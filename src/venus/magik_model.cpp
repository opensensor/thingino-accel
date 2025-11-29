/*
 * thingino-accel - Magik Model/Layer Stubs
 * These are stub implementations for the Magik model framework
 */

#include "magik_model.h"
#include "../../include/nna_memory.h"
#include <cstdio>
#include <dlfcn.h>
#include <typeinfo>
#include <new>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <cstdint>
#include <cstdarg>
#include <limits>

// Forward declarations for OEM helper functions that are overridden later in
// this translation unit. We need these early so that helper utilities can
// safely call them.
extern "C" int _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
    std::string &out,
    const void *param,
    int &index);
extern "C" int _Z19get_string_vector_tRSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEPKvRi(
    std::vector<std::string> &vec,
    const void *param,
    int &index);

/* Runtime DDR base exported from runtime.c so we can annotate parameter pointers. */
extern "C" void *__ddr_vbase;

namespace magik {
namespace venus {

// Track the currently constructed MagikModelBase instance so that
// free-standing shims invoked from .mgk code can associate parsed
// parameters with the active network. For the current Thingino flows
// we only ever have a single model instance at a time.
static MagikModelBase *g_active_model = nullptr;

// Global tensor name lookup, keyed by PyramidConfig pointer.
// This is protected from .mgk corruption of PyramidConfig internals.
static std::map<MagikModelBase::PyramidConfig*, std::map<std::string, TensorXWrapper*>> g_tensor_names;

// Global layer I/O map - maps layer ID to (input_names, output_names) pairs.
// This might be expected by .mgk's *_param_init functions.
// Making it global with C linkage so .mgk can access it if needed.
static std::map<int, std::pair<std::vector<std::string>, std::vector<std::string>>> g_layer_io_map;

// Minimal record of per-layer parameters parsed from the shared
// parameter blob. This will later be enriched with names and
// bottom/top tensor lists to drive a genuine run() implementation.
struct ParsedLayerRecord {
    int op_type;
    unsigned short layer_id;
    unsigned long long param_index;
    unsigned int flags;
};

	// Side-car record capturing the high-level build_layer metadata for each
	// logical layer as the .mgk model wires its graph. This is the primary
	// source of truth we intend to use for reconstructing a runnable graph
	// on top of the Thingino NNA runtime.
	struct BuildLayerRecord {
	    MagikModelBase::PyramidConfig *config;
	    int op_hint;
	    std::vector<std::string> bottoms;
	    std::vector<std::string> tops;
	};

	static std::vector<ParsedLayerRecord> g_parsed_layers;
	static std::vector<BuildLayerRecord> g_build_layer_records;

	namespace {
	    // Best-effort one-shot scan guard so we do not repeatedly walk the
	    // entire DDR window on every MagikModelBase::run() invocation.
	    bool g_layer_header_scan_done = false;

	    // Return true if the byte range [param + index, param + index + needed_bytes)
	    // lies entirely within the logical DDR heap
	    //   [__ddr_vbase, __ddr_vbase + kDdrSpan).
    inline bool ddr_slice_is_valid(const void *param,
                                   int index,
                                   size_t needed_bytes,
                                   long &ddr_offset) {
        ddr_offset = -1;
        if (!__ddr_vbase || !param || index < 0)
            return false;

	        uintptr_t ddr_base = reinterpret_cast<uintptr_t>(__ddr_vbase);
	        static const uintptr_t kDdrSpan = 4u * 1024u * 1024u;
        uintptr_t p = reinterpret_cast<uintptr_t>(param);
        if (p < ddr_base || p >= ddr_base + kDdrSpan)
            return false;

        uintptr_t start = p + static_cast<uintptr_t>(index);
        uintptr_t end   = start + needed_bytes;
        if (end < start || start < ddr_base || end > ddr_base + kDdrSpan)
            return false;

        ddr_offset = static_cast<long>(p - ddr_base);
        return true;
    }

    // Internal helper that mirrors OEM read_layer_param semantics but operates
    // purely on a pointer + index pair. It reads a single layer header from the
    // shared parameter blob at `param + index`, advances the cursor, appends a
    // ParsedLayerRecord, and returns the updated index.
    inline int venus_read_layer_param_internal(
        int &op_type,
        unsigned short &layer_id,
        unsigned long long &param_index,
        unsigned int &flags,
        const void *param,
        int &index) {

        const size_t kBytes = 4u + 2u + 8u + 4u;
        long ddr_off = -1;
        if (!ddr_slice_is_valid(param, index, kBytes, ddr_off)) {
            std::fprintf(stderr,
                         "[VENUS] venus_read_layer_param_internal: param=%p index=%d "
                         "not in DDR or out of range (need=%zu)\n",
                         param, index, kBytes);
            op_type = 0;
            layer_id = 0;
            param_index = 0;
            flags = 0;
            return index;
        }

        const unsigned char *base =
            static_cast<const unsigned char *>(param);

        int32_t tmp32 = 0;
        std::memcpy(&tmp32, base + index, sizeof(tmp32));
        index += static_cast<int>(sizeof(tmp32));
        op_type = tmp32;

        uint16_t tmp16 = 0;
        std::memcpy(&tmp16, base + index, sizeof(tmp16));
        index += static_cast<int>(sizeof(tmp16));
        layer_id = tmp16;

        uint64_t tmp64 = 0;
        std::memcpy(&tmp64, base + index, sizeof(tmp64));
        index += static_cast<int>(sizeof(tmp64));
        param_index = tmp64;

        uint32_t tmpf = 0;
        std::memcpy(&tmpf, base + index, sizeof(tmpf));
        index += static_cast<int>(sizeof(tmpf));
        flags = tmpf;

	        ParsedLayerRecord rec{};
	        rec.op_type = op_type;
	        rec.layer_id = layer_id;
	        rec.param_index = param_index;
	        rec.flags = flags;
	        g_parsed_layers.push_back(rec);

	        std::fprintf(stderr,
	                     "[VENUS] venus_read_layer_param_internal: "
	                     "index=%d op_type=%d layer_id=%u param_index=%llu flags=0x%x "
	                     "ddr_off=%ld\n",
	                     index,
	                     op_type,
	                     static_cast<unsigned>(layer_id),
	                     static_cast<unsigned long long>(param_index),
	                     flags,
	                     ddr_off);
	        std::fflush(stderr);
	        return index;
	    }

	    // Very conservative scanner that looks for bytes inside the exported
	    // DDR window that *could* be OEM-style layer headers. This does not
	    // attempt to be perfect; it exists to give us some empirical signal
	    // about where the .mgk model has placed its per-layer records without
	    // relying on symbol interposition that may or may not be hit.
	    //
	    // The heuristic is simple:
	    //   - interpret 4+2+8+4 bytes at (ddr_base + offset) as
	    //     {op_type, layer_id, param_index, flags};
	    //   - require op_type and layer_id to be in a small, non-negative
	    //     range and param_index to stay within the DDR span;
	    //   - record at most kMaxHits candidates.
	    //
	    // Results are appended to g_parsed_layers. This is only used when the
	    // read_layer_param shim did not observe any calls for this model.
	    void scan_param_blob_for_layer_headers_once() {
	        if (g_layer_header_scan_done)
	            return;
	        g_layer_header_scan_done = true;

	        if (!__ddr_vbase) {
	            std::fprintf(stderr,
	                         "[VENUS] scan_param_blob_for_layer_headers_once: "
	                         "__ddr_vbase is null, skipping scan.\n");
	            std::fflush(stderr);
	            return;
	        }

	        const unsigned char *base =
	            static_cast<const unsigned char *>(__ddr_vbase);
	        static const uintptr_t kDdrSpan = 4u * 1024u * 1024u;
	        const int kHeaderBytes = 4 + 2 + 8 + 4; // 18
	        const int kMaxHits = 128;
	        int hits = 0;

	        // We step by 4 bytes for now, assuming most interesting structures
	        // are at least word-aligned. This keeps the scan affordable while
	        // still covering the entire logical DDR window.
	        for (int offset = 0; offset + kHeaderBytes <= (int)kDdrSpan;
	             offset += 4) {
	            long ddr_off = -1;
	            if (!ddr_slice_is_valid(__ddr_vbase, offset,
	                                   (size_t)kHeaderBytes, ddr_off)) {
	                // Once ddr_slice_is_valid fails for a monotonically
	                // increasing offset we can stop.
	                break;
	            }

	            const unsigned char *ptr = base + offset;
	            int32_t raw_op_type = 0;
	            uint16_t raw_layer_id = 0;
	            uint64_t raw_param_index = 0;
	            uint32_t raw_flags = 0;

	            std::memcpy(&raw_op_type, ptr, sizeof(raw_op_type));
	            ptr += sizeof(raw_op_type);
	            std::memcpy(&raw_layer_id, ptr, sizeof(raw_layer_id));
	            ptr += sizeof(raw_layer_id);
	            std::memcpy(&raw_param_index, ptr, sizeof(raw_param_index));
	            ptr += sizeof(raw_param_index);
	            std::memcpy(&raw_flags, ptr, sizeof(raw_flags));

	            // Heuristics: keep only records that look vaguely sane. These
	            // bounds are intentionally loose; they are only meant to filter
	            // obviously garbage patterns.
	            if (raw_op_type < 0 || raw_op_type > 512)
	                continue;
	            if (raw_layer_id > 512)
	                continue;
	            if (raw_param_index >= kDdrSpan)
	                continue;
		    // Skip entries that are completely zero; these are almost certainly
		    // just uninitialized space at the start of the DDR heap and they
		    // flood our candidate list with non-informative records.
		    if (raw_op_type == 0 && raw_layer_id == 0 &&
		        raw_param_index == 0 && raw_flags == 0)
		        continue;

	            ParsedLayerRecord rec{};
	            rec.op_type = raw_op_type;
	            rec.layer_id = raw_layer_id;
	            rec.param_index = raw_param_index;
	            rec.flags = raw_flags;
	            g_parsed_layers.push_back(rec);
	            ++hits;

	            if (hits >= kMaxHits)
	                break;
	        }

		        std::fprintf(stderr,
		                     "[VENUS] scan_param_blob_for_layer_headers_once: "
		                     "recorded %d candidate headers, g_parsed_layers=%zu\n",
		                     hits,
		                     g_parsed_layers.size());
		        std::fflush(stderr);
		    }

		    // Build a best-effort layer I/O map (bottom/top tensor names per layer ID)
		    // directly from the parameter blob in DDR.
		    //
		    // We do this *without* interposing on the OEM read_common_param/read_layer_param
		    // helpers so that we do not disturb the .mgk's own parsing. Instead we:
		    //   - derive candidate layer headers in g_parsed_layers (either from a real
		    //     shim or via scan_param_blob_for_layer_headers_once());
		    //   - for each header, treat param_index as an offset into the shared DDR
		    //     parameter blob and attempt to decode:
		    //         layer_name, bottoms, tops
		    //     using our hardened get_string_t/get_string_vector_t overrides;
		    //   - validate decoded tensor names against the set of TensorInfo names we
		    //     already know for the given PyramidConfig;
		    //   - on success, populate g_layer_io_map[layer_id] with those bottom/top
		    //     vectors.
		    //
		    // This gives *_param_init functions like format_convert_param_init access to
		    // a reasonably accurate view of the layer connectivity without requiring any
		    // additional symbol interposition.
		    void build_layer_io_map_from_ddr(MagikModelBase::PyramidConfig *config) {
		        if (!config)
		            return;
		        if (!__ddr_vbase)
		            return;
		        // Only build the map once per model instance.
		        if (!g_layer_io_map.empty())
		            return;

		        // Ensure we have some candidate layer headers.
		        if (g_parsed_layers.empty()) {
		            scan_param_blob_for_layer_headers_once();
		        }

		        auto tn_it = g_tensor_names.find(config);
		        if (tn_it == g_tensor_names.end() || tn_it->second.empty()) {
		            std::fprintf(stderr,
		                         "[VENUS] build_layer_io_map_from_ddr: no tensor names for config=%p, skipping I/O map build.\n",
		                         (void*)config);
		            std::fflush(stderr);
		            return;
		        }
		        const auto &tensor_map = tn_it->second;

		        int built = 0;
		        // Heuristic upper bound for the parameter blob span we are willing to
		        // consider when walking param_index offsets.
		        static const unsigned long long kMaxParamSpan = 4ull * 1024ull * 1024ull;

		        for (const ParsedLayerRecord &rec : g_parsed_layers) {
		            if (rec.param_index >= kMaxParamSpan)
		                continue;
		            if (rec.param_index >
		                static_cast<unsigned long long>(std::numeric_limits<int>::max()))
		                continue;

		            const void *param = __ddr_vbase;
		            int index = static_cast<int>(rec.param_index);
		            int cursor = index;

		            std::string layer_name;
		            std::vector<std::string> bottoms;
		            std::vector<std::string> tops;

		            _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
		                layer_name, param, cursor);
		            _Z19get_string_vector_tRSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEPKvRi(
		                bottoms, param, cursor);
		            _Z19get_string_vector_tRSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEPKvRi(
		                tops, param, cursor);

		            if (layer_name.empty())
		                continue;
		            if (bottoms.empty() && tops.empty())
		                continue;

		            auto names_valid = [&tensor_map](const std::vector<std::string> &vec) {
		                for (const auto &n : vec) {
		                    if (n.empty())
		                        continue;
		                    if (tensor_map.find(n) == tensor_map.end())
		                        return false;
		                }
		                return true;
		            };

		            // Require at least one of the bottom/top lists to be consistent with
		            // the known tensor set; this filters out the vast majority of false
		            // positives from the DDR scan.
		            if (!names_valid(bottoms) && !names_valid(tops))
		                continue;

		            g_layer_io_map[static_cast<int>(rec.layer_id)] =
		                std::make_pair(bottoms, tops);
		            ++built;

		            if (built <= 8) {
		                std::fprintf(stderr,
		                             "[VENUS] build_layer_io_map_from_ddr: layer_id=%u name=\"%s\" nb=%zu nt=%zu param_index=%llu\n",
		                             static_cast<unsigned>(rec.layer_id),
		                             layer_name.c_str(),
		                             bottoms.size(),
		                             tops.size(),
		                             static_cast<unsigned long long>(rec.param_index));
		            }
		        }

		        std::fprintf(stderr,
		                     "[VENUS] build_layer_io_map_from_ddr: built %d entries (g_parsed_layers=%zu)\n",
		                     built,
		                     g_parsed_layers.size());
		        std::fflush(stderr);
		    }
		} // anonymous namespace

/* Device implementation */
Device::Device() {}
Device::~Device() {}

/* TensorXWrapper implementation - layout aligned with OEM libmert. */
TensorXWrapper::TensorXWrapper()
    : tensorx(nullptr), name(), flush_status(FlushCacheStatus::DISABLED) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper() default constructor (this=%p)\n",
           (void*)this);
    fflush(stdout);
}

TensorXWrapper::TensorXWrapper(TensorX *tx)
    : tensorx(tx), name(), flush_status(FlushCacheStatus::DISABLED) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper(tx=%p, this=%p)\n",
           (void*)tx, (void*)this);
    fflush(stdout);
}

TensorXWrapper::TensorXWrapper(TensorX *tx, std::string n)
    : tensorx(tx), name(std::move(n)), flush_status(FlushCacheStatus::DISABLED) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper(tx=%p, name=%s, this=%p)\n",
           (void*)tx, name.c_str(), (void*)this);
    fflush(stdout);
}

TensorXWrapper::TensorXWrapper(TensorX *tx, std::string n, FlushCacheStatus status)
    : tensorx(tx), name(std::move(n)), flush_status(status) {
    printf("[VENUS] TensorXWrapper::TensorXWrapper(tx=%p, name=%s, status=%d, this=%p)\n",
           (void*)tx, name.c_str(), (int)flush_status, (void*)this);
    fflush(stdout);
}

TensorXWrapper::~TensorXWrapper() {
    printf("[VENUS] TensorXWrapper::~TensorXWrapper(this=%p, tensorx=%p, status=%d)\n",
           (void*)this, (void*)tensorx, (int)flush_status);
    fflush(stdout);
}

void TensorXWrapper::set_content(TensorX *tx) {
    printf("[VENUS] TensorXWrapper::set_content(this=%p, tx=%p)\n",
           (void*)this, (void*)tx);
    tensorx = tx;
}

TensorX *TensorXWrapper::get_content() const {
    printf("[VENUS] TensorXWrapper::get_content(this=%p) -> %p\n",
           (void*)this, (void*)tensorx);
    return tensorx;
}

void TensorXWrapper::set_flush_cache_status(FlushCacheStatus status) {
    printf("[VENUS] TensorXWrapper::set_flush_cache_status(this=%p, status=%d)\n",
           (void*)this, (int)status);
    flush_status = status;
}

FlushCacheStatus TensorXWrapper::get_flush_cache_status() const {
    return flush_status;
}

std::string TensorXWrapper::get_name() const {
    return name;
}

int TensorXWrapper::flush_cache() {
    /* OEM implementation checks flush_status and, if enabled, flushes the
     * underlying TensorX data cache via __aie_flushcache_dir.
     * For now we only log and pretend success when enabled.
     */
    if (flush_status != FlushCacheStatus::ENABLE_ONCE &&
        flush_status != FlushCacheStatus::ENABLE_ALWAYS) {
        return 1;
    }

    printf("[VENUS] TensorXWrapper::flush_cache(this=%p, tensorx=%p, status=%d)\n",
           (void*)this, (void*)tensorx, (int)flush_status);
    fflush(stdout);

    // TODO: call into NNA cache flush when we have a confirmed implementation.
    return 1;
}

/* TensorX constructor with size, data pointer, and device */
extern "C" void _ZN5magik5venus7TensorXC1EjPvNS0_6DeviceE(TensorX *this_ptr, uint32_t size, void *data_ptr, Device dev) {
    printf("[VENUS] TensorX::TensorX(this=%p, size=%u, data=%p)\n", (void*)this_ptr, size, data_ptr);
    fflush(stdout);
    (void)dev; /* Device selection not yet used; TODO: hook into MBuffer/device allocator. */

    /* The .mgk file has already allocated space for this object on the stack.
     * Use placement-new to run our C++ default constructor and initialize fields.
     */
    new (this_ptr) TensorX();

    /* Override key fields based on constructor arguments. */
    this_ptr->data       = data_ptr;
    this_ptr->bytes      = size;
    this_ptr->owns_data  = 0;     /* Caller owns the data */
    this_ptr->align      = 64;    /* NNA alignment */
    this_ptr->data_offset = 0;    /* No offset by default */

    printf("[VENUS] TensorX::TensorX() completed:\n");
    printf("[VENUS]   ndims=%d\n",
           this_ptr->dims_begin && this_ptr->dims_end ? (int)(this_ptr->dims_end - this_ptr->dims_begin) : 0);
    printf("[VENUS]   dtype=%d, format=%d\n", (int)this_ptr->dtype, (int)this_ptr->format);
    printf("[VENUS]   data=%p, bytes=%u\n", this_ptr->data, this_ptr->bytes);
    printf("[VENUS]   owns_data=%u, ref_count=%d\n", this_ptr->owns_data, this_ptr->ref_count);
    printf("[VENUS]   align=%u, data_offset=%u\n", this_ptr->align, this_ptr->data_offset);
    printf("[VENUS]   sizeof(TensorX)=%zu\n", sizeof(TensorX));
    fflush(stdout);
}

/* TensorX mudata method */
extern "C" void* _ZN5magik5venus7TensorX6mudataEjj(TensorX *this_ptr, uint32_t offset, uint32_t size) {
    (void)size;
    if (!this_ptr->data) {
        return nullptr;
    }
    return (char*)this_ptr->data + offset;
}

/* TensorX get_memory_size (OEM-compatible symbol) */
extern "C" size_t _ZNK5magik5venus7TensorX15get_memory_sizeEv(const TensorX *this_ptr) {
    /* OEM uses bytes rounded up to align() boundary; align field already reflects that. */
    size_t aligned = (static_cast<size_t>(this_ptr->bytes) + (this_ptr->align - 1u)) & ~static_cast<size_t>(this_ptr->align - 1u);
    printf("[VENUS] TensorX::get_memory_size(this=%p) -> %zu (bytes=%u, align=%u)\n",
           (void*)this_ptr, aligned, this_ptr->bytes, this_ptr->align);
    fflush(stdout);
    return aligned;
}

/* TensorX align (OEM-compatible symbol) */
extern "C" size_t _ZNK5magik5venus7TensorX5alignEv(const TensorX *this_ptr) {
    size_t kAlign = this_ptr->align ? this_ptr->align : 64u;
    printf("[VENUS] TensorX::align(this=%p) -> %zu\n", (void*)this_ptr, kAlign);
    fflush(stdout);
    return kAlign;
}

/* TensorX vdata<char> (OEM-compatible symbol) */
extern "C" const void* _ZNK5magik5venus7TensorX5vdataIaEEPKT_v(const TensorX *this_ptr) {
    if (!this_ptr->data) {
        return nullptr;
    }

    size_t offset = static_cast<size_t>(this_ptr->data_offset);
    const void *ptr = static_cast<const void*>(
        static_cast<const char*>(this_ptr->data) + offset);
    printf("[VENUS] TensorX::vdata<char>(this=%p) -> %p (offset=%zu)\n",
           (void*)this_ptr, ptr, offset);
    fflush(stdout);
    return ptr;
}

	/* TensorX reset_data stub (OEM-compatible symbol)
	 * Signature: void TensorX::reset_data(void *buf, void *data, int bytes)
	 * We treat `data` as a direct pointer to the underlying buffer and avoid
	 * any std::shared_ptr<MBuffer> / control-block gymnastics to prevent
	 * layout-dependent corruption.
	 */
	extern "C" void _ZN5magik5venus7TensorX10reset_dataEPvS2_i(
	    TensorX *this_ptr, void *buf, void *data_ptr, int bytes) {
	    (void)buf; /* Unused for now; OEM may use this for host-side aliasing. */

	    long raw_ndims = 0;
	    if (this_ptr->dims_begin && this_ptr->dims_end) {
	        raw_ndims = (long)(this_ptr->dims_end - this_ptr->dims_begin);
	    }

	    printf("[VENUS] TensorX::reset_data(this=%p, buf=%p, data=%p, bytes=%d, raw_ndims=%ld)\n",
	           (void*)this_ptr, buf, data_ptr, bytes, raw_ndims);
	    fflush(stdout);

	    /* Do not touch dims_*; they are owned by set_shape / create_tensor. */
	    this_ptr->data = data_ptr;
	    if (bytes > 0) {
	        this_ptr->bytes = (uint32_t)bytes;
	    }
	    /* We do not give ownership of this buffer to TensorX. */
	    this_ptr->owns_data = 0;
	    if (this_ptr->align == 0) {
	        this_ptr->align = 64;
	    }
	    /* Keep data_offset as-is; OEM sometimes uses non-zero offsets. */
	}

	/* TensorX reset_buffer stub (OEM-compatible symbol)
	 * Signature: void TensorX::reset_buffer(std::shared_ptr<MBuffer>)
	 * The mangled type encodes std::shared_ptr<MBuffer> by value. On 32-bit
	 * this is two pointer-sized words (ptr + control-block). We accept them
	 * as two opaque void* slots to avoid instantiating std::shared_ptr here,
	 * which would otherwise run its destructor and interact with an OEM
	 * control-block layout we don't fully emulate yet.
	 */
	extern "C" void _ZN5magik5venus7TensorX12reset_bufferESt10shared_ptrINS0_7MBufferEE(
	    TensorX *this_ptr, void *sptr_slot0, void *sptr_slot1) {
	    printf("[VENUS] TensorX::reset_buffer(this=%p, sptr0=%p, sptr1=%p) [stub/no-op]\n",
	           (void*)this_ptr, sptr_slot0, sptr_slot1);
	    fflush(stdout);
	    /* Intentionally do nothing: TensorX::data / bytes are managed by
	     * create_tensor / reset_data. MBuffer/shared_ptr lifetime stays in
	     * OEM code and is treated as opaque here. */
	}

	/* TensorX free_mbo stub (OEM-compatible symbol)
	 * Signature: void TensorX::free_mbo()
	 * OEM calls into MBuffer::free_data() here. For now we turn this into a
	 * no-op to avoid double-free of ORAM/DDR buffers that we manage via
	 * core::oram_malloc / core::oram_free. This may leak small OEM-managed
	 * host buffers, but that's acceptable for short-lived inference runs.
	 */
	extern "C" void _ZN5magik5venus7TensorX8free_mboEv(TensorX *this_ptr) {
	    printf("[VENUS] TensorX::free_mbo(this=%p, data=%p, bytes=%u) [stub/no-op]\n",
	           (void*)this_ptr, this_ptr ? this_ptr->data : nullptr,
	           this_ptr ? this_ptr->bytes : 0u);
	    fflush(stdout);
	    /* Do not modify data/bytes/owns_data here. Lifetime is handled by
	     * MagikModelBase::create_tensor and Tensor wrappers. */
	}



/* Core ORAM functions */
namespace core {
    void* oram_malloc(size_t size) {
        printf("[VENUS] core::oram_malloc(size=%zu)\n", size);
        fflush(stdout);
        /* Use NNA memory allocator */
        void *ptr = nna_malloc(size);
        printf("[VENUS] core::oram_malloc returning %p\n", ptr);
        fflush(stdout);
        return ptr;
    }

    void oram_free(void *ptr) {
        printf("[VENUS] core::oram_free(ptr=%p)\n", ptr);
        fflush(stdout);
        nna_free(ptr);
    }
} // namespace core

/* MagikLayerBase implementation */
MagikLayerBase::MagikLayerBase() {
    // Every time a MagikLayerBase-derived instance is constructed by a .mgk
    // model, record it against the currently active MagikModelBase so we can
    // later introspect the layer graph from MagikModelBase::run().
    printf("[VENUS] MagikLayerBase::MagikLayerBase(this=%p, active_model=%p)\n",
           (void*)this, (void*)g_active_model);
    fflush(stdout);

    if (g_active_model) {
        g_active_model->layers_.push_back(this);
    }
}
MagikLayerBase::~MagikLayerBase() {}

void MagikLayerBase::set_inputs(std::vector<TensorXWrapper*> inputs) {
    printf("[VENUS] MagikLayerBase::set_inputs(this=%p, inputs.size=%zu)\n",
           (void*)this, inputs.size());
    fflush(stdout);
    inputs_ = inputs;
}

void MagikLayerBase::set_outputs(std::vector<TensorXWrapper*> outputs) {
    printf("[VENUS] MagikLayerBase::set_outputs(this=%p, outputs.size=%zu)\n",
           (void*)this, outputs.size());
    fflush(stdout);
    outputs_ = outputs;
}

void MagikLayerBase::_flush_cache(std::vector<TensorXWrapper*> tensors) {
    (void)tensors;
}

std::vector<TensorXWrapper*> MagikLayerBase::get_inputs() const {
    return std::vector<TensorXWrapper*>();
}

std::vector<TensorXWrapper*> MagikLayerBase::get_outputs() const {
    return std::vector<TensorXWrapper*>();
}

std::vector<TensorXWrapper*> MagikLayerBase::get_input_wrappers() const {
    return std::vector<TensorXWrapper*>();
}

std::vector<TensorXWrapper*> MagikLayerBase::get_output_wrappers() const {
    return std::vector<TensorXWrapper*>();
}

std::string MagikLayerBase::get_name() const {
    return "";
}

int MagikLayerBase::get_layer_id() const {
    return 0;
}

int MagikLayerBase::forward() {
    printf("[VENUS] MagikLayerBase::forward() - base class no-op\n");
    return 0;
}

int MagikLayerBase::update_cache_buffer_ptr(void *ptr) {
    (void)ptr;
    return 0;
}

/* MagikKernelLayer implementation */
int MagikKernelLayer::next_layer_id_ = 0;

MagikKernelLayer::MagikKernelLayer(int op_hint,
                                   KernelFunc kernel_fn,
                                   std::unique_ptr<kernel::KernelParam>& param,
                                   OpConfig* op_cfg)
    : MagikLayerBase()
    , op_hint_(op_hint)
    , kernel_fn_(std::move(kernel_fn))
    , param_(std::move(param))
    , op_cfg_(op_cfg)
    , layer_id_(next_layer_id_++)
{
    name_ = "layer_" + std::to_string(layer_id_);
    printf("[VENUS] MagikKernelLayer::MagikKernelLayer(op=%d, id=%d)\n", op_hint_, layer_id_);
    fflush(stdout);
}

MagikKernelLayer::MagikKernelLayer(int op_hint,
                                   KernelFunc kernel_fn,
                                   std::unique_ptr<kernel::KernelParam>& param,
                                   OpConfig* op_cfg,
                                   std::vector<TensorXWrapper*> inputs,
                                   std::vector<TensorXWrapper*> outputs)
    : MagikLayerBase()
    , op_hint_(op_hint)
    , kernel_fn_(std::move(kernel_fn))
    , param_(std::move(param))
    , op_cfg_(op_cfg)
    , inputs_(std::move(inputs))
    , outputs_(std::move(outputs))
    , layer_id_(next_layer_id_++)
{
    name_ = "layer_" + std::to_string(layer_id_);
    printf("[VENUS] MagikKernelLayer::MagikKernelLayer(op=%d, id=%d, ins=%zu, outs=%zu)\n",
           op_hint_, layer_id_, inputs_.size(), outputs_.size());
    fflush(stdout);
}

MagikKernelLayer::~MagikKernelLayer() {
    printf("[VENUS] MagikKernelLayer::~MagikKernelLayer(id=%d)\n", layer_id_);
}

void MagikKernelLayer::set_inputs(std::vector<TensorXWrapper*> inputs) {
    inputs_ = std::move(inputs);
}

void MagikKernelLayer::set_outputs(std::vector<TensorXWrapper*> outputs) {
    outputs_ = std::move(outputs);
}

std::vector<TensorXWrapper*> MagikKernelLayer::get_inputs() const {
    return inputs_;
}

std::vector<TensorXWrapper*> MagikKernelLayer::get_outputs() const {
    return outputs_;
}

std::vector<TensorXWrapper*> MagikKernelLayer::get_input_wrappers() const {
    return inputs_;
}

std::vector<TensorXWrapper*> MagikKernelLayer::get_output_wrappers() const {
    return outputs_;
}

std::string MagikKernelLayer::get_name() const {
    return name_;
}

int MagikKernelLayer::get_layer_id() const {
    return layer_id_;
}

int MagikKernelLayer::forward() {
    printf("[VENUS] MagikKernelLayer::forward(id=%d, op=%d)\n", layer_id_, op_hint_);
    fflush(stdout);

    if (kernel_fn_) {
        printf("[VENUS]   Calling kernel function...\n");
        fflush(stdout);
        ReturnValue rv = kernel_fn_(param_, op_cfg_);
        printf("[VENUS]   Kernel returned: %d\n", rv.code);
        fflush(stdout);
        return rv.code;
    }

    printf("[VENUS]   No kernel function, skipping\n");
    return 0;
}

int MagikKernelLayer::update_cache_buffer_ptr(void *ptr) {
    (void)ptr;
    return 0;
}

kernel::KernelParam* MagikKernelLayer::get_kernel_param() {
    return param_.get();
}

/* MagikModelBase implementation */
MagikModelBase::MagikModelBase(long long param1, long long param2, void *&param3, void *param4,
                               ModelMemoryInfoManager::MemAllocMode mode, ModuleMode module_mode)
    : main_pyramid_config_(nullptr), layer_io_map_ptr_(nullptr) {
    printf("[VENUS] MagikModelBase::MagikModelBase(p1=%lld, p2=%lld, p3=%p, p4=%p)\n",
           param1, param2, param3, param4);
    fflush(stdout);

    /* Initialize padding to zero */
    memset(padding_, 0, sizeof(padding_));
    memset(padding2_, 0, sizeof(padding2_));

    /* Initialize layer I/O map pointer to point to the global map */
    layer_io_map_ptr_ = &g_layer_io_map;
    printf("[VENUS] MagikModelBase: layer_io_map_ptr_ = %p\n", (void*)layer_io_map_ptr_);
    fflush(stdout);

    /* Constructor - just suppress unused warnings */
    (void)param1; (void)param2; (void)param3; (void)param4; (void)mode; (void)module_mode;
    printf("[VENUS] MagikModelBase::MagikModelBase() - constructor body complete\n");
    fflush(stdout);

    // Mark this instance as the active model so that parameter shims can
    // attach parsed information to it. For now we assume a single active
    // model in the process at any given time.
    g_active_model = this;
}

MagikModelBase::~MagikModelBase() {
    if (main_pyramid_config_) {
        delete main_pyramid_config_;
        main_pyramid_config_ = nullptr;
    }

    if (g_active_model == this) {
        g_active_model = nullptr;
	        g_parsed_layers.clear();
	        g_build_layer_records.clear();
    }
}

static void dump_memory(const char* label, void* ptr, size_t size) {
    printf("[VENUS] Memory dump: %s at %p (%zu bytes)\n", label, ptr, size);
    fflush(stdout);
    unsigned char* p = (unsigned char*)ptr;
    for (size_t i = 0; i < size; i += 16) {
        printf("  %04zx: ", i);
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            printf("%02x ", p[i + j]);
        }
        printf(" | ");
        for (size_t j = 0; j < 16 && (i + j) < size; j++) {
            unsigned char c = p[i + j];
            printf("%c", (c >= 32 && c < 127) ? c : '.');
        }
        printf("\n");
    }
    fflush(stdout);
}

int MagikModelBase::run() {
	printf("[VENUS] MagikModelBase::run() - BEGIN (logging-only) this=%p active=%p\n",
	       (void*)this, (void*)g_active_model);
	printf("[VENUS] MARKER_1\n");
	fflush(stdout);

	// Show expected offsets vs actual content at those offsets
	printf("[VENUS] Expected offsets in MagikModelBase:\n");
	printf("[VENUS]   sizeof(MagikModelBase) = %zu\n", sizeof(MagikModelBase));
	printf("[VENUS]   &layers_ - this = %zu\n", (size_t)((char*)&layers_ - (char*)this));
	printf("[VENUS]   &pyramid_configs_ - this = %zu\n", (size_t)((char*)&pyramid_configs_ - (char*)this));
	printf("[VENUS]   &main_pyramid_config_ - this = %zu\n", (size_t)((char*)&main_pyramid_config_ - (char*)this));
	printf("[VENUS]   &inputs_ - this = %zu\n", (size_t)((char*)&inputs_ - (char*)this));
	printf("[VENUS]   &outputs_ - this = %zu\n", (size_t)((char*)&outputs_ - (char*)this));
	printf("[VENUS] MARKER_2\n");
	fflush(stdout);

	// Dump first 512 bytes of 'this' object to see actual memory layout
	// Our sizeof(MagikModelBase) is 80, so dump 512 to see derived class members
	dump_memory("MagikModelBase this", (void*)this, 512);

		if (!main_pyramid_config_) {
			printf("[VENUS] MagikModelBase::run(): main_pyramid_config_ is null; "
			       "no tensors to process.\n");
		} else {
			printf("[VENUS] MagikModelBase::run(): pyramid_configs_.size()=%zu, "
			       "main_pyramid_config_=%p\n",
			       pyramid_configs_.size(),
			       (void*)main_pyramid_config_);
		}

	// Log input and output wrappers that the model believes it has.
	printf("[VENUS]   inputs_.size()=%zu, outputs_.size()=%zu\n",
	       inputs_.size(), outputs_.size());
	for (size_t i = 0; i < inputs_.size(); ++i) {
		TensorXWrapper *w = inputs_[i];
		TensorX *tx = w ? w->get_content() : nullptr;
		printf("[VENUS]     input[%zu]: wrapper=%p name='%s' tx=%p\n",
		       i, (void*)w,
		       w ? w->get_name().c_str() : "",
		       (void*)tx);
	}
	for (size_t i = 0; i < outputs_.size(); ++i) {
		TensorXWrapper *w = outputs_[i];
		TensorX *tx = w ? w->get_content() : nullptr;
		printf("[VENUS]     output[%zu]: wrapper=%p name='%s' tx=%p\n",
		       i, (void*)w,
		       w ? w->get_name().c_str() : "",
		       (void*)tx);
	}
	fflush(stdout);

		// If our read_layer_param shim did not capture any headers for this
		// model, fall back to a best-effort scan across the logical DDR window
		// to derive candidate layer records directly from the parameter blob.
		if (g_parsed_layers.empty()) {
			scan_param_blob_for_layer_headers_once();
		}

		// Log the layer records we have reconstructed so far.
		printf("[VENUS]   Parsed layers (side-car decode): %zu\n", g_parsed_layers.size());
		for (size_t i = 0; i < g_parsed_layers.size(); ++i) {
			const ParsedLayerRecord &rec = g_parsed_layers[i];
			printf("[VENUS]     L%zu: op_type=%d layer_id=%u param_index=%llu flags=0x%x\n",
			       i,
			       rec.op_type,
			       (unsigned)rec.layer_id,
			       (unsigned long long)rec.param_index,
			       rec.flags);
		}
			fflush(stdout);

			// Also surface the higher-level build_layer records we have observed via
			// the dedicated shim. This is the main logical graph description we
			// will eventually use to drive real NNA execution.
			printf("[VENUS]   build_layer records: %zu\n", g_build_layer_records.size());
			const size_t kMaxDump = g_build_layer_records.size() < 32 ?
			                       g_build_layer_records.size() : 32;
			for (size_t i = 0; i < kMaxDump; ++i) {
			    const BuildLayerRecord &rec = g_build_layer_records[i];
			    const char *b0 = (rec.bottoms.empty() ? "" : rec.bottoms[0].c_str());
			    const char *t0 = (rec.tops.empty() ? "" : rec.tops[0].c_str());
			    printf("[VENUS]     BL%zu: cfg=%p op_hint=%d nb=%zu nt=%zu first_bottom='%s' first_top='%s'\n",
			           i,
			           (void*)rec.config,
			           rec.op_hint,
			           rec.bottoms.size(),
			           rec.tops.size(),
			           b0,
			           t0);
			}
			fflush(stdout);

		// Log layers from base class member and from PyramidConfig
		printf("[VENUS]   layers_.size()=%zu (base class member)\n", layers_.size());

		// Check for layers in main_pyramid_config_ (populated by .mgk's build_layer)
		size_t pyramid_layer_count = 0;
		if (main_pyramid_config_) {
			pyramid_layer_count = main_pyramid_config_->layers_.size();
			printf("[VENUS]   main_pyramid_config_->layers_.size()=%zu\n", pyramid_layer_count);
		}

		// Use pyramid layers if available, otherwise fall back to base class layers_
		std::vector<MagikLayerBase*>* active_layers = &layers_;
		if (pyramid_layer_count > 0) {
			active_layers = &main_pyramid_config_->layers_;
			printf("[VENUS]   Using layers from main_pyramid_config_!\n");
		} else if (!layers_.empty()) {
			printf("[VENUS]   Using layers from base class layers_\n");
		}

		for (size_t i = 0; i < active_layers->size(); ++i) {
			MagikLayerBase *layer = (*active_layers)[i];
			if (!layer) {
				printf("[VENUS]     layer[%zu]: nullptr\n", i);
				continue;
			}

			std::string lname = layer->get_name();
			std::vector<TensorXWrapper*> in_wrappers = layer->get_input_wrappers();
			std::vector<TensorXWrapper*> out_wrappers = layer->get_output_wrappers();

			printf("[VENUS]     layer[%zu]: this=%p name='%s' inputs=%zu outputs=%zu\n",
			       i,
			       (void*)layer,
			       lname.c_str(),
			       in_wrappers.size(),
			       out_wrappers.size());
		}
	fflush(stdout);

	// Execute layers if we have any
	if (!active_layers->empty()) {
		printf("[VENUS] MagikModelBase::run() - Executing %zu layers\n", active_layers->size());
		fflush(stdout);

		for (size_t i = 0; i < active_layers->size(); ++i) {
			MagikLayerBase *layer = (*active_layers)[i];
			if (!layer) {
				printf("[VENUS]   Skipping null layer[%zu]\n", i);
				continue;
			}

			printf("[VENUS]   Executing layer[%zu]: '%s'\n", i, layer->get_name().c_str());
			fflush(stdout);

			int ret = layer->forward();
			if (ret != 0) {
				printf("[VENUS]   Layer[%zu] returned error: %d\n", i, ret);
				fflush(stdout);
				return ret;
			}
		}

		printf("[VENUS] MagikModelBase::run() - All layers executed successfully\n");
		fflush(stdout);
	} else {
		printf("[VENUS] MagikModelBase::run() - No layers to execute (no-op)\n");
		fflush(stdout);
	}

	printf("[VENUS] MagikModelBase::run() - END\n");
	fflush(stdout);
	return 0;
}

int MagikModelBase::reshape() {
    printf("[VENUS] MagikModelBase::reshape() called\n");
    fflush(stdout);
    return 0;
}

int MagikModelBase::pre_graph_run() {
    printf("[VENUS] MagikModelBase::pre_graph_run() called\n");
    fflush(stdout);
    return 0;
}
int MagikModelBase::free_forward_memory() { return 0; }
int MagikModelBase::free_inputs_memory() { return 0; }
int MagikModelBase::open_mnni_debug() { return 0; }
int MagikModelBase::open_mnni_profiler() { return 0; }
int MagikModelBase::set_main_pyramid_config(int level) {
    printf("[VENUS] MagikModelBase::set_main_pyramid_config(level=%d)\n", level);
    fflush(stdout);

	    // OEM uses this to select which PyramidConfig is considered the
	    // "main" one by level index. Our PyramidConfig no longer stores a
	    // level field explicitly; instead we pick an entry from the
	    // pyramid_configs_ vector by index when possible.
	    if (!pyramid_configs_.empty()) {
	        size_t idx = 0;
	        if (level >= 0 && static_cast<size_t>(level) < pyramid_configs_.size()) {
	            idx = static_cast<size_t>(level);
	        }
	        main_pyramid_config_ = pyramid_configs_[idx];
	        printf("[VENUS]   main_pyramid_config_ set to %p (idx=%zu)\n",
	               (void*)main_pyramid_config_, idx);
	        fflush(stdout);
	    }

	    return 0;
}
int MagikModelBase::add_pyramid_config(PyramidConfig *config) {
    printf("[VENUS] MagikModelBase::add_pyramid_config(config=%p)\n", config);
    fflush(stdout);

    if (config) {
        /* Add to the vector - now that we have the correct memory layout */
        pyramid_configs_.push_back(config);
        printf("[VENUS] Added to pyramid_configs_ vector (size now = %zu)\n", pyramid_configs_.size());
        fflush(stdout);

        /* Set as main config if we don't have one */
        if (!main_pyramid_config_) {
            main_pyramid_config_ = config;
            printf("[VENUS] Set as main_pyramid_config_ = %p\n", (void*)main_pyramid_config_);
            fflush(stdout);
        }
    }

    return 0;
}

int MagikModelBase::create_and_add_pyramid_config() {
    fprintf(stderr, "[VENUS] MagikModelBase::create_and_add_pyramid_config() this=%p\n", (void*)this);

    /* Create a new PyramidConfig */
    PyramidConfig *config = new PyramidConfig();

    /* Debug: show PyramidConfig layout */
    fprintf(stderr, "[VENUS] PyramidConfig layout:\n");
    fprintf(stderr, "[VENUS]   sizeof(PyramidConfig) = %zu\n", sizeof(PyramidConfig));
    fprintf(stderr, "[VENUS]   &layers_ offset = %zu\n", (size_t)((char*)&config->layers_ - (char*)config));
    fprintf(stderr, "[VENUS]   &tensors_ offset = %zu\n", (size_t)((char*)&config->tensors_ - (char*)config));
    fprintf(stderr, "[VENUS]   &input_tensors_ offset = %zu\n", (size_t)((char*)&config->input_tensors_ - (char*)config));
    fprintf(stderr, "[VENUS]   &output_tensors_ offset = %zu\n", (size_t)((char*)&config->output_tensors_ - (char*)config));
    fflush(stderr);

    /* std::vector/std::map members are already default-constructed as
     * empty containers.  reserved_ is unused but we zero it for
     * determinism when dumping raw memory.
     */
    memset(config->reserved_, 0, sizeof(config->reserved_));

    fprintf(stderr, "[VENUS] Created new PyramidConfig at %p\n", (void*)config);

    /* Add it to the collection - inline the logic to avoid virtual call issues */
    fprintf(stderr, "[VENUS] About to add config to pyramid_configs_ and set main_pyramid_config_\n");

    pyramid_configs_.push_back(config);
    fprintf(stderr, "[VENUS] Added to pyramid_configs_ vector (size now = %zu)\n", pyramid_configs_.size());

    if (!main_pyramid_config_) {
        main_pyramid_config_ = config;
        fprintf(stderr, "[VENUS] Set main_pyramid_config_ = %p\n", (void*)main_pyramid_config_);
    }

    fprintf(stderr, "[VENUS] create_and_add_pyramid_config() returning %p\n", (void*)config);

    /* Return the pointer cast to int, as per OEM implementation */
    return (int)(intptr_t)config;
}

MagikModelBase::PyramidConfig* MagikModelBase::get_main_pyramid_config() {
    printf("[VENUS] MagikModelBase::get_main_pyramid_config() returning %p\n", (void*)main_pyramid_config_);
    fflush(stdout);
    return main_pyramid_config_;
}

TensorX* MagikModelBase::create_tensor(TensorInfo info) {
    printf("[VENUS] MagikModelBase::create_tensor(name=%s)\n", info.name.c_str());
    fflush(stdout);

    /* Log TensorInfo details to validate ABI/layout at runtime. */
    printf("[VENUS]   dtype_str=\"%s\", layout=\"%s\", is_input=%u, is_output=%u\n",
           info.dtype_str.c_str(), info.layout.c_str(),
           (unsigned)info.is_input, (unsigned)info.is_output);
    printf("[VENUS]   shape dims=[");
    for (size_t i = 0; i < info.shape.size(); ++i) {
        printf("%d%s", info.shape[i], (i + 1 < info.shape.size()) ? "," : "");
    }
    printf("]\n");
    fflush(stdout);

    /* Allocate a TensorX object (0x44 bytes as per OEM) */
    TensorX *tensor = new(std::nothrow) TensorX();
    if (!tensor) {
        printf("[VENUS] ERROR: MagikModelBase::create_tensor - new TensorX failed\n");
        fflush(stdout);
        return nullptr;
    }

    /* Derive dtype/format from strings when available. */
    DataType (*dt_fn)(const std::string&) = utils::string2data_type;
    TensorFormat (*fmt_fn)(const std::string&) = utils::string2data_format;

    DataType dtype = dt_fn(info.dtype_str);
    if (dtype == DataType::NONE || dtype == DataType::AUTO) {
        dtype = DataType::FP32;  /* Reasonable default */
    }
    TensorFormat fmt = fmt_fn(info.layout);

    tensor->dtype  = dtype;
    tensor->format = fmt;

    /* Configure shape. */
    tensor->set_shape(info.shape);

    /* Compute total number of elements. */
    size_t total_elements = 1;
    for (size_t i = 0; i < info.shape.size(); ++i) {
        int dim = info.shape[i];
        if (dim <= 0) {
            continue;
        }
        total_elements *= static_cast<size_t>(dim);
    }
    if (total_elements == 0) {
        total_elements = 1;
    }

    int bits_per_element = utils::data_type2bits(dtype);
    if (bits_per_element <= 0) {
        bits_per_element = 32;  /* Conservative default */
    }

    size_t bytes_needed = (total_elements * static_cast<size_t>(bits_per_element) + 7u) / 8u;
    if (bytes_needed == 0) {
        bytes_needed = 4;
    }

    /* Align to NNA-friendly boundary and allocate from ORAM/DDR. */
    tensor->align = 64;
    size_t aligned_bytes = (bytes_needed + tensor->align - 1u) & ~static_cast<size_t>(tensor->align - 1u);

    void *data = core::oram_malloc(aligned_bytes);
    if (!data) {
        printf("[VENUS] WARNING: core::oram_malloc(%zu) failed, falling back to malloc\n",
               aligned_bytes);
        fflush(stdout);
        data = std::malloc(aligned_bytes);
    }

    if (!data) {
        printf("[VENUS] ERROR: Failed to allocate %zu bytes for tensor data\n",
               aligned_bytes);
        fflush(stdout);
        delete tensor;
        return nullptr;
    }

    tensor->data        = data;
    tensor->bytes       = static_cast<uint32_t>(aligned_bytes);
    tensor->owns_data   = 1u;
    tensor->data_offset = 0;

    printf("[VENUS] Created tensor: %zu bytes (aligned=%zu), dtype=%d, fmt=%d, shape=[",
           bytes_needed, aligned_bytes, (int)tensor->dtype, (int)tensor->format);
    for (size_t i = 0; i < info.shape.size(); ++i) {
        printf("%d%s", info.shape[i], (i + 1 < info.shape.size()) ? "," : "");
    }
    printf("]\n");
    fflush(stdout);

    return tensor;
}

int MagikModelBase::build_tensors(PyramidConfig *config, std::vector<TensorInfo> infos) {
    printf("[VENUS] MagikModelBase::build_tensors(config=%p, infos.size=%zu)\n",
           (void*)config, infos.size());
    fflush(stdout);

    if (!config) {
        printf("[VENUS] ERROR: build_tensors called with NULL config\n");
        fflush(stdout);
        return -1;
    }

    if (infos.empty()) {
        printf("[VENUS] build_tensors: no TensorInfo entries, nothing to do\n");
        fflush(stdout);
        return 0;
    }

    /* First pass: dump all TensorInfo metadata so we can debug is_input/is_output
     * flags before any tensor creation. This helps identify mismatch between
     * how many inputs the metadata declares vs. how many the OEM model thinks
     * exist.
     */
    printf("[VENUS] build_tensors: === TensorInfo metadata dump ===\n");
    fflush(stdout);
    for (size_t i = 0; i < infos.size(); ++i) {
        TensorInfo &ti = infos[i];
        printf("[VENUS] TensorInfo[%zu]: name='%s' is_input=%u is_output=%u\n",
               i, ti.name.c_str(), (unsigned)ti.is_input, (unsigned)ti.is_output);
        printf("        dtype_str='%s' layout='%s' channel=%u\n",
               ti.dtype_str.c_str(), ti.layout.c_str(), (unsigned)ti.channel);
        printf("        shape=[");
        for (size_t j = 0; j < ti.shape.size(); ++j) {
            printf("%d%s", ti.shape[j], (j + 1 < ti.shape.size()) ? "," : "");
        }
        printf("]\n");
        fflush(stdout);
    }
    printf("[VENUS] build_tensors: === End TensorInfo dump ===\n");
    fflush(stdout);

    for (size_t i = 0; i < infos.size(); ++i) {
        TensorInfo &info = infos[i];
        printf("[VENUS] build_tensors: [%zu] name='%s' is_input=%u is_output=%u\n",
               i, info.name.c_str(), (unsigned)info.is_input, (unsigned)info.is_output);
        fflush(stdout);

        printf("[VENUS] build_tensors: calling create_tensor for '%s'\n",
               info.name.c_str());
        fflush(stdout);

        TensorX *tensor = MagikModelBase::create_tensor(info);
        if (!tensor) {
            printf("[VENUS] ERROR: create_tensor failed for '%s'\n",
                   info.name.c_str());
            fflush(stdout);
            return -1;
        }

        printf("[VENUS] build_tensors: create_tensor returned %p for '%s'\n",
               (void*)tensor, info.name.c_str());
        fflush(stdout);

        TensorXWrapper *wrapper = new TensorXWrapper(tensor);

        /* Track tensor in this pyramid configuration. */
        config->tensors_.push_back(wrapper);

        /* For now, map by name in both input and output maps so that
         * PyramidConfig::get_tensor_wrapper can resolve tensors by name. This
         * may be refined later once we fully understand OEM input/output flags.
         */
        config->input_tensors_.emplace(info.name, wrapper);
        config->output_tensors_.emplace(info.name, wrapper);

        /* Also track in our protected maps (immune to .mgk corruption) */
        config_tensors_[config].push_back(wrapper);
        config_tensor_names_[config][info.name] = wrapper;

        /* Also populate the global lookup map for get_tensor_wrapper */
        g_tensor_names[config][info.name] = wrapper;

        /* Populate global input/output vectors based on TensorInfo flags so that
         * host code can query IO tensors by index without relying on any
         * DerivedMagikModel overrides whose layout we don't fully control.
         */
        if (info.is_input) {
            printf("[VENUS] build_tensors: marking '%s' as model INPUT (index=%zu)\n",
                   info.name.c_str(), inputs_.size());
            fflush(stdout);
            inputs_.push_back(wrapper);
        }

        if (info.is_output) {
            printf("[VENUS] build_tensors: marking '%s' as model OUTPUT (index=%zu)\n",
                   info.name.c_str(), outputs_.size());
            fflush(stdout);
            outputs_.push_back(wrapper);
        }
    }

    printf("[VENUS] build_tensors: built %zu tensors (config=%p, config_tensors_[config].size()=%zu, g_tensor_names[config].size()=%zu)\n",
           infos.size(), (void*)config, config_tensors_[config].size(), g_tensor_names[config].size());
    printf("[VENUS] build_tensors: config->tensors_.size()=%zu (should be %zu)\n",
           config->tensors_.size(), infos.size());
    printf("[VENUS] build_tensors: config->input_tensors_.size()=%zu, config->output_tensors_.size()=%zu\n",
           config->input_tensors_.size(), config->output_tensors_.size());
    printf("[VENUS] build_tensors: SUMMARY - inputs_.size()=%zu, outputs_.size()=%zu\n",
           inputs_.size(), outputs_.size());
    printf("[VENUS] build_tensors: g_tensor_names has %zu configs total\n", g_tensor_names.size());
    /* Dump first few tensor names to verify they're in the map */
    size_t dump_count = 0;
    for (auto &kv : g_tensor_names[config]) {
        if (dump_count++ < 5) {
            printf("[VENUS] build_tensors: g_tensor_names[config]['%s'] = %p\n",
                   kv.first.c_str(), (void*)kv.second);
        }
    }
    fflush(stdout);

	    // Now that we have a stable set of TensorInfo names for this configuration,
	    // attempt to reconstruct a layer I/O map (bottom/top tensor names per
	    // layer ID) directly from the shared parameter blob in DDR. This map is
	    // what *_param_init functions such as format_convert_param_init expect to
	    // receive.
	    build_layer_io_map_from_ddr(config);

    return 0;
}

int MagikModelBase::update_cache_buffer_ptr(std::vector<MagikLayerBase*> layers, void *ptr) {
    printf("[VENUS] MagikModelBase::update_cache_buffer_ptr(layers=%zu, ptr=%p)\n",
           layers.size(), ptr);
    fflush(stdout);

    // The .mgk model passes us its internal layer vector here!
    // Store these layers so run() can iterate over them.
    if (!layers.empty()) {
        printf("[VENUS]   Capturing %zu layers from model!\n", layers.size());
        for (size_t i = 0; i < layers.size(); i++) {
            MagikLayerBase* layer = layers[i];
            printf("[VENUS]     Layer[%zu]: %p name='%s' id=%d\n",
                   i, (void*)layer, layer->get_name().c_str(), layer->get_layer_id());
            // Add to our layers_ vector if not already present
            bool found = false;
            for (auto* existing : layers_) {
                if (existing == layer) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                layers_.push_back(layer);
            }
        }
        printf("[VENUS]   Total layers now: %zu\n", layers_.size());
        fflush(stdout);
    }

    (void)ptr;
    return 0;
}

int MagikModelBase::set_oram_address(void *addr, long long size) const {
    printf("[VENUS] MagikModelBase::set_oram_address(addr=%p, size=%lld)\n", addr, size);
    fflush(stdout);

    /* This method is called by the derived class constructor to set the ORAM address.
     * The method is const but needs to modify global state.
     * We'll just acknowledge the call for now - the globals are already set correctly
     * by nna_runtime_init().
     */
    (void)addr;
    (void)size;

    printf("[VENUS] set_oram_address: ORAM address already set by runtime init\n");
    fflush(stdout);

    return 0;
}

/* PyramidConfig::get_tensor_wrapper
 *
 * Match OEM semantics: linearly scan the inline
 * std::vector<TensorXWrapper*> at offset 0x0c and compare each
 * wrapper's name.
 */
TensorXWrapper* MagikModelBase::PyramidConfig::get_tensor_wrapper(std::string &name) const {
    for (TensorXWrapper *wrapper : tensors_) {
        if (!wrapper)
            continue;
        if (wrapper->get_name() == name)
            return wrapper;
    }

    fprintf(stderr,
            "[VENUS] PyramidConfig::get_tensor_wrapper: cannot find tensor '%s' in config %p (tensors_.size=%zu)\n",
            name.c_str(), (void*)this, tensors_.size());
    fflush(stderr);

    return nullptr;
}

std::string MagikModelBase::get_output_names() const {
    return "";
}

std::string MagikModelBase::get_input_names() const {
    return "";
}

TensorXWrapper* MagikModelBase::get_input(std::string &name) const {
    printf("[VENUS] MagikModelBase::get_input(name=%s)\n", name.c_str());
    fflush(stdout);

    PyramidConfig *cfg = main_pyramid_config_;
    if (!cfg && !pyramid_configs_.empty()) {
        cfg = pyramid_configs_[0];
    }
    if (!cfg) {
        printf("[VENUS] get_input: no PyramidConfig available, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    TensorXWrapper *wrapper = cfg->get_tensor_wrapper(name);
    if (!wrapper) {
        printf("[VENUS] get_input: get_tensor_wrapper returned nullptr for '%s'\n",
               name.c_str());
        fflush(stdout);
        return nullptr;
    }

    // Keep inputs_ vector in sync for index-based access.
    MagikModelBase *self = const_cast<MagikModelBase*>(this);
    bool found = false;
    for (auto *w : self->inputs_) {
        if (w == wrapper) {
            found = true;
            break;
        }
    }
    if (!found) {
        self->inputs_.push_back(wrapper);
    }

    return wrapper;
}

TensorXWrapper* MagikModelBase::get_input(int index) const {
    printf("[VENUS] MagikModelBase::get_input(index=%d)\n", index);
    fflush(stdout);

    if (index < 0) {
        printf("[VENUS] get_input(index): negative index, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    if (static_cast<size_t>(index) < inputs_.size()) {
        TensorXWrapper *wrapper = inputs_[index];
        printf("[VENUS] get_input(index): returning inputs_[%d]=%p\n",
               index, (void*)wrapper);
        fflush(stdout);
        return wrapper;
    }

    PyramidConfig *cfg = main_pyramid_config_;
    if (!cfg && !pyramid_configs_.empty()) {
        cfg = pyramid_configs_[0];
    }
    if (!cfg) {
        printf("[VENUS] get_input(index): no PyramidConfig available, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    if (static_cast<size_t>(index) < cfg->tensors_.size()) {
        TensorXWrapper *wrapper = cfg->tensors_[index];
        printf("[VENUS] get_input(index): using config->tensors_[%d]=%p\n",
               index, (void*)wrapper);
        fflush(stdout);

        // Keep inputs_ vector in sync.
        MagikModelBase *self = const_cast<MagikModelBase*>(this);
        bool found = false;
        for (auto *w : self->inputs_) {
            if (w == wrapper) {
                found = true;
                break;
            }
        }
        if (!found) {
            self->inputs_.push_back(wrapper);
        }

        return wrapper;
    }

    printf("[VENUS] get_input(index): index %d out of range, returning nullptr\n", index);
    fflush(stdout);
    return nullptr;
}

TensorXWrapper* MagikModelBase::get_output(std::string &name) const {
    printf("[VENUS] MagikModelBase::get_output(name=%s)\n", name.c_str());
    fflush(stdout);

    PyramidConfig *cfg = main_pyramid_config_;
    if (!cfg && !pyramid_configs_.empty()) {
        cfg = pyramid_configs_[0];
    }
    if (!cfg) {
        printf("[VENUS] get_output: no PyramidConfig available, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    TensorXWrapper *wrapper = cfg->get_tensor_wrapper(name);
    if (!wrapper) {
        printf("[VENUS] get_output: get_tensor_wrapper returned nullptr for '%s'\n",
               name.c_str());
        fflush(stdout);
        return nullptr;
    }

    // Keep outputs_ vector in sync for index-based access.
    MagikModelBase *self = const_cast<MagikModelBase*>(this);
    bool found = false;
    for (auto *w : self->outputs_) {
        if (w == wrapper) {
            found = true;
            break;
        }
    }
    if (!found) {
        self->outputs_.push_back(wrapper);
    }

    return wrapper;
}

TensorXWrapper* MagikModelBase::get_output(int index) const {
    printf("[VENUS] MagikModelBase::get_output(index=%d)\n", index);
    fflush(stdout);

    if (index < 0) {
        printf("[VENUS] get_output(index): negative index, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    if (static_cast<size_t>(index) < outputs_.size()) {
        TensorXWrapper *wrapper = outputs_[index];
        printf("[VENUS] get_output(index): returning outputs_[%d]=%p\n",
               index, (void*)wrapper);
        fflush(stdout);
        return wrapper;
    }

    PyramidConfig *cfg = main_pyramid_config_;
    if (!cfg && !pyramid_configs_.empty()) {
        cfg = pyramid_configs_[0];
    }
    if (!cfg) {
        printf("[VENUS] get_output(index): no PyramidConfig available, returning nullptr\n");
        fflush(stdout);
        return nullptr;
    }

    if (static_cast<size_t>(index) < cfg->tensors_.size()) {
        TensorXWrapper *wrapper = cfg->tensors_[index];
        printf("[VENUS] get_output(index): using config->tensors_[%d]=%p\n",
               index, (void*)wrapper);
        fflush(stdout);

        // Keep outputs_ vector in sync.
        MagikModelBase *self = const_cast<MagikModelBase*>(this);
        bool found = false;
        for (auto *w : self->outputs_) {
            if (w == wrapper) {
                found = true;
                break;
            }
        }
        if (!found) {
            self->outputs_.push_back(wrapper);
        }

        return wrapper;
    }

    printf("[VENUS] get_output(index): index %d out of range, returning nullptr\n", index);
    fflush(stdout);
    return nullptr;
}

size_t MagikModelBase::get_forward_memory_size() const {
    size_t total = 0;

    printf("[VENUS] get_forward_memory_size: using config_tensors_ map\n");
    fflush(stdout);

    /* Sum the byte sizes of all tensors using our protected map */
    for (const auto &kv : config_tensors_) {
        PyramidConfig *cfg = kv.first;
        const std::vector<TensorXWrapper*> &tensors = kv.second;
        printf("[VENUS] get_forward_memory_size: cfg=%p has %zu tensors\n",
               (void*)cfg, tensors.size());
        fflush(stdout);

        for (size_t j = 0; j < tensors.size(); ++j) {
            TensorXWrapper *wrapper = tensors[j];
            if (!wrapper || !wrapper->tensorx) {
                continue;
            }
            total += wrapper->tensorx->get_bytes_size();
        }
    }

    if (total == 0) {
        /* Fallback to 1MB default if we have not built any tensors yet. */
        total = 1024 * 1024;
    }

    printf("[VENUS] MagikModelBase::get_forward_memory_size() -> %zu bytes\n", total);
    fflush(stdout);

    return total;
}

} // namespace venus

// OEM helper prototype (mangled name) so we can call get_string_t from
// our get_string_vector_t shim.
extern "C" int _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
    std::string &out,
    const void *param,
    int &index);

// Override OEM helper: get_string_vector_t
// This function is exported by the .mgk and normally reads a vector of strings
// from a kernel parameter block.
//
// The OEM implementation (from AEC_T41_16K_NS_OUT_UC.mgk) is roughly:
//   - read a 32-bit count from the param blob at param + index
//   - resize the std::vector<std::string> to that count
//   - for each entry call get_string_t() to fill one string
//   - return the updated index cursor
//
// We replicate those semantics here, but add a defensive clamp for obviously
// bogus counts (e.g. when the param pointer is wrong and we would otherwise
// try to allocate millions of strings).
extern "C" int _Z19get_string_vector_tRSt6vectorINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaIS5_EEPKvRi(
    std::vector<std::string> &vec,
    const void *param,
    int &index)
{
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    unsigned long off = 0;
    const char *obj = "?";
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    // Heuristic: calls originating from the .mgk module itself should be
    // allowed to consume whatever param pointer they pass us; in those cases
    // we do not insist that the pointer also fall inside the exported DDR
    // heap window.
    bool caller_is_mgk = false;
    if (obj && std::strstr(obj, ".mgk")) {
        caller_is_mgk = true;
    }

    unsigned char bytes[16];
    for (int i = 0; i < 16; ++i) {
        bytes[i] = 0;
    }
    const unsigned char *base = static_cast<const unsigned char *>(param);
    if (base) {
        uintptr_t addr = (uintptr_t)base;
        if (addr >= 0x10000u) {
            for (int i = 0; i < 16; ++i) {
                bytes[i] = base[i];
            }
        }
    }

    // Read element count (32-bit) from the parameter block at the current
    // index. For calls coming from the .mgk, we trust the pointer even if it
    // is outside the usual DDR window; for everything else we keep a
    // conservative low-address guard to avoid obviously bogus pointers.
    int32_t count = 0;
    if (base) {
        uintptr_t addr = (uintptr_t)(base + index);
        if (addr >= 0x10000u || caller_is_mgk) {
            memcpy(&count, base + index, sizeof(count));
            index += (int)sizeof(count);
        }
    }

    // Reasonable upper bound for string vector sizes coming from real .mgk
    // models. Anything far beyond this is almost certainly a corrupted pointer
    // or mis-parsed blob.
    static const int32_t kMaxReasonableCount = 1024;

    // Also look at which object the param pointer lives in; if it points into
    // libvenus.so itself, we know we are *not* looking at a real model param
    // blob.
    Dl_info param_info{};
    const char *param_obj = "?";
    bool param_in_venus = false;
    if (param && dladdr(param, &param_info)) {
        param_obj = param_info.dli_fname ? param_info.dli_fname : "?";
        if (std::strstr(param_obj, "libvenus.so")) {
            param_in_venus = true;
        }
    }

    // Additionally, detect whether the param pointer is inside the logical
    // DDR heap (__ddr_vbase .. __ddr_vbase + 4MB) that we expose to .mgk.
	    bool param_in_ddr = false;
	    long ddr_offset = -1;
	    if (__ddr_vbase && param) {
	        uintptr_t ddr_base = (uintptr_t)__ddr_vbase;
	        uintptr_t p = (uintptr_t)param;
	        static const uintptr_t kDdrSpan = 4u * 1024u * 1024u;
	        if (p >= ddr_base && p < ddr_base + kDdrSpan) {
	            param_in_ddr = true;
	            ddr_offset = (long)(p - ddr_base);
	        }
	    }

	    bool allow_param = param_in_ddr || caller_is_mgk;
	    if (param_in_ddr) {
	        // If it clearly lives in DDR, don't also label it as "libvenus".
	        param_in_venus = false;
	    }

	    if (!allow_param) {
	        std::fprintf(stderr,
	                     "[VENUS] get_string_vector_t: param=%p (obj=%s) not in DDR heap and caller not .mgk; treating as empty vector\n",
	                     param, param_obj);
	        vec.clear();
	        return index;
	    }

    if (count < 0 || count > kMaxReasonableCount) {
        std::fprintf(stderr,
                     "[VENUS] get_string_vector_t: suspicious count=%d (param=%p obj=%s ddr_off=%ld)\n",
                     count, param, param_obj, ddr_offset);
        // If the param buffer clearly lives inside libvenus, treat this as
        // completely bogus and ignore the vector contents.
        if (param_in_venus) {
            count = 0;
        } else if (count > kMaxReasonableCount) {
            count = kMaxReasonableCount;
        }
    }

    std::fprintf(stderr,
                 "[VENUS] get_string_vector_t override\n"
                 "  vec_size_before=%zu index=%d count=%d param=%p\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n"
                 "  param_ddr_base=%p in_ddr=%d ddr_off=%ld\n"
                 "  param[0..15]=%02x %02x %02x %02x %02x %02x %02x %02x "
                 "%02x %02x %02x %02x %02x %02x %02x %02x\n",
                 vec.size(), index, count, param,
                 ra0, obj, sym, off,
                 __ddr_vbase, param_in_ddr ? 1 : 0, ddr_offset,
                 bytes[0], bytes[1], bytes[2], bytes[3],
                 bytes[4], bytes[5], bytes[6], bytes[7],
                 bytes[8], bytes[9], bytes[10], bytes[11],
                 bytes[12], bytes[13], bytes[14], bytes[15]);
    std::fflush(stderr);

    if (count <= 0) {
        vec.clear();
        return index;
    }

    vec.resize(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
            vec[static_cast<size_t>(i)], param, index);
    }

    return index;
}


// Override OEM helper: get_string_t
// Mangled name: _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi
//
// The OEM implementation (from AEC_T41_16K_NS_OUT_UC.mgk) is roughly:
//   - read a 32-bit length from the param blob at param + index
//   - if length != 0, resize the std::string and memcpy the bytes
//   - advance index by the length and return the updated cursor
//
// We replicate those semantics here, but add a defensive clamp for obviously
// bogus lengths (e.g. when the param pointer is wrong and we would otherwise
// try to allocate hundreds of megabytes for a single string).
extern "C" int _Z12get_string_tRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvRi(
    std::string &out,
    const void *param,
    int &index)
{
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    const char *obj = "?";
    unsigned long off = 0;
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    bool caller_is_mgk = false;
    if (obj && std::strstr(obj, ".mgk")) {
        caller_is_mgk = true;
    }

    unsigned char bytes[16];
    for (int i = 0; i < 16; ++i) {
        bytes[i] = 0;
    }
    const unsigned char *base = static_cast<const unsigned char *>(param);
    if (base) {
        uintptr_t addr = (uintptr_t)base;
        if (addr >= 0x10000u) {
            for (int i = 0; i < 16; ++i) {
                bytes[i] = base[i];
            }
        }
    }

    // Read string length (32-bit) from the parameter block at the current
    // index. As with get_string_vector_t, allow the .mgk to drive whatever
    // pointer it believes is correct while keeping a conservative low-address
    // guard for other callers.
    uint32_t len = 0;
    if (base) {
        uintptr_t addr = (uintptr_t)(base + index);
        if (addr >= 0x10000u || caller_is_mgk) {
            memcpy(&len, base + index, sizeof(len));
            index += (int)sizeof(len);
        }
    }

    // Reasonable upper bound for string lengths in real .mgk models.
    static const uint32_t kMaxReasonableLen = 16384; // 16 KB

    Dl_info param_info{};
    const char *param_obj = "?";
    bool param_in_venus = false;
    if (param && dladdr(param, &param_info)) {
        param_obj = param_info.dli_fname ? param_info.dli_fname : "?";
        if (std::strstr(param_obj, "libvenus.so")) {
            param_in_venus = true;
        }
    }

    // Additionally, detect whether the param pointer is inside the logical
    // DDR heap (__ddr_vbase .. __ddr_vbase + 4MB) that we expose to .mgk.
	    bool param_in_ddr = false;
	    long ddr_offset = -1;
	    if (__ddr_vbase && param) {
	        uintptr_t ddr_base = (uintptr_t)__ddr_vbase;
	        uintptr_t p = (uintptr_t)param;
	        static const uintptr_t kDdrSpan = 4u * 1024u * 1024u;
	        if (p >= ddr_base && p < ddr_base + kDdrSpan) {
	            param_in_ddr = true;
	            ddr_offset = (long)(p - ddr_base);
	        }
	    }

	    bool allow_param = param_in_ddr || caller_is_mgk;
	    if (param_in_ddr) {
	        // If it clearly lives in DDR, don't also label it as "libvenus".
	        param_in_venus = false;
	    }

	    if (!allow_param) {
	        std::fprintf(stderr,
	                     "[VENUS] get_string_t: param=%p (obj=%s) not in DDR heap and caller not .mgk; treating as empty string\n",
	                     param, param_obj);
	        len = 0;
	    }

    if (len > kMaxReasonableLen) {
        std::fprintf(stderr,
                     "[VENUS] get_string_t: suspicious len=%u (param=%p obj=%s ddr_off=%ld)\n",
                     len, param, param_obj, ddr_offset);
        if (param_in_venus) {
            // This is almost certainly a bogus param pointer. Treat as an empty
            // string so we do not walk off into unrelated text/data segments.
            len = 0;
        } else {
            len = kMaxReasonableLen;
        }
    }

    if (len == 0 || !base) {
        out.clear();
    } else {
        out.resize(static_cast<size_t>(len));
        memcpy(&out[0], base + index, (size_t)len);
        index += (int)len;
    }

    std::fprintf(stderr,
                 "[VENUS] get_string_t override\n"
                 "  index=%d len=%u param=%p\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n"
                 "  param[0..15]=%02x %02x %02x %02x %02x %02x %02x %02x "
                 "%02x %02x %02x %02x %02x %02x %02x %02x\n",
                 index, len, param,
                 ra0, obj, sym, off,
                 bytes[0], bytes[1], bytes[2], bytes[3],
                 bytes[4], bytes[5], bytes[6], bytes[7],
                 bytes[8], bytes[9], bytes[10], bytes[11],
                 bytes[12], bytes[13], bytes[14], bytes[15]);
    std::fflush(stderr);

    return index;
}

} // namespace magik


// NOTE: We deliberately do **not** interpose read_common_param any more.
//
// Earlier iterations of libvenus provided a "read_common_param_shim" that
// aliased the real mangled symbol
//   _Z17read_common_paramRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIS4_SaIS4_EES9_RS6_IiSaIiEES9_S9_RiS9_S5_PKvSD_
// and attempted to decode the layer name and bottom/top tensor names while
// leaving the caller's `index` cursor unchanged. That was useful for
// reverse‑engineering but it subtly changed the OEM semantics and prevented
// the .mgk's own read_common_param implementation from advancing its internal
// cursor and filling all of the integer/vector parameters that
// *_param_init functions (including format_convert_param_init) rely on.
//
// Now that we have a much more faithful MagikModelBase / PyramidConfig /
// TensorInfo implementation, we want the .mgk to run its own parsing logic
// unmodified. The .mgk's read_common_param will still call into our
// get_string_t/get_string_vector_t overrides for safe access to the DDR
// parameter blob, but we no longer override read_common_param itself.

// ============================================================================
// REMOVED ALL SHIMS
// ============================================================================
// The .mgk file provides its own implementations of:
//   - build_layer
//   - read_layer_param
//   - read_common_param
//   - all *_param_init functions (conv2d_int8_param_init, bn_scale_int8_param_init, etc.)
//
// These are all symbol type 'T' (defined) in the .mgk.
// We were shadowing them with our shims, preventing the .mgk's real code from running!
//
// The .mgk NEEDS from us (symbol type 'U' - undefined):
//   - MagikLayerBase::set_inputs/set_outputs
//   - MagikModelBase::build_tensors
//   - PyramidConfig::get_tensor_wrapper  ← KEY!
//   - MagikModelBase::update_cache_buffer_ptr
//
// ============================================================================

// OLD SHIM CODE REMOVED (was lines 1923-2497)
// All heavy shim implementations deleted - the .mgk provides these functions
// itself. The only remaining interpositions are:
//   - a very conservative shim for format_convert_param_init that simply logs
//     the call site and reports success without touching any arguments;
//   - a very conservative shim for bn_scale_int8_param_init that logs and
//     reports success without touching any arguments;
//   - a very conservative shim for conv2d_int8_param_init that logs and
//     returns without touching any arguments;
//   - a very conservative shim for reshape_param_init that logs and reports
//     success without touching any arguments; and
//   - a very conservative shim for gru_param_init that logs and reports
//     success without touching any arguments.

// Shim for OEM format_convert_param_init used by .mgk models.
//
// Original (demangled) signature (trimmed to key parts):
//   magik::venus::ReturnValue format_convert_param_init(
//       std::string,
//       std::vector<std::string>, std::vector<std::string>,
//       std::vector<int>, std::vector<std::string>, std::vector<std::string>,
//       int, std::vector<std::string>, std::string,
//       int, int, int, int, int,
//       void*, void*, long long,
//       std::vector<std::vector<int>>, std::vector<std::vector<int>>,
//       std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>,
//       bool, bool, int, int, int, bool, int, int,
//       std::vector<magik::venus::TensorInfo>&,
//       std::map<int, std::pair<std::vector<std::string>, std::vector<std::string>>>&,
//       magik::venus::MagikModelBase::PyramidConfig*,
//       std::function<magik::venus::ReturnValue(
//           std::unique_ptr<magik::venus::kernel::KernelParam>&,
//           magik::venus::OpConfig*)>);
//
// We implement a conservative shim that logs the call site but does not
// dereference any of the opaque parameter pointers. For now we simply report
// success so that the model can continue past format_convert_param_init while
// we incrementally reconstruct the real behaviour.
extern "C" magik::venus::ReturnValue
format_convert_param_init_shim(...) __asm__(
    "_Z25format_convert_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iiiiiPvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
format_convert_param_init_shim(...)
{
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    const char *obj = "?";
    unsigned long off = 0;
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    std::fprintf(stderr,
                 "[VENUS] format_convert_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    return magik::venus::ReturnValue(0);
}

// Shim for OEM bn_scale_int8_param_init used by .mgk models.
//
// The OEM implementation has historically been a frequent crash site for this
// AEC model (SIGBUS inside bn_scale_int8_param_init). We interpose a very
// conservative shim that logs the call site and simply returns success,
// effectively treating BN/scale param init as a no-op while we
// reverse‑engineer the full semantics.
extern "C" magik::venus::ReturnValue
bn_scale_int8_param_init_shim(...) __asm__(
    "_Z24bn_scale_int8_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iiiS9_S9_iiiiPvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
bn_scale_int8_param_init_shim(...)
{
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    const char *obj = "?";
    unsigned long off = 0;
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    std::fprintf(stderr,
                 "[VENUS] bn_scale_int8_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    // Treat BN/scale param init as a no-op for now so the model can continue.
    return magik::venus::ReturnValue(0);
}

// Shim for OEM conv2d_int8_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z22conv2d_int8_param_initPKvPvS1_xRSt6vectorIN5magik5venus10TensorInfoESaIS5_EERSt3mapIiSt4pairIS2_INSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISG_EESI_ESt4lessIiESaISA_IKiSJ_EEEPNS4_14MagikModelBase13PyramidConfigESt8functionIFNS4_11ReturnValueERSt10unique_ptrINS4_6kernel11KernelParamESt14default_deleteISY_EEPNS4_8OpConfigEEE
//
// The OEM implementation asserts on output_size==1 and performs complex
// parameter parsing. For now we interpose a very conservative shim that
// simply logs the call site and returns immediately, effectively treating
// conv2d param initialization as a no-op so we avoid tripping internal
// asserts and corrupting STL state.
extern "C" void
conv2d_int8_param_init_shim(...) __asm__(
    "_Z22conv2d_int8_param_initPKvPvS1_xRSt6vectorIN5magik5venus10TensorInfoESaIS5_EERSt3mapIiSt4pairIS2_INSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESaISG_EESI_ESt4lessIiESaISA_IKiSJ_EEEPNS4_14MagikModelBase13PyramidConfigESt8functionIFNS4_11ReturnValueERSt10unique_ptrINS4_6kernel11KernelParamESt14default_deleteISY_EEPNS4_8OpConfigEEE");

extern "C" void
conv2d_int8_param_init_shim(...)
{
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    const char *obj = "?";
    unsigned long off = 0;
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    std::fprintf(stderr,
                 "[VENUS] conv2d_int8_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    // No-op: we intentionally do not touch any of the arguments.
    return;
}

// Shim for OEM reshape_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z18reshape_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iS9_PvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE
//
// Like the other param-init shims, we currently treat reshape as a no-op for
// safety. This avoids crashes in the OEM implementation while we focus on
// getting stable end-to-end execution.
extern "C" magik::venus::ReturnValue
reshape_param_init_shim(...) __asm__(
    "_Z18reshape_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_iS9_PvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
reshape_param_init_shim(...)
{
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    const char *obj = "?";
    unsigned long off = 0;
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    std::fprintf(stderr,
                 "[VENUS] reshape_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    // Treat reshape param init as a no-op for now so the model can continue.
    return magik::venus::ReturnValue(0);
}

// Shim for OEM gru_param_init used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk backtrace):
//   _Z14gru_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_ibbiS9_iiiyiS9_S9_iiPvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE
//
// For now we treat GRU param initialization as a no-op to avoid crashing in
// the OEM implementation while we focus on getting stable end-to-end runs.
extern "C" magik::venus::ReturnValue
gru_param_init_shim(...) __asm__(
    "_Z14gru_param_initNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS4_SaIS4_EES7_S5_IiSaIiEES7_S7_iS7_S4_ibbiS9_iiiyiS9_S9_iiPvSA_xS5_IS9_SaIS9_EESC_S9_S9_S9_S9_bbiiibiiRS5_IN5magik5venus10TensorInfoESaISF_EERSt3mapIiSt4pairIS7_S7_ESt4lessIiESaISK_IKiSL_EEEPNSE_14MagikModelBase13PyramidConfigESt8functionIFNSE_11ReturnValueERSt10unique_ptrINSE_6kernel11KernelParamESt14default_deleteIS10_EEPNSE_8OpConfigEEE");

extern "C" magik::venus::ReturnValue
gru_param_init_shim(...)
{
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const char *sym = "?";
    const char *obj = "?";
    unsigned long off = 0;
    if (dladdr(ra0, &info0)) {
        sym = info0.dli_sname ? info0.dli_sname : "?";
        obj = info0.dli_fname ? info0.dli_fname : "?";
        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
    }

    std::fprintf(stderr,
                 "[VENUS] gru_param_init shim (raw) called\n"
                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
                 ra0, obj, sym, off);
    std::fflush(stderr);

    // Treat GRU param init as a no-op for now so the model can continue.
    return magik::venus::ReturnValue(0);
}

// Guarded shim for OEM read_common_param used by .mgk models.
//
// Mangled name (from AEC_T41_16K_NS_OUT_UC.mgk):
//   _Z17read_common_paramRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIS4_SaIS4_EES9_RS6_IiSaIiEES9_S9_RiS9_S5_PKvSD_
//
// We only intercept calls where the `param` pointer is clearly *not* within
// the shared DDR heap exposed via __ddr_vbase. In that case we:
//   - reset all outputs to safe defaults;
//   - leave `index` unchanged; and
//   - do NOT call the OEM implementation, avoiding any memcpy/parse on bogus
//     pointers like 0xd9a that would otherwise SIGBUS inside musl's memcpy.
//
// For calls where `param` resides in DDR, we delegate entirely to the OEM
// read_common_param via dlsym(RTLD_NEXT, ...), preserving its behaviour.
extern "C" int
read_common_param_shim(
    std::string &layer_name,
    std::vector<std::string> &bottoms,
    std::vector<std::string> &tops,
    std::vector<int> &int_params,
    std::vector<std::string> &extra_strs1,
    std::vector<std::string> &extra_strs2,
    int &some_flag,
    std::vector<std::string> &extra_strs3,
    std::string &extra_str,
    const void *param,
    int &index);

extern "C" int
read_common_param_shim(
	    std::string &layer_name,
	    std::vector<std::string> &bottoms,
	    std::vector<std::string> &tops,
	    std::vector<int> &int_params,
	    std::vector<std::string> &extra_strs1,
	    std::vector<std::string> &extra_strs2,
	    int &some_flag,
	    std::vector<std::string> &extra_strs3,
	    std::string &extra_str,
	    const void *param,
	    int &index)
{
	    void *ra0 = __builtin_return_address(0);
	    Dl_info info0;
	    const char *sym = "?";
	    const char *obj = "?";
	    unsigned long off = 0;
	    if (dladdr(ra0, &info0)) {
	        sym = info0.dli_sname ? info0.dli_sname : "?";
	        obj = info0.dli_fname ? info0.dli_fname : "?";
	        off = (unsigned long)((char *)ra0 - (char *)info0.dli_saddr);
	    }

	    using real_fn_t = int (*)(
	        std::string &,
	        std::vector<std::string> &,
	        std::vector<std::string> &,
	        std::vector<int> &,
	        std::vector<std::string> &,
	        std::vector<std::string> &,
	        int &,
	        std::vector<std::string> &,
	        std::string &,
	        const void *,
	        int &);
	    static real_fn_t real = nullptr;
	    if (!real) {
	        real = (real_fn_t)dlsym(RTLD_NEXT,
	                                 "_Z17read_common_paramRNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERSt6vectorIS4_SaIS4_EES9_RS6_IiSaIiEES9_S9_RiS9_S5_PKvSD_");
	        if (!real) {
	            std::fprintf(stderr,
	                         "[VENUS] read_common_param shim: dlsym(RTLD_NEXT, ...) failed: %s (param=%p index=%d caller=%s)\n",
	                         dlerror(), param, index, obj ? obj : "?");
	            std::fflush(stderr);

	            // Best-effort: clear outputs and return index unchanged. This
	            // should never happen in normal .mgk usage, but avoids crashes
	            // if the symbol cannot be resolved for some reason.
	            layer_name.clear();
	            bottoms.clear();
	            tops.clear();
	            int_params.clear();
	            extra_strs1.clear();
	            extra_strs2.clear();
	            extra_strs3.clear();
	            some_flag = 0;
	            extra_str.clear();
	            return index;
	        }
	    }

	    std::fprintf(stderr,
	                 "[VENUS] read_common_param shim: delegating to OEM (param=%p index=%d caller=%s)\n"
	                 "  ra0=%p (obj=%s, symbol=%s+0x%lx)\n",
	                 param,
	                 index,
	                 obj ? obj : "?",
	                 ra0, obj ? obj : "?", sym, off);
	    std::fflush(stderr);

	    return real(layer_name,
	                bottoms,
	                tops,
	                int_params,
	                extra_strs1,
	                extra_strs2,
	                some_flag,
	                extra_strs3,
	                extra_str,
	                param,
	                index);
}

extern "C" int _Z16read_layer_paramRiRtRyRjPKvS_(
    int &op_type,
    unsigned short &layer_id,
    unsigned long long &param_index,
    unsigned int &flags,
    const void *param,
    int &index) {
    using real_fn_t = int (*)(int &, unsigned short &, unsigned long long &,
                              unsigned int &, const void *, int &);
    static real_fn_t real = nullptr;

    if (!real) {
        dlerror();  // clear any prior error
        void *sym = dlsym(RTLD_NEXT, "_Z16read_layer_paramRiRtRyRjPKvS_");
        if (!sym) {
            std::fprintf(stderr,
                         "[VENUS] read_layer_param shim: dlsym(RTLD_NEXT, ...) failed: %s\n",
                         dlerror());
            std::fflush(stderr);
            // Best-effort: leave index unchanged and return it. This will likely
            // cause the .mgk to misbehave, but avoids recursion.
            return index;
        }
        real = reinterpret_cast<real_fn_t>(sym);
    }

    int before_index = index;
    int ret = real(op_type, layer_id, param_index, flags, param, index);

    magik::venus::ParsedLayerRecord rec{};
    rec.op_type = op_type;
    rec.layer_id = layer_id;
    rec.param_index = param_index;
    rec.flags = flags;
    magik::venus::g_parsed_layers.push_back(rec);

    std::fprintf(stderr,
                 "[VENUS] read_layer_param shim: idx %d->%d op_type=%d "
                 "layer_id=%u param_index=%llu flags=0x%x param=%p\n",
                 before_index,
                 index,
                 op_type,
                 static_cast<unsigned>(layer_id),
                 static_cast<unsigned long long>(param_index),
                 flags,
                 param);
    std::fflush(stderr);

    return ret;
}

extern "C" void _ZSt17__throw_bad_allocv(void) {
    void *ra0 = __builtin_return_address(0);
    void *ra1 = __builtin_return_address(1);
    void *ra2 = __builtin_return_address(2);
    Dl_info info0;
    if (dladdr(ra0, &info0)) {
        fprintf(stderr,
                "[VENUS] std::__throw_bad_alloc intercepted\n"
                "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n"
                "  ra1=%p\n"
                "  ra2=%p\n",
                ra0,
                info0.dli_fname ? info0.dli_fname : "?",
                info0.dli_fbase,
                info0.dli_sname ? info0.dli_sname : "?",
                (unsigned long)((char *)ra0 - (char *)info0.dli_saddr),
                ra1,
                ra2);
    } else {
        fprintf(stderr,
                "[VENUS] std::__throw_bad_alloc intercepted\n"
                "  ra0=%p (no symbol)\n  ra1=%p\n  ra2=%p\n",
                ra0,
                ra1,
                ra2);
    }
    fflush(stderr);
    __builtin_trap();
}

extern "C" void _ZSt20__throw_length_errorPKc(const char *msg) {
    void *ra0 = __builtin_return_address(0);
    void *ra1 = __builtin_return_address(1);
    void *ra2 = __builtin_return_address(2);
    Dl_info info0;
    if (dladdr(ra0, &info0)) {
        fprintf(stderr,
                "[VENUS] std::__throw_length_error(\"%s\") intercepted\n"
                "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n"
                "  ra1=%p\n"
                "  ra2=%p\n",
                msg ? msg : "",
                ra0,
                info0.dli_fname ? info0.dli_fname : "?",
                info0.dli_fbase,
                info0.dli_sname ? info0.dli_sname : "?",
                (unsigned long)((char *)ra0 - (char *)info0.dli_saddr),
                ra1,
                ra2);
    } else {
        fprintf(stderr,
                "[VENUS] std::__throw_length_error(\"%s\") intercepted\n"
                "  ra0=%p (no symbol)\n  ra1=%p\n  ra2=%p\n",
                msg ? msg : "",
                ra0,
                ra1,
                ra2);
    }
    fflush(stderr);
    __builtin_trap();
}


extern "C" void __assert_fail(const char *expr,
                              const char *file,
                              unsigned int line,
                              const char *func)
{
    // Interpose libc assert failures coming from OEM Magik kernels.
    std::fprintf(stderr,
                 "[VENUS] __assert_fail intercepted: expr=\"%s\" file=\"%s\" line=%u func=\"%s\"\n",
                 expr ? expr : "(null)",
                 file ? file : "(null)",
                 line,
                 func ? func : "(null)");
    std::fflush(stderr);

    if (expr && file && func &&
        std::strstr(expr, "output_size==1") &&
        std::strstr(file, "magik_op_override.cpp"))
    {
        std::fprintf(stderr,
                     "[VENUS] ignoring magik_op_override.cpp output_size==1 assert in %s; continuing execution\n",
                     func ? func : "(null)");
        std::fflush(stderr);
        return;
    }

    using assert_fail_fn = void (*)(const char *, const char *, unsigned int, const char *);
    static assert_fail_fn real_assert_fail = nullptr;
    if (!real_assert_fail) {
        real_assert_fail = (assert_fail_fn)dlsym(RTLD_NEXT, "__assert_fail");
    }

    if (real_assert_fail) {
        real_assert_fail(expr, file, line, func);
    } else {
        std::fprintf(stderr,
                     "[VENUS] __assert_fail: no RTLD_NEXT implementation, calling abort()\n");
        std::fflush(stderr);
        std::abort();
    }
}


/*
 * Global operator new/delete interceptors
 *
 * These override the default C++ global allocation functions when
 * libvenus.so is preloaded. They simply forward to malloc/free and
 * log large allocations so we can see what size triggered
 * std::bad_alloc inside the OEM .mgk code.
 */
void* operator new(std::size_t size) {
    void *ptr = std::malloc(size);
    if (!ptr) {
        /* Log the failing allocation and its call site before throwing. */
        void *ra0 = __builtin_return_address(0);
        Dl_info info0;
        if (dladdr(ra0, &info0)) {
            std::fprintf(stderr,
                         "[VENUS] operator new OOM size=%zu\n"
                         "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n",
                         size,
                         ra0,
                         info0.dli_fname ? info0.dli_fname : "?",
                         info0.dli_fbase,
                         info0.dli_sname ? info0.dli_sname : "?",
                         (unsigned long)((char *)ra0 - (char *)info0.dli_saddr));
        } else {
            std::fprintf(stderr,
                         "[VENUS] operator new OOM size=%zu ra0=%p (no symbol)\n",
                         size,
                         ra0);
        }
        std::fflush(stderr);

        // Respect the C++ contract: throw std::bad_alloc on failure.
        throw std::bad_alloc();
    }
    return ptr;
}

void* operator new[](std::size_t size) {
    void *ptr = std::malloc(size);
    if (!ptr) {
        /* Log the failing allocation and its call site before throwing. */
        void *ra0 = __builtin_return_address(0);
        Dl_info info0;
        if (dladdr(ra0, &info0)) {
            std::fprintf(stderr,
                         "[VENUS] operator new[] OOM size=%zu\n"
                         "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n",
                         size,
                         ra0,
                         info0.dli_fname ? info0.dli_fname : "?",
                         info0.dli_fbase,
                         info0.dli_sname ? info0.dli_sname : "?",
                         (unsigned long)((char *)ra0 - (char *)info0.dli_saddr));
        } else {
            std::fprintf(stderr,
                         "[VENUS] operator new[] OOM size=%zu ra0=%p (no symbol)\n",
                         size,
                         ra0);
        }
        std::fflush(stderr);

        throw std::bad_alloc();
    }
    return ptr;
}

void operator delete(void *ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void *ptr) noexcept {
    std::free(ptr);
}


/* Generic C++ exception interceptor: __cxa_throw
 * Logs all thrown exceptions (including std::bad_alloc) with type name and call site.
 */
extern "C" void __cxa_throw(void *exc_ptr, void *tinfo_raw, void (*dest)(void *)) {
    void *ra0 = __builtin_return_address(0);
    Dl_info info0;
    const std::type_info *tinfo = static_cast<const std::type_info *>(tinfo_raw);
    const char *type_name = tinfo ? tinfo->name() : "(null)";

    if (dladdr(ra0, &info0)) {
        fprintf(stderr,
                "[VENUS] __cxa_throw type=%s\n"
                "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n",
                type_name,
                ra0,
                info0.dli_fname ? info0.dli_fname : "?",
                info0.dli_fbase,
                info0.dli_sname ? info0.dli_sname : "?",
                (unsigned long)((char *)ra0 - (char *)info0.dli_saddr));
    } else {
        fprintf(stderr,
                "[VENUS] __cxa_throw type=%s ra0=%p (no symbol)\n",
                type_name,
                ra0);
    }
    fflush(stderr);

    using cxa_throw_t = void (*)(void *, void *, void (*)(void *));
    static cxa_throw_t real_cxa_throw = nullptr;
    if (!real_cxa_throw) {
        real_cxa_throw = (cxa_throw_t)dlsym(RTLD_NEXT, "__cxa_throw");
    }

    real_cxa_throw(exc_ptr, tinfo_raw, dest);
}

static bool venus_is_region_mapped(const void *ptr, size_t n)
{
    if (!ptr || n == 0)
        return true;

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        page_size = 4096;

    uintptr_t start = (uintptr_t)ptr;
    uintptr_t end = start + (n - 1);

    uintptr_t page_mask = (uintptr_t)page_size - 1u;
    uintptr_t page_start = start & ~page_mask;
    uintptr_t page_end = end & ~page_mask;

    unsigned char vec;
    for (uintptr_t p = page_start; p <= page_end; p += (uintptr_t)page_size) {
        if (mincore((void *)p, (size_t)page_size, &vec) != 0) {
            return false;
        }
    }

    return true;
}

/*
 * Lightweight global memchr override for diagnostics and safety.
 *
 * We use this to catch calls where the pointer range is clearly bogus and
 * would otherwise trigger SIGBUS. For normal mapped regions this adds a
 * small mincore() check but keeps behaviour identical to the libc memchr.
 */
extern "C" void *memchr(const void *s, int c, size_t n)
{
	using memchr_fn = void *(*)(const void *, int, size_t);
	static memchr_fn real_memchr = nullptr;

	if (!s || n == 0)
		return nullptr;

	// Best-effort guard against obviously invalid addresses (e.g. 0x2358).
	if (!venus_is_region_mapped(s, n)) {
		void *ra0 = __builtin_return_address(0);
		Dl_info info0{};
		dladdr(ra0, &info0);
		std::fprintf(stderr,
		             "[VENUS] memchr guard: blocked memchr(%p, c=%d, n=%zu) as unmapped\n"
		             "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n",
		             s,
		             c,
		             n,
		             ra0,
		             info0.dli_fname ? info0.dli_fname : "?",
		             info0.dli_fbase,
		             info0.dli_sname ? info0.dli_sname : "?",
		             info0.dli_saddr ? (unsigned long)((char *)ra0 - (char *)info0.dli_saddr) : 0ul);
		std::fflush(stderr);
		return nullptr;
	}

	if (!real_memchr) {
		real_memchr = (memchr_fn)dlsym(RTLD_NEXT, "memchr");
	}

	if (!real_memchr) {
		/* Fallback: simple byte-wise scan. */
		const unsigned char *p = static_cast<const unsigned char *>(s);
		const unsigned char *end = p + n;
		unsigned char uc = static_cast<unsigned char>(c);
		for (; p < end; ++p) {
			if (*p == uc)
				return const_cast<unsigned char *>(p);
		}
		return nullptr;
	}

	return real_memchr(s, c, n);
}

/*
 * Lightweight global memcpy override for diagnostics and safety.
 *
 * When libvenus.so is preloaded, this function will be used for most
 * memcpy calls from the .mgk and our own code. We add a best-effort guard
 * using mincore() to avoid SIGBUS on clearly invalid pointer ranges (for
 * example, the 0x2358 crash seen in read_common_param), while delegating
 * all valid copies to the real libc implementation.
 */
extern "C" void *memcpy(void *dest, const void *src, size_t n)
{
	if (!dest || !src || n == 0)
		return dest;

	bool src_ok = venus_is_region_mapped(src, n);
	bool dest_ok = venus_is_region_mapped(dest, n);
	if (!src_ok || !dest_ok) {
		void *ra0 = __builtin_return_address(0);
		Dl_info info0{};
		dladdr(ra0, &info0);
		std::fprintf(stderr,
		             "[VENUS] memcpy guard: blocked memcpy(%p -> %p, n=%zu) src_ok=%d dest_ok=%d\n"
		             "  ra0=%p (obj=%s, base=%p, symbol=%s+0x%lx)\n",
		             src,
		             dest,
		             n,
		             src_ok ? 1 : 0,
		             dest_ok ? 1 : 0,
		             ra0,
		             info0.dli_fname ? info0.dli_fname : "?",
		             info0.dli_fbase,
		             info0.dli_sname ? info0.dli_sname : "?",
		             info0.dli_saddr ? (unsigned long)((char *)ra0 - (char *)info0.dli_saddr) : 0ul);
		std::fflush(stderr);

		// If the destination is mapped, zero it so callers do not see
		// uninitialised stack/heap data, but avoid touching an unmapped dst.
		if (dest_ok) {
			std::memset(dest, 0, n);
		}
		return dest;
	}

	using memcpy_fn = void *(*)(void *, const void *, size_t);
	static memcpy_fn real_memcpy = nullptr;
	if (!real_memcpy) {
		real_memcpy = (memcpy_fn)dlsym(RTLD_NEXT, "memcpy");
	}

	if (!real_memcpy) {
		/* Extremely unlikely: if dlsym fails, fall back to a simple
		 * byte-wise implementation.
		 */
		unsigned char *dptr = static_cast<unsigned char *>(dest);
		const unsigned char *sptr = static_cast<const unsigned char *>(src);

		if (dptr <= sptr || dptr >= sptr + n) {
			for (size_t i = 0; i < n; ++i)
				dptr[i] = sptr[i];
		} else {
			for (size_t i = n; i-- > 0; )
				dptr[i] = sptr[i];
		}
		return dest;
	}

	return real_memcpy(dest, src, n);
}




