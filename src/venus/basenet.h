/*
 * thingino-accel - Venus BaseNet Implementation
 */

#ifndef THINGINO_ACCEL_VENUS_BASENET_H
#define THINGINO_ACCEL_VENUS_BASENET_H

#include "tensor.h"
#include "venus_types.h"
#include <memory>
#include <string>
#include <vector>
#include <map>

namespace magik {
namespace venus {

/* BaseNet class matching Venus API */
class BaseNet {
public:
    BaseNet();
    virtual ~BaseNet();
    
    virtual int load_model(const char *model_path, bool memory_model = false, int start_off = 0);
    virtual int get_forward_memory_size(size_t &memory_size);
    virtual int set_forward_memory(void *memory);
    virtual int free_forward_memory();
    virtual int free_inputs_memory();
    virtual void set_internal_mm_status(bool status);
    virtual bool get_internal_mm_status();
    virtual void set_profiler_per_frame(bool status = false);
    
    virtual std::unique_ptr<Tensor> get_input(int index);
    virtual std::unique_ptr<Tensor> get_input_by_name(std::string &name);
    virtual std::vector<std::string> get_input_names();
    
    virtual std::unique_ptr<const Tensor> get_output(int index);
    virtual std::unique_ptr<const Tensor> get_output_by_name(std::string &name);
    virtual std::vector<std::string> get_output_names();
    virtual std::vector<std::string> get_output_names_step(int step);
    
    virtual ChannelLayout get_input_channel_layout(std::string &name);
    virtual void set_input_channel_layout(std::string name, ChannelLayout layout);
    
    virtual int run();
    virtual int steps();
    virtual int run_step();

protected:
    /* Model data */
    void *model_handle;  /* dlopen handle for .mgk */
    std::string model_path;
    
    /* Tensors */
    std::vector<std::unique_ptr<Tensor>> inputs;
    std::vector<std::unique_ptr<Tensor>> outputs;
    std::map<std::string, int> input_name_map;
    std::map<std::string, int> output_name_map;
    
    /* Memory management */
    void *forward_memory;
    size_t forward_memory_size;
    bool owns_forward_memory;
    bool internal_mm_enabled;
    
    /* Profiling */
    bool profiling_enabled;
    
    /* Format */
    TensorFormat input_format;
    ShareMemoryMode memory_mode;
};

/* Factory function */
std::unique_ptr<BaseNet> net_create(TensorFormat input_data_fmt = TensorFormat::NHWC,
                                    ShareMemoryMode smem_mode = ShareMemoryMode::DEFAULT);

/* Global functions */
int venus_lock();
int venus_unlock();
uint32_t venus_get_version_info();
uint32_t venus_get_used_mem_size();

} // namespace venus
} // namespace magik

#endif /* THINGINO_ACCEL_VENUS_BASENET_H */

