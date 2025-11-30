/*
 * Minimal stub implementation of the Ingenic AIP API used by libvenus.so.
 *
 * The real /opt/libaip.so tries to mmap AIP hardware and aborts on this
 * Franken-rootfs with:
 *   "[AIP@]Error chain mmap: Invalid argument"
 *
 * For the purpose of capturing NNDMA/NNA traces from OEM Venus, we don't
 * actually need the AIP hardware. This shim exports just enough symbols for
 * libvenus.so to link and will return success (or benign defaults) from all
 * AIP entrypoints that Venus calls.
 *
 * IMPORTANT:
 *   - This is *not* a real implementation of the AIP.
 *   - It is only meant for experimentation on this custom rootfs.
 */

#include <stdio.h>

#include "ingenic_aip.h"

int bs_covert_cfg(data_info_s *src, const data_info_s *dst,
                  const uint32_t *coef, const uint32_t *offset,
                  const task_info_s *task_info)
{
    (void)src; (void)dst; (void)coef; (void)offset; (void)task_info;
    // In OEM code this would configure a bit-scaler chain. For our purposes
    // we just pretend it succeeded.
    fprintf(stderr, "[aip_shim] bs_covert_cfg()\n");
    return 0;
}

int bs_covert_step_start(const task_info_s *task_info,
                         uint32_t dst_ptr, const bs_data_locate_e locate)
{
    (void)task_info; (void)dst_ptr; (void)locate;
    fprintf(stderr, "[aip_shim] bs_covert_step_start()\n");
    return 0;
}

int bs_covert_step_wait(void)
{
    fprintf(stderr, "[aip_shim] bs_covert_step_wait()\n");
    return 0;
}

int bs_covert_step_exit(void)
{
    fprintf(stderr, "[aip_shim] bs_covert_step_exit()\n");
    return 0;
}

int ingenic_aip_resize_process(data_info_s *src,
                               const int box_num, const data_info_s *dst,
                               const box_resize_info_s *boxes,
                               const uint32_t *coef, const uint32_t *offset)
{
    (void)src; (void)box_num; (void)dst; (void)boxes; (void)coef; (void)offset;
    fprintf(stderr, "[aip_shim] ingenic_aip_resize_process()\n");
    return 0;
}

int ingenic_aip_affine_process(data_info_s *src,
                               const int box_num, const data_info_s *dst,
                               const box_affine_info_s *boxes,
                               const uint32_t *coef, const uint32_t *offset)
{
    (void)src; (void)box_num; (void)dst; (void)boxes; (void)coef; (void)offset;
    fprintf(stderr, "[aip_shim] ingenic_aip_affine_process()\n");
    return 0;
}

int ingenic_aip_perspective_process(data_info_s *src,
                                    const int box_num, const data_info_s *dst,
                                    const box_affine_info_s *boxes,
                                    const uint32_t *coef, const uint32_t *offset)
{
    (void)src; (void)box_num; (void)dst; (void)boxes; (void)coef; (void)offset;
    fprintf(stderr, "[aip_shim] ingenic_aip_perspective_process()\n");
    return 0;
}

int ingenic_aip_init(void)
{
    fprintf(stderr, "[aip_shim] ingenic_aip_init()\n");
    return 0;
}

int ingenic_aip_deinit(void)
{
    fprintf(stderr, "[aip_shim] ingenic_aip_deinit()\n");
    return 0;
}

int ingenic_aip_version(void)
{
    // Arbitrary non-zero version number.
    fprintf(stderr, "[aip_shim] ingenic_aip_version() -> 1\n");
    return 1;
}

int ingenic_aip_max_cnt(int aip_type)
{
    (void)aip_type;
    // Say we support a single AIP instance; Venus should treat this as
    // "at least one" and continue.
    fprintf(stderr, "[aip_shim] ingenic_aip_max_cnt(%d) -> 1\n", aip_type);
    return 1;
}
