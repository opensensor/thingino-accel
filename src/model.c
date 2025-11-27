/*
 * thingino-accel - Model Loading and Inference
 * 
 * Implementation of .mgk model loading and inference execution
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <elf.h>
#include <dlfcn.h>

#include "nna.h"
#include "nna_model.h"
#include "nna_memory.h"
#include "venus/model_loader.h"

/* Model structure */
struct nna_model {
    void *model_data;        /* Model file data (mmap or malloc) */
    size_t model_size;       /* Size of model data */
    int is_mapped;           /* 1 if mmap'd, 0 if malloc'd */
    char *model_path;        /* Path to model file (for dlopen) */

    /* ELF sections */
    Elf32_Ehdr *elf_header;
    Elf32_Shdr *section_headers;
    char *section_names;

    /* Model sections */
    void *text_section;      /* Code section */
    void *rodata_section;    /* Weights/constants */
    void *data_section;      /* Mutable data */

    /* Dynamic loading */
    void *dl_handle;         /* dlopen handle */
    void *(*model_init)(void*, size_t);  /* Model init function */
    int (*model_run)(void*);              /* Model run function */
    void (*model_cleanup)(void*);         /* Model cleanup function */
    void *model_context;     /* Model-specific context */

    /* C++ model instance (from model_loader) */
    void *mgk_model_handle;  /* Handle from load_mgk_model() */

    /* Inference state */
    nna_tensor_t **inputs;   /* Input tensors */
    nna_tensor_t **outputs;  /* Output tensors */
    uint32_t num_inputs;
    uint32_t num_outputs;

    void *forward_memory;    /* Forward pass working memory */
    size_t forward_mem_size;
    int owns_forward_mem;    /* 1 if we allocated it */
};

/* Helper: Find ELF section by name */
static Elf32_Shdr* find_section(nna_model_t *model, const char *name) {
    for (int i = 0; i < model->elf_header->e_shnum; i++) {
        Elf32_Shdr *shdr = &model->section_headers[i];
        const char *section_name = &model->section_names[shdr->sh_name];
        if (strcmp(section_name, name) == 0) {
            return shdr;
        }
    }
    return NULL;
}

/* Suppress unused function warning - will be used later */
__attribute__((unused))
static void* get_section_data(nna_model_t *model, const char *name) {
    Elf32_Shdr *shdr = find_section(model, name);
    if (shdr == NULL) {
        return NULL;
    }
    return (char*)model->model_data + shdr->sh_offset;
}

/* Load model from file */
nna_model_t* nna_model_load(const char *path, const nna_model_options_t *options) {
    int fd = -1;
    nna_model_t *model = NULL;
    struct stat st;
    
    /* Open model file */
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "nna_model_load: Failed to open %s: %s\n", 
                path, strerror(errno));
        return NULL;
    }
    
    /* Get file size */
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "nna_model_load: fstat failed: %s\n", strerror(errno));
        close(fd);
        return NULL;
    }
    
    /* Allocate model structure */
    model = calloc(1, sizeof(nna_model_t));
    if (model == NULL) {
        fprintf(stderr, "nna_model_load: Out of memory\n");
        close(fd);
        return NULL;
    }

    /* Save model path for potential dlopen */
    model->model_path = strdup(path);
    if (model->model_path == NULL) {
        fprintf(stderr, "nna_model_load: Failed to allocate path\n");
        free(model);
        close(fd);
        return NULL;
    }

    model->model_size = st.st_size;
    
    /* Map or load model data */
    if (options && options->use_file_mapping) {
        /* Use mmap for large models */
        model->model_data = mmap(NULL, model->model_size, PROT_READ,
                                 MAP_PRIVATE, fd, 0);
        if (model->model_data == MAP_FAILED) {
            fprintf(stderr, "nna_model_load: mmap failed: %s\n", strerror(errno));
            free(model);
            close(fd);
            return NULL;
        }
        model->is_mapped = 1;
    } else {
        /* Load into memory */
        model->model_data = malloc(model->model_size);
        if (model->model_data == NULL) {
            fprintf(stderr, "nna_model_load: Out of memory for model data\n");
            free(model);
            close(fd);
            return NULL;
        }
        
        if (read(fd, model->model_data, model->model_size) != (ssize_t)model->model_size) {
            fprintf(stderr, "nna_model_load: Failed to read model: %s\n", 
                    strerror(errno));
            free(model->model_data);
            free(model);
            close(fd);
            return NULL;
        }
        model->is_mapped = 0;
    }
    
    close(fd);
    
    /* Parse ELF header */
    model->elf_header = (Elf32_Ehdr*)model->model_data;
    
    /* Verify ELF magic */
    if (memcmp(model->elf_header->e_ident, ELFMAG, SELFMAG) != 0) {
        fprintf(stderr, "nna_model_load: Not a valid ELF file\n");
        nna_model_unload(model);
        return NULL;
    }
    
    /* Get section headers */
    model->section_headers = (Elf32_Shdr*)((char*)model->model_data + 
                                           model->elf_header->e_shoff);
    
    /* Get section name string table */
    Elf32_Shdr *shstrtab = &model->section_headers[model->elf_header->e_shstrndx];
    model->section_names = (char*)model->model_data + shstrtab->sh_offset;
    
    printf("Model loaded: %s (%zu bytes)\n", path, model->model_size);
    printf("  ELF sections: %u\n", model->elf_header->e_shnum);

    /* TODO: Parse model structure, extract inputs/outputs */
    model->num_inputs = 1;   /* Placeholder */
    model->num_outputs = 1;  /* Placeholder */

    return model;
}

/* Load model from memory */
nna_model_t* nna_model_load_from_memory(const void *buffer, size_t size,
                                         const nna_model_options_t *options) {
    /* For now, just allocate and copy */
    nna_model_t *model = calloc(1, sizeof(nna_model_t));
    if (model == NULL) {
        return NULL;
    }

    model->model_data = malloc(size);
    if (model->model_data == NULL) {
        free(model);
        return NULL;
    }

    memcpy(model->model_data, buffer, size);
    model->model_size = size;
    model->is_mapped = 0;

    /* Parse ELF (same as file loading) */
    model->elf_header = (Elf32_Ehdr*)model->model_data;
    model->section_headers = (Elf32_Shdr*)((char*)model->model_data +
                                           model->elf_header->e_shoff);
    Elf32_Shdr *shstrtab = &model->section_headers[model->elf_header->e_shstrndx];
    model->section_names = (char*)model->model_data + shstrtab->sh_offset;

    model->num_inputs = 1;
    model->num_outputs = 1;

    return model;
}

/* Get model info */
int nna_model_get_info(nna_model_t *model, nna_model_info_t *info) {
    if (model == NULL || info == NULL) {
        return NNA_ERROR_INVALID;
    }

    info->num_inputs = model->num_inputs;
    info->num_outputs = model->num_outputs;
    info->num_layers = 0;  /* TODO: Parse from model */
    info->model_size = model->model_size;
    info->forward_mem_req = 1024 * 1024;  /* TODO: Calculate actual requirement */

    return NNA_SUCCESS;
}

/* Get input tensor */
nna_tensor_t* nna_model_get_input(nna_model_t *model, uint32_t index) {
    if (model == NULL || index >= model->num_inputs) {
        return NULL;
    }

    if (model->inputs == NULL) {
        model->inputs = calloc(model->num_inputs, sizeof(nna_tensor_t*));
        if (model->inputs == NULL) {
            return NULL;
        }
    }

    if (model->inputs[index] == NULL) {
        /* Prefer wiring to real .mgk input tensors when available. */
        void *data = NULL;
        int dims[4] = {1, 1, 1, 1};
        int ndim = 0;
        int dtype_code = NNA_DTYPE_UINT8;
        int format_code = NNA_FORMAT_NHWC;

        /* Ensure .mgk model is loaded so that we can query its tensors. */
        if (model->mgk_model_handle == NULL && model->model_path != NULL) {
            printf("nna_model_get_input: loading .mgk model for IO tensor query: %s\n",
                   model->model_path);
            model->mgk_model_handle = load_mgk_model(model->model_path);
            if (model->mgk_model_handle == NULL) {
                fprintf(stderr, "nna_model_get_input: load_mgk_model failed, falling back to dummy tensor\n");
            } else {
                printf("nna_model_get_input: .mgk model loaded successfully\n");
            }
        }

        int wired = 0;
        if (model->mgk_model_handle != NULL) {
            int rc = mgk_model_get_io_tensor_info(model->mgk_model_handle,
                                                  1, /* is_input */
                                                  index,
                                                  &data,
                                                  dims,
                                                  &ndim,
                                                  &dtype_code,
                                                  &format_code);
            if (rc == 0 && data != NULL && ndim > 0 && ndim <= 4) {
                nna_shape_t shape;
                int i;
                for (i = 0; i < 4; i++) {
                    if (i < ndim) {
                        shape.dims[i] = (int32_t)dims[i];
                    } else {
                        shape.dims[i] = 1;
                    }
                }
                shape.ndim = ndim;

                nna_dtype_t dtype = (nna_dtype_t)dtype_code;
                nna_format_t format = (nna_format_t)format_code;

                model->inputs[index] = nna_tensor_from_data(data, &shape, dtype, format);
                if (model->inputs[index] != NULL) {
                    wired = 1;
                } else {
                    fprintf(stderr, "nna_model_get_input: nna_tensor_from_data failed, falling back to dummy tensor\n");
                }
            } else {
                fprintf(stderr, "nna_model_get_input: mgk_model_get_io_tensor_info failed for input %u, falling back to dummy tensor\n",
                        (unsigned int)index);
            }
        }

        if (!wired) {
            /* Create placeholder tensor (legacy behavior). */
            nna_shape_t shape = {{1, 16, 1, 1}, 4};  /* N=1, H=16, W=1, C=1 */
            model->inputs[index] = nna_tensor_create(&shape, NNA_DTYPE_INT8, NNA_FORMAT_NHWC);
        }
    }

    return model->inputs[index];
}

/* Get input by name */
nna_tensor_t* nna_model_get_input_by_name(nna_model_t *model, const char *name) {
    /* TODO: Implement name lookup */
    return nna_model_get_input(model, 0);
}

/* Get output tensor */
const nna_tensor_t* nna_model_get_output(nna_model_t *model, uint32_t index) {
    if (model == NULL || index >= model->num_outputs) {
        return NULL;
    }

    if (model->outputs == NULL) {
        model->outputs = calloc(model->num_outputs, sizeof(nna_tensor_t*));
        if (model->outputs == NULL) {
            return NULL;
        }
    }

    if (model->outputs[index] == NULL) {
        void *data = NULL;
        int dims[4] = {1, 1, 1, 1};
        int ndim = 0;
        int dtype_code = NNA_DTYPE_UINT8;
        int format_code = NNA_FORMAT_NHWC;

        if (model->mgk_model_handle == NULL && model->model_path != NULL) {
            printf("nna_model_get_output: loading .mgk model for IO tensor query: %s\n",
                   model->model_path);
            model->mgk_model_handle = load_mgk_model(model->model_path);
            if (model->mgk_model_handle == NULL) {
                fprintf(stderr, "nna_model_get_output: load_mgk_model failed, falling back to dummy tensor\n");
            } else {
                printf("nna_model_get_output: .mgk model loaded successfully\n");
            }
        }

        int wired = 0;
        if (model->mgk_model_handle != NULL) {
            int rc = mgk_model_get_io_tensor_info(model->mgk_model_handle,
                                                  0, /* is_input */
                                                  index,
                                                  &data,
                                                  dims,
                                                  &ndim,
                                                  &dtype_code,
                                                  &format_code);
            if (rc == 0 && data != NULL && ndim > 0 && ndim <= 4) {
                nna_shape_t shape;
                int i;
                for (i = 0; i < 4; i++) {
                    if (i < ndim) {
                        shape.dims[i] = (int32_t)dims[i];
                    } else {
                        shape.dims[i] = 1;
                    }
                }
                shape.ndim = ndim;

                nna_dtype_t dtype = (nna_dtype_t)dtype_code;
                nna_format_t format = (nna_format_t)format_code;

                model->outputs[index] = nna_tensor_from_data(data, &shape, dtype, format);
                if (model->outputs[index] != NULL) {
                    wired = 1;
                } else {
                    fprintf(stderr, "nna_model_get_output: nna_tensor_from_data failed, falling back to dummy tensor\n");
                }
            } else {
                fprintf(stderr, "nna_model_get_output: mgk_model_get_io_tensor_info failed for output %u, falling back to dummy tensor\n",
                        (unsigned int)index);
            }
        }

        if (!wired) {
            nna_shape_t shape = {{1, 16, 1, 1}, 4};
            model->outputs[index] = nna_tensor_create(&shape, NNA_DTYPE_INT8, NNA_FORMAT_NHWC);
        }
    }

    return model->outputs[index];
}

/* Get output by name */
const nna_tensor_t* nna_model_get_output_by_name(nna_model_t *model, const char *name) {
    (void)name;  /* Unused for now */
    return nna_model_get_output(model, 0);
}

/* Run inference */
int nna_model_run(nna_model_t *model) {
    if (model == NULL) {
        return NNA_ERROR_INVALID;
    }

    /* Try to load model as C++ class if not already loaded */
    if (model->mgk_model_handle == NULL && model->model_path != NULL) {
        printf("Loading .mgk model: %s\n", model->model_path);
        model->mgk_model_handle = load_mgk_model(model->model_path);
        if (model->mgk_model_handle == NULL) {
            fprintf(stderr, "nna_model_run: Failed to load .mgk model\n");
            return NNA_ERROR_DEVICE;
        }
        printf("Model loaded successfully\n");
    }

    /* Run inference using C++ model */
    if (model->mgk_model_handle != NULL) {
        printf("Running inference...\n");
        int result = run_mgk_model(model->mgk_model_handle);
        if (result == 0) {
            printf("Inference completed successfully\n");
            return NNA_SUCCESS;
        } else {
            fprintf(stderr, "nna_model_run: Inference failed with code %d\n", result);
            return NNA_ERROR_DEVICE;
        }
    }

    /* Fallback: try old C-style loading */
    if (model->dl_handle == NULL && model->model_path != NULL) {
        printf("Trying C-style model loading: %s\n", model->model_path);
        model->dl_handle = dlopen(model->model_path, RTLD_NOW | RTLD_LOCAL);
        if (model->dl_handle == NULL) {
            fprintf(stderr, "nna_model_run: dlopen failed: %s\n", dlerror());
            return NNA_ERROR_DEVICE;
        }

        model->model_init = dlsym(model->dl_handle, "model_init");
        model->model_run = dlsym(model->dl_handle, "model_run");
        model->model_cleanup = dlsym(model->dl_handle, "model_cleanup");
    }

    if (model->model_run != NULL && model->model_context != NULL) {
        return model->model_run(model->model_context);
    }

    fprintf(stderr, "nna_model_run: No executable model found\n");
    return NNA_ERROR_DEVICE;
}

/* Unload model */
void nna_model_unload(nna_model_t *model) {
    if (model == NULL) {
        return;
    }

    /* Unload C++ model if loaded */
    if (model->mgk_model_handle) {
        unload_mgk_model(model->mgk_model_handle);
        model->mgk_model_handle = NULL;
    }

    /* Cleanup model context if we have a cleanup function */
    if (model->model_cleanup && model->model_context) {
        model->model_cleanup(model->model_context);
    }

    /* Close dlopen handle */
    if (model->dl_handle) {
        dlclose(model->dl_handle);
    }

    /* Free tensors */
    if (model->inputs) {
        for (uint32_t i = 0; i < model->num_inputs; i++) {
            if (model->inputs[i]) {
                nna_tensor_destroy(model->inputs[i]);
            }
        }
        free(model->inputs);
    }

    if (model->outputs) {
        for (uint32_t i = 0; i < model->num_outputs; i++) {
            if (model->outputs[i]) {
                nna_tensor_destroy(model->outputs[i]);
            }
        }
        free(model->outputs);
    }

    /* Free forward memory if we own it */
    if (model->owns_forward_mem && model->forward_memory) {
        nna_free(model->forward_memory);
    }

    /* Free model path */
    if (model->model_path) {
        free(model->model_path);
    }

    /* Free model data */
    if (model->model_data) {
        if (model->is_mapped) {
            munmap(model->model_data, model->model_size);
        } else {
            free(model->model_data);
        }
    }

    free(model);
}

