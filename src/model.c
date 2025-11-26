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

    /* TODO: Return actual input tensor */
    /* For now, create a dummy tensor */
    if (model->inputs == NULL) {
        model->inputs = calloc(model->num_inputs, sizeof(nna_tensor_t*));
    }

    if (model->inputs[index] == NULL) {
        /* Create placeholder tensor */
        nna_shape_t shape = {{1, 16, 1, 1}, 4};  /* N=1, H=16, W=1, C=1 */
        model->inputs[index] = nna_tensor_create(&shape, NNA_DTYPE_INT8, NNA_FORMAT_NHWC);
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

    /* TODO: Return actual output tensor */
    if (model->outputs == NULL) {
        model->outputs = calloc(model->num_outputs, sizeof(nna_tensor_t*));
    }

    if (model->outputs[index] == NULL) {
        nna_shape_t shape = {{1, 16, 1, 1}, 4};
        model->outputs[index] = nna_tensor_create(&shape, NNA_DTYPE_INT8, NNA_FORMAT_NHWC);
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

    /* Try to load model as shared library if not already loaded */
    if (model->dl_handle == NULL && model->model_path != NULL) {
        printf("Loading model as shared library: %s\n", model->model_path);
        model->dl_handle = dlopen(model->model_path, RTLD_NOW | RTLD_LOCAL);
        if (model->dl_handle == NULL) {
            fprintf(stderr, "nna_model_run: dlopen failed: %s\n", dlerror());
            fprintf(stderr, "Note: .mgk models may require specific runtime environment\n");
            return NNA_ERROR_DEVICE;
        }

        /* Try to find common entry points */
        model->model_init = dlsym(model->dl_handle, "model_init");
        model->model_run = dlsym(model->dl_handle, "model_run");
        model->model_cleanup = dlsym(model->dl_handle, "model_cleanup");

        printf("Model symbols: init=%p run=%p cleanup=%p\n",
               model->model_init, model->model_run, model->model_cleanup);
    }

    /* If we have a model_run function, call it */
    if (model->model_run != NULL && model->model_context != NULL) {
        return model->model_run(model->model_context);
    }

    /* Otherwise, this is expected - models need specific runtime */
    fprintf(stderr, "nna_model_run: Model loaded but no executable entry points found\n");
    fprintf(stderr, "This is expected - .mgk models require the Venus runtime\n");
    return NNA_ERROR_DEVICE;
}

/* Unload model */
void nna_model_unload(nna_model_t *model) {
    if (model == NULL) {
        return;
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

