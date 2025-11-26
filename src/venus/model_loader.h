/*
 * thingino-accel - Model Loader for .mgk files
 */

#ifndef THINGINO_ACCEL_MODEL_LOADER_H
#define THINGINO_ACCEL_MODEL_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Load a .mgk model file and instantiate the model class
 * 
 * Returns opaque handle to model instance, or NULL on failure
 */
void* load_mgk_model(const char *path);

/*
 * Run inference on a loaded model
 * 
 * Returns 0 on success, -1 on failure
 */
int run_mgk_model(void *handle);

/*
 * Unload model and free resources
 */
void unload_mgk_model(void *handle);

/*
 * Get model instance pointer (for accessing inputs/outputs)
 */
void* get_mgk_model_instance(void *handle);

#ifdef __cplusplus
}
#endif

#endif /* THINGINO_ACCEL_MODEL_LOADER_H */

