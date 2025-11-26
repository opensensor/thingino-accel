/*
 * thingino-accel - Venus C API
 * C wrapper for C++ Venus library
 */

#include "basenet.h"
#include "../../include/nna.h"
#include <cstdio>

using namespace magik::venus;

/* Global initialization state */
static bool venus_initialized = false;

/* C API functions */
extern "C" {

/* Initialize Venus */
int magik_venus_init(int flags) {
    (void)flags;
    
    printf("magik_venus_init: Initializing Venus library\n");
    
    /* Initialize NNA hardware */
    if (nna_init() != NNA_SUCCESS) {
        fprintf(stderr, "magik_venus_init: NNA initialization failed\n");
        return -1;
    }
    
    venus_initialized = true;
    printf("magik_venus_init: Venus initialized successfully\n");
    return 0;
}

/* Check Venus status */
int magik_venus_check(void) {
    return venus_initialized ? 0 : -1;
}

/* Deinitialize Venus */
int magik_venus_deinit(void) {
    if (!venus_initialized) {
        return 0;
    }
    
    printf("magik_venus_deinit: Shutting down Venus library\n");
    
    nna_deinit();
    venus_initialized = false;
    
    return 0;
}

/* Get version string */
const char* magik_venus_get_version(void) {
    return "thingino-accel Venus 1.0.0";
}

} /* extern "C" */

/* C++ namespace functions for .mgk models */
namespace magik {
namespace venus {

/* These functions are called by .mgk model code */

/* Layer base class stubs */
class Layer {
public:
    virtual ~Layer() {}
    virtual int init() { return 0; }
    virtual int forward() { return 0; }
    virtual const char* get_name() const { return "layer"; }
};

/* LayerParam stub */
class LayerParam {
public:
    virtual ~LayerParam() {}
    virtual int load() { return 0; }
};

/* Predictor class stub */
class Predictor {
public:
    Predictor() {}
    virtual ~Predictor() {}
    virtual int run_nmem_analysis(std::vector<std::string> &names) {
        (void)names;
        return 0;
    }
};

/* EvaluateTool stub */
class EvaluateTool {
public:
    virtual int run_kernel() { return 0; }
};

} // namespace venus
} // namespace magik

