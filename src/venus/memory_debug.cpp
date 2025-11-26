/*
 * thingino-accel - Memory Debugging
 * Override global new/delete to log allocations
 */

#include <cstdio>
#include <cstdlib>
#include <new>

/* Override global new operator to log allocations */
/* Disabled - causes issues with .mgk file's own allocator
void* operator new(size_t size) {
    if (size > 100000000) {
        printf("[VENUS] WARNING: Attempting to allocate huge size: %zu bytes\n", size);
        fflush(stdout);
    }
    void *ptr = malloc(size);
    if (!ptr) {
        printf("[VENUS] ERROR: malloc(%zu) failed!\n", size);
        fflush(stdout);
        throw std::bad_alloc();
    }
    return ptr;
}
*/

/*
 * Global delete overrides were originally defined here, but we now
 * provide a unified operator new/delete implementation in
 * magik_model.cpp for allocation tracing. Disable these to avoid
 * multiple-definition linker errors.
 */
#if 0
void operator delete(void *ptr) noexcept {
    free(ptr);
}

void operator delete(void *ptr, size_t size) noexcept {
    (void)size;
    free(ptr);
}
#endif

