// Minimal __assert shim for OEM uClibc-based Venus stack
// This satisfies the undefined symbol __assert in libvenus.so
// when running on the custom uClibc rootfs where libc does not
// provide it with the expected symbol name.

#include <stdio.h>

void __assert(const char *expr, const char *file, int line, ...)
{
    if (!file)
        file = "(unknown)";
    if (!expr)
        expr = "(null)";

    fprintf(stderr,
            "[assert_shim] __assert failed: %s at %s:%d\n",
            expr, file, line);
}
