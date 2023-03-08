#ifndef ocfiowwdnymevc
#define ocfiowwdnymevc
#include <stdint.h>

uint64_t ipow(uint64_t x, uint64_t y) {
    uint64_t result = 1;
    
    for (;;) {
        if (y & 1)
            result *= x;
        y >>= 1;
        if (!y)
            break;
        x *= x;
    }

    return result;
}

#endif