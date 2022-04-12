#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "libbeam.h"


void
array_sensor_response(
    double *k,
    precision complex* G,
    size_t channels
)
{
    for(size_t ch = 0; ch < channels; ch++) {
        G[ch] = 1.0;
    }
}