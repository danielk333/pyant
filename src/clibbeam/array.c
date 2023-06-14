#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "beam.h"


void
array_sensor_response(
    double *k,
    double complex *G,
    size_t channels
)
{
    for(size_t ch = 0; ch < channels; ch++) {
        G[ch] = 1.0;
    }
}
