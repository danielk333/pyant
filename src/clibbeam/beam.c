#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "beam.h"

int
main(int argc, char *argv[])
{
    double kvec[3];
    size_t channels = 1;
    double complex G[channels];
    kvec[0] = 0.0;
    kvec[1] = 0.0;
    kvec[2] = 1.0;
    array_sensor_response(kvec, G, channels);

    printf("Hello, G = %.2f + %.2fi\n", creal(G[0]), cimag(G[0]));

    return 0;
}
