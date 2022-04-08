// inclusion guard
#ifndef DOA_H_
#define DOA_H_
#define precision double

void
array_sensor_response(
    double *k,
    precision complex* G,
    size_t channels
);
#endif // DOA_H_