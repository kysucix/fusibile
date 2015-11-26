#pragma once
#include <string.h> // memset()
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "managed.h"
#include <vector_types.h> // float4

class __align__(128) LineState : public Managed {
public:
    char *used_pixels; // disparity*/*/
    int n;
    int s; // stride
    int l; // length
    void resize(int n)
    {
        cudaMallocManaged (&used_pixels,        sizeof(char) * n);
        memset            (used_pixels,      0, sizeof(char) * n);
    }
    ~LineState()
    {
        cudaFree (used_pixels);
    }
};
