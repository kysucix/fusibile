#pragma once

#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "linestate.h"
#include "imageinfo.h"
#include "managed.h"
#include "point_cloud.h"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>

// includes, cuda
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

class GlobalState : public Managed {
public:
    CameraParameters_cu *cameras;
    //ImageInfo iminfo;
    LineState *lines;
    curandState *cs;
    AlgorithmParameters *params;
    PointCloud *pc;
   // PointCloud *pc_list;

    cudaTextureObject_t imgs  [MAX_IMAGES];
    cudaTextureObject_t normals_depths  [MAX_IMAGES]; // first 3 values normal, fourth depth
    /*cudaTextureObject_t depths  [MAX_IMAGES];*/
    //cudaTextureObject_t gradx [MAX_IMAGES];
    //cudaTextureObject_t grady [MAX_IMAGES];
    void resize(int n)
    {
        printf("Resizing globalstate to %d\n", n);
        cudaMallocManaged (&lines,     sizeof(LineState) * n);
    }
    ~GlobalState()
    {
        /*cudaFree (c);*/
        cudaFree (lines);
    }

};
