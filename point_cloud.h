#pragma once
#include <string.h> // memset()
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "managed.h"
#include <vector_types.h> // float4

class __align__(128) Point_cu : public Managed {
public:
    float4 normal; // Normal
    float4 coord; // Point coordinate
    float texture; // Average texture color
};


class __align__(128) PointCloud : public Managed {
public:
    Point_cu *points;
    int rows;
    int cols;
    int size;
    void resize(int n)
    {
        cudaMallocManaged (&points,    sizeof(Point_cu) * n);
        memset            (points,  0, sizeof(Point_cu) * n);
    }
    ~PointCloud()
    {
        cudaFree (points);
    }
};
