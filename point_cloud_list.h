#pragma once
#include <string.h> // memset()
#include <stdlib.h> // malloc(), realloc(), free()
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "managed.h"
#include <vector_types.h> // float4

class Point_li {
public:
    float4 normal; // Normal
    float4 coord; // Point coordinate
    float texture; // Average texture color
};


class PointCloudList {
public:
    Point_li *points;
    int rows;
    int cols;
    unsigned int size;
    unsigned int maximum;
    void resize(int n)
    {
        maximum=n;
        points = (Point_li *) malloc (sizeof(Point_li) * n);
        memset            (points,  0, sizeof(Point_li) * n);
    }
    void increase_size(int n)
    {
        maximum=n;
        points = (Point_li *) realloc (points, n * sizeof(Point_li));
        printf("New size of point cloud list is %d\n", n);
    }
    ~PointCloudList()
    {
        free (points);
    }
};
