#include "cuda_runtime.h"

#include <stdio.h>
#include "globalstate.h"



int main()
{
    int *c;
    GlobalState *gs = new GlobalState;
    CHECK(cudaMallocManaged(&c, sizeof(int)));
    *c = 0;
    return 0;
}
