/* vim: ft=cpp
 * */

//#include <helper_math.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include "globalstate.h"
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "linestate.h"
#include "imageinfo.h"
#include "config.h"

#include <vector_types.h> // float4
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "vector_operations.h"
#include "point_cloud_list.h"

#define SAVE_TEXTURE
//#define SMOOTHNESS

#define FORCEINLINE_FUSIBILE __forceinline__
//#define FORCEINLINE_FUSIBILE


__device__ float K[16];
__device__ float K_inv[16];

/*__device__ FORCEINLINE_FUSIBILE __constant__ float4 camerasK[32];*/

/* compute depth value from disparity or disparity value from depth
 * Input:  f         - focal length in pixel
 *         baseline  - baseline between cameras (in meters)
 *         d - either disparity or depth value
 * Output: either depth or disparity value
 */
__device__ FORCEINLINE_FUSIBILE float disparityDepthConversion_cu ( const float &f, const float &baseline, const float &d ) {
    return f * baseline / d;
}

/* compute depth value from disparity or disparity value from depth
 * Input:  f         - focal length in pixel
 *         baseline  - baseline between cameras (in meters)
 *         d - either disparity or depth value
 * Output: either depth or disparity value
 */
__device__ FORCEINLINE_FUSIBILE float disparityDepthConversion_cu2 ( const float &f, const Camera_cu &cam_ref, const Camera_cu &cam, const float &d ) {
    float baseline = l2_float4(cam_ref.C4 - cam.C4);
    return f * baseline / d;
}

__device__ FORCEINLINE_FUSIBILE void get3Dpoint_cu ( float4 * __restrict__ ptX, const Camera_cu &cam, const int2 &p, const float &depth ) {
    // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first
    const float4 pt = make_float4 (
                                   depth * (float)p.x     - cam.P_col34.x,
                                   depth * (float)p.y     - cam.P_col34.y,
                                   depth                  - cam.P_col34.z,
                                   0);

    matvecmul4 (cam.M_inv, pt, ptX);
}
__device__ FORCEINLINE_FUSIBILE void get3Dpoint_cu1 ( float4 * __restrict__ ptX, const Camera_cu &cam, const int2 &p) {
    // in case camera matrix is not normalized: see page 162, then depth might not be the real depth but w and depth needs to be computed from that first
    float4 pt;
    pt.x = (float)p.x     - cam.P_col34.x;
    pt.y = (float)p.y     - cam.P_col34.y;
    pt.z = 1.0f           - cam.P_col34.z;

    matvecmul4 (cam.M_inv, pt, ptX);
}
//get d parameter of plane pi = [nT, d]T, which is the distance of the plane to the camera center
__device__ FORCEINLINE_FUSIBILE float getPlaneDistance_cu ( const float4 &normal, const float4 &X ) {
    return -(dot4(normal,X));
}

__device__ FORCEINLINE_FUSIBILE void normalize_cu (float4 * __restrict__ v)
{
    const float normSquared = pow2(v->x) + pow2(v->y) + pow2(v->z);
    const float inverse_sqrt = rsqrtf (normSquared);
    v->x *= inverse_sqrt;
    v->y *= inverse_sqrt;
    v->z *= inverse_sqrt;
}
__device__ FORCEINLINE_FUSIBILE void getViewVector_cu (float4 * __restrict__ v, const Camera_cu &camera, const int2 &p)
{
    get3Dpoint_cu1 (v, camera, p);
    sub((*v), camera.C4);
    normalize_cu(v);
    //v->x=0;
    //v->y=0;
    //v->z=1;
}

__device__ FORCEINLINE_FUSIBILE float l1_norm(float f) {
    return fabsf(f);
}
__device__ FORCEINLINE_FUSIBILE float l1_norm(float4 f) {
    return ( fabsf (f.x) +
             fabsf (f.y) +
             fabsf (f.z))*0.3333333f;

}
__device__ FORCEINLINE_FUSIBILE float l1_norm2(float4 f) {
    return ( fabsf (f.x) +
             fabsf (f.y) +
             fabsf (f.z));

}

/* get angle between two vectors in 3D
 * Input: v1,v2 - vectors
 * Output: angle in radian
 */
__device__ FORCEINLINE_FUSIBILE float getAngle_cu ( const float4 &v1, const float4 &v2 ) {
    float angle = acosf ( dot4(v1, v2));
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if ( angle != angle )
        return 0.0f;
    //if ( acosf ( v1.dot ( v2 ) ) != acosf ( v1.dot ( v2 ) ) )
    //cout << acosf ( v1.dot ( v2 ) ) << " / " << v1.dot ( v2 )<< " / " << v1<< " / " << v2 << endl;
    return angle;
}
__device__ FORCEINLINE_FUSIBILE void project_on_camera (const float4 &X, const Camera_cu &cam, float2 *pt, float *depth) {
    float4 tmp = make_float4 (0, 0, 0, 0);
    matvecmul4P (cam.P, X, (&tmp));
    pt->x = tmp.x / tmp.z;
    pt->y = tmp.y / tmp.z;
    *depth = tmp.z;
}

/*
 * Simple and fast depth math fusion based on depth map and normal consensus
 */
__global__ void fusibile (GlobalState &gs, int ref_camera)
{
    int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    //printf("p is %d %d\n", p.x, p.y);

    const int cols = gs.cameras->cols;
    const int rows = gs.cameras->rows;

    if (p.x>=cols)
        return;
    if (p.y>=rows)
        return;

    const int center = p.y*cols+p.x;

    const CameraParameters_cu &camParams = *(gs.cameras);

    if (gs.lines[ref_camera].used_pixels[center]==1)
        return;

    //printf("ref_camera is %d\n", ref_camera);
    const float4 normal = tex2D<float4> (gs.normals_depths[ref_camera], p.x+0.5f, p.y+0.5f);
    //printf("Normal is %f %f %f\nDepth is %f\n", normal.x, normal.y, normal.z, normal.w);
    /*
     * For each point of the reference camera compute the 3d position corresponding to the corresponding depth.
     * Create a point only if the following conditions are fulfilled:
     * - Projected depths of other cameras does not differ more than gs.params.depthThresh
     * - Angle of normal does not differ more than gs.params.normalThresh
     */
    float depth = normal.w;

    float4 X;
    get3Dpoint_cu (&X, camParams.cameras[ref_camera], p, depth);
    //if (p.x<100 && p.y ==100)
    //printf("3d Point is %f %f %f\n", X.x, X.y, X.z);
    float4 consistent_X = X;
    float4 consistent_normal  = normal;
    float consistent_texture = tex2D<float> (gs.imgs[ref_camera], p.x+0.5f, p.y+0.5f);
    int number_consistent = 0;
    //int2 used_list[camParams.viewSelectionSubsetNumber];
    int2 used_list[MAX_IMAGES];
    for ( int i = 0; i < camParams.viewSelectionSubsetNumber; i++ ) {

        int idxCurr = camParams.viewSelectionSubset[i];
        used_list[idxCurr].x=-1;
        used_list[idxCurr].y=-1;
        if (idxCurr == ref_camera)
            continue;

        // Project 3d point X on camera idxCurr
        float2 tmp_pt;
        project_on_camera (X, camParams.cameras[idxCurr], &tmp_pt, &depth);
        //printf("P for camera %d is \n", i);
        //print_matrix (camParams.cameras[idxCurr].P, "camera ");
        //printf("2d point for camera %d is %f %f\n", idxCurr, tmp_pt.x, tmp_pt.y);

        // Boundary check
        if (tmp_pt.x >=0 &&
            tmp_pt.x < cols &&
            tmp_pt.y >=0 &&
            tmp_pt.y < rows) {
            //printf("Boundary check passed\n");

            // Compute interpolated depth and normal for tmp_pt w.r.t. camera ref_camera
            float4 tmp_normal_and_depth; // first 3 components normal, fourth depth
            tmp_normal_and_depth   = tex2D<float4> (gs.normals_depths[idxCurr], tmp_pt.x+0.5f, tmp_pt.y+0.5f);
            //printf("New depth is %f vs %f\n", tmp_normal_and_depth.w, depth);

            const float depth_disp = disparityDepthConversion_cu2                ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], depth );
            const float tmp_normal_and_depth_disp = disparityDepthConversion_cu2 ( camParams.cameras[ref_camera].f, camParams.cameras[ref_camera], camParams.cameras[idxCurr], tmp_normal_and_depth.w );
            // First consistency check on depth
            if (fabsf(depth_disp - tmp_normal_and_depth_disp) < gs.params->depthThresh) {
                //printf("\tFirst consistency test passed!\n");
                float angle = getAngle_cu (tmp_normal_and_depth, normal); // extract normal
                if (angle < gs.params->normalThresh)
                {
                    //printf("\tSecond consistency test passed!\n");
                    /// All conditions met:
                    //  - average 3d points and normals
                    //  - save resulting point and normal
                    //  - (optional) average texture (not done yet)
                    float4 tmp_X; // 3d point of consistent point on other view
                    int2 tmp_p = make_int2 ((int) tmp_pt.x, (int) tmp_pt.y);

                    get3Dpoint_cu (&tmp_X, camParams.cameras[idxCurr], tmp_p, tmp_normal_and_depth.w);
                    consistent_X      = consistent_X      + tmp_X;
                    //consistent_X      = tmp_X;
                    consistent_normal = consistent_normal + tmp_normal_and_depth;
                    if (gs.params->saveTexture)
                        consistent_texture = consistent_texture + tex2D<float> (gs.imgs[idxCurr], tmp_pt.x+0.5f, tmp_pt.y+0.5f);


                    // Save the point for later check
                    //printf ("Saved point on camera %d is %d %d\n", idxCurr, (int)tmp_pt.x, (int)tmp_pt.y);
                    used_list[idxCurr].x=(int)tmp_pt.x;
                    used_list[idxCurr].y=(int)tmp_pt.y;

                    number_consistent++;
                }
            }
        }
        else
            continue;
    }

    // Average normals and points
    consistent_X       = consistent_X       / ((float) number_consistent + 1.0f);
    consistent_normal  = consistent_normal  / ((float) number_consistent + 1.0f);
    consistent_texture = consistent_texture / ((float) number_consistent + 1.0f);

    // If at least numConsistentThresh point agree:
    // Create point
    // Save normal
    // (optional) save texture
    if (number_consistent >= gs.params->numConsistentThresh) {
        //printf("\tEnough consistent points!\nSaving point %f %f %f", consistent_X.x, consistent_X.y, consistent_X.z);
        if (!gs.params->remove_black_background || consistent_texture>15) // hardcoded for middlebury TODO FIX
        {
            gs.pc->points[center].coord  = consistent_X;
            gs.pc->points[center].normal = consistent_normal;

#ifdef SAVE_TEXTURE
            if (gs.params->saveTexture)
                gs.pc->points[center].texture = consistent_texture;
#endif

            //// Mark corresponding point on other views as "used"
            for ( int i = 0; i < camParams.viewSelectionSubsetNumber; i++ ) {
                int idxCurr = camParams.viewSelectionSubset[i];
                if (used_list[idxCurr].x==-1)
                    continue;
                //printf("Used list point on camera %d is %d %d\n", idxCurr, used_list[idxCurr].x, used_list[idxCurr].y);
                gs.lines[idxCurr].used_pixels [used_list[idxCurr].x + used_list[idxCurr].y*cols] = 1;
            }
        }
    }

    return;
}
/* Copy point cloud to global memory */
//template< typename T >
void copy_point_cloud_to_host(GlobalState &gs, int cam, PointCloudList &pc_list)
{
    printf("Processing camera %d\n", cam);
    unsigned int count = pc_list.size;
    for (int y=0; y<gs.pc->rows; y++) {
        for (int x=0; x<gs.pc->cols; x++) {
            Point_cu &p = gs.pc->points[x+y*gs.pc->cols];
            const float4 X      = p.coord;
            const float4 normal = p.normal;
            float texture = 127.0f;
#ifdef SAVE_TEXTURE
            if (gs.params->saveTexture)
                texture = p.texture;
#endif
            if (count==pc_list.maximum) {
                printf("Not enough space to save points :'(\n... allocating more! :)");
                pc_list.increase_size(pc_list.maximum*2);

            }
            if (X.x != 0 && X.y != 0 && X.z != 0) {
                pc_list.points[count].coord   = X;
                pc_list.points[count].normal  = normal;
#ifdef SAVE_TEXTURE
                pc_list.points[count].texture = texture;
#endif
                count++;
            }
            p.coord = make_float4(0,0,0,0);
        }
    }
    printf("Found %.2f million points\n", count/1000000.0f);
    pc_list.size = count;
}

template< typename T >
void fusibile_cu(GlobalState &gs, PointCloudList &pc_list, int num_views)
{
#ifdef SHARED
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
#endif

    int rows = gs.cameras->rows;
    int cols = gs.cameras->cols;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("Run gipuma\n");
    /*curandState* devStates;*/
    //cudaMalloc ( &gs.cs, rows*cols*sizeof( curandState ) );

    int count = 0;
    int i = 0;

    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return ;
    }

    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }
    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return ;
    }
    //float mind = gs.params.min_disparity;
    //float maxd = gs.params.max_disparity;
    //srand(0);
    //for(int x = 0; x < gs.cameras.cols; x++) {
    //for(int y = 0; y < gs.cameras.rows; y++) {
    //gs.lines.disp[y*gs.cameras.cols+x] = (float)rand()/(float)RAND_MAX * (maxd-mind) + mind;
    //[>printf("%f\n", gs.lines.disp[y*256+x]);<]
    //}
    //}
    /*printf("MAX DISP is %f\n", gs.params.max_disparity);*/
    /*printf("MIN DISP is %f\n", gs.params.min_disparity);*/
    cudaSetDevice(i);
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*128);
    dim3 grid_size;
    grid_size.x=(cols+BLOCK_W-1)/BLOCK_W;
    grid_size.y=((rows/2)+BLOCK_H-1)/BLOCK_H;
    dim3 block_size;
    block_size.x=BLOCK_W;
    block_size.y=BLOCK_H;

    dim3 grid_size_initrand;
    grid_size_initrand.x=(cols+32-1)/32;
    grid_size_initrand.y=(rows+32-1)/32;
    dim3 block_size_initrand;
    block_size_initrand.x=32;
    block_size_initrand.y=32;

/*     printf("Launching kernel with grid of size %d %d and block of size %d %d and shared size %d %d\nBlock %d %d and radius %d %d and tile %d %d\n",
           grid_size.x,
           grid_size.y,
           block_size.x,
           block_size.y,
           SHARED_SIZE_W,
           SHARED_SIZE_H,
           BLOCK_W,
           BLOCK_H,
           WIN_RADIUS_W,
           WIN_RADIUS_H,
           TILE_W,
           TILE_H
          );
 */    printf("Grid size initrand is grid: %d-%d block: %d-%d\n", grid_size_initrand.x, grid_size_initrand.y, block_size_initrand.x, block_size_initrand.y);

    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    printf("Device memory used: %fMB\n", used/1000000.0f);
    printf("Number of iterations is %d\n", gs.params->iterations);
    printf("Blocksize is %dx%d\n", gs.params->box_hsize,gs.params->box_vsize);
    printf("Disparity threshold is \t%f\n", gs.params->depthThresh);
    printf("Normal threshold is \t%f\n", gs.params->normalThresh);
    printf("Number of consistent points is \t%d\n", gs.params->numConsistentThresh);
    printf("Cam scale is \t%f\n", gs.params->cam_scale);

    //int shared_memory_size = sizeof(float)  * SHARED_SIZE ;
    printf("Fusing points\n");
    cudaEventRecord(start);

    //printf("Computing final disparity\n");
    //for (int cam=0; cam<10; cam++) {
    for (int cam=0; cam<num_views; cam++) {
        fusibile<<< grid_size_initrand, block_size_initrand, cam>>>(gs, cam);
        cudaDeviceSynchronize();
        copy_point_cloud_to_host(gs, cam, pc_list); // slower but saves memory
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\t\tELAPSED %f seconds\n", milliseconds/1000.f);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));

    // print results to file
}

int runcuda(GlobalState &gs, PointCloudList &pc_list, int num_views)
{
    printf("Run cuda\n");
    /*GlobalState *gs = new GlobalState;*/
    if(gs.params->color_processing)
        fusibile_cu<float4>(gs, pc_list, num_views);
    else
        fusibile_cu<float>(gs, pc_list, num_views);
    return 0;
}
