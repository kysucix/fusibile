#pragma once

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#if CV_MAJOR_VERSION == 3
#include "opencv2/core/utility.hpp"
#endif

#include <omp.h>
#include <stdint.h>

using namespace cv;
using namespace std;

typedef Vec<uint16_t, 2> Vec2us;

// parameters for comparison with ground truth
struct GTcheckParameters {
    GTcheckParameters () : gtCheck ( false ), noccCheck ( false ), scale ( 150.0f ), dispTolGT ( 0.5f ), divFactor ( 4.0f ) {}
    bool gtCheck;
    bool noccCheck;
    float scale; // scaling factor just for visualization of error
    float dispTolGT;
    float dispTolGT2;
    //division factor dependent on ground truth data (to get disparity value: imgValue/divFactor)
    float divFactor; //Middleburry small images: 4, big images third: 3, Kitti: 255
};

//pathes to input images (camera images, ground truth, ...)
struct InputFiles {
    InputFiles () : gt_filename ( "" ), gt_nocc_filename ( "" ), occ_filename ( "" ), gt_normal_filename ( "" ), calib_filename ( "" ), images_folder ( "" ), p_folder ( "" ), camera_folder ( "" ), pmvs_folder("") {}
    vector<string> img_filenames; // input camera images (only filenames, path is set in images_folder), names can also be used for calibration data (e.g. for Strecha P, camera)
    string gt_filename; // ground truth image
    string gt_nocc_filename; // non-occluded ground truth image (as provided e.g. by Kitti)
    string occ_filename; // occlusion mask (binary map of all the points that are occluded) (as provided e.g. by Middleburry)
    string gt_normal_filename; // ground truth normal map (for Strecha)
    string calib_filename; // calibration file containing camera matrices (P) (as provided e.g. by Kitti)
    string images_folder; // path to camera input images
    string p_folder; // path to camera projection matrix P (Strecha)
    string krt_file; // path to camera projection matrix P (Strecha)
    string camera_folder; // path to camera calibration matrix K (Strecha)
    string bounding_folder; //path to bounding volume (Strecha)
    string seed_file; //path to bounding volume (Strecha)
    string pmvs_folder; //path to bounding volume (Strecha)
};

//pathes to output files
struct OutputFiles {
    OutputFiles () : parentFolder ( "results" ), disparity_filename ( 0 ) {}
    const char* parentFolder;
    char* disparity_filename;
};

struct Camera {
    Camera () : baseline ( 0.54f ), P ( Mat::eye ( 3,4,CV_32F ) ), R ( Mat::eye ( 3,3,CV_32F ) ), reference ( false ), depthMin ( 2.0f ), depthMax ( 20.0f ) {}
    float baseline;
    Mat_<float> P;
    Mat_<float> P_inv;
    Mat_<float> M_inv;
    //Mat_<float> K;
    Mat_<float> R;
    Mat_<float> t;
    Vec3f C;
    bool reference;
    float depthMin; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    float depthMax; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    //int id; //corresponds to the image name id (eg. 0-10), independent of order in argument list, just dependent on name
    string id;
    Mat_<float> K;
    Mat_<float> K_inv;
    //float f;
};

//parameters for camera geometry setup (assuming that K1 = K2 = K, P1 = K [I | 0] and P2 = K [R | t])
struct CameraParameters {
    CameraParameters () : rectified ( false ), idRef ( 0 ) {}
    Mat_<float> K; //if K varies from camera to camera: K and f need to be stored within Camera
    Mat_<float> K_inv; //if K varies from camera to camera: K and f need to be stored within Camera
    float f;
    bool rectified;
    vector<Camera> cameras;
    int idRef;
    vector<int> viewSelectionSubset;
};

struct Results {
    Results () : error_occ ( 1.0f ), error_noc ( 1.0f ), valid_pixels ( 0.0f ), error_valid ( 1.0f ), error_valid_all ( 1.0f ), total_runtime ( 0.0f ), runtime_per_pixel ( 0.0f ) {}
    float error_occ;
    float error_noc;
    float valid_pixels; // passed occlusion check
    float valid_pixels_gt;
    float error_valid;
    float error_valid_all;
    double total_runtime;
    double runtime_per_pixel;
};
