#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <ctime>
#include <direct.h>
#include "win32_dirent.h"
#define access _access
#else
#include <dirent.h>
#endif


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <sys/types.h>

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <vector_types.h>


#ifdef _MSC_VER
#include <io.h>
#define R_OK 04
#else
#include <unistd.h>
#endif

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check

#include <map> // multimap

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir


//#include "camera.h"
#include "algorithmparameters.h"
#include "globalstate.h"
#include "fusibile.h"

#include "main.h"
#include "fileIoUtils.h"
#include "cameraGeometryUtils.h"
#include "mathUtils.h"
#include "displayUtils.h"
#include "point_cloud_list.h"

#define MAX_NR_POINTS 500000

struct InputData{
    string path;
    //int id;
    string id;
    int camId;
    Camera cam;
    Mat_<float> depthMap;
    Mat_<Vec3b> inputImage;
    Mat_<Vec3f> normals;
};

int getCameraFromId(string id, vector<Camera> &cameras){
    for(size_t i =0; i< cameras.size(); i++) {
        //cout << "Checking camera id " << i << " cameraid " << cameras[i].id << endl;
        if(cameras[i].id.compare(id) == 0)
            return i;
    }
    return -1;
}
static void get_subfolders(
                           const char *dirname,
                           vector<string> &subfolders)
{
    DIR *dir;
    struct dirent *ent;

    // Open directory stream
    dir = opendir (dirname);
    if (dir != NULL) {
        //cout << "Dirname is " << dirname << endl;
        //cout << "Dirname type is " << ent->d_type << endl;
        //cout << "Dirname type DT_DIR " << DT_DIR << endl;

        // Print all files and directories within the directory
        while ((ent = readdir (dir)) != NULL) {
            //cout << "INSIDE" << endl;
            //if(ent->d_type == DT_DIR)
            {
                char* name = ent->d_name;
                if(strcmp(name,".") == 0 || strcmp(ent->d_name,"..") == 0)
                    continue;
                //printf ("dir %s/\n", name);
                subfolders.push_back(string(name));
            }
        }

        closedir (dir);

    } else {
        // Could not open directory
        printf ("Cannot open directory %s\n", dirname);
        exit (EXIT_FAILURE);
    }
}

static void print_help ()
{
    printf ( "\nfusibile\n" );
}

/* process command line arguments
 * Input: argc, argv - command line arguments
 * Output: inputFiles, outputFiles, parameters, gt_parameters, no_display - algorithm parameters
 */
static int getParametersFromCommandLine ( int argc,
                                          char** argv,
                                          InputFiles &inputFiles,
                                          OutputFiles &outputFiles,
                                          AlgorithmParameters &parameters,
                                          GTcheckParameters &gt_parameters,
                                          bool &no_display )
{
    const char* algorithm_opt     = "--algorithm=";
    const char* maxdisp_opt       = "--max-disparity=";
    const char* blocksize_opt     = "--blocksize=";
    const char* cost_tau_color_opt   = "--cost_tau_color=";
    const char* cost_tau_gradient_opt   = "--cost_tau_gradient=";
    const char* cost_alpha_opt   = "--cost_alpha=";
    const char* cost_gamma_opt   = "--cost_gamma=";
    const char* disparity_tolerance_opt = "--disp_tol=";
    const char* normal_tolerance_opt = "--norm_tol=";
    const char* border_value = "--border_value="; //either constant scalar or -1 = REPLICATE
    const char* gtDepth_divFactor_opt = "--gtDepth_divisionFactor=";
    const char* gtDepth_tolerance_opt = "--gtDepth_tolerance=";
    const char* gtDepth_tolerance2_opt = "--gtDepth_tolerance2=";
    const char* nodisplay_opt     = "-no_display";
    const char* colorProc_opt = "-color_processing";
    const char* num_iterations_opt = "--iterations=";
    const char* self_similariy_n_opt = "--ss_n=";
    const char* ct_epsilon_opt = "--ct_eps=";
    const char* cam_scale_opt = "--cam_scale=";
    const char* num_img_processed_opt = "--num_img_processed=";
    const char* n_best_opt = "--n_best=";
    const char* cost_comb_opt = "--cost_comb=";
    const char* cost_good_factor_opt = "--good_factor=";
    const char* depth_min_opt = "--depth_min=";
    const char* depth_max_opt = "--depth_max=";
    //    const char* scale_opt         = "--scale=";
    const char* outputPath_opt = "-output_folder";
    const char* calib_opt = "-calib_file";
    const char* gt_opt = "-gt";
    const char* gt_nocc_opt = "-gt_nocc";
    const char* occl_mask_opt = "-occl_mask";
    const char* gt_normal_opt = "-gt_normal";
    const char* images_input_folder_opt = "-images_folder";
    const char* p_input_folder_opt = "-p_folder";
    const char* krt_file_opt = "-krt_file";
    const char* camera_input_folder_opt = "-camera_folder";
    const char* bounding_folder_opt = "-bounding_folder";
    const char* viewSelection_opt = "-view_selection";
    const char* initial_seed_opt = "--initial_seed";

    const char* disp_thresh_opt = "--disp_thresh=";
    const char* normal_thresh_opt = "--normal_thresh=";
    const char* num_consistent_opt = "--num_consistent=";

    //read in arguments
    for ( int i = 1; i < argc; i++ )
    {
        if ( argv[i][0] != '-' )
        {
            inputFiles.img_filenames.push_back ( argv[i] );
            /*if( inputFiles.imgLeft_filename.empty() )
              inputFiles.imgLeft_filename = argv[i];
              else if( inputFiles.imgRight_filename.empty() )
              inputFiles.imgRight_filename = argv[i];
              */
        }
        else if ( strncmp ( argv[i], algorithm_opt, strlen ( algorithm_opt ) ) == 0 )
        {
            char* _alg = argv[i] + strlen ( algorithm_opt );
            parameters.algorithm = strcmp ( _alg, "pm" ) == 0 ? PM_COST :
            strcmp ( _alg, "ct" ) == 0 ? CENSUS_TRANSFORM :
            strcmp ( _alg, "sct" ) == 0 ? SPARSE_CENSUS :
            strcmp ( _alg, "ct_ss" ) == 0 ? CENSUS_SELFSIMILARITY :
            strcmp ( _alg, "adct" ) == 0 ? ADCENSUS :
            strcmp ( _alg, "adct_ss" ) == 0 ? ADCENSUS_SELFSIMILARITY :
            strcmp ( _alg, "pm_ss" ) == 0 ? PM_SELFSIMILARITY : -1;
            if ( parameters.algorithm < 0 )
            {
                printf ( "Command-line parameter error: Unknown stereo algorithm\n\n" );
                print_help ();
                return -1;
            }
        }
        else if ( strncmp ( argv[i], cost_comb_opt, strlen ( cost_comb_opt ) ) == 0 )
        {
            char* _alg = argv[i] + strlen ( algorithm_opt );
            parameters.cost_comb = strcmp ( _alg, "all" ) == 0 ? COMB_ALL :
            strcmp ( _alg, "best_n" ) == 0 ? COMB_BEST_N :
            strcmp ( _alg, "angle" ) == 0 ? COMB_ANGLE :
            strcmp ( _alg, "good" ) == 0 ? COMB_GOOD : -1;
            if ( parameters.cost_comb < 0 )
            {
                printf ( "Command-line parameter error: Unknown cost combination method\n\n" );
                print_help ();
                return -1;
            }
        }
        else if ( strncmp ( argv[i], maxdisp_opt, strlen ( maxdisp_opt ) ) == 0 )
        {
            if ( sscanf ( argv[i] + strlen ( maxdisp_opt ), "%f", &parameters.max_disparity ) != 1 ||
                 parameters.max_disparity < 1  )
            {
                printf ( "Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer \n" );
                print_help ();
                return -1;
            }
        }
        else if ( strncmp ( argv[i], blocksize_opt, strlen ( blocksize_opt ) ) == 0 )
        {
            int k_size;
            if ( sscanf ( argv[i] + strlen ( blocksize_opt ), "%d", &k_size ) != 1 ||
                 k_size < 1 || k_size % 2 != 1 )
            {
                printf ( "Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n" );
                return -1;
            }
            parameters.box_hsize = k_size;
            parameters.box_vsize = k_size;
        }
        else if ( strncmp ( argv[i], cost_good_factor_opt, strlen ( cost_good_factor_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_good_factor_opt ), "%f", &parameters.good_factor );
        }
        else if ( strncmp ( argv[i], cost_tau_color_opt, strlen ( cost_tau_color_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_tau_color_opt ), "%f", &parameters.tau_color );
        }
        else if ( strncmp ( argv[i], cost_tau_gradient_opt, strlen ( cost_tau_gradient_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_tau_gradient_opt ), "%f", &parameters.tau_gradient );
        }
        else if ( strncmp ( argv[i], cost_alpha_opt, strlen ( cost_alpha_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_alpha_opt ), "%f", &parameters.alpha );
        }
        else if ( strncmp ( argv[i], cost_gamma_opt, strlen ( cost_gamma_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cost_gamma_opt ), "%f", &parameters.gamma );
        }
        else if ( strncmp ( argv[i], border_value, strlen ( border_value ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( border_value ), "%d", &parameters.border_value );
        }
        else if ( strncmp ( argv[i], num_iterations_opt, strlen ( num_iterations_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( num_iterations_opt ), "%d", &parameters.iterations );
        }
        else if ( strncmp ( argv[i], disparity_tolerance_opt, strlen ( disparity_tolerance_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( disparity_tolerance_opt ), "%f", &parameters.dispTol );
        }
        else if ( strncmp ( argv[i], normal_tolerance_opt, strlen ( normal_tolerance_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( normal_tolerance_opt ), "%f", &parameters.normTol );
        }
        else if ( strncmp ( argv[i], self_similariy_n_opt, strlen ( self_similariy_n_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( self_similariy_n_opt ), "%d", &parameters.self_similarity_n );
        }
        else if ( strncmp ( argv[i], ct_epsilon_opt, strlen ( ct_epsilon_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( ct_epsilon_opt ), "%f", &parameters.census_epsilon );
        }
        else if ( strncmp ( argv[i], cam_scale_opt, strlen ( cam_scale_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( cam_scale_opt ), "%f", &parameters.cam_scale );
        }
        else if ( strncmp ( argv[i], num_img_processed_opt, strlen ( num_img_processed_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( num_img_processed_opt ), "%d", &parameters.num_img_processed );
        }
        else if ( strncmp ( argv[i], n_best_opt, strlen ( n_best_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( n_best_opt ), "%d", &parameters.n_best );
        }
        else if ( strncmp ( argv[i], gtDepth_divFactor_opt, strlen ( gtDepth_divFactor_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( gtDepth_divFactor_opt ), "%f", &gt_parameters.divFactor );
        }
        else if ( strncmp ( argv[i], gtDepth_tolerance_opt, strlen ( gtDepth_tolerance_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( gtDepth_tolerance_opt ), "%f", &gt_parameters.dispTolGT );
        }
        else if ( strncmp ( argv[i], gtDepth_tolerance2_opt, strlen ( gtDepth_tolerance2_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( gtDepth_tolerance2_opt ), "%f", &gt_parameters.dispTolGT2 );
        }
        else if ( strncmp ( argv[i], depth_min_opt, strlen ( depth_min_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( depth_min_opt ), "%f", &parameters.depthMin );
        }
        else if ( strncmp ( argv[i], depth_max_opt, strlen ( depth_max_opt ) ) == 0 )
        {
            sscanf ( argv[i] + strlen ( depth_max_opt ), "%f", &parameters.depthMax );
        }
        else if ( strcmp ( argv[i], viewSelection_opt ) == 0 )
            parameters.viewSelection = true;
        else if ( strcmp ( argv[i], nodisplay_opt ) == 0 )
            no_display = true;
        else if ( strcmp ( argv[i], colorProc_opt ) == 0 )
            parameters.color_processing = true;
        else if ( strcmp ( argv[i], "-o" ) == 0 )
            outputFiles.disparity_filename = argv[++i];
        else if ( strcmp ( argv[i], outputPath_opt ) == 0 )
            outputFiles.parentFolder = argv[++i];
        else if ( strcmp ( argv[i], calib_opt ) == 0 )
            inputFiles.calib_filename = argv[++i];
        else if ( strcmp ( argv[i], gt_opt ) == 0 )
            inputFiles.gt_filename = argv[++i];
        else if ( strcmp ( argv[i], gt_nocc_opt ) == 0 )
            inputFiles.gt_nocc_filename = argv[++i];
        else if ( strcmp ( argv[i], occl_mask_opt ) == 0 )
            inputFiles.occ_filename = argv[++i];
        else if ( strcmp ( argv[i], gt_normal_opt ) == 0 )
            inputFiles.gt_normal_filename = argv[++i];
        else if ( strcmp ( argv[i], images_input_folder_opt ) == 0 )
            inputFiles.images_folder = argv[++i];
        else if ( strcmp ( argv[i], p_input_folder_opt ) == 0 )
            inputFiles.p_folder = argv[++i];
        else if ( strcmp ( argv[i], krt_file_opt ) == 0 )
            inputFiles.krt_file = argv[++i];
        else if ( strcmp ( argv[i], camera_input_folder_opt ) == 0 )
            inputFiles.camera_folder = argv[++i];
        else if ( strcmp ( argv[i], initial_seed_opt ) == 0 )
            inputFiles.seed_file = argv[++i];
        else if ( strcmp ( argv[i], bounding_folder_opt ) == 0 )
            inputFiles.bounding_folder = argv[++i];
        else if ( strncmp ( argv[i], disp_thresh_opt, strlen (disp_thresh_opt) ) == 0 )
            sscanf ( argv[i] + strlen (disp_thresh_opt), "%f", &parameters.depthThresh );
        else if ( strncmp ( argv[i], normal_thresh_opt, strlen (normal_thresh_opt) ) == 0 ) {
            float angle_degree;
            sscanf ( argv[i] + strlen (normal_thresh_opt), "%f", &angle_degree );
            parameters.normalThresh = angle_degree * M_PI / 180.0f;
        }
        else if ( strncmp ( argv[i], num_consistent_opt, strlen (num_consistent_opt) ) == 0 )
            sscanf ( argv[i] + strlen (num_consistent_opt), "%d", &parameters.numConsistentThresh );
        else
        {
            printf ( "Command-line parameter error: unknown option %s\n", argv[i] );
            //return -1;
        }
    }
    //cout << "KRt file is " << inputFiles.krt_file << endl;

    return 0;
}

static void selectViews ( CameraParameters &cameraParams, int imgWidth, int imgHeight, bool viewSel ) {
    vector<Camera> cameras = cameraParams.cameras;
    Camera ref = cameras[cameraParams.idRef];

    int x = imgWidth / 2;
    int y = imgHeight / 2;

    cameraParams.viewSelectionSubset.clear ();

    Vec3f viewVectorRef = getViewVector ( ref, x, y);

    // TODO hardcoded value makes it a parameter
    float minimum_angle_degree = 10;
    float maximum_angle_degree = 30;
    float minimum_angle_radians = minimum_angle_degree * M_PI / 180.0f;
    float maximum_angle_radians = maximum_angle_degree * M_PI / 180.0f;
    printf("Accepted intersection angle of central rays is %f to %f degrees\n", minimum_angle_degree, maximum_angle_degree);
    for ( size_t i = 0; i < cameras.size (); i++ ) {
        //if ( i == cameraParams.idRef && !cameraParams.rectified )
        //  continue;

        if ( !viewSel ) { //select all views, dont perform selection
            cameraParams.viewSelectionSubset.push_back ( i );
            continue;
        }

        Vec3f vec = getViewVector ( cameras[i], x, y);

        float angle = getAngle ( viewVectorRef, vec );
        if ( angle > minimum_angle_radians && angle < maximum_angle_radians ) //0.6 select if angle between 5.7 and 34.8 (0.6) degrees (10 and 30 degrees suggested by some paper)
        {
            cameraParams.viewSelectionSubset.push_back ( i );
            printf("Accepting camera %ld with angle\t %f degree (%f radians)\n", i, angle*180.0f/M_PI, angle);
        }
        else
            printf("Discarding camera %ld with angle\t %f degree (%f radians)\n", i, angle*180.0f/M_PI, angle);
    }

}

static void addImageToTextureUint (vector<Mat_<uint8_t> > &imgs, cudaTextureObject_t texs[])
{
    for (unsigned int i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with uint8_t point type
        cudaChannelFormatDesc channelDesc =
        //cudaCreateChannelDesc (8,
        //0,
        //0,
        //0,
        //cudaChannelFormatKindUnsigned);
        cudaCreateChannelDesc<char>();
        // Allocate array with correct size and number of channels
        cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray,
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray,
                                              0,
                                              0,
                                              imgs[i].ptr<uint8_t>(),
                                              imgs[i].step[0],
                                              cols*sizeof(uint8_t),
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModePoint;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        //cudaTextureObject_t &texObj = texs[i];
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        //texs[i] = texObj;
    }
    return;
}
static void addImageToTextureFloatColor (vector<Mat > &imgs, cudaTextureObject_t texs[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

        // Allocate array with correct size and number of channels
        cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray,
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray,
                                              0,
                                              0,
                                              imgs[i].ptr<float>(),
                                              imgs[i].step[0],
                                              cols*sizeof(float)*4,
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
    }
    return;
}

static void addImageToTextureFloatGray (vector<Mat > &imgs, cudaTextureObject_t texs[])
{
    for (size_t i=0; i<imgs.size(); i++)
    {
        int rows = imgs[i].rows;
        int cols = imgs[i].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc (32,
                               0,
                               0,
                               0,
                               cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray,
                                        &channelDesc,
                                        cols,
                                        rows));

        checkCudaErrors (cudaMemcpy2DToArray (cuArray,
                                              0,
                                              0,
                                              imgs[i].ptr<float>(),
                                              imgs[i].step[0],
                                              cols*sizeof(float),
                                              rows,
                                              cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeWrap;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        checkCudaErrors(cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL));
        //texs[i] = texObj;
    }
    return;
}
static int runFusibile (int argc,
                        char **argv,
                        AlgorithmParameters &algParameters
                       )
{
    InputFiles inputFiles;
    string ext = ".png";

    string results_folder = "results/";

    const char* results_folder_opt     = "-input_folder";
    const char* p_input_folder_opt = "-p_folder";
    const char* krt_file_opt = "-krt_file";
    const char* images_input_folder_opt = "-images_folder";
    const char* gt_opt = "-gt";
    const char* gt_nocc_opt = "-gt_nocc";
    const char* pmvs_folder_opt = "--pmvs_folder";
    const char* remove_black_background_opt = "-remove_black_background";
    //read in arguments
    for ( int i = 1; i < argc; i++ )
    {
        if ( strcmp ( argv[i], results_folder_opt ) == 0 ){
            results_folder = argv[++i];
            cout << "input folder is " << results_folder << endl;

        }else if ( strcmp ( argv[i], p_input_folder_opt ) == 0 ){
            inputFiles.p_folder = argv[++i];
        }
        else if ( strcmp ( argv[i], krt_file_opt ) == 0 )
            inputFiles.krt_file = argv[++i];
        else if ( strcmp ( argv[i], images_input_folder_opt ) == 0 ){
            inputFiles.images_folder = argv[++i];

        }else if ( strcmp ( argv[i], gt_opt ) == 0 ){
            inputFiles.gt_filename = argv[++i];

        }else if ( strcmp ( argv[i], gt_nocc_opt ) == 0 ){
            inputFiles.gt_nocc_filename = argv[++i];
        }
        else if ( strncmp ( argv[i], pmvs_folder_opt, strlen ( pmvs_folder_opt ) ) == 0 ) {
            inputFiles.pmvs_folder = argv[++i];
        }
        else if ( strcmp ( argv[i], remove_black_background_opt ) == 0 )
            algParameters.remove_black_background = true;
    }

    if (inputFiles.pmvs_folder.size()>0) {
        inputFiles.images_folder = inputFiles.pmvs_folder + "/visualize/";
        inputFiles.p_folder = inputFiles.pmvs_folder + "/txt/";
    }
    cout <<"image folder is " << inputFiles.images_folder << endl;
    cout <<"p folder is " << inputFiles.p_folder << endl;
    cout <<"pmvs folder is " << inputFiles.pmvs_folder << endl;

    GTcheckParameters gtParameters;

    gtParameters.dispTolGT = 0.1f;
    gtParameters.dispTolGT2 = 0.02f;
    gtParameters.divFactor = 1.0f;
    // create folder to store result images
    time_t timeObj;
    time ( &timeObj );
    tm *pTime = localtime ( &timeObj );

    vector <Mat_<Vec3f> > view_vectors;


    time(&timeObj);
    pTime = localtime(&timeObj);

    char output_folder[256];
    sprintf(output_folder, "%s/consistencyCheck-%04d%02d%02d-%02d%02d%02d/",results_folder.c_str(), pTime->tm_year+1900, pTime->tm_mon+1,pTime->tm_mday,pTime->tm_hour, pTime->tm_min, pTime->tm_sec);
#if defined(_WIN32)
    _mkdir(output_folder);
#else
    mkdir(output_folder, 0777);
#endif

    vector<string> subfolders;
    get_subfolders(results_folder.c_str(), subfolders);
    std::sort(subfolders.begin(), subfolders.end());

    vector< Mat_<Vec3b> > warpedImages;
    vector< Mat_<Vec3b> > warpedImages_inverse;
    //vector< Mat_<float> > depthMaps;
    vector< Mat_<float> > updateMaps;
    vector< Mat_<Vec3f> > updateNormals;
    vector< Mat_<float> > depthMapConsistent;
    vector< Mat_<Vec3f> > normalsConsistent;
    vector< Mat_<Vec3f> > groundTruthNormals;
    vector< Mat_<uint8_t> > valid;


    map< int,string> consideredIds;
    for(size_t i=0;i<subfolders.size();i++) {
        //make sure that it has the right format (DATE_TIME_INDEX)
        size_t n = std::count(subfolders[i].begin(), subfolders[i].end(), '_');
        if(n < 2)
            continue;
        if (subfolders[i][0] != '2')
            continue;

        //get index
        //unsigned found = subfolders[i].find_last_of("_");
        //find second index
        unsigned posFirst = subfolders[i].find_first_of("_") +1;
        unsigned found = subfolders[i].substr(posFirst).find_first_of("_") + posFirst +1;
        string id_string = subfolders[i].substr(found);
        //InputData dat;

        //consideredIds.push_back(id_string);
        consideredIds.insert(pair<int,string>(i,id_string));
        //cout << "id_string is " << id_string << endl;
        //cout << "i is " << i << endl;
        //char outputPath[256];
        //sprintf(outputPath, "%s.png", id_string);

        if( access( (inputFiles.images_folder + id_string + ".png").c_str(), R_OK ) != -1 )
            inputFiles.img_filenames.push_back((id_string + ".png"));
        else if( access( (inputFiles.images_folder + id_string + ".jpg").c_str(), R_OK ) != -1 )
            inputFiles.img_filenames.push_back((id_string + ".jpg"));
        else if( access( (inputFiles.images_folder + id_string + ".ppm").c_str(), R_OK ) != -1 )
            inputFiles.img_filenames.push_back((id_string + ".ppm"));
    }
    size_t numImages = inputFiles.img_filenames.size ();
    cout << "numImages is " << numImages << endl;
    cout << "img_filenames is " << inputFiles.img_filenames.size() << endl;
    algParameters.num_img_processed = min ( ( int ) numImages, algParameters.num_img_processed );

    vector<Mat_<Vec3b> > img_color; // imgLeft_color, imgRight_color;
    vector<Mat_<uint8_t> > img_grayscale;
    for ( size_t i = 0; i < numImages; i++ ) {
        //printf ( "Opening image %ld: %s\n", i, ( inputFiles.images_folder + inputFiles.img_filenames[i] ).c_str () );
        img_grayscale.push_back ( imread ( ( inputFiles.images_folder + inputFiles.img_filenames[i] ), IMREAD_GRAYSCALE ) );
        if ( algParameters.color_processing ) {
            img_color.push_back ( imread ( ( inputFiles.images_folder + inputFiles.img_filenames[i] ), IMREAD_COLOR ) );
        }

        if ( img_grayscale[i].rows == 0 ) {
            printf ( "Image seems to be invalid\n" );
            return -1;
        }
    }

    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    size_t used = total - avail;
    printf("Device memory used: %fMB\n", used/1000000.0f);

    GlobalState *gs = new GlobalState;
	gs->cameras = new CameraParameters_cu;
	gs->pc = new PointCloud;
    cudaMemGetInfo( &avail, &total );
    used = total - avail;
    printf("Device memory used: %fMB\n", used/1000000.0f);

    uint32_t rows = img_grayscale[0].rows;
    uint32_t cols = img_grayscale[0].cols;

    CameraParameters camParams = getCameraParameters (*(gs->cameras),
                                                      inputFiles,algParameters.depthMin,
                                                      algParameters.depthMax,
                                                      algParameters.cam_scale,
                                                      false);
    printf("Camera size is %lu\n", camParams.cameras.size());

    for ( int i = 0; i < algParameters.num_img_processed; i++ ) {
        algParameters.min_disparity = disparityDepthConversion ( camParams.f, camParams.cameras[i].baseline, camParams.cameras[i].depthMax );
        algParameters.max_disparity = disparityDepthConversion ( camParams.f, camParams.cameras[i].baseline, camParams.cameras[i].depthMin );
    }

    selectViews ( camParams, cols, rows, false);
    int numSelViews = camParams.viewSelectionSubset.size ();
    cout << "Selected views: " << numSelViews << endl;
    gs->cameras->viewSelectionSubsetNumber = numSelViews;
    ofstream myfile;
    for ( int i = 0; i < numSelViews; i++ ) {
        cout << camParams.viewSelectionSubset[i] << ", ";
        gs->cameras->viewSelectionSubset[i] = camParams.viewSelectionSubset[i];
    }
    cout << endl;

    vector<InputData> inputData;

    cout << "Reading normals and depth from disk" << endl;
    cout << "Size consideredIds is " << consideredIds.size() << endl;
    for (map<int,string>::iterator it=consideredIds.begin(); it!=consideredIds.end(); ++it){

        //get corresponding camera
        int i = it->first;
        string id = it->second;//consideredIds[i];
        //int id = atoi(id_string.c_str());
        int camIdx = getCameraFromId(id,camParams.cameras);
        //cout << "id is " << id << endl;
        //cout << "camIdx is " << camIdx << endl;
        if(camIdx < 0)// || camIdx == camParams.idRef)
            continue;

        InputData dat;
        dat.id = id;
        dat.camId = camIdx;
        dat.cam = camParams.cameras[camIdx];
        dat.path = results_folder + subfolders[i];
        dat.inputImage = imread((inputFiles.images_folder + id + ext), IMREAD_COLOR);

        //read normal
        cout << "Reading normal " << i << endl;
        readDmbNormal((dat.path + "/normals.dmb").c_str(),dat.normals);

        //read depth
        cout << "Reading disp " << i << endl;
        readDmb((dat.path + "/disp.dmb").c_str(),dat.depthMap);

        //inputData.push_back(move(dat));
        inputData.push_back(dat);

    }
    // run gpu run
    // Init parameters
    gs->params = &algParameters;


    // Init ImageInfo
    //gs->iminfo.cols = img_grayscale[0].cols;
    //gs->iminfo.rows = img_grayscale[0].rows;
    gs->cameras->cols = img_grayscale[0].cols;
    gs->cameras->rows = img_grayscale[0].rows;
    gs->params->cols = img_grayscale[0].cols;
    gs->params->rows = img_grayscale[0].rows;
    gs->resize (img_grayscale.size());
    gs->pc->resize (img_grayscale[0].rows * img_grayscale[0].cols);
	PointCloudList pc_list;
    pc_list.resize (img_grayscale[0].rows * img_grayscale[0].cols);
    pc_list.size=0;
    pc_list.rows = img_grayscale[0].rows;
    pc_list.cols = img_grayscale[0].cols;
    gs->pc->rows = img_grayscale[0].rows;
    gs->pc->cols = img_grayscale[0].cols;

    // Resize lines
    for (size_t i = 0; i<img_grayscale.size(); i++)
    {
        gs->lines[i].resize(img_grayscale[0].rows * img_grayscale[0].cols);
        gs->lines[i].n = img_grayscale[0].rows * img_grayscale[0].cols;
        //gs->lines.s = img_grayscale[0].step[0];
        gs->lines[i].s = img_grayscale[0].cols;
        gs->lines[i].l = img_grayscale[0].cols;
    }

    vector<Mat > img_grayscale_float           (img_grayscale.size());
    vector<Mat > img_color_float               (img_grayscale.size());
    vector<Mat > img_color_float_alpha         (img_grayscale.size());
    vector<Mat > normals_and_depth             (img_grayscale.size());
    vector<Mat_<uint16_t> > img_grayscale_uint (img_grayscale.size());
    for (size_t i = 0; i<img_grayscale.size(); i++)
    {
        //img_grayscale[i].convertTo(img_grayscale_float[i], CV_32FC1, 1.0/255.0); // or CV_32F works (too)
        img_grayscale[i].convertTo(img_grayscale_float[i], CV_32FC1); // or CV_32F works (too)
        img_grayscale[i].convertTo(img_grayscale_uint[i], CV_16UC1); // or CV_32F works (too)
        if(algParameters.color_processing) {
            vector<Mat_<float> > rgbChannels ( 3 );
            img_color_float_alpha[i] = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC4 );
            img_color[i].convertTo (img_color_float[i], CV_32FC3); // or CV_32F works (too)
            Mat alpha( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC1 );
            split (img_color_float[i], rgbChannels);
            rgbChannels.push_back( alpha);
            merge (rgbChannels, img_color_float_alpha[i]);
        }
        /* Create vector of normals and disparities */
        vector<Mat_<float> > normal ( 3 );
        normals_and_depth[i] = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC4 );
        split (inputData[i].normals, normal);
        normal.push_back( inputData[i].depthMap);
        merge (normal, normals_and_depth[i]);

    }
    //int64_t t = getTickCount ();

    // Copy images to texture memory
    if (algParameters.saveTexture) {
        if (algParameters.color_processing)
            addImageToTextureFloatColor (img_color_float_alpha, gs->imgs);
        else
            addImageToTextureFloatGray (img_grayscale_float, gs->imgs);
    }

    addImageToTextureFloatColor (normals_and_depth, gs->normals_depths);

#define pow2(x) ((x)*(x))
#define get_pow2_norm(x,y) (pow2(x)+pow2(y))

    runcuda(*gs, pc_list, numSelViews);
    //Mat_<Vec3f> norm0 = Mat::zeros ( img_grayscale[0].rows, img_grayscale[0].cols, CV_32FC3 );
    Mat_<float> distImg;
    char plyFile[256];
    sprintf ( plyFile, "%s/final3d_model.ply", output_folder);
    printf("Writing ply file %s\n", plyFile);
    //storePlyFileAsciiPointCloud ( plyFile, pc_list, inputData[0].cam, distImg);
    storePlyFileBinaryPointCloud ( plyFile, pc_list, distImg);
    //char xyzFile[256];
    //sprintf ( xyzFile, "%s/final3d_model.xyz", output_folder);
    //printf("Writing ply file %s\n", xyzFile);
    //storeXYZPointCloud ( xyzFile, pc_list, inputData[0].cam, distImg);

    return 0;
}

int main(int argc, char **argv)
{
    if ( argc < 3 )
    {
        print_help ();
        return 0;
    }

    InputFiles inputFiles;
    OutputFiles outputFiles;
	AlgorithmParameters* algParameters = new AlgorithmParameters;
    GTcheckParameters gtParameters;
    bool no_display = false;

    int ret = getParametersFromCommandLine ( argc, argv, inputFiles, outputFiles, *algParameters, gtParameters, no_display );
    if ( ret != 0 )
        return ret;

    Results results;
    ret = runFusibile ( argc, argv, *algParameters);

    return 0;
}

