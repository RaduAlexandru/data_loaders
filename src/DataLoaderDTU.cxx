#include "data_loaders/DataLoaderDTU.h"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//cnpy
#include "cnpy.h"

#include <opencv2/core/eigen.hpp>


//my stuff
#include "data_loaders/DataTransformer.h"
#include "easy_pbr/Frame.h"
#include "Profiler.h"
#include "string_utils.h"
#include "numerical_utils.h"
#include "opencv_utils.h"
#include "eigen_utils.h"
#include "RandGenerator.h"
#include "easy_pbr/LabelMngr.h"
// #include "UtilsGL.h"

//json
// #include "json11/json11.hpp"

//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;


DataLoaderDTU::DataLoaderDTU(const std::string config_file):
    m_is_running(false),
    m_autostart(false),
    m_idx_scene_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator),
    m_nr_scenes_read_so_far(0)
{
    init_params(config_file);
    if(m_autostart){
        start();
    }

    // init_data_reading();
    // start_reading_next_scene();

}

DataLoaderDTU::~DataLoaderDTU(){

    m_is_running=false;
    m_loader_thread.join();
}

void DataLoaderDTU::init_params(const std::string config_file){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_dtu"];

    m_autostart=loader_config["autostart"];
    m_read_with_bg_thread = loader_config["read_with_bg_thread"];
    m_shuffle=loader_config["shuffle"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_do_overfit=loader_config["do_overfit"];
    // m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are
    m_restrict_to_scan_idx= loader_config["restrict_to_scan_idx"];
    m_load_as_shell= loader_config["load_as_shell"];
    m_mode= (std::string)loader_config["mode"];
    m_scene_scale_multiplier= loader_config["scene_scale_multiplier"];


}

void DataLoaderDTU::start(){
    CHECK(m_scene_folders.empty()) << " The loader has already been started before. Make sure that you have m_autostart to false";

    init_data_reading();
    read_poses_and_intrinsics();
    start_reading_next_scene();
}

void DataLoaderDTU::init_data_reading(){

    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }


    //load the corresponding file and get from there the scene that we need to read
    fs::path scene_file_path= m_dataset_path/("new_"+m_mode+".lst");

    std::ifstream scene_file(scene_file_path.string() );
    if(!scene_file.is_open()){
        LOG(FATAL) << "Could not open labels file " << scene_file_path;
    }
    // int nr_scenes_read=0;
    for( std::string line; getline( scene_file, line ); ){
        if(line.empty()){
            continue;
        }
        std::string scan=trim_copy(line); //this scan is a string with format "scanNUMBER". We want just the number
        int scan_idx=std::stoi(radu::utils::erase_substring(scan, "scan"));
        VLOG(1) << "from scan line " << scan << "scan idx is " << scan_idx;
        //if we want to load only one of the scans except for all of them
        //push only one of the scenes
        if(m_restrict_to_scan_idx>=0){
            if(m_restrict_to_scan_idx==scan_idx){
                m_scene_folders.push_back(m_dataset_path/scan);;
            }
        }else{
            //push all scenes
            m_scene_folders.push_back(m_dataset_path/scan);
        }

        // nr_scenes_read++;
    }



    VLOG(1) << "loaded nr of scenes " << m_scene_folders.size() << " for mode " << m_mode;

    // shuffle the data if neccsary
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_scene_folders), std::end(m_scene_folders), rng_0);
    }

    CHECK(m_scene_folders.size()!=0 ) << "We have read zero scene folders";


}

void DataLoaderDTU::start_reading_next_scene(){
    CHECK(m_is_running==false) << "The loader thread is already running. Wait until the scene is finished loading before loading a new one. You can check this with finished_reading_scene()";

    std::string scene_path;
    if ( m_idx_scene_to_read< m_scene_folders.size()){
        scene_path=m_scene_folders[m_idx_scene_to_read].string();
    }

    // VLOG(1) << " mode "<< m_mode << "m dof overfit" << m_do_overfit << " scnee size "<< m_scene_folders.size() << " scnee path is " << scene_path;


    if(!m_do_overfit){
        m_idx_scene_to_read++;
    }



    //start the reading
    if (m_loader_thread.joinable()){
        m_loader_thread.join(); //join the thread from the previous iteration of running
    }
    if(!scene_path.empty()){
        if(m_read_with_bg_thread){
            m_is_running=true;
            m_loader_thread=std::thread(&DataLoaderDTU::read_scene, this, scene_path);  //starts to read in another thread
        }else{
            read_scene(scene_path);
        }
    }
}


void DataLoaderDTU::read_scene(const std::string scene_path){
    // VLOG(1) <<" read from path " << scene_path;

    TIME_SCOPE("read_scene");

    m_frames_for_scene.clear();

    std::vector<fs::path> paths;
    for (fs::directory_iterator itr( fs::path(scene_path)/"image"); itr!=fs::directory_iterator(); ++itr){
        fs::path img_path= itr->path();
        paths.push_back(img_path);
    }

    //shuffle the images from this scene
    unsigned seed1 = m_nr_scenes_read_so_far;
    auto rng_1 = std::default_random_engine(seed1);
    if(m_mode=="train"){
        std::shuffle(std::begin(paths), std::end(paths), rng_1);
    }

    //load all the scene for the chosen object
    // for (fs::directory_iterator itr(scene_path); itr!=fs::directory_iterator(); ++itr){
    for (size_t i=0; i<paths.size(); i++){
        // fs::path img_path= itr->path();
        fs::path img_path= paths[i];
        //get only files that end in png
        // VLOG(1) << "img_path" <<img_path;
        if(img_path.filename().string().find("png")!= std::string::npos){
            // VLOG(1) << "png img path " << img_path;

            int img_idx=std::stoi( img_path.stem().string() );
            // VLOG(1) << "img idx is " << img_idx;

            Frame frame;
            frame.frame_idx=img_idx;

            //sets the paths and all the things necessary for the loading of images
            frame.rgb_path=img_path.string();
            frame.subsample_factor=m_subsample_factor;


            //load the images if necessary or delay it for whne it's needed
            frame.load_images=[this]( easy_pbr::Frame& frame ) -> void{ this->load_images_in_frame(frame); };
            if (m_load_as_shell){
                //set the function to load the images whenever it's neede
                frame.is_shell=true;
            }else{
                frame.is_shell=false;
                frame.load_images(frame);
            }







            // //read pose and camera params needs to be read from the camera.npz
            // std::string pose_and_intrinsics_path=(fs::path(scene_path)/"cameras.npz").string();

            // //read npz
            // cnpy::npz_t npz_file = cnpy::npz_load( pose_and_intrinsics_path );
            // cnpy::NpyArray projection_mat_array = npz_file["world_mat_"+std::to_string(img_idx) ]; //one can obtain the keys with https://stackoverflow.com/a/53901903
            // cnpy::NpyArray scale_array = npz_file["scale_mat_"+std::to_string(img_idx) ]; //one can obtain the keys with https://stackoverflow.com/a/53901903

            // // VLOG(1) << " projection_mat_array size" <<  projection_mat_array.shape.size();
            // // VLOG(1) << " scale_array size" <<  scale_array.shape.size();
            // // VLOG(1) << " projection_mat_array shape0 " <<  projection_mat_array.shape[0];
            // // VLOG(1) << " projection_mat_array shape1 " <<  projection_mat_array.shape[1];
            // // VLOG(1) << " scale_array shape0 " <<  scale_array.shape[0];
            // // VLOG(1) << " scale_array shape1 " <<  scale_array.shape[1];

            // //get the P matrix which containst both K and the pose
            // Eigen::Affine3d P;
            // double* projection_mat_data = projection_mat_array.data<double>();
            // P.matrix()= Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor> >(projection_mat_data);
            // // VLOG(1) << "P is " << P.matrix();
            // Eigen::Matrix<double,3,4> P_block = P.matrix().block<3,4>(0,0);
            // // VLOG(1) << P_block;
            // //get scale
            // Eigen::Affine3d S;
            // double* scale_array_data = scale_array.data<double>();
            // S.matrix()= Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor> >(scale_array_data);
            // // VLOG(1) << "S is " << S.matrix();


            // //Get the P_block into K and R and T as done in this line: K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            // cv::Mat P_mat;
            // cv::eigen2cv(P_block, P_mat);
            // cv::Mat K_mat, R_mat, t_mat;
            // cv::decomposeProjectionMatrix(P_mat, K_mat, R_mat, t_mat);
            // // VLOG(1) << "K_Mat has size " << K_mat.rows << " " << K_mat.cols;
            // // VLOG(1) << "T_Mat has size " << R_mat.rows << " " << R_mat.cols;
            // // VLOG(1) << "t_Mat has size " << t_mat.rows << " " << t_mat.cols;
            // Eigen::Matrix3d K, R;
            // Eigen::Vector4d t_full;
            // cv::cv2eigen(K_mat, K);
            // cv::cv2eigen(R_mat, R);
            // cv::cv2eigen(t_mat, t_full);
            // K = K / K(2, 2);
            // // VLOG(1) << "K is " << K;
            // // VLOG(1) << "R is " << R;
            // // VLOG(1) << "t_full is " << t_full;
            // Eigen::Vector3d t;
            // t.x()= t_full.x()/t_full.w();
            // t.y()= t_full.y()/t_full.w();
            // t.z()= t_full.z()/t_full.w();
            // // VLOG(1) << "t is "<<t;


            // // //get the pose into a mat
            // Eigen::Affine3f tf_cam_world;
            // tf_cam_world.linear() = R.transpose().cast<float>();
            // tf_cam_world.translation() = t.cast<float>();
            // // VLOG(1) << "tf_cam_world " << tf_cam_world.matrix();


            // //get S
            // // Eigen::Matrix3d S_block=
            // Eigen::Vector3d norm_trans=S.translation();
            // // VLOG(1) << "norm trans is " << norm_trans;
            // Eigen::Vector3d norm_scale;
            // norm_scale << S(0,0), S(1,1), S(2,2);
            // // VLOG(1) << "norm scale " << norm_scale;
            // tf_cam_world.translation()-=norm_trans.cast<float>();
            // tf_cam_world.translation()=tf_cam_world.translation().array()/norm_scale.cast<float>().array();
            // // VLOG(1) << "pose after the weird scaling " << tf_cam_world.matrix();


            // //transform so the up is in the positive y for a right handed system
            // // self._coord_trans_world = torch.tensor(
            //     // [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            //     // dtype=torch.float32,
            // // )
            // // Eigen::Affine3f rot_world, rot_cam;
            // // rot_world.matrix()<< 1, 0, 0, 0, 0, -1, 0, 0,  0, 0, -1, 0,  0, 0, 0, 1;
            // // rot_cam.matrix()<< 1, 0, 0, 0, 0, -1, 0, 0,  0, 0, -1, 0,  0, 0, 0, 1;
            // // tf_cam_world= rot_world*tf_cam_world*rot_cam;


            // //atteptm2
            // //rotate
            // Eigen::Quaternionf q = Eigen::Quaternionf( Eigen::AngleAxis<float>( -60 * M_PI / 180.0 ,  Eigen::Vector3f::UnitX() ) );
            // Eigen::Affine3f tf_rot;
            // tf_rot.setIdentity();
            // tf_rot.linear()=q.toRotationMatrix();
            // // tf_world_cam=tf_rot*tf_world_cam;
            // tf_cam_world=tf_rot*tf_cam_world;
            // //flip
            // Eigen::Affine3f tf_world_cam=tf_cam_world.inverse();
            // Eigen::DiagonalMatrix<float, 4> diag;
            // diag.diagonal() <<1, -1, 1, 1;
            // tf_world_cam.matrix()=diag*tf_world_cam.matrix()*diag;
            // //flip again the x
            // diag.diagonal() <<-1, 1, 1, 1;
            // tf_world_cam.matrix()=tf_world_cam.matrix()*diag;
            // //flip locally
            // tf_cam_world=tf_world_cam.inverse();





            // frame.K=K.cast<float>();
            // frame.tf_cam_world=tf_cam_world.inverse();



            ///////////////////////////////////////just get it from the hashmap
            frame.K = m_scene2frame_idx2K[scene_path][img_idx];
            frame.tf_cam_world = m_scene2frame_idx2tf_cam_world[scene_path][img_idx];


            if(m_subsample_factor>1){
                // frame.K/=m_subsample_factor;
                // frame.K(2,2)=1.0;
                frame.rescale_K(1.0/m_subsample_factor);
            }




            // if (img_idx==0){
                // exit(1);
            // }




            // CHECK(arr.shape.size()==2) << "arr should have 2 dimensions and it has " << arr.shape.size();
            // CHECK(arr.shape[1]==4) << "arr second dimension should be 4 (x,y,z,label) but it is " << arr.shape[1];

            // //read intensity
            // fs::path absolute_path=fs::absolute(npz_filename).parent_path();
            // fs::path file_name=npz_filename.stem();
            // // fs::path npz_intensity_path=absolute_path/(file_name.string()+"_i"+".npz");
            // // cnpy::npz_t npz_intensity_file = cnpy::npz_load(npz_intensity_path.string());
            // // cnpy::NpyArray arr_intensity = npz_intensity_file["arr_0"]; //one can obtain the keys with https://stackoverflow.com/a/53901903
            // // CHECK(arr_intensity.shape.size()==1) << "arr should have 1 dimensions and it has " << arr.shape.size();


            // //copy into EigenMatrix
            // int nr_points=arr.shape[0];
            // MeshSharedPtr cloud=Mesh::create();
            // cloud->V.resize(nr_points,3);
            // cloud->V.setZero();
            // cloud->L_gt.resize(nr_points,1);
            // cloud->L_gt.setZero();
            // // cloud->I.resize(nr_points,1);
            // // cloud->I.setZero();
            // double* arr_data = arr.data<double>();
            // // float* arr_intensity_data = arr_intensity.data<float>(); //the intensities are as floats while xyz is double. You can check by reading the npz in python
            // for(int i=0; i<nr_points*4; i=i+4){
            //     int row_insert=i/4;

            //     double x=arr_data[i];
            //     double y=arr_data[i+1];
            //     double z=arr_data[i+2];
            //     int label=arr_data[i+3];
            //     // double intensity=arr_intensity_data[row_insert];

            //     cloud->V.row(row_insert) << x,y,z;
            //     cloud->L_gt.row(row_insert) << label;
            //     // cloud->I.row(row_insert) << intensity;


            //     // VLOG(1) << "xyz is " << x << " " << y << " " << z << " " << label;
            //     // exit(1);

            // }














            //rescale things if necessary
            if(m_scene_scale_multiplier>0.0){
                Eigen::Affine3f tf_world_cam_rescaled = frame.tf_cam_world.inverse();
                tf_world_cam_rescaled.translation()*=m_scene_scale_multiplier;
                frame.tf_cam_world=tf_world_cam_rescaled.inverse();
            }









            m_frames_for_scene.push_back(frame);

            // if(m_nr_imgs_to_read>0 && m_frames_for_scene.size()>=m_nr_imgs_to_read){
                // break; //we finished reading how many images we need so we stop the thread
            // }

        }
    }

    // VLOG(1) << "loaded a scene with nr of frames " << m_frames_for_scene.size();
    CHECK(m_frames_for_scene.size()!=0) << "Clouldn't load any images for this scene in path " << scene_path;

    m_nr_scenes_read_so_far++;

    //shuffle the images from this scene
    unsigned seed = m_nr_scenes_read_so_far;
    auto rng_0 = std::default_random_engine(seed);
    std::shuffle(std::begin(m_frames_for_scene), std::end(m_frames_for_scene), rng_0);

    m_is_running=false;
}

void DataLoaderDTU::load_images_in_frame(easy_pbr::Frame& frame){

    frame.is_shell=false;


    // VLOG(1) << "load image from" << frame.rgb_path ;
    cv::Mat rgb_8u=cv::imread(frame.rgb_path );
    if(frame.subsample_factor>1){
        cv::Mat resized;
        cv::resize(rgb_8u, resized, cv::Size(), 1.0/frame.subsample_factor, 1.0/frame.subsample_factor, cv::INTER_AREA);
        rgb_8u=resized;
    }
    frame.rgb_8u=rgb_8u;

    // VLOG(1) << "img type is " << radu::utils::type2string( frame.rgb_8u.type() );
    frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
    frame.width=frame.rgb_32f.cols;
    frame.height=frame.rgb_32f.rows;
    // VLOG(1) << " frame width ad height " << frame.width << " " << frame.height;




}

void DataLoaderDTU::read_poses_and_intrinsics(){

    // std::unordered_map<std::string,      std::unordered_map<int, Eigen::Affine3f>     > m_scene2frame_idx2tf_cam_world;
    // std::unordered_map<std::string,      std::unordered_map<int, Eigen::Matrix3f>    > m_scene2frame_idx2K;


    for(size_t scene_idx; scene_idx<m_scene_folders.size(); scene_idx++){

        std::string scene_path=m_scene_folders[scene_idx].string();
        VLOG(1) << "reading poses and intrinsics for scene " << fs::path(scene_path).stem();


        std::vector<fs::path> paths;
        for (fs::directory_iterator itr( fs::path(scene_path)/"image"); itr!=fs::directory_iterator(); ++itr){
            fs::path img_path= itr->path();
            paths.push_back(img_path);
        }



        //read pose and camera params needs to be read from the camera.npz
        std::string pose_and_intrinsics_path=(fs::path(scene_path)/"cameras.npz").string();
        cnpy::npz_t npz_file = cnpy::npz_load( pose_and_intrinsics_path );



        //load all the scene for the chosen object
        // for (fs::directory_iterator itr(scene_path); itr!=fs::directory_iterator(); ++itr){
        for (size_t i=0; i<paths.size(); i++){
            // fs::path img_path= itr->path();
            fs::path img_path= paths[i];
            //get only files that end in png
            // VLOG(1) << "img_path" <<img_path;
            if(img_path.filename().string().find("png")!= std::string::npos){
                // VLOG(1) << "png img path " << img_path;

                int img_idx=std::stoi( img_path.stem().string() );
                // VLOG(1) << "img idx is " << img_idx;



                //read npz
                cnpy::NpyArray projection_mat_array = npz_file["world_mat_"+std::to_string(img_idx) ]; //one can obtain the keys with https://stackoverflow.com/a/53901903
                cnpy::NpyArray scale_array = npz_file["scale_mat_"+std::to_string(img_idx) ]; //one can obtain the keys with https://stackoverflow.com/a/53901903

                // VLOG(1) << " projection_mat_array size" <<  projection_mat_array.shape.size();
                // VLOG(1) << " scale_array size" <<  scale_array.shape.size();
                // VLOG(1) << " projection_mat_array shape0 " <<  projection_mat_array.shape[0];
                // VLOG(1) << " projection_mat_array shape1 " <<  projection_mat_array.shape[1];
                // VLOG(1) << " scale_array shape0 " <<  scale_array.shape[0];
                // VLOG(1) << " scale_array shape1 " <<  scale_array.shape[1];

                //get the P matrix which containst both K and the pose
                Eigen::Affine3d P;
                double* projection_mat_data = projection_mat_array.data<double>();
                P.matrix()= Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor> >(projection_mat_data);
                // VLOG(1) << "P is " << P.matrix();
                Eigen::Matrix<double,3,4> P_block = P.matrix().block<3,4>(0,0);
                // VLOG(1) << P_block;
                //get scale
                Eigen::Affine3d S;
                double* scale_array_data = scale_array.data<double>();
                S.matrix()= Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor> >(scale_array_data);
                // VLOG(1) << "S is " << S.matrix();


                //Get the P_block into K and R and T as done in this line: K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
                cv::Mat P_mat;
                cv::eigen2cv(P_block, P_mat);
                cv::Mat K_mat, R_mat, t_mat;
                cv::decomposeProjectionMatrix(P_mat, K_mat, R_mat, t_mat);
                // VLOG(1) << "K_Mat has size " << K_mat.rows << " " << K_mat.cols;
                // VLOG(1) << "T_Mat has size " << R_mat.rows << " " << R_mat.cols;
                // VLOG(1) << "t_Mat has size " << t_mat.rows << " " << t_mat.cols;
                Eigen::Matrix3d K, R;
                Eigen::Vector4d t_full;
                cv::cv2eigen(K_mat, K);
                cv::cv2eigen(R_mat, R);
                cv::cv2eigen(t_mat, t_full);
                K = K / K(2, 2);
                // VLOG(1) << "K is " << K;
                // VLOG(1) << "R is " << R;
                // VLOG(1) << "t_full is " << t_full;
                Eigen::Vector3d t;
                t.x()= t_full.x()/t_full.w();
                t.y()= t_full.y()/t_full.w();
                t.z()= t_full.z()/t_full.w();
                // VLOG(1) << "t is "<<t;


                // //get the pose into a mat
                Eigen::Affine3f tf_world_cam;
                tf_world_cam.linear() = R.transpose().cast<float>();
                tf_world_cam.translation() = t.cast<float>();
                // VLOG(1) << "tf_world_cam " << tf_world_cam.matrix();


                //get S
                // Eigen::Matrix3d S_block=
                Eigen::Vector3d norm_trans=S.translation();
                // VLOG(1) << "norm trans is " << norm_trans;
                Eigen::Vector3d norm_scale;
                norm_scale << S(0,0), S(1,1), S(2,2);
                // VLOG(1) << "norm scale " << norm_scale;
                tf_world_cam.translation()-=norm_trans.cast<float>();
                tf_world_cam.translation()=tf_world_cam.translation().array()/norm_scale.cast<float>().array();
                // VLOG(1) << "pose after the weird scaling " << tf_world_cam.matrix();



                //atteptm2
                //rotate
                Eigen::Quaternionf q = Eigen::Quaternionf( Eigen::AngleAxis<float>( -60 * M_PI / 180.0 ,  Eigen::Vector3f::UnitX() ) );
                Eigen::Affine3f tf_rot;
                tf_rot.setIdentity();
                tf_rot.linear()=q.toRotationMatrix();
                // tf_world_cam=tf_rot*tf_world_cam;
                tf_world_cam=tf_rot*tf_world_cam;
                //flip
                Eigen::Affine3f tf_cam_world=tf_world_cam.inverse();
                Eigen::DiagonalMatrix<float, 4> diag;
                diag.diagonal() <<1, -1, 1, 1;
                tf_cam_world.matrix()=diag*tf_cam_world.matrix()*diag;
                //flip again the x
                diag.diagonal() <<-1, 1, 1, 1;
                tf_cam_world.matrix()=tf_cam_world.matrix()*diag;
                //flip locally
                // tf_world_cam=tf_cam_world.inverse();

                //add it to the hashmaps
                m_scene2frame_idx2tf_cam_world[scene_path][img_idx]=tf_cam_world;
                m_scene2frame_idx2K[scene_path][img_idx]=K.cast<float>();


            }
        }





    }

}




bool DataLoaderDTU::finished_reading_scene(){
    return !m_is_running;
}
bool DataLoaderDTU::has_data(){
    return finished_reading_scene();
}

Frame DataLoaderDTU::get_random_frame(){
    int random_idx=m_rand_gen->rand_int(0, m_frames_for_scene.size()-1);
    // int random_idx=0;
    return m_frames_for_scene[random_idx];
}

Frame DataLoaderDTU::get_frame_at_idx( const int idx){
    CHECK(idx<m_frames_for_scene.size()) << "idx is out of bounds. It is " << idx << " while m_frames has size " << m_frames_for_scene.size();

    Frame  frame= m_frames_for_scene[idx];

    return frame;
}




bool DataLoaderDTU::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_scene_to_read<m_scene_folders.size()){
        return false; //there is still more files to read
    }


    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderDTU::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_scene_folders), std::end(m_scene_folders), rng_0);
    }

    m_idx_scene_to_read=0;
}

int DataLoaderDTU::nr_samples(){
    return m_frames_for_scene.size();
}
int DataLoaderDTU::nr_scenes(){
    return m_scene_folders.size();
}


void DataLoaderDTU::set_restrict_to_scan_idx(const int scan_idx){
    m_restrict_to_scan_idx=scan_idx;
}

void DataLoaderDTU::set_mode_train(){
    m_mode="train";
}
void DataLoaderDTU::set_mode_test(){
    m_mode="test";
}
void DataLoaderDTU::set_mode_validation(){
    m_mode="val";
}


std::unordered_map<std::string, std::string> DataLoaderDTU::create_mapping_classnr2classname(){

    //from https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/datasets/shapenet.py

    std::unordered_map<std::string, std::string> classnr2classname;

    classnr2classname["04379243"]="table";
    classnr2classname["03211117"]="monitor";
    classnr2classname["04401088"]="phone";

    classnr2classname["04530566"]="watercraft";
    classnr2classname["03001627"]="chair";
    classnr2classname["03636649"]="lamp";

    classnr2classname["03691459"]="speaker";
    classnr2classname["02828884"]="bench";
    classnr2classname["02691156"]="plane";

    classnr2classname["02808440"]="bathtub";
    classnr2classname["02871439"]="bookcase";
    classnr2classname["02773838"]="bag";

    classnr2classname["02801938"]="basket";
    classnr2classname["02880940"]="bowl";
    classnr2classname["02924116"]="bus";

    classnr2classname["02933112"]="cabinet";
    classnr2classname["02942699"]="camera";
    classnr2classname["02958343"]="car";

    classnr2classname["03207941"]="dishwasher";
    classnr2classname["03337140"]="file";
    classnr2classname["03624134"]="knife";

    classnr2classname["03642806"]="laptop";
    classnr2classname["03710193"]="mailbox";
    classnr2classname["03761084"]="microwave";

    classnr2classname["03928116"]="piano";
    classnr2classname["03938244"]="pillow";
    classnr2classname["03948459"]="pistol";

    classnr2classname["04004475"]="printer";
    classnr2classname["04099429"]="rocket";
    classnr2classname["04256520"]="sofa";

    classnr2classname["04554684"]="washer";
    classnr2classname["04090263"]="rifle";
    classnr2classname["02946921"]="can";

    return classnr2classname;
}

Eigen::Affine3f DataLoaderDTU::process_extrinsics_line(const std::string line){

    // //remove any "[" or "]" in the line
    // std::string line_processed=line;
    // line_processed.erase(std::remove(line_processed.begin(), line_processed.end(), '['), line_processed.end());
    // line_processed.erase(std::remove(line_processed.begin(), line_processed.end(), ']'), line_processed.end());


    // // std::vector<std::string> tokens = radu::utils::split(line_processed, " ");
    // std::vector<std::string> tokens = radu::utils::split(line_processed, ",");
    // float azimuth = std::stof(tokens[0]);
    // float elevation = std::stof(tokens[1]);
    // float distance = std::stof(tokens[3]);
    // VLOG(1) << "line is " << line;
    // VLOG(1) << "azimuth elev and dist " << azimuth << " " << elevation << " " << distance;

    // Eigen::Affine3f tf;

    // //from compute_camera_params() in https://github.com/NVIDIAGameWorks/kaolin/blob/a76a004ada95280c6a0a821678cf1b886bcb3625/kaolin/mathutils/geometry/transformations.py
    // float theta = radu::utils::degrees2radians(azimuth);
    // float phi = radu::utils::degrees2radians(elevation);

    // float camY = distance * std::sin(phi);
    // float temp = distance * std::cos(phi);
    // float camX = temp * std::cos(theta);
    // float camZ = temp * std::sin(theta);
    // // cam_pos = np.array([camX, camY, camZ])
    // Eigen::Vector3f t;
    // t << camX,camY,camZ;

    // Eigen::Vector3f axisZ = t;
    // Eigen::Vector3f axisY = Eigen::Vector3f::UnitY();
    // Eigen::Vector3f axisX = axisY.cross(axisZ);
    // axisY = axisZ.cross(axisX);

    // // cam_mat = np.array([axisX, axisY, axisZ])
    // Eigen::Matrix3f R;
    // R.col(0)=axisX;
    // R.col(1)=axisY;
    // R.col(2)=-axisZ;
    // // l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    // // l2[l2 == 0] = 1
    // // cam_mat = cam_mat / np.expand_dims(l2, 1)

    // Eigen::Vector3f norm_vec=R.colwise().norm();
    // VLOG(1) << "norm is " << norm_vec;
    // // R=R.colwise()/norm;
    // for (int i=0; i<3; i++){
    //     float norm=norm_vec(i);
    //     for (int j=0; j<3; j++){
    //         // R(i,j) = R(i,j)/norm;
    //         R(j,i) = R(j,i)/norm;
    //     }
    // }
    // norm_vec=R.colwise().norm();
    // VLOG(1) << "norm is " << norm_vec;

    // tf.translation() = t;
    // tf.linear() = R;

    // //just to make sure it's orthogonal
    // // Eigen::AngleAxisf aa(R);    // RotationMatrix to AxisAngle
    // // R = aa.toRotationMatrix();  // AxisAngle      to RotationMatrix
    // // tf.linear() = R;

    // Eigen::Affine3f tf_ret=tf.inverse();
    // // Eigen::Affine3f tf_ret=tf;


















    //attempt 2 by looking at https://github.com/Xharlie/ShapenetRender_more_variation/blob/master/cam_read.py
    //  F_MM = 35.  # Focal length
    // SENSOR_SIZE_MM = 32.
    // PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    // RESOLUTION_PCT = 100.
    // SKEW = 0.
    // CAM_MAX_DIST = 1.75
    // CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
    //                       [1.0, -4.371138828673793e-08, -0.0],
    //                       [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    float cam_max_dist=1.75;
    Eigen::Matrix3f cam_rot;
    cam_rot <<1.910685676922942e-15, 4.371138828673793e-08, 1.0,
            1.0, -4.371138828673793e-08, -0.0,
            4.371138828673793e-08, 1.0, -4.371138828673793e-08;


    std::string line_processed=line;
    line_processed.erase(std::remove(line_processed.begin(), line_processed.end(), '['), line_processed.end());
    line_processed.erase(std::remove(line_processed.begin(), line_processed.end(), ']'), line_processed.end());


    // std::vector<std::string> tokens = radu::utils::split(line_processed, " ");
    std::vector<std::string> tokens = radu::utils::split(line_processed, ",");
    float az = std::stof(tokens[0]);
    float el = std::stof(tokens[1]);
    float distance_ratio = std::stof(tokens[3]);
    // float ox = std::stof(tokens[7]);
    // float oy = std::stof(tokens[8]);
    // float oz = std::stof(tokens[9]);

    // # Calculate rotation and translation matrices.
    // # Step 1: World coordinate to object coordinate.
    float sa = std::sin(radu::utils::degrees2radians(-az));
    float ca = std::cos(radu::utils::degrees2radians(-az));
    float se = std::sin(radu::utils::degrees2radians(-el));
    float ce = std::cos(radu::utils::degrees2radians(-el));
    // R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
    //                                       (sa * ce, ca, sa * se),
    //                                       (-se, 0, ce))))
    Eigen::Matrix3f R_world2obj;
    R_world2obj <<ca * ce, -sa, ca * se,
                    sa * ce, ca, sa * se,
                    -se, 0, ce;
    Eigen::Matrix3f trans;
    trans=R_world2obj.transpose();
    R_world2obj=trans;


    // # Step 2: Object coordinate to camera coordinate.
    // R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    Eigen::Matrix3f R_obj2cam=cam_rot.transpose();
    Eigen::Matrix3f R_world2cam = R_obj2cam * R_world2obj;
    // cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
    //                                        0,
    //                                        0)))
    Eigen::Vector3f cam_location;
    cam_location <<  distance_ratio * cam_max_dist, 0, 0;
    // # print('distance', distance_ratio * CAM_MAX_DIST)
    Eigen::Vector3f T_world2cam = -1 * R_obj2cam * cam_location;

    // // # Step 3: Fix blender camera's y and z axis direction.
    // R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    Eigen::Matrix3f R_camfix;
    R_camfix <<1, 0, 0,
              0,1,0,
              0,0,-1;
    R_world2cam = R_camfix * R_world2cam;
    T_world2cam = R_camfix * T_world2cam;



    Eigen::Affine3f tf_ret;
    tf_ret.linear()=R_world2cam;
    tf_ret.translation()=T_world2cam;


    //rotate 90 degrees
    Eigen::Affine3f tf_worldGL_worldROS;
    tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3f worldGL_worldROS_rot;
    worldGL_worldROS_rot = Eigen::AngleAxisf(-0.5*M_PI, Eigen::Vector3f::UnitX());
    tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
    Eigen::Affine3f tf_worldROS_worldGL=tf_worldGL_worldROS.inverse();
    Eigen::Affine3f tf_ret_cor=tf_worldGL_worldROS*tf_ret.inverse();
    tf_ret=tf_ret_cor.inverse();






    return tf_ret;

}
