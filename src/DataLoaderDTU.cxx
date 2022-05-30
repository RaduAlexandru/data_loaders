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
    m_restrict_to_scene_name= (std::string)loader_config["restrict_to_scene_name"];
    m_load_as_shell= loader_config["load_as_shell"];
    m_mode= (std::string)loader_config["mode"];
    m_load_mask=loader_config["load_mask"];
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


    //we find this the new_train.lst file only in the pixelnerf DTU version but not in the neus one
    bool found_scene_file=boost::filesystem::exists(scene_file_path);

    if (found_scene_file){
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
            // int scan_idx=std::stoi(radu::utils::erase_substring(scan, "scan"));
            VLOG(1) << "from scan line " << scan;
            //if we want to load only one of the scans except for all of them
            //push only one of the scenes
            if(!m_restrict_to_scene_name.empty()){
                if(m_restrict_to_scene_name==scan){
                    m_scene_folders.push_back(m_dataset_path/scan);;
                }
            }else{
                //push all scenes
                m_scene_folders.push_back(m_dataset_path/scan);
            }

            // nr_scenes_read++;
        }
    }else{
        //LOADING ALL SCENES because we are loading from the neus dataset
        LOG(WARNING) << "Loading all scenes because we could not find a new_train.lst file";
        for (fs::directory_iterator itr(m_dataset_path); itr!=fs::directory_iterator(); ++itr){
            fs::path scene_path= itr->path();
            VLOG(1) << "scene_path" << scene_path;
            //get scene_name
            // int scan_idx=std::stoi(radu::utils::erase_substring(scene_path.filename().string(), "dtu_scan"));
            std::string scene_name=scene_path.filename().string();
            if(!m_restrict_to_scene_name.empty()){
                if(m_restrict_to_scene_name==scene_name){
                    m_scene_folders.push_back(scene_path);;
                }
            }else{
                //push all scenes
                m_scene_folders.push_back(scene_path);
            }
        }


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
    if ( m_idx_scene_to_read< (int)m_scene_folders.size()){
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

            if (m_load_mask){
                std::string mask_path=(fs::path(scene_path)/"mask"/img_path.filename()).string();
                CHECK(boost::filesystem::exists(mask_path)) << "Mask does not exist under path" << mask_path;

                frame.mask_path=mask_path;
            }


            //load the images if necessary or delay it for whne it's needed
            frame.load_images=[this]( easy_pbr::Frame& frame ) -> void{ this->load_images_in_frame(frame); };
            if (m_load_as_shell){
                //set the function to load the images whenever it's neede
                frame.is_shell=true;
            }else{
                frame.is_shell=false;
                frame.load_images(frame);
            }



            //just get it from the hashmap
            frame.K = m_scene2frame_idx2K[scene_path][img_idx];
            frame.tf_cam_world = m_scene2frame_idx2tf_cam_world[scene_path][img_idx];


            if(m_subsample_factor>1){
                frame.rescale_K(1.0/m_subsample_factor);
            }




            










            //rescale things if necessary
            if(m_scene_scale_multiplier>0.0){
                Eigen::Affine3f tf_world_cam_rescaled = frame.tf_cam_world.inverse();
                tf_world_cam_rescaled.translation()*=m_scene_scale_multiplier;
                frame.tf_cam_world=tf_world_cam_rescaled.inverse();
            }









            m_frames_for_scene.push_back(frame);


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


    //load also mask if it's there
    if (!frame.mask_path.empty()){
        cv::Mat mask=cv::imread(frame.mask_path );
        if(frame.subsample_factor>1){
            cv::Mat resized;
            cv::resize(mask, resized, cv::Size(), 1.0/frame.subsample_factor, 1.0/frame.subsample_factor, cv::INTER_NEAREST);
            mask=resized;
        }
        mask.convertTo(frame.mask, CV_32FC3, 1.0/255.0);
        // VLOG(1) << "read mask of type "<< type2string( mask.type() );
        // frame.mask=mask;

        //multiply the rgb with the mask
        // frame.rgb_8u=frame.rgb_8u*mask;
    }

    // VLOG(1) << "img type is " << radu::utils::type2string( frame.rgb_8u.type() );
    frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
    frame.width=frame.rgb_32f.cols;
    frame.height=frame.rgb_32f.rows;
    // VLOG(1) << " frame width ad height " << frame.width << " " << frame.height;

    //multiply mask with rgb
    // frame.rgb_32f=frame.rgb_32f*frame.mask;


}

void DataLoaderDTU::read_poses_and_intrinsics(){



    for(size_t scene_idx=0; scene_idx<m_scene_folders.size(); scene_idx++){

        std::string scene_path=m_scene_folders[scene_idx].string();
        VLOG(1) << "reading poses and intrinsics for scene " << fs::path(scene_path).stem();


        std::vector<fs::path> paths;
        for (fs::directory_iterator itr( fs::path(scene_path)/"image"); itr!=fs::directory_iterator(); ++itr){
            fs::path img_path= itr->path();
            paths.push_back(img_path);
        }



        //read pose and camera params needs to be read from the camera.npz
        std::string pose_and_intrinsics_path=(fs::path(scene_path)/"cameras.npz").string();
        //if it doesn't exists it means we might be using the neus dataset which means we have to load camera_sphere.npz
        bool found_attempt1=boost::filesystem::exists(pose_and_intrinsics_path);
        if (!found_attempt1){
            pose_and_intrinsics_path=(fs::path(scene_path)/"cameras_sphere.npz").string();
        }
        cnpy::npz_t npz_file = cnpy::npz_load( pose_and_intrinsics_path );

        bool using_pixelnerf_format=true;
        if (!found_attempt1){
            using_pixelnerf_format=false;
        }



        //load all the scene for the chosen object
        // for (fs::directory_iterator itr(scene_path); itr!=fs::directory_iterator(); ++itr){
        for (size_t i=0; i<paths.size(); i++){
            fs::path img_path= paths[i];
            //get only files that end in png
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
                Eigen::Affine3d S;
                //pixelenrf format uses doubles in the nzp files and the neus format uses floats for some reason...
                if (using_pixelnerf_format){
                    //P
                    Eigen::Affine3d P_tmp;
                    double* projection_mat_data = projection_mat_array.data<double>();
                    P_tmp.matrix()= Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor> >(projection_mat_data);
                    //S
                    Eigen::Affine3d S_tmp;
                    double* scale_array_data = scale_array.data<double>();
                    S_tmp.matrix()= Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor> >(scale_array_data);
                    //cast
                    P=P_tmp.cast<double>();
                    S=S_tmp.cast<double>();
                }else{
                    //P
                    Eigen::Affine3f P_tmp;
                    float* projection_mat_data = projection_mat_array.data<float>();
                    P_tmp.matrix()= Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor> >(projection_mat_data);
                    //S
                    Eigen::Affine3f S_tmp;
                    float* scale_array_data = scale_array.data<float>();
                    S_tmp.matrix()= Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor> >(scale_array_data);
                    //cast
                    P=P_tmp.cast<double>();
                    S=S_tmp.cast<double>();
                }
                Eigen::Matrix<double,3,4> P_block = P.matrix().block<3,4>(0,0);
                


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
                Eigen::Quaternionf q = Eigen::Quaternionf( Eigen::AngleAxis<float>( 115 * M_PI / 180.0 ,  Eigen::Vector3f::UnitX() ) );
                Eigen::Affine3f tf_rot;
                tf_rot.setIdentity();
                tf_rot.linear()=q.toRotationMatrix();
                tf_world_cam=tf_rot*tf_world_cam;
                Eigen::Affine3f tf_cam_world=tf_world_cam.inverse();
                

                //add it to the hashmaps
                m_scene2frame_idx2tf_cam_world[scene_path][img_idx]=tf_cam_world;
                m_scene2frame_idx2K[scene_path][img_idx]=K.cast<float>();

                // exit(1);


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
    CHECK(idx<(int)m_frames_for_scene.size()) << "idx is out of bounds. It is " << idx << " while m_frames has size " << m_frames_for_scene.size();

    Frame  frame= m_frames_for_scene[idx];

    return frame;
}




bool DataLoaderDTU::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_scene_to_read<(int)m_scene_folders.size()){
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



void DataLoaderDTU::set_scene_scale_multiplier(const float scene_scale_multiplier){
    m_scene_scale_multiplier=scene_scale_multiplier;
}

void DataLoaderDTU::set_load_mask(bool load_mask){
    m_load_mask=load_mask;
}

void DataLoaderDTU::set_dataset_path(const std::string dataset_path){
    m_dataset_path=dataset_path;
}

void DataLoaderDTU::set_restrict_to_scene_name(const std::string scene_name){
    m_restrict_to_scene_name=scene_name;
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

