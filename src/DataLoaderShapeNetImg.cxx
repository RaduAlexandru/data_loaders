#include "data_loaders/DataLoaderShapeNetImg.h"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;


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
#include "UtilsGL.h"

//json 
// #include "json11/json11.hpp"

//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;


DataLoaderShapeNetImg::DataLoaderShapeNetImg(const std::string config_file):
    m_is_running(false),
    m_idx_scene_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator)
{
    init_params(config_file);
    // if(m_autostart){
        // start();
    // }

    init_data_reading();
    start_reading_next_scene();

}

DataLoaderShapeNetImg::~DataLoaderShapeNetImg(){

    m_is_running=false;
    m_loader_thread.join();
}

void DataLoaderShapeNetImg::init_params(const std::string config_file){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_shapenet_img"];

    // m_autostart=loader_config["autostart"];
    m_nr_samples_to_skip=loader_config["nr_samples_to_skip"];
    m_nr_samples_to_read=loader_config["nr_samples_to_read"];
    m_shuffle=loader_config["shuffle"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_do_overfit=loader_config["do_overfit"];
    m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are 


    //data transformer
    // Config transformer_config=loader_config["transformer"];
    // m_transformer=std::make_shared<DataTransformer>(transformer_config);

}



void DataLoaderShapeNetImg::init_data_reading(){
    
    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }
    
    //make the mapping between the weird numbers in the files and the class label
    std::unordered_map<std::string, std::string> classnr2classname = create_mapping_classnr2classname( );


    //go to the folder for that specific object. Read through all the folders which will give me a gibberish of numbers and map that to the class name. If we found a match then we read the scenes
    fs::path chosen_object_path;
    for (fs::directory_iterator itr(m_dataset_path); itr!=fs::directory_iterator(); ++itr){
        fs::path object_path= itr->path();
        if (!fs::is_regular_file(object_path)){
            //check that this number matched the object we chose
            std::string class_nr=object_path.stem().string();
            // VLOG(1) << "class nr is " << class_nr;
            std::string class_name=classnr2classname[class_nr];
            if(class_name==m_restrict_to_object){
                VLOG(1) << "found " << class_name << " in path " << object_path;
                chosen_object_path=object_path;
                break;
            }

        }
    }
    CHECK( !chosen_object_path.empty() ) << "Could not find the object " << m_restrict_to_object;


    //load all the scene for the chosen object
    int nr_read=0;
    for (fs::directory_iterator itr(chosen_object_path); itr!=fs::directory_iterator(); ++itr){
        if( nr_read>=m_nr_samples_to_skip && ((int)m_scene_folders.size()<m_nr_samples_to_read || m_nr_samples_to_read<0 ) ){
            fs::path scene_path= itr->path()/"rendering";
            m_scene_folders.push_back(scene_path);
        }
        nr_read++;
    }

    // shuffle the data if neccsary
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_scene_folders), std::end(m_scene_folders), rng_0);
    }

    CHECK(m_scene_folders.size()!=0 ) << "We have read zero scene folders";


}

void DataLoaderShapeNetImg::start_reading_next_scene(){
    CHECK(m_is_running==false) << "The loader thread is already running. Wait until the scene is finished loading before loading a new one. You can check this with finished_reading_scene()";

    std::string scene_path;
    if ( m_idx_scene_to_read< m_scene_folders.size()){
        scene_path=m_scene_folders[m_idx_scene_to_read].string();
    }



    if(!m_do_overfit){
        m_idx_scene_to_read++;
    }

    

    //start the reading
    if (m_loader_thread.joinable()){
        m_loader_thread.join(); //join the thread from the previous iteration of running
    }
    if(!scene_path.empty()){
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderShapeNetImg::read_scene, this, scene_path);  //starts to read in another thread
    }
}


void DataLoaderShapeNetImg::read_scene(const std::string scene_path){
    // VLOG(1) <<" read from path " << scene_path;

    m_frames_for_scene.clear();

    //load all the scene for the chosen object
    for (fs::directory_iterator itr(scene_path); itr!=fs::directory_iterator(); ++itr){
        fs::path img_path= itr->path();
        //get only files that end in png
        // VLOG(1) << "img_path" <<img_path.filename();
        if(img_path.filename().string().find("png")!= std::string::npos){
            // VLOG(1) << "png img path " << img_path;

            int img_idx=std::stoi( img_path.stem().string() );
            // VLOG(1) << "img idx is " << img_idx;

            Frame frame;

            //get rgba image and get the alpha in a mask
            cv::Mat rgba_8u=cv::imread(img_path.string(), cv::IMREAD_UNCHANGED );
            if(m_subsample_factor>1){
                cv::Mat resized;
                cv::resize(rgba_8u, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
                rgba_8u=resized;
            }
            std::vector<cv::Mat> channels(4);
            cv::split(rgba_8u, channels);
            cv::threshold( channels[3], frame.mask, 0.0, 1.0, cv::THRESH_BINARY);
            channels.pop_back();
            cv::merge(channels, frame.rgb_8u);

            // frame.rgb_8u=cv::imread(img_path.string(), cv::IMREAD_UNCHANGED );
            // VLOG(1) << "img type is " << radu::utils::type2string( frame.rgb_8u.type() );
            frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
            frame.width=frame.rgb_32f.cols;
            frame.height=frame.rgb_32f.rows;

            //read pose and camera params

            //intrisncis are from here 
            // https://github.com/facebookresearch/pytorch3d/blob/778383eef77a23686f3d0e68834b29d6d73f8501/pytorch3d/datasets/r2n2/r2n2.py
            // and from https://github.com/facebookresearch/meshrcnn/blob/master/shapenet/utils/coords.py
            // ther we also have zmin and zmax
            // but it seems that it's not actually  a K matrix but rather a projection matrix as  an opengl projection matrix like in here http://www.songho.ca/opengl/gl_projectionmatrix.html
            // so it projects from camera coordinates to clip coordinates but we want a K matrix that projects to screen coords
            Eigen::Matrix4f P;
            P << 
            2.1875, 0.0, 0.0, 0.0,
            0.0, 2.1875, 0.0, 0.0,
            0.0, 0.0, -1.002002, -0.2002002,
            0.0, 0.0, -1.0, 0.0;
            Eigen::Matrix3f K = opengl_proj_to_intrinsics(P, 137, 137);
            // VLOG(1) << "K is " << K;
            frame.K=K;
            frame.K/=m_subsample_factor;
            frame.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0

            //the extrinsics are stored in rendering_metadata.txt, stored as azimuth elevation and distance 
            //processing of this can be seen here: https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/datasets/shapenet.py
            Eigen::Affine3f tf_cam_world;
            int lines_read=0;
            bool found=false;
            std::ifstream metadata_file( (fs::path(scene_path)/"rendering_metadata.txt").string() );
            if(!metadata_file.is_open()){
                LOG(FATAL) << "Could not open the rendering metadata file ";
            }
            for( std::string line; getline( metadata_file, line ); ){
                if (lines_read==img_idx){
                    // VLOG(1) << "img idx" << img_idx << "reading line " << lines_read << " line " << line;
                    tf_cam_world=process_extrinsics_line(line);
                    found=true;
                    break;
                }
                lines_read++;
            }
            CHECK(found) << "Could not find a corrsponding line in the metadata for img " << img_idx;
            // VLOG(1) << "TF is " << tf_cam_world.matrix();
            frame.tf_cam_world=tf_cam_world;


            m_frames_for_scene.push_back(frame);

        }
    }

    CHECK(m_frames_for_scene.size()!=0) << "Clouldn't load any images for this scene in path " << scene_path; 

    //shuffle the images from this scene 
    unsigned seed = m_nr_resets;
    auto rng_0 = std::default_random_engine(seed);
    std::shuffle(std::begin(m_frames_for_scene), std::end(m_frames_for_scene), rng_0);

    m_is_running=false;
}


bool DataLoaderShapeNetImg::finished_reading_scene(){
    return !m_is_running;
}

Frame DataLoaderShapeNetImg::get_random_frame(){
    int random_idx=m_rand_gen->rand_int(0, m_frames_for_scene.size()-1);
    // int random_idx=0;
    return m_frames_for_scene[random_idx];
}




bool DataLoaderShapeNetImg::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_scene_to_read<m_scene_folders.size()){
        return false; //there is still more files to read
    }
   

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderShapeNetImg::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed); 
        std::shuffle(std::begin(m_scene_folders), std::end(m_scene_folders), rng_0);
    }

    m_idx_scene_to_read=0;
}

int DataLoaderShapeNetImg::nr_samples(){
    return m_scene_folders.size();
}

std::unordered_map<std::string, std::string> DataLoaderShapeNetImg::create_mapping_classnr2classname(){

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

Eigen::Affine3f DataLoaderShapeNetImg::process_extrinsics_line(const std::string line){

    std::vector<std::string> tokens = radu::utils::split(line, " ");
    float azimuth = std::stof(tokens[0]);
    float elevation = std::stof(tokens[1]);
    float distance = std::stof(tokens[3]);

    Eigen::Affine3f tf;

    //from compute_camera_params() in https://github.com/NVIDIAGameWorks/kaolin/blob/a76a004ada95280c6a0a821678cf1b886bcb3625/kaolin/mathutils/geometry/transformations.py 
    float theta = radu::utils::degrees2radians(azimuth);
    float phi = radu::utils::degrees2radians(elevation);

    float camY = distance * std::sin(phi);
    float temp = distance * std::cos(phi);
    float camX = temp * std::cos(theta);
    float camZ = temp * std::sin(theta);
    // cam_pos = np.array([camX, camY, camZ])
    Eigen::Vector3f t;
    t << camX,camY,camZ;

    Eigen::Vector3f axisZ = t;
    Eigen::Vector3f axisY = Eigen::Vector3f::UnitY();
    Eigen::Vector3f axisX = axisY.cross(axisZ);
    axisY = axisZ.cross(axisX);

    // cam_mat = np.array([axisX, axisY, axisZ])
    Eigen::Matrix3f R;
    R.col(0)=axisX; 
    R.col(1)=axisY; 
    R.col(2)=-axisZ; 
    // l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    // l2[l2 == 0] = 1
    // cam_mat = cam_mat / np.expand_dims(l2, 1)

    tf.translation() = t;
    tf.linear() = R;

    //just to make sure it's orthogonal
    Eigen::AngleAxisf aa(R);    // RotationMatrix to AxisAngle
    R = aa.toRotationMatrix();  // AxisAngle      to RotationMatrix

    Eigen::Affine3f tf_ret=tf.inverse(); 
    // Eigen::Affine3f tf_ret=tf; 



    return tf_ret;

}

// std::shared_ptr<LabelMngr> DataLoaderShapeNetImg::label_mngr(){
//     CHECK(m_label_mngr) << "label_mngr was not created";
//     return m_label_mngr;
// }

// void DataLoaderShapeNetImg::set_mode_train(){
//     m_mode="train";
// }
// void DataLoaderShapeNetImg::set_mode_test(){
//     m_mode="test";
// }
// void DataLoaderShapeNetImg::set_mode_validation(){
//     m_mode="val";
// }


// std::string DataLoaderShapeNetImg::get_object_name(){
//     return m_restrict_to_object;
// }

// void DataLoaderShapeNetImg::set_object_name(const std::string object_name){
//     //kill data loading thread 
//     m_is_running=false;
//     m_loader_thread.join();

//     //clear all data 
//     m_idx_img_to_read=0;
//     m_nr_resets=0;
//     m_pts_filenames.clear();
//     m_labels_filenames.clear();
//     // m_imgs_buffer.clear();
//     //deque until ihe cloud buffer is empty 
//     bool has_data=true;
//     MeshSharedPtr dummy_cloud;
//     while(has_data){
//         has_data=m_imgs_buffer.try_dequeue(dummy_cloud);
//     }

//     //set the new object_name
//     m_restrict_to_object=object_name;

//     //start loading thread again
//     start();

// }



// void DataLoaderShapeNetImg::create_transformation_matrices(){

//     m_tf_worldGL_worldROS.setIdentity();
//     Eigen::Matrix3d worldGL_worldROS_rot;
//     worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
//     m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
// }
