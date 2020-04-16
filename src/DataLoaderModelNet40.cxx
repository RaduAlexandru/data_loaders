#include "data_loaders/DataLoaderModelNet40.h"

//ros
#include <ros/ros.h>

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//my stuff 
#include "data_loaders/core/MeshCore.h"


using namespace easy_pbr;

DataLoaderModelNet40::DataLoaderModelNet40(const std::string config_file):
    m_clouds_buffer(BUFFER_SIZE),
    m_is_running(true),
    m_idx_cloud_to_read(0)
{
    init_params(config_file);
    // read_pose_file();
    create_transformation_matrices();
    // std::cout << " creating thread" << "\n";
    m_loader_thread=std::thread(&DataLoaderModelNet40::read_data, this);  //starts the spin in another thread
    // std::cout << " finidhed creating thread" << "\n";

}

DataLoaderModelNet40::~DataLoaderModelNet40(){

    m_is_running=false;
    m_loader_thread.join();
}

void DataLoaderModelNet40::init_params(const std::string config_file){
    //get the config filename
    // ros::NodeHandle private_nh("~");
    // std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    // std::string config_file= "config.cfg";

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config loader_config=cfg["loader_modelnet40"];

    // m_nr_clouds_to_skip=loader_config["nr_clouds_to_skip"];
    // m_nr_clouds_to_read=loader_config["nr_clouds_to_read"];
    m_mode=(std::string)loader_config["mode"];
    m_normalize=loader_config["normalize"];

    //get the path where all the off files are 
    fs::path dataset_path = (std::string)loader_config["dataset_path"];
    // fs::path sequence = (std::string)loader_config["sequence"];
    fs::path full_path= dataset_path;

    if(!fs::is_directory(full_path)) {
        LOG(FATAL) << "No directory " << full_path;
    }

    //see how many clouds we have and read the files paths into a vector
    std::vector<fs::path> off_filenames_all;
    for (fs::directory_iterator itr(full_path); itr!=fs::directory_iterator(); ++itr){
        fs::path class_path=itr->path()/m_mode; //will point to the class and then train or test depending on the mode. Eg /dataset/airplane/train
        for (fs::directory_iterator c(class_path); c!=fs::directory_iterator(); ++c){
            off_filenames_all.push_back(c->path());
        }
    }
    std::sort(off_filenames_all.begin(), off_filenames_all.end());

    //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    for (size_t i = 0; i < off_filenames_all.size(); i++) {
        // if( (int)i>=m_nr_clouds_to_skip && ((int)m_npz_filenames.size()<m_nr_clouds_to_read || m_nr_clouds_to_read<0 ) ){
        m_off_filenames.push_back(off_filenames_all[i]);
        // }
    }
    std::cout << "About to read " << m_off_filenames.size() << " clouds" <<std::endl; 


    //label file and colormap



}

void DataLoaderModelNet40::read_data(){

    loguru::set_thread_name("loader_thread_kitti");

    while (m_is_running) {

        //we finished reading so we wait here for a reset
        if(m_idx_cloud_to_read>=m_off_filenames.size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue

            fs::path off_filename=m_off_filenames[ m_idx_cloud_to_read ];
            m_idx_cloud_to_read++;
            // VLOG(1) << "reading " << npz_filename;

            //read off
            MeshCore cloud;
            cloud.load_from_file(off_filename.string());
            if(m_normalize){
                cloud.normalize_size();
                cloud.normalize_position();
            }

            //transform
            cloud.apply_transform(m_tf_worldGL_worldROS); // from worldROS to worldGL

            //some sensible visualization options
            cloud.m_vis.m_show_mesh=true;
            cloud.m_vis.m_show_points=true;
            

            m_clouds_buffer.enqueue(cloud);;

        }

    }

}

bool DataLoaderModelNet40::has_data(){
    if(m_clouds_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    }
}


MeshCore DataLoaderModelNet40::get_cloud(){

    MeshCore cloud;
    m_clouds_buffer.try_dequeue(cloud);

    return cloud;
}



void DataLoaderModelNet40::create_transformation_matrices(){

    m_tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3d worldGL_worldROS_rot;
    worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
}