#include "data_loaders/DataLoaderStanford3DScene.h"

#include "Profiler.h"
//c++
#include <algorithm>
#include <random>

#include <opencv2/imgcodecs.hpp>  //for imread
#include "opencv2/imgproc/imgproc.hpp" //for cv::resize

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//boost
#include <boost/range.hpp>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


//my stuff
#include "RandGenerator.h"

using namespace radu::utils;
using namespace easy_pbr;

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

DataLoaderStanford3DScene::DataLoaderStanford3DScene(const std::string config_file):
    m_is_modified(false),
    m_is_running(false),
    m_frames_color_buffer(BUFFER_SIZE),
    m_frames_depth_buffer(BUFFER_SIZE),
    m_idx_sample_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator),
    m_rgb_subsample_factor(1),
    m_depth_subsample_factor(1)
{

    init_params(config_file);
    if(m_autostart){
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderStanford3DScene::read_data, this);  //starts the spin in another thread
    }

}

DataLoaderStanford3DScene::~DataLoaderStanford3DScene(){

    m_is_running=false;

    if (m_loader_thread.joinable()){
        m_loader_thread.join();
    }
}

void DataLoaderStanford3DScene::init_params(const std::string config_file){

    //read all the parameters
    // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);

    Config loader_config=cfg["loader_stanford_3D_scene"];
    m_autostart=loader_config["autostart"];
    m_nr_samples_to_skip=loader_config["nr_samples_to_skip"];
    m_nr_samples_to_read=loader_config["nr_samples_to_read"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_dataset_path=(std::string)loader_config["dataset_path"];
    m_pose_file_path=(std::string)loader_config["pose_file_path"];
    m_rgb_subsample_factor=loader_config["rgb_subsample_factor"];
    m_depth_subsample_factor=loader_config["depth_subsample_factor"];

}

void DataLoaderStanford3DScene::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";

    init_data_reading();

    m_is_running=true;
    m_loader_thread=std::thread(&DataLoaderStanford3DScene::read_data, this);  //starts the spin in another thread
}

void DataLoaderStanford3DScene::init_data_reading(){

    read_pose_file(m_pose_file_path.string());

    std::vector<fs::path> samples_filenames_all;


    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }
    fs::path dataset_path_full=m_dataset_path;
    for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(dataset_path_full/"color"), {})){

        //get the path name
        std::string img_name=entry.path().stem().string();
        // VLOG(1) << "img name is " << img_name;
        // int img_nr=std::stoi(img_name);
        // VLOG(1) << "img nr is " << img_nr;

        samples_filenames_all.push_back(entry);
    }

    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(samples_filenames_all), std::end(samples_filenames_all), rng);
    }else{
        std::sort(samples_filenames_all.begin(), samples_filenames_all.end());
    }





    //ADDS THE samples to the member std_vector of paths
    //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    for (size_t i = 0; i <samples_filenames_all.size(); i++) {
        if( (int)i>=m_nr_samples_to_skip && ((int)m_samples_filenames.size()<m_nr_samples_to_read || m_nr_samples_to_read<0 ) ){
            m_samples_filenames.push_back(samples_filenames_all[i]);
        }
    }

    std::cout << "About to read " << m_samples_filenames.size() << " samples" <<std::endl;


    CHECK(m_samples_filenames.size()>0) <<"We did not find any samples files to read";


    // //read the intrinsics which in this case are the same for both color and depth
    // the values are from here http://qianyi.info/scenedata.html
    m_K.setIdentity();
    m_K(0,0)=  525.0;
    m_K(1,1)=  525.0;
    m_K(0,2)= 319.5;
    m_K(1,2)= 239.5;


}

void DataLoaderStanford3DScene::read_data(){

    loguru::set_thread_name("loader_thread_vol_ref");


    while (m_is_running ) {

        //we finished reading so we wait here for a reset
        if(m_idx_sample_to_read>=m_samples_filenames.size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_frames_color_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue

            fs::path sample_filename=m_samples_filenames[ m_idx_sample_to_read ];
            if(!m_do_overfit){
                m_idx_sample_to_read++;
            }

            //read frame color and frame depth
            Frame frame_color;
            Frame frame_depth;
            read_sample(frame_color, frame_depth, sample_filename);



            m_frames_color_buffer.enqueue(frame_color);
            m_frames_depth_buffer.enqueue(frame_depth);

        }

    }

}


void DataLoaderStanford3DScene::read_sample(Frame& frame_color, Frame& frame_depth, const boost::filesystem::path& sample_filename){

    TIME_SCOPE("data_loader_stanford3d")

    int frame_idx= std::stoi(sample_filename.stem().string());

    //read color img
    frame_color.rgb_8u=cv::imread(sample_filename.string());
    if(m_rgb_subsample_factor>1){
        cv::Mat resized;
        cv::resize(frame_color.rgb_8u, resized, cv::Size(), 1.0/m_rgb_subsample_factor, 1.0/m_rgb_subsample_factor, cv::INTER_AREA);
        // frame.rgb_8u=resized.clone();
        frame_color.rgb_8u=resized;
    }
    frame_color.rgb_8u.convertTo(frame_color.rgb_32f, CV_32FC3, 1.0/255.0);
    frame_color.width=frame_color.rgb_32f.cols;
    frame_color.height=frame_color.rgb_32f.rows;
    frame_color.frame_idx=frame_idx;


    //read depth
    std::string rgb_name = sample_filename.filename().string();
    std::string depth_path = (sample_filename.parent_path().parent_path()/"depth"/rgb_name).string();
    cv::Mat depth=cv::imread(depth_path, cv::IMREAD_ANYDEPTH);
    if(m_depth_subsample_factor>1){
        cv::Mat resized;
        cv::resize(depth, resized, cv::Size(), 1.0/m_depth_subsample_factor, 1.0/m_depth_subsample_factor, cv::INTER_NEAREST);
        depth=resized;
    }
    depth.convertTo(frame_depth.depth, CV_32FC1, 1.0/1000.0); //the depth was stored in mm but we want it in meters
    // depth.convertTo(frame_depth.depth, CV_32FC1 ); //the depth was stored in meters already so there's nothing to be done with it
    frame_depth.width=frame_depth.depth.cols;
    frame_depth.height=frame_depth.depth.rows;
    frame_depth.frame_idx=frame_idx;

    //get the pose
    for(size_t i=0; i<m_poses_vec.size(); i++){
        if(frame_idx==m_poses_vec[i].frame_idx){
            frame_color.tf_cam_world=m_poses_vec[i].pose.cast<float>().inverse();
            frame_depth.tf_cam_world=m_poses_vec[i].pose.cast<float>().inverse();
            break;
        }
    }

    //rotate so that it looks ok in opengl
    Eigen::Affine3d m_tf_worldGL_world;
    m_tf_worldGL_world.setIdentity();
    Eigen::Matrix3d worldGL_world_rot;
    worldGL_world_rot = Eigen::AngleAxisd(1.0*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_world.matrix().block<3,3>(0,0)=worldGL_world_rot;
    frame_color.tf_cam_world= frame_color.tf_cam_world * m_tf_worldGL_world.cast<float>().inverse(); //from worldgl to world ros, from world ros to cam
    frame_depth.tf_cam_world= frame_depth.tf_cam_world * m_tf_worldGL_world.cast<float>().inverse(); //from worldgl to world ros, from world ros to cam

    //assign K matrix
    frame_color.K=m_K.cast<float>()/m_rgb_subsample_factor;
    frame_depth.K=m_K.cast<float>()/m_depth_subsample_factor;
    frame_color.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0
    frame_depth.K(2,2)=1.0;
}






void DataLoaderStanford3DScene::read_pose_file(std::string pose_file){
    std::ifstream infile( pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << pose_file;
    }

    m_poses_vec.clear();


    // int line_read=0;
    std::string line;
    Eigen::Affine3d pose;
    pose.setIdentity();
    int id1, id2, frame_idx;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >> id1 >> id2 >> frame_idx;
        // VLOG(1) << "pose for frame_idx " << frame_idx;
        //read the poose
        for(int i=0; i<4; i++){
            std::getline(infile, line);
            iss=std::istringstream(line);
            iss >>  pose.matrix()(i,0)>> pose.matrix()(i,1)>> pose.matrix()(i,2)>> pose.matrix()(i,3);
        }

        // VLOG(1) << " pose read is \n " << pose.matrix();
        PoseStanford3DScene pose_stanford;
        pose_stanford.frame_idx=frame_idx;
        pose_stanford.pose=pose;
        m_poses_vec.push_back(pose_stanford);


    }


}

Eigen::Matrix3d DataLoaderStanford3DScene::read_intrinsics_file(std::string intrinsics_file){
    std::ifstream infile( intrinsics_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open intrinsics file " << intrinsics_file;
    }
    int line_read=0;
    std::string line;
    Eigen::Matrix3d K;
    K.setIdentity();
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >>  K.matrix()(line_read,0) >> K.matrix()(line_read,1) >> K.matrix()(line_read,2);
        line_read++;
        if(line_read>=3){
            break;
        }
    }

    return K;
}


bool DataLoaderStanford3DScene::has_data(){
    if(m_frames_color_buffer.peek()==nullptr || m_frames_depth_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    }
}


Frame DataLoaderStanford3DScene::get_color_frame(){

    Frame frame;
    m_frames_color_buffer.try_dequeue(frame);

    return frame;
}

Frame DataLoaderStanford3DScene::get_depth_frame(){

    Frame frame;
    m_frames_depth_buffer.try_dequeue(frame);

    return frame;
}


bool DataLoaderStanford3DScene::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_sample_to_read<m_samples_filenames.size()){
        return false; //there is still more files to read
    }

    //check that there is nothing in the ring buffers
    if(m_frames_color_buffer.peek()!=nullptr || m_frames_depth_buffer.peek()!=nullptr){
        return false; //there is still something in the buffer
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderStanford3DScene::is_finished_reading(){
    //check if this loader has loaded everything
    if(m_idx_sample_to_read<m_samples_filenames.size()){
        return false; //there is still more files to read
    }

    return true; //there is nothing more to read and so we are finished reading

}

void DataLoaderStanford3DScene::reset(){
    m_nr_resets++;
    // we shuffle again the data so as to have freshly shuffled data for the next epoch
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // auto rng = std::default_random_engine(seed);
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(m_samples_filenames), std::end(m_samples_filenames), rng);
    }

    m_idx_sample_to_read=0;
}

int DataLoaderStanford3DScene::nr_samples(){
    return m_samples_filenames.size();
}
