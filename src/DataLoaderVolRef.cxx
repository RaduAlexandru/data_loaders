#include "data_loaders/DataLoaderVolRef.h"

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
#include "string_utils.h"

using namespace radu::utils;
using namespace easy_pbr;

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

DataLoaderVolRef::DataLoaderVolRef(const std::string config_file):
    m_is_modified(false),
    m_is_running(false),
    m_frames_color_buffer(BUFFER_SIZE),
    m_frames_depth_buffer(BUFFER_SIZE),
    m_idx_sample_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator),
    m_rgb_subsample_factor(1),
    m_depth_subsample_factor(1),
    m_idx_colorframe_to_return(0),
    m_idx_depthframe_to_return(0)

{

    init_params(config_file);
    if(m_autostart){
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderVolRef::read_data, this);  //starts the spin in another thread
    }

}

DataLoaderVolRef::~DataLoaderVolRef(){

    m_is_running=false;

    if (m_loader_thread.joinable()){
        m_loader_thread.join();
    }
}

void DataLoaderVolRef::init_params(const std::string config_file){

    //read all the parameters
    // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);

    Config loader_config=cfg["loader_vol_ref"];
    m_autostart=loader_config["autostart"];
    m_preload=loader_config["preload"];
    m_load_rgb_with_valid_depth= loader_config["load_rgb_with_valid_depth"];
    m_nr_samples_to_skip=loader_config["nr_samples_to_skip"];
    m_nr_samples_to_read=loader_config["nr_samples_to_read"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_dataset_path=(std::string)loader_config["dataset_path"];
    m_rgb_subsample_factor=loader_config["rgb_subsample_factor"];
    m_depth_subsample_factor=loader_config["depth_subsample_factor"];

    m_scene_translation=loader_config["scene_translation"];
    m_scene_scale_multiplier= loader_config["scene_scale_multiplier"];

}

void DataLoaderVolRef::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";

    init_data_reading();

    m_is_running=true;
     if (m_preload){
        read_data(); //if we prelaod we don't need to use any threads and it may cause some other issues
    }else{
        m_loader_thread=std::thread(&DataLoaderVolRef::read_data, this);  //starts the spin in another thread
    }
}

void DataLoaderVolRef::init_data_reading(){

    std::vector<fs::path> samples_filenames_all;


    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }
    fs::path dataset_path_full=m_dataset_path;
    for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(dataset_path_full), {})){
        //grab only the color png images
        // bool found_color=entry.path().stem().string().find("color")!= std::string::npos;
        // bool found_png=entry.path().stem().string().find("png")!= std::string::npos;
        // VLOG(1) << "entry is " << entry.path().stem().string() << " has found color " << found_color;
        // VLOG(1) << "entry is " << entry.path().stem().string() << " has found png " << found_png;
        if( entry.path().stem().string().find("color")!= std::string::npos && entry.path().stem().string().find("frame")!= std::string::npos  ){
            samples_filenames_all.push_back(entry);
        }
    }





    //ADDS THE samples to the member std_vector of paths
    //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    for (size_t i = 0; i <samples_filenames_all.size(); i++) {
        if( (int)i>=m_nr_samples_to_skip && ((int)m_samples_filenames.size()<m_nr_samples_to_read || m_nr_samples_to_read<0 ) ){


            //if we are loading only from selected idxs
            // VLOG(1) << "m_load_from_idxs has size " << m_load_from_idxs.size();
            if(m_load_from_idxs.size()){
                // VLOG(1) << "loading from selected indexes";
                bool do_we_load_from_idx=false;
                for(int j=0; j < m_load_from_idxs.rows(); j++){
                    if((int)i==m_load_from_idxs[j]){
                        do_we_load_from_idx=true;
                        // VLOG(1) << "loading for this idx";
                    }
                }
                if (!do_we_load_from_idx){
                    continue;
                    // VLOG(1) << "NOT loading for this idx";
                }
            }


            m_samples_filenames.push_back(samples_filenames_all[i]);
        }
    }


    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(m_samples_filenames), std::end(m_samples_filenames), rng);
    }else{
        std::sort(m_samples_filenames.begin(), m_samples_filenames.end());
    }



    std::cout << "About to read " << m_samples_filenames.size() << " samples" <<std::endl;


    CHECK(m_samples_filenames.size()>0) <<"We did not find any samples files to read";


    //read the intrinsics for color and depth
    m_K_color.setIdentity();
    std::string intrinsics_color=(m_dataset_path/"colorIntrinsics.txt").string();
    m_K_color=read_intrinsics_file(intrinsics_color);
    m_K_depth.setIdentity();
    std::string intrinsics_depth=(m_dataset_path/"depthIntrinsics.txt").string();
    m_K_depth=read_intrinsics_file(intrinsics_depth);



}

void DataLoaderVolRef::read_data(){

    loguru::set_thread_name("loader_thread_vol_ref");

    //if we preload, we just read the meshes and store them in memory, data transformation will be done while reading the mesh
    if (m_preload){
        for(size_t i=0; i<m_samples_filenames.size(); i++ ){

            fs::path sample_filename=m_samples_filenames[ m_idx_sample_to_read ];
            VLOG(1) << "preloading from " << sample_filename;
            if(!m_do_overfit){
                m_idx_sample_to_read++;
            }
            // MeshSharedPtr cloud=read_sample(sample_filename);
            // m_clouds_vec.push_back(cloud);


            //read frame color and frame depth
            Frame frame_color;
            Frame frame_depth;
            read_sample(frame_color, frame_depth, sample_filename);

            m_frames_color_vec.push_back(frame_color);
            m_frames_depth_vec.push_back(frame_depth);

        }

    }else{ //we continously read from disk


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

}


void DataLoaderVolRef::read_sample(Frame& frame_color, Frame& frame_depth, const boost::filesystem::path& sample_filename){

    //get frame idx
    std::string sample_filename_basename= sample_filename.stem().string(); //the stem has format frame-00000.color.png  We want just the number
    //remove the frame and color so we are left only with the number
    std::string frame_idx_str=  radu::utils::erase_substrings(sample_filename_basename, {".color", "frame-"});
    frame_color.frame_idx= std::stoi(frame_idx_str);
    frame_depth.frame_idx= std::stoi(frame_idx_str);

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

    //read depth
    std::string name = sample_filename.string().substr(0, sample_filename.string().size()-9); //removes the last 5 characters corresponding to "color"
    cv::Mat depth=cv::imread(name+"depth.png", cv::IMREAD_ANYDEPTH);
    if(m_depth_subsample_factor>1){
        cv::Mat resized;
        cv::resize(depth, resized, cv::Size(), 1.0/m_depth_subsample_factor, 1.0/m_depth_subsample_factor, cv::INTER_NEAREST);
        depth=resized;
    }
    depth.convertTo(frame_depth.depth, CV_32FC1, 1.0/1000.0); //the depth was stored in mm but we want it in meters
    // depth.convertTo(frame_depth.depth, CV_32FC1 ); //the depth was stored in mm but we want it in meters
    frame_depth.width=frame_depth.depth.cols;
    frame_depth.height=frame_depth.depth.rows;

    //read pose file
    std::string pose_file=name+"pose.txt";
    Eigen::Affine3d tf_world_cam=read_pose_file(pose_file);
    // VLOG(1) << "pose from tf_world_cam" << pose_file << " is " << tf_world_cam.matrix();
    // frame_color.tf_cam_world=tf_world_cam.inverse().cast<float>();
    // frame_depth.tf_cam_world=tf_world_cam.inverse().cast<float>();
    // frame_color.tf_cam_world=tf_world_cam.cast<float>();
    // frame_depth.tf_cam_world=tf_world_cam.cast<float>();

    //for some reason the y is flipped so we unflip it
    // tf_world_cam.linear().col(1)=-tf_world_cam.linear().col(1);

    Eigen::Affine3d m_tf_worldGL_world;
    m_tf_worldGL_world.setIdentity();
    Eigen::Matrix3d worldGL_world_rot;
    worldGL_world_rot = Eigen::AngleAxisd(1.0*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_world.matrix().block<3,3>(0,0)=worldGL_world_rot;
    frame_color.tf_cam_world= tf_world_cam.cast<float>().inverse() * m_tf_worldGL_world.cast<float>().inverse(); //from worldgl to world ros, from world ros to cam
    frame_depth.tf_cam_world= tf_world_cam.cast<float>().inverse() * m_tf_worldGL_world.cast<float>().inverse(); //from worldgl to world ros, from world ros to cam

   
    //assign K matrix
    // frame_color.K=m_K_color.cast<float>()/m_rgb_subsample_factor;
    // frame_depth.K=m_K_depth.cast<float>()/m_depth_subsample_factor;
    // frame_color.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0
    // frame_depth.K(2,2)=1.0;
    frame_color.K=m_K_color.cast<float>();
    frame_depth.K=m_K_depth.cast<float>();
    if(m_rgb_subsample_factor>1){
        frame_color.rescale_K(1.0/m_rgb_subsample_factor);
    }
    if(m_depth_subsample_factor>1){
        frame_depth.rescale_K(1.0/m_depth_subsample_factor);
    }

    //if the depth and the rgb have the same size then we can use the depth to compute a mask for the rgb part
    if (frame_color.rgb_32f.size == frame_depth.depth.size){
        cv::Mat mask;
        cv::threshold( frame_depth.depth, mask, 0.0, 1.0, cv::THRESH_BINARY);
        frame_color.mask=mask;
        frame_depth.mask=mask;
    }

    //if we want we can get the RGB onyl for the valid part which means the parts that has a depth
    if(m_load_rgb_with_valid_depth){
        cv::Mat rgb_with_valid_depth=frame_color.rgb_with_valid_depth(frame_depth);
        frame_color.rgb_32f= rgb_with_valid_depth;
    }

    //rescale things if necessary
    if(m_scene_scale_multiplier>0.0 || !m_scene_translation.isZero() ){
        Eigen::Affine3f tf_world_cam_rescaled = frame_color.tf_cam_world.inverse().cast<float>();
        tf_world_cam_rescaled.translation()+=m_scene_translation;
        tf_world_cam_rescaled.translation()*=m_scene_scale_multiplier;
        frame_color.tf_cam_world=tf_world_cam_rescaled.inverse();
        frame_depth.tf_cam_world=tf_world_cam_rescaled.inverse();
    }
    //if the scene is rescaled the depth map also needs to be
    if(m_scene_scale_multiplier>0.0 ){
        frame_depth.depth*= m_scene_scale_multiplier;

    }

}

Frame DataLoaderVolRef::closest_color_frame(const Frame& frame){

double lowest_score=std::numeric_limits<double>::max();
fs::path best_path;

for(int i=0; i<(int)m_samples_filenames.size(); i++){
    fs::path sample_filename=m_samples_filenames[i];

    std::string name = sample_filename.string().substr(0, sample_filename.string().size()-9); //removes the last 5 characters corresponding to "color"
    std::string pose_file=name+"pose.txt";
    Eigen::Affine3d tf_worldros_cam=read_pose_file(pose_file);

    //get the difference in transaltion and difference in angle of looking at
    Eigen::Affine3d m_tf_worldGL_world;
    m_tf_worldGL_world.setIdentity();
    Eigen::Matrix3d worldGL_world_rot;
    worldGL_world_rot = Eigen::AngleAxisd(1.0*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_world.matrix().block<3,3>(0,0)=worldGL_world_rot;
    Eigen::Affine3d other_tf_cam_world= tf_worldros_cam.cast<double>().inverse() * m_tf_worldGL_world.inverse(); //from worldgl to world ros, from world ros to cam

    double diff_translation= (frame.tf_cam_world.cast<double>().inverse().translation() - other_tf_cam_world.inverse().translation()).norm();

    //diff in angle
    double diff_angle= 1.0- frame.tf_cam_world.cast<double>().inverse().linear().col(2).dot( other_tf_cam_world.inverse().linear().col(2) );

    double score= 0.5*diff_translation + 0.5*diff_angle;

    if (score<0.00001){
        //if the score is exactly zero then we are just comparing with the same frame
        continue;
    }

        if (score<lowest_score){
            best_path=sample_filename;
            lowest_score=score;
        }



    }


    //read frame color and frame depth
    Frame frame_color;
    Frame frame_depth;
    read_sample(frame_color, frame_depth, best_path);

    return frame_color;

}


Frame DataLoaderVolRef::closest_depth_frame(const Frame& frame){

    double lowest_score=std::numeric_limits<double>::max();
    fs::path best_path;

    for(int i=0; i<(int)m_samples_filenames.size(); i++){
        fs::path sample_filename=m_samples_filenames[i];

        std::string name = sample_filename.string().substr(0, sample_filename.string().size()-9); //removes the last 5 characters corresponding to "color"
        std::string pose_file=name+"pose.txt";
        Eigen::Affine3d tf_worldros_cam=read_pose_file(pose_file);

        //get the difference in transaltion and difference in angle of looking at
        Eigen::Affine3d m_tf_worldGL_world;
        m_tf_worldGL_world.setIdentity();
        Eigen::Matrix3d worldGL_world_rot;
        worldGL_world_rot = Eigen::AngleAxisd(1.0*M_PI, Eigen::Vector3d::UnitX());
        m_tf_worldGL_world.matrix().block<3,3>(0,0)=worldGL_world_rot;
        Eigen::Affine3d other_tf_cam_world= tf_worldros_cam.cast<double>().inverse() * m_tf_worldGL_world.inverse(); //from worldgl to world ros, from world ros to cam

        double diff_translation= (frame.tf_cam_world.cast<double>().inverse().translation() - other_tf_cam_world.inverse().translation()).norm();

        //diff in angle
        double diff_angle= 1.0- frame.tf_cam_world.cast<double>().inverse().linear().col(2).dot( other_tf_cam_world.inverse().linear().col(2) );

        double score= 0.5*diff_translation + 0.5*diff_angle;

        if (score<0.00001){
            //if the score is exactly zero then we are just comparing with the same frame
            continue;
        }

        if (score<lowest_score){
            best_path=sample_filename;
            lowest_score=score;
        }



    }

    // VLOG(1) << "Best depth frame has score " << lowest_score;
    //read frame color and frame depth
    Frame frame_color;
    Frame frame_depth;
    read_sample(frame_color, frame_depth, best_path);

    return frame_depth;

}


void DataLoaderVolRef::load_only_from_idxs(const Eigen::VectorXi& vec){
    // if(m_shuffle){
    //     LOG(WARNING) << "We are shuffling after every reset so selecting some indexes now will change every time we reset. This may not be what you want so you may consider setting shuffle to false";
    // }
    m_load_from_idxs=vec;
}




Eigen::Affine3d DataLoaderVolRef::read_pose_file(std::string pose_file){
    std::ifstream infile( pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << pose_file;
    }
    int line_read=0;
    std::string line;
    Eigen::Affine3d pose;
    pose.setIdentity();
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >>  pose.matrix()(line_read,0)>> pose.matrix()(line_read,1)>> pose.matrix()(line_read,2)>> pose.matrix()(line_read,3);
        line_read++;
    }

    return pose;
}

Eigen::Matrix3d DataLoaderVolRef::read_intrinsics_file(std::string intrinsics_file){
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


bool DataLoaderVolRef::has_data(){
    if (m_preload){
        return true;
    }else{

        if(m_frames_color_buffer.peek()==nullptr || m_frames_depth_buffer.peek()==nullptr){
            return false;
        }else{
            return true;
        }

    }
}


Frame DataLoaderVolRef::get_color_frame(){

    if (m_preload){
        CHECK(m_idx_colorframe_to_return<m_frames_color_vec.size()) << " m_idx_colorframe_to_return is out of bounds. m_idx_colorframe_to_return is " << m_idx_colorframe_to_return << " and colorframe vec is " << m_frames_color_vec.size();

        Frame frame =  m_frames_color_vec[m_idx_colorframe_to_return];

        m_idx_colorframe_to_return++;

        return frame;


    }else{

        Frame frame;
        m_frames_color_buffer.try_dequeue(frame);

        return frame;
    }
}

Frame DataLoaderVolRef::get_depth_frame(){

    if (m_preload){
        CHECK(m_idx_depthframe_to_return<m_frames_depth_vec.size()) << " m_idx_depthframe_to_return is out of bounds. m_idx_depthframe_to_return is " << m_idx_depthframe_to_return << " and depthframe vec is " << m_frames_depth_vec.size();

        Frame frame =  m_frames_depth_vec[m_idx_depthframe_to_return];

        m_idx_depthframe_to_return++;

        return frame;


    }else{

        Frame frame;
        m_frames_depth_buffer.try_dequeue(frame);

        return frame;

    }
}

Frame DataLoaderVolRef::get_frame_at_idx( const int idx){
    CHECK(idx<(int)m_frames_color_vec.size()) << "idx is out of bounds. It is " << idx << " while m_frames_color_vec has size " << m_frames_color_vec.size();
    CHECK(m_preload) <<"Getting frame of a certain index only works when preloading";

    Frame  frame= m_frames_color_vec[idx];

    return frame;
}

Frame DataLoaderVolRef::get_depth_frame_at_idx( const int idx){
    CHECK(idx<(int)m_frames_depth_vec.size()) << "idx is out of bounds. It is " << idx << " while m_frames_depth_vec has size " << m_frames_depth_vec.size();
    CHECK(m_preload) <<"Getting frame of a certain index only works when preloading";

    Frame  frame= m_frames_depth_vec[idx];

    return frame;
}


bool DataLoaderVolRef::is_finished(){

    if(m_preload){
        if (m_idx_colorframe_to_return>=m_frames_color_vec.size() || m_idx_depthframe_to_return>m_frames_depth_vec.size()  ){
            return true;
        }else{
            return false;
        }


    }else{


        //check if this loader has loaded everything
        if(m_idx_sample_to_read<m_samples_filenames.size()){
            return false; //there is still more files to read
        }

        //if ANY of the two frame buffers is empty then we say that we finished reading. This is because not always we want to read the depth bffer
        if(m_frames_color_buffer.peek()==nullptr || m_frames_depth_buffer.peek()==nullptr){
            return true;
        }

        return false; //there is still something in at least one of the buffers

    }

}


bool DataLoaderVolRef::is_finished_reading(){

    if(m_preload){
        if (m_idx_colorframe_to_return>=m_frames_color_vec.size() || m_idx_depthframe_to_return>m_frames_depth_vec.size()  ){
            return true;
        }else{
            return false;
        }


    }else{

        //check if this loader has loaded everything
        if(m_idx_sample_to_read<m_samples_filenames.size()){
            return false; //there is still more files to read
        }

        return true; //there is nothing more to read and so we are finished reading

    }

}

void DataLoaderVolRef::reset(){
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
    m_idx_colorframe_to_return=0;
    m_idx_depthframe_to_return=0;
}

int DataLoaderVolRef::nr_samples(){
    return m_samples_filenames.size();
}


void DataLoaderVolRef::set_shuffle(bool val){
    m_shuffle=val;
}

void DataLoaderVolRef::set_overfit(bool val){
    m_do_overfit=val;
}
