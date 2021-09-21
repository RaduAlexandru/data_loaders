#include "data_loaders/DataLoaderImg.h"

//c++
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <random>


//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//My stuff
#include "Profiler.h"
#include "string_utils.h"

//cv
//#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

//ros
// #include "data_loaders/utils/RosTools.h"


//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;


using namespace radu::utils;
using namespace easy_pbr;


#define BUFFER_SIZE 16

DataLoaderImg::DataLoaderImg(const std::string config_file):
    m_nr_resets(0),
    m_is_running(false)
    {

    init_params(config_file);


    // LOG(FATAL) << "DO NOT USE THIS CLASS! At the moment this class has a really big issue with loading which may lead to even bigger issues down the line. The problem is that the frames are stored in a ringbuffer, however the ringbuffer has not Eigen::aligned_alocator as te DataLoaderImgRos or the DataLoaderCloudRos has. This means that it can fail in very nasty ways down the line. The solution would be to implement my own ringbuffer with an aligned alocator the same way it is down in the other two DataLoaders. However I don't have time for this right now..";

    create_transformation_matrices();
    if(!m_only_rgb){
        if(m_dataset_type==DatasetType::ETH){
            read_pose_file_eth();
        }else if(m_dataset_type==DatasetType::ICL){
            read_pose_file_icl();
        }else if(m_dataset_type==DatasetType::NTS){
            read_pose_file_nts();
        }
    }


    //start reading
    // m_loader_threads.resize(m_nr_cams);
    // for (size_t i = 0; i < m_nr_cams; i++) {
        // m_loader_threads[i]=std::thread(&DataLoaderImg::read_data_for_cam, this, i);
    // }

    if(m_autostart){
        start();
    }



}

DataLoaderImg::~DataLoaderImg(){

    m_is_running=false;
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_loader_threads[i].join();
    }
}

void DataLoaderImg::init_params(const std::string config_file){
    //get the config filename
    // ros::NodeHandle private_nh("~");
    // std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    // std::string config_file="config.cfg";

    //read all the parameters
    // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);

    Config loader_config=cfg["loader_img"];
    m_autostart=loader_config["autostart"];
    VLOG(1) << "Autostart is " << m_autostart;
    m_nr_cams = loader_config["nr_cams"];
    m_imgs_to_skip=loader_config["imgs_to_skip"];
    m_nr_images_to_read=loader_config["nr_images_to_read"];
    m_do_overfit=loader_config["do_overfit"];
    m_shuffle=loader_config["shuffle"];
    m_sort_by_filename=loader_config["sort_by_filename"];


    for (size_t i = 0; i < m_nr_cams; i++) {
        m_rgb_imgs_path_per_cam.push_back( fs::path( (std::string)loader_config["rgb_path_cam_"+std::to_string(i)]) );
        m_frames_buffer_per_cam.push_back( moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE));
    }


    m_only_rgb=loader_config["only_rgb"];
    if(!m_only_rgb){
        std::string dataset_type_string=(std::string)loader_config["dataset_type"];
        if(dataset_type_string=="eth") m_dataset_type=DatasetType::ETH;
        else if(dataset_type_string=="icl") m_dataset_type=DatasetType::ICL;
        else if(dataset_type_string=="nts") m_dataset_type=DatasetType::NTS;
        else LOG(FATAL) << " Dataset type is not known " << dataset_type_string;
        m_pose_file= (std::string)loader_config["pose_file"];
    }



    // Config vis_config=cfg["visualization"];
    // m_tf_worldGL_worldROS_angle=vis_config["tf_worldGL_worldROS_angle"];
    // m_tf_worldGL_worldROS_axis=(std::string)vis_config["tf_worldGL_worldROS_axis"];

    // //input for the images
    m_rgb_subsample_factor=loader_config["rgb_subsample_factor"];

}

void DataLoaderImg::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";

    init_data_reading();

    m_is_running=true;
    m_loader_threads.resize(m_nr_cams);
    for (size_t i = 0; i < m_nr_cams; i++) {
        VLOG(1) <<"starting thread for cam " << i;
        m_loader_threads[i]=std::thread(&DataLoaderImg::read_data_for_cam, this, i);
    }

}

void DataLoaderImg::init_data_reading(){
    std::cout << "init data reading" << '\n';

    m_idx_img_to_read_per_cam.resize(m_nr_cams,0);
    m_rgb_filenames_per_cam.resize(m_nr_cams);
    m_last_frame_per_cam.resize(m_nr_cams);
    m_get_last_published_frame_for_cam.resize(m_nr_cams,false);
    m_undistort_map_x_per_cam.resize(m_nr_cams);
    m_undistort_map_y_per_cam.resize(m_nr_cams);


    for (size_t i = 0; i < m_nr_cams; i++) {
        if(!fs::is_directory(m_rgb_imgs_path_per_cam[i])) {
            LOG(FATAL) << "No directory " << m_rgb_imgs_path_per_cam[i];
        }

        //see how many images we have and read the files paths into a vector
        std::vector<fs::path> rgb_filenames_all;
        for (fs::directory_iterator itr(m_rgb_imgs_path_per_cam[i]); itr!=fs::directory_iterator(); ++itr){
            if(fs::is_regular_file( itr->path()) ){
                rgb_filenames_all.push_back(itr->path());
                // VLOG(1) << "pushed" << itr->path();
            }
        }


        //TODO sort by name so that we process the frames in the correct order
        //sorting assumes that the filename is a numericla value eg 35.png Check that it is so.
        if (m_sort_by_filename && !m_shuffle){ //we don't sort if we arre shuffling afterwards as it makes no difference
            for (size_t i = 0; i < rgb_filenames_all.size(); i++){
                try {
                    std::stod(rgb_filenames_all[i].stem().string() );
                } catch (const std::invalid_argument&) {
                    LOG(FATAL) << "We are assuming that the filename is a numerical value like 45.png. However for this file it is not so for file: " << rgb_filenames_all[i] << " at index: " << i;
                }
            }
            if(m_dataset_type==DatasetType::NTS){
                std::sort(rgb_filenames_all.begin(), rgb_filenames_all.end(), nts_file_comparator());
            }else{
                std::sort(rgb_filenames_all.begin(), rgb_filenames_all.end(), file_timestamp_comparator());
            }
        }



        //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
        for (size_t img_idx = 0; img_idx < rgb_filenames_all.size(); img_idx++) {
            if( (int)img_idx>=m_imgs_to_skip && ((int)m_rgb_filenames_per_cam[i].size()<m_nr_images_to_read || m_nr_images_to_read<0 ) ){
                m_rgb_filenames_per_cam[i].push_back(rgb_filenames_all[img_idx]);
            }
        }

        //shuffle the filles to be read if necessary
        if(m_shuffle){
            unsigned seed = m_nr_resets;
            for (size_t i = 0; i < m_nr_cams; i++){
                auto rng = std::default_random_engine(seed); //create engines with the same states so the vector are randomized in the same way
                std::shuffle(std::begin(m_rgb_filenames_per_cam[i]), std::end(m_rgb_filenames_per_cam[i]), rng);
            }
        }

        std::cout << "Nr rgb images on cam " << i << ": " << rgb_filenames_all.size() << std::endl;
        // std::cout << "Nr rgb images on cam resized  " << i << ": " << m_rgb_filenames_per_cam[i].size() << std::endl;


        // std::cout << "stems are "  << '\n';
        // for (size_t d = 0; d < m_rgb_filenames_per_cam[i].size(); d++) {
        //     std::cout << "path is is " << m_rgb_filenames_per_cam[i][d] << '\n';
        //     std::cout << "stem is " << m_rgb_filenames_per_cam[i][d].stem() << '\n';
        // }

        // std::cout << "stem is " << m_rgb_filenames_per_cam[i][0].stem().string() << '\n';
    }



}



void DataLoaderImg::read_data_for_cam(const int cam_id){
    std::cout << "----------READING DATA for cam " << cam_id << '\n';
    // loguru::set_thread_name(("loader_thread_"+std::to_string(cam_id)).c_str());

    int nr_frames_read_for_cam=0;
    while (m_is_running) {

        //we finished reading so we wait here for a reset
        if(m_idx_img_to_read_per_cam[cam_id]>=(int)m_rgb_filenames_per_cam[cam_id].size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_frames_buffer_per_cam[cam_id].size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue


            Frame frame;
            frame.cam_id=cam_id;
            frame.frame_idx=nr_frames_read_for_cam;
            // VLOG(1) <<"Created frame";

            if(m_dataset_type==DatasetType::NTS){
                //because the very last frame of new tsukuba acually has a bad pose for some reason we set the penultima one to be the last
                if(m_idx_img_to_read_per_cam[cam_id]==(int)m_rgb_filenames_per_cam[cam_id].size()-2){
                    frame.is_last=true; //is the last frame in the dataset
                }
            }else{
                if(m_idx_img_to_read_per_cam[cam_id]==(int)m_rgb_filenames_per_cam[cam_id].size()-1){
                    frame.is_last=true; //is the last frame in the dataset
                }
            }


            fs::path rgb_filename=m_rgb_filenames_per_cam[cam_id][ m_idx_img_to_read_per_cam[cam_id] ];
            if(!m_do_overfit){
                m_idx_img_to_read_per_cam[cam_id]++;
            }
            uint64_t timestamp=-1;
            if(m_dataset_type==DatasetType::NTS){
                std::string filename=rgb_filename.stem().string();
                filename.erase(0,6);
                timestamp=std::stoull(filename);
            }else{
                timestamp=std::stoull(rgb_filename.stem().string());
            }
            frame.timestamp=timestamp; //store the unrounded one because when we check for the labels we check for the same filename

            //POSE---
            if (!m_only_rgb && !get_pose_at_timestamp(frame.tf_cam_world, timestamp, cam_id )){
                LOG(WARNING) << "Not found any pose at timestamp " << timestamp << " Discarding";
                continue;
            }

            //intrinsics
            if(!m_only_rgb){
                get_intrinsics(frame.K, frame.distort_coeffs, cam_id);
                frame.K/=m_rgb_subsample_factor;
                frame.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0
            }


            //Get images, rgb, gradients etc
            // TIME_START("read_imgs");
            frame.rgb_8u=cv::imread(rgb_filename.string());

            // std::cout << "reading " << rgb_filename.string() << '\n';

            // std::cout << "frame rgb is " << type2string(frame.rgb.type()) << '\n';

            // double min, max;
            // cv::minMaxLoc(frame.rgb, &min, &max);
            // std::cout << "min max of frame.rgb is " << min << " " << max << '\n';

            if(m_rgb_subsample_factor>1){
                cv::Mat resized;
                cv::resize(frame.rgb_8u, resized, cv::Size(), 1.0/m_rgb_subsample_factor, 1.0/m_rgb_subsample_factor, cv::INTER_AREA);
                frame.rgb_8u=resized;
            }
            frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
            frame.width=frame.rgb_32f.cols;
            frame.height=frame.rgb_32f.rows;


            // create_alpha_mat(frame.rgb, frame.rgba_8u); //because gpus like rgba otherwise the driver might copy back to system memory the add the 4th channel

            // //convert rgba to float
            // cv::Mat rgba_float;
            // cv::Mat original_rgba;
            // original_rgba=frame.rgba.clone();
            // std::cout << "starting to convert" << '\n';
            // frame.rgba.release();
            // original_rgba.convertTo(frame.rgba, CV_32FC4, 1.0/255.0);
            // std::cout << "finishing convert" << '\n';
            // // frame.rgba=cv::Mat(rgba_float.size(), rgba_float.type());
            // // rgba_float.copyTo(frame.rgba);
            // // frame.rgba=rgba_float.clone();
            // std::cout << "the new rgba is " << type2string(rgba_float.type()) << '\n';
            // std::cout << "copying into the frame rgba yields " << type2string(frame.rgba.type()) << '\n';

            // //gray
            cv::cvtColor ( frame.rgb_32f, frame.gray_32f, cv::COLOR_BGR2GRAY);
            // frame.gray.convertTo(frame.gray, CV_32F, 1.0/255.0);
            // if(!m_only_rgb || !frame.distort_coeffs.isZero() ){
            //     frame.gray=undistort_image(frame.gray, frame.K, frame.distort_coeffs, cam_id); //undistort only the gray image because rgb is only used for visualization
            //     //TODO remove this as we only use the rgb for visualization and debug
            //     frame.rgb=undistort_image(frame.rgb, frame.K, frame.distort_coeffs, cam_id);
            // }

            // frame.gray/=255.0;

            // cv::minMaxLoc(frame.gray, &min, &max);
            // std::cout << "min max of frame.gray is " << min << " " << max << '\n';



            // //gradients
            // cv::Scharr( frame.gray, frame.grad_x, CV_32F, 1, 0);
            // cv::Scharr( frame.gray, frame.grad_y, CV_32F, 0, 1);
            // frame.grad_x = cv::abs(frame.grad_x);
            // frame.grad_y = cv::abs(frame.grad_y);

            // merge the gray image and the gradients into one 3 channel image
            // std::vector<cv::Mat> channels;
            // channels.push_back(frame.gray);
            // channels.push_back(frame.grad_x);
            // channels.push_back(frame.grad_y);
            // // channels.push_back(frame.grad_y);
            // cv::merge(channels, frame.gray_with_gradients);
            // frame.gray_with_gradients = cv::abs(frame.gray_with_gradients);
            // TIME_END("read_imgs");


            // std::cout << "pusing frame with tf corld of " << frame.tf_cam_world.matrix() << '\n';
            // std::cout << "pusing frame with K of " << frame.K << '\n';
            // std::cout << "pushing frame with frame_idxs " << frame.frame_idx << '\n';

            //just to see if we can get a callback from the DataLoaderRos
            // publish_stereo_frame(frame);

            m_frames_buffer_per_cam[cam_id].enqueue(frame);
            nr_frames_read_for_cam++;

        }
    }
    VLOG(1) << "Finished reading all the images";
}

bool DataLoaderImg::is_finished(){
    //check if this loader has loaded everything for every camera
    for (size_t cam_id = 0; cam_id < m_nr_cams; cam_id++) {
        if(m_idx_img_to_read_per_cam[cam_id]<(int)m_rgb_filenames_per_cam[cam_id].size()){
            // VLOG(1) << "there is still more files to read for cam " << cam_id << " " << m_idx_img_to_read_per_cam[cam_id] << " out of " <<  m_rgb_filenames_per_cam[cam_id].size() ;
            return false; //there is still more files to read
        }
    }

    //check that there is nothing in the ring buffers
    for (size_t cam_id = 0; cam_id < m_nr_cams; cam_id++) {
        if(m_frames_buffer_per_cam[cam_id].peek()!=nullptr){
            // VLOG(1) << "There is still smething in the buffer";
            return false; //there is still something in the buffer
        }
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderImg::is_finished_reading(){
    //check if this loader has loaded everything for every camera
    for (size_t cam_id = 0; cam_id < m_nr_cams; cam_id++) {
        if(m_idx_img_to_read_per_cam[cam_id]<(int)m_rgb_filenames_per_cam[cam_id].size()){
            return false; //there is still more files to read
        }
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}

bool DataLoaderImg::has_data_for_cam(const int cam_id){
    // return !m_queue.empty();
    if(m_frames_buffer_per_cam[cam_id].peek()==nullptr){
        return false;
    }else{
        return true;
    }
}

bool DataLoaderImg::has_data_for_all_cams(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        if(!has_data_for_cam(i)){
            return false;
        }
    }
    return true;
}

Frame DataLoaderImg::get_frame_for_cam(const int cam_id){
    // TIME_SCOPE("get_next_frame");

    if(m_get_last_published_frame_for_cam[cam_id] && m_last_frame_per_cam[cam_id].rgb_8u.data){
        m_get_last_published_frame_for_cam[cam_id]=false;

        //this frame has to be recorrected with the spline
        Frame frame=m_last_frame_per_cam[cam_id];

        // VLOG(1) << "after pose is : \n" << frame.tf_cam_world.matrix() << std::endl;
        //if the pose was not valid then we will get a nan. TODO make it a bit nicer, maybe return a bool for success or check why id fails in the first place
        return frame;
    }

    Frame frame ;
    m_frames_buffer_per_cam[cam_id].try_dequeue(frame);

    //store also the last frame in case we need to republish it
    m_last_frame_per_cam[cam_id]=frame;

    return frame;

}

int DataLoaderImg::nr_samples_for_cam(const int cam_id){
    return m_rgb_filenames_per_cam[cam_id].size();
}

void DataLoaderImg::reset(){


    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        for (size_t i = 0; i < m_nr_cams; i++){
            auto rng = std::default_random_engine(seed); //create engines with the same states so the vector are randomized in the same way
            std::shuffle(std::begin(m_rgb_filenames_per_cam[i]), std::end(m_rgb_filenames_per_cam[i]), rng);
        }
    }

    //restarts the indexes for reading
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_idx_img_to_read_per_cam[i]=0;
        //deque everything (we can do it safely from here because while this is running, the core is not reading since the Core and GUI share thread)
        m_frames_buffer_per_cam[i]=moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE);
    }

}

void DataLoaderImg::clear_buffers(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        //deque everything (we can do it safely from here because while this is running, the core is not reading since the Core and GUI share thread)
        m_frames_buffer_per_cam[i]=moodycamel::ReaderWriterQueue<Frame>(BUFFER_SIZE);
    }
}





void DataLoaderImg::read_pose_file_eth(){
    std::ifstream infile( m_pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
    VLOG(1) << "Reading pose file for ETH mav dataset";

    // uint64_t scan_nr;
    uint64_t timestamp;
    Eigen::Vector3f position;
    Eigen::Quaternionf quat;

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        //skip comments
        if(line.at(0)=='#'){
            continue;
        }

        std::vector<std::string> tokens=split(line," ");
        timestamp=stod(tokens[0]);
        position(0)=stod(tokens[1]);
        position(1)=stod(tokens[2]);
        position(2)=stod(tokens[3]);
        quat.w()=stod(tokens[4]);
        quat.x()=stod(tokens[5]);
        quat.y()=stod(tokens[6]);
        quat.z()=stod(tokens[7]);

        quat.normalize();
        // quat=quat.conjugate();

        // std::cout << "input is \n" << " " << timestamp << " " << position << " " << quat.matrix()  << "\n";
        Eigen::Affine3f pose;
        pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
        pose.matrix().block<3,1>(0,3)=position;

        m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3f>(timestamp,pose) );
    }

}

void DataLoaderImg::read_pose_file_icl(){
    std::ifstream infile( m_pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
    VLOG(1) << "Reading pose file for ICL-NUIM dataset";

    // uint64_t scan_nr;
    uint64_t timestamp;
    Eigen::Vector3f position;
    Eigen::Quaternionf quat;

    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        //skip comments
        if(line.at(0)=='#'){
            continue;
        }

        iss >> timestamp
            >> position(0) >> position(1) >> position(2)
            >> quat.x() >> quat.y() >> quat.z() >> quat.w();

        // timestamp=timestamp-1; //timestamp starts at 1 and filenames starts at 0. We substract 1 from the timestamp to make it match

        // std::cout << "input is \n" << " " << timestamp << " " << position << " " << quat.matrix()  << "\n";
        Eigen::Affine3f pose;
        pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
        pose.matrix().block<3,1>(0,3)=position;



        m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3f>(timestamp,pose) );
    }

}

void DataLoaderImg::read_pose_file_nts(){

    std::ifstream infile( m_pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
    VLOG(1) << "Reading pose file for NTS dataset";

    std::string line;
    uint64_t timestamp=0;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        double tx, ty, tz, ex, ey, ez;

        std::vector<std::string> tokens=split(line,",");
        tx=stod(tokens[0]);
        ty=stod(tokens[1]);
        tz=stod(tokens[2]);
        ex=stod(tokens[3]);
        ey=stod(tokens[4]);
        ez=stod(tokens[5]);


        // rad 2 deg
        ex = ex * M_PI / 180.;
        ey = (180-ey) * M_PI / 180.;
        ez = (180-ez) * M_PI / 180.;

        Eigen::Affine3f pose_wb = Eigen::Affine3f::Identity();
        pose_wb.translation() << tx,ty,tz;

        pose_wb.translation() = pose_wb.translation() / 100.; // cm to m
        pose_wb.linear () = (Eigen::AngleAxisf(ez, Eigen::Vector3f::UnitZ())
                * Eigen::AngleAxisf(ey, Eigen::Vector3f::UnitY())
                * Eigen::AngleAxisf(ex, Eigen::Vector3f::UnitX())).toRotationMatrix();

        m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3f>(timestamp,pose_wb) );

        timestamp++;
    }

}

bool DataLoaderImg::get_pose_at_timestamp(Eigen::Affine3f& pose, const uint64_t timestamp, const uint64_t cam_id){


    //return the closest one
    uint64_t closest_idx=-1;
    double smallest_timestamp_diff=std::numeric_limits<double>::max();
    for (size_t i = 0; i < m_worldROS_baselink_vec.size(); i++) {
        uint64_t recorded_timestamp=m_worldROS_baselink_vec[i].first;
        // Eigen::Affine3f pose=m_worldROS_baselink_vec[i].second;
        // std::cout << "comparing recorded_timestamp to timestmp \n" << recorded_timestamp << "\n" << timestamp << '\n';
        double diff=fabs((double)timestamp- (double)recorded_timestamp);
        if (  diff < smallest_timestamp_diff){
            closest_idx=i;
            smallest_timestamp_diff=diff;
            // std::cout << "smallest_timestamp_diff " << smallest_timestamp_diff << '\n';
        }
    }
    // if ( smallest_timestamp_diff > 1e7 )
    // {
    //     LOG(WARNING) << "time difference for pose is way too large! " << (smallest_timestamp_diff/1e6) << "s." << '\n';
    //     return false;
    // }

    if ( smallest_timestamp_diff!=0 ){
        LOG(WARNING) << "time difference for pose is way too large! " << smallest_timestamp_diff << '\n';
        return false;
    }

    // std::cout << "smallest_timestamp_diff is " << smallest_timestamp_diff << '\n';
    // std::cout << "smallest_timestamp_diff_no_abs is " << smallest_timestamp_diff_no_abs << '\n';
    // std::cout << "deviation_ms is " << deviation_ms << '\n';
    Eigen::Affine3f pose_from_file=m_worldROS_baselink_vec[closest_idx].second;


    //this pose may be already the correct one or it may be transfromed ot another frame depending on the dataset type
    if(m_dataset_type==DatasetType::ETH){
        // pose_from_file is only the transformation from base to world
        if(cam_id==0){
            // camera to base link
            Eigen::Matrix4f tf_baselink_cam;
            tf_baselink_cam.setIdentity();
            // tf_baselink_cam.row(0) << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975;
            // tf_baselink_cam.row(1) << 0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768;
            // tf_baselink_cam.row(2) << -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949;
            // tf_baselink_cam.row(3) <<  0.0, 0.0, 0.0, 1.0;

            //pose is only from base to world but we need to return a pose that is tf_cam_world (so from world to cam)
            pose= Eigen::Affine3f(tf_baselink_cam).inverse() *  pose_from_file.inverse(); //world to base and base to cam
        }else if(cam_id==1){
            // camera to base link

            Eigen::Matrix4f tf_baselink_cam_left;
            // tf_baselink_cam.setIdentity();
            tf_baselink_cam_left.row(0) << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975;
            tf_baselink_cam_left.row(1) << 0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768;
            tf_baselink_cam_left.row(2) << -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949;
            tf_baselink_cam_left.row(3) <<  0.0, 0.0, 0.0, 1.0;

            Eigen::Matrix4f tf_baselink_cam_right;
            tf_baselink_cam_right.row(0) << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556;
            tf_baselink_cam_right.row(1) << 0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024;
            tf_baselink_cam_right.row(2) << -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038;
            tf_baselink_cam_right.row(3) << 0.0, 0.0, 0.0, 1.0;

            Eigen::Matrix4f tf_baselink_cam;
            tf_baselink_cam=tf_baselink_cam_left.inverse()*tf_baselink_cam_right;


            //pose is only from base to world but we need to return a pose that is tf_cam_world (so from world to cam)
            pose= Eigen::Affine3f(tf_baselink_cam).inverse() *  pose_from_file.inverse(); //world to base and base to cam
        }else{
            LOG(FATAL) << "Now a known cam_id at " << cam_id;
        }

    }else if(m_dataset_type==DatasetType::ICL){
        pose=pose_from_file.inverse();
    }else if(m_dataset_type==DatasetType::NTS){
        //pose from file is only from baselink to world

        Eigen::Affine3f leftPose_cb = Eigen::Affine3f::Identity();
        Eigen::Affine3f rightPose_cb = Eigen::Affine3f::Identity();
        leftPose_cb.translation()(0)+=0.05; // 5 cm, since stereo origin is in the middle.
        rightPose_cb.translation()(0)-=0.05; // 5 cm, since stereo origin is in the middle.

        if(cam_id==0){
            //pose is only from base to world but we need to return a pose that is tf_cam_world (so from world to cam)
            pose= leftPose_cb *  pose_from_file.inverse(); //world to base and base to cam
        }else if(cam_id==1){
            pose= rightPose_cb *  pose_from_file.inverse(); //world to base and base to cam
        }else{
            LOG(FATAL) << "Now a known cam_id at " << cam_id;
        }



    }else{
        LOG(FATAL) << "Unknown dataset";
    }

    // std::cout << "closest idx is " << closest_idx << '\n';
    // std::cout << " timestamp is " << timestamp << " closest timestamp is " << m_worldROS_baselink_vec[closest_idx].first << '\n';
    // std::cout << "returning cam pose \n" << pose.matrix()  << '\n';


    return true;


}

void DataLoaderImg::get_intrinsics(Eigen::Matrix3f& K, Eigen::Matrix<float, 5, 1>& distort_coeffs, const uint64_t cam_id){
    K.setIdentity();

    if(m_dataset_type==DatasetType::ETH){
        K.setIdentity();
        if(cam_id==0){
            K(0,0) = 458.654;
            K(1,1) = 457.296;
            K(0,2) = 367.215;
            K(1,2) = 248.375;
            distort_coeffs(0) = -0.28340811;
            distort_coeffs(1) = 0.07395907;
            distort_coeffs(2) = 0.00019359;
            distort_coeffs(3) = 1.76187114e-05;
            distort_coeffs(4) = 0.;
        }else if(cam_id==1){
            K(0,0) = 457.587;
            K(1,1) = 456.134;
            K(0,2) = 379.999;
            K(1,2) = 255.238;
            distort_coeffs(0) = -0.28368365;
            distort_coeffs(1) = 0.07451284;
            distort_coeffs(2) = -0.00010473;
            distort_coeffs(3) = -3.55590700e-05;
            distort_coeffs(4) = 0.;
        }
    }else if(m_dataset_type==DatasetType::ICL){
        K.setIdentity();
        if(cam_id==0){
            K(0,0)=481.2; //fx
            K(1,1)=-480; //fy
            K(0,2)=319.5; // cx
            K(1,2)=239.5; //cy
            K(2,2)=1.0;
            distort_coeffs.setZero();
        }else if(cam_id==1){
            //even though we have one cam we set this one too because it's easier to deal with it like this for now.
            K(0,0)=481.2; //fx
            K(1,1)=-480; //fy
            K(0,2)=319.5; // cx
            K(1,2)=239.5; //cy
            K(2,2)=1.0;
            distort_coeffs.setZero();
        }
    }else if(m_dataset_type==DatasetType::NTS){
        K.setIdentity();
        if(cam_id==0){
            K(0,0) = 615; //fx
            K(1,1) = 615; //fy
            K(0,2) = 320; //cx
            K(1,2) = 240; //cy
            K(2,2)=1.0;
            distort_coeffs.setZero();
        }else if(cam_id==1){
            //even the two cameras are the same K
            K(0,0) = 615; //fx
            K(1,1) = 615; //fy
            K(0,2) = 320; //cx
            K(1,2) = 240; //cy
            K(2,2)=1.0;
            distort_coeffs.setZero();
        }
    }else{
        LOG(FATAL) << "Unknown dataset";
    }
}



void DataLoaderImg::create_transformation_matrices(){



    /*
     *
     *
     *           Z
     *           |
     *           |       X
     *           |     /
     *           |   /
     *           | /
     *   Y-------
     *

     * ROS world frame
     * Explained here: http://www.ros.org/reps/rep-0103.html
     *
     * */


    // m_tf_worldGL_worldROS.setIdentity();
    // Eigen::Matrix3f worldGL_worldROS_rot;
    // Eigen::Vector3f axis;
    // axis.setZero();
    // if(m_tf_worldGL_worldROS_axis=="x"){
    //     axis=Eigen::Vector3f::UnitX();
    // }else if(m_tf_worldGL_worldROS_axis=="y"){
    //     axis=Eigen::Vector3f::UnitY();
    // }else if(m_tf_worldGL_worldROS_axis=="z"){
    //     axis=Eigen::Vector3f::UnitZ();
    // }else{
    //     LOG(FATAL) << "No valid m_tf_worldGL_worldROS_axis. Need to be either x,y or z";
    // }
    // worldGL_worldROS_rot = Eigen::AngleAxisf(m_tf_worldGL_worldROS_angle, axis);
    // m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;

    m_tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3d worldGL_worldROS_rot;
    worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;



}

void DataLoaderImg::republish_last_frame_from_cam(const int cam_id){
    m_get_last_published_frame_for_cam[cam_id]=true;
    // std::cout << "republish check" << '\n';
    // if(m_last_frame_per_cam[cam_id].rgb.data){
    //     std::cout << "republish " << '\n';
    //     m_frames_buffer_per_cam[cam_id].enqueue(m_last_frame_per_cam[cam_id]);
    // }

    //set it so that the next

}

void DataLoaderImg::republish_last_frame_all_cams(){
    for (size_t i = 0; i < m_nr_cams; i++) {
        m_get_last_published_frame_for_cam[i]=true;
    }
}
