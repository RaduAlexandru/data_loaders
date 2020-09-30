#include "data_loaders/DataLoaderImgRos.h"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//ros
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include "ros_utils.h"
#include "opencv_utils.h"
#include <image_transport/image_transport.h>
#include <tf2_eigen/tf2_eigen.h>

//my stuff 


using namespace radu::utils;
using namespace easy_pbr;

#define BUFFER_SIZE 3

DataLoaderImgRos::DataLoaderImgRos(const std::string config_file):
    m_is_thread_running(false),
    m_nr_callbacks(0)
{
    init_params(config_file);
    m_loader_thread=std::thread(&DataLoaderImgRos::init_ros, this);  //starts the spin in another thread
    m_is_thread_running=true;

}

DataLoaderImgRos::~DataLoaderImgRos(){
    std::cout << "DESTRUCTOR" <<std::endl;

    ros::shutdown();
    std::cout << "trying to kill the loader thread" << std::endl;
    m_loader_thread.join();
}

void DataLoaderImgRos::init_params(const std::string config_file){
 
    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_img_ros"];

    int nr_cams = loader_config["nr_cams"];
    // m_do_pose = loader_config["do_pose"];
    m_pose_source=(std::string)loader_config["pose_source"];
    m_tf_reference_frame=(std::string)loader_config["tf_reference_frame"];
    m_cam_info_source=(std::string)loader_config["cam_info_source"];

    //create the cams 
    m_cams.resize(nr_cams);
    for(int i = 0; i < nr_cams; i++){
        m_cams[i].m_img_topic = (std::string)loader_config["img_topic_"+std::to_string(i)];
        m_cams[i].m_cam_info_topic = (std::string)loader_config["cam_info_topic_"+std::to_string(i)];
        m_cams[i].m_img_subsample_factor = loader_config["img_subsample_factor_"+std::to_string(i)]; 
        m_cams[i].m_is_compressed = loader_config["is_compressed_"+std::to_string(i)];
        // m_cams[i].m_intrinsics_string = (std::string)loader_config["intrinsics_"+std::to_string(i)];
        // m_cams[i].m_timestamp_offset = (u_int64_t)loader_config["timestamp_offset_"+std::to_string(i)];
    }
    // m_pose_file = (std::string)loader_config["pose_file"];
}

void DataLoaderImgRos::init_ros(){
    std::vector<std::pair<std::string, std::string> > dummy_remappings;
    ros::init(dummy_remappings, "loader_img_ros");

    ros::NodeHandle private_nh("~");

    image_transport::ImageTransport img_transport( private_nh );
    std::vector<image_transport::Subscriber> img_subs;
    img_subs.resize(m_cams.size());
    std::vector<ros::Subscriber> cam_info_subs;
    cam_info_subs.resize(m_cams.size());

    //make the subscribers
    for(size_t i = 0; i < m_cams.size(); i++){
        if(m_cams[i].m_is_compressed){
            img_subs[i] = img_transport.subscribe(m_cams[i].m_img_topic, 5, boost::bind(&DataLoaderImgRos::callback_img, this, _1, i), ros::VoidPtr(), image_transport::TransportHints("compressed"));
        }else{
            img_subs[i] = img_transport.subscribe(m_cams[i].m_img_topic, 5,  boost::bind(&DataLoaderImgRos::callback_img, this, _1, i) );
        }

        //cam info only if the source contains the word "topic"
        bool source_is_from_topic = m_cam_info_source.find("topic")!=std::string::npos;
        if(source_is_from_topic){
            cam_info_subs[i]=private_nh.subscribe<sensor_msgs::CameraInfo>(m_cams[i].m_cam_info_topic, 5, boost::bind(&DataLoaderImgRos::callback_cam_info, this, _1, i) );
        }
        // VLOG(1) << "source is from topic " << source_is_from_topic;
    }

    //start the tf listener here because it has to be started only after ros_init
    m_tf_buf=std::make_shared<tf2_ros::Buffer>();
    m_tf_listener=std::make_shared<tf2_ros::TransformListener>(*m_tf_buf);

    //multithreaded spinning, each callback (from different cameras) will run on different threads in paralel
    // ros::MultiThreadedSpinner spinner(m_cams.size()); // Use as many threads as cameras
    // spinner.spin();

    //so that we can have both the ImgRos and the cloud rus running we need a asyncspinner
    ros::AsyncSpinner spinner(m_cams.size()); // Use as many threads as cameras
    spinner.start();
    ros::waitForShutdown();

    LOG(INFO) << "finished ros communication";
    std::cout << "inside thread" << "\n";
    m_is_thread_running=false;
}

void DataLoaderImgRos::callback_cam_info(const sensor_msgs::CameraInfoConstPtr& msg, const int cam_id){
    // VLOG(1) << "callback cma info";
    m_cams[cam_id].m_cam_info=msg; 
}

void DataLoaderImgRos::callback_img(const sensor_msgs::ImageConstPtr& img_msg, const int cam_id) {

    // std::cout << "loll callback from " << cam_id <<std::endl;
    // VLOG(1)<< "callback from cam " << cam_id;
    Frame frame;
    frame.cam_id=cam_id;
    frame.frame_idx=m_nr_callbacks;

    Cam& cam=m_cams[cam_id];

    //Get msg as a opencv
    cv::Mat cv_img;
    cv_bridge::CvImageConstPtr cv_ptr;
    try{
        cv_ptr = cv_bridge::toCvShare( img_msg );
        cv_ptr->image.copyTo(cv_img);
        //cv::flip(img_cv,img_cv, -1); //TODO this line needs to be commented
    }catch (cv_bridge::Exception& e){
        LOG(FATAL) << "cv_bridge exception " << e.what();
    }

    frame.img_original_size=cv_img;

    //resize if the downsample factor is anything ther than 1
    if (cam.m_img_subsample_factor!=1){
        cv::Mat resized;
        cv::resize(cv_img, resized, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor, cv::INTER_LANCZOS4 );
        cv_img=resized;
    }


    //get intrinsics 
    if (m_cam_info_source=="none"){
        frame.K.setZero();
        frame.distort_coeffs.setZero();
    }else if (m_cam_info_source=="topic"){
        //check if we already got a callback from the camera_info topic
        if (!cam.m_cam_info){
            LOG(WARNING) << "No camera info yet"; 
            return;
        }
        //The K is stored in the K vector
        frame.K.setIdentity();
        frame.K(0,0) = cam.m_cam_info->K[0]; //fx
        frame.K(1,1) = cam.m_cam_info->K[4]; //fy
        frame.K(0,2) = cam.m_cam_info->K[2]; //cx
        frame.K(1,2) = cam.m_cam_info->K[5]; //cy
        for(int i = 0; i < 5; ++i){
            frame.distort_coeffs[i] = cam.m_cam_info->D[i];
        }
        // // Divide fx, fy, cx, cy by image size to obtain coordinates in [0..1]
        // distortion[2] /= camInfo->width;
        // distortion[4] /= camInfo->width;
        // distortion[3] /= camInfo->height;
        // distortion[5] /= camInfo->height;

    }else if (m_cam_info_source=="topic_with_double_sphere"){
        //check if we already got a callback from the camera_info topic
        if (!cam.m_cam_info){
            LOG(WARNING) << "No camera info yet"; 
            return;
        }
        //The K is kinda hackly stored inside the D vector of the cam info in this case
        frame.K.setIdentity();
        frame.K(0,0) = cam.m_cam_info->D[2]; //fx
        frame.K(1,1) = cam.m_cam_info->D[3]; //fy
        frame.K(0,2) = cam.m_cam_info->D[4]; //cx
        frame.K(1,2) = cam.m_cam_info->D[5]; //cy
        for(int i = 0; i < 5; ++i){
            frame.distort_coeffs[i] = cam.m_cam_info->D[i];
            // frame.distort_coeffs[i] = cam.m_cam_info->D[i]/cam.m_img_subsample_factor;
        }

    
    }else{
        LOG(FATAL) << "m_cam_info_source is not known";
    }

    //adjust the K matrix to the size of the img 
    frame.K/=cam.m_img_subsample_factor;
    frame.K(2,2)=1.0;


    //get pose
    // tf2_ros::Buffer m_tf_buf;
    // tf2_ros::TransformListener m_tf_listener{m_tf_buf};
    if (m_pose_source=="none"){
        frame.tf_cam_world.setIdentity();

    }else if (m_pose_source=="tf"){
        geometry_msgs::TransformStamped transform;
        try{
            // transform = m_tf_buf.lookupTransform(img_msg->header.frame_id, m_tf_reference_frame, img_msg->header.stamp);
            transform = m_tf_buf->lookupTransform(img_msg->header.frame_id, m_tf_reference_frame, img_msg->header.stamp);
        }catch(tf2::TransformException& e){
            LOG(ERROR) << "Could not obtain camera transform: " << e.what();
            return;
        }
        Eigen::Affine3d tf_cam_world = tf2::transformToEigen(transform);
        frame.tf_cam_world=tf_cam_world.cast<float>();

    }else{
        LOG(FATAL) << "pose_source is not known";
    } 

    //check that the transform form depth to color coincides with /camera/extrinsics/depth_to_color
    // VLOG(1) << "frame " << img_msg->header.frame_id;
    // geometry_msgs::TransformStamped transform;
    // transform = m_tf_buf->lookupTransform("camera_color_optical_frame", "camera_depth_optical_frame", img_msg->header.stamp); //transform from depth to color
    // Eigen::Affine3d tf_color_depth = tf2::transformToEigen(transform); 
    // VLOG(1) << "tf_color_depth" << tf_color_depth.translation();


    
    // std::cout << "K is " << frame.K << std::endl;

    // VLOG(1) << " cv img type  " << type2string(cv_img.type());

    //img 
    if(cv_img.depth()==CV_8U){
        if(cv_img.channels()==3){
            frame.rgb_8u=cv_img;
        }else if(cv_img.channels()==1){
            frame.gray_8u=cv_img;
        }
    }else if(cv_img.depth()==CV_16U){
        if(cv_img.channels()==1){
            // cv_img.convertTo(frame.depth, CV_32FC1);
            cv_img.convertTo(frame.depth, CV_32FC1, 1.0/1000.0);
        }

    }

    frame.width=cv_img.cols;
    frame.height=cv_img.rows;
 

    // VLOG(1) << "cv_img has size " <<cv_img.rows << " " << cv_img.cols;


    //set K
    // std::vector<std::string> intrinsics_split=split(cam.m_intrinsics_string," ");
    // frame.K.setIdentity();
    // frame.K(0,0)=std::stof(intrinsics_split[0]); //fx
    // frame.K(1,1)=std::stof(intrinsics_split[1]); //fy
    // frame.K(0,2)=std::stof(intrinsics_split[2]); // cx
    // frame.K(1,2)=std::stof(intrinsics_split[3]); //cy
    // frame.K*=1.0/cam.m_img_subsample_factor;



    //subsample (cannot subsample the cv_img directly because it may be a bayered image)
    // cv::resize(cv_img, cv_img, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor);
    // std::cout << " got an image of size" << cv_img.rows << " " << cv_img.cols << " type is " << type2string(cv_img.type()) << std::endl;

  
    // if(cv_img.depth()==CV_16U){
    //     std::cout << "got a thermal image" << std::endl;
    //     //it is most likely a thermal image
    //     frame.thermal_16u=cv_img;
    //     cv::resize(frame.thermal_16u, frame.thermal_16u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor);
    //     frame.thermal_16u.convertTo(frame.thermal_32f, CV_32FC1, 1.0/65536.0);
    //     //for showing we put it also in 3 channels
    //     std::vector<cv::Mat> matChannels;
    //     matChannels.push_back(frame.thermal_32f);
    //     matChannels.push_back(frame.thermal_32f);
    //     matChannels.push_back(frame.thermal_32f);
    //     // cv::Mat_<cv::Vec<float,3> > out;
    //     cv::merge(matChannels, frame.thermal_vis_32f);

    //     // Gui::show(frame.thermal_32f, "thermal_32f");
    //     frame.width=frame.thermal_32f.cols;
    //     frame.height=frame.thermal_32f.rows;
    // }else if(cv_img.depth()==CV_8U){
    //     //just an rgb image
    //     //convert to float
    //     if(cv_img.channels()==1){
    //         // frame.rgb_8u.convertTo(frame.gray_32f, CV_32FC1, 1.0/255.0);
    //         cv::cvtColor ( cv_img, frame.rgb_8u, CV_BayerBG2RGB );  // only works on 8u and 16u
    //         cv::cvtColor ( frame.rgb_8u, frame.rgb_8u, CV_BGR2RGB );  
    //         cv::resize(frame.rgb_8u, frame.rgb_8u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor); // can only do it here because cv image is a bayered image
    //         frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
    //         // std::cout << "After debayer the type is " << type2string(frame.rgb_32f.type()) << std::endl;
    //         cv::cvtColor ( frame.rgb_32f, frame.gray_32f, CV_BGR2GRAY );      
    //         //gradients
    //         cv::Scharr( frame.gray_32f, frame.grad_x_32f, CV_32F, 1, 0);
    //         cv::Scharr( frame.gray_32f, frame.grad_y_32f, CV_32F, 0, 1);
    //     }else if(cv_img.channels()==2){
    //         frame.rgb_8u=cv_img;
    //         cv::resize(frame.rgb_8u, frame.rgb_8u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor); 
    //         frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC2, 1.0/255.0);
    //     }else if(cv_img.channels()==3){
    //         frame.rgb_8u=cv_img;
    //         cv::resize(frame.rgb_8u, frame.rgb_8u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor); 
    //         frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
    //         cv::cvtColor ( frame.rgb_32f, frame.gray_32f, CV_BGR2GRAY );      
    //         //gradients
    //         cv::Scharr( frame.gray_32f, frame.grad_x_32f, CV_32F, 1, 0);
    //         cv::Scharr( frame.gray_32f, frame.grad_y_32f, CV_32F, 0, 1);
    //     }else if(cv_img.channels()==4){
    //         frame.rgb_8u=cv_img;
    //         cv::resize(frame.rgb_8u, frame.rgb_8u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor); 
    //         frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC4, 1.0/255.0);
    //     }else{
    //         LOG(FATAL) << "Not a known number of channels";
    //     }

    //     frame.width=frame.rgb_32f.cols;
    //     frame.height=frame.rgb_32f.rows;
    // }else{
    //     LOG(WARNING) << "Unknown image type";
    // } 


    // Gui::show(frame.rgb_32f, "data_loader_img"); // DO NOT CALL thus
    // m_last_retreived_idx=m_working_idx;
    
    // frame.rgb=cv_img;



    //put float images inside
    // cv::Mat cv_img_float;
    // cv_img.convertTo(cv_img_float, CV_32FC3, 1.0/255.0);
    // frame.rgb=cv_img_float;
    // frame.rgba_32f=create_alpha_mat<float>(cv_img_float);
    // VLOG(1) << "frame.rgb has type" << type2string(frame.rgb.type());
    // VLOG(1) << "frame.rgba_32f has type" << type2string(frame.rgba_32f.type());



   
    // VLOG(1) << "working_idx is " << m_working_idx << " finsihed is " << m_finished_idx ;
    // VLOG(1) << "m_last_retrieved is " << m_last_retreived_idx ;



    // cam.m_push_pull_mutex->lock();  
    // bool about_to_overwrite= (cam.m_working_idx==cam.m_last_retrieved_idx);
    // cam.m_push_pull_mutex->unlock();  
    // if(about_to_overwrite){
    //     //we are about to overwrite on top of the finished one which might be at the moment used by the gpu
    //     LOG(WARNING) << "About to overwrite, maybe your consumer thread is too slow or maybe this queue in the dataloader is too small. Discarding";
    //     cam.m_is_modified = true;
    //     return; 
    // }
    // cam.m_frames_buffer[cam.m_working_idx]=frame;
    // cam.m_finished_idx = cam.m_working_idx;
    // cam.m_working_idx = (cam.m_working_idx + 1) % NUM_FRAMES_BUFFER;
    // cam.m_is_modified = true;

    cam.m_frames_buffer.try_enqueue(frame);

    m_nr_callbacks++; 

}

bool DataLoaderImgRos::is_loader_thread_alive(){

    return m_is_thread_running;
}

bool DataLoaderImgRos::has_data_for_all_cams(){
    for(int i=0; i<nr_cams(); i++){
        if (!has_data_for_cam(i)){
            return false;
        }
    }

    return true;
}

bool DataLoaderImgRos::has_data_for_cam(const int cam_id){
    if( m_cams[cam_id].m_frames_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    } 
}

Frame DataLoaderImgRos::get_frame_for_cam(const int cam_id){
    Cam& cam=m_cams[cam_id];

    // std::lock_guard<std::mutex> lock(*cam.m_push_pull_mutex);  
    // cam.m_is_modified=false;
    // cam.m_last_retrieved_idx=cam.m_finished_idx;
    // // VLOG(1) << "RETRIEVING m_last_retrieved is " << m_last_retreived_idx ;
    // return cam.m_frames_buffer[cam.m_finished_idx];

    Frame frame;
    cam.m_frames_buffer.try_dequeue(frame);

    return frame;
}


int DataLoaderImgRos::nr_cams(){
    return m_cams.size();
}

// void DataLoaderImgRos::read_pose_file(){
//     std::ifstream infile( m_pose_file );
//     if(!infile.is_open()){
//         LOG(FATAL) << "Could not open pose file " << m_pose_file;
//     }
//     uint64_t timestamp;
//     Eigen::Vector3d position;
//     Eigen::Quaterniond quat;


//     std::string line;
// //     while (std::getline(infile, line)) {
// //         std::istringstream iss(line);
// //         iss >> scan_nr >> timestamp
// //             >> position(0) >> position(1) >> position(2)
// //             >> quat.w() >> quat.x() >> quat.y() >> quat.z();
// //     //    std::cout << "input is \n" << scan_nr << " " << timestamp << "  \n Position is" << position << " " << quat.matrix()  << "\n";
// //         Eigen::Affine3d pose;
// //         pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
// //         pose.matrix().block<3,1>(0,3)=position;
// //         // if(m_do_timestamp_rounding_when_reading_file){
// //         //     timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
// //         // }
// //        // VLOG(2) << "recorded tmestamp is " << timestamp;
// // //        VLOG(2) << "recorded scan_nr is " << scan_nr;
// //         m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3d>(timestamp,pose) );
// //     }



//     while (std::getline(infile, line)) {
//         std::istringstream iss(line);
//         iss >> timestamp
//             >> position.x() >> position.y() >> position.z()
//             >> quat.x() >> quat.y() >> quat.z() >> quat.w();
//     //    std::cout << "input is \n" << " " << timestamp << "  \n Position is" << position << " " << quat.matrix()  << "\n";
//         Eigen::Affine3d pose;
//         pose.setIdentity();
//         pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
//         pose.matrix().block<3,1>(0,3)=position;
//         // if(m_do_timestamp_rounding_when_reading_file){
//         //     timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
//         // }
//        // VLOG(2) << "recorded tmestamp is " << timestamp;
// //        VLOG(2) << "recorded scan_nr is " << scan_nr;
//         // Eigen::Affine3d pose_inv=pose.inverse();
        
//         m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3d>(timestamp,pose) );
//     }



//     std::sort ( m_worldROS_baselink_vec.begin(), m_worldROS_baselink_vec.end(), [](const std::pair<uint64_t,Eigen::Affine3d> & a, const std::pair<uint64_t,Eigen::Affine3d> & b ){return a.first < b.first;});


//     // auto last_valid=std::unique ( m_worldROS_baselink_vec.begin(), m_worldROS_baselink_vec.end(), [](const std::pair<double,Eigen::Affine3d> & a, const std::pair<double,Eigen::Affine3d> & b ){return a.first == b.first;});
//     // // m_worldROS_baselink_vec.erase(last_valid, m_worldROS_baselink_vec.end()); 
//     // m_worldROS_baselink_vec.erase(m_worldROS_baselink_vec.begin(),last_valid.base());

//     //remove duplicates but keep the last one 
//     std::vector<std::pair<uint64_t, Eigen::Affine3d>, Eigen::aligned_allocator<std::pair<uint64_t, Eigen::Affine3d>>  >m_worldROS_baselink_vec_no_duplicates;
//     for(size_t i = 0; i < m_worldROS_baselink_vec.size(); i++){
//         //if the duplicates vec is empty just insert 
//         if(m_worldROS_baselink_vec_no_duplicates.empty()){
//             m_worldROS_baselink_vec_no_duplicates.push_back(m_worldROS_baselink_vec[i]);
//         }else{
//             //if not empty, check if the last one has the same timestamp, and if it does then just replace it
//             uint64_t ts_dup=m_worldROS_baselink_vec_no_duplicates.back().first;
//             uint64_t ts_cur=m_worldROS_baselink_vec[i].first;
//             if(ts_dup==ts_cur){
//                 // std::cout << "replacing " << m_worldROS_baselink_vec_no_duplicates.back().first << " with " << m_worldROS_baselink_vec[i].first << std::endl;;
//                 m_worldROS_baselink_vec_no_duplicates.back()=m_worldROS_baselink_vec[i]; //replace
//             }else{
//                 m_worldROS_baselink_vec_no_duplicates.push_back(m_worldROS_baselink_vec[i]);
//             }

//         }
//     }
//     m_worldROS_baselink_vec=m_worldROS_baselink_vec_no_duplicates;
    

// }

// bool DataLoaderImgRos::get_pose_at_timestamp(Eigen::Affine3d& pose, double& deviation_ms, const double timestamp){

//     // auto got = m_worldROS_baselink_map.find (timestamp);

//     //return the closest one
//     uint64_t closest_idx=-1;
//     double smallest_timestamp_diff=std::numeric_limits<double>::max();
//     double smallest_timestamp_diff_no_abs=std::numeric_limits<double>::max();
//     for (size_t i = 0; i < m_worldROS_baselink_vec.size(); i++) {
//         uint64_t recorded_timestamp=m_worldROS_baselink_vec[i].first;
//         // Eigen::Affine3d pose=m_worldROS_baselink_vec[i].second;
//         // std::cout << "comparing recorded_timestamp to timestmp \n" << recorded_timestamp << "\n" << timestamp << '\n';
//         double diff=fabs((double)timestamp- (double)recorded_timestamp);
//         if (  diff < smallest_timestamp_diff){
//             closest_idx=i;
//             smallest_timestamp_diff=diff;
//             smallest_timestamp_diff_no_abs=(double)timestamp - (double)recorded_timestamp;
//         }
//     }
//     if ( smallest_timestamp_diff*1e-6 > 60 ){
//         LOG(WARNING) << "time difference for pose is way too large! " << (smallest_timestamp_diff*1e-6) << "ms." << '\n';
//         return false;
//     }
//     // std::cout << "smallest_timestamp_diff is " << smallest_timestamp_diff << '\n';
//     // std::cout << "smallest_timestamp_diff ms is " << smallest_timestamp_diff*1e-6 << '\n';
//     // std::cout << "closest_idx is " << closest_idx << '\n';
//     // std::cout << "smallest_timestamp_diff_no_abs is " << smallest_timestamp_diff_no_abs << '\n';
//     deviation_ms=(double)(smallest_timestamp_diff_no_abs)*1e-6;
//     // std::cout << "deviation_ms is " << deviation_ms << '\n';
//     pose=m_worldROS_baselink_vec[closest_idx].second;
//     // std::cout << "pose is \n " << pose.matrix() << '\n';
//     return true;


// }

// bool DataLoaderImgRos::interpolate_pose_along_spline(Eigen::Affine3d& pose_world_baselink_interpolated, const Eigen::Affine3d pose, const double cur_timestamp_ns, const float deviation_ms){

//     // double new_time_ns=cur_timestamp_ns + deviation_ms*1e6;
//     double new_time_ns=cur_timestamp_ns;
//     // new_time_ns= new_time_ns/1e6;
//     // VLOG(4) << "Spline offset: " << offset_ms << " newTime: " << new_time_ns << " frame.ts: " << frame.timestamp <<" o[s]: "<<offset_ms / 1000 << " camId: "<< frame.cam_id << " dt: "<< (newTime-m_splines_per_cam[frame.cam_id]->m_min_ts);

//     bool valid=m_spline->getPose ( new_time_ns, pose_world_baselink_interpolated );

//     VLOG(2) << std::fixed<< "m_min_ts is " << m_spline->m_min_ts;
//     VLOG(2) << std::fixed<< "new_time is " << new_time_ns << " dt is " << new_time_ns-m_spline->m_min_ts;
//     VLOG(2) << std::fixed<< "pose is \n" << pose.matrix(); 
//     VLOG(2) << std::fixed<< "pose_interpolated is \n" << pose_world_baselink_interpolated.matrix(); 

//     //sometimes the rotation gets fucked up and I want to see why 
//     // Eigen::Vector3d angles_pose = pose.linear().eulerAngles(0, 1, 2); 
//     // Eigen::Vector3d angles_pose_interpolated = pose_world_baselink_interpolated.linear().eulerAngles(0, 1, 2); 
//     // //check if the difference is big
//     // Eigen::Vector3d diff=angles_pose-angles_pose_interpolated;
//     // if( diff.norm() >0.1 ){
//     //     LOG(ERROR) << "Something went wrong and we are jumping too much in the angles. Diff norm is " << diff.norm();
//     // }

//     Eigen::Affine3d pose_diff= pose* pose_world_baselink_interpolated.inverse();
//     VLOG(1) << "pose_diff: " <<pose_diff.matrix();
//     Eigen::Matrix3d rot_diff=pose_diff.linear() - Eigen::Matrix3d::Identity();
//     // Eigen::Vector3d angles_pose = .eulerAngles(0, 1, 2); 
//      if( rot_diff.norm() > 1.0 ){
//         LOG(FATAL) << "Something went wrong and we are jumping too much in the angles. Diff norm is " << rot_diff.norm();
//     }

//     if(!valid){
//         LOG(WARNING) << "Spline pose not valid, returning the original one";
//         return false; // return the old pose
//     }else{
//         return true;
//     }


// }