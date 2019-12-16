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
#include "data_loaders/utils/RosTools.h"
#include <image_transport/image_transport.h>

//my stuff 
#include "data_loaders/viewer/Mesh.h"
#include "data_loaders/viewer/Gui.h"
#include "data_loaders/utils/MiscUtils.h"

using namespace er::utils;


DataLoaderImgRos::DataLoaderImgRos(const std::string config_file):
    // m_frames_buffer(NUM_FRAMES_BUFFER),
    // m_finished_idx(-1),
    // m_working_idx(0),
    // m_last_retrieved_idx(-1),
    m_is_thread_running(false),
    m_nr_callbacks(0)
{
    init_params(config_file);
    if (m_do_pose){
        read_pose_file();
        init_spline();
    }
    // init_ros();
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
    //get the config filename
    // ros::NodeHandle private_nh("~");
    // std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    // std::string config_file="config.cfg";

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config loader_config=cfg["loader_img_ros"];
    int nr_cams = loader_config["nr_cams"];
    m_do_pose = loader_config["do_pose"];

    //create the cams 
    m_cams.resize(nr_cams);
    for(int i = 0; i < nr_cams; i++){
        m_cams[i].m_img_topic = (std::string)loader_config["img_topic_"+std::to_string(i)];
        m_cams[i].m_img_subsample_factor = loader_config["img_subsample_factor_"+std::to_string(i)]; 
        m_cams[i].m_is_compressed = loader_config["is_compressed_"+std::to_string(i)];
        m_cams[i].m_intrinsics_string = (std::string)loader_config["intrinsics_"+std::to_string(i)];
        m_cams[i].m_timestamp_offset = (u_int64_t)loader_config["timestamp_offset_"+std::to_string(i)];
    }
    m_pose_file = (std::string)loader_config["pose_file"];
}

void DataLoaderImgRos::init_ros(){
    std::vector<std::pair<std::string, std::string> > dummy_remappings;
    ros::init(dummy_remappings, "dummy_name");

    ros::NodeHandle private_nh("~");
    // ros::Subscriber sub = private_nh.subscribe(m_img_topic, 5, &DataLoaderImgRos::callback, this);

    image_transport::ImageTransport img_transport( private_nh );
    std::vector<image_transport::Subscriber> img_subs;
    img_subs.resize(m_cams.size());

    //make the subscribers
    for(size_t i = 0; i < m_cams.size(); i++){
        if(m_cams[i].m_is_compressed){
            img_subs[i] = img_transport.subscribe(m_cams[i].m_img_topic, 5, boost::bind(&DataLoaderImgRos::callback, this, _1, i), ros::VoidPtr(), image_transport::TransportHints("compressed"));
        }else{
            img_subs[i] = img_transport.subscribe(m_cams[i].m_img_topic, 5,  boost::bind(&DataLoaderImgRos::callback, this, _1, i) );
        }
    }

//    std::cout <<"subscribing" << std::endl;
    // image_transport::Subscriber sub;
    // if(m_is_compressed){
        // sub = img_transport.subscribe("/stereo_left/image_raw", 5, &DataLoaderImgRos::callback, this, image_transport::TransportHints("compressed"));
        // sub = img_transport.subscribe("/stereo_left/image_raw", 5, boost::bind(&DataLoaderImgRos::callback, this, _1, 0), ros::VoidPtr(),  image_transport::TransportHints("compressed"));
    // }else{
    //     sub = img_transport.subscribe(m_img_topic, 5, &DataLoaderImgRos::callback, this );
    // }
    // image_transport::Subscriber sub = img_transport.subscribe(m_img_topic, 5, &DataLoaderImgRos::callback, this, image_transport::TransportHints("compressed"));
    // ros::Subscriber sub = private_nh.subscribe(m_img_topic, 5, &DataLoaderImgRos::callback, this,  ros::TransportHints("compressed"));

    // ros::spin();

    //multithreaded spinning, each callback (from different cameras) will run on different threads in paralel
    ros::MultiThreadedSpinner spinner(m_cams.size()); // Use as many threads as cameras
    spinner.spin();

    LOG(INFO) << "finished ros communication";
    std::cout << "inside thread" << "\n";
    m_is_thread_running=false;
}

void DataLoaderImgRos::init_spline(){
    m_spline = std::make_shared<CeresSplineOptimization>();
    m_spline->initSpline( m_worldROS_baselink_vec );
}


void DataLoaderImgRos::callback(const sensor_msgs::ImageConstPtr& img_msg, const int cam_id) {

    // std::cout << "loll callback from " << cam_id <<std::endl;
    VLOG(1)<< "callback from cam " << cam_id;
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

    //pose
    Eigen::Affine3d sensor_pose;  //maps from velodyne to world ros
    sensor_pose.setIdentity();
    if (m_do_pose){
        double timestamp=(double)img_msg->header.stamp.toNSec();
        // timestamp+=1548857368082465171;
        timestamp+=cam.m_timestamp_offset;
        double deviation_ms=-1;
        if (!get_pose_at_timestamp(sensor_pose,deviation_ms, timestamp)){
            LOG(WARNING) << "Not found any pose at timestamp " << timestamp << " Discarding";
            return;
        }
    }



    Eigen::Affine3d m_tf_vel_cam; 
    m_tf_vel_cam.setIdentity();
//     - Translation: [-0.111, -0.135, -0.053]                                                                                                                                                                                                                                          
// - Rotation: in Quaternion [0.683, -0.010, -0.006, 0.730]                                                                                                                                                                                                                         
//             in RPY (radian) [1.505, -0.006, -0.023]                                                                                                                                                                                                                              
//             in RPY (degree) [86.230, -0.332, -1.308]
    Eigen::Vector3d vel_cam_t(-0.111, -0.135, -0.053);
    Eigen::AngleAxisd rollAngle(1.505, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd yawAngle(-0.023, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(-0.006, Eigen::Vector3d::UnitZ());
    Eigen::Quaternion<double> vel_cam_quat = pitchAngle * yawAngle * rollAngle;
    m_tf_vel_cam.matrix().block<3,3>(0,0)=vel_cam_quat.toRotationMatrix();
    m_tf_vel_cam.matrix().block<3,1>(0,3)=vel_cam_t;

    Eigen::Affine3d m_tf_worldGL_worldROS;
    m_tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3d worldGL_worldROS_rot;
    worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;

    // Eigen::Affine3d spline_pose=interpolate_pose_along_spline(sensor_pose, frame.timestamp, -deviation_ms);
    Eigen::Affine3d sensor_pose_interpolated=sensor_pose;
    // bool valid=interpolate_pose_along_spline(sensor_pose_interpolated, sensor_pose, timestamp, -deviation_ms);
    // bool valid=interpolate_pose_along_spline(sensor_pose_interpolated, sensor_pose, timestamp, 0.0);
    // if(!valid){
    //     LOG(WARNING) << "Interpolating along spline failed. Discarding";
    //     return;
    // }
    

    // frame.tf_cam_world= m_tf_cam_vel.cast<float>() *sensor_pose.cast<float>().inverse(); //from world to vel and from vel to cam
    // frame.tf_cam_world= m_tf_vel_cam.cast<float>().inverse() * sensor_pose.cast<float>().inverse() * m_tf_worldGL_worldROS.cast<float>().inverse(); //from worldgl to world ros, from world ros to velodyne, from velodyne to cam
    frame.tf_cam_world= m_tf_vel_cam.cast<float>().inverse() * sensor_pose_interpolated.cast<float>().inverse() * m_tf_worldGL_worldROS.cast<float>().inverse(); //from worldgl to world ros, from world ros to velodyne, from velodyne to cam

    if(cam_id==1){
        //this it the thermal camera so we now make another transform which moves it a bit to the right
        Eigen::Affine3f tf_camThermal_camLeft; 
        tf_camThermal_camLeft.setIdentity();
        float deviation_m=-0.1113778; //this moves it to the right of the left cam
        tf_camThermal_camLeft.translation()<< deviation_m, 0.0, 0.0;
        frame.tf_cam_world= tf_camThermal_camLeft* frame.tf_cam_world;
        
        //CHECK how this one looks like. The translation between val and thermal should have at least one component equal to 0
        Eigen::Affine3f tf_camThermal_vel; 
        tf_camThermal_vel= tf_camThermal_camLeft * m_tf_vel_cam.cast<float>().inverse();
        Eigen::Affine3f tf_cam_vel;
        tf_cam_vel= m_tf_vel_cam.cast<float>().inverse();
        VLOG(1) << "translation of cam_vel: " << tf_cam_vel.translation().transpose();
        VLOG(1) << "translation of camthemal_vel: " << tf_camThermal_vel.translation().transpose();

    }

    //set K
    // std::string intrinsics_string="727.92526982,  730.38109657,  326.47574019,  260.87943595"; //for thermal
    // std::string intrinsics_string="2404.76891592 2400.9170077 1245.04813256 1003.61646891"; //
    // std::vector<std::string> intrinsics_split=split(intrinsics_string," ");
    std::vector<std::string> intrinsics_split=split(cam.m_intrinsics_string," ");
    frame.K.setIdentity();
    frame.K(0,0)=std::stof(intrinsics_split[0]); //fx
    frame.K(1,1)=std::stof(intrinsics_split[1]); //fy
    frame.K(0,2)=std::stof(intrinsics_split[2]); // cx
    frame.K(1,2)=std::stof(intrinsics_split[3]); //cy
    frame.K*=1.0/cam.m_img_subsample_factor;



    //subsample (cannot subsample the cv_img directly because it may be a bayered image)
    // cv::resize(cv_img, cv_img, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor);
    std::cout << " got an image of size" << cv_img.rows << " " << cv_img.cols << " type is " << type2string(cv_img.type()) << std::endl;

  
    if(cv_img.depth()==CV_16U){
        std::cout << "got a thermal image" << std::endl;
        //it is most likely a thermal image
        frame.thermal_16u=cv_img;
        cv::resize(frame.thermal_16u, frame.thermal_16u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor);
        frame.thermal_16u.convertTo(frame.thermal_32f, CV_32FC1, 1.0/65536.0);
        //for showing we put it also in 3 channels
        std::vector<cv::Mat> matChannels;
        matChannels.push_back(frame.thermal_32f);
        matChannels.push_back(frame.thermal_32f);
        matChannels.push_back(frame.thermal_32f);
        // cv::Mat_<cv::Vec<float,3> > out;
        cv::merge(matChannels, frame.thermal_vis_32f);

        // Gui::show(frame.thermal_32f, "thermal_32f");
        frame.width=frame.thermal_32f.cols;
        frame.height=frame.thermal_32f.rows;
    }else if(cv_img.depth()==CV_8U){
        //just an rgb image
        //convert to float
        if(cv_img.channels()==1){
            // frame.rgb_8u.convertTo(frame.gray_32f, CV_32FC1, 1.0/255.0);
            cv::cvtColor ( cv_img, frame.rgb_8u, CV_BayerBG2RGB );  // only works on 8u and 16u
            cv::cvtColor ( frame.rgb_8u, frame.rgb_8u, CV_BGR2RGB );  
            cv::resize(frame.rgb_8u, frame.rgb_8u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor); // can only do it here because cv image is a bayered image
            frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
            // std::cout << "After debayer the type is " << type2string(frame.rgb_32f.type()) << std::endl;
            cv::cvtColor ( frame.rgb_32f, frame.gray_32f, CV_BGR2GRAY );      
            //gradients
            cv::Scharr( frame.gray_32f, frame.grad_x_32f, CV_32F, 1, 0);
            cv::Scharr( frame.gray_32f, frame.grad_y_32f, CV_32F, 0, 1);
        }else if(cv_img.channels()==2){
            frame.rgb_8u=cv_img;
            cv::resize(frame.rgb_8u, frame.rgb_8u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor); 
            frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC2, 1.0/255.0);
        }else if(cv_img.channels()==3){
            frame.rgb_8u=cv_img;
            cv::resize(frame.rgb_8u, frame.rgb_8u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor); 
            frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
            cv::cvtColor ( frame.rgb_32f, frame.gray_32f, CV_BGR2GRAY );      
            //gradients
            cv::Scharr( frame.gray_32f, frame.grad_x_32f, CV_32F, 1, 0);
            cv::Scharr( frame.gray_32f, frame.grad_y_32f, CV_32F, 0, 1);
        }else if(cv_img.channels()==4){
            frame.rgb_8u=cv_img;
            cv::resize(frame.rgb_8u, frame.rgb_8u, cv::Size(), 1.0/cam.m_img_subsample_factor, 1.0/cam.m_img_subsample_factor); 
            frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC4, 1.0/255.0);
        }else{
            LOG(FATAL) << "Not a known number of channels";
        }

        frame.width=frame.rgb_32f.cols;
        frame.height=frame.rgb_32f.rows;
    }else{
        LOG(WARNING) << "Unknown image type";
    } 


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



    cam.m_push_pull_mutex->lock();  
    bool about_to_overwrite= (cam.m_working_idx==cam.m_last_retrieved_idx);
    cam.m_push_pull_mutex->unlock();  
    if(about_to_overwrite){
        //we are about to overwrite on top of the finished one which might be at the moment used by the gpu
        LOG(WARNING) << "About to overwrite, maybe your consumer thread is too slow or maybe this queue in the dataloader is too small. Discarding";
        cam.m_is_modified = true;
        return; 
    }
    cam.m_frames_buffer[cam.m_working_idx]=frame;
    cam.m_finished_idx = cam.m_working_idx;
    cam.m_working_idx = (cam.m_working_idx + 1) % NUM_FRAMES_BUFFER;
    cam.m_is_modified = true;

    m_nr_callbacks++; 

}

bool DataLoaderImgRos::is_loader_thread_alive(){
    // std::cout << " is thread alive is " << m_is_thread_running;
    //HACK asfuck because for some reason pybind cannot read correctly the bool that I am returning
    // if(!m_is_thread_running){
        // delete this;
    // }

    return m_is_thread_running;
}


int DataLoaderImgRos::has_data_for_cam(const int cam_id){
    return m_cams[cam_id].m_is_modified;
}

Frame DataLoaderImgRos::get_frame_for_cam(const int cam_id){
    Cam& cam=m_cams[cam_id];

    std::lock_guard<std::mutex> lock(*cam.m_push_pull_mutex);  
    cam.m_is_modified=false;
    cam.m_last_retrieved_idx=cam.m_finished_idx;
    // VLOG(1) << "RETRIEVING m_last_retrieved is " << m_last_retreived_idx ;
    return cam.m_frames_buffer[cam.m_finished_idx];
}

Frame DataLoaderImgRos::get_last_retrieved_frame(){
    // std::lock_guard<std::mutex> lock(m_push_pull_mutex);  
    // return m_frames_buffer[m_last_retrieved_idx];
}

int DataLoaderImgRos::nr_cams(){
    return m_cams.size();
}

void DataLoaderImgRos::read_pose_file(){
    std::ifstream infile( m_pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
    uint64_t timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond quat;


    std::string line;
//     while (std::getline(infile, line)) {
//         std::istringstream iss(line);
//         iss >> scan_nr >> timestamp
//             >> position(0) >> position(1) >> position(2)
//             >> quat.w() >> quat.x() >> quat.y() >> quat.z();
//     //    std::cout << "input is \n" << scan_nr << " " << timestamp << "  \n Position is" << position << " " << quat.matrix()  << "\n";
//         Eigen::Affine3d pose;
//         pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
//         pose.matrix().block<3,1>(0,3)=position;
//         // if(m_do_timestamp_rounding_when_reading_file){
//         //     timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
//         // }
//        // VLOG(2) << "recorded tmestamp is " << timestamp;
// //        VLOG(2) << "recorded scan_nr is " << scan_nr;
//         m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3d>(timestamp,pose) );
//     }



    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >> timestamp
            >> position.x() >> position.y() >> position.z()
            >> quat.x() >> quat.y() >> quat.z() >> quat.w();
    //    std::cout << "input is \n" << " " << timestamp << "  \n Position is" << position << " " << quat.matrix()  << "\n";
        Eigen::Affine3d pose;
        pose.setIdentity();
        pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
        pose.matrix().block<3,1>(0,3)=position;
        // if(m_do_timestamp_rounding_when_reading_file){
        //     timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
        // }
       // VLOG(2) << "recorded tmestamp is " << timestamp;
//        VLOG(2) << "recorded scan_nr is " << scan_nr;
        // Eigen::Affine3d pose_inv=pose.inverse();
        
        m_worldROS_baselink_vec.push_back ( std::pair<uint64_t, Eigen::Affine3d>(timestamp,pose) );
    }



    std::sort ( m_worldROS_baselink_vec.begin(), m_worldROS_baselink_vec.end(), [](const std::pair<uint64_t,Eigen::Affine3d> & a, const std::pair<uint64_t,Eigen::Affine3d> & b ){return a.first < b.first;});


    // auto last_valid=std::unique ( m_worldROS_baselink_vec.begin(), m_worldROS_baselink_vec.end(), [](const std::pair<double,Eigen::Affine3d> & a, const std::pair<double,Eigen::Affine3d> & b ){return a.first == b.first;});
    // // m_worldROS_baselink_vec.erase(last_valid, m_worldROS_baselink_vec.end()); 
    // m_worldROS_baselink_vec.erase(m_worldROS_baselink_vec.begin(),last_valid.base());

    //remove duplicates but keep the last one 
    std::vector<std::pair<uint64_t, Eigen::Affine3d>, Eigen::aligned_allocator<std::pair<uint64_t, Eigen::Affine3d>>  >m_worldROS_baselink_vec_no_duplicates;
    for(size_t i = 0; i < m_worldROS_baselink_vec.size(); i++){
        //if the duplicates vec is empty just insert 
        if(m_worldROS_baselink_vec_no_duplicates.empty()){
            m_worldROS_baselink_vec_no_duplicates.push_back(m_worldROS_baselink_vec[i]);
        }else{
            //if not empty, check if the last one has the same timestamp, and if it does then just replace it
            uint64_t ts_dup=m_worldROS_baselink_vec_no_duplicates.back().first;
            uint64_t ts_cur=m_worldROS_baselink_vec[i].first;
            if(ts_dup==ts_cur){
                // std::cout << "replacing " << m_worldROS_baselink_vec_no_duplicates.back().first << " with " << m_worldROS_baselink_vec[i].first << std::endl;;
                m_worldROS_baselink_vec_no_duplicates.back()=m_worldROS_baselink_vec[i]; //replace
            }else{
                m_worldROS_baselink_vec_no_duplicates.push_back(m_worldROS_baselink_vec[i]);
            }

        }
    }
    m_worldROS_baselink_vec=m_worldROS_baselink_vec_no_duplicates;
    

}

bool DataLoaderImgRos::get_pose_at_timestamp(Eigen::Affine3d& pose, double& deviation_ms, const double timestamp){

    // auto got = m_worldROS_baselink_map.find (timestamp);

    //return the closest one
    uint64_t closest_idx=-1;
    double smallest_timestamp_diff=std::numeric_limits<double>::max();
    double smallest_timestamp_diff_no_abs=std::numeric_limits<double>::max();
    for (size_t i = 0; i < m_worldROS_baselink_vec.size(); i++) {
        uint64_t recorded_timestamp=m_worldROS_baselink_vec[i].first;
        // Eigen::Affine3d pose=m_worldROS_baselink_vec[i].second;
        // std::cout << "comparing recorded_timestamp to timestmp \n" << recorded_timestamp << "\n" << timestamp << '\n';
        double diff=fabs((double)timestamp- (double)recorded_timestamp);
        if (  diff < smallest_timestamp_diff){
            closest_idx=i;
            smallest_timestamp_diff=diff;
            smallest_timestamp_diff_no_abs=(double)timestamp - (double)recorded_timestamp;
        }
    }
    if ( smallest_timestamp_diff*1e-6 > 60 ){
        LOG(WARNING) << "time difference for pose is way too large! " << (smallest_timestamp_diff*1e-6) << "ms." << '\n';
        return false;
    }
    // std::cout << "smallest_timestamp_diff is " << smallest_timestamp_diff << '\n';
    // std::cout << "smallest_timestamp_diff ms is " << smallest_timestamp_diff*1e-6 << '\n';
    // std::cout << "closest_idx is " << closest_idx << '\n';
    // std::cout << "smallest_timestamp_diff_no_abs is " << smallest_timestamp_diff_no_abs << '\n';
    deviation_ms=(double)(smallest_timestamp_diff_no_abs)*1e-6;
    // std::cout << "deviation_ms is " << deviation_ms << '\n';
    pose=m_worldROS_baselink_vec[closest_idx].second;
    // std::cout << "pose is \n " << pose.matrix() << '\n';
    return true;


}

bool DataLoaderImgRos::interpolate_pose_along_spline(Eigen::Affine3d& pose_world_baselink_interpolated, const Eigen::Affine3d pose, const double cur_timestamp_ns, const float deviation_ms){

    // double new_time_ns=cur_timestamp_ns + deviation_ms*1e6;
    double new_time_ns=cur_timestamp_ns;
    // new_time_ns= new_time_ns/1e6;
    // VLOG(4) << "Spline offset: " << offset_ms << " newTime: " << new_time_ns << " frame.ts: " << frame.timestamp <<" o[s]: "<<offset_ms / 1000 << " camId: "<< frame.cam_id << " dt: "<< (newTime-m_splines_per_cam[frame.cam_id]->m_min_ts);

    bool valid=m_spline->getPose ( new_time_ns, pose_world_baselink_interpolated );

    VLOG(2) << std::fixed<< "m_min_ts is " << m_spline->m_min_ts;
    VLOG(2) << std::fixed<< "new_time is " << new_time_ns << " dt is " << new_time_ns-m_spline->m_min_ts;
    VLOG(2) << std::fixed<< "pose is \n" << pose.matrix(); 
    VLOG(2) << std::fixed<< "pose_interpolated is \n" << pose_world_baselink_interpolated.matrix(); 

    //sometimes the rotation gets fucked up and I want to see why 
    // Eigen::Vector3d angles_pose = pose.linear().eulerAngles(0, 1, 2); 
    // Eigen::Vector3d angles_pose_interpolated = pose_world_baselink_interpolated.linear().eulerAngles(0, 1, 2); 
    // //check if the difference is big
    // Eigen::Vector3d diff=angles_pose-angles_pose_interpolated;
    // if( diff.norm() >0.1 ){
    //     LOG(ERROR) << "Something went wrong and we are jumping too much in the angles. Diff norm is " << diff.norm();
    // }

    Eigen::Affine3d pose_diff= pose* pose_world_baselink_interpolated.inverse();
    VLOG(1) << "pose_diff: " <<pose_diff.matrix();
    Eigen::Matrix3d rot_diff=pose_diff.linear() - Eigen::Matrix3d::Identity();
    // Eigen::Vector3d angles_pose = .eulerAngles(0, 1, 2); 
     if( rot_diff.norm() > 1.0 ){
        LOG(FATAL) << "Something went wrong and we are jumping too much in the angles. Diff norm is " << rot_diff.norm();
    }

    if(!valid){
        LOG(WARNING) << "Spline pose not valid, returning the original one";
        return false; // return the old pose
    }else{
        return true;
    }


}