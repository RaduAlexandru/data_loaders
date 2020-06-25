#pragma once 

//c++
#include <thread>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <atomic>

//ros
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>


//eigen 
#include <Eigen/Core>
#include<Eigen/StdVector>

#include "easy_pbr/Frame.h"


//readerwriterqueue
#include "readerwriterqueue/readerwriterqueue.h"

//boost
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

struct Cam{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //params
    std::string m_img_topic;
    int m_img_subsample_factor=1;
    bool m_is_compressed=false;
    std::string m_cam_info_topic;

    //frames
    moodycamel::ReaderWriterQueue<easy_pbr::Frame> m_frames_buffer;
    sensor_msgs::CameraInfoConstPtr m_cam_info;

};

class DataLoaderImgRos
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderImgRos(const std::string config_file);
    ~DataLoaderImgRos();
    easy_pbr::Frame get_frame_for_cam(const int cam_id);
    int nr_cams();
    bool has_data_for_all_cams();
    bool has_data_for_cam(const int cam_id);
    bool is_loader_thread_alive(); //hacky way of checking if the thread is active and then killing the python process that created this loader


private:

    void init_params(const std::string config_file);
    void init_ros();
    void callback_img(const sensor_msgs::ImageConstPtr& msg, const int cam_id);
    void callback_cam_info(const sensor_msgs::CameraInfoConstPtr& msg, const int cam_id);
    // void read_pose_file();
    // bool get_pose_at_timestamp(Eigen::Affine3d& pose, double& deviation_ms, const double timestamp);
    // bool interpolate_pose_along_spline(Eigen::Affine3d& pose_world_baselink_interpolated, const Eigen::Affine3d pose, const double cur_timestamp_ns, const float deviation_ms);


    //params
    std::thread m_loader_thread; //there is one loader thread for the moment but we will need different ones for different cameras
    std::vector<Cam, Eigen::aligned_allocator<Cam> > m_cams;
    std::string m_pose_source;
    std::string m_tf_reference_frame;
    std::string m_cam_info_source;
    // bool m_do_pose;

 

    std::atomic<bool> m_is_thread_running;
    // std::string m_pose_file;
    // std::vector<std::pair<uint64_t, Eigen::Affine3d>, Eigen::aligned_allocator<std::pair<uint64_t, Eigen::Affine3d>>  >m_worldROS_baselink_vec;
    int m_nr_callbacks;
    //spline interpolation
    // CeresSplineOptimizationPtr m_spline;

    // ros::NodeHandle private_nh;
    // tf2_ros::Buffer m_tf_buf;
    // tf2_ros::TransformListener m_tf_listener{m_tf_buf};



};