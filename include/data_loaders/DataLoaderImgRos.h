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

//eigen 
#include <Eigen/Core>
#include<Eigen/StdVector>

#include "easy_pbr/Frame.h"

//spline
// #include "ceres_spline_opt.h"

#define NUM_FRAMES_BUFFER 8


struct Cam{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //params
    std::string m_img_topic;
    int m_img_subsample_factor=1;
    bool m_is_compressed=false;
    std::string m_intrinsics_string; // stores the fx fy cx cy
    uint64_t m_timestamp_offset=0; // offset applied to the timestamp of the message in case we need to manually syncronize something with ws/scripts/rosbag_time_diff.py 

    //frames
    std::vector<Frame, Eigen::aligned_allocator<Frame>  > m_frames_buffer = std::vector<Frame, Eigen::aligned_allocator<Frame>  > (NUM_FRAMES_BUFFER); //need to use a buffer of Frames because the Loader needs to keep in memory both the calculated Frame and the one its currently working on
    int m_finished_idx=-1; //idx pointing to the most recent finished frame
    int m_working_idx=0; //idx poiting to the frame we are currently working on
    int m_last_retrieved_idx=-1; //idx pointing to the one that was last retraived with get_frame() //this one might still be in usage
    // std::mutex m_push_pull_mutex; 
    std::shared_ptr<std::mutex> m_push_pull_mutex=std::make_shared<std::mutex>();; //when retriving new data with get_data or pushing new data, we check and update m_last_retreived_idx. This has to be mutexed its a ptr because othewise the class becomes non copyable

    bool m_is_modified=false; //indicate that a frame for this cam was finished processind and you are ready to get it 

};

class DataLoaderImgRos
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderImgRos(const std::string config_file);
    ~DataLoaderImgRos();
    // Frame get_frame_for_cam();
    Frame get_frame_for_cam(const int cam_id);
    Frame get_last_retrieved_frame();
    // bool is_modified(){return m_is_modified;};
    int nr_cams();
    int has_data_for_cam(const int cam_id);
    bool is_loader_thread_alive(); //hacky way of checking if the thread is active and then killing the python process that created this loader


private:

    void init_params(const std::string config_file);
    void init_ros();
    void init_spline();
    void callback(const sensor_msgs::ImageConstPtr& msg, const int cam_id);
    void read_pose_file();
    bool get_pose_at_timestamp(Eigen::Affine3d& pose, double& deviation_ms, const double timestamp);
    bool interpolate_pose_along_spline(Eigen::Affine3d& pose_world_baselink_interpolated, const Eigen::Affine3d pose, const double cur_timestamp_ns, const float deviation_ms);


    //params
    std::thread m_loader_thread; //there is one loader thread for the moment but we will need different ones for different cameras
    // std::vector<std::string> m_img_topic_per_cam;
    // std::vector<int> m_img_subsample_factor_per_cam; 
    // std::vector<bool> m_is_compressed_per_cam;
    std::vector<Cam, Eigen::aligned_allocator<Cam> > m_cams;
    bool m_do_pose;

    //internal
    // std::vector<Frame, Eigen::aligned_allocator<Frame>  > m_frames_buffer; //need to use a buffer of Frames because the Loader needs to keep in memory both the calculated Frame and the one its currently working on
    // int m_finished_idx; //idx pointing to the most recent finished frame
    // int m_working_idx; //idx poiting to the frame we are currently working on
    // int m_last_retrieved_idx; //idx pointing to the one that was last retraived with get_frame() //this one might still be in usage
    // std::mutex m_push_pull_mutex; //when retriving new data with get_data or pushing new data, we check and update m_last_retreived_idx. This has to be mutexed 

    std::atomic<bool> m_is_thread_running;
    std::string m_pose_file;
    std::vector<std::pair<uint64_t, Eigen::Affine3d>, Eigen::aligned_allocator<std::pair<uint64_t, Eigen::Affine3d>>  >m_worldROS_baselink_vec;
    int m_nr_callbacks;
    //spline interpolation
    CeresSplineOptimizationPtr m_spline;


};