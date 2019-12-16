#pragma once 

//c++
#include <thread>
#include <unordered_map>
#include <vector>
#include <memory>

//ros
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>

//eigen 
#include <Eigen/Core>
#include<Eigen/StdVector>

#include "data_loaders/core/MeshCore.h"

#define NUM_CLOUDS_BUFFER 5

class RandGenerator;

class DataLoaderCloudRos
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderCloudRos(const std::string config_file);
    ~DataLoaderCloudRos();
    MeshCore get_cloud();
    bool has_data(){return m_is_modified;};
    bool is_loader_thread_alive(); //hacky way of checking if the thread is active and then killing the python process that created this loader


private:

    void init_params(const std::string config_file);
    void init_ros();
    void callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    Eigen::MatrixXd pcl2eigen(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    void read_pose_file();
    void read_pose_file_vel2lasermap();
    bool get_pose_at_timestamp(Eigen::Affine3d& pose, double& deviation_ms, const double timestamp);
    bool get_pose_vel2lasermap_at_timestamp(Eigen::Affine3d& pose, double& deviation_ms, const double timestamp);
    void create_transformation_matrices();
    void apply_transform(Eigen::MatrixXd& V, const Eigen::Affine3d& trans);

    //objects 
    std::shared_ptr<RandGenerator> m_rand_gen;

    //params
    std::thread m_loader_thread;
    std::string m_cloud_topic;
    bool m_do_pose; //if it's true we do a multiplication with the poses from the file, otherwise we just leave it in whatever frame we receive it at
    bool m_do_random_gap_removal; //remove a random small gap in the cloud so that the Mesher has a chance to unwrap around that direction and use delaunay in 2D
    std::string m_pose_file;
    std::string m_pose_file_format;
    int m_timestamp_multiplier;
    bool m_exact_time;
    float m_min_dist_filter; //removed all the points that have D(i) set to lower than this value. Useful for removing self intersection with the copter in the laser scans
    bool m_hacky_fix_for_razlaws_ma_bags;


    //internal
    std::atomic<bool> m_is_thread_running;
    bool m_is_modified; //indicate that a cloud was finished processind and you are ready to get it 
    std::vector<MeshCore, Eigen::aligned_allocator<MeshCore>> m_clouds_buffer; //need to use a buffer of meshes because the Loader needs to keep in memory both the calculated Cloud and the one its currently working on
    int m_finished_cloud_idx; //idx pointing to the most recent finished mesh
    int m_working_cloud_idx; //idx poiting to the mesh we are currently working on
    std::vector<std::pair<double, Eigen::Affine3d>, Eigen::aligned_allocator<std::pair<double, Eigen::Affine3d>>  >m_worldROS_baselink_vec;
    std::vector<std::pair<double, Eigen::Affine3d>, Eigen::aligned_allocator<std::pair<double, Eigen::Affine3d>>  >m_lasermap_vel_vec;
    Eigen::Affine3d m_tf_baselink_vel;
    Eigen::Affine3d m_tf_worldGL_worldROS;

};