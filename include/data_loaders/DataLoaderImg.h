#pragma once
//C++
#include <iosfwd>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <thread>
#include <math.h>       /* log2 */


//OpenCV
#include <opencv2/highgui/highgui.hpp>

//My stuff
#include "easy_pbr/Frame.h"
// #include "core/MeshCore.h"

//ros
// #include <ros/ros.h>

//Eigen
// #include<Eigen/StdVector>


//readerwriterqueue
#include "readerwriterqueue/readerwriterqueue.h"

//boost
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


struct file_timestamp_comparator{
    inline bool operator() (const fs::path& lhs, const fs::path& rhs)
    {
        double lhs_val=stod(lhs.stem().string());
        double rhs_val=stod(rhs.stem().string());
        return (lhs_val < rhs_val);
    }
};

struct nts_file_comparator{
    inline bool operator() (const fs::path& lhs, const fs::path& rhs)
    {
        //nts has frame_x, we remove the "frame_"
        std::string lhs_s = lhs.stem().string();
        std::string rhs_s = rhs.stem().string();
        lhs_s.erase(0,6);
        rhs_s.erase(0,6);
        double lhs_val=stod(lhs_s );
        double rhs_val=stod(rhs_s );
        return (lhs_val < rhs_val);
    }
};

enum DatasetType
{
    ICL = 0, // Icl nuim
    RPG, //
    TUM, // tum rgbd
    TUM3,
    NTS, // new tsukuba
    ETH, // euroc mav dataset
    BFS, // our bfs with small base line
    WBFS, // our bfs with wide base line + thermal
};




class DataLoaderImg{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderImg(const std::string config_file);
    ~DataLoaderImg();

    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    bool has_data_for_cam(const int cam_id);
    bool has_data_for_all_cams();
    easy_pbr::Frame get_frame_for_cam(const int cam_id);
    int get_nr_cams(){return m_nr_cams; };
    int nr_samples_for_cam(const int cam_id);
    void set_mask_for_cam(const std::string mask_filename, const int cam_id); //set a mask which will cause parts of the rgb, classes and probs images to be ignored
    void republish_last_frame_from_cam(const int cam_id); //put the last frame back into the ringbuffer so we cna read it from the core
    void republish_last_frame_all_cams();


    void reset(); //starts reading back from the start of the data log
    void clear_buffers(); //empties the ringbuffers, usefull for when scrolling through time


    //params
 

    //transforms
    Eigen::Affine3d m_tf_worldGL_worldROS;
    std::vector<std::pair<uint64_t, Eigen::Affine3f>,  Eigen::aligned_allocator<std::pair<uint64_t, Eigen::Affine3f>>   >m_worldROS_baselink_vec;

    std::vector< std::vector<fs::path> > m_rgb_filenames_per_cam; //list of images paths for each cam to read
    std::vector<int> m_idx_img_to_read_per_cam;
    std::vector<cv::Mat> m_undistort_map_x_per_cam; //vector containing the undistort map in x direction for each cam
    std::vector<cv::Mat> m_undistort_map_y_per_cam; //vector containing the undistort map in x direction for each cam




private:

    std::vector< moodycamel::ReaderWriterQueue<easy_pbr::Frame> , Eigen::aligned_allocator< moodycamel::ReaderWriterQueue<easy_pbr::Frame>>   > m_frames_buffer_per_cam;



    std::vector<std::thread> m_loader_threads;
    std::vector< fs::path > m_rgb_imgs_path_per_cam;
    std::vector< fs::path > m_labels_imgs_path_per_cam;
    std::vector< Eigen::Matrix3f,  Eigen::aligned_allocator<Eigen::Matrix3f> > m_intrinsics_per_cam;
    std::vector<cv::Mat > m_mask_per_cam;
    unsigned int m_nr_cams;


    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-
    bool m_only_rgb;
    bool m_shuffle;
    bool m_sort_by_filename;
    // float m_tf_worldGL_worldROS_angle;
    // std::string m_tf_worldGL_worldROS_axis;
    // std::string m_dataset_type;
    DatasetType m_dataset_type;
    std::string m_pose_file;
    std::vector<easy_pbr::Frame, Eigen::aligned_allocator<easy_pbr::Frame>> m_last_frame_per_cam; //stores the last frame for each of the cameras
    std::vector<bool> m_get_last_published_frame_for_cam; //if we shoudl return the last published frame or not

    float m_rgb_subsample_factor;
    int m_imgs_to_skip;
    int m_nr_images_to_read; //nr images to read starting from m_imgs_to_skip
    bool m_do_overfit;
    int m_nr_resets;


    void init_params(const std::string config_file);
    void init_params_configuru();
    void init_data_reading();
    // void read_pose_file();

    //read pose files
    void read_pose_file_eth();
    void read_pose_file_icl();
    void read_pose_file_nts();

    //get poses depending on the datset
    bool get_pose_at_timestamp(Eigen::Affine3f& pose, const uint64_t timestamp, const uint64_t cam_id);

    //get the intrinsics depending on the dataset
    void get_intrinsics(Eigen::Matrix3f& K, Eigen::Matrix<float, 5, 1>& distort_coeffs, const uint64_t cam_id);


    void read_data_for_cam(const int cam_id);
    // void read_pose_file_semantic_fusion();
    bool get_pose_at_timestamp(Eigen::Affine3f& pose, const uint64_t timestamp);
    void create_transformation_matrices();



};


