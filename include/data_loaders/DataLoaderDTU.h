#pragma once


#include <thread>
#include <unordered_map>
#include <vector>
#include <atomic>



//eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

//readerwriterqueue
// #include "readerwriterqueue/readerwriterqueue.h"

//boost
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>

#include "easy_pbr/Frame.h"



namespace radu { namespace utils{
    class RandGenerator;
}}

// namespace easy_pbr{
//     class Frame;
// }
// class DataTransformer;


class DataLoaderDTU
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderDTU(const std::string config_file);
    ~DataLoaderDTU();
    void start(); //starts reading the data from disk. This gets called automatically if we have autostart=true
    easy_pbr::Frame get_random_frame();
    easy_pbr::Frame get_frame_at_idx( const int idx);
    easy_pbr::Frame get_closest_frame( const easy_pbr::Frame& frame);
    std::vector< easy_pbr::Frame > get_all_frames(); 
    void start_reading_next_scene(); //switch to another scene from this object and start reading it
    bool finished_reading_scene(); //returns true when we have finished reading everything for that one scene of the corresponding object and we can safely use get_random_frame
    bool has_data(); //calls internally finished_reading scene. It's mostly a convenience function
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of images for the object that we selected
    int nr_scenes(); //returns the number of scenes that we have for this mode
    bool is_finished(); //check if we finished reading all the scenes
    std::string get_object_name();
    void set_dataset_path(const std::string dataset_path);
    void set_object_name(const std::string object_name);
    void set_restrict_to_scene_name(const std::string scene_name);
    std::string get_restrict_to_scene_name();
    std::string get_current_scene_name(); //the scene we have currently active
    void set_subsample_factor(const int val);
    void set_scene_scale_multiplier(const float scene_scale_multiplier);
    void set_rotate_scene_x_axis_degrees(const float degrees);
    void set_load_mask(bool load_mask);
    void set_mode_train(); //set the loader so that it starts reading form the training set
    void set_mode_test();
    void set_mode_validation();
    void set_mode_all(); //load all images
    void set_preload_to_gpu_tensors(const bool val);
    Eigen::Affine3f get_tf_easypbr_dtu();




private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void read_scene(const std::string scene_path); //a path to the scene which contains all the  images and the pose and so on
    std::unordered_map<std::string, std::string> create_mapping_classnr2classname(); //create the mapping between the weird nr of a class to the actual class name
    void load_images_in_frame(easy_pbr::Frame& frame);
    void read_poses_and_intrinsics();


    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    // std::shared_ptr<DataTransformer> m_transformer;

    //params
    // bool m_autostart;
    std::atomic<bool> m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    bool m_autostart;
    bool m_read_with_bg_thread;
    std::string m_mode; // train or test or val
    bool m_load_mask;
    int m_subsample_factor;
    bool m_shuffle;
    bool m_do_overfit; // return all the time just images from the the first scene of that specified object class
    std::string m_restrict_to_scene_name; //restrict to only one of the scans
    float m_scene_scale_multiplier; //multiplier the scene scale with this value so that we keep it in a range that we can expect
    float m_rotate_scene_x_axis_degrees;
    boost::filesystem::path m_dataset_path;  //get the path where all the off files are
    bool m_load_as_shell;
    bool m_preload_to_gpu_tensors;
    std::thread m_loader_thread;
    int m_nr_resets;
    int m_idx_scene_to_read;
    // Eigen::Affine3f m_tf_easypbr_dtu;



    //internal
    std::vector<boost::filesystem::path> m_scene_folders; //contains all the folders of the scenes for this objects
    std::vector< easy_pbr::Frame > m_frames_for_scene;
    std::unordered_map<std::string,      std::unordered_map<int, Eigen::Affine3f>     > m_scene2frame_idx2tf_cam_world;
    std::unordered_map<std::string,      std::unordered_map<int, Eigen::Matrix3f>    > m_scene2frame_idx2K;
    std::unordered_map<std::string,      Eigen::Affine3f    > m_scene2tf_easypbr_dtu; //they key is the scan name eg: dtu_scan65

    int m_nr_scenes_read_so_far;
    std::string m_current_scene_name;

};
