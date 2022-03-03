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

namespace easy_pbr{
    class Mesh;
}
// class DataTransformer;


class DataLoaderEasyPBR
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderEasyPBR(const std::string config_file);
    ~DataLoaderEasyPBR();
    void start(); //starts reading the data from disk. This gets called automatically if we have autostart=true
    easy_pbr::Frame get_next_frame();
    std::vector<easy_pbr::Frame> get_all_frames();
    easy_pbr::Frame get_frame_at_idx( const int idx);
    easy_pbr::Frame get_closest_frame( const easy_pbr::Frame& frame); //return the one closest frame
    std::vector<easy_pbr::Frame> get_close_frames( const easy_pbr::Frame& frame, const int nr_frames, const bool discard_same_idx ); //return a certain number of frames ordered by proximity,
    // std::vector<float> compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames);
    easy_pbr::Frame get_random_frame();
    bool loaded_scene_mesh(){ return m_loaded_scene_mesh;  };
    std::shared_ptr<easy_pbr::Mesh> get_scene_mesh(){ return m_scene_mesh;   };
    bool has_data(); //will reeturn always true because this dataloader preloads all the frames and keeps them in memory all the time. They are not so many
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of scenes for the object that we selected
    bool is_finished(); //check if we finished reading all the images from the scene
    void set_mode_train(); //set the loader so that it starts reading form the training set
    void set_mode_test();
    void set_mode_validation();
    void set_shuffle(bool val);




private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void init_poses(); //rad the pose json file and fills m_filename2pose
    void read_data(); //a scene (depending on the mode) and all the images contaned in it together with the poses and so on


    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    std::shared_ptr<easy_pbr::Mesh> m_scene_mesh;


    //params
    bool m_autostart;
    // std::atomic<bool> m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    int m_subsample_factor;
    std::string m_mode; // train or test or val
    bool m_shuffle;
    int m_limit_to_nr_imgs;
    bool m_do_overfit; // return all the time just the first image
    float m_scene_scale_multiplier; //multiplier the scene scale with this value so that we keep it in a range that we can expect
    // std::string m_restrict_to_object;  //makes it load clouds only from a specific object
    boost::filesystem::path m_dataset_path;  //get the path where all the off files are
    std::string m_object_name;
    // std::thread m_loader_thread;
    int m_nr_resets;
    int m_idx_img_to_read; //corresponds to the idx of the frame we will return since we have them all in memory


    //internal
    std::unordered_map<std::string, Eigen::Affine3d> m_filename2pose; //maps from the filename of the image to the corresponding pose in tf_world_cam
    std::unordered_map<std::string, Eigen::Matrix3d> m_filename2intrinsics; //maps from the filename of the image to the corresponding intrinsics
    std::vector<boost::filesystem::path> m_imgs_paths; //contains all the filenames that of the images we want to read
    std::vector< easy_pbr::Frame > m_frames;
    float m_camera_angle_x;
    float m_loaded_scene_mesh;

};
