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


class DataLoaderNerf
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderNerf(const std::string config_file);
    ~DataLoaderNerf();
    easy_pbr::Frame get_next_frame();
    bool has_data(); //will reeturn always true because this dataloader preloads all the frames and keeps them in memory all the time. They are not so many
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of scenes for the object that we selected
    bool is_finished(); //check if we finished reading all the images from the scene




private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void init_poses(); //rad the pose json file and fills m_filename2pose
    void read_data(); //a scene (depending on the mode) and all the images contaned in it together with the poses and so on
    

    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    // std::shared_ptr<DataTransformer> m_transformer;

    //params
    // bool m_autostart;
    // std::atomic<bool> m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    std::string m_mode; // train or test or val
    bool m_shuffle;
    bool m_do_overfit; // return all the time just the first image
    // std::string m_restrict_to_object;  //makes it load clouds only from a specific object
    boost::filesystem::path m_dataset_path;  //get the path where all the off files are 
    // std::thread m_loader_thread;
    int m_nr_resets;
    int m_idx_img_to_read; //corresponds to the idx of the frame we will return since we have them all in memory


    //internal
    std::unordered_map<std::string, Eigen::Affine3d> m_filename2pose; //maps from the filename of the image to the corresponding pose
    std::vector<boost::filesystem::path> m_imgs_paths; //contains all the filenames that of the images we want to read
    std::vector< easy_pbr::Frame > m_frames;
    float m_camera_angle_x;

};