#include <thread>
#include <unordered_map>
#include <vector>

//eigen 
#include <Eigen/Core>
#include<Eigen/StdVector>

//readerwriterqueue
#include "readerwriterqueue/readerwriterqueue.h"

//boost
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "easy_pbr/Frame.h"

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

namespace radu { namespace utils{
    class RandGenerator;
}}

struct PoseStanford3DScene{
    Eigen::Affine3d pose;
    int frame_idx;
};

class DataLoaderStanford3DScene
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderStanford3DScene(const std::string config_file);
    ~DataLoaderStanford3DScene();
    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    easy_pbr::Frame get_color_frame();
    easy_pbr::Frame get_depth_frame();
    bool has_data();
    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of samples/examples that this loader will iterate over

    // easy_pbr::Frame closest_color_frame(const easy_pbr::Frame& frame); //returns the frame color that is closest to the current one and also looks in a similar direction
    // easy_pbr::Frame closest_depth_frame(const easy_pbr::Frame& frame); //returns the frame depth that is closest to the current one and looks in a similar direction

private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void read_pose_file(std::string pose_file);
    Eigen::Matrix3d read_intrinsics_file(std::string intrinsics_file);
    void read_data();
    void read_sample(easy_pbr::Frame& frame_color, easy_pbr::Frame& frame_depth, const boost::filesystem::path& sample_filename); //reads one data sample

    //objects 
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;

    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    fs::path m_dataset_path; 
    fs::path m_pose_file_path; 
    int m_nr_samples_to_skip;
    int m_nr_samples_to_read;
    bool m_shuffle;
    bool m_do_overfit; // return all the time just one of the samples, specifically the first one
    std::thread m_loader_thread;
    uint32_t m_idx_sample_to_read;
    int m_nr_resets;
    int m_rgb_subsample_factor; //reduces the size of the color frames
    int m_depth_subsample_factor; //reduces the size of the depth frames


    //internal
    bool m_is_modified; //indicate that a cloud was finished processind and you are ready to get it 
    std::vector<fs::path> m_samples_filenames;
    moodycamel::ReaderWriterQueue<easy_pbr::Frame> m_frames_color_buffer;
    moodycamel::ReaderWriterQueue<easy_pbr::Frame> m_frames_depth_buffer;
    std::vector<PoseStanford3DScene> m_poses_vec;
    Eigen::Affine3d m_tf_worldGL_worldROS;
    Eigen::Matrix3d m_K;

};