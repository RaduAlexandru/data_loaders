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

class RandGenerator;

class DataLoaderVolRef
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderVolRef(const std::string config_file);
    ~DataLoaderVolRef();
    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    Frame get_color_frame();
    Frame get_depth_frame();
    bool has_data();
    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of samples/examples that this loader will iterate over

    Frame closest_color_frame(const Frame& frame); //returns the frame color that is closest to the current one and also looks in a similar direction
    Frame closest_depth_frame(const Frame& frame); //returns the frame depth that is closest to the current one and looks in a similar direction

private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    Eigen::Affine3d read_pose_file(std::string pose_file);
    Eigen::Matrix3d read_intrinsics_file(std::string intrinsics_file);
    void read_data();
    void read_sample(Frame& frame_color, Frame& frame_depth, const boost::filesystem::path& sample_filename); //reads one data sample

    //objects 
    std::shared_ptr<RandGenerator> m_rand_gen;

    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    fs::path m_dataset_path; 
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
    moodycamel::ReaderWriterQueue<Frame> m_frames_color_buffer;
    moodycamel::ReaderWriterQueue<Frame> m_frames_depth_buffer;
    Eigen::Affine3d m_tf_worldGL_worldROS;
    Eigen::MatrixXd m_K_color;
    Eigen::MatrixXd m_K_depth;

};