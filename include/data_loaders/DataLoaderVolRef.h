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


namespace radu { namespace utils{
    class RandGenerator;
}}

class DataLoaderVolRef
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderVolRef(const std::string config_file);
    ~DataLoaderVolRef();
    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    easy_pbr::Frame get_color_frame();
    easy_pbr::Frame get_depth_frame();
    easy_pbr::Frame get_frame_at_idx( const int idx); //convenience function so that it has the same API as the other loaders. WARNING returns only the color frame
    easy_pbr::Frame get_depth_frame_at_idx( const int idx); //convenience function so that it has the same API as the other loaders. WARNING returns only the color frame
    bool has_data();
    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of samples/examples that this loader will iterate over

    easy_pbr::Frame closest_color_frame(const easy_pbr::Frame& frame); //returns the frame color that is closest to the current one and also looks in a similar direction
    easy_pbr::Frame closest_depth_frame(const easy_pbr::Frame& frame); //returns the frame depth that is closest to the current one and looks in a similar direction


    void load_only_from_idxs(const Eigen::VectorXi& vec); //Set a vector of ints and we will load only color and depth frames from those
    void set_shuffle(bool val);
    void set_overfit(bool val);

private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    Eigen::Affine3d read_pose_file(std::string pose_file);
    Eigen::Matrix3d read_intrinsics_file(std::string intrinsics_file);
    void read_data();
    void read_sample(easy_pbr::Frame& frame_color, easy_pbr::Frame& frame_depth, const boost::filesystem::path& sample_filename); //reads one data sample

    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;

    //params
    bool m_autostart;
    bool m_preload;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    fs::path m_dataset_path;
    bool m_load_rgb_with_valid_depth;
    int m_nr_samples_to_skip;
    int m_nr_samples_to_read;
    bool m_shuffle;
    bool m_do_overfit; // return all the time just one of the samples, specifically the first one
    std::thread m_loader_thread;
    uint32_t m_idx_sample_to_read;
    int m_nr_resets;
    int m_rgb_subsample_factor; //reduces the size of the color frames
    int m_depth_subsample_factor; //reduces the size of the depth frames
    Eigen::Vector3f m_scene_translation; //moves the scene so that we have it at the origin more or less
    float m_scene_scale_multiplier; //multiplier the scene scale with this value so that we keep it in a range that we can expect


    //internal
    bool m_is_modified; //indicate that a cloud was finished processind and you are ready to get it
    std::vector<fs::path> m_samples_filenames;
    moodycamel::ReaderWriterQueue<easy_pbr::Frame> m_frames_color_buffer;
    moodycamel::ReaderWriterQueue<easy_pbr::Frame> m_frames_depth_buffer;
    std::vector<easy_pbr::Frame> m_frames_color_vec;
    std::vector<easy_pbr::Frame> m_frames_depth_vec;
    Eigen::Affine3d m_tf_worldGL_worldROS;
    Eigen::MatrixXd m_K_color;
    Eigen::MatrixXd m_K_depth;
    Eigen::VectorXi m_load_from_idxs;
    uint32_t m_idx_colorframe_to_return;
    uint32_t m_idx_depthframe_to_return;

};
