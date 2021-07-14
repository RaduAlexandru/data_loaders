#include <thread>
#include <unordered_map>
#include <vector>

//ros
// #include <ros/ros.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <pcl_ros/point_cloud.h>

//eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

//readerwriterqueue
#include "readerwriterqueue/readerwriterqueue.h"

//boost
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;


namespace radu { namespace utils{
    class RandGenerator;
}}

namespace easy_pbr{
    class LabelMngr;
    class Mesh;
}
class DataTransformer;


class DataLoaderUSCHair
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderUSCHair(const std::string config_file);
    ~DataLoaderUSCHair();
    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    std::shared_ptr<easy_pbr::Mesh> get_cloud();
    bool has_data();
    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of samples/examples that this loader will iterate over
    void set_mode_train(); //set the loader so that it starts reading form the training set
    void set_mode_test();
    void set_mode_validation();


private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  >read_pose_file(std::string m_pose_file);
    void read_data();
    std::tuple<
        std::vector< std::shared_ptr<easy_pbr::Mesh> >,
        std::shared_ptr<easy_pbr::Mesh>
    > read_hair_sample(const std::string data_filepath); //returns a full hair mesha and also a vector of meshes corresponding with the strands


    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;

    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    std::string m_mode; // train or test or val
    fs::path m_dataset_path;
    int m_nr_clouds_to_skip;
    int m_nr_clouds_to_read;
    bool m_shuffle;
    bool m_do_overfit; // return all the time just one of the clouds, specifically the first one
    // bool m_do_adaptive_subsampling; //randomly drops points from the cloud, dropping with more probability the ones that are closes and with less the ones further
    std::thread m_loader_thread;
    uint32_t m_idx_cloud_to_read;
    int m_nr_resets;
    bool m_load_buffered; //if true, we start another thread an load clouds in a rinbuffer. If false, we just load everything in memory


    //internal
    bool m_is_modified; //indicate that a cloud was finished processind and you are ready to get it
    int m_nr_sequences;
    std::vector<fs::path> m_data_filenames;
    moodycamel::ReaderWriterQueue<std::shared_ptr<easy_pbr::Mesh> > m_clouds_buffer;
    std::vector<std::shared_ptr<easy_pbr::Mesh> > m_clouds_vec;

};
