#include <thread>
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


class DataLoaderPhenorob
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderPhenorob(const std::string config_file);
    ~DataLoaderPhenorob();
    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    std::shared_ptr<easy_pbr::Mesh> get_cloud();
    bool has_data();
    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of samples/examples that this loader will iterate over
    std::shared_ptr<easy_pbr::LabelMngr> label_mngr();
    void set_nr_clouds_to_skip(const int new_val);
    void set_nr_clouds_to_read(const int new_val);


private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void read_data();

    //objects 
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    std::shared_ptr<DataTransformer> m_transformer;

    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    std::string m_mode; // train or test or val
    fs::path m_dataset_path; 
    int m_nr_clouds_to_skip;
    int m_nr_clouds_to_read;
    bool m_shuffle_points; //When splatting in a permutohedral lattice it's better to have adyancent point in 3D be in different parts in memoru to aboid hashing conflicts
    bool m_normalize; //normalizes the point cloud between [-1,1]
    bool m_shuffle;
    bool m_do_overfit; // return all the time just one of the clouds, specifically the first one
    // bool m_do_adaptive_subsampling; //randomly drops points from the cloud, dropping with more probability the ones that are closes and with less the ones further
    std::thread m_loader_thread;
    uint32_t m_idx_cloud_to_read;
    int m_nr_resets;
    // std::string m_pose_file;
    // std::string m_pose_file_format;


    //internal
    bool m_is_modified; //indicate that a cloud was finished processind and you are ready to get it 
    int m_nr_sequences;
    std::vector<fs::path> m_sample_filenames;
    moodycamel::ReaderWriterQueue<std::shared_ptr<easy_pbr::Mesh> > m_clouds_buffer;
    // std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  >m_worldROS_cam_vec; //actually the semantic kitti expressed the clouds in the left camera coordinate so it should be m_worldRos_cam_vec 

    //label mngr to link to all the meshes that will have a semantic information
    std::shared_ptr<easy_pbr::LabelMngr> m_label_mngr;

};