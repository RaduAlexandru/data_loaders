#include <thread>
#include <unordered_map>
#include <vector>



//eigen 
#include <Eigen/Core>

//readerwriterqueue
#include "readerwriterqueue/readerwriterqueue.h"

//boost
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "data_loaders/core/MeshCore.h"

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items


class DataLoaderModelNet40
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderModelNet40(const std::string config_file);
    ~DataLoaderModelNet40();
    MeshCore get_cloud();
    bool has_data();


private:

    void init_params(const std::string config_file);
    void read_data();
    void create_transformation_matrices();
    // void apply_transform(Eigen::MatrixXd& V, const Eigen::Affine3d& trans);
    // void compute_normals(Eigen::MatrixXd& NV, const Eigen::MatrixXd& V);


    //params
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    std::string m_mode; // train or test
    bool m_normalize; //normalizes the point cloud between [-1,1]
    // int m_nr_clouds_to_skip;
    // int m_nr_clouds_to_read;
    std::thread m_loader_thread;
    uint32_t m_idx_cloud_to_read;
    // std::string m_pose_file;
    // std::string m_pose_file_format;


    //internal
    std::vector<fs::path> m_off_filenames; //contains all the off filenames from all the classes in alphabetical order
    moodycamel::ReaderWriterQueue<MeshCore> m_clouds_buffer;
    Eigen::Affine3d m_tf_worldGL_worldROS;

};