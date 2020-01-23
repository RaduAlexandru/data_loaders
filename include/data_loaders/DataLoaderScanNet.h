#include <thread>
#include <unordered_map>
#include <vector>

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


#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

class LabelMngr;
class RandGenerator;
class DataTransformer;
class Mesh;

class DataLoaderScanNet
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderScanNet(const std::string config_file);
    ~DataLoaderScanNet();
    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    std::shared_ptr<Mesh> get_cloud();
    bool has_data();
    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of samples/examples that this loader will iterate over
    std::shared_ptr<LabelMngr> label_mngr();
    void set_mode_train(); //set the loader so that it starts reading form the training set
    void set_mode_test();
    void set_mode_validation();
    void write_for_evaluating_on_scannet_server(std::shared_ptr<Mesh>& cloud, const std::string path_for_eval); //the test set need to be evaluated on the their server so we write it in the format they want

private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void read_data();
    Eigen::MatrixXi read_labels(const std::string labels_file); //the labels of the point cloud are stored in a separate ply file. We read it the same way as the ReadPLY.cpp in libigl.
    Eigen::Affine3d read_alignment_matrix(const std::string alignment_file); //scannet provides and alignment files as a 4x4 matrix stored in row major that aligns the walls and so on
    // std::unordered_map<std::string, bool>  read_data_split(const std::string data_split_file);
    void create_transformation_matrices();

    //objects 
    std::shared_ptr<RandGenerator> m_rand_gen;
    std::shared_ptr<DataTransformer> m_transformer;

    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    std::string m_mode; // train or test or val
    fs::path m_dataset_path; 
    int m_nr_clouds_to_skip;
    int m_nr_clouds_to_read;
    int m_max_nr_points_per_cloud;
    bool m_shuffle_points; //When splatting in a permutohedral lattice it's better to have adyancent point in 3D be in different parts in memoru to aboid hashing conflicts
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
    std::vector<fs::path> m_ply_filenames;
    std::unordered_map<std::string, bool> m_files_train; 
    std::unordered_map<std::string, bool> m_files_test; 
    std::unordered_map<std::string, bool> m_files_validation; 
    moodycamel::ReaderWriterQueue<std::shared_ptr<Mesh> > m_clouds_buffer;
    // std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  >m_worldROS_cam_vec; //actually the semantic kitti expressed the clouds in the left camera coordinate so it should be m_worldRos_cam_vec 
    Eigen::Affine3d m_tf_worldGL_worldROS;

    //label mngr to link to all the meshes that will have a semantic information
    std::shared_ptr<LabelMngr> m_label_mngr;

    //for sanity checking
    int m_min_label_written;
    int m_max_label_written;

};