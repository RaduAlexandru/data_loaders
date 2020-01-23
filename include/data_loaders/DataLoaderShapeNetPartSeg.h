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


#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

class LabelMngr;
class RandGenerator;
class DataTransformer;
class Mesh;

class DataLoaderShapeNetPartSeg
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderShapeNetPartSeg(const std::string config_file);
    ~DataLoaderShapeNetPartSeg();
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
    std::string get_object_name();
    void set_object_name(const std::string object_name);




private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void read_data();
    Eigen::MatrixXd read_pts(const std::string file_path);
    Eigen::MatrixXi read_labels(const std::string file_path);
    std::unordered_map<std::string, std::string> read_mapping_synsetoffset2category(const std::string file_path);
    void create_transformation_matrices();
    // void apply_transform(Eigen::MatrixXd& V, const Eigen::Affine3d& trans);
    // void compute_normals(Eigen::MatrixXd& NV, const Eigen::MatrixXd& V);

    //objects
    std::shared_ptr<RandGenerator> m_rand_gen;
    std::shared_ptr<DataTransformer> m_transformer;

    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    std::string m_mode; // train or test or val
    bool m_shuffle_points; //When splatting in a permutohedral lattice it's better to have adyancent point in 3D be in different parts in memoru to aboid hashing conflicts
    bool m_normalize; //normalizes the point cloud between [-1,1]
    bool m_shuffle;
    bool m_do_overfit; // return all the time just one of the clouds, specifically the first one
    std::string m_restrict_to_object;  //makes it load clouds only from a specific object
    boost::filesystem::path m_dataset_path;  //get the path where all the off files are 
    // int m_nr_clouds_to_skip;
    // int m_nr_clouds_to_read;
    std::thread m_loader_thread;
    uint32_t m_idx_cloud_to_read;
    int m_nr_resets;
    // std::string m_pose_file;
    // std::string m_pose_file_format;


    //internal
    std::vector<boost::filesystem::path> m_pts_filenames; //contains all the pts filenames from all the classes
    std::vector<boost::filesystem::path> m_labels_filenames; //contains all the labels for the correspinding pts files
    // std::unordered_map<std::string, std::string> m_synsetoffset2category; //mapping from the filename which a bunch of number to the class name;
    moodycamel::ReaderWriterQueue<std::shared_ptr<Mesh> > m_clouds_buffer;
    Eigen::Affine3d m_tf_worldGL_worldROS;

    //label mngr to link to all the meshes that will have a semantic information
    std::shared_ptr<LabelMngr> m_label_mngr;

};