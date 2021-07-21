#include <thread>
#include <unordered_map>
#include <vector>
#include <memory>

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

#include "torch/torch.h"

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


// Struct to contain everything we need for one hair sample
class USCHair : public std::enable_shared_from_this<USCHair> {
    public:
        std::shared_ptr<easy_pbr::Mesh> full_hair_cloud; //cloud containing the points of the hair
        std::vector< std::shared_ptr<easy_pbr::Mesh> > strand_meshes; //vector containing a mesh for each strand
        // Eigen::MatrixXd points; //Nx3 points of the hair
        torch::Tensor points_tensor; //nr_strands x nr_points_per_strand x 3
        Eigen::MatrixXi per_point_strand_idx; //Nx1 index of strand for each point. Points that belong to the same strand will have the same idx
        Eigen::MatrixXd uv_roots; // nr_strands x 2 uv for only the points on the roots
        torch::Tensor tbn_roots_tensor; //nr_strands x 3 x 3  tanent-bitangent-normal for the root points
        Eigen::MatrixXd position_roots; //nr_strands x 3 positions of the roots in world coords
        Eigen::MatrixXd strand_lengths; //nr_strands x 1 strand lengths
        Eigen::MatrixXd full_hair_cumulative_strand_length; //Nx 1  for each point on the hair store the cumulative lenght along it's corresponding strand
        torch::Tensor per_point_rotation_next_cur_tensor; // nr_strands X nr_points_per_strand x 3 of rodrigues towards the next point. Expressed in the local coordinate system of the current point
        torch::Tensor per_point_delta_dist_tensor; // nr_strands X nr_points_per_strand x 1  of delta movement applied to the average segment lenght. This is applied to the per_point_rotation_next_cur
        torch::Tensor per_point_direction_to_next_tensor; // nr_strands X nr_points_per_strand-1 x 3 direction in world coordinates from one point to the next one on the same strand
} ;


class DataLoaderUSCHair
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderUSCHair(const std::string config_file);
    ~DataLoaderUSCHair();
    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    std::shared_ptr<easy_pbr::Mesh> get_cloud();
    std::shared_ptr<USCHair> get_hair(); //return the whole usc struct of hair
    std::shared_ptr<easy_pbr::Mesh> get_mesh_head();
    std::shared_ptr<easy_pbr::Mesh> get_mesh_scalp();
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
    // std::tuple<
    //     std::vector< std::shared_ptr<easy_pbr::Mesh> >,
    //     std::shared_ptr<easy_pbr::Mesh>
    // > read_hair_sample(const std::string data_filepath); //returns a full hair mesha and also a vector of meshes corresponding with the strands
    std::shared_ptr<USCHair> read_hair_sample(const std::string data_filepath); //returns a full hair mesha and also a vector of meshes corresponding with the strands
    void compute_root_points_atributes(Eigen::MatrixXd& uv, std::vector<Eigen::Matrix3d>& tbn_per_point, std::shared_ptr<easy_pbr::Mesh> mesh, std::vector<Eigen::Vector3d> points_vec); //project the points onto the closest point on the mesh and get the uv from there

    //compute a local representation of the strands
    // strands_xyz : tensor of nr_strands x nr_points_per_strand x 3 of coordinates in world coordinates
    // strands_lengths: tensor of nr_strands x 1
    // tbn_roots nr_strands x 3 x 3 #Tangent-bitangent-normal for each point at the root of the strand in world coordinates
    // OUTPUT:
    // per_point_rotation_next_cur tensor of nr_strands X nr_points_per_strand x 3 of rodrigues towards the next point. Expressed in the local coordinate system of the current point
    // per_point_delta_dist tensor of nr_strands X nr_points_per_strand x 1  of delta movement applied to the average segment lenght. This is applied to the per_point_rotation_next_cur
    // per_point_direction_to_next  nr_strands X nr_points_per_strand-1 x 3 direction in world coordinates from one point to the next one on the same strand
    void xyz2local(int nr_strands, int nr_verts_per_strand, const Eigen::MatrixXd& points, const Eigen::MatrixXd& strand_lengths, std::vector<Eigen::Matrix3d>& tbn_roots, torch::Tensor& per_point_rotation_next_cur_tensor, torch::Tensor& per_point_delta_dist_tensor, torch::Tensor& per_point_direction_to_next_tensor);


    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;

    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    std::string m_mode; // train or test or val
    fs::path m_dataset_path;
    fs::path m_scalp_mesh_path;
    int m_nr_clouds_to_skip;
    int m_nr_clouds_to_read;
    float m_percentage_strand_drop;
    int m_load_only_strand_with_idx;
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
    // moodycamel::ReaderWriterQueue<std::shared_ptr<easy_pbr::Mesh> > m_clouds_buffer;
    // std::vector<std::shared_ptr<easy_pbr::Mesh> > m_clouds_vec;
    moodycamel::ReaderWriterQueue<std::shared_ptr<USCHair > > m_hairs_buffer;
    std::vector<std::shared_ptr<USCHair> > m_hairs_vec;
    std::shared_ptr<easy_pbr::Mesh>  m_mesh_head;
    std::shared_ptr<easy_pbr::Mesh>  m_mesh_scalp;


};
