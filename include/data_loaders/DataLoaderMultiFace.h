#pragma once

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

namespace easy_pbr{
    class Mesh;
}
// class DataTransformer;


struct GenesisCam{
    int cam_idx; //index of the camera which is something like cam400262
    std::vector< std::string > imgs_paths;
};


class DataLoaderMultiFace
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderMultiFace(const std::string config_file, const int subject_id);
    ~DataLoaderMultiFace();
    void start(); //starts reading the data from disk. This gets called automatically if we have autostart=true
    easy_pbr::Frame get_next_frame();
    std::vector<easy_pbr::Frame> get_all_frames();
    easy_pbr::Frame get_frame_at_idx( const int idx);
    easy_pbr::Frame get_frame_for_cam_id( const int cam_id);
    easy_pbr::Frame get_closest_frame( const easy_pbr::Frame& frame); //return the one closest frame
    std::vector<easy_pbr::Frame> get_close_frames( const easy_pbr::Frame& frame, const int nr_frames, const bool discard_same_idx ); //return a certain number of frames ordered by proximity,
    // std::vector<float> compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames);
    easy_pbr::Frame get_random_frame();
    std::shared_ptr<easy_pbr::Mesh>  get_mesh_head();
    // std::shared_ptr<easy_pbr::Mesh>  get_mesh_head_bald();
    // std::shared_ptr<easy_pbr::Mesh>  get_mesh_shoulders();
    // std::shared_ptr<easy_pbr::Mesh>  get_mesh_scalp();
    // std::shared_ptr<easy_pbr::Mesh>  get_mesh_hair();
    // std::shared_ptr<GenesisHair>   get_hair();
    std::vector<int> get_cam_indices();
    std::vector<int> get_cam_indices_lin();
    bool has_data(); //will reeturn always true because this dataloader preloads all the frames and keeps them in memory all the time. They are not so many
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of scenes for the object that we selected
    int nr_cameras();
    bool is_finished(); //check if we finished reading all the images from the scene
    void set_mode_train(); //set the loader so that it starts reading form the training set
    void set_mode_test();
    void set_mode_validation();
    void set_mode_all();
    std::string sequence();
    std::string dataset_path();
    std::string mesh_name_for_cur_timestep(); //return the mesh string like 017867 for this sequence and this timestep
    int subsample_factor();
    void set_subsample_factor(const int val);
    // std::shared_ptr<GenesisHair> get_random_roots(const int nr_strands);
    // Eigen::MatrixXd compute_uv_wrt_mesh( std::shared_ptr<easy_pbr::Mesh> mesh, Eigen::MatrixXd& points );
    std::shared_ptr<easy_pbr::Mesh> transform_mesh_from_mugsy_to_easypbr(std::shared_ptr<easy_pbr::Mesh> mesh);
    std::shared_ptr<easy_pbr::Mesh> transform_mesh_from_easypbr_to_mugsy(std::shared_ptr<easy_pbr::Mesh> mesh);
    // bool is_genesis();
    // bool is_minisis();
    // bool is_synthetic();

    // Eigen::Affine3f m_tf_frame_to_world_post;
    // Eigen::Affine3f m_tf_frame_to_world_pre;




private:

    void init_params(const std::string config_file, const int subject_id);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void init_poses(); //rad the pose json file and fills m_filename2pose
    // Eigen::Affine3f  init_transforms(); //initialize the transforms that transform from the Mugsy frame to my easypbr frame so that it looks good
    Eigen::Affine3f  init_transforms(); 
    Eigen::Affine3f transform_from_world_mugsy_to_world_easypbr(const Eigen::Affine3f& tf_world_obj, const bool do_scaling);
    void read_data(); //a scene (depending on the mode) and all the images contaned in it together with the poses and so on
    void load_images_in_frame(easy_pbr::Frame& frame);
    std::shared_ptr<easy_pbr::Mesh>  read_mesh(const std::string path, bool load_texture, bool transform, bool check_frame_nr);
    // std::shared_ptr<GenesisHair>  read_hair_recon(const std::string path_bin_file);
    // void compute_root_points_atributes(Eigen::MatrixXd& uv, std::vector<Eigen::Matrix3d>& tbn_per_point, std::shared_ptr<easy_pbr::Mesh> mesh, std::vector<Eigen::Vector3d> points_vec);



    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    // std::shared_ptr<DataTransformer> m_transformer;

    //params
    boost::filesystem::path m_dataset_path;
    //subject spefic params
    std::string m_subject_name;
    std::string m_sequence; //a string like A_good_morrow_to_you_my_boy
    // int m_frame_nr;
    int m_timestep;
    std::vector<int> m_indices_cameras_test;
    //rest of params
    bool m_autostart;
    int m_subsample_factor;
    // bool m_load_as_float;
    std::string m_mode; // train or test or val
    bool m_shuffle;
    bool m_load_as_shell;
    bool m_do_overfit; // return all the time just the first image
    float m_scene_rotate_x_angle;
    Eigen::Vector3f m_scene_translation; //moves the scene so that we have it at the origin more or less
    float m_scene_scale_multiplier; //multiplier the scene scale with this value so that we keep it in a range that we can expect



    //other
    int m_nr_resets;
    int m_idx_img_to_read; //corresponds to the idx of the frame we will return since we have them all in memory


    //internal
    std::unordered_map<int, Eigen::Affine3d> m_camidx2pose; //maps from the cam_idx of the image to the corresponding pose
    std::unordered_map<int, Eigen::Matrix3d> m_camidx2intrinsics; //maps from the cam_idx of the image to the corresponding K
    std::unordered_map<int, Eigen::VectorXd> m_camidx2distorsion; //maps from the cam_idx of the image to the corresponding distorsion params
    Eigen::Affine3f m_tf_world_easypbr_world_mugsy;
    std::vector<boost::filesystem::path> m_imgs_paths; //contains all the filenames that of the images we want to read
    std::vector< easy_pbr::Frame > m_frames;
    // std::vector< std::vector< std::string > > m_camidx_to_timestepidx_to_path;   //first index is the camera idx, second index is the time index of that camera, the value is the path
    std::vector< GenesisCam > m_cameras;
    std::vector< std::string > m_meshes_paths_for_timesteps;
    // std::vector< std::string > m_meshes_scalp_paths_for_timesteps;
    // std::vector< std::string > m_meshes_bald_paths_for_timesteps;
    // std::vector< std::string > m_meshes_shoulders_paths_for_timesteps;
    std::shared_ptr<easy_pbr::Mesh>  m_mesh_for_timestep;
    // std::shared_ptr<easy_pbr::Mesh>  m_mesh_head_bald_for_timestep;
    // std::shared_ptr<easy_pbr::Mesh>  m_mesh_shoulders_for_timestep;
    // std::shared_ptr<easy_pbr::Mesh>  m_mesh_scalp_for_timestep;
    // std::shared_ptr<easy_pbr::Mesh>  m_mesh_hair_for_timestep;
    // std::shared_ptr<GenesisHair>  m_hair_for_timestep;

};