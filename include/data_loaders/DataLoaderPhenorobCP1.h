#include <thread>
#include <unordered_map>
#include <vector>
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


//better enums
#include <enum.h>

BETTER_ENUM(PHCP1DatasetType, int, Raw = 0, Processed )




namespace radu { namespace utils{
    class RandGenerator;
}}

// namespace easy_pbr{
//     class Frame;
// }
// class DataTransformer;



// Struct to contain from one block of cameras that are triggered syncronously
class PRCP1Block : public std::enable_shared_from_this<PRCP1Block> {
    public:
        int nr_frames();
        std::shared_ptr<easy_pbr::Frame> get_rgb_frame_with_idx( const int idx);
        easy_pbr::Frame get_photoneo_frame(){ return m_photoneo_frame; };
        std::shared_ptr<easy_pbr::Mesh> get_photoneo_mesh(){ return m_photoneo_mesh; };
        std::string name(){ return m_name;};

        easy_pbr::Frame m_photoneo_frame; 
        std::shared_ptr<easy_pbr::Mesh> m_photoneo_mesh; 
        // std::vector<easy_pbr::Frame> m_rgb_frames; 
        std::unordered_map<int, std::shared_ptr<easy_pbr::Frame> > m_rgb_frames; //the idx is the cam_id
        std::string m_photoneo_cfg_file_path; //is the path of the cfg file of the photoneo which contains the intrinsics and so one
        std::string m_name;
        boost::filesystem::path m_path;
} ;
//class that contains a full scan of a plant, so all N blocks
class PRCP1Scan : public std::enable_shared_from_this<PRCP1Scan> {
    public:
        int nr_blocks(); //returns the number of scenes for the object that we selected
        std::string name(){ return m_name;};
        std::shared_ptr<PRCP1Block> get_block_with_idx(const int idx);

        std::vector<  std::shared_ptr<PRCP1Block>  > m_blocks;
        std::string m_name;
        boost::filesystem::path m_path;
};





class DataLoaderPhenorobCP1
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderPhenorobCP1(const std::string config_file);
    ~DataLoaderPhenorobCP1();
    void start(); //starts reading the data from disk. This gets called automatically if we have autostart=true
    std::shared_ptr<PRCP1Scan> get_scan_with_idx(const int idx);
    // std::vector<easy_pbr::Frame> get_all_frames();
    // easy_pbr::Frame get_frame_at_idx( const int idx);
    // easy_pbr::Frame get_closest_frame( const easy_pbr::Frame& frame); //return the one closest frame
    // std::vector<easy_pbr::Frame> get_close_frames( const easy_pbr::Frame& frame, const int nr_frames, const bool discard_same_idx ); //return a certain number of frames ordered by proximity,
    // std::vector<float> compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames);
    // easy_pbr::Frame get_random_frame();
    bool has_data(); //will reeturn always true because this dataloader preloads all the frames and keeps them in memory all the time. They are not so many
    void reset(); //starts reading from the beggining
    int nr_scans(); //returns the number of scenes for the object that we selected
    std::string dataset_path();
    std::string scan_date();
    std::string rgb_pose_file();
    bool is_finished(); //check if we finished reading all the images from the scene
    void set_mode_train(); //set the loader so that it starts reading form the training set
    void set_mode_test();
    void set_mode_validation();
    void set_mode_all();




private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void init_poses_kalibr(); //rad the pose json file for poses and sets the poses for the nikons and photoneos
    void init_intrinsics_and_poses_krt(); //init poses and intrinsics from a krt format and set them for all the nikons
    void init_intrinsics_kalibr(); //rad the pose json file and sets the intrinsics for the nikon 
    void init_stereo_pairs();
    void read_data(); //a scene (depending on the mode) and all the images contaned in it together with the poses and so on
    void load_images_in_frame(easy_pbr::Frame& frame);


    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    // std::shared_ptr<DataTransformer> m_transformer;

    //params
    bool m_autostart;
    // std::atomic<bool> m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    int m_rgb_subsample_factor;
    int m_photoneo_subsample_factor;
    // float m_exposure_change;
    // bool m_load_as_float;
    bool m_transform_to_easypbr_world; //if true then the extrinsics will be transformed to the easypbr origin
    float m_rotation_alignment_degrees; //rotate by this in order to align with wour world coordinate
    std::string m_mode; // train or test or val
    bool m_shuffle;
    bool m_load_as_shell;
    bool m_do_overfit; // return all the time just the first image
    float m_scene_scale_multiplier; //multiplier the scene scale with this value so that we keep it in a range that we can expect
    boost::filesystem::path m_dataset_path;  //get the path where all the the scans are
    boost::filesystem::path m_scan_date; //the date of the scan
    // std::string m_dataset_type;
    PHCP1DatasetType m_dataset_type=PHCP1DatasetType::Raw;
    bool m_load_poses;
    bool m_load_intrinsics;
    // int m_scan_idx;  //the idx of the scan that was made on a certain date
    // std::thread m_loader_thread;
    int m_nr_resets;
    int m_idx_img_to_read; //corresponds to the idx of the frame we will return since we have them all in memory


    //internal
    std::unordered_map<int, Eigen::Affine3d> m_camidx2pose; //maps from the cam_idx of the image to the corresponding pose
    std::unordered_map<int, Eigen::Vector2i> m_camidx2resolution;
    // std::unordered_map<int, Eigen::Matrix3d> m_camidx2intrinsics; //maps from the cam_idx of the image to the corresponding K
    // std::unordered_map<int, Eigen::VectorXd> m_camidx2distorsion; //maps from the cam_idx of the image to the corresponding distorsion params
    // std::vector<boost::filesystem::path> m_imgs_paths; //contains all the filenames that of the images we want to read
    // std::vector<  std::shared_ptr<PRCP1Block>  > m_blocks;
    std::vector<  std::shared_ptr<PRCP1Scan>  > m_scans;
    std::string m_rgb_pose_file;
    std::unordered_map<int, int> m_stereo_pairs; // two indicex for the left and right pairs, if the right pair doesnt exist, then it is -1

};
