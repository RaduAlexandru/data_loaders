#pragma once


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


class DataLoaderPheno4D
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderPheno4D(const std::string config_file);
    ~DataLoaderPheno4D();
    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    std::shared_ptr<easy_pbr::Mesh> get_cloud();
    std::shared_ptr<easy_pbr::Mesh> get_cloud_with_idx(const int idx);
    bool has_data();
    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of samples/examples that this loader will iterate over
    std::shared_ptr<easy_pbr::LabelMngr> label_mngr();
    void set_day(const std::string day_format); // Set a concrete day from which we read The format of the string is something like 0325 in which the first two characters is the month and the last 2 is the day
    void set_plant_nr(const int nr);
    void set_nr_plants_to_skip(const int new_val);
    void set_nr_plants_to_read(const int new_val);
    void set_nr_days_to_skip(const int new_val);
    void set_nr_days_to_read(const int new_val);
    void set_do_augmentation(const bool val);
    void set_segmentation_method(const std::string method);
    void set_preload(const bool val); //if we preload, then we read the meshes only once and store them in memory

    //TODO



private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void read_data();
    std::shared_ptr<easy_pbr::Mesh> read_sample(const fs::path sample_filename); //reads one data sample

    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    std::shared_ptr<DataTransformer> m_transformer;

    //params
    fs::path m_dataset_path;
    bool m_autostart;
    //plant
    std::string m_plant_type; //maize or tomato
    std::string m_segmentation_method; //leaf_collar or leaf_tip (only valid if we are loading maize)
    //which plants to read
    int m_nr_plants_to_skip;
    int m_nr_plants_to_read; //how many plants of the selected type we should read, set to -1 to read all plants
    int m_selected_plant_nr;
    //which days to read
    int m_nr_days_to_skip;
    int m_nr_days_to_read; //how many days to read for the selected plants, set to -1 to read all days
    std::string m_selected_day; //To read one concrete single day, day for eg can be 0325 which is march 25 from which we will read
    //params for after reading
    bool m_shuffle_points;
    bool m_normalize;
    bool m_shuffle_days;
    bool m_do_overfit; //return only one of the samples the whole time, concretely the first sample in the dataset
    bool m_do_augmentation;
    bool m_preload;




    //internal
    std::thread m_loader_thread;
    uint32_t m_idx_cloud_to_read;
    uint32_t m_idx_cloud_to_return;
    int m_nr_resets;
    bool m_is_modified; //indicate that a cloud was finished processind and you are ready to get it
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    int m_nr_sequences;
    std::vector<fs::path> m_sample_filenames;
    moodycamel::ReaderWriterQueue<std::shared_ptr<easy_pbr::Mesh> > m_clouds_buffer;
    std::vector< std::shared_ptr<easy_pbr::Mesh>  > m_clouds_vec;
    // std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  >m_worldROS_cam_vec; //actually the semantic kitti expressed the clouds in the left camera coordinate so it should be m_worldRos_cam_vec

    //label mngr to link to all the meshes that will have a semantic information
    std::shared_ptr<easy_pbr::LabelMngr> m_label_mngr;

};
