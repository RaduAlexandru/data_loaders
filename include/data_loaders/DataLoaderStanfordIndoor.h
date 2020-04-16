// this is used to loader the S3DIS (Stanford 3D Indoor Spaces) dataset http://buildingparser.stanford.edu/method.html

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

#include "data_loaders/core/MeshCore.h"

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

namespace radu { namespace utils{
    class RandGenerator;
}}

namespace easy_pbr{
    class LabelMngr;
}
class DataTransformer;


class DataLoaderStanfordIndoor
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderStanfordIndoor(const std::string config_file);
    ~DataLoaderStanfordIndoor();
    void start(); //starts the thread that reads the data from disk. This gets called automatically if we have autostart=true
    MeshCore get_cloud();
    bool has_data();
    bool is_finished(); //returns true when we have finished reading AND processing everything
    bool is_finished_reading(); //returns true when we have finished reading everything but maybe not processing
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of samples/examples that this loader will iterate over
    void set_mode_train(); //set the loader so that it starts reading form the training set
    void set_mode_test();
    void set_mode_validation();

    // void set_sequence(const std::string sequence);

private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void read_data_and_reparse();
    void read_data();
    bool should_read_area(const int area_number); //depending on the mode (train or test) and the the m_fold we may need to read or not one of the 6 areas

    //objects 
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    std::shared_ptr<DataTransformer> m_transformer;

    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    std::string m_mode; // train or test or val
    int m_fold; //The dataset is divided in 6 areas, the fold number indicates which area we use for training and which for testing. Explained here http://buildingparser.stanford.edu/dataset.html
    fs::path m_dataset_path; 
    fs::path m_sequence; 
    int m_nr_clouds_to_skip;
    int m_nr_clouds_to_read;
    int m_max_nr_points_per_cloud;
    bool m_shuffle_points; //When splatting in a permutohedral lattice it's better to have adyancent point in 3D be in different parts in memoru to aboid hashing conflicts
    bool m_shuffle;
    bool m_do_overfit; // return all the time just one of the clouds, specifically the first one
    std::thread m_loader_thread;
    uint32_t m_idx_cloud_to_read;
    int m_nr_resets;
    bool m_read_original_data_and_reparse; //if we are reparsing the data we are only reading the original files whcih are stored in horrible slow ASCII and store them in a binary of floats per room.


    //internal
    bool m_is_modified; //indicate that a cloud was finished processind and you are ready to get it 
    std::vector<fs::path> m_room_paths;
    moodycamel::ReaderWriterQueue<MeshCore> m_clouds_buffer;

    //label mngr to link to all the meshes that will have a semantic information
    std::shared_ptr<LabelMngr> m_label_mngr;

};