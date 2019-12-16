#include <thread>
#include <unordered_map>
#include <vector>



//eigen 
#include <Eigen/Core>
#include <Eigen/StdVector>

//readerwriterqueue
#include "readerwriterqueue/readerwriterqueue.h"

//boost
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "data_loaders/core/MeshCore.h"

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

class LabelMngr;

class DataLoaderToyExample
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderToyExample(const std::string config_file);
    ~DataLoaderToyExample();
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



private:

    void init_params(const std::string config_file);
    void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void read_data();
    MeshCore sine_wave_3D(const int nr_points, const int nr_cycles, const float wave_amplitude);


    //params
    bool m_autostart;
    bool m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    int m_sine_nr_points;
    int m_sine_nr_cycles;
    float m_sine_amplitude;

    std::thread m_loader_thread;


    //internal
    moodycamel::ReaderWriterQueue<MeshCore> m_clouds_buffer;

    //label mngr to link to all the meshes that will have a semantic information
    std::shared_ptr<LabelMngr> m_label_mngr;

};