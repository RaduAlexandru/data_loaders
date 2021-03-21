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



namespace radu { namespace utils{
    class RandGenerator;
}}

// namespace easy_pbr{
//     class Frame;
// }
// class DataTransformer;


class DataLoaderColmap
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataLoaderColmap(const std::string config_file);
    ~DataLoaderColmap();
    void start(); //starts reading the data from disk. This gets called automatically if we have autostart=true
    easy_pbr::Frame get_next_frame();
    easy_pbr::Frame get_frame_at_idx( const int idx);
    easy_pbr::Frame get_closest_frame( const easy_pbr::Frame& frame);
    std::vector<easy_pbr::Frame> get_close_frames( const easy_pbr::Frame& frame, const int nr_frames, const bool discard_same_idx ); //return a certain number of frames ordered by proximity, 
    easy_pbr::Frame get_random_frame();
    bool has_data(); //will reeturn always true because this dataloader preloads all the frames and keeps them in memory all the time. They are not so many
    void reset(); //starts reading from the beggining
    int nr_samples(); //returns the number of scenes for the object that we selected
    bool is_finished(); //check if we finished reading all the images from the scene

    void set_mode_train(); //set the loader so that it starts reading form the training set
    void set_mode_test();
    void set_mode_validation();
    void set_mode_all();




private:

    void init_params(const std::string config_file);
    // void init_data_reading(); //after the parameters this uses the params to initiate all the structures needed for the susequent read_data
    void init_extrinsics_and_intrinsics(); //rad the pose json file and fills m_filename2pose
    void read_data(); //a scene (depending on the mode) and all the images contaned in it together with the poses and so on

    // Read data in little endian format for cross-platform support. From colmap github src/util/endian.h
    template <typename T>
    T ReadBinaryLittleEndian(std::istream* stream);
    template <typename T>
    void ReadBinaryLittleEndian(std::istream* stream, std::vector<T>* data);
    // Convert data between endianness and the native format. Note that, for float
    // and double types, these functions are only valid if the format is IEEE-754.
    // This is the case for pretty much most processors.
    template <typename T>
    T LittleEndianToNative(const T x);
    template <typename T>
    T BigEndianToNative(const T x);
    template <typename T>
    T NativeToLittleEndian(const T x);
    template <typename T>
    T NativeToBigEndian(const T x);
    // Reverse the order of each byte.
    template <typename T>
    T ReverseBytes(const T& data);
    // Check the order in which bytes are stored in computer memory.
    bool IsLittleEndian();
    bool IsBigEndian();

        

    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;
    // std::shared_ptr<DataTransformer> m_transformer;

    //params
    bool m_autostart;
    // std::atomic<bool> m_is_running;// if the loop of loading is running, it is used to break the loop when the user ctrl-c
    int m_subsample_factor;
    std::string m_mode; // train or test or val
    bool m_shuffle;
    bool m_do_overfit; // return all the time just the first image
    float m_scene_scale_multiplier; //multiplier the scene scale with this value so that we keep it in a range that we can expect
    bool m_load_imgs_with_transparency; //loads images with transparency and therefore also loads the frame.mask or without which is faster and only loads the rgb part of the frame
    // std::string m_restrict_to_object;  //makes it load clouds only from a specific object
    boost::filesystem::path m_dataset_path;  //get the path where all the off files are 
    // std::thread m_loader_thread;
    int m_nr_resets;
    int m_idx_img_to_read; //corresponds to the idx of the frame we will return since we have them all in memory


    //internal
    std::unordered_map<std::string, Eigen::Affine3d> m_filename2pose; //maps from the filename of the image to the corresponding pose
    std::vector<boost::filesystem::path> m_imgs_paths; //contains all the filenames that of the images we want to read
    std::vector< easy_pbr::Frame > m_frames;

};