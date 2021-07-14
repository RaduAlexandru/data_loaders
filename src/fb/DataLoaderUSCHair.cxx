#include "data_loaders/fb/DataLoaderUSCHair.h"

//c++
#include <algorithm>
#include <random>

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//cnpy
// #include "cnpy.h"

//boost
#include <boost/range/iterator_range.hpp>

//my stuff
#include "easy_pbr/Mesh.h"
// #include "easy_pbr/LabelMngr.h"
// #include "data_loaders/DataTransformer.h"
#include "Profiler.h"
#include "string_utils.h"
#include "eigen_utils.h"
#include "RandGenerator.h"

using namespace radu::utils;
using namespace easy_pbr;

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

DataLoaderUSCHair::DataLoaderUSCHair(const std::string config_file):
    m_is_modified(false),
    m_is_running(false),
    m_clouds_buffer(BUFFER_SIZE),
    m_idx_cloud_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator)
{

    init_params(config_file);
    // read_pose_file();
    // create_transformation_matrices();
    // std::cout << " creating thread" << "\n";
    if(m_autostart){
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderUSCHair::read_data, this);  //starts the spin in another thread
    }
    // std::cout << " finidhed creating thread" << "\n";

}

DataLoaderUSCHair::~DataLoaderUSCHair(){

    // std::cout << "finishing" << std::endl;
    m_is_running=false;

    m_loader_thread.join();
}

void DataLoaderUSCHair::init_params(const std::string config_file){
    //get the config filename
    // ros::NodeHandle private_nh("~");
    // std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    // std::string config_file="config.cfg";

    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_semantic_kitti"];

    m_autostart=loader_config["autostart"];
    m_mode=(std::string)loader_config["mode"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    // m_do_adaptive_subsampling=loader_config["do_adaptive_subsampling"];
    m_dataset_path=(std::string)loader_config["dataset_path"];

}

void DataLoaderUSCHair::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";

    init_data_reading();

    m_is_running=true;
    m_loader_thread=std::thread(&DataLoaderUSCHair::read_data, this);  //starts the spin in another thread
}

void DataLoaderUSCHair::init_data_reading(){

    std::vector<fs::path> data_filenames_all;
    // if(m_sequence!="all"){
    //     m_nr_sequences=1; //we usually get only one sequence, unless m_sequence is set to "all"
    //     fs::path full_path= m_dataset_path/m_mode/m_sequence;

    //     if(!fs::is_directory(full_path)) {
    //         LOG(FATAL) << "No directory " << full_path;
    //     }

    //     //see how many images we have and read the files paths into a vector
    //     for (fs::directory_iterator itr(full_path); itr!=fs::directory_iterator(); ++itr){
    //         //all the files in the folder might include also the pose file so we ignore that one
    //         //we also ignore the files that contain intensity, for now we only read the general ones and then afterwards we append _i to the file and read the intensity if neccesarry
    //         if( !(itr->path().stem()=="poses")  &&  itr->path().stem().string().find("_i")== std::string::npos ){
    //             npz_filenames_all.push_back(itr->path());
    //         }
    //     }
    //     if(!m_shuffle){ //if we are shuffling, there is no need to sort them
    //         std::sort(npz_filenames_all.begin(), npz_filenames_all.end());
    //     }



	// if(m_do_pose){
	//     //pose file
    //         std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  > poses;
    //         fs::path pose_file = full_path/"poses.txt";
    //         poses=read_pose_file( pose_file.string() );
    //         m_poses_per_sequence[m_sequence.string()] = poses;
	// }

    // }else if(m_sequence=="all"){
    //     //iterate thrugh all the sequences and load all of them

    //     //get how many sequnces we have here
    //     fs::path dataset_path_with_mode= m_dataset_path/m_mode;
    //     if(!fs::is_directory(dataset_path_with_mode)) {
    //         LOG(FATAL) << "No directory " << dataset_path_with_mode;
    //     }
    //     m_nr_sequences=0;
    //     for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(dataset_path_with_mode), {})){
    //         if(fs::is_directory(entry)){
    //             fs::path full_path= entry;
    //             std::string sequence= full_path.stem().string();
    //             VLOG(1) << "full path is " << full_path;
    //             VLOG(1) << "sequence is " << sequence;

    //             m_nr_sequences++;
    //             //read the npz of each sequence
    //             std::vector<fs::path> npz_filenames_for_sequence;
    //             for (fs::directory_iterator itr(full_path); itr!=fs::directory_iterator(); ++itr){
    //                 //all the files in the folder might include also the pose file so we ignore that one
    //                 //we also ignore the files that contain intensity, for now we only read the general ones and then afterwards we append _i to the file and read the intensity if neccesarry
    //                 if( !(itr->path().stem()=="poses")  && itr->path().stem().string().find("_i")== std::string::npos ){
    //                     npz_filenames_for_sequence.push_back(itr->path());
    //                 }
    //             }
    //             if(!m_shuffle){ //if we are shuffling, there is no need to sort them
    //                 std::sort(npz_filenames_for_sequence.begin(), npz_filenames_for_sequence.end());
    //             }

    //             npz_filenames_all.insert(npz_filenames_all.end(), npz_filenames_for_sequence.begin(), npz_filenames_for_sequence.end());

    //         	if(m_do_pose){
    //           	    //read poses for this sequence
    //                 std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  > poses;
    //                 fs::path pose_file = full_path/"poses.txt";
    //                 poses=read_pose_file( pose_file.string() );
    //                 m_poses_per_sequence[sequence] = poses;
    //             }
    //         }
    //     }
    //     VLOG(1) << "m_nr_sequences is " << m_nr_sequences;


    // }else{
    //     LOG(FATAL) << "Sequence is not known" << m_sequence;
    // }


    // //shuffle the filles to be read if necessary
    // if(m_shuffle){
    //     // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //     unsigned seed = m_nr_resets;
    //     auto rng = std::default_random_engine(seed);
    //     std::shuffle(std::begin(npz_filenames_all), std::end(npz_filenames_all), rng);
    // }


    // //ADDS THE clouds to the member std_vector of paths
    // //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    // for (size_t i = 0; i < npz_filenames_all.size(); i++) {
    //     if( (int)i>=m_nr_clouds_to_skip && ((int)m_npz_filenames.size()<m_nr_clouds_to_read || m_nr_clouds_to_read<0 ) ){
    //         m_npz_filenames.push_back(npz_filenames_all[i]);
    //     }
    // }

    // std::cout << "About to read " << m_npz_filenames.size() << " clouds" <<std::endl;


    // CHECK(m_npz_filenames.size()>0) <<"We did not find any npz files to read";

}

void DataLoaderUSCHair::read_data(){

    loguru::set_thread_name("loader_thread_kitti");


    while (m_is_running ) {

        //we finished reading so we wait here for a reset
        if(m_idx_cloud_to_read>=m_data_filenames.size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue

            fs::path data_filename=m_data_filenames[ m_idx_cloud_to_read ];
            if(!m_do_overfit){
                m_idx_cloud_to_read++;
            }



            // m_clouds_buffer.enqueue(cloud);;

        }

    }

}

bool DataLoaderUSCHair::has_data(){
    if(m_clouds_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    }
}


std::shared_ptr<Mesh> DataLoaderUSCHair::get_cloud(){

    std::shared_ptr<Mesh> cloud;
    m_clouds_buffer.try_dequeue(cloud);

    return cloud;
}

bool DataLoaderUSCHair::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<m_data_filenames.size()){
        return false; //there is still more files to read
    }

    //check that there is nothing in the ring buffers
    if(m_clouds_buffer.peek()!=nullptr){
        return false; //there is still something in the buffer
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderUSCHair::is_finished_reading(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<m_data_filenames.size()){
        return false; //there is still more files to read
    }

    return true; //there is nothing more to read and so we are finished reading

}

void DataLoaderUSCHair::reset(){
    m_nr_resets++;
    // we shuffle again the data so as to have freshly shuffled data for the next epoch
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // auto rng = std::default_random_engine(seed);
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(m_data_filenames), std::end(m_data_filenames), rng);
    }

    m_idx_cloud_to_read=0;
}

int DataLoaderUSCHair::nr_samples(){
    return m_data_filenames.size();
}
void DataLoaderUSCHair::set_mode_train(){
    m_mode="train";
}
void DataLoaderUSCHair::set_mode_test(){
    m_mode="test";
}
void DataLoaderUSCHair::set_mode_validation(){
    m_mode="val";
}
