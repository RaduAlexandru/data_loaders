#include "data_loaders/fb/DataLoaderUSCHair.h"

//c++
#include <algorithm>
#include <random>
#include <cstdio>

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
    Config loader_config=cfg["loader_usc_hair"];

    m_autostart=loader_config["autostart"];
    m_mode=(std::string)loader_config["mode"];
    m_nr_clouds_to_skip=loader_config["nr_clouds_to_skip"];
    m_nr_clouds_to_read=loader_config["nr_clouds_to_read"];
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

    for (fs::directory_iterator itr(m_dataset_path); itr!=fs::directory_iterator(); ++itr){

        // VLOG(1) << "checing" << itr->path();
        // VLOG(1) << "filename " <<itr->path().filename().string();
        std::string filename_with_ext=itr->path().filename().string();
        if(   radu::utils::contains(filename_with_ext, "data") ){
            // VLOG(1) << "itr->path() " << itr->path();
            data_filenames_all.push_back(itr->path());
        }
    }


    //shuffle the filles to be read if necessary
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(data_filenames_all), std::end(data_filenames_all), rng);
    }


    //ADDS THE clouds to the member std_vector of paths
    //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    for (size_t i = 0; i < data_filenames_all.size(); i++) {
        if( (int)i>=m_nr_clouds_to_skip && ((int)m_data_filenames.size()<m_nr_clouds_to_read || m_nr_clouds_to_read<0 ) ){
            m_data_filenames.push_back(data_filenames_all[i]);
        }
    }

    std::cout << "About to read " << m_data_filenames.size() << " clouds" <<std::endl;


    CHECK(m_data_filenames.size()>0) <<"We did not find any npz files to read";


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

            std::string data_filepath=m_data_filenames[ m_idx_cloud_to_read ].string();
            if(!m_do_overfit){
                m_idx_cloud_to_read++;
            }

            //read data. most of the code is from http://www-scf.usc.edu/~liwenhu/SHM/Hair.cc
            VLOG(1) << "reading from " <<  data_filepath;

            std::FILE *f = std::fopen(data_filepath.c_str(), "rb");
            CHECK(f) << "Couldn't open" << data_filepath;


            TIME_START("load");


            std::vector< std::shared_ptr<easy_pbr::Mesh> > strands;
            std::shared_ptr<easy_pbr::Mesh> full_hair=easy_pbr::Mesh::create();
            std::vector<Eigen::Vector3d> full_hair_points_vec;

            int nstrands = 0;
            fread(&nstrands, 4, 1, f);
            VLOG(1) << "nr strands" << nstrands;
            // if (!fread(&nstrands, 4, 1, f)) {
            //     fprintf(stderr, "Couldn't read number of strands\n");
            //     fclose(f);
            //     return false;
            // }
            strands.resize(nstrands);

            for (int i = 0; i < nstrands; i++) {
                // VLOG(1) << "strand " <<i;
                int nverts = 0;
                std::shared_ptr<easy_pbr::Mesh> strand= easy_pbr::Mesh::create();
                fread(&nverts, 4, 1, f);
                // if (!fread(&nverts, 4, 1, f)) {
                //     fprintf(stderr, "Couldn't read number of vertices\n");
                //     fclose(f);
                //     return false;
                // }
                // strands[i].resize(nverts);
                strand->V.resize(nverts,3);
                // Eigen::VectorXf strand_points_float;
                // strand_points_float.resize(nverts,3);

                for (int j = 0; j < nverts; j++) {
                    // VLOG(1) << "vert " <<j;
                    // fread(&strand_points_float(i,0), 12, 1, f);
                    float x,y,z;
                    fread(&x, 4, 1, f);
                    fread(&y, 4, 1, f);
                    fread(&z, 4, 1, f);
                    strand->V.row(j) << x,y,z;

                    Eigen::Vector3d point;
                    point << x,y,z;
                    full_hair_points_vec.push_back(point);

                }

                //finished reading this strand
                strands.push_back(strand);
            }

            VLOG(1) << "finished reading everything";

            fclose(f);
            // return true;

            //get the full cloud into one
            // std::shared_ptr<easy_pbr::Mesh> full_hair=easy_pbr::Mesh::create();
            full_hair->V=vec2eigen(full_hair_points_vec);
            full_hair->m_vis.m_show_points=true;
            VLOG(1) << "adding";
            VLOG(1) << "finished adding";


            TIME_END("load");

            m_clouds_buffer.enqueue(full_hair);;

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
