#include "data_loaders/DataLoaderToyExample.h"

//ros
#include <ros/ros.h>

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//ros
#include "data_loaders/utils/RosTools.h"

//my stuff 
#include "data_loaders/core/MeshCore.h"
#include "data_loaders/utils/MiscUtils.h"
#include "data_loaders/utils/Profiler.h"
#include "data_loaders/LabelMngr.h"

//json 
#include "json11.hpp"


using namespace rady::utils;
using namespace easy_pbr;

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

DataLoaderToyExample::DataLoaderToyExample(const std::string config_file):
    m_clouds_buffer(BUFFER_SIZE),
    m_is_running(false)
{
    init_params(config_file);
    if(m_autostart){
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderToyExample::read_data, this);  //starts the spin in another thread
    }

}

DataLoaderToyExample::~DataLoaderToyExample(){

    m_is_running=false;
    m_loader_thread.join();
}

void DataLoaderToyExample::init_params(const std::string config_file){

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config loader_config=cfg["loader_toy_example"];

    m_autostart=loader_config["autostart"];
    m_sine_nr_points=loader_config["sine_nr_points"];
    m_sine_nr_cycles=loader_config["sine_nr_cycles"];
    m_sine_amplitude=loader_config["sine_amplitude"];

    //label file and colormap
    Config mngr_config=loader_config["label_mngr"];
    m_label_mngr=std::make_shared<LabelMngr>(mngr_config);

}

void DataLoaderToyExample::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";
    m_is_running=true;
    m_loader_thread=std::thread(&DataLoaderToyExample::read_data, this);  //starts the spin in another thread
}



void DataLoaderToyExample::init_data_reading(){
    

}

void DataLoaderToyExample::read_data(){

    loguru::set_thread_name("loader_thread_toyexample");

    init_data_reading();

    while (m_is_running) {


        if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space

            //make sinewave cloud
            MeshCore cloud;
            cloud=sine_wave_3D(m_sine_nr_points, m_sine_nr_cycles, m_sine_amplitude);
            cloud.L_gt.resize(cloud.V.rows(),1);
            cloud.L_gt.setConstant(1);//we consider 0 as being the background and 1 as being the one and only class
            cloud.D=cloud.V.rowwise().norm();

            cloud.normalize_size();
            cloud.normalize_position(); //will segfault because V is Nx2
           


            //some sensible visualization options
            cloud.m_vis.m_show_mesh=false;
            cloud.m_vis.m_show_points=true;
            cloud.m_vis.m_color_type=+MeshColorType::SemanticGT;
            
            //set the labelmngr which will be used by the viewer to put correct colors for the semantics
            cloud.m_label_mngr=m_label_mngr->shared_from_this();

            m_clouds_buffer.enqueue(cloud);;

        }

    }

}

MeshCore DataLoaderToyExample::sine_wave_3D(const int nr_points, const int nr_cycles, const float wave_amplitude){
    CHECK(nr_cycles<nr_points) << "The nr of points is not sufficient to represent such a high frequency sine wave. Please reduce the frequency or increase the nr of points";

    MeshCore sine;
    sine.V.resize(nr_points,3);
    sine.V.setZero();

    int nr_points_in_cycle=nr_points/nr_cycles; //the same as frequency actually
    for(int i=0; i<nr_points; i++){
        int idx_in_cycle=i%nr_points_in_cycle;
        float rads=map(idx_in_cycle, 0,nr_points_in_cycle, 0, 2*M_PI); //withing that cycle of points we go from 0 to 2pi radians
        float val= wave_amplitude * std::sin( rads );
        sine.V(i,0)=i;
        sine.V(i,1)=val;
    }

    sine.m_vis.m_show_points=true;
    return sine;    

}


bool DataLoaderToyExample::has_data(){
    if(m_clouds_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    }
}


MeshCore DataLoaderToyExample::get_cloud(){

    MeshCore cloud;
    m_clouds_buffer.try_dequeue(cloud);

    return cloud;
}

bool DataLoaderToyExample::is_finished(){
    //check if this loader has loaded everything
    // if(m_idx_cloud_to_read<(int)m_pts_filenames.size()){
    //     return false; //there is still more files to read
    // }

    //check that there is nothing in the ring buffers
    if(m_clouds_buffer.peek()!=nullptr){
        return false; //there is still something in the buffer
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderToyExample::is_finished_reading(){
    //check if this loader has loaded everything
    // if(m_idx_cloud_to_read<(int)m_pts_filenames.size()){
    //     return false; //there is still more files to read
    // }

    return true; //there is nothing more to read and so we are finished reading

}


int DataLoaderToyExample::nr_samples(){
    return 1.0;
}

void DataLoaderToyExample::set_mode_train(){
}
void DataLoaderToyExample::set_mode_test(){
}
void DataLoaderToyExample::set_mode_validation(){
}


