#include "data_loaders/RosBagPlayer.h"

//c++
#include<iostream>
#include<thread>
#include<chrono>

// //loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//ros
#include <ros/ros.h>
#include "ros_utils.h"


RosBagPlayer::RosBagPlayer(const std::string config_file):
        m_player_should_do_one_step(false),
        m_player_should_continue_after_step(false),
        m_nr_resets(0),
        m_rosbag(new TinyProcessLib::Process("rosbag play"))
        {

        init_params(config_file);
        play(m_cmd);
}
RosBagPlayer::~RosBagPlayer(){
    m_rosbag->kill(true);
}

void RosBagPlayer::init_params(const std::string config_file){
    //get the config filename
    // ros::NodeHandle private_nh("~");
    // std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    // std::string config_file="config.cfg";

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config rosbag_config=cfg["ros_bag"];
    std::string bag_path = (std::string)rosbag_config["bag_path"];
    std::string bag_args = (std::string)rosbag_config["bag_args"];
    m_cmd=" " + bag_path + " " + bag_args;
}

void RosBagPlayer::play(std::string args){
    //Check for pause in the arguments
    m_paused= args.find("pause") != std::string::npos ? true : false;
    VLOG(2) << "starting bag with pause set to " << m_paused;
    m_rosbag->add_arguments(args);
    m_rosbag->run();
}

void RosBagPlayer::pause(){
    if(!m_paused){
        VLOG(2) << "pausing the bag";
        m_rosbag->write(" ");
        m_paused=!m_paused;
    }
}

void RosBagPlayer::play(){
    if(m_paused){
        VLOG(2) << "playing the bag";
        m_rosbag->write(" ");
        m_paused=!m_paused;
    }
}

void RosBagPlayer::reset(){
    m_nr_resets++;
    m_rosbag->kill(true);
    m_rosbag->run();
}

bool RosBagPlayer::is_paused(){
    return m_paused;
}

//when the bag finishes running this will return true
bool RosBagPlayer::is_finished(){
    int exit_status;
    bool running=m_rosbag->try_get_exit_status(&exit_status); //the documentation of tinyprocess library seems to say that it returns the oposite but whatever..
    return !running;
}

void RosBagPlayer::kill(){
    LOG(WARNING) << "Rosbag is killed";
    m_rosbag->kill(true);
}