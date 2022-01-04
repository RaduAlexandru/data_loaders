#include "data_loaders/DataLoaderPhenorobCP1.h"

#include <limits>

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;


//my stuff
#include "data_loaders/DataTransformer.h"
#include "easy_pbr/Frame.h"
#include "Profiler.h"
#include "string_utils.h"
#include "numerical_utils.h"
#include "opencv_utils.h"
#include "eigen_utils.h"
#include "RandGenerator.h"

//json
#include "json11/json11.hpp"


//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;


DataLoaderPhenorobCP1::DataLoaderPhenorobCP1(const std::string config_file):
    // m_is_running(false),
    m_idx_img_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator)
{
    init_params(config_file);

    if(m_autostart){
        start();
    }

}

DataLoaderPhenorobCP1::~DataLoaderPhenorobCP1(){

    // m_is_running=false;
    // m_loader_thread.join();
}

void DataLoaderPhenorobCP1::init_params(const std::string config_file){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_phenorob_cp1"];

    m_autostart=loader_config["autostart"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_shuffle=loader_config["shuffle"];
    m_load_as_shell= loader_config["load_as_shell"];
    m_do_overfit=loader_config["do_overfit"];
    m_scene_scale_multiplier= loader_config["scene_scale_multiplier"];
    m_mode=(std::string)loader_config["mode"];

    m_dataset_path = (std::string)loader_config["dataset_path"];   
    m_scan_date = (std::string)loader_config["scan_date"];   
    m_scan_idx = loader_config["scan_idx"];   
    // m_pose_file_path = (std::string)loader_config["pose_file_path"];    //get the path where all the off files are


}

void DataLoaderPhenorobCP1::start(){
    init_data_reading();
    init_poses();
    read_data();
}


void DataLoaderPhenorobCP1::init_data_reading(){

    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }

    fs::path scan_path= m_dataset_path/m_scan_date/ std::to_string(m_scan_idx);
    CHECK( fs::is_directory(scan_path) ) << "Scan path does not exist under " << scan_path;


    //iterate through the scan and get all the blocks
    for (fs::directory_iterator itr(scan_path); itr!=fs::directory_iterator(); ++itr){
        fs::path block_path= itr->path();
        VLOG(1) << "Block path " << block_path;

        //load the paths for this block
        if (fs::is_directory(block_path) ){
            //create a block 
            std::shared_ptr<PRCP1Block> block= std::make_shared<PRCP1Block>();;

            //iterate through the block and get the paths of the images and the photoneo
            for (fs::directory_iterator itr_blk(block_path); itr_blk!=fs::directory_iterator(); ++itr_blk){
                VLOG(1) << "inside block" << itr_blk->path(); 
                fs::path inside_blk=itr_blk->path();
                if(radu::utils::contains( inside_blk.string(), "nikon" ) ){
                    Frame new_rgb_frame;
                    new_rgb_frame.rgb_path= (inside_blk/"img.jpeg").string();
                    block->m_rgb_frames.push_back(new_rgb_frame);
                }
                if(radu::utils::contains( inside_blk.string(), "photoneo" ) ){
                    //mesh
                    easy_pbr::MeshSharedPtr mesh= Mesh::create();
                    mesh->m_disk_path=(inside_blk/"cloud.pcd").string();
                    block->m_photoneo_mesh=mesh;
                    //frame
                    Frame new_photoneo_frame;
                    new_photoneo_frame.rgb_path= (inside_blk/"texture.jpeg").string();
                    block->m_photoneo_frame=new_photoneo_frame;
                }
            }

            //finsihed block
            m_blocks.push_back(block);




        }
    }



}

void DataLoaderPhenorobCP1::init_poses(){

    std::string rgb_pose_file="/media/rosu/Data/data/phenorob/days_on_field/2021_05_20_incomplete_just_9/rgb_calib/camchain-.img.yaml";
    
    // // //read json
    // std::string file_list_string=radu::utils::file_to_string(rgb_pose_file);
    // std::string err;
    // const auto json = json11::Json::parse(file_list_string, err);
    // std::string topic = json["cam0"]["rostopic"].string_value();
    // VLOG(1) << "topic is " << topic;
    // VLOG(1) << "err is " <<err;

    // Config cfg = configuru::parse_file(rgb_pose_file, JSON);
    //  std::string topic =  (std::string)cfg["cam0"]["rostopic"];
    // VLOG(1) << "topic is " << topic;



}

void DataLoaderPhenorobCP1::read_data(){

    


    for (size_t i = 0; i < m_blocks.size(); i++){

        //load the rgb frame
        for (size_t j = 0; j < m_blocks[i]->m_rgb_frames.size(); j++){
            Frame &frame=m_blocks[i]->m_rgb_frames[j];

            //load the images if necessary or delay it for whne it's needed
            frame.load_images=[this]( easy_pbr::Frame& frame ) -> void{ this->load_images_in_frame(frame); };
            if (m_load_as_shell){   //set the function to load the images whenever it's neede
                frame.is_shell=true;
            }else{
                frame.is_shell=false;
                frame.load_images(frame);
            }


            //get cam_id 
            std::string filename=fs::path(frame.rgb_path).parent_path().filename().string();
            VLOG(1) << "filename: " << filename;
            std::vector<std::string> filename_tokens=radu::utils::split(filename, "_");
            CHECK(filename_tokens.size()==2) << "We should have only two tokens here for example nikon_3 but the filename is " << filename;
            frame.cam_id= std::stoi(filename_tokens[1]);

            //extrinsics

            //intrinsics


            //distorsion


            //rescale things if necessary
            if(m_scene_scale_multiplier>0.0){
                Eigen::Affine3f tf_world_cam_rescaled = frame.tf_cam_world.inverse();
                tf_world_cam_rescaled.translation()*=m_scene_scale_multiplier;
                frame.tf_cam_world=tf_world_cam_rescaled.inverse();
            }
        }


        //load the photoneo data

    }


}


void DataLoaderPhenorobCP1::load_images_in_frame(easy_pbr::Frame& frame){

    frame.is_shell=false;

    //read rgba and split into rgb and alpha mask
    cv::Mat rgb_32f;

    cv::Mat rgb_8u = cv::imread( frame.rgb_path );
    //resize the rgb8u mat and then convert to float because its faster
    if(m_subsample_factor>1){
        cv::Mat resized;
        cv::resize(rgb_8u, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
        rgb_8u=resized;
    }
    // frame.rgb_8u=rgb_8u;
    rgb_8u.convertTo(rgb_32f, CV_32FC3, 1.0/255.0);
    // VLOG(1) << " type is  " << radu::utils::type2string(rgba_32f.type());

    frame.rgb_32f= rgb_32f;
    // cv::cvtColor(frame.rgb_32f, frame.gray_32f, CV_BGR2GRAY);

    frame.width=frame.rgb_32f.cols;
    frame.height=frame.rgb_32f.rows;

}

//block functions
Frame PRCP1Block::get_rgb_frame_at_idx( const int idx){
    CHECK(idx<m_rgb_frames.size()) << "idx is out of bounds. It is " << idx << " while m_rgb_frames has size " << m_rgb_frames.size();
    Frame  frame= m_rgb_frames[idx];
    return frame;
}




std::shared_ptr<PRCP1Block> DataLoaderPhenorobCP1::get_block_with_idx(const int idx){
    CHECK(idx<m_blocks.size()) << "idx is out of bounds. It is " << idx << " while m_blocks has size " << m_blocks.size();

    std::shared_ptr<PRCP1Block>  block = m_blocks[idx];

    return block;
}



bool DataLoaderPhenorobCP1::is_finished(){
    // //check if this loader has returned all the images it has
    // if(m_idx_img_to_read<m_frames.size()){
    //     return false; //there is still more files to read
    // }


    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderPhenorobCP1::reset(){

    m_nr_resets++;

    // //reshuffle for the next epoch
    // if(m_shuffle && m_mode=="train"){
    //     unsigned seed = m_nr_resets;
    //     auto rng_0 = std::default_random_engine(seed);
    //     std::shuffle(std::begin(m_frames), std::end(m_frames), rng_0);
    // }

    m_idx_img_to_read=0;
}

// int DataLoaderPhenorobCP1::nr_samples(){
//     return m_frames.size();
// }

bool DataLoaderPhenorobCP1::has_data(){
    return true; //we always have data since the loader stores all the image in memory and keeps them there
}

void DataLoaderPhenorobCP1::set_mode_train(){
    m_mode="train";
}
void DataLoaderPhenorobCP1::set_mode_test(){
    m_mode="test";
}
void DataLoaderPhenorobCP1::set_mode_validation(){
    m_mode="val";
}
void DataLoaderPhenorobCP1::set_mode_all(){
    m_mode="all";
}
