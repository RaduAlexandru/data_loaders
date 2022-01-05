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
// #include "json11/json11.hpp"

//json 
// #include "yaml-cpp/parser.h"
// #include "yaml-cpp/node/node.h"
// #include "yaml-cpp/node/parse.h"
#include "yaml-cpp/yaml.h"


//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;



// template <typename T>
// T tryReadYamlNode( const YAML::Node & node, const std::string & key, const std::string & camName )
// {
//     try
//     {
//         return node[key].as<T>();
//     }
//     catch (...)
//     {
//         ROS_ERROR_STREAM( "Could not retrieve field \"" << key << "\" for camera " << camName );
//     }
//     return T();
// }






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
    m_rotation_alignment_degrees= loader_config["rotation_alignment_degrees"];

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
                    //get cam id 
                    std::string filename=fs::path(new_rgb_frame.rgb_path).parent_path().filename().string();
                    std::vector<std::string> filename_tokens=radu::utils::split(filename, "_");
                    CHECK(filename_tokens.size()==2) << "We should have only two tokens here for example nikon_3 but the filename is " << filename;
                    new_rgb_frame.cam_id= std::stoi(filename_tokens[1]);
                    //push
                    // block->m_rgb_frames.push_back(new_rgb_frame);
                    block->m_rgb_frames[new_rgb_frame.cam_id]=new_rgb_frame;
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

    Eigen::Affine3d tf_camcur_cam0;
    tf_camcur_cam0.setIdentity();

    YAML::Node config = YAML::LoadFile(rgb_pose_file);
    int nr_calibrated_cams=config.size();
    //go through the chain of poses and get the poses of tf_cam_c0; 
    // for(YAML::const_iterator it=config.begin();it != config.end();++it) {
        // std::string cam_name = it->first.as<std::string>();       // <- key
        // int cam_idx = std::stoi( radu::utils::erase_substring(cam_name, "cam")  );
    for (size_t cam_idx = 0; cam_idx < m_blocks[0]->m_rgb_frames.size(); cam_idx++){
        std::string cam_name= "cam"+std::to_string(cam_idx); 


        //run through the chain of cameras to get the calibration for this one 

        if(cam_idx!=0){
            Eigen::Affine3d T_cn_cnm1 = Eigen::Affine3d::Identity();
            VLOG(1) << "accessing T for camera " << cam_name;
            std::vector<std::vector<double> > vT_cn_cnm1 =config[cam_name]["T_cn_cnm1"].as< std::vector<std::vector<double> > >();
            // std::vector<std::vector<double> > vT_cn_cnm1 = tryReadYamlNode< std::vector<std::vector<double> > >( camConfig, "T_cn_cnm1", camName );
            // VLOG(1) << "it worked";
            for ( size_t j = 0; j < 3; ++j )            {
                T_cn_cnm1.translation()(j) = vT_cn_cnm1[j][3];
                for ( size_t k = 0; k < 3; ++k){
                    T_cn_cnm1.linear()(j,k) = vT_cn_cnm1[j][k];
                }
            }
            VLOG(1) << " T_cn_cnm1 is " << T_cn_cnm1.matrix();
            VLOG(1) << " before tf_camcur_cam0 is " << tf_camcur_cam0.matrix();
            // follow chain
            tf_camcur_cam0=T_cn_cnm1*tf_camcur_cam0;
            VLOG(1) << " ater tf_camcur_cam0 is " << tf_camcur_cam0.matrix();
            VLOG(1) << "";
        }
        m_camidx2pose[cam_idx]=tf_camcur_cam0;
        Eigen::Vector2i res;
        std::vector<int> res_vec=config[cam_name]["resolution"].as< std::vector<int> >();
        res<< res_vec[0],res_vec[1];
        m_camidx2resolution[cam_idx]=res;

    }
    //get extrinsics 
    // for (size_t kalibr_cam_idx = 0; kalibr_cam_idx < nr_calibrated_cams; kalibr_cam_idx++){

        // std::string cam_name= "cam"+std::to_string(cam_idx); 


            //run through the chain of cameras to get the calibration for this one 
            // Eigen::Affine3d tf_camcur_cam0;
            // tf_camcur_cam0.setIdentity();
            // for (size_t kalibr_cam_idx = 1; kalibr_cam_idx <= cam_idx; kalibr_cam_idx++){
            //     Eigen::Affine3d T_cn_cnm1 = Eigen::Affine3d::Identity();
            //     // VLOG(1) << "accessing T for camera " << kalibr_cam_idx;
            //     std::vector<std::vector<double> > vT_cn_cnm1 =config[cam_name]["T_cn_cnm1"].as< std::vector<std::vector<double> > >();
            //     // std::vector<std::vector<double> > vT_cn_cnm1 = tryReadYamlNode< std::vector<std::vector<double> > >( camConfig, "T_cn_cnm1", camName );
            //     // VLOG(1) << "it worked";
            //     for ( size_t j = 0; j < 3; ++j )            {
            //         T_cn_cnm1.translation()(j) = vT_cn_cnm1[j][3];
            //         for ( size_t k = 0; k < 3; ++k){
            //             T_cn_cnm1.linear()(j,k) = vT_cn_cnm1[j][k];
            //         }
            //     }
            //     // follow chain
            //     tf_camcur_cam0=T_cn_cnm1*tf_camcur_cam0;
            // }
            // frame.tf_cam_world=tf_camcur_cam0.cast<float>();
    // }





    //set the poses for every cam in every block
    for (size_t blk_idx = 0; blk_idx < m_blocks.size(); blk_idx++){
        // CHECK( nr_calirated_cams==m_blocks[blk_idx]->m_rgb_frames.size() ) << "We need calibration for each camera. We have nr calibrated cams " << nr_calibrated_cams << " but we have nr frames " << m_blocks[blk_idx]->m_rgb_frames.size();

        for (size_t cam_idx = 0; cam_idx < m_blocks[blk_idx]->m_rgb_frames.size(); cam_idx++){
            Frame &frame=m_blocks[blk_idx]->m_rgb_frames[cam_idx];

            std::string cam_name= "cam"+std::to_string(cam_idx);

            //get the intrinsics
            std::vector<float> intrinsics_vec = config[cam_name]["intrinsics"].as<std::vector<float>>();
            CHECK(intrinsics_vec.size()==4) << "Intrinsics_vec should be size of 4 but it is " << intrinsics_vec.size();
            // VLOG(1) << "intrinsics vec is " << intrinsics_vec;
            frame.K(0,0)=intrinsics_vec[0];
            frame.K(1,1)=intrinsics_vec[1];
            frame.K(0,2)=intrinsics_vec[2];
            frame.K(1,2)=intrinsics_vec[3];
            //the y principal point needs to be flipped because it actually measures the distance from the bottom but we measure the distance from the top
            int height=m_camidx2resolution[cam_idx].y();
            frame.K(1,2) = height - frame.K(1,2);
            // VLOG(1) << "K is " << frame.K;
            frame.rescale_K(1.0/m_subsample_factor);


            //get extrinsics 
            //run through the chain of cameras to get the calibration for this one 
            // Eigen::Affine3d tf_camcur_cam0;
            // tf_camcur_cam0.setIdentity();
            // for (size_t kalibr_cam_idx = 1; kalibr_cam_idx <= cam_idx; kalibr_cam_idx++){
            //     Eigen::Affine3d T_cn_cnm1 = Eigen::Affine3d::Identity();
            //     // VLOG(1) << "accessing T for camera " << kalibr_cam_idx;
            //     std::vector<std::vector<double> > vT_cn_cnm1 =config[cam_name]["T_cn_cnm1"].as< std::vector<std::vector<double> > >();
            //     // std::vector<std::vector<double> > vT_cn_cnm1 = tryReadYamlNode< std::vector<std::vector<double> > >( camConfig, "T_cn_cnm1", camName );
            //     // VLOG(1) << "it worked";
            //     for ( size_t j = 0; j < 3; ++j )            {
            //         T_cn_cnm1.translation()(j) = vT_cn_cnm1[j][3];
            //         for ( size_t k = 0; k < 3; ++k){
            //             T_cn_cnm1.linear()(j,k) = vT_cn_cnm1[j][k];
            //         }
            //     }
            //     // follow chain
            //     tf_camcur_cam0=T_cn_cnm1*tf_camcur_cam0;
            // }
            // frame.tf_cam_world=tf_camcur_cam0.cast<float>();
            // frame.tf_cam_world=m_camidx2pose[cam_idx].cast<float>();

            Eigen::Affine3f pre_rotate;
            pre_rotate.setIdentity();
            Eigen::Matrix3f r = (Eigen::AngleAxisf( radu::utils::degrees2radians(m_rotation_alignment_degrees), Eigen::Vector3f::UnitX()) ).toRotationMatrix();
            pre_rotate.linear()=r;


            Eigen::Affine3f tf_cam_world = m_camidx2pose[cam_idx].cast<float>()*pre_rotate;
            Eigen::Affine3f tf_world_cam;
            tf_world_cam= tf_cam_world.inverse();
            //flip y locally
            tf_world_cam.matrix().col(1) = -tf_world_cam.matrix().col(1);

            
            tf_cam_world = tf_world_cam.inverse();
            frame.tf_cam_world=tf_cam_world;

            VLOG(1) << "final tf for cam " << cam_idx << "is " << frame.tf_cam_world.matrix();


            //get distorsion 




        }
    }
    //check that the nr of cams in the kalibr yaml file is the same as rgb frames
    // const std::string topic = config["cam0"]["rostopic"].as<std::string>();
    // VLOG(1) << "topic is " << topic;
    // VLOG(1) << "nr_cams " <<nr_cams;
    // VLOG(1) << "nr_frames" <<m_blocks[0]->m_rgb_frames.size();
    // for (size_t i = 0; i < nr_cams; i++){
    // }


    // YAML::Node config;



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
