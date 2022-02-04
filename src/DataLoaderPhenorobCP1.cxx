#include "data_loaders/DataLoaderPhenorobCP1.h"

#include <fstream>
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
    m_rgb_subsample_factor=loader_config["rgb_subsample_factor"];
    m_photoneo_subsample_factor=loader_config["photoneo_subsample_factor"];
    m_shuffle=loader_config["shuffle"];
    m_load_as_shell= loader_config["load_as_shell"];
    m_do_overfit=loader_config["do_overfit"];
    m_scene_scale_multiplier= loader_config["scene_scale_multiplier"];
    m_mode=(std::string)loader_config["mode"];
    m_rotation_alignment_degrees= loader_config["rotation_alignment_degrees"];
    m_transform_to_easypbr_world =  loader_config["transform_to_easypbr_world"];

    m_dataset_path = (std::string)loader_config["dataset_path"];   
    m_scan_date = (std::string)loader_config["scan_date"];   
    std::string dataset_type = (std::string)loader_config["dataset_type"];   
    if (dataset_type=="raw"){
        m_dataset_type=PHCP1DatasetType::Raw;
    }else if(dataset_type=="processed"){
        m_dataset_type=PHCP1DatasetType::Processed;
    }else{
        LOG(FATAL) << "Dataset type not known " << dataset_type;
    }
    m_load_poses=loader_config["load_poses"];   
    m_load_intrinsics=loader_config["load_intrinsics"];   
    m_load_dense_cloud=loader_config["load_dense_cloud"]; 
    m_load_sparse_cloud=loader_config["load_sparse_cloud"]; 
    m_load_visible_points=loader_config["load_visible_points"]; 


}

void DataLoaderPhenorobCP1::start(){
    init_data_reading();
    if (m_load_poses){
        // init_poses();
    }
    if (m_load_intrinsics){
        // init_intrinsics();
    }
    if (m_load_poses && m_load_intrinsics){
        init_intrinsics_and_poses_krt();
    }

    init_stereo_pairs();
    read_data();
}


void DataLoaderPhenorobCP1::init_data_reading(){

    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }

    // fs::path scan_path= m_dataset_path/m_scan_date/ std::to_string(m_scan_idx);
    // CHECK( fs::is_directory(scan_path) ) << "Scan path does not exist under " << scan_path;

    fs::path day_path= m_dataset_path/m_scan_date;
    CHECK( fs::is_directory(day_path) ) << "Day path does not exist under " << day_path;


    //get the scans from this day
    std::vector<fs::path> scans_vec;
    std::copy(fs::directory_iterator(day_path), fs::directory_iterator(), std::back_inserter(scans_vec));
    std::sort(scans_vec.begin(), scans_vec.end());
    // for(int i=0; i<scans_vec.size(); i++){
    //     VLOG(1) << "scan is " << scans_vec[i];
    // }



    //iterate through all the scans in that day
    for (const fs::path & scan_path : scans_vec){
        std::string scan_name=scan_path.filename().string();

        //skip the calib things 
        if (radu::utils::contains(scan_name, "calib") or radu::utils::contains(scan_name, "txt") or radu::utils::contains(scan_name, "extrinsics") ){
            continue;
        }

        //get either the raw one or the processed version of this scan
        if (m_dataset_type==+PHCP1DatasetType::Raw){
            if (  radu::utils::contains(scan_name, "processed") ){
                continue; //skip all the scan that DO have the word processed
            }
        }else if(m_dataset_type==+PHCP1DatasetType::Processed){
            if ( ! radu::utils::contains(scan_name, "processed") ){
                continue; //skip all the scan that don't have the word processed
            }
        }else{
            LOG(FATAL) << "Unknown data type";
        }

        //make a scan object
        VLOG(1) << "scan_name is " << scan_name;
        std::shared_ptr<PRCP1Scan> scan= std::make_shared<PRCP1Scan>();;
        scan->m_name=scan_name;
        scan->m_path=scan_path;




        //get the blocks
        std::vector<fs::path> blocks_vec;
        std::copy(fs::directory_iterator(scan_path), fs::directory_iterator(), std::back_inserter(blocks_vec));
        std::sort(blocks_vec.begin(), blocks_vec.end());




        // //iterate through the scan and get all the blocks
        for (const fs::path & block_path : blocks_vec){
            VLOG(1) << "Block path " << block_path;

            //load the paths for this block
            if (fs::is_directory(block_path) ){
                //create a block 
                std::shared_ptr<PRCP1Block> block= std::make_shared<PRCP1Block>();
                std::string block_name=block_path.filename().string();
                block->m_name=block_name;
                block->m_path=block_path;

                //get the block_nr
                std::vector<std::string> block_tokens=radu::utils::split(block_name, "_");
                CHECK(block_tokens.size()==2) <<"We should have 2 tokens for the blocks";
                int block_nr=std::stoi(block_tokens[1]);



                // //get the photoneo for this block
                fs::path photoneo_path=block_path/("photoneo_"+std::to_string(block_nr));
                if (fs::is_directory(photoneo_path) ){
                    //mesh
                    easy_pbr::MeshSharedPtr mesh= Mesh::create();
                    mesh->m_disk_path=(photoneo_path/"cloud.pcd").string();
                    block->m_photoneo_mesh=mesh;
                    //frame
                    Frame new_photoneo_frame;
                    new_photoneo_frame.rgb_path= (photoneo_path/"texture.jpeg").string();
                    new_photoneo_frame.depth_path= (photoneo_path/"depth.exr").string();
                    new_photoneo_frame.add_extra_field("is_photoneo", true);
                    //get the name of this frame which will be something like nikon_x
                    std::string frame_name=photoneo_path.filename().string();
                    new_photoneo_frame.m_name=frame_name;
                    new_photoneo_frame.cam_id=block_nr;
                    VLOG(1) << "Loaded photoneo with cam_id" << new_photoneo_frame.cam_id << " with depth " << new_photoneo_frame.depth_path;
                    //add it to the block
                    block->m_photoneo_frame=new_photoneo_frame;
                    block->m_photoneo_cfg_file_path= (photoneo_path/"info.cfg").string();
                }


                //get the dense and sparse cloud
                if (m_load_dense_cloud){
                    block->m_dense_cloud=easy_pbr::Mesh::create();
                    block->m_dense_cloud->m_disk_path=(block_path/"colmap_data/cloud_dense.ply").string();
                }
                if (m_load_sparse_cloud){
                    block->m_sparse_cloud=easy_pbr::Mesh::create();
                    block->m_sparse_cloud->m_disk_path=(block_path/"colmap_data/cloud_sparse.ply").string();
                }




                //get the nikon cameras. if we are loading from a RAW dataset, then the nikon are in block_x/nikon_y/ if we are loading from processed it is in block_x/nikons_subsample_y
                if (m_dataset_type==+PHCP1DatasetType::Raw){

                    //get the raw nikon cams
                    std::vector<fs::path> cams_vec;
                    std::copy(fs::directory_iterator(block_path), fs::directory_iterator(), std::back_inserter(cams_vec));
                    std::sort(cams_vec.begin(), cams_vec.end());

                    //iterate through the block and get the paths of the images and the photoneo
                    for (const fs::path & rgb_cam_path : cams_vec){
                        if(radu::utils::contains( rgb_cam_path.filename().string(), "nikon" ) ){

                            std::shared_ptr<Frame> new_rgb_frame= std::make_shared<easy_pbr::Frame>();
                            new_rgb_frame->rgb_path= (rgb_cam_path/"img.jpeg").string();

                            //get the name of this frame which will be something like nikon_x
                            std::string frame_name=rgb_cam_path.filename().string();
                            new_rgb_frame->m_name=frame_name;

                            //get cam id 
                            std::string filename=fs::path(new_rgb_frame->rgb_path).parent_path().filename().string();
                            std::vector<std::string> filename_tokens=radu::utils::split(filename, "_");
                            CHECK(filename_tokens.size()==2) << "We should have only two tokens here for example nikon_3 but the filename is " << filename;
                            new_rgb_frame->cam_id= std::stoi(filename_tokens[1]);
                            //push
                            block->m_rgb_frames[new_rgb_frame->cam_id]=new_rgb_frame;
                        }
                    }

                }else if(m_dataset_type==+PHCP1DatasetType::Processed){
                    //choose nikon_folder depending on the subsample
                    fs::path rgb_cam_path;
                    if (m_rgb_subsample_factor==1){
                        rgb_cam_path=block_path/"nikons";
                    }else{
                        rgb_cam_path=block_path/("nikons_subsample_"+std::to_string( (int)m_rgb_subsample_factor ) );
                    }

                    CHECK(fs::is_directory(rgb_cam_path)) << "Path for the rgb cams does nto exist, check if the subsample is not too large " << rgb_cam_path; 


                    //get the processed nikon cams
                    std::vector<fs::path> imgs_vec;
                    std::copy(fs::directory_iterator(rgb_cam_path), fs::directory_iterator(), std::back_inserter(imgs_vec));
                    std::sort(imgs_vec.begin(), imgs_vec.end());

                    for (const fs::path & rgb_img_path : imgs_vec){
                        std::shared_ptr<Frame> new_rgb_frame= std::make_shared<easy_pbr::Frame>();

                        //get cam id 
                        std::string filename=rgb_img_path.filename().string();
                        std::vector<std::string> filename_tokens=radu::utils::split(filename, ".");
                        CHECK(filename_tokens.size()==2) << "We should have only two tokens here for example 0.jpeg but the filename is " << filename;
                        int cam_id=std::stoi(filename_tokens[0]);
                        new_rgb_frame->cam_id= cam_id;


                        new_rgb_frame->rgb_path= rgb_img_path.string();
                        new_rgb_frame->depth_path =  (block_path/"colmap_data/depth_maps"/(std::to_string(cam_id)+"_geometric.exr")).string();

                        //get the name of this frame which will be something like nikon_x
                        std::string frame_name=rgb_img_path.filename().string();
                        new_rgb_frame->m_name=frame_name;

                        //make also a mesh for the visible points
                        if(m_load_visible_points){
                            auto mesh=easy_pbr::Mesh::create();
                            mesh->m_disk_path=  (block_path/"colmap_data/visible_clouds"/(std::to_string(cam_id)+".ply")).string();
                            new_rgb_frame->add_extra_field("visible_points", mesh);
                        }

                        //push
                        block->m_rgb_frames[new_rgb_frame->cam_id]=new_rgb_frame;
                    }
                    

                }

                //finsihed block
                scan->m_blocks.push_back(block);




            }
        }

        m_scans.push_back(scan);
    }





}

void DataLoaderPhenorobCP1::init_poses_kalibr(){

    // std::string rgb_pose_file="/media/rosu/Data/data/phenorob/days_on_field/2021_05_20_incomplete_just_9/rgb_calib/camchain-.img.yaml";
    // m_rgb_pose_file=(m_dataset_path/m_scan_date/"rgb_calib/camchain-.img.yaml").string();
    if (m_dataset_type==+PHCP1DatasetType::Raw){
        m_rgb_pose_file=(m_dataset_path/m_scan_date/"rgb_calib/camchain-.img.yaml").string();
    }else if(m_dataset_type==+PHCP1DatasetType::Processed){
        m_rgb_pose_file=(m_dataset_path/m_scan_date/"rgb_calib_processed/camchain-.img.yaml").string();
    }

    Eigen::Affine3d tf_camcur_cam0;
    tf_camcur_cam0.setIdentity();

    YAML::Node config = YAML::LoadFile(m_rgb_pose_file);
    int nr_calibrated_cams=config.size();
    for (size_t cam_idx = 0; cam_idx < m_scans[0]->m_blocks[0]->m_rgb_frames.size(); cam_idx++){
        std::string cam_name= "cam"+std::to_string(cam_idx); 


        //run through the chain of cameras to get the calibration for this one 
        if(cam_idx!=0){
            Eigen::Affine3d T_cn_cnm1 = Eigen::Affine3d::Identity();
            // VLOG(1) << "accessing T for camera " << cam_name;
            std::vector<std::vector<double> > vT_cn_cnm1 =config[cam_name]["T_cn_cnm1"].as< std::vector<std::vector<double> > >();
            // std::vector<std::vector<double> > vT_cn_cnm1 = tryReadYamlNode< std::vector<std::vector<double> > >( camConfig, "T_cn_cnm1", camName );
            // VLOG(1) << "it worked";
            for ( size_t j = 0; j < 3; ++j )            {
                T_cn_cnm1.translation()(j) = vT_cn_cnm1[j][3];
                for ( size_t k = 0; k < 3; ++k){
                    T_cn_cnm1.linear()(j,k) = vT_cn_cnm1[j][k];
                }
            }
            // follow chain
            tf_camcur_cam0=T_cn_cnm1*tf_camcur_cam0;
        }
        m_camidx2pose[cam_idx]=tf_camcur_cam0;
        Eigen::Vector2i res;
        std::vector<int> res_vec=config[cam_name]["resolution"].as< std::vector<int> >();
        res<< res_vec[0],res_vec[1];
        m_camidx2resolution[cam_idx]=res;

    }

    //a prerotation so as to align it a bit better to our world frame
    Eigen::Affine3f pre_rotate;
    pre_rotate.setIdentity();
    Eigen::Matrix3f r = (Eigen::AngleAxisf( radu::utils::degrees2radians(m_rotation_alignment_degrees), Eigen::Vector3f::UnitX()) ).toRotationMatrix();
    pre_rotate.linear()=r;
   


    //set the poses for every cam in even scan
    for (size_t scan_idx = 0; scan_idx < m_scans.size(); scan_idx++){

        //set the poses for every cam in every block
        for (size_t blk_idx = 0; blk_idx < m_scans[scan_idx]->m_blocks.size(); blk_idx++){
            CHECK( nr_calibrated_cams==m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames.size() ) << "We need calibration for each camera. We have nr calibrated cams " << nr_calibrated_cams << " but we have nr frames " << m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames.size();

            for (size_t cam_idx = 0; cam_idx < m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames.size(); cam_idx++){
                std::shared_ptr<Frame> frame=m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames[cam_idx];

                std::string cam_name= "cam"+std::to_string(cam_idx);

                // //get the intrinsics
                // std::vector<float> intrinsics_vec = config[cam_name]["intrinsics"].as<std::vector<float>>();
                // CHECK(intrinsics_vec.size()==4) << "Intrinsics_vec should be size of 4 but it is " << intrinsics_vec.size();
                // frame->K(0,0)=intrinsics_vec[0];
                // frame->K(1,1)=intrinsics_vec[1];
                // frame->K(0,2)=intrinsics_vec[2];
                // frame->K(1,2)=intrinsics_vec[3];
                // if (m_transform_to_easypbr_world){ //the y principal point needs to be flipped because we flip the y locally so we need to also flip y here
                //     // int height=m_camidx2resolution[cam_idx].y();
                //     // frame->K(1,2) = height - frame->K(1,2);
                // }
                // // VLOG(1) << "K is " << frame.K;
                // frame->rescale_K(1.0/m_rgb_subsample_factor);


                // //get distorsion 
                // std::vector<float> distorsion_vec = config[cam_name]["distortion_coeffs"].as<std::vector<float>>();
                // CHECK(distorsion_vec.size()==4) << "distorsion_vec should be size of 4 but it is " << distorsion_vec.size();
                // frame->distort_coeffs(0)=distorsion_vec[0];
                // frame->distort_coeffs(1)=distorsion_vec[1];
                // frame->distort_coeffs(2)=distorsion_vec[2];
                // frame->distort_coeffs(3)=distorsion_vec[3];



                if (m_transform_to_easypbr_world){
                    Eigen::Affine3f tf_cam_world = m_camidx2pose[cam_idx].cast<float>()*pre_rotate;
                    frame->tf_cam_world=tf_cam_world;
                }else{
                    frame->tf_cam_world= m_camidx2pose[cam_idx].cast<float>();
                }

               

            }
       

            //load the intrinsics and extrinsics for the photoneo
            auto block=m_scans[scan_idx]->m_blocks[blk_idx];

            //intrinsics
            Config photoneo_cfg = configuru::parse_file(block->m_photoneo_cfg_file_path, CFG);
            std::vector<float> intrinsics_vec = (std::vector<float>)photoneo_cfg["intrinsics"];
            CHECK(intrinsics_vec.size()==4) << "Intrinsics vec should have 4 elements";
            block->m_photoneo_frame.K(0,0)=intrinsics_vec[0];
            block->m_photoneo_frame.K(1,1)=intrinsics_vec[1];
            block->m_photoneo_frame.K(0,2)=intrinsics_vec[2];
            block->m_photoneo_frame.K(1,2)=intrinsics_vec[3];
            if (m_transform_to_easypbr_world){ //the y principal point needs to be flipped because we flip the y locally so we need to also flip y here
                //load the image just to get the height
                // cv::Mat photoneo_texture=cv::imread(block->m_photoneo_frame.rgb_path);
                // int height_photoneo=photoneo_texture.rows;
                // block->m_photoneo_frame.K(1,2) = height_photoneo - block->m_photoneo_frame.K(1,2);
            }
            block->m_photoneo_frame.rescale_K(1.0/m_photoneo_subsample_factor);

            //extrinsics
            if (m_dataset_type==+PHCP1DatasetType::Processed){
                std::string pose_file_path=(m_dataset_path/m_scan_date/"photoneo_extrinsics"/("pose_xyzquat_photoneo_world_"+std::to_string(blk_idx)+".txt")  ).string();
                std::string pose_file_string=radu::utils::file_to_string(pose_file_path);
                std::vector<std::string> pose_tokens=radu::utils::split(pose_file_string, " ");
                CHECK(pose_tokens.size()==7) <<"Should have 7 tokens corresponding to xyz, qx, qy, qz, qw";
                //make photoneo_world_matrix
                Eigen::Affine3f tf_photoneo_world;
                tf_photoneo_world.setIdentity();
                tf_photoneo_world.translation() << std::stof(pose_tokens[0]), std::stof(pose_tokens[1]), std::stof(pose_tokens[2]);
                Eigen::Quaternion<float> q;
                q.coeffs()<< stof(pose_tokens[3]), std::stof(pose_tokens[4]), std::stof(pose_tokens[5]),  std::stof(pose_tokens[6]) ;
                tf_photoneo_world.linear()=q.toRotationMatrix();
                //set matrix
                if (m_transform_to_easypbr_world){
                    Eigen::Affine3f tf_cam_world =tf_photoneo_world.cast<float>()*pre_rotate;
                    // Eigen::Affine3f tf_world_cam;
                    // tf_world_cam= tf_cam_world.inverse();
                    //flip y locally
                    // tf_world_cam.matrix().col(1) = -tf_world_cam.matrix().col(1);
                    // tf_cam_world = tf_world_cam.inverse();
                    block->m_photoneo_frame.tf_cam_world=tf_cam_world;
                }else{
                    // frame.tf_cam_world= tf_photoneo_world;
                    block->m_photoneo_frame.tf_cam_world=tf_photoneo_world;
                }

                //set also the extrinsics for the mesh
                // block->m_photoneo_mesh->set_model_matrix( block->m_photoneo_frame.tf_cam_world.inverse().cast<double>() );

                //attempt 2
                Eigen::Affine3f tf_cam_world;
                if (m_transform_to_easypbr_world){
                    tf_cam_world =tf_photoneo_world.cast<float>()*pre_rotate;
                }else{
                    tf_cam_world =tf_photoneo_world.cast<float>();
                }
                block->m_photoneo_mesh->set_model_matrix( tf_cam_world.inverse().cast<double>() );


            }




        }
    }
  

}

void DataLoaderPhenorobCP1::init_intrinsics_kalibr(){
    if (m_dataset_type==+PHCP1DatasetType::Raw){
        m_rgb_pose_file=(m_dataset_path/m_scan_date/"rgb_calib/camchain-.img.yaml").string();
    }else if(m_dataset_type==+PHCP1DatasetType::Processed){
        m_rgb_pose_file=(m_dataset_path/m_scan_date/"rgb_calib_processed/camchain-.img.yaml").string();
    }

    YAML::Node config = YAML::LoadFile(m_rgb_pose_file);
    int nr_calibrated_cams=config.size();


    //set the poses for every cam in even scan
    for (size_t scan_idx = 0; scan_idx < m_scans.size(); scan_idx++){

        //set the poses for every cam in every block
        for (size_t blk_idx = 0; blk_idx < m_scans[scan_idx]->m_blocks.size(); blk_idx++){
            CHECK( nr_calibrated_cams==m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames.size() ) << "We need calibration for each camera. We have nr calibrated cams " << nr_calibrated_cams << " but we have nr frames " << m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames.size();

            for (size_t cam_idx = 0; cam_idx < m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames.size(); cam_idx++){
                std::shared_ptr<Frame> frame=m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames[cam_idx];

                std::string cam_name= "cam"+std::to_string(cam_idx);

                //get the intrinsics
                std::vector<float> intrinsics_vec = config[cam_name]["intrinsics"].as<std::vector<float>>();
                CHECK(intrinsics_vec.size()==4) << "Intrinsics_vec should be size of 4 but it is " << intrinsics_vec.size();
                frame->K(0,0)=intrinsics_vec[0];
                frame->K(1,1)=intrinsics_vec[1];
                frame->K(0,2)=intrinsics_vec[2];
                frame->K(1,2)=intrinsics_vec[3];
                if (m_transform_to_easypbr_world){ //the y principal point needs to be flipped because we flip the y locally so we need to also flip y here
                    // int height=m_camidx2resolution[cam_idx].y();
                    // frame->K(1,2) = height - frame->K(1,2);
                }
                // VLOG(1) << "K is " << frame.K;
                frame->rescale_K(1.0/m_rgb_subsample_factor);


                //get distorsion 
                std::vector<float> distorsion_vec = config[cam_name]["distortion_coeffs"].as<std::vector<float>>();
                CHECK(distorsion_vec.size()==4) << "distorsion_vec should be size of 4 but it is " << distorsion_vec.size();
                frame->distort_coeffs(0)=distorsion_vec[0];
                frame->distort_coeffs(1)=distorsion_vec[1];
                frame->distort_coeffs(2)=distorsion_vec[2];
                frame->distort_coeffs(3)=distorsion_vec[3];
            }

        }

    }

}

void DataLoaderPhenorobCP1::init_intrinsics_and_poses_krt(){
    //read KRT2_maya file which has format:
    // camera_idx (width height)
    // intrinsics [3x3]
    // lens distortion [1x5]
    // extrinsics [3x4]


    //a prerotation so as to align it a bit better to our world frame
    Eigen::Affine3f pre_rotate;
    pre_rotate.setIdentity();
    Eigen::Matrix3f r = (Eigen::AngleAxisf( radu::utils::degrees2radians(m_rotation_alignment_degrees), Eigen::Vector3f::UnitX()) ).toRotationMatrix();
    pre_rotate.linear()=r;

    for (size_t scan_idx = 0; scan_idx < m_scans.size(); scan_idx++){
        auto scan=m_scans[scan_idx];
        for (size_t blk_idx = 0; blk_idx < scan->m_blocks.size(); blk_idx++){
            auto block=scan->m_blocks[blk_idx];

            std::unordered_map<int, Eigen::Affine3d> camidx2pose; //maps from the cam_idx of the image to the corresponding pose
            std::unordered_map<int, Eigen::Matrix3d> camidx2intrinsics; //maps from the cam_idx of the image to the corresponding K

            //for this block read the krt file
            fs::path krt_path=block->m_path/"KRT";
            CHECK(fs::is_regular_file(krt_path));

            std::ifstream infile( krt_path.string() );
            std::string line;
            std::vector<std::string> tokens;

            while (std::getline(infile, line)){

                //skip lines which start with #
                if (line.at(0)=='#'){
                    continue;
                }

                tokens=split(line," ");
                int cam_idx=std::stoi(tokens[0]);
                int width=std::stoi(tokens[1]);
                int height=std::stoi(tokens[2]);

                //3 lines for intrinsics
                std::string intrinsics_string_full;
                std::getline(infile, line);  intrinsics_string_full+=line+" ";
                std::getline(infile, line);  intrinsics_string_full+=line+" ";
                std::getline(infile, line);  intrinsics_string_full+=line+" ";
                tokens=split(intrinsics_string_full," ");
                Eigen::Matrix3d K;
                K.setIdentity();
                radu::utils::tokens2matrix(tokens,K);


                //distorsion
                std::getline(infile, line);
                tokens=split(line," ");
                Eigen::VectorXd distorsion;
                distorsion.resize(5);
                radu::utils::tokens2matrix(tokens,distorsion);


                //pose
                std::string pose_string_full;
                std::getline(infile, line);  pose_string_full+=line+" ";
                std::getline(infile, line);  pose_string_full+=line+" ";
                std::getline(infile, line);  pose_string_full+=line+" ";
                tokens=split(pose_string_full," ");
                Eigen::Matrix<double,3,4>  pose3x4;
                radu::utils::tokens2matrix(tokens, pose3x4 );
                // VLOG(1) << "pose3x4 is " << pose3x4;
                //convert to4x4
                Eigen::Matrix4d pose4x4;
                pose4x4.setIdentity();
                pose4x4.block<3,4>(0,0) = pose3x4;
                Eigen::Affine3d pose_affine;
                pose_affine.matrix()=pose4x4;
                // VLOG(1) << "poseaffine is " << pose_affine.matrix();


                //empty line
                std::getline(infile, line);


                //push things
                camidx2pose[cam_idx]=pose_affine;
                camidx2intrinsics[cam_idx]=K;
            }


            //set intrinsics and extrisnics of all nikons
            for (size_t cam_idx = 0; cam_idx < block->m_rgb_frames.size(); cam_idx++){
                std::shared_ptr<Frame> frame=block->m_rgb_frames[cam_idx];
                frame->K=camidx2intrinsics[cam_idx].cast<float>();
                frame->rescale_K(1.0/m_rgb_subsample_factor);
                if (m_transform_to_easypbr_world){
                    Eigen::Affine3f tf_cam_world = camidx2pose[cam_idx].cast<float>()*pre_rotate;
                    frame->tf_cam_world=tf_cam_world;

                    //transform also the visible clouds
                    if(m_load_visible_points){
                        auto mesh=frame->get_extra_field<std::shared_ptr<easy_pbr::Mesh> >("visible_points");
                        Eigen::Affine3d tf_world_obj = mesh->model_matrix();
                        Eigen::Affine3d tf_obj_world = tf_world_obj.inverse();
                        tf_obj_world=tf_obj_world*pre_rotate.cast<double>();
                        mesh->set_model_matrix( tf_obj_world.inverse() );
                    }

                }else{
                    frame->tf_cam_world= camidx2pose[cam_idx].cast<float>();
                }
            }


            //set the pose for the clouds
            if(m_load_dense_cloud && m_transform_to_easypbr_world){
                Eigen::Affine3d tf_world_obj =  block->m_dense_cloud->model_matrix();
                Eigen::Affine3d tf_obj_world = tf_world_obj.inverse();
                tf_obj_world=tf_obj_world*pre_rotate.cast<double>();
                block->m_dense_cloud->set_model_matrix( tf_obj_world.inverse() );
            }
            if(m_load_sparse_cloud){
                Eigen::Affine3d tf_world_obj =  block->m_sparse_cloud->model_matrix();
                Eigen::Affine3d tf_obj_world = tf_world_obj.inverse();
                tf_obj_world=tf_obj_world*pre_rotate.cast<double>();
                block->m_sparse_cloud->set_model_matrix( tf_obj_world.inverse() );
            }
        }
    }





    

}

void DataLoaderPhenorobCP1::init_stereo_pairs(){
    std::string pairs_file=(m_dataset_path/m_scan_date/"stereo_pairs.txt").string();

    std::ifstream infile(pairs_file);
    CHECK(infile.good()) << "Could not open file " << pairs_file;

    std::string line;
    while (std::getline(infile, line)){
        std::istringstream iss(line);
        int idx_left, idx_right;
        if (!(iss >> idx_left >> idx_right)) { break; } // error

        // process pair (a,b)
        // VLOG(1) << idx_left << " " << idx_right;
        m_stereo_pairs[idx_left] = idx_right;
    }

    //set the right frame 
    for (size_t scan_idx = 0; scan_idx < m_scans.size(); scan_idx++){
        for (size_t blk_idx = 0; blk_idx < m_scans[scan_idx]->m_blocks.size(); blk_idx++){
            //load the rgb frame
            for (size_t frame_idx = 0; frame_idx < m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames.size(); frame_idx++){
                std::shared_ptr<Frame> frame=m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames[frame_idx];

                int idx_right=m_stereo_pairs[frame_idx];
                if(idx_right!=-1){
                    
                    // frame->m_right_stereo_pair=std::make_shared<easy_pbr::Frame>( m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames[idx_right] );
                    frame->m_right_stereo_pair= m_scans[scan_idx]->m_blocks[blk_idx]->m_rgb_frames[idx_right] ;

                }

            }
        }
    }


}

void DataLoaderPhenorobCP1::read_data(){

    

    for (size_t scan_idx = 0; scan_idx < m_scans.size(); scan_idx++){

        for (size_t i = 0; i < m_scans[scan_idx]->m_blocks.size(); i++){

            //load the rgb frame
            for (size_t j = 0; j < m_scans[scan_idx]->m_blocks[i]->m_rgb_frames.size(); j++){
                std::shared_ptr<Frame> frame=m_scans[scan_idx]->m_blocks[i]->m_rgb_frames[j];

                //load the images if necessary or delay it for whne it's needed
                frame->load_images=[this]( easy_pbr::Frame& frame ) -> void{ this->load_images_in_frame(frame); };
                if (m_load_as_shell){   //set the function to load the images whenever it's neede
                    frame->is_shell=true;
                }else{
                    frame->is_shell=false;
                    frame->load_images(*frame);
                }



                //rescale things if necessary
                if(m_scene_scale_multiplier>0.0){
                    Eigen::Affine3f tf_world_cam_rescaled = frame->tf_cam_world.inverse();
                    tf_world_cam_rescaled.translation()*=m_scene_scale_multiplier;
                    frame->tf_cam_world=tf_world_cam_rescaled.inverse();
                }
            }


            //load the photoneo data
            auto block=m_scans[scan_idx]->m_blocks[i];
            Frame &photoneo_frame=block->m_photoneo_frame;
            photoneo_frame.load_images=[this]( easy_pbr::Frame& frame ) -> void{ this->load_images_in_frame(frame); };
            if (m_load_as_shell){   //set the function to load the images whenever it's neede
                photoneo_frame.is_shell=true;
            }else{
                photoneo_frame.is_shell=false;
                photoneo_frame.load_images(photoneo_frame);
            }

        }

    }


    


}


void DataLoaderPhenorobCP1::load_images_in_frame(easy_pbr::Frame& frame){

    frame.is_shell=false;

    //read rgba and split into rgb and alpha mask
    cv::Mat rgb_32f;

    cv::Mat rgb_8u = cv::imread( frame.rgb_path );
    //resize the rgb8u mat and then convert to float because its faster
    int subsample_factor=m_rgb_subsample_factor;
    if ( frame.has_extra_field("is_photoneo") ){
        subsample_factor=m_photoneo_subsample_factor;
    }
    //if it's a processed frame then it's already subsampled
    if(subsample_factor>1 && m_dataset_type==+PHCP1DatasetType::Raw){
        cv::Mat resized;
        cv::resize(rgb_8u, resized, cv::Size(), 1.0/subsample_factor, 1.0/subsample_factor, cv::INTER_AREA);
        rgb_8u=resized;
    }
    frame.rgb_8u=rgb_8u;
    rgb_8u.convertTo(rgb_32f, CV_32FC3, 1.0/255.0);
    // VLOG(1) << " type is  " << radu::utils::type2string(rgba_32f.type());

    frame.rgb_32f= rgb_32f;
    // cv::cvtColor(frame.rgb_32f, frame.gray_32f, CV_BGR2GRAY);

    frame.width=frame.rgb_32f.cols;
    frame.height=frame.rgb_32f.rows;


    //if we have depth load also that one
    if (!frame.depth_path.empty()){
        frame.depth = cv::imread(frame.depth_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        //resize to match the rgb frame if needed
        if(frame.depth.rows!=frame.height || frame.depth.cols!=frame.width){
            cv::Mat resized;
            cv::resize(frame.depth, resized, cv::Size(frame.width, frame.height), cv::INTER_NEAREST);
            frame.depth=resized;
        }



        if ( frame.has_extra_field("is_photoneo") ){
            frame.depth*=1.0/1000;
        }
        CHECK(frame.height==frame.depth.rows) << "We are assuming we have an equal size depth otherwise we should maybe make another frame";
        CHECK(frame.width==frame.depth.cols) << "We are assuming we have an equal size depth otherwise we should maybe make another frame";
    }

}

//BLOCK functions------------------
std::shared_ptr<Frame> PRCP1Block::get_rgb_frame_with_idx( const int idx){
    CHECK(idx<m_rgb_frames.size()) << "idx is out of bounds. It is " << idx << " while m_rgb_frames has size " << m_rgb_frames.size();
    std::shared_ptr<Frame>  frame= m_rgb_frames[idx];
    return frame;
}
int PRCP1Block::nr_frames(){
    return m_rgb_frames.size();
}
//SCAN functions----------------
std::shared_ptr<PRCP1Block> PRCP1Scan::get_block_with_idx(const int idx){
    CHECK(idx<m_blocks.size()) << "idx is out of bounds. It is " << idx << " while m_blocks has size " << m_blocks.size();
    std::shared_ptr<PRCP1Block>  block = m_blocks[idx];
    return block;
}
int PRCP1Scan::nr_blocks(){
    return m_blocks.size();
}




std::shared_ptr<PRCP1Scan> DataLoaderPhenorobCP1::get_scan_with_idx(const int idx){
    CHECK(idx<m_scans.size()) << "idx is out of bounds. It is " << idx << " while m_scans has size " << m_scans.size();
    std::shared_ptr<PRCP1Scan>  scan = m_scans[idx];
    return scan;
}


std::string DataLoaderPhenorobCP1::dataset_path(){
    return m_dataset_path.string();
}
std::string DataLoaderPhenorobCP1::scan_date(){
    return m_scan_date.string();
}
std::string DataLoaderPhenorobCP1::rgb_pose_file(){
    return m_rgb_pose_file;
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

int DataLoaderPhenorobCP1::nr_scans(){
    return m_scans.size();
}

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
