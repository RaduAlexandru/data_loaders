#include "data_loaders/DataLoaderMultiFace.h"

// #include "UtilsPytorch.h" //need to include first otherwise it will overlap loguru

#include <limits>
#include <fstream>
#include <iterator>
#include <algorithm>

#include <opencv2/imgcodecs.hpp>  //for imread
#include "opencv2/imgproc/imgproc.hpp" //for cv::resize

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;


//my stuff
// #include "data_loaders/DataTransformer.h"
#include "easy_pbr/Frame.h"
#include "easy_pbr/Mesh.h"
#include "Profiler.h"
#include "string_utils.h"
#include "numerical_utils.h"
#include "opencv_utils.h"
#include "eigen_utils.h"
#include "RandGenerator.h"
// #include "easy_pbr/LabelMngr.h"
// #include "UtilsGL.h"

//json
// #include "json11/json11.hpp"


// #include <igl/point_mesh_squared_distance.h>
// #include <igl/barycentric_coordinates.h>
// #include <igl/random_points_on_mesh.h>


//cnpy
// #include "cnpy.h"


//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;


struct {
    bool operator()(fs::path a, fs::path b) const {
        std::string a_filename=a.stem().string();
        std::string b_filename=b.stem().string();
        //The files have a format of <NR> so we want to get the NR 
        int a_nr=std::stoi( a_filename );
        int b_nr=std::stoi( b_filename );
        return a_nr < b_nr;
    }
} ImgsFileComparatorFunc;

struct {
    bool operator()(fs::path a, fs::path b) const {
        std::string a_filename=a.stem().string();
        std::string b_filename=b.stem().string();
        //The files have a format of <NR> so we want to get the NR
        int a_nr=std::stoi( a_filename );
        int b_nr=std::stoi( b_filename );
        return a_nr < b_nr;
    }
} MeshFileComparatorFunc;

struct {
    bool operator()(GenesisCam& a, GenesisCam& b) const {
        return a.cam_idx < b.cam_idx;
    }
} GenesisCamIdxComparatorFunc;

DataLoaderMultiFace::DataLoaderMultiFace(const std::string config_file, const int subject_id):
    // m_is_running(false),
    m_idx_img_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator)
{
    init_params(config_file, subject_id);

    if(m_autostart){
        start();
    }

}

DataLoaderMultiFace::~DataLoaderMultiFace(){

    // m_is_running=false;
    // if (m_loader_thread.joinable()){
    //     m_loader_thread.join();
    // }
}

void DataLoaderMultiFace::init_params(const std::string config_file, const int subject_id){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_multiface"];

    m_dataset_path=(std::string)loader_config["dataset_path"];

    //get the config of the subject
    std::string subject="subject_"+std::to_string(subject_id);
    Config subject_config=loader_config["subjects"][subject];
    m_subject_name=(std::string)subject_config["subject_name"];
    m_sequence = (std::string)subject_config["sequence"];
    // m_frame_nr= subject_config["frame_nr"];
    m_timestep= subject_config["timestep"];
    m_indices_cameras_test=subject_config["test_cameras"];
    m_scene_rotate_x_angle = subject_config["scene_rotate_x_angle"];
    m_scene_translation=subject_config["scene_translation"];
    m_scene_scale_multiplier= subject_config["scene_scale_multiplier"];


    //rest of params
    m_autostart=loader_config["autostart"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_shuffle=loader_config["shuffle"];
    m_load_as_shell= loader_config["load_as_shell"];
    m_do_overfit=loader_config["do_overfit"];
    m_mode=(std::string)loader_config["mode"];
    
   

}

void DataLoaderMultiFace::start(){
    init_data_reading();
    init_poses();
    // m_tf_world_easypbr_world_mugsy=init_transforms();
    // m_tf_world_easypbr_world_mugsy_for_frames=init_transforms(true);
    // m_tf_frame_to_world_post=init_transforms(true);
    // m_tf_frame_to_world_pre.setIdentity();
    // m_tf_frame_to_world_pre.linear().col(1) = -m_tf_frame_to_world_pre.linear().col(1);
    read_data();
}


void DataLoaderMultiFace::init_data_reading(){

    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }

    //contains the path towards all the folders called  400002, 400003 etc
    fs::path cameras_path= m_dataset_path/m_subject_name/"images/"/m_sequence;
    if(!fs::is_directory(cameras_path)) {
        LOG(FATAL) << "No directory " << cameras_path;
    }
    


    //iterate through all the cameras and add the ones that are non-empty
    // std::vector<fs::path> valid_cameras_paths;
    for (fs::directory_iterator c_itr(cameras_path); c_itr!=fs::directory_iterator(); ++c_itr){
        fs::path cam_path= c_itr->path();
        if (!fs::is_empty(cam_path)    ){ 

           

            // VLOG(1) << "camera: " << cam_path.string();
            //get the camera idx from the cam_path which is something like cam400262
            int cam_idx =  std::stoi( radu::utils::erase_substring( fs::basename(cam_path), "cam" ));




            // if (m_restrict_to_cam_nr>0 && cam_idx!=m_restrict_to_cam_nr){
            //     continue;
            // }

            // VLOG(1) << "cam idx " << cam_idx;
            GenesisCam genesis_cam;
            genesis_cam.cam_idx=cam_idx;


            //get all the images from this camera
            std::vector< std::string > imgs_paths;
            //iterate through the images of this camera and get the images for the specific timestep
            // int nr_images_read=0;
            for (fs::directory_iterator i_itr(cam_path); i_itr!=fs::directory_iterator(); ++i_itr){
                fs::path img_path= i_itr->path();
                // VLOG(1) << img_path.string();
                imgs_paths.push_back( img_path.string() );
            }

            //sort them incrementally by their number
            std::sort(imgs_paths.begin(), imgs_paths.end(), ImgsFileComparatorFunc);

            genesis_cam.imgs_paths=imgs_paths;
            m_cameras.push_back(genesis_cam);

        }
    }
    CHECK(!m_cameras.empty()) << "We could not read any camera";


    //sort the cameras by their cam idx
    std::sort(m_cameras.begin(), m_cameras.end(), GenesisCamIdxComparatorFunc);



    //get the paths for the tracked meshes at each timestep
    fs::path meshes_path= m_dataset_path/m_subject_name/"tracked_mesh/"/m_sequence;
    for (fs::directory_iterator itr(meshes_path); itr!=fs::directory_iterator(); ++itr){
        fs::path mesh_path= itr->path();
        // VLOG(1) << mesh_path.string();
        //get only the obj and ignore the bin
        if (  radu::utils::contains(mesh_path.string(),"obj")  ){
            m_meshes_paths_for_timesteps.push_back(mesh_path.string());
        }
    }
    std::sort(m_meshes_paths_for_timesteps.begin(), m_meshes_paths_for_timesteps.end(), MeshFileComparatorFunc);



  


}

void DataLoaderMultiFace::init_poses(){
    //read KRT2_maya file which has format:
    // camera_idx (width height)
    // intrinsics [3x3]
    // lens distortion [1x5]
    // extrinsics [3x4]

    fs::path pose_file=m_dataset_path/m_subject_name/"KRT";
    


    //get the path to this json file
    if(!fs::is_regular_file(pose_file) ) {
        LOG(FATAL) << "Pose file for could not be found in " << pose_file;
    }


    //read the params for each camera
    std::ifstream infile( pose_file.string() );
    std::string line;
    std::vector<std::string> tokens;
    // std::getline(infile, line);
    // tokens=split(line," ");
    // int cam_idx=std::stoi(tokens[0]);
    // VLOG(1) <<"reading params for cam_idx " << cam_idx;

    while (std::getline(infile, line))
    {

        tokens=split(line," ");
        int cam_idx=std::stoi(tokens[0]);
        // int width, height;
        // if(m_capture_type!="synthetic"){ //the synthetic one doesnt have height and width
        //     width=std::stoi(tokens[1]);
        //     height=std::stoi(tokens[2]);
        // }
        // VLOG(1) <<"reading params for cam_idx " << cam_idx;

        //3 lines for intrinsics
        std::string intrinsics_string_full;
        std::getline(infile, line);  intrinsics_string_full+=line+" ";
        std::getline(infile, line);  intrinsics_string_full+=line+" ";
        std::getline(infile, line);  intrinsics_string_full+=line+" ";
        tokens=split(intrinsics_string_full," ");
        Eigen::Matrix3d K;
        K.setIdentity();
        radu::utils::tokens2matrix(tokens,K);

        // if(m_capture_type=="synthetic"){
        //     width=K(0,2)*2;
        //     height=K(1,2)*2;
        // }


        //the y principal point needs to be flipped because it actually measures the distance from the bottom but we measure the distance from the top
        // float difference_from_midpoint=height/2-K(1,2);
        // K(1,2) = height - K(1,2);

        //distorsion
        std::getline(infile, line);
        tokens=split(line," ");
        Eigen::VectorXd distorsion;
        distorsion.resize(5);
        // VLOG(1) <<"distorsion " << distorsion.rows() << " " << distorsion.cols();
        radu::utils::tokens2matrix(tokens,distorsion);
        // VLOG(1) << "distorsion is " << distorsion;


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

        // //debug
        // if (cam_idx==400357){
        //     VLOG(1) << "cam idx is " << cam_idx;
        //     VLOG(1) << "debug pose is " << pose_affine.matrix();
        //     VLOG(1) << "debug K is " << K;
        // }


        //empty line
        std::getline(infile, line);


        //push things
        m_camidx2pose[cam_idx]=pose_affine;
        m_camidx2intrinsics[cam_idx]=K;
        m_camidx2distorsion[cam_idx]=distorsion;
    }




}

Eigen::Affine3f  DataLoaderMultiFace::init_transforms(){


    Eigen::Affine3f tf_world_easypbr_world_mugsy;
    tf_world_easypbr_world_mugsy.setIdentity();


    //make a matrix that   is T,R,S  so first ti does scaling, then rotation and then translation
    Eigen::Affine3f scaling;
    scaling.setIdentity();
    scaling.matrix()(0,0)*=m_scene_scale_multiplier;
    scaling.matrix()(1,1)*=m_scene_scale_multiplier;
    scaling.matrix()(2,2)*=m_scene_scale_multiplier;
    Eigen::Affine3f rot;
    rot.setIdentity();
    Eigen::Affine3f trans;
    trans.setIdentity();

    //combine the matrices
    tf_world_easypbr_world_mugsy=trans*rot*scaling;


    return tf_world_easypbr_world_mugsy;
}

Eigen::Affine3f DataLoaderMultiFace::transform_from_world_mugsy_to_world_easypbr(const Eigen::Affine3f& tf_world_obj, const bool do_scaling){
    //we do the transofmrations in this order, first, scaling, then rotating and then translating
    Eigen::Affine3f new_tf_world_obj=tf_world_obj;

    //Scaling
    //affects only the translation
    if (do_scaling){
        float s=m_scene_scale_multiplier;
        new_tf_world_obj.translation()*=s;
    }

    //Rotation
    Eigen::Affine3f tf_rot;
    tf_rot.setIdentity();
    Eigen::Matrix3f mat_rot;
    mat_rot = Eigen::AngleAxisf(radu::utils::degrees2radians(m_scene_rotate_x_angle), Eigen::Vector3f::UnitX());
    tf_rot.matrix().block<3,3>(0,0)=mat_rot;
    new_tf_world_obj=tf_rot*new_tf_world_obj;

    //translation
    new_tf_world_obj.translation()+=m_scene_translation;

    return new_tf_world_obj;
}

void DataLoaderMultiFace::read_data(){
    CHECK(m_timestep>=0) << "Timestep should be positive or zero. It is " << m_timestep;



    //read mesh
    CHECK(m_timestep<(int)m_meshes_paths_for_timesteps.size()) << "Timestep should be less that the nr of meshes. Timestep is " << m_timestep << " nr of meshes for all timesteps is "<< m_meshes_paths_for_timesteps.size();
    


    // MeshSharedPtr mesh=Mesh::create();
    // mesh->read_obj(m_meshes_paths_for_timesteps[m_timestep], true, false);
    // //place in world
    // Eigen::Affine3f tf_world_obj =mesh->model_matrix().cast<float>();
    // Eigen::Affine3f tf_obj_world;
    // {
    //     Eigen::Affine3f tf_rot;
    //     tf_rot.setIdentity();
    //     Eigen::Matrix3f mat_rot;
    //     mat_rot = Eigen::AngleAxisf(radu::utils::degrees2radians(m_scene_rotate_x_angle), Eigen::Vector3f::UnitX());
    //     tf_rot.matrix().block<3,3>(0,0)=mat_rot;
    //     tf_world_obj=tf_rot*tf_world_obj;
    //     tf_obj_world=tf_world_obj.inverse();
    //     //apply
    //     // tf_obj_world.translation()*=m_scene_scale_multiplier;
    //     // tf_obj_world.translation()+=m_scene_translation;
    //     mesh->set_model_matrix(tf_obj_world.cast<double>());
    //     mesh->apply_model_matrix_to_cpu(true);
    //     mesh->scale_mesh(m_scene_scale_multiplier);
    //     mesh->translate_model_matrix(m_scene_translation.cast<double>());
    // }
    // m_mesh_for_timestep=mesh;

    //attempt 2
    MeshSharedPtr mesh=Mesh::create();
    mesh->read_obj(m_meshes_paths_for_timesteps[m_timestep], true, false);
    mesh->scale_mesh(m_scene_scale_multiplier);
    mesh->apply_model_matrix_to_cpu(true);
    //place in world
    Eigen::Affine3f tf_world_obj =mesh->model_matrix().cast<float>();
    tf_world_obj=transform_from_world_mugsy_to_world_easypbr(tf_world_obj, false); //we don't scale here because the scaling of translation doesn't make a any difference here because we already have a translation of 0,0,0
    mesh->set_model_matrix(tf_world_obj.cast<double>());
    mesh->apply_model_matrix_to_cpu(true);
    m_mesh_for_timestep=mesh;
   

    // for (size_t i = 0; i < m_imgs_paths.size(); i++){
    int nr_cameras_valid=0;

    for (size_t i = 0; i < m_cameras.size(); i++){

        int cam_idx=m_cameras[i].cam_idx;
        // VLOG(1) << "readin from cam_idx" << cam_idx;
        // VLOG(1) << "this cam has nr of imgs " << m_cameras[i].imgs_paths.size();
        CHECK(m_timestep<(int)m_cameras[i].imgs_paths.size()) << "Timestep should be less that the nr of images. Timestep is " << m_timestep << " nr of images for this cam is "<< m_cameras[i].imgs_paths.size();


        //see if we add this camera based on the mode that we are in
        //we do the check HERE because if we were do discard the cameras before they are even read, then the cam_id_lin is wrong
        bool is_cam_in_test_list= (std::find(m_indices_cameras_test.begin(), m_indices_cameras_test.end(), cam_idx) != m_indices_cameras_test.end());
        if (m_mode=="test" && !is_cam_in_test_list){
            continue;
        }else if (m_mode=="train" && is_cam_in_test_list){
            continue;
        }


        Frame frame;
        frame.cam_id=cam_idx;
        frame.add_extra_field("cam_id_lin", (int)i);
        frame.frame_idx=m_frames.size();
        fs::path img_path=m_cameras[i].imgs_paths[m_timestep];
        frame.rgb_path=img_path.string();
        //add also a maks path
        fs::path img_filename=img_path.filename();
        // fs::path mask_path=m_hair_recon_path/"segmentation/predictions/segmentation"/ ("cam"+std::to_string(cam_idx))/ img_filename;
        // frame.mask_path=mask_path.string();


        if (nr_cameras_valid!=0 and m_do_overfit){ //if we are overfitting, we jsut read one camera
            continue;
        }
        // VLOG(1) << "reading " << img_path;



        //load the images if necessary or delay it for whne it's needed
        frame.load_images=[this]( easy_pbr::Frame& frame ) -> void{ this->load_images_in_frame(frame); };
        if (m_load_as_shell){
            //set the function to load the images whenever it's neede
            frame.is_shell=true;
        }else{
            frame.is_shell=false;
            frame.load_images(frame);
        }

        // VLOG(1) << "Cam idx is " << cam_idx;

        //extrinsics
        Eigen::Affine3f tf_cam_world = m_camidx2pose[cam_idx].cast<float>();
        Eigen::Affine3f tf_world_cam= tf_cam_world.inverse();


        //attempt 3
        // tf_world_cam=m_tf_frame_to_world_post*tf_world_cam * m_tf_frame_to_world_pre;
        // tf_world_cam=m_tf_frame_to_world_post*tf_world_cam * m_tf_frame_to_world_pre;
        // tf_cam_world = tf_world_cam.inverse();

        //attmepmt 4
        // VLOG(1) << "m_tf_world_easypbr_world_mugsy" << m_tf_world_easypbr_world_mugsy.matrix();
        // VLOG(1) << "tf_cam_world before" << tf_cam_world.matrix();
        // tf_world_cam=m_tf_world_easypbr_world_mugsy*tf_world_cam;
        // tf_cam_world = tf_world_cam.inverse();
        // VLOG(1) << "tf_cam_world after" << tf_cam_world.matrix();


        //attempt 5
        //rotate 180 degrees in x because things seem to be upside down
        // Eigen::Affine3f tf_rot;
        // tf_rot.setIdentity();
        // Eigen::Matrix3f mat_rot;
        // mat_rot = Eigen::AngleAxisf(radu::utils::degrees2radians(m_scene_rotate_x_angle), Eigen::Vector3f::UnitX());
        // tf_rot.matrix().block<3,3>(0,0)=mat_rot;
        // tf_world_cam=tf_rot*tf_world_cam;
        // tf_cam_world=tf_world_cam.inverse();
        // //apply
        // tf_cam_world.translation()*=m_scene_scale_multiplier;
        // // tf_cam_world.translation()+=m_scene_translation;
        // tf_world_cam= tf_cam_world.inverse();
        // tf_world_cam.translation()+=m_scene_translation;
        // tf_cam_world=tf_world_cam.inverse();
        // frame.tf_cam_world=tf_cam_world;



        //attempt 6
        // tf_world_cam=m_tf_world_easypbr_world_mugsy*tf_world_cam;
        // tf_cam_world=tf_world_cam.inverse();
        // frame.tf_cam_world=tf_cam_world;


        //attempt 7
        tf_world_cam=transform_from_world_mugsy_to_world_easypbr(tf_world_cam, true);
        tf_cam_world=tf_world_cam.inverse();
        frame.tf_cam_world=tf_cam_world;



        //intrinsics got mostly from here https://github.com/bmild/nerf/blob/0247d6e7ede8d918bc1fab2711f845669aee5e03/load_blender.py
        frame.K=m_camidx2intrinsics[cam_idx].cast<float>();
        if(m_subsample_factor>1){
            //based on the post from tomas simon and https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
            frame.rescale_K(1.0/m_subsample_factor);
        }

        //distorsion
        frame.distort_coeffs=m_camidx2distorsion[cam_idx].cast<float>();


        m_frames.push_back(frame);

        nr_cameras_valid++;

    }


}

// MeshSharedPtr DataLoaderMultiFace::read_mesh(const std::string path, bool load_texture, bool transform, bool check_frame_nr){

//     CHECK( fs::exists(fs::path(path)) ) << "Could not find path " << path;
//     if ( check_frame_nr ){
//         CHECK( radu::utils::contains(path, std::to_string(m_frame_nr))  ) <<"The path to the mesh has to contain the correct frame nr that we are trying to load. Something is weird. We are trying to load path " << path << " but the frame nr to load is " << m_frame_nr;
//     }

//     MeshSharedPtr mesh=Mesh::create();

//     // mesh->load_from_file(m_meshes_paths_for_timesteps[m_timestep]);
//     std::string file_ext = path.substr(path.find_last_of(".") + 1);
//     if (file_ext == "obj" || file_ext == "OBJ"){
//         mesh->read_obj(path, true, false);
//     }else{
//         mesh->load_from_file(path);
//     }

//     if (transform){
//         mesh=transform_mesh_from_mugsy_to_easypbr(mesh);
//     }
//     mesh->apply_model_matrix_to_cpu(true);
//     if (load_texture){
//         //get a texture as the average texture
//         std::string avg_tex_path;
//         if (m_capture_type=="minisis"){
//             avg_tex_path= (m_dataset_path/"minisis/processed_GHS_v2/codec/tex_mean.png").string();
//         }else if(m_capture_type=="synthetic"){
//             avg_tex_path= (m_dataset_path/"gt/texture.png_mean.png").string();
//         }else if(m_capture_type=="genesis"){
//             avg_tex_path= (m_dataset_path/"processed_GHS_v2/codec/tex_mean.png").string();
//         }
//         mesh->set_diffuse_tex(avg_tex_path);
//     }



//     return mesh;
// }

// std::shared_ptr<GenesisHair> DataLoaderMultiFace::read_hair_recon(const std::string path_bin_file){
//     //bin file contains the strands xyz and normal and so on. The script to read it is here:
//     // https://ghe.oculus-rep.com/giljoonam/HairReconstruction/blob/master/scripts/kevin/bin2npy.py

//     //return a mesh of the points on the strands, and also a vector of strands
//     std::shared_ptr<GenesisHair> strand_data(new GenesisHair);

//     std::vector<Eigen::Vector3f> full_points_vec;

//     std::ifstream input( path_bin_file, std::ios::binary );
//     int nr_strands=0;
//     int nr_strands_added=0;

//     input.read(reinterpret_cast<char*>(&nr_strands), sizeof(int));
//     // VLOG(1) << "nr strands " << nr_strands;
//     for(int i=0; i<nr_strands; i++){

//         // bool is_strand_valid=true;
//         // //randomly drop some strands
//         // if (m_rand_gen->rand_bool(m_percentage_strand_drop)){
//         //     is_strand_valid=false;
//         // }



//         int num_verts;
//         input.read(reinterpret_cast<char*>(&num_verts), sizeof(int));
//         // num_verts = struct.unpack('i', file.read(4))[0]
//         // strand = np.zeros((num_verts, 6), dtype=np.float32)
//         std::vector<Eigen::Vector3f> points_vec;
//         std::vector<Eigen::Vector3f> normals_vec;

//         for(int j=0; j<num_verts; j++){
//             Eigen::Vector3f point, normal;
//             // point.resize(3);
//             // normal.resize(3);
//             float label;
//             // x = struct.unpack('f', file.read(4))[0]
//             // y = struct.unpack('f', file.read(4))[0]
//             // z = struct.unpack('f', file.read(4))[0]
//             // nx = struct.unpack('f', file.read(4))[0]
//             // ny = struct.unpack('f', file.read(4))[0]
//             // nz = struct.unpack('f', file.read(4))[0]
//             // label = struct.unpack('f', file.read(4))[0]
//             input.read(reinterpret_cast<char*>(&point.x()), sizeof(float));
//             input.read(reinterpret_cast<char*>(&point.y()), sizeof(float));
//             input.read(reinterpret_cast<char*>(&point.z()), sizeof(float));
//             //normal
//             input.read(reinterpret_cast<char*>(&normal.x()), sizeof(float));
//             input.read(reinterpret_cast<char*>(&normal.y()), sizeof(float));
//             input.read(reinterpret_cast<char*>(&normal.z()), sizeof(float));
//             //label
//             input.read(reinterpret_cast<char*>(&label), sizeof(float));
//             // VLOG(1) << "point is " << point;
//             // VLOG(1) << "normal is " << normal;
//             // VLOG(1) << "label is " << label;

//             points_vec.push_back(point);
//             normals_vec.push_back(normal);
//             full_points_vec.push_back(point); ////for the full hair




//             //DEBUG
//             // VLOG(1) << "dim is " << point.size();
//             // break;
//         }

//         //make it into a mesh
//         // if (is_strand_valid){
//         MeshSharedPtr mesh=Mesh::create();
//         Eigen::MatrixXf points = vec2eigen(points_vec);
//         // VLOG(1) << " points has rows and cols " << points.rows() << " x " << points.cols();
//         mesh->V=points.cast<double>();
//         mesh=transform_mesh_from_mugsy_to_easypbr(mesh);
//         //assign also a random color for each strand
//         Eigen::MatrixXd rand_color= Eigen::MatrixXd::Random(1,3);
//         mesh->C=rand_color.replicate(points.size(), 1);
//         strand_data->strand_meshes.push_back(mesh);
//         nr_strands_added++;
//         // }


//     }

//     strand_data->generate_full_hair_cloud();
//     if(m_load_consistent_hair_directions){
//         fs::path path_for_consistent_dirs=m_hair_recon_path/"/line_mvs/consistent_directions.ply";
//         MeshSharedPtr mesh_hair_with_consistent_directions=Mesh::create(path_for_consistent_dirs.string());
//         mesh_hair_with_consistent_directions=transform_mesh_from_mugsy_to_easypbr(mesh_hair_with_consistent_directions);
//         Eigen::MatrixXf dirs=mesh_hair_with_consistent_directions->NV.cast<float>();
//         strand_data->per_point_direction_to_next_tensor = eigen2tensor(dirs);
//         strand_data->per_point_direction_to_next_tensor=strand_data->per_point_direction_to_next_tensor.squeeze(0);
//     }else{
//         strand_data->generate_per_point_directions();
//     }








//     return strand_data;

// }



// void DataLoaderMultiFace::compute_root_points_atributes(Eigen::MatrixXd& uv, std::vector<Eigen::Matrix3d>& tbn_per_point, std::shared_ptr<easy_pbr::Mesh> mesh, std::vector<Eigen::Vector3d> points_vec){

//     Eigen::MatrixXd points=vec2eigen(points_vec);

//     Eigen::VectorXd distances;
//     Eigen::MatrixXd closest_points;
//     Eigen::VectorXi closest_face_indices;
//     // VLOG(1) << "points mesh squared";
//     igl::point_mesh_squared_distance(points, mesh->V, mesh->F, distances, closest_face_indices, closest_points );
//     // VLOG(1) << "finished points mesh squared";

//     // VLOG(1) << "points" << points.rows() << "x " << points.cols();
//     // VLOG(1) << "distances" << distances.rows() << "x " << distances.cols();
//     // VLOG(1) << "closest_points" << closest_points.rows() << "x " << closest_points.cols();
//     // VLOG(1) << "closest_face_indices" << closest_face_indices.rows() << "x " << closest_face_indices.cols();

//     //get the uv
//     // Eigen::MatrixXd UV;
//     uv.resize(points.rows(), 2);
//     //get also the TBN in world coords
//     tbn_per_point.resize( points.rows() );

//     for(int i=0; i<points.rows(); i++){
//         // VLOG(1) << "getting atributes for point " << i << " out of " << points.rows();
//         Eigen::MatrixXd closest_point = closest_points.row(i);
//         int face_idx=closest_face_indices(i);
//         Eigen::Vector3i face=mesh->F.row(face_idx);
//         int idx_p0 = face.x();
//         int idx_p1 = face.y();
//         int idx_p2 = face.z();
//         Eigen::Vector3d p0 = mesh->V.row(idx_p0);
//         Eigen::Vector3d p1 = mesh->V.row(idx_p1);
//         Eigen::Vector3d p2 = mesh->V.row(idx_p2);

//         Eigen::MatrixXd barycentric;
//         igl::barycentric_coordinates(closest_point, p0.transpose(), p1.transpose(), p2.transpose(), barycentric);

//         float b0=barycentric(0,0);
//         float b1=barycentric(0,1);
//         float b2=barycentric(0,2);

//         // if (i==0){
//         //     VLOG(1) << " baryc is " << b0 << " " << b1 << " " << b2;
//         //     VLOG(1) << " idx_p0 is " << idx_p0 << " " << idx_p1 << " " << idx_p2;
//         //     VLOG(1) << "idx0 uv " << mesh->UV.row(idx_p0);
//         // }

//         Eigen::Vector2d uv_for_point = mesh->UV.row(idx_p0)*b0 + mesh->UV.row(idx_p1)*b1 + mesh->UV.row(idx_p2)*b2;

//         uv.row(i) = uv_for_point;

//         //get also the TBN per point
//         Eigen::Vector3d T,B,N;
//         N= mesh->NV.row(idx_p0)*b0 + mesh->NV.row(idx_p1)*b1 + mesh->NV.row(idx_p2)*b2;
//         T= mesh->V_tangent_u.row(idx_p0)*b0 + mesh->V_tangent_u.row(idx_p1)*b1 + mesh->V_tangent_u.row(idx_p2)*b2;
//         N.normalize();
//         T.normalize();
//         B=N.cross(T);
//         Eigen::Vector3d Tw,Bw,Nw;
//         Tw=T;
//         Bw=B;
//         Nw=N;
//         CHECK(!N.isZero()) << "N is zero why ";
//         CHECK(!T.isZero()) << "T is zero why ";
//         CHECK(!B.isZero()) << "B is zero why ";
//         //rotate from model coordinates to world
//         T=mesh->model_matrix().linear()*Tw;
//         B=mesh->model_matrix().linear()*Bw;
//         N=mesh->model_matrix().linear()*Nw;
//         CHECK(!N.isZero()) << "N is zero why " << " linear is " << mesh->model_matrix().linear() << " Nw is " << Nw << " n is " << N;
//         CHECK(!T.isZero()) << "T is zero why " << " linear is " << mesh->model_matrix().linear() << " Tw is " << Tw << " t is " << T;
//         CHECK(!B.isZero()) << "B is zero why " << " linear is " << mesh->model_matrix().linear() << " Bw is " << Bw << " n is " << B;
//         Eigen::Matrix3d TBN;
//         TBN.col(0)=T;
//         TBN.col(1)=B;
//         TBN.col(2)=N;
//         CHECK(!TBN.isZero()) << "TBN is zero! " << TBN << " i is " << i << " points is " << points.rows();
//         tbn_per_point[i] = TBN;
//         CHECK(!tbn_per_point[i].col(0).isZero()) << "tbn_per_point[i] col0 is zero! " << tbn_per_point[i] << " i is " << i << " tbn roots size is " <<tbn_per_point.size();
//         CHECK(!tbn_per_point[i].col(1).isZero()) << "tbn_per_point[i] col1 is zero! " << tbn_per_point[i] << " i is " << i << " tbn roots size is " <<tbn_per_point.size();
//         CHECK(!tbn_per_point[i].col(2).isZero()) << "tbn_per_point[i] col2 is zero! " << tbn_per_point[i] << " i is " << i << " tbn roots size is " <<tbn_per_point.size();



//     }


//     // //show the mesh
//     // std::shared_ptr<easy_pbr::Mesh> closest_mesh= easy_pbr::Mesh::create();
//     // // closest_mesh->V= closest_points;
//     // closest_mesh->V= mesh->V;
//     // closest_mesh->F.resize(1,3);
//     // closest_mesh->F.row(0) =  mesh->F.row(closest_face_indices(0));
//     // easy_pbr::Scene::show(closest_mesh,"closest_mesh");

//     // return UV;

//     // uv=UV;


// }

// std::shared_ptr<GenesisHair> DataLoaderMultiFace::get_random_roots(const int nr_strands){
//     //create random points that are on the scalp mesh

//     Eigen::MatrixXd barycentric;
//     Eigen::MatrixXi face_indices; //nr_strand x 1 indices onto F

//     // VLOG(1) << "straitng igl::random_points_on_mesh ";
//     igl::random_points_on_mesh(nr_strands, m_mesh_scalp_for_timestep->V, m_mesh_scalp_for_timestep->F, barycentric, face_indices);
//     // VLOG(1) << "got igl::random_points_on_mesh ";
//     //get the points
//     std::vector<Eigen::Vector3d> points_vec;
//     for(int i=0; i<nr_strands; i++){
//         // VLOG(1) << "strand " << i;
//         Eigen::Vector3d point;
//         point.setZero();
//         int face_index= face_indices(i);
//         int vertex_index_0 = m_mesh_scalp_for_timestep->F(face_index, 0);
//         int vertex_index_1 = m_mesh_scalp_for_timestep->F(face_index, 1);
//         int vertex_index_2 = m_mesh_scalp_for_timestep->F(face_index, 2);
//         float barycentric_0= barycentric(i, 0);
//         float barycentric_1= barycentric(i, 1);
//         float barycentric_2= barycentric(i, 2);
//         point=  barycentric_0*m_mesh_scalp_for_timestep->V.row(vertex_index_0) +
//                 barycentric_1*m_mesh_scalp_for_timestep->V.row(vertex_index_1) +
//                 barycentric_2*m_mesh_scalp_for_timestep->V.row(vertex_index_2);
//         points_vec.push_back(point);
//     }


//     //from the points we sampled on the scalp, get the uv towards the bald mesh, afterwards, samples the bald mesh at it's uv coordinates to check if we are within the hairline
//     std::string hairline_tex_path= (m_hair_recon_path/ "/textures/exact_harline.png").string();
//     cv::Mat harline_mat;
//     harline_mat=cv::imread(hairline_tex_path);
//     // VLOG(1) << "hairline mat is " << type2string(harline_mat.type());
//     CHECK(!harline_mat.empty()) <<"Could not find exact hairline texture at" << hairline_tex_path;
//     Eigen::MatrixXd uv_roots_bald_mesh;
//     std::vector<Eigen::Matrix3d> tbn_roots_bald_mesh;
//     compute_root_points_atributes(uv_roots_bald_mesh, tbn_roots_bald_mesh, m_mesh_head_bald_for_timestep, points_vec);
//     //filter
//     std::vector<Eigen::Vector3d> points_filtered_vec;
//     for(int i=0; i<nr_strands; i++){
//         Eigen::Vector2d uv=uv_roots_bald_mesh.row(i);
//         int x=uv.x()*harline_mat.cols;
//         int y=(1-uv.y())*harline_mat.rows;
//         if(x<harline_mat.cols && y<harline_mat.rows && x>=0 && y>=0){
//             float val=harline_mat.at<cv::Vec3b>(y, x)[0];
//             if (val>0){
//                 points_filtered_vec.push_back(points_vec[i]);
//             }

//         }
//     }
//     points_vec=points_filtered_vec;





//     std::shared_ptr<GenesisHair> hair(new GenesisHair);
//     Eigen::MatrixXd uv_roots;
//     std::vector<Eigen::Matrix3d> tbn_roots;
//     // VLOG(1) << "computing root points atributes ";
//     compute_root_points_atributes(uv_roots, tbn_roots, m_mesh_scalp_for_timestep, points_vec);
//     // VLOG(1) << "got root points atributes ";
//     hair->position_roots=vec2eigen(points_vec);
//     hair->uv_roots=uv_roots;


//     //tbn roots to tensor
//     hair->tbn_roots_tensor = torch::empty({ (long int)points_vec.size(),3,3 }, torch::dtype(torch::kFloat32) );
//     auto tbn_roots_tensor_accesor = hair->tbn_roots_tensor.accessor<float,3>();
//     for(int i=0; i<tbn_roots.size(); i++){
//         // //row 0
//         tbn_roots_tensor_accesor[i][0][0]=tbn_roots[i](0,0);
//         tbn_roots_tensor_accesor[i][0][1]=tbn_roots[i](0,1);
//         tbn_roots_tensor_accesor[i][0][2]=tbn_roots[i](0,2);
//         //row 1
//         tbn_roots_tensor_accesor[i][1][0]=tbn_roots[i](1,0);
//         tbn_roots_tensor_accesor[i][1][1]=tbn_roots[i](1,1);
//         tbn_roots_tensor_accesor[i][1][2]=tbn_roots[i](1,2);
//         //row 2
//         tbn_roots_tensor_accesor[i][2][0]=tbn_roots[i](2,0);
//         tbn_roots_tensor_accesor[i][2][1]=tbn_roots[i](2,1);
//         tbn_roots_tensor_accesor[i][2][2]=tbn_roots[i](2,2);
//     }

//     return hair;
// }

// Eigen::MatrixXd DataLoaderMultiFace::compute_uv_wrt_mesh( std::shared_ptr<easy_pbr::Mesh> mesh, Eigen::MatrixXd& points ){

//     //get points to a vector
//     std::vector<Eigen::Vector3d> points_vec(points.rows());
//     for (int i = 0; i < points.rows(); ++i) {
//         points_vec[i] =  points.row(i);
//     }


//     Eigen::MatrixXd uv_roots_bald_mesh;
//     std::vector<Eigen::Matrix3d> tbn_roots_bald_mesh;
//     compute_root_points_atributes(uv_roots_bald_mesh, tbn_roots_bald_mesh, mesh, points_vec);

//     return uv_roots_bald_mesh;
// }



std::shared_ptr<easy_pbr::Mesh> DataLoaderMultiFace::transform_mesh_from_mugsy_to_easypbr(std::shared_ptr<easy_pbr::Mesh> mesh){



    // // // flio the model matrix in the same way we modified the frame of the image
    // Eigen::Affine3f tf_obj_world = mesh->model_matrix().inverse().cast<float>();
    // //flip z
    // tf_obj_world.matrix().col(2) = -tf_obj_world.matrix().col(2);
    // // //flip y
    // tf_obj_world.matrix().col(1) = -tf_obj_world.matrix().col(1);
    // // //flip y locally
    // Eigen::Affine3f tf_world_obj= tf_obj_world.inverse();


    // // //add scene translation
    // tf_world_obj.translation()+=m_scene_translation;
    // Eigen::Affine3f scaling;
    // scaling.setIdentity();
    // scaling.matrix()(0,0)*=m_scene_scale_multiplier;
    // scaling.matrix()(1,1)*=m_scene_scale_multiplier;
    // scaling.matrix()(2,2)*=m_scene_scale_multiplier;
    // tf_world_obj=scaling*tf_world_obj;


    // // mesh->set_model_matrix(tf_world_obj.cast<double>() );
    mesh->transform_model_matrix(m_tf_world_easypbr_world_mugsy.cast<double>() );
    mesh->apply_model_matrix_to_cpu(true);



    return mesh;

}

std::shared_ptr<easy_pbr::Mesh> DataLoaderMultiFace::transform_mesh_from_easypbr_to_mugsy(std::shared_ptr<easy_pbr::Mesh> mesh){

    mesh->transform_model_matrix(m_tf_world_easypbr_world_mugsy.inverse().cast<double>() );
    mesh->apply_model_matrix_to_cpu(true);


    return mesh;
}


void DataLoaderMultiFace::load_images_in_frame(easy_pbr::Frame& frame){

    // CHECK( radu::utils::contains(frame.rgb_path, std::to_string(m_frame_nr))  ) <<"The path to the rgb has to contain the correct frame nr that we are trying to load. Something is weird. We are trying to load path " << frame.rgb_path << " but the frame nr to load is " << m_frame_nr;
    // CHECK( radu::utils::contains(frame.mask_path, std::to_string(m_frame_nr))  ) <<"The path to the mask has to contain the correct frame nr that we are trying to load. Something is weird. We are trying to load path " << frame.mask_path << " but the frame nr to load is " << m_frame_nr;

    frame.is_shell=false;



    //read rgba and split into rgb and alpha mask
    cv::Mat rgb_32f;
    // if (m_load_as_float){
    //     rgb_32f = cv::imread( frame.rgb_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH );
    //     ///resize the float mat
    //     if(m_subsample_factor>1){
    //         cv::Mat resized;
    //         cv::resize(rgb_32f, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
    //         rgb_32f=resized;
    //     }
    // }else{
        cv::Mat rgb_8u = cv::imread( frame.rgb_path );
        //resize the rgb8u mat and then convert to float because its faster
        if(m_subsample_factor>1){
            cv::Mat resized;
            cv::resize(rgb_8u, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
            rgb_8u=resized;
        }
        // frame.rgb_8u=rgb_8u;
        rgb_8u.convertTo(rgb_32f, CV_32FC3, 1.0/255.0);
    // }


    frame.rgb_32f= rgb_32f;
    

    frame.width=frame.rgb_32f.cols;
    frame.height=frame.rgb_32f.rows;



}

std::shared_ptr<easy_pbr::Mesh>  DataLoaderMultiFace::get_mesh_head(){

    return m_mesh_for_timestep;
}
// std::shared_ptr<easy_pbr::Mesh>  DataLoaderMultiFace::get_mesh_head_bald(){
//     CHECK(m_load_bald_mesh) << "load_bald_mesh should be true in the cfg file";
//     return m_mesh_head_bald_for_timestep;
// }
// std::shared_ptr<easy_pbr::Mesh>  DataLoaderMultiFace::get_mesh_shoulders(){
//     CHECK(m_load_shoulders_mesh) << "load_shoulders_mesh should be true in the cfg file";
//     return m_mesh_shoulders_for_timestep;
// }


// std::shared_ptr<easy_pbr::Mesh>  DataLoaderMultiFace::get_mesh_scalp(){
//     CHECK(m_load_scalp) << "load_scalp should be true in the cfg file";
//     return m_mesh_scalp_for_timestep;
// }
// std::shared_ptr<easy_pbr::Mesh>  DataLoaderMultiFace::get_mesh_hair(){
//     return m_mesh_hair_for_timestep;
// }
// std::shared_ptr<GenesisHair>  DataLoaderMultiFace::get_hair(){
//     return m_hair_for_timestep;
// }



Frame DataLoaderMultiFace::get_next_frame(){
    CHECK(m_idx_img_to_read<(int)m_frames.size()) << "m_idx_img_to_read is out of bounds. It is " << m_idx_img_to_read << " while m_frames has size " << m_frames.size();
    Frame  frame= m_frames[m_idx_img_to_read];

    if(!m_do_overfit){
        m_idx_img_to_read++;
    }

    return frame;
}
std::vector<easy_pbr::Frame> DataLoaderMultiFace::get_all_frames(){
    return m_frames;
}
// easy_pbr::Frame DataLoaderMultiFace::get_frame_for_cam_id( const int cam_id){
//    for (size_t i = 0; i < m_frames.size(); i++){
//        if (m_frames[i].cam_id==cam_id){
//            return m_frames[i];
//        }
//    }
//    LOG(FATAL) << "Could not find cam_id " << cam_id << " m_frames has size " << m_frames.size();

// }
Frame DataLoaderMultiFace::get_frame_at_idx( const int idx){
    CHECK(idx<(int)m_frames.size()) << "idx is out of bounds. It is " << idx << " while m_frames has size " << m_frames.size();

    Frame  frame= m_frames[idx];

    return frame;
}

Frame DataLoaderMultiFace::get_random_frame(){
    CHECK(m_frames.size()>0 ) << "m_frames has size 0";

    int random_idx=m_rand_gen->rand_int(0, m_frames.size()-1);
    Frame  frame= m_frames[random_idx];

    return frame;
}
// Frame DataLoaderMultiFace::get_closest_frame( const easy_pbr::Frame& frame){

//     float closest_distance=std::numeric_limits<float>::max();
//     int closest_idx=-1;
//     for(size_t i=0; i<m_frames.size(); i++){
//         float dist =  ( m_frames[i].tf_cam_world.inverse().translation() - frame.tf_cam_world.inverse().translation() ).norm();
//         if (dist < closest_distance && dist>1e-7){
//             closest_distance=dist;
//             closest_idx=i;
//         }
//     }

//     Frame  frame_closest= m_frames[closest_idx];

//     return frame_closest;

// }


// std::vector<easy_pbr::Frame>  DataLoaderMultiFace::get_close_frames( const easy_pbr::Frame& frame, const int nr_frames, const bool discard_same_idx){

//     CHECK(nr_frames<m_frames.size()) << "Cannot select more close frames than the total nr of frames that we have in the loader. Required select of " << nr_frames << " out of a total of " << m_frames.size() << " available in the loader";

//     std::vector<easy_pbr::Frame> selected_close_frames;

//     for(size_t i=0; i<nr_frames; i++){

//         //select a close frame
//         float closest_distance=std::numeric_limits<float>::max();
//         int closest_idx=-1;
//         for(size_t j=0; j<m_frames.size(); j++){

//             //ignore if the current frame we are checking is THIS
//             if (discard_same_idx){
//                 if( m_frames[j].frame_idx == frame.frame_idx ){
//                     continue;
//                 }
//             }

//             //ignore the current frame that we are checking if it's any of the ones already selected
//             bool is_already_selected=false;
//             for(size_t k=0; k<selected_close_frames.size(); k++){
//                 if( m_frames[j].frame_idx == selected_close_frames[k].frame_idx ){
//                     is_already_selected=true;
//                 }
//             }
//             if(is_already_selected){
//                 continue;
//             }

//             //now get the closest one
//             float dist =  ( m_frames[j].tf_cam_world.inverse().translation() - frame.tf_cam_world.inverse().translation() ).norm();
//             // float dist = 1.0 -  m_frames[j].look_dir().dot(frame.look_dir());
//             if (dist < closest_distance){
//                 closest_distance=dist;
//                 closest_idx=j;
//             }
//         }

//         Frame  frame_closest= m_frames[closest_idx];
//         selected_close_frames.push_back(frame_closest);


//     }


//     return selected_close_frames;


// }

// std::vector<int> DataLoaderMultiFace::get_cam_indices(){
//     std::vector<int> cam_indices;
//     for (size_t i = 0; i < m_frames.size(); i++){
//         cam_indices.push_back( m_frames[i].cam_id );
//     }

//     return cam_indices;
// }

// std::vector<int> DataLoaderMultiFace::get_cam_indices_lin(){
//     std::vector<int> cam_indices;
//     for (size_t i = 0; i < m_frames.size(); i++){
//         cam_indices.push_back( m_frames[i].get_extra_field<int>("cam_id_lin") );
//     }

//     return cam_indices;
// }

// // //compute weights
// // std::vector<float> DataLoaderMultiFace::compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames){
// //     // https://people.cs.clemson.edu/~dhouse/courses/404/notes/barycentric.pdf
// //     // https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
// //     // https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle

// //     //to compute the weights we use barycentric coordinates.
// //     //this has several steps, first project the current frame into the triangle defiend by the close_frames.
// //     //compute barycentric coords
// //     //if the barycentric coords are not within [0,1], clamp them

// //     //checks
// //     CHECK(close_frames.size()==3) <<"This assumes we are using 3 frames as close frames because we want to compute barycentric coords";

// //     //make triangle
// //     Eigen::Vector3d cur_pos= frame.pos_in_world().cast<double>();
// //     Eigen::Vector3d p1= close_frames[0].pos_in_world().cast<double>();
// //     Eigen::Vector3d p2= close_frames[1].pos_in_world().cast<double>();
// //     Eigen::Vector3d p3= close_frames[2].pos_in_world().cast<double>();

// //     //get barycentirc coords of the projection https://math.stackexchange.com/a/544947
// //     Eigen::Vector3d u=p2-p1;
// //     Eigen::Vector3d v=p3-p1;
// //     Eigen::Vector3d n=u.cross(v);
// //     Eigen::Vector3d w=cur_pos-p1;

// //     float w_p3= u.cross(w).dot(n)/ (n.dot(n));
// //     float w_p2= w.cross(v).dot(n)/ (n.dot(n));
// //     float w_p1= 1.0-w_p2-w_p3;

// //     //to get weights as if the point was inside the triangle, we clamp the barycentric coordinates (I don't know if this is needed yeat)

// //     //return tha values
// //     std::vector<float> vals;
// //     vals.push_back(w_p1);
// //     vals.push_back(w_p2);
// //     vals.push_back(w_p3);

// //     return vals;


// // }





bool DataLoaderMultiFace::is_finished(){
    //check if this loader has returned all the images it has
    if(m_idx_img_to_read<(int)m_frames.size()){
        return false; //there is still more files to read
    }


    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderMultiFace::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle && m_mode=="train"){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_frames), std::end(m_frames), rng_0);
    }

    m_idx_img_to_read=0;
}

int DataLoaderMultiFace::nr_samples(){
    return m_frames.size();
}
// int DataLoaderMultiFace::nr_cameras(){
//     return m_cameras.size();
// }


bool DataLoaderMultiFace::has_data(){
    return true; //we always have data since the loader stores all the image in memory and keeps them there
}

void DataLoaderMultiFace::set_mode_train(){
    m_mode="train";
}
void DataLoaderMultiFace::set_mode_test(){
    m_mode="test";
}
void DataLoaderMultiFace::set_mode_validation(){
    m_mode="val";
}
void DataLoaderMultiFace::set_mode_all(){
    m_mode="all";
}
void DataLoaderMultiFace::set_dataset_path(const std::string dataset_path){
    m_dataset_path=dataset_path;
}

void DataLoaderMultiFace::set_subsample_factor(const int val){
    m_subsample_factor=val;
}
// std::string DataLoaderMultiFace::sequence(){
//     return m_sequence;
// }
// std::string DataLoaderMultiFace::dataset_path(){
//     return m_dataset_path.string();
// }
// std::string DataLoaderMultiFace::mesh_name_for_cur_timestep(){
//     return fs::path(m_meshes_paths_for_timesteps[m_timestep]).stem().string();

// }
// int DataLoaderMultiFace::subsample_factor(){
//     return m_subsample_factor;
// }
// bool DataLoaderMultiFace::is_genesis(){
//     return m_capture_type=="genesis";
// }
// bool DataLoaderMultiFace::is_minisis(){
//     return m_capture_type=="minisis";
// }
// bool DataLoaderMultiFace::is_synthetic(){
//     return m_capture_type=="synthetic";
// }