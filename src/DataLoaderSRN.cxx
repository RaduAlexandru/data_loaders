#include "data_loaders/DataLoaderSRN.h"

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
#include "data_loaders/DataTransformer.h"
#include "easy_pbr/Frame.h"
#include "Profiler.h"
#include "string_utils.h"
#include "numerical_utils.h"
#include "opencv_utils.h"
#include "eigen_utils.h"
#include "RandGenerator.h"
#include "easy_pbr/LabelMngr.h"
// #include "UtilsGL.h"

//json
// #include "json11/json11.hpp"

//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;


DataLoaderSRN::DataLoaderSRN(const std::string config_file):
    m_is_running(false),
    m_autostart(false),
    m_idx_scene_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator),
    m_nr_scenes_read_so_far(0)
{
    init_params(config_file);
    if(m_autostart){
        start();
    }

    // init_data_reading();
    // start_reading_next_scene();

}

DataLoaderSRN::~DataLoaderSRN(){

    m_is_running=false;
    if (m_loader_thread.joinable()){
        m_loader_thread.join();
    }
}

void DataLoaderSRN::init_params(const std::string config_file){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_srn"];

    m_autostart=loader_config["autostart"];
    m_nr_samples_to_skip=loader_config["nr_samples_to_skip"];
    m_nr_samples_to_read=loader_config["nr_samples_to_read"];
    m_nr_imgs_to_read=loader_config["nr_imgs_to_read"];
    m_shuffle=loader_config["shuffle"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_do_overfit=loader_config["do_overfit"];
    // m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are
    m_object_name= (std::string)loader_config["object_name"];
    // m_dataset_depth_path =(std::string)loader_config["dataset_depth_path"];
    // m_difficulty =(std::string)loader_config["difficulty"];
    // m_load_depth= loader_config["load_depth"];
    m_load_as_shell= loader_config["load_as_shell"];
    m_mode= (std::string)loader_config["mode"];
    m_get_spiral_test_else_split_train= loader_config["get_spiral_test_else_split_train"];

    bool found_scene_multiplier_for_cur_obj=false;
    for (auto& p : loader_config["scene_scale_multiplier"].as_object()) {
        std::cout << "Key: " << p.key() << std::endl;
        std::cout << "Value: " << p.value() << std::endl;
        // p.value() = "new value";
        if( p.key() == m_object_name){
            found_scene_multiplier_for_cur_obj=true;
            m_scene_scale_multiplier= p.value();
            break;
        }
    }
    CHECK(found_scene_multiplier_for_cur_obj)  << "Could not find a scene scale multiplier for the object " << m_object_name;


    // CHECK(m_difficulty=="easy") << "We only implemented the reader for the easy dataset. The hard version just moves the model randomly but maybe you can do that by just moving the mesh";


    //data transformer
    // Config transformer_config=loader_config["transformer"];
    // m_transformer=std::make_shared<DataTransformer>(transformer_config);

}

void DataLoaderSRN::start(){
    CHECK(m_scene_folders.empty()) << " The loader has already been started before. Make sure that you have m_autostart to false";

    init_data_reading();
    start_reading_next_scene();
}

void DataLoaderSRN::init_data_reading(){

    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }

    std::string mode_to_load=m_mode;
    if(!m_get_spiral_test_else_split_train){
        mode_to_load="train"; // we load the train set and we split it differntly
    }

    //find the folder for this mode (train, test, val)
    fs::path dataset_for_mode;
    for (fs::directory_iterator itr(m_dataset_path/("srn_"+m_object_name+"s")  ); itr!=fs::directory_iterator(); ++itr){
        std::string basename=itr->path().filename().string();
        VLOG(1) << "checking basename" << basename;
        if ( radu::utils::contains(basename, mode_to_load)  ){
            dataset_for_mode=itr->path();
            //if the filename is chairs_train we need to go one level deeper into chairs_2.0_train because whoever made the dataset just really wanted to make life for other people difficult...
            if (dataset_for_mode.filename().string()=="chairs_train"){
                dataset_for_mode=dataset_for_mode/"chairs_2.0_train";
            }
            break;
        }
    }
    VLOG(1) << "found path for this mode at " << dataset_for_mode;




    //load all the scene for the chosen object
    int nr_read=0;
    for (fs::directory_iterator itr(dataset_for_mode); itr!=fs::directory_iterator(); ++itr){
        nr_read++;

        if(!m_get_spiral_test_else_split_train){
            if(m_mode=="train"){
                //allow about 66% to be added
                if(nr_read%3==0){ //skip the 3 from a sequence of 1,2,3
                    continue;
                }
            }else if(m_mode=="test"){
                //allow about 33% to be added
                if(nr_read%3!=0){ //skip 1 and 2 but adds the 3
                    continue;
                }
            }
        }

        if( nr_read>=m_nr_samples_to_skip && ((int)m_scene_folders.size()<m_nr_samples_to_read || m_nr_samples_to_read<0 ) ){
            // fs::path scene_path= itr->path()/"rendering";
            fs::path scene_path= itr->path();
            m_scene_folders.push_back(scene_path);
        }
    }
    VLOG(1) << "loaded nr of scenes " << m_scene_folders.size() << " for mode " << m_mode;

    // shuffle the data if neccsary
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_scene_folders), std::end(m_scene_folders), rng_0);
    }

    CHECK(m_scene_folders.size()!=0 ) << "We have read zero scene folders";


}

void DataLoaderSRN::start_reading_next_scene(){
    CHECK(m_is_running==false) << "The loader thread is already running. Wait until the scene is finished loading before loading a new one. You can check this with finished_reading_scene()";

    std::string scene_path;
    if ( m_idx_scene_to_read< (int)m_scene_folders.size()){
        scene_path=m_scene_folders[m_idx_scene_to_read].string();
    }



    if(!m_do_overfit){
        m_idx_scene_to_read++;
    }



    //start the reading
    if (m_loader_thread.joinable()){
        m_loader_thread.join(); //join the thread from the previous iteration of running
    }
    if(!scene_path.empty()){
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderSRN::read_scene, this, scene_path);  //starts to read in another thread
    }
}


void DataLoaderSRN::read_scene(const std::string scene_path){
    // VLOG(1) <<" read from path " << scene_path;

    m_frames_for_scene.clear();

    std::vector<fs::path> paths;
    for (fs::directory_iterator itr( fs::path(scene_path)/"rgb"); itr!=fs::directory_iterator(); ++itr){
        fs::path img_path= itr->path();
        paths.push_back(img_path);
    }

    //shuffle the images from this scene
    unsigned seed1 = m_nr_scenes_read_so_far;
    auto rng_1 = std::default_random_engine(seed1);
    std::shuffle(std::begin(paths), std::end(paths), rng_1);

    //load all the scene for the chosen object
    // for (fs::directory_iterator itr(scene_path); itr!=fs::directory_iterator(); ++itr){
    for (size_t i=0; i<paths.size(); i++){
        // fs::path img_path= itr->path();
        fs::path img_path= paths[i];
        //get only files that end in png
        // VLOG(1) << "img_path" <<img_path;
        if(img_path.filename().string().find("png")!= std::string::npos){
            // VLOG(1) << "png img path " << img_path;

            int img_idx=std::stoi( img_path.stem().string() );
            // VLOG(1) << "img idx is " << img_idx;

            Frame frame;
            frame.frame_idx=img_idx;

            //sets the paths and all the things necessary for the loading of images
            frame.rgb_path=img_path.string();
            frame.subsample_factor=m_subsample_factor;


            //load the images if necessary or delay it for whne it's needed
            frame.load_images=[this]( easy_pbr::Frame& frame ) -> void{ this->load_images_in_frame(frame); };
            if (m_load_as_shell){
                //set the function to load the images whenever it's neede
                frame.is_shell=true;
            }else{
                frame.is_shell=false;
                frame.load_images(frame);
            }







            //read pose and camera params

            //intrisncis are directly from the intrisnics.txt file


            frame.K.setIdentity();
            frame.K(0,0) =  131.250000;
            frame.K(1,1) =  131.250000;
            frame.K(0,2) =  64;
            frame.K(1,2) =  64;
            // frame.K/=m_subsample_factor;
            // frame.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0
            frame.rescale_K(1.0/m_subsample_factor);



            //the extrinsics are stored in poses folder
            fs::path pose_file_path= fs::path(scene_path)/"pose"/(img_path.stem().string()+".txt");
            std::string pose_string= radu::utils::file_to_string(pose_file_path.string());
            //the pose is stored directly as a 4x4 matrix in a rowmajor way so we just load it directly
            std::vector<std::string> tokens=radu::utils::split(pose_string, " ");
            CHECK(tokens.size()==16) << "We expect to have 16 tokens because we want a 4x4 matrix. However we got tokens " << tokens.size();
            Eigen::Affine3f tf_world_cam;
            //row1
            tf_world_cam.matrix()(0,0)= std::stof(tokens[0]);
            tf_world_cam.matrix()(0,1)= std::stof(tokens[1]);
            tf_world_cam.matrix()(0,2)= std::stof(tokens[2]);
            tf_world_cam.matrix()(0,3)= std::stof(tokens[3]);
            //row2
            tf_world_cam.matrix()(1,0)= std::stof(tokens[4]);
            tf_world_cam.matrix()(1,1)= std::stof(tokens[5]);
            tf_world_cam.matrix()(1,2)= std::stof(tokens[6]);
            tf_world_cam.matrix()(1,3)= std::stof(tokens[7]);
            //row3
            tf_world_cam.matrix()(2,0)= std::stof(tokens[8]);
            tf_world_cam.matrix()(2,1)= std::stof(tokens[9]);
            tf_world_cam.matrix()(2,2)= std::stof(tokens[10]);
            tf_world_cam.matrix()(2,3)= std::stof(tokens[11]);
            //row4
            tf_world_cam.matrix()(3,0)= std::stof(tokens[12]);
            tf_world_cam.matrix()(3,1)= std::stof(tokens[13]);
            tf_world_cam.matrix()(3,2)= std::stof(tokens[14]);
            tf_world_cam.matrix()(3,3)= std::stof(tokens[15]);

            //the pose is weird so we multiply with a coord transformatiuon as seen here: https://github.com/sxyu/pixel-nerf/blob/master/src/data/SRNDataset.py
            Eigen::DiagonalMatrix<float, 4> diag;
            diag.diagonal() <<1, -1, 1, 1;
            tf_world_cam.matrix()=tf_world_cam.matrix()*diag;

            //rotate 90 degrees

            Eigen::Quaternionf q = Eigen::Quaternionf( Eigen::AngleAxis<float>( -90 * M_PI / 180.0 ,  Eigen::Vector3f::UnitX() ) );
            Eigen::Affine3f tf_rot;
            tf_rot.setIdentity();
            tf_rot.linear()=q.toRotationMatrix();
            tf_world_cam=tf_rot*tf_world_cam;


            frame.tf_cam_world=tf_world_cam.inverse();


            //rescale things if necessary
            if(m_scene_scale_multiplier>0.0){
                Eigen::Affine3f tf_world_cam_rescaled = frame.tf_cam_world.inverse();
                tf_world_cam_rescaled.translation()*=m_scene_scale_multiplier;
                frame.tf_cam_world=tf_world_cam_rescaled.inverse();
            }









            m_frames_for_scene.push_back(frame);

            if(m_nr_imgs_to_read>0 && (int)m_frames_for_scene.size()>=m_nr_imgs_to_read){
                break; //we finished reading how many images we need so we stop the thread
            }

        }
    }

    // VLOG(1) << "loaded a scene with nr of frames " << m_frames_for_scene.size();
    CHECK(m_frames_for_scene.size()!=0) << "Clouldn't load any images for this scene in path " << scene_path;

    m_nr_scenes_read_so_far++;

    //shuffle the images from this scene
    unsigned seed = m_nr_scenes_read_so_far;
    auto rng_0 = std::default_random_engine(seed);
    std::shuffle(std::begin(m_frames_for_scene), std::end(m_frames_for_scene), rng_0);

    m_is_running=false;
}

void DataLoaderSRN::load_images_in_frame(easy_pbr::Frame& frame){

    frame.is_shell=false;


    // VLOG(1) << "load image from" << frame.rgb_path ;
    cv::Mat rgb_8u=cv::imread(frame.rgb_path );
    if(frame.subsample_factor>1){
        cv::Mat resized;
        cv::resize(rgb_8u, resized, cv::Size(), 1.0/frame.subsample_factor, 1.0/frame.subsample_factor, cv::INTER_AREA);
        rgb_8u=resized;
    }
    frame.rgb_8u=rgb_8u;

    // VLOG(1) << "img type is " << radu::utils::type2string( frame.rgb_8u.type() );
    frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
    frame.width=frame.rgb_32f.cols;
    frame.height=frame.rgb_32f.rows;
    // VLOG(1) << " frame width ad height " << frame.width << " " << frame.height;




}


bool DataLoaderSRN::finished_reading_scene(){
    return !m_is_running;
}
bool DataLoaderSRN::has_data(){
    return finished_reading_scene();
}

Frame DataLoaderSRN::get_random_frame(){
    int random_idx=m_rand_gen->rand_int(0, m_frames_for_scene.size()-1);
    // int random_idx=0;
    return m_frames_for_scene[random_idx];
}

Frame DataLoaderSRN::get_frame_at_idx( const int idx){
    CHECK(idx<(int)m_frames_for_scene.size()) << "idx is out of bounds. It is " << idx << " while m_frames has size " << m_frames_for_scene.size();

    Frame  frame= m_frames_for_scene[idx];

    return frame;
}




bool DataLoaderSRN::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_scene_to_read<(int)m_scene_folders.size()){
        return false; //there is still more files to read
    }


    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderSRN::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_scene_folders), std::end(m_scene_folders), rng_0);
    }

    m_idx_scene_to_read=0;
}

int DataLoaderSRN::nr_samples(){
    return m_frames_for_scene.size();
}

void DataLoaderSRN::set_mode_train(){
    m_mode="train";
}
void DataLoaderSRN::set_mode_test(){
    m_mode="test";
}
void DataLoaderSRN::set_mode_validation(){
    m_mode="val";
}


std::unordered_map<std::string, std::string> DataLoaderSRN::create_mapping_classnr2classname(){

    //from https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/datasets/shapenet.py

    std::unordered_map<std::string, std::string> classnr2classname;

    classnr2classname["04379243"]="table";
    classnr2classname["03211117"]="monitor";
    classnr2classname["04401088"]="phone";

    classnr2classname["04530566"]="watercraft";
    classnr2classname["03001627"]="chair";
    classnr2classname["03636649"]="lamp";

    classnr2classname["03691459"]="speaker";
    classnr2classname["02828884"]="bench";
    classnr2classname["02691156"]="plane";

    classnr2classname["02808440"]="bathtub";
    classnr2classname["02871439"]="bookcase";
    classnr2classname["02773838"]="bag";

    classnr2classname["02801938"]="basket";
    classnr2classname["02880940"]="bowl";
    classnr2classname["02924116"]="bus";

    classnr2classname["02933112"]="cabinet";
    classnr2classname["02942699"]="camera";
    classnr2classname["02958343"]="car";

    classnr2classname["03207941"]="dishwasher";
    classnr2classname["03337140"]="file";
    classnr2classname["03624134"]="knife";

    classnr2classname["03642806"]="laptop";
    classnr2classname["03710193"]="mailbox";
    classnr2classname["03761084"]="microwave";

    classnr2classname["03928116"]="piano";
    classnr2classname["03938244"]="pillow";
    classnr2classname["03948459"]="pistol";

    classnr2classname["04004475"]="printer";
    classnr2classname["04099429"]="rocket";
    classnr2classname["04256520"]="sofa";

    classnr2classname["04554684"]="washer";
    classnr2classname["04090263"]="rifle";
    classnr2classname["02946921"]="can";

    return classnr2classname;
}

Eigen::Affine3f DataLoaderSRN::process_extrinsics_line(const std::string line){

    // //remove any "[" or "]" in the line
    // std::string line_processed=line;
    // line_processed.erase(std::remove(line_processed.begin(), line_processed.end(), '['), line_processed.end());
    // line_processed.erase(std::remove(line_processed.begin(), line_processed.end(), ']'), line_processed.end());


    // // std::vector<std::string> tokens = radu::utils::split(line_processed, " ");
    // std::vector<std::string> tokens = radu::utils::split(line_processed, ",");
    // float azimuth = std::stof(tokens[0]);
    // float elevation = std::stof(tokens[1]);
    // float distance = std::stof(tokens[3]);
    // VLOG(1) << "line is " << line;
    // VLOG(1) << "azimuth elev and dist " << azimuth << " " << elevation << " " << distance;

    // Eigen::Affine3f tf;

    // //from compute_camera_params() in https://github.com/NVIDIAGameWorks/kaolin/blob/a76a004ada95280c6a0a821678cf1b886bcb3625/kaolin/mathutils/geometry/transformations.py
    // float theta = radu::utils::degrees2radians(azimuth);
    // float phi = radu::utils::degrees2radians(elevation);

    // float camY = distance * std::sin(phi);
    // float temp = distance * std::cos(phi);
    // float camX = temp * std::cos(theta);
    // float camZ = temp * std::sin(theta);
    // // cam_pos = np.array([camX, camY, camZ])
    // Eigen::Vector3f t;
    // t << camX,camY,camZ;

    // Eigen::Vector3f axisZ = t;
    // Eigen::Vector3f axisY = Eigen::Vector3f::UnitY();
    // Eigen::Vector3f axisX = axisY.cross(axisZ);
    // axisY = axisZ.cross(axisX);

    // // cam_mat = np.array([axisX, axisY, axisZ])
    // Eigen::Matrix3f R;
    // R.col(0)=axisX;
    // R.col(1)=axisY;
    // R.col(2)=-axisZ;
    // // l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    // // l2[l2 == 0] = 1
    // // cam_mat = cam_mat / np.expand_dims(l2, 1)

    // Eigen::Vector3f norm_vec=R.colwise().norm();
    // VLOG(1) << "norm is " << norm_vec;
    // // R=R.colwise()/norm;
    // for (int i=0; i<3; i++){
    //     float norm=norm_vec(i);
    //     for (int j=0; j<3; j++){
    //         // R(i,j) = R(i,j)/norm;
    //         R(j,i) = R(j,i)/norm;
    //     }
    // }
    // norm_vec=R.colwise().norm();
    // VLOG(1) << "norm is " << norm_vec;

    // tf.translation() = t;
    // tf.linear() = R;

    // //just to make sure it's orthogonal
    // // Eigen::AngleAxisf aa(R);    // RotationMatrix to AxisAngle
    // // R = aa.toRotationMatrix();  // AxisAngle      to RotationMatrix
    // // tf.linear() = R;

    // Eigen::Affine3f tf_ret=tf.inverse();
    // // Eigen::Affine3f tf_ret=tf;


















    //attempt 2 by looking at https://github.com/Xharlie/ShapenetRender_more_variation/blob/master/cam_read.py
    //  F_MM = 35.  # Focal length
    // SENSOR_SIZE_MM = 32.
    // PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    // RESOLUTION_PCT = 100.
    // SKEW = 0.
    // CAM_MAX_DIST = 1.75
    // CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
    //                       [1.0, -4.371138828673793e-08, -0.0],
    //                       [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    float cam_max_dist=1.75;
    Eigen::Matrix3f cam_rot;
    cam_rot <<1.910685676922942e-15, 4.371138828673793e-08, 1.0,
            1.0, -4.371138828673793e-08, -0.0,
            4.371138828673793e-08, 1.0, -4.371138828673793e-08;


    std::string line_processed=line;
    line_processed.erase(std::remove(line_processed.begin(), line_processed.end(), '['), line_processed.end());
    line_processed.erase(std::remove(line_processed.begin(), line_processed.end(), ']'), line_processed.end());


    // std::vector<std::string> tokens = radu::utils::split(line_processed, " ");
    std::vector<std::string> tokens = radu::utils::split(line_processed, ",");
    float az = std::stof(tokens[0]);
    float el = std::stof(tokens[1]);
    float distance_ratio = std::stof(tokens[3]);
    // float ox = std::stof(tokens[7]);
    // float oy = std::stof(tokens[8]);
    // float oz = std::stof(tokens[9]);

    // # Calculate rotation and translation matrices.
    // # Step 1: World coordinate to object coordinate.
    float sa = std::sin(radu::utils::degrees2radians(-az));
    float ca = std::cos(radu::utils::degrees2radians(-az));
    float se = std::sin(radu::utils::degrees2radians(-el));
    float ce = std::cos(radu::utils::degrees2radians(-el));
    // R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
    //                                       (sa * ce, ca, sa * se),
    //                                       (-se, 0, ce))))
    Eigen::Matrix3f R_world2obj;
    R_world2obj <<ca * ce, -sa, ca * se,
                    sa * ce, ca, sa * se,
                    -se, 0, ce;
    Eigen::Matrix3f trans;
    trans=R_world2obj.transpose();
    R_world2obj=trans;


    // # Step 2: Object coordinate to camera coordinate.
    // R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    Eigen::Matrix3f R_obj2cam=cam_rot.transpose();
    Eigen::Matrix3f R_world2cam = R_obj2cam * R_world2obj;
    // cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
    //                                        0,
    //                                        0)))
    Eigen::Vector3f cam_location;
    cam_location <<  distance_ratio * cam_max_dist, 0, 0;
    // # print('distance', distance_ratio * CAM_MAX_DIST)
    Eigen::Vector3f T_world2cam = -1 * R_obj2cam * cam_location;

    // // # Step 3: Fix blender camera's y and z axis direction.
    // R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    Eigen::Matrix3f R_camfix;
    R_camfix <<1, 0, 0,
              0,1,0,
              0,0,-1;
    R_world2cam = R_camfix * R_world2cam;
    T_world2cam = R_camfix * T_world2cam;



    Eigen::Affine3f tf_ret;
    tf_ret.linear()=R_world2cam;
    tf_ret.translation()=T_world2cam;


    //rotate 90 degrees
    Eigen::Affine3f tf_worldGL_worldROS;
    tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3f worldGL_worldROS_rot;
    worldGL_worldROS_rot = Eigen::AngleAxisf(-0.5*M_PI, Eigen::Vector3f::UnitX());
    tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
    // Eigen::Affine3f tf_worldROS_worldGL=tf_worldGL_worldROS.inverse();
    Eigen::Affine3f tf_ret_cor=tf_worldGL_worldROS*tf_ret.inverse();
    tf_ret=tf_ret_cor.inverse();






    return tf_ret;

}
