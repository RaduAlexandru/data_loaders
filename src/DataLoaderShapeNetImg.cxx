#include "data_loaders/DataLoaderShapeNetImg.h"

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
#include "easy_gl/UtilsGL.h"

//json
// #include "json11/json11.hpp"

//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;


DataLoaderShapeNetImg::DataLoaderShapeNetImg(const std::string config_file):
    m_is_running(false),
    m_idx_scene_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator),
    m_nr_scenes_read_so_far(0)
{
    init_params(config_file);
    // if(m_autostart){
        // start();
    // }

    init_data_reading();
    start_reading_next_scene();

}

DataLoaderShapeNetImg::~DataLoaderShapeNetImg(){

    m_is_running=false;
    m_loader_thread.join();
}

void DataLoaderShapeNetImg::init_params(const std::string config_file){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_shapenet_img"];

    // m_autostart=loader_config["autostart"];
    m_nr_samples_to_skip=loader_config["nr_samples_to_skip"];
    m_nr_samples_to_read=loader_config["nr_samples_to_read"];
    m_nr_imgs_to_read=loader_config["nr_imgs_to_read"];
    m_shuffle=loader_config["shuffle"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_do_overfit=loader_config["do_overfit"];
    m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are
    m_dataset_depth_path =(std::string)loader_config["dataset_depth_path"];
    m_difficulty =(std::string)loader_config["difficulty"];
    m_load_depth= loader_config["load_depth"];
    m_load_as_shell= loader_config["load_as_shell"];

    CHECK(m_difficulty=="easy") << "We only implemented the reader for the easy dataset. The hard version just moves the model randomly but maybe you can do that by just moving the mesh";


    //data transformer
    // Config transformer_config=loader_config["transformer"];
    // m_transformer=std::make_shared<DataTransformer>(transformer_config);

}



void DataLoaderShapeNetImg::init_data_reading(){

    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }

    //make the mapping between the weird numbers in the files and the class label
    std::unordered_map<std::string, std::string> classnr2classname = create_mapping_classnr2classname( );


    //go to the folder for that specific object. Read through all the folders which will give me a gibberish of numbers and map that to the class name. If we found a match then we read the scenes
    fs::path chosen_object_path;
    for (fs::directory_iterator itr(m_dataset_path); itr!=fs::directory_iterator(); ++itr){
        fs::path object_path= itr->path();
        if (!fs::is_regular_file(object_path)){
            //check that this number matched the object we chose
            std::string class_nr=object_path.stem().string();
            // VLOG(1) << "class nr is " << class_nr;
            std::string class_name=classnr2classname[class_nr];
            if(class_name==m_restrict_to_object){
                VLOG(1) << "found " << class_name << " in path " << object_path;
                chosen_object_path=object_path;
                break;
            }

        }
    }
    CHECK( !chosen_object_path.empty() ) << "Could not find the object " << m_restrict_to_object;


    //load all the scene for the chosen object
    int nr_read=0;
    for (fs::directory_iterator itr(chosen_object_path); itr!=fs::directory_iterator(); ++itr){
        if( nr_read>=m_nr_samples_to_skip && ((int)m_scene_folders.size()<m_nr_samples_to_read || m_nr_samples_to_read<0 ) ){
            // fs::path scene_path= itr->path()/"rendering";
            fs::path scene_path= itr->path() / m_difficulty;
            m_scene_folders.push_back(scene_path);
        }
        nr_read++;
    }

    // shuffle the data if neccsary
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_scene_folders), std::end(m_scene_folders), rng_0);
    }

    CHECK(m_scene_folders.size()!=0 ) << "We have read zero scene folders";


}

void DataLoaderShapeNetImg::start_reading_next_scene(){
    CHECK(m_is_running==false) << "The loader thread is already running. Wait until the scene is finished loading before loading a new one. You can check this with finished_reading_scene()";

    std::string scene_path;
    if ( m_idx_scene_to_read< m_scene_folders.size()){
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
        m_loader_thread=std::thread(&DataLoaderShapeNetImg::read_scene, this, scene_path);  //starts to read in another thread
    }
}


void DataLoaderShapeNetImg::read_scene(const std::string scene_path){
    // VLOG(1) <<" read from path " << scene_path;

    m_frames_for_scene.clear();

    std::vector<fs::path> paths;
    for (fs::directory_iterator itr(scene_path); itr!=fs::directory_iterator(); ++itr){
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
            if (m_load_depth){
                fs::path filename=img_path.stem();
                fs::path scene_name=img_path.parent_path().parent_path().filename();
                fs::path object_name=img_path.parent_path().parent_path().parent_path().filename();
                // VLOG(1) << "filename " << filename << " " << scene_name << " " << object_name ;
                fs::path depth_path = m_dataset_depth_path/object_name/scene_name/m_difficulty/ (filename.string() + ".exr" );
                frame.depth_path=depth_path.string();
            }
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

            //intrisncis are from here
            // https://github.com/facebookresearch/pytorch3d/blob/778383eef77a23686f3d0e68834b29d6d73f8501/pytorch3d/datasets/r2n2/r2n2.py
            // and from https://github.com/facebookresearch/meshrcnn/blob/master/shapenet/utils/coords.py
            // ther we also have zmin and zmax
            // but it seems that it's not actually  a K matrix but rather a projection matrix as  an opengl projection matrix like in here http://www.songho.ca/opengl/gl_projectionmatrix.html
            // so it projects from camera coordinates to clip coordinates but we want a K matrix that projects to screen coords
            Eigen::Matrix4f P;
            P <<
            2.1875, 0.0, 0.0, 0.0,
            0.0, 2.1875, 0.0, 0.0,
            0.0, 0.0, -1.002002, -0.2002002,
            0.0, 0.0, -1.0, 0.0;
            // Eigen::Matrix3f K = opengl_proj_to_intrinsics(P, 137, 137);
            Eigen::Matrix3f K = opengl_proj_to_intrinsics(P, 224, 224);
            // VLOG(1) << "K is " << K;
            frame.K=K;
            // frame.K/=m_subsample_factor;
            // frame.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0
            frame.rescale_K(1.0/m_subsample_factor);

            // frame.K(1,1)=435.55555555555554 ;

            // VLOG(1) << "K is " << frame.K;


            //the extrinsics are stored in rendering_metadata.txt, stored as azimuth elevation and distance
            //processing of this can be seen here: https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/datasets/shapenet.py
            Eigen::Affine3f tf_cam_world;
            int lines_read=0;
            bool found=false;
            std::ifstream metadata_file( (fs::path(scene_path)/"rendering_metadata.txt").string() );
            if(!metadata_file.is_open()){
                LOG(FATAL) << "Could not open the rendering metadata file ";
            }
            for( std::string line; getline( metadata_file, line ); ){
                if (lines_read==img_idx){
                    // VLOG(1) << "img idx" << img_idx << "reading line " << lines_read << " line " << line;
                    tf_cam_world=process_extrinsics_line(line);
                    found=true;
                    break;
                }
                lines_read++;
            }
            CHECK(found) << "Could not find a corrsponding line in the metadata for img " << img_idx;
            // VLOG(1) << "TF is " << tf_cam_world.matrix();
            frame.tf_cam_world=tf_cam_world;




            auto tf=frame.tf_cam_world;
            // frame.tf_cam_world=tf.inverse();







            m_frames_for_scene.push_back(frame);

            if(m_nr_imgs_to_read>0 && m_frames_for_scene.size()>=m_nr_imgs_to_read){
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

void DataLoaderShapeNetImg::load_images_in_frame(easy_pbr::Frame& frame){

    frame.is_shell=false;


    // VLOG(1) << "load image from" << frame.rgb_path ;
    cv::Mat rgba_8u=cv::imread(frame.rgb_path, cv::IMREAD_UNCHANGED ); //correct
    // cv::Mat rgba_8u=cv::imread(img_path.string(), cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH );
    if(frame.subsample_factor>1){
        cv::Mat resized;
        cv::resize(rgba_8u, resized, cv::Size(), 1.0/frame.subsample_factor, 1.0/frame.subsample_factor, cv::INTER_AREA);
        rgba_8u=resized;
    }
    std::vector<cv::Mat> channels(4);
    cv::split(rgba_8u, channels);
    // cv::threshold( channels[3], frame.mask, 0.01, 1.0, cv::THRESH_BINARY);
    // frame.mask=channels[3];
    channels[3].convertTo(frame.mask, CV_32FC1, 1.0/255.0);
    cv::threshold( frame.mask, frame.mask, 0.0, 1.0, cv::THRESH_BINARY);
    channels.pop_back();
    cv::merge(channels, frame.rgb_8u);

    // frame.rgb_8u=cv::imread(img_path.string(), cv::IMREAD_UNCHANGED );
    // VLOG(1) << "img type is " << radu::utils::type2string( frame.rgb_8u.type() );
    frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
    frame.width=frame.rgb_32f.cols;
    frame.height=frame.rgb_32f.rows;
    // VLOG(1) << " frame width ad height " << frame.width << " " << frame.height;



    //read also the depth
    if (m_load_depth){
        fs::path depth_path= frame.depth_path;
        CHECK( fs::is_regular_file(depth_path) ) << "Could not find depth under " << depth_path;
        // cv::Mat depth=cv::imread(depth_path.string() , cv::IMREAD_UNCHANGED);
        // cv::Mat depth=cv::imread(depth_path.string() , cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH );
        cv::Mat depth=cv::imread(depth_path.string() ,  cv::IMREAD_UNCHANGED  );
        // cv::Mat depth=cv::imread(depth_path.string() , CV_LOAD_IMAGE_ANYDEPTH );
        // VLOG(1) << "depth has type " << radu::utils::type2string(depth.type());
        // VLOG(1) << "depth has rows and cols " << depth.rows << " " << depth.cols;
        cv::threshold( depth, depth, 99999, 0.0, cv::THRESH_TOZERO_INV ); //vlaues above 9999 are set to zero depth
        // double min, max;
        // cv::minMaxLoc(depth, &min, &max);
        // VLOG(1) << "min max is " << min <<" " << max;
        std::vector<cv::Mat> channels_depth(3);
        cv::split(depth, channels_depth);
        // depth.convertTo(frame.depth, CV_32FC1, 1.0/1000.0); //the depth was stored in mm but we want it in meters
        // channels_depth[0].convertTo(frame.depth, CV_32FC1, 1.0/1.0); //the depth was stored in mm but we want it in meters
        frame.depth=channels_depth[0];
        // depth.convertTo(frame.depth, CV_32FC1, 1.0/1000.0); //the depth was stored in cm but we want it in meters
        // frame.depth=1.0/frame.depth; //seems to be the inverse depth
        if(frame.subsample_factor>1){
            cv::Mat resized;
            cv::resize(frame.depth, resized, cv::Size(), 1.0/frame.subsample_factor, 1.0/frame.subsample_factor, cv::INTER_NEAREST);
            frame.depth=resized;
        }
    }



}


bool DataLoaderShapeNetImg::finished_reading_scene(){
    return !m_is_running;
}
bool DataLoaderShapeNetImg::has_data(){
    return finished_reading_scene();
}

Frame DataLoaderShapeNetImg::get_random_frame(){
    int random_idx=m_rand_gen->rand_int(0, m_frames_for_scene.size()-1);
    // int random_idx=0;
    return m_frames_for_scene[random_idx];
}

Frame DataLoaderShapeNetImg::get_frame_at_idx( const int idx){
    CHECK(idx<m_frames_for_scene.size()) << "idx is out of bounds. It is " << idx << " while m_frames has size " << m_frames_for_scene.size();

    Frame  frame= m_frames_for_scene[idx];

    return frame;
}




bool DataLoaderShapeNetImg::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_scene_to_read<m_scene_folders.size()){
        return false; //there is still more files to read
    }


    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderShapeNetImg::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_scene_folders), std::end(m_scene_folders), rng_0);
    }

    m_idx_scene_to_read=0;
}

int DataLoaderShapeNetImg::nr_samples(){
    return m_frames_for_scene.size();
}

std::unordered_map<std::string, std::string> DataLoaderShapeNetImg::create_mapping_classnr2classname(){

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

Eigen::Affine3f DataLoaderShapeNetImg::process_extrinsics_line(const std::string line){

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
    Eigen::Affine3f tf_worldROS_worldGL=tf_worldGL_worldROS.inverse();
    Eigen::Affine3f tf_ret_cor=tf_worldGL_worldROS*tf_ret.inverse();
    tf_ret=tf_ret_cor.inverse();






    return tf_ret;

}
