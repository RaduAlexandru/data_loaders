#include "data_loaders/DataLoaderNerf.h"

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
#include "easy_pbr/LabelMngr.h"
#include "UtilsGL.h"

//json
#include "json11/json11.hpp"

//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;


struct {
    bool operator()(fs::path a, fs::path b) const {
        std::string a_filename=a.stem().string();
        std::string b_filename=b.stem().string();
        //The files have a format of r_NUMBER so we want to get the number
        std::vector<std::string> a_tokens=split(a_filename,"_");
        std::vector<std::string> b_tokens=split(b_filename,"_");
        int a_nr=std::stoi(a_tokens[1]);
        int b_nr=std::stoi(b_tokens[1]);
        return a_nr < b_nr;
    }
} FileComparatorFunc;


DataLoaderNerf::DataLoaderNerf(const std::string config_file):
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

DataLoaderNerf::~DataLoaderNerf(){

    // m_is_running=false;
    // m_loader_thread.join();
}

void DataLoaderNerf::init_params(const std::string config_file){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_nerf"];

    m_autostart=loader_config["autostart"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_scene_scale_multiplier= loader_config["scene_scale_multiplier"];
    m_mode=(std::string)loader_config["mode"];
    // m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are


    //data transformer
    // Config transformer_config=loader_config["transformer"];
    // m_transformer=std::make_shared<DataTransformer>(transformer_config);

}

void DataLoaderNerf::start(){
    init_data_reading();
    init_poses();
    read_data();
}


void DataLoaderNerf::init_data_reading(){

    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }

    //go to the folder of train val or test depending on the mode in which we are one
    for (fs::directory_iterator itr(m_dataset_path/m_mode); itr!=fs::directory_iterator(); ++itr){
        fs::path img_path= itr->path();
        //we disregard the images that contain depth and normals, we load only the rgb
        if (fs::is_regular_file(img_path) &&
        img_path.filename().string().find("png") != std::string::npos &&
        img_path.stem().string().find("depth")== std::string::npos &&
        img_path.stem().string().find("normal")== std::string::npos   ){
            m_imgs_paths.push_back(img_path);
        }
    }
    CHECK( !m_imgs_paths.empty() ) << "Could not find any images in path " << m_dataset_path/m_mode;

    std::sort(m_imgs_paths.begin(), m_imgs_paths.end(), FileComparatorFunc);


    // shuffle the data if neccsary
    if(m_shuffle && m_mode=="train"){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_imgs_paths), std::end(m_imgs_paths), rng_0);
    }


}

void DataLoaderNerf::init_poses(){
    //read transforms_test.json (or whichever file is corresponding to the mode we are on)

    //get the path to this json file
    fs::path pose_file_json= m_dataset_path/("transforms_"+m_mode+".json");
    if(!fs::is_regular_file(pose_file_json) ) {
        LOG(FATAL) << "Json file for the poses could not be found in " << pose_file_json;
    }


    //read json
    std::string file_list_string=radu::utils::file_to_string(pose_file_json.string());
    std::string err;
    const auto json = json11::Json::parse(file_list_string, err);
    m_camera_angle_x = json["camera_angle_x"].number_value();

    //read all the poses
    for (auto &k : json["frames"].array_items()) {
        fs::path key= k.string_value();

        std::string file_path=k["file_path"].string_value(); //will be something like ./test/r_0 but we only want the r_0 part
        std::vector<std::string> tokens= radu::utils::split(file_path, "/");
        std::string file_name= tokens[2];
        // VLOG(1) << "filename is" << file_name;


        //read the psoe as a 4x4 matrix
        Eigen::Affine3d tf_world_cam;
        int rows=4;
        for (int r = 0; r < 4; r++){
            for (int c = 0; c < 4; c++){
                // VLOG(1) << "tf matri is " << k["transform_matrix"][r][c].number_value();
                tf_world_cam.matrix()(r,c) =k["transform_matrix"][r][c].number_value();
            }
        }
        tf_world_cam.linear().col(2)=-tf_world_cam.linear().col(2); //make it look in the correct direction

        //rotate from their world to our opengl world by rotating along the x axis
        Eigen::Affine3d tf_worldGL_worldROS;
        tf_worldGL_worldROS.setIdentity();
        Eigen::Matrix3d worldGL_worldROS_rot;
        worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
        tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
        // transform_vertices_cpu(tf_worldGL_worldROS);
        tf_world_cam=tf_worldGL_worldROS*tf_world_cam;


        Eigen::Affine3d tf_cam_world=tf_world_cam.inverse();



        m_filename2pose[file_name]=tf_cam_world; //we want to store here the transrom from world to cam so the tf_cam_world


    }

}

void DataLoaderNerf::read_data(){

    for (size_t i = 0; i < m_imgs_paths.size(); i++){

        Frame frame;

        fs::path img_path=m_imgs_paths[i];
        // VLOG(1) << "reading " << img_path;

        //get the idx
        std::string filename=img_path.stem().string();
        std::vector<std::string> tokens=radu::utils::split(filename,"_");
        frame.frame_idx=std::stoi(tokens[1]);

        //read rgba and split into rgb and alpha mask
        cv::Mat rgba_8u = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
        cv::Mat rgb_8u;
        if(m_subsample_factor>1){
            cv::Mat resized;
            cv::resize(rgba_8u, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
            rgba_8u=resized;
        }
        std::vector<cv::Mat> channels(4);
        cv::split(rgba_8u, channels);
        cv::threshold( channels[3], frame.mask, 0.0, 1.0, cv::THRESH_BINARY);
        channels.pop_back();
        cv::merge(channels, rgb_8u);


        // cv::cvtColor(frame.rgb_8u, frame.gray_8u, CV_BGR2GRAY);
        rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
        // cv::cvtColor(frame.rgb_32f, frame.gray_32f, CV_BGR2GRAY);
        frame.width=frame.rgb_32f.cols;
        frame.height=frame.rgb_32f.rows;

        //if we are loading the test, get also the normals
        // BUG WE CANNOT LOAD THE NORMAL BECAUSE THEY ARE STORED AS PNG AND THEREFORE FROM THE range of [-1,1], only the [0,1] is stored...
        // if(m_mode=="test"){
        //     fs::path parent=img_path.parent_path();
        //     std::string img_filename=img_path.stem().string();
        //     fs::path normal_img_path=parent/(img_filename+"_normal_0001.png");
        //     // VLOG(1) << "normal img path" << normal_img_path;

        //     cv::Mat normal=cv::imread(normal_img_path.string() ,  cv::IMREAD_UNCHANGED );
        //     VLOG(1) << "normal type is " << radu::utils::type2string(normal.type());
        //     //get the normal form 4 channels( with alpha, to just 3 channels)
        //     std::vector<cv::Mat> channels(4);
        //     cv::split(normal, channels);
        //     channels.pop_back();
        //     cv::merge(channels, normal);
        //     normal.convertTo(normal, CV_32FC3, 1.0/255.0);
        //     CHECK(!normal.empty()) << "The normal image is empty at path " << normal_img_path;
        //     //subsample the normals and renormalzie them
        //     if(m_subsample_factor>1){
        //         cv::Mat resized;
        //         cv::resize(normal, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
        //         normal=resized;
        //         //normalize becuase we subsample and therefore the normal may not be equal to 1 anymore
        //         // for(int y=0; y<normal.rows; y++){
        //     //         for(int x=0; x<normal.cols; x++){
        //     //             Eigen::Vector3f normal_vec;
        //     //             normal_vec.x()= normal.at<cv::Vec3f>(y,x)[0];
        //     //             normal_vec.y()= normal.at<cv::Vec3f>(y,x)[1];
        //     //             normal_vec.z()= normal.at<cv::Vec3f>(y,x)[2];
        //     //             // normal_vec=normal_vec.normalized();
        //     //             // normal_vec.normalize();
        //     //             // normal_vec=(normal_vec.array()+1.0)*0.5;
        //     //             // if (!normal_vec.isZero()){
        //     //                 // VLOG(1) << "norm is " << normal_vec.norm();
        //     //             // }
        //     //             normal_vec.x()=(normal_vec.x()+1.0)*0.5;
        //     //             normal_vec.y()=(normal_vec.y()+1.0)*0.5;
        //     //             normal_vec.z()=(normal_vec.z()+1.0)*0.5;

        //     //             normal.at<cv::Vec3f>(y,x)[0]=normal_vec.x();
        //     //             normal.at<cv::Vec3f>(y,x)[1]=normal_vec.y();
        //     //             normal.at<cv::Vec3f>(y,x)[2]=normal_vec.z();
        //     //         }
        //     //     }
        //     }

        //     frame.normal_32f=normal;
        // }

        //if we are loading the test one, get also the depth
        // THE DEPTH is shitty because is stored as png therefore discretized in 0,255 so it has stepwise artifacts
        // if(m_mode=="test"){
        //     fs::path parent=img_path.parent_path();
        //     std::string img_filename=img_path.stem().string();
        //     // VLOG(1) << "parent" << parent;
        //     // fs::path depth_img_path=
        //     fs::path depth_img_path=parent/(img_filename+"_depth_0001.png");
        //     VLOG(1) << "depth img path" << depth_img_path;

        //     cv::Mat depth=cv::imread(depth_img_path.string() , CV_LOAD_IMAGE_ANYDEPTH);
        //     CHECK(!depth.empty()) << "The depth image is empty at path " << depth_img_path;
        //     // depth.convertTo(frame.depth, CV_32FC1, 1.0/1000.0); //the depth was stored in mm but we want it in meters
        //     depth.convertTo(frame.depth, CV_32FC1, 1.0/1000.0); //the depth was stored in cm but we want it in meters
        //     frame.depth=1.0/frame.depth; //seems to be the inverse depth

        // }

        //load gradients
        // cv::cvtColor(frame.rgb_32f, frame.gray_32f, CV_BGR2GRAY);
        // cv::Scharr( frame.gray_32f, frame.grad_x_32f, CV_32F, 1, 0);
        // cv::Scharr( frame.gray_32f, frame.grad_y_32f, CV_32F, 0, 1);


        //extrinsics
        frame.tf_cam_world=m_filename2pose[img_path.stem().string()].cast<float>();

        //intrinsics got mostly from here https://github.com/bmild/nerf/blob/0247d6e7ede8d918bc1fab2711f845669aee5e03/load_blender.py
        frame.K.setIdentity();
        float focal = 0.5 * frame.width / std::tan(0.5 * m_camera_angle_x);
        frame.K(0,0) = focal;
        frame.K(1,1) = focal;
        frame.K(0,2) = frame.width/2.0; //no need to subsample the cx and cy because the frame width already refers to the subsampled iamge
        frame.K(1,2) = frame.height/2.0;
        frame.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0


        //rescale things if necessary
        if(m_scene_scale_multiplier>0.0){
            Eigen::Affine3f tf_world_cam_rescaled = frame.tf_cam_world.inverse();
            tf_world_cam_rescaled.translation()*=m_scene_scale_multiplier;
            frame.tf_cam_world=tf_world_cam_rescaled.inverse();
        }

        m_frames.push_back(frame);
        // VLOG(1) << "pushback and frames is " << m_frames.size();


    }


}



Frame DataLoaderNerf::get_next_frame(){
    CHECK(m_idx_img_to_read<m_frames.size()) << "m_idx_img_to_read is out of bounds. It is " << m_idx_img_to_read << " while m_frames has size " << m_frames.size();
    Frame  frame= m_frames[m_idx_img_to_read];

    if(!m_do_overfit){
        m_idx_img_to_read++;
    }

    return frame;
}
std::vector<easy_pbr::Frame> DataLoaderNerf::get_all_frames(){
    return m_frames;
}
Frame DataLoaderNerf::get_frame_at_idx( const int idx){
    CHECK(idx<m_frames.size()) << "idx is out of bounds. It is " << idx << " while m_frames has size " << m_frames.size();

    Frame  frame= m_frames[idx];

    return frame;
}

Frame DataLoaderNerf::get_random_frame(){
    CHECK(m_frames.size()>0 ) << "m_frames has size 0";

    int random_idx=m_rand_gen->rand_int(0, m_frames.size()-1);
    Frame  frame= m_frames[random_idx];

    return frame;
}
Frame DataLoaderNerf::get_closest_frame( const easy_pbr::Frame& frame){

    float closest_distance=std::numeric_limits<float>::max();
    int closest_idx=-1;
    for(size_t i=0; i<m_frames.size(); i++){
        float dist =  ( m_frames[i].tf_cam_world.inverse().translation() - frame.tf_cam_world.inverse().translation() ).norm();
        if (dist < closest_distance && dist>1e-7){
            closest_distance=dist;
            closest_idx=i;
        }
    }

    Frame  frame_closest= m_frames[closest_idx];

    return frame_closest;

}


std::vector<easy_pbr::Frame>  DataLoaderNerf::get_close_frames( const easy_pbr::Frame& frame, const int nr_frames, const bool discard_same_idx){

    CHECK(nr_frames<m_frames.size()) << "Cannot select more close frames than the total nr of frames that we have in the loader. Required select of " << nr_frames << " out of a total of " << m_frames.size() << " available in the loader";

    std::vector<easy_pbr::Frame> selected_close_frames;

    for(size_t i=0; i<nr_frames; i++){

        //select a close frame
        float closest_distance=std::numeric_limits<float>::max();
        int closest_idx=-1;
        for(size_t j=0; j<m_frames.size(); j++){

            //ignore if the current frame we are checking is THIS
            if (discard_same_idx){
                if( m_frames[j].frame_idx == frame.frame_idx ){
                    continue;
                }
            }

            //ignore the current frame that we are checking if it's any of the ones already selected
            bool is_already_selected=false;
            for(size_t k=0; k<selected_close_frames.size(); k++){
                if( m_frames[j].frame_idx == selected_close_frames[k].frame_idx ){
                    is_already_selected=true;
                }
            }
            if(is_already_selected){
                continue;
            }

            //now get the closest one
            float dist =  ( m_frames[j].tf_cam_world.inverse().translation() - frame.tf_cam_world.inverse().translation() ).norm();
            // float dist = 1.0 -  m_frames[j].look_dir().dot(frame.look_dir());
            if (dist < closest_distance){
                closest_distance=dist;
                closest_idx=j;
            }
        }

        Frame  frame_closest= m_frames[closest_idx];
        selected_close_frames.push_back(frame_closest);


    }


    return selected_close_frames;


}

// //compute weights
// std::vector<float> DataLoaderNerf::compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames){
//     // https://people.cs.clemson.edu/~dhouse/courses/404/notes/barycentric.pdf
//     // https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
//     // https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle

//     //to compute the weights we use barycentric coordinates.
//     //this has several steps, first project the current frame into the triangle defiend by the close_frames.
//     //compute barycentric coords
//     //if the barycentric coords are not within [0,1], clamp them

//     //checks
//     CHECK(close_frames.size()==3) <<"This assumes we are using 3 frames as close frames because we want to compute barycentric coords";

//     //make triangle
//     Eigen::Vector3d cur_pos= frame.pos_in_world().cast<double>();
//     Eigen::Vector3d p1= close_frames[0].pos_in_world().cast<double>();
//     Eigen::Vector3d p2= close_frames[1].pos_in_world().cast<double>();
//     Eigen::Vector3d p3= close_frames[2].pos_in_world().cast<double>();

//     //get barycentirc coords of the projection https://math.stackexchange.com/a/544947
//     Eigen::Vector3d u=p2-p1;
//     Eigen::Vector3d v=p3-p1;
//     Eigen::Vector3d n=u.cross(v);
//     Eigen::Vector3d w=cur_pos-p1;

//     float w_p3= u.cross(w).dot(n)/ (n.dot(n));
//     float w_p2= w.cross(v).dot(n)/ (n.dot(n));
//     float w_p1= 1.0-w_p2-w_p3;

//     //to get weights as if the point was inside the triangle, we clamp the barycentric coordinates (I don't know if this is needed yeat)

//     //return tha values
//     std::vector<float> vals;
//     vals.push_back(w_p1);
//     vals.push_back(w_p2);
//     vals.push_back(w_p3);

//     return vals;


// }





bool DataLoaderNerf::is_finished(){
    //check if this loader has returned all the images it has
    if(m_idx_img_to_read<m_frames.size()){
        return false; //there is still more files to read
    }


    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderNerf::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle && m_mode=="train"){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_frames), std::end(m_frames), rng_0);
    }

    m_idx_img_to_read=0;
}

int DataLoaderNerf::nr_samples(){
    return m_frames.size();
}

bool DataLoaderNerf::has_data(){
    return true; //we always have data since the loader stores all the image in memory and keeps them there
}

void DataLoaderNerf::set_mode_train(){
    m_mode="train";
}
void DataLoaderNerf::set_mode_test(){
    m_mode="test";
}
void DataLoaderNerf::set_mode_validation(){
    m_mode="val";
}
