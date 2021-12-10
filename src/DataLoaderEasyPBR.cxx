#include "data_loaders/DataLoaderEasyPBR.h"

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
// #include "easy_gl/UtilsGL.h"

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
        int a_nr=std::stoi(a_filename);
        int b_nr=std::stoi(b_filename);
        return a_nr < b_nr;
    }
} FileComparatorFunc;


DataLoaderEasyPBR::DataLoaderEasyPBR(const std::string config_file):
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

DataLoaderEasyPBR::~DataLoaderEasyPBR(){

    // m_is_running=false;
    // m_loader_thread.join();
}

void DataLoaderEasyPBR::init_params(const std::string config_file){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_easypbr"];

    m_autostart=loader_config["autostart"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_mode=(std::string)loader_config["mode"];
    // m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are
    m_object_name= (std::string)loader_config["object_name"];

    // m_scene_scale_multiplier= loader_config["scene_scale_multiplier"];
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

    //data transformer
    // Config transformer_config=loader_config["transformer"];
    // m_transformer=std::make_shared<DataTransformer>(transformer_config);

}

void DataLoaderEasyPBR::start(){
    init_data_reading();
    init_poses();
    read_data();
}


void DataLoaderEasyPBR::init_data_reading(){

    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }

    //go to the folder of train val or test depending on the mode in which we are one
    for (fs::directory_iterator itr(m_dataset_path/m_object_name/("imgs_"+m_mode) ); itr!=fs::directory_iterator(); ++itr){
        fs::path img_path= itr->path();
        //we disregard the images that contain depth and normals, we load only the rgb
        if (fs::is_regular_file(img_path) &&
        img_path.filename().string().find("png") != std::string::npos){
            m_imgs_paths.push_back(img_path);
        }
    }
    CHECK( !m_imgs_paths.empty() ) << "Could not find any images in path " << m_dataset_path/m_object_name;

    std::sort(m_imgs_paths.begin(), m_imgs_paths.end(), FileComparatorFunc);


    // shuffle the data if neccsary
    if(m_shuffle && m_mode=="train"){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_imgs_paths), std::end(m_imgs_paths), rng_0);
    }


}

void DataLoaderEasyPBR::init_poses(){
    //read transforms_test.json (or whichever file is corresponding to the mode we are on)

    //the file post_and_instrincis.txt has format filename tx ty tz qx qy qz qw fx fy cx cy

    //get the path to this json file
    fs::path pose_and_intrinsics_path= m_dataset_path/m_object_name/"poses_and_intrinsics.txt";
    if(!fs::is_regular_file(pose_and_intrinsics_path) ) {
        LOG(FATAL) << "File for the poses could not be found in " << pose_and_intrinsics_path;
    }


    std::ifstream file(pose_and_intrinsics_path.string() );
    if(! file.is_open()){
        LOG(FATAL) << "Could not open labels file " << pose_and_intrinsics_path;
    }
    int nr_poses_read=0;
    for( std::string line; getline( file, line ); ){
        //skip lines that start with # because that's a comment
        if( (ltrim_copy(line)).at(0)!='#' && !line.empty() ){

            //get from the line, the
            std::vector<std::string> tokens=radu::utils::split(line," ");
            std::string filename=tokens[0];

            // VLOG(1) << "line is " << line;

            //t
            float tx=std::stof(tokens[1]);
            float ty=std::stof(tokens[2]);
            float tz=std::stof(tokens[3]);
            //q
            float qx=std::stof(tokens[4]);
            float qy=std::stof(tokens[5]);
            float qz=std::stof(tokens[6]);
            float qw=std::stof(tokens[7]);
            //intrinsics
            float fx=std::stof(tokens[8]);
            float fy=std::stof(tokens[9]);
            float cx=std::stof(tokens[10]);
            float cy=std::stof(tokens[11]);

            //Make the matrices
            Eigen::Matrix3d K;
            K.setIdentity();
            Eigen::Affine3d tf_world_cam;
            tf_world_cam.setIdentity();
            Eigen::Quaterniond q;
            q.x()=qx;
            q.y()=qy;
            q.z()=qz;
            q.w()=qw;
            // q.normalize();
            tf_world_cam.translation()<<tx,ty,tz;
            tf_world_cam.linear()<<q.toRotationMatrix();
            K(0,0)=fx;
            K(1,1)=fy;
            K(0,2)=cx;
            K(1,2)=cy;
            K(2,2)=1.0;

            //push
            // VLOG(1) << "pushing for filename " << filename << " pose " <<tf_world_cam.matrix();
            m_filename2pose[filename]=tf_world_cam;
            m_filename2intrinsics[filename]=K;
            nr_poses_read++;

        }
    }

    CHECK(nr_poses_read!=0) << "There are not poses written in the pose_file " << pose_and_intrinsics_path;


}

void DataLoaderEasyPBR::read_data(){

    for (size_t i = 0; i < m_imgs_paths.size(); i++){

        Frame frame;

        fs::path img_path=m_imgs_paths[i];
        // VLOG(1) << "reading " << img_path;

        //get the idx
        std::string filename=img_path.stem().string();
        frame.frame_idx=std::stoi(filename);

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



        //extrinsics
        VLOG(1) << "img_path " <<img_path.filename().string();
        std::string key=img_path.filename().string();
        CHECK( m_filename2pose.find(key) != m_filename2pose.end() ) <<"Could not find the key " << key << " in the pose hashmap";
        CHECK( m_filename2intrinsics.find(key) != m_filename2intrinsics.end() ) <<"Could not find the key " << key << " in the intrinsics hashmap";

        frame.tf_cam_world=m_filename2pose[key].cast<float>().inverse();

        //flip z axis
        Eigen::Affine3f tf_world_cam=frame.tf_cam_world.inverse();
        Eigen::Matrix3f cam_axes;
        cam_axes=tf_world_cam.linear();
        cam_axes.col(2)=-cam_axes.col(2);
        tf_world_cam.linear()= cam_axes;
        frame.tf_cam_world=tf_world_cam.inverse();


        //intrinsics
        frame.K=m_filename2intrinsics[key].cast<float>();
        if(m_subsample_factor>1){
            // frame.K/=m_subsample_factor;
            // frame.K(2,2)=1.0;
            frame.rescale_K(1.0/m_subsample_factor);
        }
        // VLOG(1) << "K is" << frame.K;
        // VLOG(1) << "width and height is " << frame.width <<  " " << frame.height;


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



Frame DataLoaderEasyPBR::get_next_frame(){
    CHECK(m_idx_img_to_read<m_frames.size()) << "m_idx_img_to_read is out of bounds. It is " << m_idx_img_to_read << " while m_frames has size " << m_frames.size();
    Frame  frame= m_frames[m_idx_img_to_read];

    if(!m_do_overfit){
        m_idx_img_to_read++;
    }

    return frame;
}
std::vector<easy_pbr::Frame> DataLoaderEasyPBR::get_all_frames(){
    return m_frames;
}
Frame DataLoaderEasyPBR::get_frame_at_idx( const int idx){
    CHECK(idx<m_frames.size()) << "idx is out of bounds. It is " << idx << " while m_frames has size " << m_frames.size();

    Frame  frame= m_frames[idx];

    return frame;
}

Frame DataLoaderEasyPBR::get_random_frame(){
    CHECK(m_frames.size()>0 ) << "m_frames has size 0";

    int random_idx=m_rand_gen->rand_int(0, m_frames.size()-1);
    Frame  frame= m_frames[random_idx];

    return frame;
}
Frame DataLoaderEasyPBR::get_closest_frame( const easy_pbr::Frame& frame){

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


std::vector<easy_pbr::Frame>  DataLoaderEasyPBR::get_close_frames( const easy_pbr::Frame& frame, const int nr_frames, const bool discard_same_idx){

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
// std::vector<float> DataLoaderEasyPBR::compute_frame_weights( const easy_pbr::Frame& frame, std::vector<easy_pbr::Frame>& close_frames){
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





bool DataLoaderEasyPBR::is_finished(){
    //check if this loader has returned all the images it has
    if(m_idx_img_to_read<m_frames.size()){
        return false; //there is still more files to read
    }


    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderEasyPBR::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle && m_mode=="train"){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_frames), std::end(m_frames), rng_0);
    }

    m_idx_img_to_read=0;
}

int DataLoaderEasyPBR::nr_samples(){
    return m_frames.size();
}

bool DataLoaderEasyPBR::has_data(){
    return true; //we always have data since the loader stores all the image in memory and keeps them there
}

void DataLoaderEasyPBR::set_mode_train(){
    m_mode="train";
}
void DataLoaderEasyPBR::set_mode_test(){
    m_mode="test";
}
void DataLoaderEasyPBR::set_mode_validation(){
    m_mode="val";
}

void DataLoaderEasyPBR::set_shuffle(bool val){
    m_shuffle=val;
}
