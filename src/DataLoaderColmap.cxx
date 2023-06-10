#include "data_loaders/DataLoaderColmap.h"

#include <limits>

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
#include "json11/json11.hpp"

//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;


DataLoaderColmap::DataLoaderColmap(const std::string config_file):
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

DataLoaderColmap::~DataLoaderColmap(){

    // m_is_running=false;
    // if (m_loader_thread.joinable()){
    //   m_loader_thread.join();
    // }
}

void DataLoaderColmap::init_params(const std::string config_file){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_colmap"];

    m_autostart=loader_config["autostart"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_scene_scale_multiplier= loader_config["scene_scale_multiplier"];
    m_load_imgs_with_transparency=loader_config["load_imgs_with_transparency"];
    // m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are


    //data transformer
    // Config transformer_config=loader_config["transformer"];
    // m_transformer=std::make_shared<DataTransformer>(transformer_config);

}

void DataLoaderColmap::start(){
    // init_data_reading();
    // init_extrinsics_and_intrinsics();
    read_data();
}


// void DataLoaderColmap::init_data_reading(){

//     // if(!fs::is_directory(m_dataset_path)) {
//     //     LOG(FATAL) << "No directory " << m_dataset_path;
//     // }

//     // //go to the folder of train val or test depending on the mode in which we are one
//     // for (fs::directory_iterator itr(m_dataset_path/m_mode); itr!=fs::directory_iterator(); ++itr){
//     //     fs::path img_path= itr->path();
//     //     //we disregard the images that contain depth and normals, we load only the rgb
//     //     if (fs::is_regular_file(img_path) &&
//     //     img_path.filename().string().find("png") != std::string::npos &&
//     //     img_path.stem().string().find("depth")== std::string::npos &&
//     //     img_path.stem().string().find("normal")== std::string::npos   ){
//     //         m_imgs_paths.push_back(img_path);
//     //     }
//     // }
//     // CHECK( !m_imgs_paths.empty() ) << "Could not find any images in path " << m_dataset_path/m_mode;


//     // // shuffle the data if neccsary
//     // if(m_shuffle){
//     //     unsigned seed = m_nr_resets;
//     //     auto rng_0 = std::default_random_engine(seed);
//     //     std::shuffle(std::begin(m_imgs_paths), std::end(m_imgs_paths), rng_0);
//     // }


// }

void DataLoaderColmap::read_data(){
    // //read transforms_test.json (or whichever file is corresponding to the mode we are on)

    // //get the path to this json file
    // fs::path pose_file_json= m_dataset_path/("transforms_"+m_mode+".json");
    // if(!fs::is_regular_file(pose_file_json) ) {
    //     LOG(FATAL) << "Json file for the poses could not be found in " << pose_file_json;
    // }


    // //read json
    // std::string file_list_string=radu::utils::file_to_string(pose_file_json.string());
    // std::string err;
    // const auto json = json11::Json::parse(file_list_string, err);
    // m_camera_angle_x = json["camera_angle_x"].number_value();

    // //read all the poses
    // for (auto &k : json["frames"].array_items()) {
    //     fs::path key= k.string_value();

    //     std::string file_path=k["file_path"].string_value(); //will be something like ./test/r_0 but we only want the r_0 part
    //     std::vector<std::string> tokens= radu::utils::split(file_path, "/");
    //     std::string file_name= tokens[2];
    //     // VLOG(1) << "filename is" << file_name;


    //     //read the psoe as a 4x4 matrix
    //     Eigen::Affine3d tf_world_cam;
    //     int rows=4;
    //     for (int r = 0; r < 4; r++){
    //         for (int c = 0; c < 4; c++){
    //             // VLOG(1) << "tf matri is " << k["transform_matrix"][r][c].number_value();
    //             tf_world_cam.matrix()(r,c) =k["transform_matrix"][r][c].number_value();
    //         }
    //     }
    //     tf_world_cam.linear().col(2)=-tf_world_cam.linear().col(2); //make it look in the correct direction

    //     //rotate from their world to our opengl world by rotating along the x axis
    //     Eigen::Affine3d tf_worldGL_worldROS;
    //     tf_worldGL_worldROS.setIdentity();
    //     Eigen::Matrix3d worldGL_worldROS_rot;
    //     worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
    //     tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
    //     // transform_vertices_cpu(tf_worldGL_worldROS);
    //     tf_world_cam=tf_worldGL_worldROS*tf_world_cam;


    //     Eigen::Affine3d tf_cam_world=tf_world_cam.inverse();



    //     m_filename2pose[file_name]=tf_cam_world; //we want to store here the transrom from world to cam so the tf_cam_world


    // }








    //attempt 2
    fs::path pose_file=m_dataset_path/"sparse"/"images.bin";

    //read the bin file according to this format https://colmap.github.io/format.html#binary-file-format
    //reading a binary file is also guided by src/base/reconstruction.cc from the colmap github at least from version 3.6 at function Reconstruction::ReadImagesBinary
    //some types from https://github.com/colmap/colmap/blob/ff9a463067a2656d1f59d12109fe2931e29e3ca0/src/util/types.h
    // Unique identifier for cameras.
    typedef uint32_t camera_t;
    // Unique identifier for images.
    typedef uint32_t image_t;
    // Index per image, i.e. determines maximum number of 2D points per image.
    // typedef uint32_t point2D_t;
    // Unique identifier per added 3D point. Since we add many 3D points,
    // delete them, and possibly re-add them again, the maximum number of allowed
    // unique indices should be large.
    typedef uint64_t point3D_t;

    std::ifstream file(pose_file.string(), std::ios::binary);
    CHECK(file.is_open()) << "Could not open file " << pose_file;
    const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&file);
    VLOG(1) << "Reading nr of images: " << num_reg_images;
    for (size_t i = 0; i < num_reg_images; ++i) {

      image_t image_id = ReadBinaryLittleEndian<image_t>(&file);

      Eigen::Quaterniond q;
      q.w() =  ReadBinaryLittleEndian<double>(&file);
      q.x() =  ReadBinaryLittleEndian<double>(&file);
      q.y() =  ReadBinaryLittleEndian<double>(&file);
      q.z() =  ReadBinaryLittleEndian<double>(&file);
      // q.normalize();

      Eigen::Vector3d t;
      t.x() = ReadBinaryLittleEndian<double>(&file);
      t.y() = ReadBinaryLittleEndian<double>(&file);
      t.z() = ReadBinaryLittleEndian<double>(&file);

      camera_t camera_id = ReadBinaryLittleEndian<camera_t>(&file);


      VLOG(1) << "image_id" << image_id;


      std::string image_name;
      char name_char;
      do {
          file.read(&name_char, 1);
          if (name_char != '\0') {
          image_name += name_char;
          }
      } while (name_char != '\0');

      const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&file);


      std::vector<Eigen::Vector2d> points2D;
      points2D.reserve(num_points2D);
      std::vector<point3D_t> point3D_ids;
      point3D_ids.reserve(num_points2D);
      for (size_t j = 0; j < num_points2D; ++j) {
          const double x = ReadBinaryLittleEndian<double>(&file);
          const double y = ReadBinaryLittleEndian<double>(&file);
          points2D.emplace_back(x, y);
          point3D_ids.push_back(ReadBinaryLittleEndian<point3D_t>(&file));
      }


      //NOW we finished reading everything from the binary file regarding this image,  so now we can read the image itself
      // VLOG(1) << "Read pose for frame_idx " << camera_id;


      Frame frame;


      fs::path img_path;
      if(m_load_imgs_with_transparency){
        image_name.erase(image_name.length()-3);
        image_name=image_name+"png";
        img_path=m_dataset_path/"images_transparency"/image_name;
      }else{
        img_path=m_dataset_path/"images"/image_name;
      }

      //get the idx
      frame.cam_id=camera_id;
      frame.frame_idx=i;

      //depending on the mode we read this image or not
      if (m_mode=="train" && frame.frame_idx%3==0){
        continue;
      }
      if (m_mode=="test" && frame.frame_idx%3!=0){
        continue;
      }

      VLOG(1) << "Read pose for image " << image_name;

      //load actually the TRANSAPRENCY ONE
      if (m_load_imgs_with_transparency){
        cv::Mat rgba_8u = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
        if(m_subsample_factor>1){
            cv::Mat resized;
            cv::resize(rgba_8u, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
            rgba_8u=resized;
        }
        std::vector<cv::Mat> channels(4);
        cv::split(rgba_8u, channels);
        cv::threshold( channels[3], frame.mask, 0.0, 1.0, cv::THRESH_BINARY);
        channels.pop_back();
        cv::merge(channels, frame.rgb_8u);
      }else{
        // read rgb
        frame.rgb_8u = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
        if(m_subsample_factor>1){
            cv::Mat resized;
            cv::resize(frame.rgb_8u, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
            frame.rgb_8u=resized;
        }
      }






      cv::cvtColor(frame.rgb_8u, frame.gray_8u, cv::COLOR_BGR2GRAY);
      frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
      // cv::cvtColor(frame.rgb_32f, frame.gray_32f, cv::COLOR_BGR2GRAY);
      frame.width=frame.rgb_32f.cols;
      frame.height=frame.rgb_32f.rows;

      //load gradients
      cv::cvtColor(frame.rgb_32f, frame.gray_32f, cv::COLOR_BGR2GRAY);
      cv::Scharr( frame.gray_32f, frame.grad_x_32f, CV_32F, 1, 0);
      cv::Scharr( frame.gray_32f, frame.grad_y_32f, CV_32F, 0, 1);


      //extrinsics
      Eigen::Affine3d tf_cam_world;
      tf_cam_world.linear()=q.toRotationMatrix();
      tf_cam_world.translation()=t;
      //rotate it a bit
      Eigen::Quaterniond q_rot = Eigen::Quaterniond( Eigen::AngleAxis<double>( -90 * M_PI / 180.0 ,  Eigen::Vector3d::UnitX() ) );
      Eigen::Affine3d rot;
      rot.setIdentity();
      rot.linear()=q_rot.toRotationMatrix();
      tf_cam_world=tf_cam_world*rot;
      //flip the y axis because for some reason colmap stores the positive Y towards down but I want it towards up
      Eigen::Affine3d tf_world_cam =tf_cam_world.inverse();
      tf_world_cam.matrix().col(1) = - tf_world_cam.matrix().col(1);
      // tf_world_cam.translation()/=3.0;
      tf_cam_world=tf_world_cam.inverse();


      frame.tf_cam_world=tf_cam_world.cast<float>();

      // //intrinsics we get later whne we read the cameras.bin

      //rescale things if necessary
      if(m_scene_scale_multiplier>0.0){
          Eigen::Affine3f tf_world_cam_rescaled = frame.tf_cam_world.inverse();
          tf_world_cam_rescaled.translation()*=m_scene_scale_multiplier;
          frame.tf_cam_world=tf_world_cam_rescaled.inverse();
      }


      m_frames.push_back(frame);




    }



    //read cameras intrinsics
    fs::path cameras_path=m_dataset_path/"sparse"/"cameras.bin";
    std::ifstream camera_file(cameras_path.string(), std::ios::binary);
    CHECK(camera_file.is_open()) << "Could not open file " << pose_file;

    const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&camera_file);
    VLOG(1) << "Reading intrinsics for nr of cameras: " << num_cameras;
    for (size_t i = 0; i < num_cameras; ++i) {
      // class Camera camera;
      camera_t camera_id = ReadBinaryLittleEndian<camera_t>(&camera_file);
      int model_id =ReadBinaryLittleEndian<int>(&camera_file);
      uint64_t width = ReadBinaryLittleEndian<uint64_t>(&camera_file);
      uint64_t height = ReadBinaryLittleEndian<uint64_t>(&camera_file);
      std::vector<double> params;
      params.resize(4);
      ReadBinaryLittleEndian<double>(&camera_file, &params );


      VLOG(1) << "width and height" << width << " " << height;
      VLOG(1) << "model id " << model_id;

      // CHECK(params.size()==4) << " params should have size of 4, so it should contain fx,fy,cx,cy. So the camera model should be simple_pinhole. However the size is " << params.size();

      // VLOG(1) << "Read intrinsics for frame_idx " << camera_id;

      //get all the frames which have frame_idx to be camera_id and we set the params;
      for (size_t j = 0; j < m_frames.size(); j++) {
        Frame& frame = m_frames[j];
        if (frame.cam_id==(int)camera_id){
          //this correspond so we set it
          double fx,fy,cx,cy;
          fx=params[0];
          fy=params[1];
          cx=params[2];
          cy=params[3];

          frame.K.setIdentity();
          frame.K(0,0) = fx; //fx
          frame.K(1,1) = fy; //fy
          frame.K(0,2) = cx; //cx
          frame.K(1,2) = cy; //cy
          // frame.K = frame.K/m_subsample_factor;
          // frame.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0
          frame.rescale_K(1.0/m_subsample_factor);

          // VLOG(1) << "K is " << frame.K;

        }

      }


    }



}

// void DataLoaderColmap::read_data(){

//     // for (size_t i = 0; i < m_imgs_paths.size(); i++){

//     //     Frame frame;

//     //     fs::path img_path=m_imgs_paths[i];
//     //     // VLOG(1) << "reading " << img_path;

//     //     //get the idx
//     //     std::string filename=img_path.stem().string();
//     //     std::vector<std::string> tokens=radu::utils::split(filename,"_");
//     //     frame.frame_idx=std::stoi(tokens[1]);

//     //     //read rgba and split into rgb and alpha mask
//     //     cv::Mat rgba_8u = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
//     //     if(m_subsample_factor>1){
//     //         cv::Mat resized;
//     //         cv::resize(rgba_8u, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
//     //         rgba_8u=resized;
//     //     }
//     //     std::vector<cv::Mat> channels(4);
//     //     cv::split(rgba_8u, channels);
//     //     cv::threshold( channels[3], frame.mask, 0.0, 1.0, cv::THRESH_BINARY);
//     //     channels.pop_back();
//     //     cv::merge(channels, frame.rgb_8u);


//     //     cv::cvtColor(frame.rgb_8u, frame.gray_8u, cv::COLOR_BGR2GRAY);
//     //     frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
//     //     // cv::cvtColor(frame.rgb_32f, frame.gray_32f, cv::COLOR_BGR2GRAY);
//     //     frame.width=frame.rgb_32f.cols;
//     //     frame.height=frame.rgb_32f.rows;

//     //     // //if we are loading the test one, get also the depth
//     //     // if(m_mode=="test"){
//     //     //     fs::path parent=img_path.parent_path();
//     //     //     std::string img_filename=img_path.stem().string();
//     //     //     // VLOG(1) << "parent" << parent;
//     //     //     // fs::path depth_img_path=
//     //     //     fs::path depth_img_path=parent/(img_filename+"_depth_0001.png");
//     //     //     VLOG(1) << "depth img path" << depth_img_path;

//     //     //     cv::Mat depth=cv::imread(depth_img_path.string() , cv::IMREAD_ANYDEPTH);
//     //     //     CHECK(!depth.empty()) << "The depth image is empty at path " << depth_img_path;
//     //     //     // depth.convertTo(frame.depth, CV_32FC1, 1.0/1000.0); //the depth was stored in mm but we want it in meters
//     //     //     depth.convertTo(frame.depth, CV_32FC1, 1.0/1000.0); //the depth was stored in cm but we want it in meters
//     //     //     // frame.depth=1.0/frame.depth; //seems to be the inverse depth
//     //     // }


//     //     //extrinsics
//     //     frame.tf_cam_world=m_filename2pose[img_path.stem().string()].cast<float>();

//     //     //intrinsics got mostly from here https://github.com/bmild/nerf/blob/0247d6e7ede8d918bc1fab2711f845669aee5e03/load_blender.py
//     //     frame.K.setIdentity();
//     //     float focal = 0.5 * frame.width / std::tan(0.5 * m_camera_angle_x);
//     //     frame.K(0,0) = focal;
//     //     frame.K(1,1) = focal;
//     //     frame.K(0,2) = frame.width/2.0; //no need to subsample the cx and cy because the frame width already refers to the subsampled iamge
//     //     frame.K(1,2) = frame.height/2.0;
//     //     frame.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0

//     //     m_frames.push_back(frame);
//     //     // VLOG(1) << "pushback and frames is " << m_frames.size();


//     // }







// }



Frame DataLoaderColmap::get_next_frame(){
    CHECK(m_idx_img_to_read<(int)m_frames.size()) << "m_idx_img_to_read is out of bounds. It is " << m_idx_img_to_read << " while m_frames has size " << m_frames.size();
    Frame  frame= m_frames[m_idx_img_to_read];

    if(!m_do_overfit){
        m_idx_img_to_read++;
    }

    return frame;
}
Frame DataLoaderColmap::get_frame_at_idx( const int idx){
    CHECK(idx<(int)m_frames.size()) << "idx is out of bounds. It is " << idx << " while m_frames has size " << m_frames.size();

    Frame  frame= m_frames[idx];

    return frame;
}

Frame DataLoaderColmap::get_random_frame(){
    CHECK(m_frames.size()>0 ) << "m_frames has size 0";

    int random_idx=m_rand_gen->rand_int(0, m_frames.size()-1);
    Frame  frame= m_frames[random_idx];

    return frame;
}
Frame DataLoaderColmap::get_closest_frame( const easy_pbr::Frame& frame){

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

std::vector<easy_pbr::Frame>  DataLoaderColmap::get_close_frames( const easy_pbr::Frame& frame, const int nr_frames, const bool discard_same_idx){

    CHECK(nr_frames<(int)m_frames.size()) << "Cannot select more close frames than the total nr of frames that we have in the loader. Required select of " << nr_frames << " out of a total of " << m_frames.size() << " available in the loader";

    std::vector<easy_pbr::Frame> selected_close_frames;

    for(int i=0; i<nr_frames; i++){

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





bool DataLoaderColmap::is_finished(){
    //check if this loader has returned all the images it has
    if(m_idx_img_to_read<(int)m_frames.size()){
        return false; //there is still more files to read
    }


    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderColmap::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_frames), std::end(m_frames), rng_0);
    }

    m_idx_img_to_read=0;
}

int DataLoaderColmap::nr_samples(){
    return m_frames.size();
}

bool DataLoaderColmap::has_data(){
    return true; //we always have data since the loader stores all the image in memory and keeps them there
}


void DataLoaderColmap::set_mode_train(){
    m_mode="train";
}
void DataLoaderColmap::set_mode_test(){
    m_mode="test";
}
void DataLoaderColmap::set_mode_validation(){
    m_mode="val";
}
void DataLoaderColmap::set_mode_all(){
    m_mode="all";
}





//Bunch of functions required to read binary files
//from colmap github src/util/endian.h
template <typename T>
T DataLoaderColmap::ReadBinaryLittleEndian(std::istream* stream) {
  T data_little_endian;
  stream->read(reinterpret_cast<char*>(&data_little_endian), sizeof(T));
  return LittleEndianToNative(data_little_endian);
}

template <typename T>
void DataLoaderColmap::ReadBinaryLittleEndian(std::istream* stream, std::vector<T>* data) {
  for (size_t i = 0; i < data->size(); ++i) {
    (*data)[i] = ReadBinaryLittleEndian<T>(stream);
  }
}
template <typename T>
T DataLoaderColmap::LittleEndianToNative(const T x) {
  if (IsLittleEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}
template <typename T>
T DataLoaderColmap::BigEndianToNative(const T x) {
  if (IsBigEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T DataLoaderColmap::NativeToLittleEndian(const T x) {
  if (IsLittleEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}
template <typename T>
T DataLoaderColmap::NativeToBigEndian(const T x) {
  if (IsBigEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}
template <typename T>
T DataLoaderColmap::ReverseBytes(const T& data) {
  T data_reversed = data;
  std::reverse(reinterpret_cast<char*>(&data_reversed),
               reinterpret_cast<char*>(&data_reversed) + sizeof(T));
  return data_reversed;
}

inline bool DataLoaderColmap::IsLittleEndian() {
#ifdef BOOST_BIG_ENDIAN
  return false;
#else
  return true;
#endif
}

inline bool DataLoaderColmap::IsBigEndian() {
#ifdef BOOST_BIG_ENDIAN
  return true;
#else
  return false;
#endif
}
