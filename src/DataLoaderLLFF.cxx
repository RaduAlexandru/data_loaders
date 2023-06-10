#include "data_loaders/DataLoaderLLFF.h"

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


//cnpy
#include "cnpy.h"


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


DataLoaderLLFF::DataLoaderLLFF(const std::string config_file):
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

DataLoaderLLFF::~DataLoaderLLFF(){

    // m_is_running=false;
    // m_loader_thread.join();
}

void DataLoaderLLFF::init_params(const std::string config_file){


    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_llff"];

    m_autostart=loader_config["autostart"];
    m_subsample_factor=loader_config["subsample_factor"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_scene_scale_multiplier= loader_config.get_float_else_nan("scene_scale_multiplier");
    // m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are


    //data transformer
    // Config transformer_config=loader_config["transformer"];
    // m_transformer=std::make_shared<DataTransformer>(transformer_config);

}

void DataLoaderLLFF::start(){
    // init_data_reading();
    // init_extrinsics_and_intrinsics();
    read_data();
}


void DataLoaderLLFF::read_data(){

    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }




    //following the convention from https://github.com/googleinterns/IBRNet/blob/master/ibrnet/data_loaders/llff_data_utils.py
    std::string pose_path=(m_dataset_path/"poses_bounds.npy").string();

    //read npz
    cnpy::NpyArray arr = cnpy::npy_load( pose_path );

    // VLOG(1) << " array size" <<  arr.shape.size(); //returns 2
    // VLOG(1) << " array shape0 " <<  arr.shape[0]; //rreturns nr_poses
    // VLOG(1) << " array shape1 " <<  arr.shape[1]; //returns 17 , the first 15 are a 3x5 pose matrix and then 2 depth values fro near and far
    int rows = arr.shape[0];
    int cols = arr.shape[1];

    Eigen::MatrixXd arr_eigen;
    arr_eigen.resize( arr.shape[0], arr.shape[1] );
    const double* arr_data = arr.data<double>();
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            double val = arr_data[j+i*cols];
            arr_eigen(i,j) = val;
        }
    }
    arr_eigen.transposeInPlace();
    rows = arr.shape[1];
    cols = arr.shape[0];


    // VLOG(1) << "arr_eigen" << arr_eigen;

    //last to rows are 2xnr_poses for the near and far bound
    Eigen::MatrixXd bounds;
    bounds.resize(2, cols);
    // bounds = arr_eigen.block(rows-2,0,rows,2);
    bounds = arr_eigen.block(rows-2,0,2,cols);
    bounds.transposeInPlace();
    //if the scene_scale_multiplier was set to auto, thne we use a scene scale multiplier so that the far plane is at 1.0, so the scene is in the unit cube
    if (std::isnan(m_scene_scale_multiplier)){
        float max_far =bounds.col(1).maxCoeff();
        m_scene_scale_multiplier= 1.0/max_far;
    }

    bounds.array()*=m_scene_scale_multiplier;

    // VLOG(1) << "bounds " << bounds;
















    //try to read the imgs bin
    //attempt 2
    fs::path pose_file=m_dataset_path/"sparse/0"/"images.bin";

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


        std::string image_name;
        char name_char;
        do {
            file.read(&name_char, 1);
            if (name_char != '\0') {
            image_name += name_char;
            }
        } while (name_char != '\0');

        VLOG(1) << "image name " << image_name << "with img id" << image_id;

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

        Frame frame;
        fs::path img_path;
        img_path=m_dataset_path/"images"/image_name;

        //get the idx
        frame.cam_id=camera_id;
        frame.frame_idx=i;

        //depending on the mode we read this image or not
        //we use every 8th image because that is what IBRNet uses also since we use a modulo we actually need to do a modulo 9 to get the same results, we tested this using the dataloader llff_test from ibrner
        //so what it does is that the first frames counts as test then the next 8 for training, then one for test, next 8 for training and so on
        if (m_mode=="train" && frame.frame_idx%9==0){
            continue;
        }
        if (m_mode=="test" && frame.frame_idx%9!=0){
            continue;
        }

        VLOG(1) << "For mode " << m_mode << " read pose for image " << image_name;

        // read rgb
        frame.rgb_8u = cv::imread(img_path.string(), cv::IMREAD_UNCHANGED);
        if(m_subsample_factor>1){
            cv::Mat resized;
            cv::resize(frame.rgb_8u, resized, cv::Size(), 1.0/m_subsample_factor, 1.0/m_subsample_factor, cv::INTER_AREA);
            frame.rgb_8u=resized;
        }




        cv::cvtColor(frame.rgb_8u, frame.gray_8u, cv::COLOR_BGR2GRAY);
        frame.rgb_8u.convertTo(frame.rgb_32f, CV_32FC3, 1.0/255.0);
        // cv::cvtColor(frame.rgb_32f, frame.gray_32f, cv::COLOR_BGR2GRAY);
        frame.width=frame.rgb_32f.cols;
        frame.height=frame.rgb_32f.rows;

        //load gradients
        // cv::cvtColor(frame.rgb_32f, frame.gray_32f, cv::COLOR_BGR2GRAY);
        // cv::Scharr( frame.gray_32f, frame.grad_x_32f, CV_32F, 1, 0);
        // cv::Scharr( frame.gray_32f, frame.grad_y_32f, CV_32F, 0, 1);


        //extrinsics
        Eigen::Affine3d tf_cam_world;
        tf_cam_world.linear()=q.toRotationMatrix();
        tf_cam_world.translation()=t;

        // //rotate it a bit
        // Eigen::Quaterniond q_rot = Eigen::Quaterniond( Eigen::AngleAxis<double>( -180 * M_PI / 180.0 ,  Eigen::Vector3d::UnitX() ) );
        // Eigen::Affine3d rot;
        // rot.setIdentity();
        // rot.linear()=q_rot.toRotationMatrix();
        // tf_cam_world=tf_cam_world*rot;
        // //flip the y axis because for some reason colmap stores the positive Y towards down but I want it towards up
        // Eigen::Affine3d tf_world_cam =tf_cam_world.inverse();
        // tf_world_cam.matrix().col(1) = - tf_world_cam.matrix().col(1);
        // tf_cam_world=tf_world_cam.inverse();

        //attempt 2
        Eigen::Affine3d tf_world_cam =tf_cam_world.inverse();
        //nerf uses a opengl system https://github.com/bmild/nerf#already-have-poses
        //so the x is towards right, y up and z is backwards. 
        //we need x is right y down and z towards the frame
        tf_world_cam.linear().col(2)=-tf_world_cam.linear().col(2); //make it look in the correct direction (the z vector of the frame should point towards the image frame)
        // tf_world_cam.linear().col(1)=-tf_world_cam.linear().col(1);
        tf_cam_world=tf_world_cam.inverse();
        //since we flip y we also need to change the cy intrinsic


        // //rotate the world so that we have the top fo the dome in the y direction instead of z
        // Eigen::Affine3d m_tf_worldGL_world;
        // m_tf_worldGL_world.setIdentity();
        // Eigen::Matrix3d worldGL_world_rot;
        // worldGL_world_rot = Eigen::AngleAxisd(0.5*M_PI, Eigen::Vector3d::UnitX());
        // m_tf_worldGL_world.matrix().block<3,3>(0,0)=worldGL_world_rot;
        // tf_cam_world=tf_cam_world*m_tf_worldGL_world;

        //flip the world upside down
        tf_cam_world.linear().col(1) = -tf_cam_world.linear().col(1);



        frame.tf_cam_world=tf_cam_world.cast<float>();

        // //intrinsics we get later whne we read the cameras.bin

        //rescale things if necessary
        if(m_scene_scale_multiplier>0.0){
            Eigen::Affine3f tf_world_cam_rescaled = frame.tf_cam_world.inverse();
            tf_world_cam_rescaled.translation()*=m_scene_scale_multiplier;
            frame.tf_cam_world=tf_world_cam_rescaled.inverse();
        }

        //also add the near and far bound for the frame
        float near =  bounds(i,0);
        float far =  bounds(i,1);
        frame.add_extra_field("near", near);
        frame.add_extra_field("far", far);
        VLOG(1) << "Near and far that we have set is " << frame.get_extra_field<float>("near") << " " << frame.get_extra_field<float>("far");
        //add also the average of the near and far fields so that we can use the same near and far for all cameras and that the NDC calculated for each camera is the same
        float avg_near =bounds.col(0).mean();
        float avg_far =bounds.col(1).mean();
        frame.add_extra_field("avg_near", avg_near );
        frame.add_extra_field("avg_far", avg_far );
        VLOG(1) << "Average Near and far that we have set is " << frame.get_extra_field<float>("avg_near") << " " << frame.get_extra_field<float>("avg_far");
        float min_near =bounds.col(0).minCoeff();
        float max_far =bounds.col(1).maxCoeff();
        frame.add_extra_field("min_near", min_near );
        frame.add_extra_field("max_far", max_far );
        VLOG(1) << "Min and Max Near and far that we have set is " << frame.get_extra_field<float>("min_near") << " " << frame.get_extra_field<float>("max_far");



        m_frames.push_back(frame);


    }









     //read cameras intrinsics
    fs::path cameras_path=m_dataset_path/m_object_name/"sparse/0"/"cameras.bin";
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
    params.resize(6);
    ReadBinaryLittleEndian<double>(&camera_file, &params );

    VLOG(1) << "width and height" << width << " " << height ;
    VLOG(1) << "model id "  << model_id;
    for (size_t j = 0; j < params.size(); j++) {
        VLOG(1) << "param " << j << " is " <<params[j];
    }


    //   CHECK(params.size()==4) << " params should have size of 4, so it should contain fx,fy,cx,cy. So the camera model should be simple_pinhole. However the size is " << params.size();


      // VLOG(1) << "Read intrinsics for frame_idx " << camera_id;

      //get all the frames which have frame_idx to be camera_id and we set the params;
      for (size_t j = 0; j < m_frames.size(); j++) {
            Frame& frame = m_frames[j];
            if (frame.cam_id==(int)camera_id){
                //this correspond so we set it
                double fx,fy,cx,cy;
                fx=params[0];
                fy=params[0];
                cx=params[1];
                cy=params[2];

                frame.K.setIdentity();
                frame.K(0,0) = fx; //fx
                frame.K(1,1) = fy; //fy
                frame.K(0,2) = cx; //cx
                frame.K(1,2) = cy; //cy
                // frame.K = frame.K/m_subsample_factor;
                // frame.K(2,2)=1.0; //dividing by 2,4,8 etc depending on the subsample shouldn't affect the coordinate in the last row and last column which is always 1.0
                frame.rescale_K(1.0/m_subsample_factor);

                //since we flipped the pose y axis (from y pointing up like opengl to y pointing down like opencv), we need to also change cy
                //the y principal point needs to be flipped because it actually measures the distance from the bottom but we measure the distance from the top
                // frame.K(1,2) = frame.height - frame.K(1,2);

                //   VLOG(1) << "K is " << frame.K;

            }
      }


    }





    // shuffle the data if neccsary
    if(m_shuffle and m_mode=="train"){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_imgs_paths), std::end(m_imgs_paths), rng_0);
    }


}




Frame DataLoaderLLFF::get_next_frame(){
    CHECK(m_idx_img_to_read<(int)m_frames.size()) << "m_idx_img_to_read is out of bounds. It is " << m_idx_img_to_read << " while m_frames has size " << m_frames.size();
    Frame  frame= m_frames[m_idx_img_to_read];

    if(!m_do_overfit){
        m_idx_img_to_read++;
    }

    return frame;
}
Frame DataLoaderLLFF::get_frame_at_idx( const int idx){
    CHECK(idx<(int)m_frames.size()) << "idx is out of bounds. It is " << idx << " while m_frames has size " << m_frames.size();

    Frame  frame= m_frames[idx];

    return frame;
}

Frame DataLoaderLLFF::get_random_frame(){
    CHECK(m_frames.size()>0 ) << "m_frames has size 0";

    int random_idx=m_rand_gen->rand_int(0, m_frames.size()-1);
    Frame  frame= m_frames[random_idx];

    return frame;
}
Frame DataLoaderLLFF::get_closest_frame( const easy_pbr::Frame& frame){

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

std::vector<easy_pbr::Frame>  DataLoaderLLFF::get_close_frames( const easy_pbr::Frame& frame, const int nr_frames, const bool discard_same_idx){

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





bool DataLoaderLLFF::is_finished(){
    //check if this loader has returned all the images it has
    if(m_idx_img_to_read<(int)m_frames.size()){
        return false; //there is still more files to read
    }


    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


void DataLoaderLLFF::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_frames), std::end(m_frames), rng_0);
    }

    m_idx_img_to_read=0;
}

int DataLoaderLLFF::nr_samples(){
    return m_frames.size();
}

bool DataLoaderLLFF::has_data(){
    return true; //we always have data since the loader stores all the image in memory and keeps them there
}


void DataLoaderLLFF::set_mode_train(){
    m_mode="train";
}
void DataLoaderLLFF::set_mode_test(){
    m_mode="test";
}
void DataLoaderLLFF::set_mode_validation(){
    m_mode="val";
}
void DataLoaderLLFF::set_mode_all(){
    m_mode="all";
}





//Bunch of functions required to read binary files
//from colmap github src/util/endian.h
template <typename T>
T DataLoaderLLFF::ReadBinaryLittleEndian(std::istream* stream) {
  T data_little_endian;
  stream->read(reinterpret_cast<char*>(&data_little_endian), sizeof(T));
  return LittleEndianToNative(data_little_endian);
}

template <typename T>
void DataLoaderLLFF::ReadBinaryLittleEndian(std::istream* stream, std::vector<T>* data) {
  for (size_t i = 0; i < data->size(); ++i) {
    (*data)[i] = ReadBinaryLittleEndian<T>(stream);
  }
}
template <typename T>
T DataLoaderLLFF::LittleEndianToNative(const T x) {
  if (IsLittleEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}
template <typename T>
T DataLoaderLLFF::BigEndianToNative(const T x) {
  if (IsBigEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}

template <typename T>
T DataLoaderLLFF::NativeToLittleEndian(const T x) {
  if (IsLittleEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}
template <typename T>
T DataLoaderLLFF::NativeToBigEndian(const T x) {
  if (IsBigEndian()) {
    return x;
  } else {
    return ReverseBytes(x);
  }
}
template <typename T>
T DataLoaderLLFF::ReverseBytes(const T& data) {
  T data_reversed = data;
  std::reverse(reinterpret_cast<char*>(&data_reversed),
               reinterpret_cast<char*>(&data_reversed) + sizeof(T));
  return data_reversed;
}

inline bool DataLoaderLLFF::IsLittleEndian() {
#ifdef BOOST_BIG_ENDIAN
  return false;
#else
  return true;
#endif
}

inline bool DataLoaderLLFF::IsBigEndian() {
#ifdef BOOST_BIG_ENDIAN
  return true;
#else
  return false;
#endif
}
