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
#include "eigen_utils.h"
#include "RandGenerator.h"
#include "easy_pbr/LabelMngr.h"

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
    m_rand_gen(new RandGenerator)
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
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are 


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
    for (fs::directory_iterator itr(chosen_object_path); itr!=fs::directory_iterator(); ++itr){
        fs::path scene_path= itr->path()/"rendering";
        m_scene_folders.push_back(scene_path);
    }

    // shuffle the data if neccsary
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed);
        std::shuffle(std::begin(m_scene_folders), std::end(m_scene_folders), rng_0);
    }


}

void DataLoaderShapeNetImg::start_reading_next_scene(){
    CHECK(m_is_running==false) << "The loader thread is already running. Wait until the scene is finished loading before loading a new one. You can check this with finished_reading_scene()";

    std::string scene_path=m_scene_folders[m_idx_scene_to_read].string();

    read_scene(scene_path);


    if(!m_do_overfit){
        m_idx_scene_to_read++;
    }

    m_is_running=true;
    m_loader_thread=std::thread(&DataLoaderShapeNetImg::read_scene, this, scene_path);  //starts to read in another thread
}


void DataLoaderShapeNetImg::read_scene(const std::string scene_path){
    // VLOG(1) <<" read from path " << scene_path;

    m_frames_for_scene.clear();

    //load all the scene for the chosen object
    for (fs::directory_iterator itr(scene_path); itr!=fs::directory_iterator(); ++itr){
        fs::path img_path= itr->path();
        //get only files that end in png
        // VLOG(1) << "img_path" <<img_path.filename();
        if(img_path.filename().string().find("png")!= std::string::npos){
            // VLOG(1) << "img path " << img_path;

            std::shared_ptr<Frame> frame=std::make_shared<Frame>();

            frame->rgb_8u=cv::imread(img_path.string());
            frame->rgb_8u.convertTo(frame->rgb_32f, CV_32FC3, 1.0/255.0);
            frame->width=frame->rgb_32f.cols;
            frame->height=frame->rgb_32f.rows;

            m_frames_for_scene.push_back(frame);

        }
    }

    CHECK(m_frames_for_scene.size()!=0) << "Clouldn't load any images for this scene in path " << scene_path; 

    
    //shuffle the images from this scene 
    unsigned seed = m_nr_resets;
    auto rng_0 = std::default_random_engine(seed);
    std::shuffle(std::begin(m_frames_for_scene), std::end(m_frames_for_scene), rng_0);


    m_is_running=false;
}


bool DataLoaderShapeNetImg::finished_reading_scene(){
    return !m_is_running;
}

std::shared_ptr<Frame> DataLoaderShapeNetImg::get_random_frame(){
    VLOG(1) << "calling get random frame ";
    int random_idx=m_rand_gen->rand_int(0, m_frames_for_scene.size()-1);
    // int random_idx=m_rand_gen->rand_int(0, 10);
    VLOG(1) << "random idx is " << random_idx << " m_frames_for_scene is " << m_frames_for_scene.size();
    return m_frames_for_scene[random_idx];
}


// void DataLoaderShapeNetImg::read_data(){

//     loguru::set_thread_name("loader_thread_shapenet");


//     while (m_is_running) {

//         //we finished reading so we wait here for a reset
//         if(m_idx_img_to_read>=m_pts_filenames.size()){
//             std::this_thread::sleep_for(std::chrono::milliseconds(300));
//             continue;
//         }


//         // std::cout << "size approx is " << m_queue.size_approx() << '\n';
//         // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
//         if(m_imgs_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
//             //read the frame and everything else and push it to the queue

//             TIME_SCOPE("load_shapenet")

//             fs::path pts_filename=m_pts_filenames[ m_idx_img_to_read ];
//             fs::path labels_filename=m_labels_filenames[ m_idx_img_to_read ];
//             if(!m_do_overfit){
//                 m_idx_img_to_read++;
//             }

//             // VLOG(1) << "Reading from object" << m_restrict_to_object;

//             //read pts
//             MeshSharedPtr cloud=Mesh::create();
//             cloud->V=read_pts(pts_filename.string());
//             cloud->L_gt=read_labels(labels_filename.string());
//             cloud->D=cloud->V.rowwise().norm();
//             if(m_normalize){
//                 cloud->normalize_size();
//                 cloud->normalize_position();
//             }

//             // VLOG(1) << "cloud->v is " << cloud->V;

//             //transform
//             // cloud.apply_transform(m_tf_worldGL_worldROS); // from worldROS to worldGL

//             if(m_mode=="train"){
//                 cloud=m_transformer->transform(cloud);
//             }

//             if(m_shuffle_points){ //when splattin it is better if adyacent points in 3D space are not adyancet in memory so that we don't end up with conflicts or race conditions
//                 // https://stackoverflow.com/a/15866196
//                 Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(cloud->V.rows());
//                 perm.setIdentity();
//                 std::shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size(), m_rand_gen->generator());
//                 // VLOG(1) << "permutation matrix is " << perm.indices();
//                 // A_perm = A * perm; // permute columns
//                 cloud->V = perm * cloud->V; // permute rows
//                 cloud->L_gt = perm * cloud->L_gt; // permute rows
//                 cloud->D = perm * cloud->D; // permute rows
//             }

//             //some sensible visualization options
//             cloud->m_vis.m_show_mesh=false;
//             cloud->m_vis.m_show_points=true;
//             cloud->m_vis.m_color_type=+MeshColorType::SemanticGT;
            
//             //set the labelmngr which will be used by the viewer to put correct colors for the semantics
//             // cloud->m_label_mngr=m_label_mngr->shared_from_this();
//             cloud->m_label_mngr=m_label_mngr;

//             // VLOG(1) << "Label uindx is " << cloud->m_label_mngr->get_idx_unlabeled();
//             cloud->m_disk_path=pts_filename.string();

//             m_imgs_buffer.enqueue(cloud);

//         }

//     }

// }


// Eigen::MatrixXd DataLoaderShapeNetImg::read_pts(const std::string file_path){
//     std::ifstream infile( file_path );
//     if(!infile.is_open()){
//         LOG(FATAL) << "Could not open pts file " << file_path;
//     }

//     std::vector<Eigen::Vector3d,  Eigen::aligned_allocator<Eigen::Vector3d>  > points_vec;
//     std::string line;
//     while (std::getline(infile, line)) {
//         std::istringstream iss(line);


//         std::vector<std::string> tokens=split(line," ");
//         Eigen::Vector3d point;
//         point.x()=stod(tokens[0]);
//         point.y()=stod(tokens[1]);
//         point.z()=stod(tokens[2]);

//         points_vec.push_back(point);

//     }

//     return vec2eigen(points_vec);
// }

// Eigen::MatrixXi DataLoaderShapeNetImg::read_labels(const std::string file_path){
//     std::ifstream infile( file_path );
//     if(!infile.is_open()){
//         LOG(FATAL) << "Could not open labels file " << file_path;
//     }

//     std::vector< int> labels_vec;
//     std::string line;
//     while (std::getline(infile, line)) {
//         std::istringstream iss(line);


//         std::vector<std::string> tokens=split(line," ");
//         int label=stoi(tokens[0]);

//         labels_vec.push_back(label);

//     }

//     return vec2eigen(labels_vec);
// }

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


// bool DataLoaderShapeNetImg::has_data(){
//     if(m_imgs_buffer.peek()==nullptr){
//         return false;
//     }else{
//         return true;
//     }
// }


// std::shared_ptr<Mesh> DataLoaderShapeNetImg::get_cloud(){

//     std::shared_ptr<Mesh> cloud;
//     m_imgs_buffer.try_dequeue(cloud);

//     return cloud;
// }

// bool DataLoaderShapeNetImg::is_finished(){
//     //check if this loader has loaded everything
//     if(m_idx_img_to_read<m_pts_filenames.size()){
//         return false; //there is still more files to read
//     }

//     //check that there is nothing in the ring buffers
//     if(m_imgs_buffer.peek()!=nullptr){
//         return false; //there is still something in the buffer
//     }

//     return true; //there is nothing more to read and nothing more in the buffer so we are finished

// }


// bool DataLoaderShapeNetImg::is_finished_reading(){
//     //check if this loader has loaded everything
//     if(m_idx_img_to_read<m_pts_filenames.size()){
//         return false; //there is still more files to read
//     }

//     return true; //there is nothing more to read and so we are finished reading

// }

// void DataLoaderShapeNetImg::reset(){

//     m_nr_resets++;

//     //reshuffle for the next epoch
//     if(m_shuffle){
//         // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//         unsigned seed = m_nr_resets;
//         auto rng_0 = std::default_random_engine(seed); //create two engines with the same states so the vector are randomized in the same way
//         auto rng_1 = rng_0;
//         std::shuffle(std::begin(m_pts_filenames), std::end(m_pts_filenames), rng_0);
//         std::shuffle(std::begin(m_labels_filenames), std::end(m_labels_filenames), rng_1);
//     }

//     m_idx_img_to_read=0;
// }

// int DataLoaderShapeNetImg::nr_samples(){
//     return m_pts_filenames.size();
// }

// std::shared_ptr<LabelMngr> DataLoaderShapeNetImg::label_mngr(){
//     CHECK(m_label_mngr) << "label_mngr was not created";
//     return m_label_mngr;
// }

// void DataLoaderShapeNetImg::set_mode_train(){
//     m_mode="train";
// }
// void DataLoaderShapeNetImg::set_mode_test(){
//     m_mode="test";
// }
// void DataLoaderShapeNetImg::set_mode_validation(){
//     m_mode="val";
// }


// std::string DataLoaderShapeNetImg::get_object_name(){
//     return m_restrict_to_object;
// }

// void DataLoaderShapeNetImg::set_object_name(const std::string object_name){
//     //kill data loading thread 
//     m_is_running=false;
//     m_loader_thread.join();

//     //clear all data 
//     m_idx_img_to_read=0;
//     m_nr_resets=0;
//     m_pts_filenames.clear();
//     m_labels_filenames.clear();
//     // m_imgs_buffer.clear();
//     //deque until ihe cloud buffer is empty 
//     bool has_data=true;
//     MeshSharedPtr dummy_cloud;
//     while(has_data){
//         has_data=m_imgs_buffer.try_dequeue(dummy_cloud);
//     }

//     //set the new object_name
//     m_restrict_to_object=object_name;

//     //start loading thread again
//     start();

// }



// void DataLoaderShapeNetImg::create_transformation_matrices(){

//     m_tf_worldGL_worldROS.setIdentity();
//     Eigen::Matrix3d worldGL_worldROS_rot;
//     worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
//     m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
// }
