#include "data_loaders/DataLoaderShapeNetPartSeg.h"

//ros
#include <ros/ros.h>

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//ros
#include "ros_utils.h"

//my stuff 
#include "data_loaders/DataTransformer.h"
#include "easy_pbr/Mesh.h"
// #include "data_loaders/utils/MiscUtils.h"
#include "Profiler.h"
#include "string_utils.h"
#include "eigen_utils.h"
#include "easy_pbr/LabelMngr.h"

//json 
#include "json11/json11.hpp"

//boost
namespace fs = boost::filesystem;


// using namespace er::utils;
using namespace easy_pbr::utils;


DataLoaderShapeNetPartSeg::DataLoaderShapeNetPartSeg(const std::string config_file):
    m_clouds_buffer(BUFFER_SIZE),
    m_is_running(false),
    m_idx_cloud_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator)
{
    init_params(config_file);
    // read_pose_file();
    create_transformation_matrices();
    // std::cout << " creating thread" << "\n";
    if(m_autostart){
        start();
    }
    // std::cout << " finidhed creating thread" << "\n";

}

DataLoaderShapeNetPartSeg::~DataLoaderShapeNetPartSeg(){

    m_is_running=false;
    m_loader_thread.join();
}

void DataLoaderShapeNetPartSeg::init_params(const std::string config_file){
    //get the config filename
    // ros::NodeHandle private_nh("~");
    // std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    // std::string config_file= "config.cfg";

    //read all the parameters
    // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_shapenet_partseg"];

    // m_nr_clouds_to_skip=loader_config["nr_clouds_to_skip"];
    // m_nr_clouds_to_read=loader_config["nr_clouds_to_read"];
    m_autostart=loader_config["autostart"];
    m_mode=(std::string)loader_config["mode"];
    m_shuffle_points=loader_config["shuffle_points"];
    m_normalize=loader_config["normalize"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_restrict_to_object= (std::string)loader_config["restrict_to_object"]; //makes it load clouds only from a specific object
    m_dataset_path = (std::string)loader_config["dataset_path"];    //get the path where all the off files are 


    //data transformer
    Config transformer_config=loader_config["transformer"];
    m_transformer=std::make_shared<DataTransformer>(transformer_config);

}

void DataLoaderShapeNetPartSeg::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";

    init_data_reading();

    m_is_running=true;
    m_loader_thread=std::thread(&DataLoaderShapeNetPartSeg::read_data, this);  //starts the spin in another thread
}



void DataLoaderShapeNetPartSeg::init_data_reading(){
    
    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }
    
    //read the mapping between the weird numbers in the files and the class label
    std::unordered_map<std::string, std::string> synsetoffset2category = read_mapping_synsetoffset2category( (m_dataset_path/"synsetoffset2category.txt").string() );

    //read the files from the train or test dataset depending on what the mode is
    fs::path file_list_json= m_dataset_path/"train_test_split"/ ("shuffled_"+m_mode+"_file_list.json");
    if(!fs::is_regular_file(file_list_json) ) { 
        LOG(FATAL) << "Json file list could not be found in " << file_list_json;
    }

    //read json
    std::string file_list_string=file_to_string(file_list_json.string());
    std::string err;
    const auto json = json11::Json::parse(file_list_string, err);
    for (auto &k : json.array_items()) {
        fs::path file_path= k.string_value();
        file_path=fs::relative(file_path, "shape_data");
        std::string pts_file_name=file_path.stem().string()+".pts";
        std::string label_file_name=file_path.stem().string()+".seg";
        // std::cout << file_path << std::endl;

        //if we do a restrict class grab only the files that are of the corresponding label
        if(!m_restrict_to_object.empty()){
            std::string current_synsetoffset=file_path.parent_path().string();
            // std::cout << " current synsetoffset is " << current_synsetoffset << std::endl;
            auto search = synsetoffset2category.find(current_synsetoffset);
            if (search == synsetoffset2category.end()) {
                LOG(FATAL) << "Could not find in the mapping the synsetoffset of " << current_synsetoffset;
            } 
            std::string current_object=synsetoffset2category[current_synsetoffset];
            // std::cout << " current class is " << current_class << std::endl;
            if(current_object!=m_restrict_to_object){
                continue;
            }
        }

        //for some reason the file_path stored in the json file is not exactly right and we shoudl acually put a "point" in between
        fs::path full_file_path= m_dataset_path/file_path.parent_path()/"points"/pts_file_name;
        m_pts_filenames.push_back(full_file_path);

        fs::path full_label_path= m_dataset_path/file_path.parent_path()/"points_label"/label_file_name;
        m_labels_filenames.push_back(full_label_path);
    }

    // shuffle the data if neccsary
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed); //create two engines with the same states so the vector are randomized in the same way
        auto rng_1 = rng_0;
        std::shuffle(std::begin(m_pts_filenames), std::end(m_pts_filenames), rng_0);
        std::shuffle(std::begin(m_labels_filenames), std::end(m_labels_filenames), rng_1);
    }

    //label file and colormap
    fs::path labels_file = fs::path(m_dataset_path).parent_path().parent_path() / "colorscheme_and_labels" / m_restrict_to_object/"labels.txt";
    fs::path colorscheme_file = fs::path(m_dataset_path).parent_path().parent_path() / "colorscheme_and_labels" / m_restrict_to_object/"color_scheme.txt";
    fs::path frequency_file = fs::path(m_dataset_path).parent_path().parent_path() / "colorscheme_and_labels" / m_restrict_to_object/"frequency_uniform.txt";
    int unlabeled_idx=0;
    m_label_mngr=std::make_shared<LabelMngr>(labels_file.string(), colorscheme_file.string(), frequency_file.string(), unlabeled_idx );
}

void DataLoaderShapeNetPartSeg::read_data(){

    loguru::set_thread_name("loader_thread_shapenet");


    while (m_is_running) {

        //we finished reading so we wait here for a reset
        if(m_idx_cloud_to_read>=m_pts_filenames.size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }


        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue

            fs::path pts_filename=m_pts_filenames[ m_idx_cloud_to_read ];
            fs::path labels_filename=m_labels_filenames[ m_idx_cloud_to_read ];
            if(!m_do_overfit){
                m_idx_cloud_to_read++;
            }

            // VLOG(1) << "Reading from object" << m_restrict_to_object;

            //read pts
            MeshSharedPtr cloud=Mesh::create();
            cloud->V=read_pts(pts_filename.string());
            cloud->L_gt=read_labels(labels_filename.string());
            cloud->D=cloud->V.rowwise().norm();
            if(m_normalize){
                cloud->normalize_size();
                cloud->normalize_position();
            }

            // VLOG(1) << "cloud->v is " << cloud->V;

            //transform
            // cloud.apply_transform(m_tf_worldGL_worldROS); // from worldROS to worldGL

            if(m_mode=="train"){
                cloud=m_transformer->transform(cloud);
            }

            if(m_shuffle_points){ //when splattin it is better if adyacent points in 3D space are not adyancet in memory so that we don't end up with conflicts or race conditions
                // https://stackoverflow.com/a/15866196
                Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(cloud->V.rows());
                perm.setIdentity();
                std::shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size(), m_rand_gen->generator());
                // VLOG(1) << "permutation matrix is " << perm.indices();
                // A_perm = A * perm; // permute columns
                cloud->V = perm * cloud->V; // permute rows
                cloud->L_gt = perm * cloud->L_gt; // permute rows
                cloud->D = perm * cloud->D; // permute rows
            }

            //some sensible visualization options
            cloud->m_vis.m_show_mesh=false;
            cloud->m_vis.m_show_points=true;
            cloud->m_vis.m_color_type=+MeshColorType::SemanticGT;
            
            //set the labelmngr which will be used by the viewer to put correct colors for the semantics
            // cloud->m_label_mngr=m_label_mngr->shared_from_this();
            cloud->m_label_mngr=m_label_mngr;

            // VLOG(1) << "Label uindx is " << cloud->m_label_mngr->get_idx_unlabeled();

            m_clouds_buffer.enqueue(cloud);

        }

    }

}


Eigen::MatrixXd DataLoaderShapeNetPartSeg::read_pts(const std::string file_path){
    std::ifstream infile( file_path );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pts file " << file_path;
    }

    std::vector<Eigen::Vector3d,  Eigen::aligned_allocator<Eigen::Vector3d>  > points_vec;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);


        std::vector<std::string> tokens=split(line," ");
        Eigen::Vector3d point;
        point.x()=stod(tokens[0]);
        point.y()=stod(tokens[1]);
        point.z()=stod(tokens[2]);

        points_vec.push_back(point);

    }

    return vec2eigen(points_vec);
}

Eigen::MatrixXi DataLoaderShapeNetPartSeg::read_labels(const std::string file_path){
    std::ifstream infile( file_path );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open labels file " << file_path;
    }

    std::vector< int> labels_vec;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);


        std::vector<std::string> tokens=split(line," ");
        int label=stoi(tokens[0]);

        labels_vec.push_back(label);

    }

    return vec2eigen(labels_vec);
}

std::unordered_map<std::string, std::string> DataLoaderShapeNetPartSeg::read_mapping_synsetoffset2category(const std::string file_path){
    std::ifstream infile( file_path );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open mapping file " << file_path;
    }

    std::unordered_map<std::string, std::string> synsetoffset2category; //mapping from the filename which a bunch of number to the class name;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);

        std::vector<std::string> tokens=split(line,"\t");
        CHECK(tokens.size()==2) << "The mapping file should have 2 columns only. However, this line has " << tokens.size() << "line is " << line ;
        std::string synsetoffset =trim_copy(tokens[1]);
        std::string label =lowercase(trim_copy(tokens[0]));

        // std::cout << "synsetoffset and labels are" << synsetoffset << " " << label << std::endl;


        synsetoffset2category[synsetoffset]=label;


    }

    return synsetoffset2category;
}


bool DataLoaderShapeNetPartSeg::has_data(){
    if(m_clouds_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    }
}


std::shared_ptr<Mesh> DataLoaderShapeNetPartSeg::get_cloud(){

    std::shared_ptr<Mesh> cloud;
    m_clouds_buffer.try_dequeue(cloud);

    return cloud;
}

bool DataLoaderShapeNetPartSeg::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<(int)m_pts_filenames.size()){
        return false; //there is still more files to read
    }

    //check that there is nothing in the ring buffers
    if(m_clouds_buffer.peek()!=nullptr){
        return false; //there is still something in the buffer
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderShapeNetPartSeg::is_finished_reading(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<(int)m_pts_filenames.size()){
        return false; //there is still more files to read
    }

    return true; //there is nothing more to read and so we are finished reading

}

void DataLoaderShapeNetPartSeg::reset(){

    m_nr_resets++;

    //reshuffle for the next epoch
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        unsigned seed = m_nr_resets;
        auto rng_0 = std::default_random_engine(seed); //create two engines with the same states so the vector are randomized in the same way
        auto rng_1 = rng_0;
        std::shuffle(std::begin(m_pts_filenames), std::end(m_pts_filenames), rng_0);
        std::shuffle(std::begin(m_labels_filenames), std::end(m_labels_filenames), rng_1);
    }

    m_idx_cloud_to_read=0;
}

int DataLoaderShapeNetPartSeg::nr_samples(){
    return m_pts_filenames.size();
}

std::shared_ptr<LabelMngr> DataLoaderShapeNetPartSeg::label_mngr(){
    CHECK(m_label_mngr) << "label_mngr was not created";
    return m_label_mngr;
}

void DataLoaderShapeNetPartSeg::set_mode_train(){
    m_mode="train";
}
void DataLoaderShapeNetPartSeg::set_mode_test(){
    m_mode="test";
}
void DataLoaderShapeNetPartSeg::set_mode_validation(){
    m_mode="val";
}


std::string DataLoaderShapeNetPartSeg::get_object_name(){
    return m_restrict_to_object;
}

void DataLoaderShapeNetPartSeg::set_object_name(const std::string object_name){
    //kill data loading thread 
    m_is_running=false;
    m_loader_thread.join();

    //clear all data 
    m_idx_cloud_to_read=0;
    m_nr_resets=0;
    m_pts_filenames.clear();
    m_labels_filenames.clear();
    // m_clouds_buffer.clear();
    //deque until ihe cloud buffer is empty 
    bool has_data=true;
    MeshSharedPtr dummy_cloud;
    while(has_data){
        has_data=m_clouds_buffer.try_dequeue(dummy_cloud);
    }

    //set the new object_name
    m_restrict_to_object=object_name;

    //start loading thread again
    start();

}



void DataLoaderShapeNetPartSeg::create_transformation_matrices(){

    m_tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3d worldGL_worldROS_rot;
    worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
}
