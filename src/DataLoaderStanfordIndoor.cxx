#include "data_loaders/DataLoaderStanfordIndoor.h"

//c++
#include <algorithm>
#include <random>
#include <iostream>

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

//pybind so you can read the cloud from python
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

//my stuff
#include "data_loaders/DataTransformer.h"
#include "data_loaders/core/MeshCore.h"
#include "data_loaders/LabelMngr.h"
#include "data_loaders/utils/MiscUtils.h"
#include "data_loaders/utils/Profiler.h"

#include <boost/range.hpp>

using namespace radu::utils;
using namespace easy_pbr;

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

DataLoaderStanfordIndoor::DataLoaderStanfordIndoor(const std::string config_file):
    m_is_modified(false),
    m_is_running(false),
    m_clouds_buffer(BUFFER_SIZE),
    m_idx_cloud_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator)
{

    init_params(config_file);
    if(m_autostart){
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderStanfordIndoor::read_data, this);  //starts the spin in another thread
    }

}

DataLoaderStanfordIndoor::~DataLoaderStanfordIndoor(){

    m_is_running=false;

    if (m_loader_thread.joinable()){
        m_loader_thread.join();
    }
}

void DataLoaderStanfordIndoor::init_params(const std::string config_file){
    //get the config filename
    // ros::NodeHandle private_nh("~");
    // std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    // std::string config_file="config.cfg";

    //read all the parameters
    Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    Config loader_config=cfg["loader_stanford_indoor"];

    m_autostart=loader_config["autostart"];
    m_mode=(std::string)loader_config["mode"];
    m_read_original_data_and_reparse=loader_config["read_original_data_and_reparse"];
    m_fold=loader_config["fold"];
    m_nr_clouds_to_skip=loader_config["nr_clouds_to_skip"];
    m_nr_clouds_to_read=loader_config["nr_clouds_to_read"];
    m_max_nr_points_per_cloud=loader_config["max_nr_points_per_cloud"];
    m_shuffle_points=loader_config["shuffle_points"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_dataset_path=(std::string)loader_config["dataset_path"];

    //label file and colormap
    Config mngr_config=loader_config["label_mngr"];
    m_label_mngr=std::make_shared<LabelMngr>(mngr_config);

    //data transformer
    Config transformer_config=loader_config["transformer"];
    m_transformer=std::make_shared<DataTransformer>(transformer_config);


}

void DataLoaderStanfordIndoor::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";
    m_is_running=true;
    if(m_read_original_data_and_reparse){
        m_loader_thread=std::thread(&DataLoaderStanfordIndoor::read_data_and_reparse, this);  //starts the spin in another thread
    }else{
        m_loader_thread=std::thread(&DataLoaderStanfordIndoor::read_data, this);  //starts the spin in another thread
    }
}

bool DataLoaderStanfordIndoor::should_read_area(const int area_number){
    //indicates if I should read this area depending on the fold number and the mode we are in

    std::vector< std::vector<int> > fold2train; //maps from the fold number to the areas that should be used for training
    std::vector< std::vector<int> > fold2test; //maps from the fold number to the areas that should be used for testing

    fold2train.push_back( {1, 2, 3, 4, 6} );
    fold2train.push_back( {1, 3, 5, 6} );
    fold2train.push_back( {2, 4, 5} );
    fold2test.push_back( {5} );
    fold2test.push_back( {2, 4} );
    fold2test.push_back( {1, 3, 6} );

    if(m_mode=="train"){
        //search the vector of train in the corresponding fold
        //keep into acount that we use fold 1,2,3 but we index with them at 0,1,2

        std::vector<int> vec_train=fold2train[m_fold-1];
        //will return true if the number if found and false otherwise
        return std::find(vec_train.begin(), vec_train.end(), area_number) != vec_train.end();

    }else if(m_mode=="test"){
        std::vector<int> vec_test=fold2test[m_fold-1];
        //will return true if the number if found and false otherwise
        return std::find(vec_test.begin(), vec_test.end(), area_number) != vec_test.end();
    }else{
        LOG(FATAL) << "Not a known mode" << m_mode;
    }



}

void DataLoaderStanfordIndoor::init_data_reading(){

    //The dataset is decomposed in 6 Areas, each with different rooms and each has certain objects of a class. Each object is described by a file with columns xyz,rgb
    //depending on the mode we are in and the Fold we are using, we are going to read form one area or another one
    //the Fold is the split we use for training and testing. They are described here: http://buildingparser.stanford.edu/dataset.html
    //Minkovki engine uses Fold 1 so we shall too https://arxiv.org/pdf/1904.08755.pdf

    std::vector<fs::path> room_paths_all;

    if(m_read_original_data_and_reparse){
        //get how many areas we have here
        fs::path dataset_path_with_mode= m_dataset_path;
        if(!fs::is_directory(m_dataset_path)) {
            LOG(FATAL) << "No directory " << dataset_path_with_mode;
        }
        for(auto& area_path : boost::make_iterator_range(boost::filesystem::directory_iterator(m_dataset_path), {})){
            if(fs::is_directory(area_path)){

                //AREA

                //get the number of this area (1,2,3,4,5 or 6)
                int area_number= atoi( &area_path.path().stem().string().back() );
                VLOG(1) << "area number is " << area_number;
                if (should_read_area(area_number)){

                    VLOG(1) <<"reading area " << area_number << " when mode is " << m_mode;
                    for(auto& room_path : boost::make_iterator_range(boost::filesystem::directory_iterator(area_path.path() ), {})){
                        if(fs::is_directory(room_path)){

                            //ROOM
                            room_paths_all.push_back(room_path);


                        }
                    }

                }




            }
        }

    }else{
        //we are reading our binary data which should be much faster
        // LOG(FATAL) << "Not implemented yet";

        //get how many areas we have here
        fs::path dataset_path_with_mode= m_dataset_path;
        if(!fs::is_directory(m_dataset_path)) {
            LOG(FATAL) << "No directory " << dataset_path_with_mode;
        }
        for(auto& area_path : boost::make_iterator_range(boost::filesystem::directory_iterator(m_dataset_path), {})){
            if(fs::is_directory(area_path)){

                //AREA

                //get the number of this area (1,2,3,4,5 or 6)
                int area_number= atoi( &area_path.path().stem().string().back() );
                VLOG(1) << "area number is " << area_number;
                if (should_read_area(area_number)){

                    VLOG(1) <<"reading area " << area_number << " when mode is " << m_mode;
                    for(auto& room_path : boost::make_iterator_range(boost::filesystem::directory_iterator(area_path.path() ), {})){
                        if(fs::is_directory(room_path)){

                            //ROOM
                            room_paths_all.push_back(room_path/"room_binary.bin");


                        }
                    }

                }




            }
        }


    }




    //shuffle the filles to be read if necessary
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(room_paths_all), std::end(room_paths_all), rng);
    }


    //ADDS THE clouds to the member std_vector of paths
    //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    for (size_t i = 0; i < room_paths_all.size(); i++) {
        if( (int)i>=m_nr_clouds_to_skip && ((int)m_room_paths.size()<m_nr_clouds_to_read || m_nr_clouds_to_read<0 ) ){
            m_room_paths.push_back(room_paths_all[i]);
        }
    }

    std::cout << "About to read " << m_room_paths.size() << " clouds" <<std::endl;


    CHECK(m_room_paths.size()>0) <<"We did not find any rooms to read";

}

void DataLoaderStanfordIndoor::read_data_and_reparse(){

    loguru::set_thread_name("loader_thread_stanford");

    init_data_reading();

    while (m_is_running ) {

        //we finished reading so we wait here for a reset
        if(m_idx_cloud_to_read>=m_room_paths.size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue

            fs::path room_path=m_room_paths[ m_idx_cloud_to_read ];
            if(!m_do_overfit){
                m_idx_cloud_to_read++;
            }
            VLOG(1) <<"reading room " << room_path;


            std::vector<Eigen::VectorXd> xyz_vec;
            std::vector<Eigen::VectorXd> rgb_vec;
            std::vector<int> labels_vec;
            std::vector<float> all_room_data;

            fs::path annotations_path=room_path/"Annotations";
            for(auto& object_path : boost::make_iterator_range(boost::filesystem::directory_iterator(annotations_path), {})){

                //OBJECTS

                //Now we have to read each object in this room
                //get the name of the object
                std::string stem=object_path.path().stem().string();
                // VLOG(1) << "reading object " << object_path << "stem is " << stem;
                std::vector<std::string> tokens=er::utils::split(stem, "_");
                if (tokens.size()!=2){
                    continue; //some files are actually not objects. There are fucked up files there like Icon and stuff like that...
                }
                std::string object_name=tokens[0];
                int label_id=m_label_mngr->label2idx(object_name);

                //read all the points for this object and put them into a vector of xyz,rgb and labels for this room
                std::ifstream infile( object_path.path().string() );
                if(!infile.is_open()){
                    LOG(FATAL) << "Could not open object file " << object_path;
                    continue;
                }
                std::string line;
                Eigen::Vector3d xyz, rgb;

                // attempt 1 (about 400 ms)
                while (std::getline(infile, line)) {
                    std::istringstream iss(line);
                    std::vector<std::string> tokens=split(line," ");
                    if (tokens.size()!=6) {
                        LOG(WARNING) << "line " << line << "could not be parsed into 6 tokens. Dropping point";
                        continue;
                    }
                    // CHECK(tokens.size()==6) << "We have more than 6 tokens on line " << line;
                    xyz.x()=stod(tokens[0]);
                    xyz.y()=stod(tokens[1]);
                    xyz.z()=stod(tokens[2]);
                    rgb.x()=stod(tokens[3])/255.0;
                    rgb.y()=stod(tokens[4])/255.0;
                    rgb.z()=stod(tokens[5])/255.0;
                    xyz_vec.push_back(xyz);
                    rgb_vec.push_back(rgb);
                    labels_vec.push_back(label_id);
                    all_room_data.push_back(xyz.x());
                    all_room_data.push_back(xyz.y());
                    all_room_data.push_back(xyz.z());
                    all_room_data.push_back(rgb.x());
                    all_room_data.push_back(rgb.y());
                    all_room_data.push_back(rgb.z());
                    all_room_data.push_back(label_id);
                }

            }
            //add also the umber of elements as first
            // all_room_data.insert(all_room_data.begin(), xyz_vec.size() );
            // Profiler_ns::Profiler::print_all_stats();

            //Finished readin all the oject in the rooom. Now we combine the whole room into a mesh and return it
            MeshCore cloud;
            cloud.V=er::utils::vec2eigen(xyz_vec);
            cloud.C=er::utils::vec2eigen(rgb_vec);
            cloud.L_gt=er::utils::vec2eigen(labels_vec);
            // cloud.D=cloud.V.rowwise().norm();


            //the rooms are aligned in a weird manner. We rotate them as we see fit
            Eigen::Affine3d tf_worldGL_worldROS;
            tf_worldGL_worldROS.setIdentity();
            Eigen::Matrix3d worldGL_worldROS_rot;
            worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
            tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
            cloud.apply_transform(tf_worldGL_worldROS);

            //place the room at the 0.0.0 point otherwise it's just randomly in the space somewhere
            Eigen::Vector3d min=cloud.V.colwise().minCoeff();
            cloud.V.rowwise()-=min.transpose();



            // if(m_shuffle_points){ //when splattin it is better if adyacent points in 3D space are not adyancet in memory so that we don't end up with conflicts or race conditions
            //     // https://stackoverflow.com/a/15866196
            //     Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(cloud.V.rows());
            //     perm.setIdentity();
            //     std::shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size(), m_rand_gen->generator());
            //     // VLOG(1) << "permutation matrix is " << perm.indices();
            //     // A_perm = A * perm; // permute columns
            //     cloud.V = perm * cloud.V; // permute rows
            //     cloud.L_gt = perm * cloud.L_gt; // permute rows
            //     cloud.D = perm * cloud.D; // permute rows
            // }


            //store the cloud as binary of floats
            fs::path room_binary_file_path=room_path/"room_binary.bin";
            auto myfile = std::fstream(room_binary_file_path.string(), std::ios::out | std::ios::binary);
            myfile.write((char*)&all_room_data[0], all_room_data.size()*sizeof(float));
            myfile.close();





            //some sensible visualization options
            cloud.m_vis.m_show_mesh=false;
            cloud.m_vis.m_show_points=true;
            cloud.m_vis.m_color_type=+MeshColorType::SemanticGT;

            //set the labelmngr which will be used by the viewer to put correct colors for the semantics
            cloud.m_label_mngr=m_label_mngr->shared_from_this();

            m_clouds_buffer.enqueue(cloud);;



        }

    }

}

void DataLoaderStanfordIndoor::read_data(){

    loguru::set_thread_name("loader_thread_stanford");

    init_data_reading();

    while (m_is_running ) {

        //we finished reading so we wait here for a reset
        if(m_idx_cloud_to_read>=m_room_paths.size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue

            fs::path room_path=m_room_paths[ m_idx_cloud_to_read ];
            if(!m_do_overfit){
                m_idx_cloud_to_read++;
            }
            VLOG(1) <<"reading room " << room_path;


            TIME_START("read");
            TIME_START("seekign");
            //https://stackoverflow.com/questions/6755111/read-input-files-fastest-way-possible
            std::ifstream is(room_path.string());
            // Determine the file length
            is.seekg(0, std::ios_base::end);
            std::size_t size=is.tellg();
            is.seekg(0, std::ios_base::beg);
            TIME_END("seekign");
            // Create a vector to store the data
            VLOG(1) << "allocating vector of size" << size/sizeof(float);
            if (size/sizeof(float)>100000000){
                LOG(WARNING)<<"Size to allocate is to big. Ignoring this cloud";
                continue;
            }
            std::vector<float> v(size/sizeof(float));
            // Load the data
            is.read((char*) &v[0], size);
            // Close the file
            is.close();

            //parse the data from the file into the cloud taking into account that each 7 floats represent xyz and rgb ad label id
            int nr_points=v.size()/7;
            VLOG(1) << "reading nr of points" << nr_points;
            MeshCore cloud;
            cloud.V.resize(nr_points,3);
            cloud.C.resize(nr_points,3);
            cloud.L_gt.resize(nr_points,1);
            for(int i=0; i<nr_points; i++){
                // VLOG(1) << "reading point " << i << " of " << nr_points;
                cloud.V(i,0)=v[i*7+0];
                cloud.V(i,1)=v[i*7+1];
                cloud.V(i,2)=v[i*7+2];
                cloud.C(i,0)=v[i*7+3];
                cloud.C(i,1)=v[i*7+4];
                cloud.C(i,2)=v[i*7+5];
                cloud.L_gt(i,0)=std::round( v[i*7+6] );
            }

            //the stanford dataset is gigantic and sometimes we can't process all points, we establish a maximum amount of points we can process and drop the rest
            if (nr_points>m_max_nr_points_per_cloud && m_max_nr_points_per_cloud>0){
                LOG(WARNING)<< "Overstepping theshold of max nr of points of " << m_max_nr_points_per_cloud << " because we have nr of points " << nr_points << ". Dropping points until we only are left with the maximum we can process." ;
                //percentage of points we have to drop
                float percentage_to_drop=1.0-(float)m_max_nr_points_per_cloud/(float)nr_points;
                float prob_of_death=percentage_to_drop;
                std::vector<bool> is_vertex_to_be_removed(cloud.V.rows(), false);
                for(int i = 0; i < cloud.V.rows(); i++){
                    float random= m_rand_gen->rand_float(0.0, 1.0);
                    if(random<prob_of_death){
                        is_vertex_to_be_removed[i]=true;
                    }
                }
                cloud.remove_marked_vertices(is_vertex_to_be_removed, false);
            }



            //the rooms are aligned in a weird manner. We rotate them as we see fit
            Eigen::Affine3d tf_worldGL_worldROS;
            tf_worldGL_worldROS.setIdentity();
            Eigen::Matrix3d worldGL_worldROS_rot;
            worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
            tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
            cloud.apply_transform(tf_worldGL_worldROS);

            //place the room at the 0.0.0 point otherwise it's just randomly in the space somewhere
            Eigen::Vector3d min=cloud.V.colwise().minCoeff();
            cloud.V.rowwise()-=min.transpose();

            if(m_mode=="train"){
                cloud=m_transformer->transform(cloud);
            }

            if(m_shuffle_points){ //when splattin it is better if adyacent points in 3D space are not adyancet in memory so that we don't end up with conflicts or race conditions
                // https://stackoverflow.com/a/15866196
                Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(cloud.V.rows());
                perm.setIdentity();
                std::shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size(), m_rand_gen->generator());
                // VLOG(1) << "permutation matrix is " << perm.indices();
                // A_perm = A * perm; // permute columns
                cloud.V = perm * cloud.V; // permute rows
                cloud.L_gt = perm * cloud.L_gt; // permute rows
                cloud.D = perm * cloud.D; // permute rows
            }
            TIME_END("read");

            //some sensible visualization options
            cloud.m_vis.m_show_mesh=false;
            cloud.m_vis.m_show_points=true;
            cloud.m_vis.m_color_type=+MeshColorType::SemanticGT;

            //set the labelmngr which will be used by the viewer to put correct colors for the semantics
            cloud.m_label_mngr=m_label_mngr->shared_from_this();

            m_clouds_buffer.enqueue(cloud);;



        }

    }

}

bool DataLoaderStanfordIndoor::has_data(){
    if(m_clouds_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    }
}


MeshCore DataLoaderStanfordIndoor::get_cloud(){

    MeshCore cloud;
    m_clouds_buffer.try_dequeue(cloud);

    return cloud;
}

bool DataLoaderStanfordIndoor::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<(int)m_room_paths.size()){
        return false; //there is still more files to read
    }

    //check that there is nothing in the ring buffers
    if(m_clouds_buffer.peek()!=nullptr){
        return false; //there is still something in the buffer
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderStanfordIndoor::is_finished_reading(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<(int)m_room_paths.size()){
        return false; //there is still more files to read
    }

    return true; //there is nothing more to read and so we are finished reading

}

void DataLoaderStanfordIndoor::reset(){
    m_nr_resets++;
    // we shuffle again the data so as to have freshly shuffled data for the next epoch
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // auto rng = std::default_random_engine(seed);
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(m_room_paths), std::end(m_room_paths), rng);
    }

    //during training we do a mode of train and then a mode of test. After finishing test we call reset on it and if the only purpose was to just repasrse the data in binary then we are actually done
    if(m_read_original_data_and_reparse && m_mode=="test"){
        LOG(FATAL) << "finished writing everything and reparsing";
    }

    m_idx_cloud_to_read=0;
}

int DataLoaderStanfordIndoor::nr_samples(){
    return m_room_paths.size();
}
void DataLoaderStanfordIndoor::set_mode_train(){
    m_mode="train";
}
void DataLoaderStanfordIndoor::set_mode_test(){
    m_mode="test";
}
