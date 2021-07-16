#include "data_loaders/fb/DataLoaderUSCHair.h"

//c++
#include <algorithm>
#include <random>
#include <cstdio>

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//cnpy
// #include "cnpy.h"

#include <igl/point_mesh_squared_distance.h>
#include <igl/barycentric_coordinates.h>


//boost
#include <boost/range/iterator_range.hpp>

//my stuff
#include "easy_pbr/Mesh.h"
#include "easy_pbr/Scene.h"
// #include "easy_pbr/LabelMngr.h"
// #include "data_loaders/DataTransformer.h"
#include "Profiler.h"
#include "string_utils.h"
#include "eigen_utils.h"
#include "RandGenerator.h"

using namespace radu::utils;
using namespace easy_pbr;

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

DataLoaderUSCHair::DataLoaderUSCHair(const std::string config_file):
    m_is_modified(false),
    m_is_running(false),
    m_clouds_buffer(BUFFER_SIZE),
    m_idx_cloud_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator)
{

    init_params(config_file);
    // read_pose_file();
    // create_transformation_matrices();
    // std::cout << " creating thread" << "\n";
    if(m_autostart){
        // m_is_running=true;
        start();

        // if(m_load_buffered){
            // m_loader_thread=std::thread(&DataLoaderUSCHair::start, this);  //starts the spin in another thread
        // }else{
            // init_data_reading();
            // read_data(); //starts reading but on the main thread
        // }
    }
    // std::cout << " finidhed creating thread" << "\n";

}

DataLoaderUSCHair::~DataLoaderUSCHair(){

    // std::cout << "finishing" << std::endl;
    m_is_running=false;

    if(m_load_buffered){
        m_loader_thread.join();
    }
}

void DataLoaderUSCHair::init_params(const std::string config_file){
    //get the config filename
    // ros::NodeHandle private_nh("~");
    // std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    // std::string config_file="config.cfg";

    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config loader_config=cfg["loader_usc_hair"];

    m_autostart=loader_config["autostart"];
    m_mode=(std::string)loader_config["mode"];
    m_nr_clouds_to_skip=loader_config["nr_clouds_to_skip"];
    m_nr_clouds_to_read=loader_config["nr_clouds_to_read"];
    m_percentage_strand_drop=loader_config["percentage_strand_drop"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    m_load_buffered=loader_config["load_buffered"];
    // m_do_adaptive_subsampling=loader_config["do_adaptive_subsampling"];
    m_dataset_path=(std::string)loader_config["dataset_path"];
    m_scalp_mesh_path = (std::string)loader_config["scalp_mesh_path"];

}

void DataLoaderUSCHair::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";

    init_data_reading();

    m_is_running=true;
    if(m_load_buffered){
        m_loader_thread=std::thread(&DataLoaderUSCHair::read_data, this);  //starts the spin in another thread
    }else{
        read_data();
    }
}

void DataLoaderUSCHair::init_data_reading(){

    std::vector<fs::path> data_filenames_all;

    for (fs::directory_iterator itr(m_dataset_path); itr!=fs::directory_iterator(); ++itr){

        // VLOG(1) << "checing" << itr->path();
        // VLOG(1) << "filename " <<itr->path().filename().string();
        std::string filename_with_ext=itr->path().filename().string();
        if(   radu::utils::contains(filename_with_ext, "data") ){
            // VLOG(1) << "itr->path() " << itr->path();
            data_filenames_all.push_back(itr->path());
        }
    }


    //shuffle the filles to be read if necessary
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(data_filenames_all), std::end(data_filenames_all), rng);
    }


    //ADDS THE clouds to the member std_vector of paths
    //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    int nr_filenames_to_check= m_do_overfit? 1 : data_filenames_all.size(); //if we do overfit we only add one cloud
    for (size_t i = 0; i < nr_filenames_to_check; i++) {
        if( (int)i>=m_nr_clouds_to_skip && ((int)m_data_filenames.size()<m_nr_clouds_to_read || m_nr_clouds_to_read<0 ) ){
            m_data_filenames.push_back(data_filenames_all[i]);
        }
    }

    std::cout << "About to read " << m_data_filenames.size() << " clouds" <<std::endl;


    CHECK(m_data_filenames.size()>0) <<"We did not find any npz files to read";


}

void DataLoaderUSCHair::read_data(){

    VLOG(1) << "read data";

    //read head mesh
    m_mesh_head=Mesh::create( (m_dataset_path/"head_model.obj").string()  );
    m_mesh_scalp=Mesh::create(m_scalp_mesh_path.string());




    if(m_load_buffered){

        loguru::set_thread_name("loader_thread_kitti");


        while (m_is_running ) {

            //we finished reading so we wait here for a reset
            if(m_idx_cloud_to_read>=m_data_filenames.size()){
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                continue;
            }

            // std::cout << "size approx is " << m_queue.size_approx() << '\n';
            // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
            if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
                //read the frame and everything else and push it to the queue

                std::string data_filepath=m_data_filenames[ m_idx_cloud_to_read ].string();
                if(!m_do_overfit){
                    m_idx_cloud_to_read++;
                }


                std::vector< std::shared_ptr<easy_pbr::Mesh> > strands;
                std::shared_ptr<easy_pbr::Mesh> full_hair;
                auto ret=read_hair_sample(data_filepath);
                strands= std::get<0>(ret);
                full_hair= std::get<1>(ret);

                m_clouds_buffer.enqueue(full_hair);

            }

        }

    } else{


        //load into vec directly
        // int filenames_to_read= m_do_overfit? 1: m_data_filenames.size();
        // VLOG(1) << "filenames_to_read" << filenames_to_read;

        for(int i=0; i< m_data_filenames.size(); i++){
            std::string data_filepath=m_data_filenames[ i ].string();

            VLOG(1) << "data_filepath " << data_filepath;

            std::vector< std::shared_ptr<easy_pbr::Mesh> > strands;
            std::shared_ptr<easy_pbr::Mesh> full_hair;
            auto ret=read_hair_sample(data_filepath);
            strands= std::get<0>(ret);
            full_hair= std::get<1>(ret);

            VLOG(1) << "push";

            m_clouds_vec.push_back(full_hair);

        }

    }

}


std::tuple<
        std::vector< std::shared_ptr<easy_pbr::Mesh> >,
        std::shared_ptr<easy_pbr::Mesh>
    >
DataLoaderUSCHair::read_hair_sample(const std::string data_filepath){


    //read data. most of the code is from http://www-scf.usc.edu/~liwenhu/SHM/Hair.cc
    // VLOG(1) << "reading from " <<  data_filepath;

    std::FILE *f = std::fopen(data_filepath.c_str(), "rb");
    CHECK(f) << "Couldn't open" << data_filepath;


    TIME_START("load");


    std::vector< std::shared_ptr<easy_pbr::Mesh> > strands;
    std::shared_ptr<easy_pbr::Mesh> full_hair=easy_pbr::Mesh::create();
    std::vector<Eigen::Vector3d> full_hair_points_vec;
    std::vector<int> full_hair_strand_idx_vec;
    std::vector<Eigen::Vector3d> first_strand_points_vec;
    std::vector<double> strand_lengths_vec;
    int nr_strands_added=0;
    //debug
    // std::vector<Eigen::Vector3d> first_strand_hair_points_vec;

    int nstrands = 0;
    fread(&nstrands, 4, 1, f);
    // VLOG(1) << "nr strands" << nstrands;
    // if (!fread(&nstrands, 4, 1, f)) {
    //     fprintf(stderr, "Couldn't read number of strands\n");
    //     fclose(f);
    //     return false;
    // }
    strands.resize(nstrands);

    for (int i = 0; i < nstrands; i++) {
        // VLOG(1) << "strand " <<i;
        int nverts = 0;
        std::shared_ptr<easy_pbr::Mesh> strand= easy_pbr::Mesh::create();
        fread(&nverts, 4, 1, f);

        // if (nverts==1){
            // continue; //if the nr verts is 1 it means that there is no actual strand
        // }

        // VLOG(1) << "nrverts is " << nverts;
        // if (!fread(&nverts, 4, 1, f)) {
        //     fprintf(stderr, "Couldn't read number of vertices\n");
        //     fclose(f);
        //     return false;
        // }
        // strands[i].resize(nverts);
        strand->V.resize(nverts,3);
        // Eigen::VectorXf strand_points_float;
        // strand_points_float.resize(nverts,3);



        bool is_strand_valid=true;
        if (nverts==1){ //if the nr of vertices per strand is 1 it means that this is no actual strand, it's jsut the root node
            is_strand_valid=false;
        }
        //randomly drop some strands
        if (m_rand_gen->rand_bool(m_percentage_strand_drop)){
            is_strand_valid=false;
        }

        //store also the previous point on the strand so we can compute length
        double strand_length=0;
        Eigen::Vector3d prev_point;

        for (int j = 0; j < nverts; j++) {
            // VLOG(1) << "vert " <<j;
            // fread(&strand_points_float(i,0), 12, 1, f);
            float x,y,z;
            fread(&x, 4, 1, f);
            fread(&y, 4, 1, f);
            fread(&z, 4, 1, f);
            strand->V.row(j) << x,y,z;

            Eigen::Vector3d point;
            point << x,y,z;

            // VLOG(1) << "awdawd nrverts is " << nverts;
            if (is_strand_valid){
                // VLOG(1) << "adding";
                // if (j==0)
                full_hair_points_vec.push_back(point);
                // full_hair_strand_idx_vec.push_back(i);
                full_hair_strand_idx_vec.push_back(nr_strands_added);

                //if its the frist point, compute the uv coordinate of this first point by splatting it onto the scalp mesh
                if(j==0){
                    // Eigen::Vector2d = compute_closest_point_uv(m_mesh_scalp, point);
                    first_strand_points_vec.push_back(point);
                }

                //debug
                // if (i==0){
                    // first_strand_hair_points_vec.push_back(point);
                // }

            }

            //compute also the lenght of the strand
            if(j==1){ //if we are the first vertex, there is no previous
                float cur_segment_length= (point-prev_point).norm();
                strand_length+=cur_segment_length;
            }
            prev_point=point;




        }


        //finished reading this strand
        if (is_strand_valid){
            strands.push_back(strand);
            strand_lengths_vec.push_back(strand_length);
            nr_strands_added++;
        }
    }

    // VLOG(1) << "finished reading everything";

    fclose(f);
    // return true;

    //get the full cloud into one
    // std::shared_ptr<easy_pbr::Mesh> full_hair=easy_pbr::Mesh::create();

    // VLOG(1) << " full_hair_points_vec" << full_hair_points_vec.size();
    full_hair->V=vec2eigen(full_hair_points_vec);
    // full_hair->V=vec2eigen(first_strand_hair_points_vec);
    Eigen::MatrixXi strand_idx=vec2eigen(full_hair_strand_idx_vec);
    // VLOG(1) << "strand_idx" << strand_idx.rows() << " " <<strand_idx.cols();
    full_hair->add_extra_field("strand_idx", strand_idx);
    full_hair->m_vis.m_show_points=true;
    // VLOG(1) << "adding";
    // VLOG(1) << "finished adding";



    //compute the uv for the first points on the strand
    Eigen::MatrixXd uv_roots = compute_closest_point_uv(m_mesh_scalp, first_strand_points_vec);
    full_hair->add_extra_field("uv_roots", uv_roots);

    //get also the roots positions for each strand
    Eigen::MatrixXd position_roots=vec2eigen(first_strand_points_vec);
    full_hair->add_extra_field("position_roots", position_roots);

    //add also the strand length
    Eigen::MatrixXd strand_lengths=vec2eigen(strand_lengths_vec);
    full_hair->add_extra_field("strand_lengths", strand_lengths);



    TIME_END("load");

    auto ret= std::make_tuple(strands, full_hair);

    return ret;

}


Eigen::MatrixXd DataLoaderUSCHair::compute_closest_point_uv(std::shared_ptr<easy_pbr::Mesh> mesh, std::vector<Eigen::Vector3d> points_vec){

    Eigen::MatrixXd points=vec2eigen(points_vec);

    Eigen::VectorXd distances;
    Eigen::MatrixXd closest_points;
    Eigen::VectorXi closest_face_indices;
    igl::point_mesh_squared_distance(points, mesh->V, mesh->F, distances, closest_face_indices, closest_points );

    // VLOG(1) << "points" << points.rows() << "x " << points.cols();
    // VLOG(1) << "distances" << distances.rows() << "x " << distances.cols();
    // VLOG(1) << "closest_points" << closest_points.rows() << "x " << closest_points.cols();
    // VLOG(1) << "closest_face_indices" << closest_face_indices.rows() << "x " << closest_face_indices.cols();

    //get the uv
    Eigen::MatrixXd UV;
    UV.resize(points.rows(), 2);
    for(int i=0; i<points.rows(); i++){
        Eigen::MatrixXd closest_point = closest_points.row(i);
        int face_idx=closest_face_indices(i);
        Eigen::Vector3i face=mesh->F.row(face_idx);
        int idx_p0 = face.x();
        int idx_p1 = face.y();
        int idx_p2 = face.z();
        Eigen::Vector3d p0 = mesh->V.row(idx_p0);
        Eigen::Vector3d p1 = mesh->V.row(idx_p1);
        Eigen::Vector3d p2 = mesh->V.row(idx_p2);

        Eigen::MatrixXd barycentric;
        igl::barycentric_coordinates(closest_point, p0.transpose(), p1.transpose(), p2.transpose(), barycentric);

        float b0=barycentric(0,0);
        float b1=barycentric(0,1);
        float b2=barycentric(0,2);

        // if (i==0){
        //     VLOG(1) << " baryc is " << b0 << " " << b1 << " " << b2;
        //     VLOG(1) << " idx_p0 is " << idx_p0 << " " << idx_p1 << " " << idx_p2;
        //     VLOG(1) << "idx0 uv " << mesh->UV.row(idx_p0);
        // }

        Eigen::Vector2d uv_for_point = mesh->UV.row(idx_p0)*b0 + mesh->UV.row(idx_p1)*b1 + mesh->UV.row(idx_p2)*b2;

        // if(i==0){
            // VLOG(1) << "uv interpolated is " << uv_for_point;
        // }

        UV.row(i) = uv_for_point;

    }


    // //show the mesh
    // std::shared_ptr<easy_pbr::Mesh> closest_mesh= easy_pbr::Mesh::create();
    // // closest_mesh->V= closest_points;
    // closest_mesh->V= mesh->V;
    // closest_mesh->F.resize(1,3);
    // closest_mesh->F.row(0) =  mesh->F.row(closest_face_indices(0));
    // easy_pbr::Scene::show(closest_mesh,"closest_mesh");

    return UV;


}



bool DataLoaderUSCHair::has_data(){

    if(m_load_buffered){


        if(m_clouds_buffer.peek()==nullptr){
            return false;
        }else{
            return true;
        }

    }else{
        //if we dont laod buffered we always have some data because we store it persitently in the loader
        return true;
    }
}


std::shared_ptr<Mesh> DataLoaderUSCHair::get_cloud(){

    std::shared_ptr<Mesh> cloud;

    if(m_load_buffered){
        m_clouds_buffer.try_dequeue(cloud);
    }else{
        // VLOG(1) << "returning" << m_idx_cloud_to_read;
        cloud=m_clouds_vec[m_idx_cloud_to_read];
        cloud->m_is_dirty=true; //if we visualized this mesh before and we want to update it, we need to set the is_dirty
        cloud->m_is_shadowmap_dirty=true;
        m_idx_cloud_to_read++;
        // m_idx_cloud_to_return++;
        // if (m_idx_cloud_to_return>=m_clouds_vec.size()){
            // m_idx_cloud_to_return=0;
        // }
    }

    return cloud;
}

std::shared_ptr<Mesh> DataLoaderUSCHair::get_mesh_head(){
    return m_mesh_head;
}

std::shared_ptr<Mesh> DataLoaderUSCHair::get_mesh_scalp(){
    return m_mesh_scalp;
}

bool DataLoaderUSCHair::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<m_data_filenames.size()){
        return false; //there is still more files to read
    }

    //check that there is nothing in the ring buffers
    if(m_clouds_buffer.peek()!=nullptr){
        return false; //there is still something in the buffer
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderUSCHair::is_finished_reading(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<m_data_filenames.size()){
        return false; //there is still more files to read
    }

    return true; //there is nothing more to read and so we are finished reading

}

void DataLoaderUSCHair::reset(){
    m_nr_resets++;
    // we shuffle again the data so as to have freshly shuffled data for the next epoch
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // auto rng = std::default_random_engine(seed);
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(m_data_filenames), std::end(m_data_filenames), rng);
    }

    m_idx_cloud_to_read=0;
}

int DataLoaderUSCHair::nr_samples(){
    return m_data_filenames.size();
}
void DataLoaderUSCHair::set_mode_train(){
    m_mode="train";
}
void DataLoaderUSCHair::set_mode_test(){
    m_mode="test";
}
void DataLoaderUSCHair::set_mode_validation(){
    m_mode="val";
}
