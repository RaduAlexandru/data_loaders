#include "data_loaders/DataLoaderPhenorob.h"

//c++
#include <algorithm>
#include <random>

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

//boost
#include <boost/range/iterator_range.hpp>

//my stuff 
#include "easy_pbr/Mesh.h"
#include "easy_pbr/LabelMngr.h"
#include "data_loaders/DataTransformer.h"
#include "Profiler.h"
#include "string_utils.h"
#include "eigen_utils.h"
#include "RandGenerator.h"

using namespace radu::utils;
using namespace easy_pbr;

#define BUFFER_SIZE 5 //clouds are stored in a queue until they are acessed, the queue stores a maximum of X items

DataLoaderPhenorob::DataLoaderPhenorob(const std::string config_file):
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
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderPhenorob::read_data, this);  //starts the spin in another thread
    }
    // std::cout << " finidhed creating thread" << "\n";

}

DataLoaderPhenorob::~DataLoaderPhenorob(){

    // std::cout << "finishing" << std::endl;
    m_is_running=false;

    m_loader_thread.join();
}

void DataLoaderPhenorob::init_params(const std::string config_file){
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
    Config loader_config=cfg["loader_phenorob"];

    m_autostart=loader_config["autostart"];
    // m_mode=(std::string)loader_config["mode"];
    m_nr_clouds_to_skip=loader_config["nr_clouds_to_skip"];
    m_nr_clouds_to_read=loader_config["nr_clouds_to_read"];
    m_shuffle_points=loader_config["shuffle_points"];
    m_normalize=loader_config["normalize"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    // m_do_adaptive_subsampling=loader_config["do_adaptive_subsampling"];
    m_dataset_path=(std::string)loader_config["dataset_path"];
    // m_sequence=(std::string)loader_config["sequence"];

    //label file and colormap
    Config mngr_config=loader_config["label_mngr"];
    m_label_mngr=std::make_shared<LabelMngr>(mngr_config);

    //data transformer
    Config transformer_config=loader_config["transformer"];
    m_transformer=std::make_shared<DataTransformer>(transformer_config);


}

void DataLoaderPhenorob::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";

    init_data_reading();

    m_is_running=true;
    m_loader_thread=std::thread(&DataLoaderPhenorob::read_data, this);  //starts the spin in another thread
}

void DataLoaderPhenorob::init_data_reading(){

    std::vector<fs::path> sample_filenames_all;

    //get how many sequnces we have here
    if(!fs::is_directory(m_dataset_path)) {
        LOG(FATAL) << "No directory " << m_dataset_path;
    }
    for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(m_dataset_path), {})){
        sample_filenames_all.push_back(entry );
    }




    if(!m_shuffle){ //if we are shuffling, there is no need to sort them
        std::sort(sample_filenames_all.begin(), sample_filenames_all.end());
    }



    //ADDS THE clouds to the member std_vector of paths 
    //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    for (size_t i = 0; i < sample_filenames_all.size(); i++) {
        if( (int)i>=m_nr_clouds_to_skip && ((int)m_sample_filenames.size()<m_nr_clouds_to_read || m_nr_clouds_to_read<0 ) ){
            m_sample_filenames.push_back(sample_filenames_all[i]);
        }
    }

    //shuffle the filles to be read if necessary
    if(m_shuffle){
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(m_sample_filenames), std::end(m_sample_filenames), rng);
    }

    std::cout << "About to read " << m_sample_filenames.size() << " clouds" <<std::endl; 


    CHECK(m_sample_filenames.size()>0) <<"We did not find any files to read";

}

void DataLoaderPhenorob::read_data(){

    loguru::set_thread_name("loader_thread_kitti");


    while (m_is_running ) {

        //we finished reading so we wait here for a reset
        if(m_idx_cloud_to_read>=m_sample_filenames.size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue

            fs::path sample_filename=m_sample_filenames[ m_idx_cloud_to_read ];
            if(!m_do_overfit){
                m_idx_cloud_to_read++;
            }
            // VLOG(1) << "reading " << sample_filename;


            std::vector<Eigen::VectorXd> points_vec;
            std::vector<int> labels_vec;

            //read the text and each line contains xyz,label
            std::ifstream f(sample_filename.string());
            std::string line;    
            while (std::getline(f, line)) {
                // std::cout << line << std::endl;
                std::vector<std::string> tokens=split(line, " ");
                // VLOG(1) << "lne is " << line;
                CHECK(tokens.size()==4) << "We expect to have 4 tokens corresponding to xyz,label but we got " << tokens.size() << " and the line is " << line;

                Eigen::VectorXd point_eigen(3);
                point_eigen<< std::stof(tokens[0]),  std::stof(tokens[1]),  std::stof(tokens[2]);
                points_vec.push_back(point_eigen);

                int label=std::stoi(tokens[3]);
                if(label>=2){
                    label=2;
                }
                labels_vec.push_back(label);

            }


            //copy into EigenMatrix 
            MeshSharedPtr cloud=Mesh::create();
            cloud->V=vec2eigen(points_vec);
            cloud->L_gt=vec2eigen(labels_vec);
        
            cloud=m_transformer->transform(cloud);

            if(m_normalize){
                cloud->normalize_size();
                // cloud->normalize_position();
            }

            // cloud->normalize_position();
            Eigen::Vector3d axis;
            axis << 1.0, 0.0, 0.0;
            // Eigen::Vector3d mov;
            // mov << 0.0, 0.0, -10.0;
            Eigen::Affine3d move;
            move.setIdentity();
            move.translation()<<50.0, 0.0, -740.0;
            cloud->rotate_model_matrix(axis, -90);
            cloud->apply_model_matrix_to_cpu(true);
            cloud->transform_vertices_cpu(move);


            if(m_shuffle_points){ //when splattin it is better if adyacent points in 3D space are not adyancet in memory so that we don't end up with conflicts or race conditions
                // https://stackoverflow.com/a/15866196
                Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(cloud->V.rows());
                perm.setIdentity();
                std::shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size(), m_rand_gen->generator());
                // VLOG(1) << "permutation matrix is " << perm.indices();
                // A_perm = A * perm; // permute columns
                cloud->V = perm * cloud->V; // permute rows
                cloud->L_gt = perm * cloud->L_gt; // permute rows
                // cloud->D = perm * cloud->D; // permute rows
            }

            //some sensible visualization options
            cloud->m_vis.m_show_mesh=false;
            cloud->m_vis.m_show_points=true;
            cloud->m_vis.m_color_type=+MeshColorType::SemanticGT;

            //set the labelmngr which will be used by the viewer to put correct colors for the semantics
            cloud->m_label_mngr=m_label_mngr->shared_from_this();

            cloud->m_disk_path=sample_filename.string();
            

            m_clouds_buffer.enqueue(cloud);;

        }

    }

}

bool DataLoaderPhenorob::has_data(){
    if(m_clouds_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    }
}


std::shared_ptr<Mesh> DataLoaderPhenorob::get_cloud(){

    std::shared_ptr<Mesh> cloud;
    m_clouds_buffer.try_dequeue(cloud);

    return cloud;
}

bool DataLoaderPhenorob::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<m_sample_filenames.size()){
        return false; //there is still more files to read
    }

    //check that there is nothing in the ring buffers
    if(m_clouds_buffer.peek()!=nullptr){
        return false; //there is still something in the buffer
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderPhenorob::is_finished_reading(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<m_sample_filenames.size()){
        return false; //there is still more files to read
    }

    return true; //there is nothing more to read and so we are finished reading

}

void DataLoaderPhenorob::reset(){
    m_nr_resets++;
    // we shuffle again the data so as to have freshly shuffled data for the next epoch
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // auto rng = std::default_random_engine(seed);
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(m_sample_filenames), std::end(m_sample_filenames), rng);
    }

    m_idx_cloud_to_read=0;
}

int DataLoaderPhenorob::nr_samples(){
    return m_sample_filenames.size();
}
std::shared_ptr<LabelMngr> DataLoaderPhenorob::label_mngr(){
    CHECK(m_label_mngr) << "label_mngr was not created";
    return m_label_mngr;
}
// void DataLoaderPhenorob::set_mode_train(){
//     m_mode="train";
// }
// void DataLoaderPhenorob::set_mode_test(){
//     m_mode="test";
// }
// void DataLoaderPhenorob::set_mode_validation(){
//     m_mode="val";
// }
// void DataLoaderPhenorob::set_sequence(const std::string sequence){
//     m_sequence=sequence;
// }
// void DataLoaderSemanticKitti::set_adaptive_subsampling(const bool adaptive_subsampling){
//     m_do_adaptive_subsampling=adaptive_subsampling;
// }

void DataLoaderPhenorob::set_nr_clouds_to_skip(const int new_val){
    m_nr_clouds_to_skip=new_val;
}
void DataLoaderPhenorob::set_nr_clouds_to_read(const int new_val){
    m_nr_clouds_to_read=new_val;
}





// PYBIND11_MODULE(DataLoader, m) {
//     // py::class_<DataLoaderSemanticKitti> loader(m, "DataLoaderSemanticKitti", R"EOS(
//     //         An axis-aligned 3D range (bounding box).
//     //     )EOS"
//     // )
//     // // m.doc() = "pybind11 example plugin"; // optional module docstring

//     // // m.def("add", &add, "A function which adds two numbers");
//     // .def("pybind_example_add", &DataLoaderSemanticKitti::pybind_example_add, R"EOS(
//     //             Read class index map.

//     //             If CUDA support is active, the tensor will reside on the GPU
//     //             which was used during rendering. The background index is 0.

//     //             Returns:
//     //                 tensor: (H x W) short tensor with instance values.
//     //         )EOS"
//     //     )
//     // ;


//     // Basic geometric types
//     py::class_<DataLoaderSemanticKitti>(m, "Range3D", R"EOS(
//             An axis-aligned 3D range (bounding box).
//         )EOS")

//         // .def_property("min",
//         //     [](const Magnum::Range3D& range){ return toTorch<Magnum::Vector3>::convert(range.min()); },
//         //     [](Magnum::Range3D& range, at::Tensor min){ range.min() = fromTorch<Magnum::Vector3>::convert(min); }
//         // )
//         // .def_property("max",
//         //     [](const Magnum::Range3D& range){ return toTorch<Magnum::Vector3>::convert(range.max()); },
//         //     [](Magnum::Range3D& range, at::Tensor max){ range.max() = fromTorch<Magnum::Vector3>::convert(max); }
//         // )
//         // .def_property_readonly("center",
//         //     [](const Magnum::Range3D& range){ return toTorch<Magnum::Vector3>::convert(range.center()); }
//         // )
//         // .def_property_readonly("size",
//         //     [](const Magnum::Range3D& range){ return toTorch<Magnum::Vector3>::convert(range.size()); }
//         // )
//         // .def_property_readonly("diagonal",
//         //     [](const Magnum::Range3D& range){ return range.size().length(); }
//         // )
//         // .def("__repr__", [](const Magnum::Range3D& range){
//         //     using Corrade::Utility::Debug;
//         //     std::ostringstream ss;
//         //     Debug{&ss, Debug::Flag::NoNewlineAtTheEnd}
//         //         << "Range3D(" << range.min() << "," << range.max() << ")";
//         //     return ss.str();
//         // })

//         //     .def("pybind_example_add", &DataLoaderSemanticKitti::pybind_example_add, R"EOS(
//         //         Read class index map.

//         //         If CUDA support is active, the tensor will reside on the GPU
//         //         which was used during rendering. The background index is 0.

//         //         Returns:
//         //             tensor: (H x W) short tensor with instance values.
//         //     )EOS"
//         // )


//     ;

// }