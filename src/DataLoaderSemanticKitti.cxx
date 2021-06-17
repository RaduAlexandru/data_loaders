#include "data_loaders/DataLoaderSemanticKitti.h"

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

DataLoaderSemanticKitti::DataLoaderSemanticKitti(const std::string config_file):
    m_is_modified(false),
    m_is_running(false),
    m_clouds_buffer(BUFFER_SIZE),
    m_idx_cloud_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator)
{

    init_params(config_file);
    // read_pose_file();
    create_transformation_matrices();
    // std::cout << " creating thread" << "\n";
    if(m_autostart){
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderSemanticKitti::read_data, this);  //starts the spin in another thread
    }
    // std::cout << " finidhed creating thread" << "\n";

}

DataLoaderSemanticKitti::~DataLoaderSemanticKitti(){

    // std::cout << "finishing" << std::endl;
    m_is_running=false;

    m_loader_thread.join();
}

void DataLoaderSemanticKitti::init_params(const std::string config_file){
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
    Config loader_config=cfg["loader_semantic_kitti"];

    m_autostart=loader_config["autostart"];
    m_mode=(std::string)loader_config["mode"];
    m_nr_clouds_to_skip=loader_config["nr_clouds_to_skip"];
    m_nr_clouds_to_read=loader_config["nr_clouds_to_read"];
    m_cap_distance=loader_config["cap_distance"];
    m_shuffle_points=loader_config["shuffle_points"];
    m_do_pose=loader_config["do_pose"];
    m_normalize=loader_config["normalize"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    // m_do_adaptive_subsampling=loader_config["do_adaptive_subsampling"];
    m_dataset_path=(std::string)loader_config["dataset_path"];
    m_sequence=(std::string)loader_config["sequence"];

    //label file and colormap
    Config mngr_config=loader_config["label_mngr"];
    m_label_mngr=std::make_shared<LabelMngr>(mngr_config);

    //data transformer
    Config transformer_config=loader_config["transformer"];
    m_transformer=std::make_shared<DataTransformer>(transformer_config);


}

void DataLoaderSemanticKitti::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";

    init_data_reading();

    m_is_running=true;
    m_loader_thread=std::thread(&DataLoaderSemanticKitti::read_data, this);  //starts the spin in another thread
}

void DataLoaderSemanticKitti::init_data_reading(){

    std::vector<fs::path> npz_filenames_all;
    if(m_sequence!="all"){
        m_nr_sequences=1; //we usually get only one sequence, unless m_sequence is set to "all"
        fs::path full_path= m_dataset_path/m_mode/m_sequence;

        if(!fs::is_directory(full_path)) {
            LOG(FATAL) << "No directory " << full_path;
        }

        //see how many images we have and read the files paths into a vector
        for (fs::directory_iterator itr(full_path); itr!=fs::directory_iterator(); ++itr){
            //all the files in the folder might include also the pose file so we ignore that one
            //we also ignore the files that contain intensity, for now we only read the general ones and then afterwards we append _i to the file and read the intensity if neccesarry
            if( !(itr->path().stem()=="poses")  &&  itr->path().stem().string().find("_i")== std::string::npos ){
                npz_filenames_all.push_back(itr->path());
            }
        }
        if(!m_shuffle){ //if we are shuffling, there is no need to sort them
            std::sort(npz_filenames_all.begin(), npz_filenames_all.end());
        }



	if(m_do_pose){
	    //pose file
            std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  > poses;
            fs::path pose_file = full_path/"poses.txt";
            poses=read_pose_file( pose_file.string() );
            m_poses_per_sequence[m_sequence.string()] = poses;
	}

    }else if(m_sequence=="all"){
        //iterate thrugh all the sequences and load all of them

        //get how many sequnces we have here
        fs::path dataset_path_with_mode= m_dataset_path/m_mode;
        if(!fs::is_directory(dataset_path_with_mode)) {
            LOG(FATAL) << "No directory " << dataset_path_with_mode;
        }
        m_nr_sequences=0;
        for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(dataset_path_with_mode), {})){
            if(fs::is_directory(entry)){
                fs::path full_path= entry;
                std::string sequence= full_path.stem().string();
                VLOG(1) << "full path is " << full_path;
                VLOG(1) << "sequence is " << sequence;

                m_nr_sequences++;
                //read the npz of each sequence
                std::vector<fs::path> npz_filenames_for_sequence;
                for (fs::directory_iterator itr(full_path); itr!=fs::directory_iterator(); ++itr){
                    //all the files in the folder might include also the pose file so we ignore that one
                    //we also ignore the files that contain intensity, for now we only read the general ones and then afterwards we append _i to the file and read the intensity if neccesarry
                    if( !(itr->path().stem()=="poses")  && itr->path().stem().string().find("_i")== std::string::npos ){
                        npz_filenames_for_sequence.push_back(itr->path());
                    }
                }
                if(!m_shuffle){ //if we are shuffling, there is no need to sort them
                    std::sort(npz_filenames_for_sequence.begin(), npz_filenames_for_sequence.end());
                }

                npz_filenames_all.insert(npz_filenames_all.end(), npz_filenames_for_sequence.begin(), npz_filenames_for_sequence.end());

            	if(m_do_pose){
              	    //read poses for this sequence
                    std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  > poses;
                    fs::path pose_file = full_path/"poses.txt";
                    poses=read_pose_file( pose_file.string() );
                    m_poses_per_sequence[sequence] = poses;
                }
            }
        }
        VLOG(1) << "m_nr_sequences is " << m_nr_sequences;


    }else{
        LOG(FATAL) << "Sequence is not known" << m_sequence;
    }


    //shuffle the filles to be read if necessary
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(npz_filenames_all), std::end(npz_filenames_all), rng);
    }


    //ADDS THE clouds to the member std_vector of paths
    //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    for (size_t i = 0; i < npz_filenames_all.size(); i++) {
        if( (int)i>=m_nr_clouds_to_skip && ((int)m_npz_filenames.size()<m_nr_clouds_to_read || m_nr_clouds_to_read<0 ) ){
            m_npz_filenames.push_back(npz_filenames_all[i]);
        }
    }

    std::cout << "About to read " << m_npz_filenames.size() << " clouds" <<std::endl;


    CHECK(m_npz_filenames.size()>0) <<"We did not find any npz files to read";

}

void DataLoaderSemanticKitti::read_data(){

    loguru::set_thread_name("loader_thread_kitti");


    while (m_is_running ) {

        //we finished reading so we wait here for a reset
        if(m_idx_cloud_to_read>=m_npz_filenames.size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue

            fs::path npz_filename=m_npz_filenames[ m_idx_cloud_to_read ];
            if(!m_do_overfit){
                m_idx_cloud_to_read++;
            }
            // VLOG(1) << "reading " << npz_filename;

            //read npz
            cnpy::npz_t npz_file = cnpy::npz_load(npz_filename.string());
            cnpy::NpyArray arr = npz_file["arr_0"]; //one can obtain the keys with https://stackoverflow.com/a/53901903
            CHECK(arr.shape.size()==2) << "arr should have 2 dimensions and it has " << arr.shape.size();
            CHECK(arr.shape[1]==4) << "arr second dimension should be 4 (x,y,z,label) but it is " << arr.shape[1];

            //read intensity
            fs::path absolute_path=fs::absolute(npz_filename).parent_path();
            fs::path file_name=npz_filename.stem();
            // fs::path npz_intensity_path=absolute_path/(file_name.string()+"_i"+".npz");
            // cnpy::npz_t npz_intensity_file = cnpy::npz_load(npz_intensity_path.string());
            // cnpy::NpyArray arr_intensity = npz_intensity_file["arr_0"]; //one can obtain the keys with https://stackoverflow.com/a/53901903
            // CHECK(arr_intensity.shape.size()==1) << "arr should have 1 dimensions and it has " << arr.shape.size();


            //copy into EigenMatrix
            int nr_points=arr.shape[0];
            MeshSharedPtr cloud=Mesh::create();
            cloud->V.resize(nr_points,3);
            cloud->V.setZero();
            cloud->L_gt.resize(nr_points,1);
            cloud->L_gt.setZero();
            // cloud->I.resize(nr_points,1);
            // cloud->I.setZero();
            double* arr_data = arr.data<double>();
            // float* arr_intensity_data = arr_intensity.data<float>(); //the intensities are as floats while xyz is double. You can check by reading the npz in python
            for(int i=0; i<nr_points*4; i=i+4){
                int row_insert=i/4;

                double x=arr_data[i];
                double y=arr_data[i+1];
                double z=arr_data[i+2];
                int label=arr_data[i+3];
                // double intensity=arr_intensity_data[row_insert];

                cloud->V.row(row_insert) << x,y,z;
                cloud->L_gt.row(row_insert) << label;
                // cloud->I.row(row_insert) << intensity;


                // VLOG(1) << "xyz is " << x << " " << y << " " << z << " " << label;
                // exit(1);

            }
            cloud->D=cloud->V.rowwise().norm();

            // if(m_do_adaptive_subsampling){
            //     std::vector<bool> marked_to_be_removed(cloud.V.rows(), false);
            //     for(int i=0; i<cloud.V.rows(); i++){
            //         float dist=cloud.V.row(i).norm();
            //         float prob_to_remove= map(dist, 0.0, 60.0, 1.0, 0.0 ); //the closer verts have a high prob to be removed and the further away ones have one that is close to 0
            //         float r_val = m_rand_gen->rand_float(0.0, 1.0);
            //         if(r_val < prob_to_remove) { //the r_val will have no chance in going very low so it will not remove the points with prob_to_remove close to 0.0
            //             marked_to_be_removed[i]=true;
            //         }
            //     }
            //     cloud.remove_marked_vertices(marked_to_be_removed, false);
            // }

            //get pose
            int scan_nr=std::stoull( npz_filename.stem().string() ); //scan_nr corresponds to the file name (without the extension of course)
            std::string sequence= npz_filename.parent_path().stem().string();
            // VLOG(1) << "sequence is " << sequence;
            Eigen::Affine3d tf_worldROS_cam;
            if(m_do_pose){
                    tf_worldROS_cam=get_pose_for_scan_nr_and_sequence(scan_nr, sequence);
            }
            cloud->t=scan_nr;


            if(m_cap_distance>0.0){
                std::vector<bool> is_too_far(cloud->V.rows(),false);
                for(int i=0; i<cloud->V.rows(); i++){
                    float dist=cloud->V.row(i).norm();
                    if(dist>m_cap_distance){
                        is_too_far[i]=true;
                    }
                }
                cloud->remove_marked_vertices(is_too_far, false);
            }

            //transform
            if(m_do_pose){
                LOG(FATAL) << "Doing poses is at the moment disabled because the poses are wrong. I thought the matrix m_tf_cam_velodyne is the same for all sequences, however that is not the case.";
                cloud->transform_vertices_cpu(m_tf_cam_velodyne); //from velodyne frame to the camera frame
                cloud->transform_vertices_cpu(tf_worldROS_cam); // from camera to worldROS
            }
            cloud->transform_vertices_cpu(m_tf_worldGL_worldROS); // from worldROS to worldGL


            if(m_mode=="train"){
                cloud=m_transformer->transform(cloud);
            }


            if(m_normalize){
                cloud->normalize_size();
                cloud->normalize_position();
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
            cloud->m_label_mngr=m_label_mngr->shared_from_this();

            cloud->m_disk_path=npz_filename.string();


            m_clouds_buffer.enqueue(cloud);;

        }

    }

}

bool DataLoaderSemanticKitti::has_data(){
    if(m_clouds_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    }
}


std::shared_ptr<Mesh> DataLoaderSemanticKitti::get_cloud(){

    std::shared_ptr<Mesh> cloud;
    m_clouds_buffer.try_dequeue(cloud);

    return cloud;
}

bool DataLoaderSemanticKitti::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<m_npz_filenames.size()){
        return false; //there is still more files to read
    }

    //check that there is nothing in the ring buffers
    if(m_clouds_buffer.peek()!=nullptr){
        return false; //there is still something in the buffer
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderSemanticKitti::is_finished_reading(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<m_npz_filenames.size()){
        return false; //there is still more files to read
    }

    return true; //there is nothing more to read and so we are finished reading

}

void DataLoaderSemanticKitti::reset(){
    m_nr_resets++;
    // we shuffle again the data so as to have freshly shuffled data for the next epoch
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // auto rng = std::default_random_engine(seed);
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(m_npz_filenames), std::end(m_npz_filenames), rng);
    }

    m_idx_cloud_to_read=0;
}

int DataLoaderSemanticKitti::nr_samples(){
    return m_npz_filenames.size();
}
std::shared_ptr<LabelMngr> DataLoaderSemanticKitti::label_mngr(){
    CHECK(m_label_mngr) << "label_mngr was not created";
    return m_label_mngr;
}
void DataLoaderSemanticKitti::set_mode_train(){
    m_mode="train";
}
void DataLoaderSemanticKitti::set_mode_test(){
    m_mode="test";
}
void DataLoaderSemanticKitti::set_mode_validation(){
    m_mode="val";
}
void DataLoaderSemanticKitti::set_sequence(const std::string sequence){
    m_sequence=sequence;
}
// void DataLoaderSemanticKitti::set_adaptive_subsampling(const bool adaptive_subsampling){
//     m_do_adaptive_subsampling=adaptive_subsampling;
// }


std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  > DataLoaderSemanticKitti::read_pose_file(const std::string m_pose_file){
    std::ifstream infile( m_pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
    std::vector<Eigen::Affine3d,  Eigen::aligned_allocator<Eigen::Affine3d>  > poses;
    Eigen::Vector3d position;
    double r00, r01, r02,   r10, r11, r12,   r20, r21, r22; //elements of the rotation matrix in row major order


    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >>  r00>> r01>> r02>> position.x()
            >>  r10>> r11>> r12>> position.y()
            >>  r20>> r21>> r22>> position.z();


        Eigen::Affine3d pose;
        pose.matrix().block<3,3>(0,0) << r00, r01, r02,   r10, r11, r12,   r20, r21, r22;;
        pose.matrix().block<3,1>(0,3)=position;

        poses.push_back ( pose );
    }

    return poses;

}

Eigen::Affine3d DataLoaderSemanticKitti::get_pose_for_scan_nr_and_sequence(const int scan_nr, const std::string sequence){
    CHECK(scan_nr<(int)m_poses_per_sequence[sequence].size()) << "scan_nr out of range. Maximum pose would be for scan_nr " << m_poses_per_sequence[sequence].size() << " and you are trying to index at " <<scan_nr;

    return m_poses_per_sequence[sequence][scan_nr];
}

void DataLoaderSemanticKitti::create_transformation_matrices(){
    m_tf_cam_velodyne.linear()<<7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02;
    m_tf_cam_velodyne.translation()<< -4.069766e-03, -7.631618e-02, -2.717806e-01;

    m_tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3d worldGL_worldROS_rot;
    float angle=0.0;
    if (m_do_pose){
        angle=1.0;
    }else{
        angle=-0.5;
    }
    worldGL_worldROS_rot = Eigen::AngleAxisd(angle*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
}

int DataLoaderSemanticKitti::add(const int a, const int b ){
    return a+b;
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
