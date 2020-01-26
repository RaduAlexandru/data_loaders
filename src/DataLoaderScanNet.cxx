#include "data_loaders/DataLoaderScanNet.h"

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

// #include "igl/ply.h"
#include "tinyply.h"

//boost
#include <boost/range/iterator_range.hpp>

//my stuff 
#include "data_loaders/DataTransformer.h"
#include "easy_pbr/Mesh.h"
#include "Profiler.h"
#include "string_utils.h"
#include "eigen_utils.h"
#include "RandGenerator.h"
#include "easy_pbr/LabelMngr.h"

using namespace easy_pbr::utils;


DataLoaderScanNet::DataLoaderScanNet(const std::string config_file):
    m_is_modified(false),
    m_is_running(false),
    m_clouds_buffer(BUFFER_SIZE),
    m_idx_cloud_to_read(0),
    m_nr_resets(0),
    m_rand_gen(new RandGenerator),
    m_min_label_written(999999),
    m_max_label_written(-999999)
{

    init_params(config_file);
    // read_pose_file();
    create_transformation_matrices();
    // std::cout << " creating thread" << "\n";
    if(m_autostart){
        m_is_running=true;
        m_loader_thread=std::thread(&DataLoaderScanNet::read_data, this);  //starts the spin in another thread
    }
    // std::cout << " finidhed creating thread" << "\n";

}

DataLoaderScanNet::~DataLoaderScanNet(){

    // std::cout << "finishing" << std::endl;
    m_is_running=false;

    m_loader_thread.join();
}

void DataLoaderScanNet::init_params(const std::string config_file){
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
    Config loader_config=cfg["loader_scannet"];

    m_autostart=loader_config["autostart"];
    m_mode=(std::string)loader_config["mode"];
    m_nr_clouds_to_skip=loader_config["nr_clouds_to_skip"];
    m_nr_clouds_to_read=loader_config["nr_clouds_to_read"];
    m_max_nr_points_per_cloud=loader_config["max_nr_points_per_cloud"];
    m_shuffle_points=loader_config["shuffle_points"];
    m_shuffle=loader_config["shuffle"];
    m_do_overfit=loader_config["do_overfit"];
    // m_do_adaptive_subsampling=loader_config["do_adaptive_subsampling"];
    m_dataset_path=(std::string)loader_config["dataset_path"];

    //label file and colormap
    Config mngr_config=loader_config["label_mngr"];
    m_label_mngr=std::make_shared<LabelMngr>(mngr_config);
    m_label_mngr->compact("UNUSED");

    //data transformer
    Config transformer_config=loader_config["transformer"];
    m_transformer=std::make_shared<DataTransformer>(transformer_config);


}

void DataLoaderScanNet::start(){
    CHECK(m_is_running==false) << "The loader thread is already running. Please check in the config file that autostart is not already set to true. Or just don't call start()";
    m_is_running=true;
    m_loader_thread=std::thread(&DataLoaderScanNet::read_data, this);  //starts the spin in another thread
}

void DataLoaderScanNet::init_data_reading(){

    //Read the files for the train/test/val splits
    // m_files_train=read_data_split(  (m_dataset_path/"data_splits/scannetv2_train.txt").string() );
    // m_files_test=read_data_split(  (m_dataset_path/"data_splits/scannetv2_test.txt").string() );
    // m_files_validation=read_data_split(  (m_dataset_path/"data_splits/scannetv2_val.txt").string() );



    std::vector<fs::path> ply_filenames_all;
    fs::path full_path= m_dataset_path/"data"/m_mode;

    if(!fs::is_directory(full_path)) {
        LOG(FATAL) << "No directory " << full_path;
    }


    //each room is stored in a different file 
    for(auto& room_dir : boost::make_iterator_range(boost::filesystem::directory_iterator(full_path), {})){
        if(fs::is_directory(room_dir)){


            //ROOM which contains a ply file with the cloud and the rgb colors adn a labels.ply which has a property called label which contains the 40 class of nyu40
            fs::path room_path=room_dir.path();
            // VLOG(1) << "room path is " << room_path;
            fs::path cloud_file=room_path/ (room_path.stem().string()+"_vh_clean_2.ply");

            ply_filenames_all.push_back(cloud_file);

        }
    }



    if(!m_shuffle){ //if we are shuffling, there is no need to sort them
        std::sort(ply_filenames_all.begin(), ply_filenames_all.end());
    }
  



    //shuffle the filles to be read if necessary
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(ply_filenames_all), std::end(ply_filenames_all), rng);
    }


    //ADDS THE clouds to the member std_vector of paths 
    //read a maximum nr of images HAVE TO DO IT HERE BECAUSE WE HAVE TO SORT THEM FIRST
    for (size_t i = 0; i < ply_filenames_all.size(); i++) {
        if( (int)i>=m_nr_clouds_to_skip && ((int)m_ply_filenames.size()<m_nr_clouds_to_read || m_nr_clouds_to_read<0 ) ){
            m_ply_filenames.push_back(ply_filenames_all[i]);
        }
    }

    std::cout << "About to read " << m_ply_filenames.size() << " clouds" <<std::endl; 


    CHECK(m_ply_filenames.size()>0) <<"We did not find any ply files to read";



   

}

void DataLoaderScanNet::read_data(){

    loguru::set_thread_name("loader_thread_scannet");

    init_data_reading();

    while (m_is_running ) {

        //we finished reading so we wait here for a reset
        if(m_idx_cloud_to_read>=m_ply_filenames.size()){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            continue;
        }

        // std::cout << "size approx is " << m_queue.size_approx() << '\n';
        // std::cout << "m_idx_img_to_read is " << m_idx_img_to_read << '\n';
        if(m_clouds_buffer.size_approx()<BUFFER_SIZE-1){ //there is enough space
            //read the frame and everything else and push it to the queue
            // LOG(WARNING) << "At the moment this loader doesnt care about train and test and just reads all clouds in the dataset. NEeds ot be fixed";

            fs::path ply_filename=m_ply_filenames[ m_idx_cloud_to_read ];
            if(!m_do_overfit){
                m_idx_cloud_to_read++;
            }
            // VLOG(1) << "reading " << ply_filename;
            // VLOG(1) << "nr of classes is " << m_label_mngr->nr_classes();


            MeshSharedPtr cloud=Mesh::create();

            //put the name of the scene (eg: scene0707_00) as the name of the mesh. This will help with writing the predictions afterwards
            cloud->name=fs::absolute(ply_filename).parent_path().filename().string();

            //read xyz positions
            cloud->load_from_file(ply_filename.string());
            // cloud->C.array()/=255.0;
            cloud->D=cloud->V.rowwise().norm();
            cloud->recalculate_normals();
            cloud->I.resize(cloud->V.rows(),1);
            for(int i=0; i<cloud->V.rows(); i++){
                cloud->I(i) = 0.3*cloud->C(i,0) + 0.59*cloud->C(i,1) + 0.11*cloud->C(i,2);
            }

            if(m_mode!="test"){
                // read labels 
                fs::path labels_file=fs::absolute(ply_filename).parent_path()/ (ply_filename.stem().string()+".labels.ply");
                // VLOG(1)<< "Reading labels from " << labels_file;
                cloud->L_gt=read_labels(labels_file.string());

                //the labels indices have to be reindexed because the label_manager compacted the labels so that their indices are conscutive
                m_label_mngr->reindex_into_compacted_labels(cloud->L_gt);

                // CHECK(cloud.L_gt.maxCoeff()<m_label_mngr->nr_classes()) << "We have read a cloud which have a label idx higher than the nr of classes. The max label is " << cloud.L_gt.maxCoeff() << " and the nr of classes is m_label_mngr->nr_classes()";
                //some clouds are messed up and have a label idx higher that the nr of classes. Set those vertices to unlabeled
                if(cloud->L_gt.maxCoeff()>=m_label_mngr->nr_classes()){
                    int nr_wrong_labels=0;
                    for(int i=0; i<cloud->L_gt.rows(); i++){
                        if(cloud->L_gt(i)>=m_label_mngr->nr_classes()){
                            cloud->L_gt(i)=m_label_mngr->get_idx_unlabeled();
                            nr_wrong_labels++;
                        }
                    }
                    LOG(ERROR) << "Found a cloud with higher label idx than nr_classes. We have set " << nr_wrong_labels << " vertices to unlabeled";
                }
            }




     
            



            fs::path alignment_file=fs::absolute(ply_filename).parent_path()/ (fs::absolute(ply_filename).parent_path().filename().string()+".txt");
            // VLOG(1) << "reading alignment file from " << alignment_file;
            Eigen::Affine3d alignment;
            alignment=read_alignment_matrix(alignment_file.string());

            //the scannet dataset is gigantic and sometimes we can't process all points, we establish a maximum amount of points we can process and drop the rest
            int nr_points=cloud->V.rows();
            if (nr_points>m_max_nr_points_per_cloud && m_max_nr_points_per_cloud>0){
                LOG(WARNING)<< "Overstepping theshold of max nr of points of " << m_max_nr_points_per_cloud << " because we have nr of points " << nr_points << ". Dropping points until we only are left with the maximum we can process." ; 
                //percentage of points we have to drop 
                float percentage_to_drop=1.0-(float)m_max_nr_points_per_cloud/(float)nr_points;
                float prob_of_death=percentage_to_drop;
                std::vector<bool> is_vertex_to_be_removed(cloud->V.rows(), false);
                for(int i = 0; i < cloud->V.rows(); i++){
                    float random= m_rand_gen->rand_float(0.0, 1.0);
                    if(random<prob_of_death){
                        is_vertex_to_be_removed[i]=true;
                    }
                }
                cloud->remove_marked_vertices(is_vertex_to_be_removed, false);
            }




            cloud->transform_vertices_cpu(alignment);
            cloud->transform_vertices_cpu(m_tf_worldGL_worldROS); // from worldROS to worldGL


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
                cloud->C = perm * cloud->C; // permute rows
            }

            //some sensible visualization options
            cloud->m_vis.m_show_mesh=false;
            cloud->m_vis.m_show_points=true;
            cloud->m_vis.m_color_type=+MeshColorType::SemanticGT;

            //set the labelmngr which will be used by the viewer to put correct colors for the semantics
            cloud->m_label_mngr=m_label_mngr->shared_from_this();
            

            m_clouds_buffer.enqueue(cloud);;

        }

    }

}

//read labels (took some inspiration from the file readPLY.cpp in libigl library)
Eigen::MatrixXi DataLoaderScanNet::read_labels(const std::string labels_file){

    //open file
    std::ifstream ss(labels_file, std::ios::binary);
    CHECK(ss.is_open()) << "Failed to open " << labels_file;
    tinyply::PlyFile file;
    file.parse_header(ss);


    // Tinyply treats parsed data as untyped byte buffers. See below for examples.
    std::shared_ptr<tinyply::PlyData> labels;

    // The header information can be used to programmatically extract properties on elements
    // known to exist in the header prior to reading the data. For brevity of this sample, properties 
    // like vertex position are hard-coded: 
    try { labels = file.request_properties_from_element("vertex", { "label" }, 1); }
    catch (const std::exception & e) { LOG(FATAL) <<  e.what();  }


    file.read(ss);

    // type casting to your own native types - Option B
    typedef Eigen::Matrix<unsigned short,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RowMatrixXus;

    //parse data
    //labels
    Eigen::MatrixXi L_gt;
    if (labels->t == tinyply::Type::UINT16) {
        Eigen::Map<RowMatrixXus> mf( (unsigned short*)labels->buffer.get(), labels->count, 1);
        L_gt=mf.cast<int>();
    }else{ LOG(FATAL) <<" labels should be unsigned short. "; }
   

    return L_gt;

}

Eigen::Affine3d DataLoaderScanNet::read_alignment_matrix(const std::string alignment_file){
    std::ifstream infile(alignment_file);
    CHECK(infile.is_open())<<"Could not open file" <<alignment_file;
    std::string line;
    Eigen::Affine3d mat;
    mat.setIdentity();
    while( std::getline(infile, line)){
        std::vector<std::string> tokens=split(line, " ");

        if(tokens[0]=="axisAlignment"){
            //Put the values into an eigen matrix
            mat.matrix()(0,0)=stod(tokens[2]);
            mat.matrix()(0,1)=stod(tokens[3]);
            mat.matrix()(0,2)=stod(tokens[4]);
            mat.matrix()(0,3)=stod(tokens[5]);

            mat.matrix()(1,0)=stod(tokens[6]);
            mat.matrix()(1,1)=stod(tokens[7]);
            mat.matrix()(1,2)=stod(tokens[8]);
            mat.matrix()(1,3)=stod(tokens[9]);

            mat.matrix()(2,0)=stod(tokens[10]);
            mat.matrix()(2,1)=stod(tokens[11]);
            mat.matrix()(2,2)=stod(tokens[12]);
            mat.matrix()(2,3)=stod(tokens[13]);

            mat.matrix()(3,0)=stod(tokens[14]);
            mat.matrix()(3,1)=stod(tokens[15]);
            mat.matrix()(3,2)=stod(tokens[16]);
            mat.matrix()(3,3)=stod(tokens[17]);
        }
    }


    // VLOG(1) << "Read matrix " << mat.matrix();

    return mat;
}
// std::unordered_map<std::string, bool>  DataLoaderScanNet::read_data_split(const std::string data_split_file){
//     std::ifstream infile(data_split_file);
//     CHECK(infile.is_open())<<"Could not open file" <<data_split_file;

//     std::string line;
//     std::unordered_map<std::string, bool> map;
//     while( std::getline(infile, line)){
//         map[line]=true;
//     }

//     return map;
// }

//the test set need to be evaluated on the their server so we write it in the format they want
void DataLoaderScanNet::write_for_evaluating_on_scannet_server( std::shared_ptr<Mesh>& cloud, const std::string path_for_eval){
    //the predictions for each vertex need to be decompacted

    std::ofstream pred_file;
    std::string full_path= (fs::path(path_for_eval)/ fs::path(cloud->name+".txt")).string() ;
    VLOG(1) << "writing prediction to full_path" << full_path;
    pred_file.open (full_path);
    CHECK(pred_file.is_open()) << "Could not open for writing into file " << full_path;

    //sanity checking
    VLOG(1) << "max class compacted is " << cloud->L_pred.maxCoeff();
    VLOG(1) << "min class compacted is " << cloud->L_pred.minCoeff();

    Eigen::MatrixXi L_compacted=cloud->L_pred; //we save the compacted one for future reference
    cloud->m_label_mngr->reindex_into_uncompacted_labels(cloud->L_pred);

    for(int i=0; i<cloud->V.rows(); i++){
        int label_uncompacted=cloud->L_pred(i,0);
        pred_file << label_uncompacted << "\n";

        // for sanity checking, we check that we write only labels between 1 and 39 inclusive
        if (label_uncompacted<1 || label_uncompacted>39){
            LOG(FATAL) << "Something went wrong, the scannet labels should only be between 1 and 39 inclusive, however we are trying to write " << label_uncompacted << "which was originated by a a compacted label of " << L_compacted(i,0); 
        }

        if(label_uncompacted>m_max_label_written){
            m_max_label_written=label_uncompacted;
        }
        if(label_uncompacted<m_min_label_written){
            m_min_label_written=label_uncompacted;
        }
    }

    VLOG(1) << "the min and the max ever written should be 1 and 39, It is: " << m_min_label_written << " " << m_max_label_written;

    //resestablish back the compacted predictions
    cloud->L_pred=L_compacted;

    pred_file.close();

}


bool DataLoaderScanNet::has_data(){
    if(m_clouds_buffer.peek()==nullptr){
        return false;
    }else{
        return true;
    }
}


std::shared_ptr<Mesh> DataLoaderScanNet::get_cloud(){

    std::shared_ptr<Mesh> cloud;
    m_clouds_buffer.try_dequeue(cloud);

    return cloud;
}

bool DataLoaderScanNet::is_finished(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<(int)m_ply_filenames.size()){
        return false; //there is still more files to read
    }

    //check that there is nothing in the ring buffers
    if(m_clouds_buffer.peek()!=nullptr){
        return false; //there is still something in the buffer
    }

    return true; //there is nothing more to read and nothing more in the buffer so we are finished

}


bool DataLoaderScanNet::is_finished_reading(){
    //check if this loader has loaded everything
    if(m_idx_cloud_to_read<(int)m_ply_filenames.size()){
        return false; //there is still more files to read
    }

    return true; //there is nothing more to read and so we are finished reading

}

void DataLoaderScanNet::reset(){
    m_nr_resets++;
    // we shuffle again the data so as to have freshly shuffled data for the next epoch
    if(m_shuffle){
        // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        // auto rng = std::default_random_engine(seed);
        unsigned seed = m_nr_resets;
        auto rng = std::default_random_engine(seed);
        std::shuffle(std::begin(m_ply_filenames), std::end(m_ply_filenames), rng);
    }

    m_idx_cloud_to_read=0;
}

int DataLoaderScanNet::nr_samples(){
    return m_ply_filenames.size();
}

std::shared_ptr<LabelMngr> DataLoaderScanNet::label_mngr(){
    CHECK(m_label_mngr) << "label_mngr was not created";
    return m_label_mngr;
}


void DataLoaderScanNet::set_mode_train(){
    m_mode="train";
}
void DataLoaderScanNet::set_mode_test(){
    m_mode="test";
}
void DataLoaderScanNet::set_mode_validation(){
    m_mode="val";
}


void DataLoaderScanNet::create_transformation_matrices(){

    m_tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3d worldGL_worldROS_rot;
    float angle=-0.5;
    worldGL_worldROS_rot = Eigen::AngleAxisd(angle*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
}


