#include "data_loaders/DataLoaderCloudRos.h"

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//ros
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/PCLPointCloud2.h>

//my stuff 
#include "ros_utils.h"
#include "easy_pbr/Mesh.h"
#include "easy_pbr/LabelMngr.h"
#include "data_loaders/DataTransformer.h"
#include "Profiler.h"
#include "string_utils.h"
#include "eigen_utils.h"
#include "RandGenerator.h"

//boost
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

using namespace radu::utils;
using namespace easy_pbr;

#define NUM_CLOUDS_BUFFER 5

DataLoaderCloudRos::DataLoaderCloudRos(const std::string config_file):
    m_is_modified(false),
    m_clouds_buffer(NUM_CLOUDS_BUFFER),
    m_finished_cloud_idx(-1),
    m_working_cloud_idx(0),
    m_min_dist_filter(0.0),
    m_exact_time(true),
    m_rand_gen(new RandGenerator() )
{
    init_params(config_file);
    if(m_do_pose){
        read_pose_file(); 
        // if(m_hacky_fix_for_razlaws_ma_bags){
        //     read_pose_file_vel2lasermap(); //this reads the transform tf_lasermap_vel/ From this we will set the pose.translation to lasermap_vel
        // }
    }
    create_transformation_matrices();
    std::cout << " creating thread" << "\n";
    m_loader_thread=std::thread(&DataLoaderCloudRos::init_ros, this);  //starts the spin in another thread
    m_is_thread_running=true;
    std::cout << " finidhed creating thread" << "\n";

}

DataLoaderCloudRos::~DataLoaderCloudRos(){
    ros::shutdown();

    m_loader_thread.join();
}

void DataLoaderCloudRos::init_params(const std::string config_file){
    //get the config filename
    // ros::NodeHandle private_nh("~");
    // std::string config_file= getParamElseThrow<std::string>(private_nh, "config_file");
    // std::string config_file="config.cfg";

    //read all the parameters
    //read all the parameters
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }
    Config cfg = configuru::parse_file(config_file_abs, CFG);

    Config loader_config=cfg["loader_cloud_ros"];
    m_cloud_topic = (std::string)loader_config["cloud_topic"];
    m_do_pose = loader_config["do_pose"];
    // m_do_random_gap_removal = loader_config["do_random_gap_removal"];
    m_pose_file = (std::string)loader_config["pose_file"];
    m_pose_file_format = (std::string)loader_config["pose_file_format"];
    // m_timestamp_multiplier = loader_config["timestamp_multiplier"];
    // m_exact_time = loader_config["exact_time"];
    m_min_dist_filter = loader_config["min_dist_filter"]; 
    // m_hacky_fix_for_razlaws_ma_bags = loader_config["hacky_fix_for_razlaws_ma_bags"]; 


    //data transformer
    Config transformer_config=loader_config["transformer"];
    m_transformer=std::make_shared<DataTransformer>(transformer_config);

    // std::cout << "-------------" << m_timestamp_multiplier << std::endl;
}

void DataLoaderCloudRos::init_ros(){
    std::vector<std::pair<std::string, std::string> > dummy_remappings;
    ros::init(dummy_remappings, "loader_cloud_ros");

    ros::NodeHandle private_nh("~");
    ros::Subscriber sub = private_nh.subscribe(m_cloud_topic, 5, &DataLoaderCloudRos::callback, this);

    // ros::spin();

    //so that we can have both the ImgRos and the cloud rus running we need a asyncspinner
    ros::AsyncSpinner spinner(1); // Use x threads
    spinner.start();
    ros::waitForShutdown();


    LOG(INFO) << "finished ros communication";
    std::cout << "inside thread" << "\n";
    m_is_thread_running=false;
}

void DataLoaderCloudRos::callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {

    // std::cout << "callbck" << "\n";

    //get the laser data into a point cloud
    pcl::PCLPointCloud2::Ptr temp_cloud(new pcl::PCLPointCloud2());
    pcl_conversions::toPCL(*cloud_msg, *temp_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(*temp_cloud, *cloud);


    // MeshCore mesh_core;
    MeshSharedPtr mesh=Mesh::create();
    mesh->V=pcl2eigen(cloud);
    mesh->m_width=cloud->width;
    mesh->m_height=cloud->height;
    // VLOG(1) << "width and height is " << mesh->m_width << " " << mesh->m_height;

    //the distance to the sensor is just the norm of every point as the cloud starts in velodyne frame 
    mesh->D=mesh->V.rowwise().norm();

    double timestamp=(double)cloud_msg->header.stamp.toNSec();


    //remove all points that have a lower distance to the densor than the min_dist_filter. IT DOES NOT REMOVE THEM BECAUSE THIS WILL MAKE THE CLOUD UNORGANIZED
    if(m_min_dist_filter>0.0){
        for(int i=0; i<mesh->D.rows(); i++){
            if(mesh->D(i)<m_min_dist_filter){
                mesh->V(i,0)=0.0;
                mesh->V(i,1)=0.0;
                mesh->V(i,2)=0.0;
            }    
        }
    }

    //establish a random view_direction
    // mesh->m_view_direction=m_rand_gen->rand_float(0,2*M_PI); 
    

    // if(m_do_random_gap_removal){
    //     //remove point in the gap around the random direction so that the unwrapping to 2D can be done by the Mesher
    //     //roate it so that it looks towards the gap. This will be called algorithm frame
    //     Eigen::Affine3d tf_alg_vel;
    //     tf_alg_vel.setIdentity();
    //     Eigen::Matrix3d alg_vel_rot;
    //     alg_vel_rot = Eigen::AngleAxisd(-0.5*M_PI+mesh_core.m_view_direction, Eigen::Vector3d::UnitY())  //this rotation is done second and rotates around the Y axis of alg frame
    //     * Eigen::AngleAxisd(0.5*M_PI, Eigen::Vector3d::UnitX());   //this rotation is done first. Performed on the X axis of alg frame (after this the y is pointing towards camera, x is right and z is down)
    //     tf_alg_vel.matrix().block<3,3>(0,0)=alg_vel_rot;

    //     mesh_core.apply_transform(tf_alg_vel); //from velodyne frame to the algorithm frame
        
    //     for (size_t i = 0; i < mesh_core.V.rows(); i++) {
    //         if (!mesh_core.V.row(i).isZero()) {
    //             //calculate an angle to it
    //             double theta, phi;
    //             phi = std::atan2(mesh_core.V.row(i).x(), - mesh_core.V.row(i).z()); // atan goes from -pi to pi

    //             //if phi (the angle in the horizontal direction) is is within a certain range of 0 then set the points to nan
    //             float gap_angle=0.1415;
    //             // float gap_angle=0.8415;
    //             if(phi< gap_angle && phi > -gap_angle ){
    //                 mesh_core.V.row(i).setZero();
    //             }

    //         }
    //     }

    //     Eigen::Affine3d tf_vel_alg= tf_alg_vel.inverse();
    //     mesh_core.apply_transform(tf_vel_alg); //from algorithm frame back to velodyne frame
    // }


    mesh=m_transformer->transform(mesh);



    if(m_do_pose){
        //move into a baselink, then into world frame and then into gl frame
        Eigen::Affine3d sensor_pose;  //maps from baselink to world ros
        double deviation_ms=-1;
        if (!get_pose_at_timestamp(sensor_pose,deviation_ms, timestamp)){
            LOG(WARNING) << "Not found any pose at timestamp " << timestamp << " Discarding";
            return;
        }

        // if(m_pose_file_format=="david_old"){
            //the old format had the pose file expressed in velodyne frame, the new one is already in baselink
            // mesh_core.apply_transform(m_tf_baselink_vel); // from velodyne frame to baselink 
        // } 
        mesh->transform_vertices_cpu(sensor_pose); // from baselonk to worldROS
        mesh->transform_vertices_cpu(m_tf_worldGL_worldROS);
    }else{
        mesh->transform_vertices_cpu(m_tf_worldGL_worldROS.inverse()); //for some reason it has to be inverse which is probably because the ouster is mounter upside down...
    }

    // VLOG(1) << "meshcore the m_cur_pose is " << mesh_core.m_cur_pose.matrix();



    // // mesh_core.apply_transform(m_tf_worldGL_worldROS); // from worldROS to worldGL


    // //attemp3 just make the cur pose,be where the velodyne frame should be now
    // if(m_hacky_fix_for_razlaws_ma_bags){
    //     Eigen::Affine3d tf_lasermap_vel;
    //     double deviation_ms;
    //     get_pose_vel2lasermap_at_timestamp(tf_lasermap_vel,deviation_ms, timestamp);
    //     Eigen::Affine3d tf_worldGL_vel=m_tf_worldGL_worldROS*tf_lasermap_vel;
    //     mesh_core.m_cur_pose.translation() = tf_worldGL_vel.translation();
    // }



    // //debug the pose
    // MeshCore pose_debug;
    // pose_debug.V.resize(1,3);
    // pose_debug.V.row(0) = mesh_core.m_cur_pose.translation().cast<double>();
    // pose_debug.m_vis.m_show_points=true;
    // MeshCore mesh_dummy=mesh_core;
    // Scene::show(mesh_dummy, "dummy"); //just showing this first because otherwise the camera scene and centroid fucks up because the next one is only a point
    // Scene::show(pose_debug, "pose_debug_from_loader");


    //some sensible visualization options
    mesh->m_vis.m_show_mesh=false;
    mesh->m_vis.m_show_points=true;


    m_clouds_buffer[m_working_cloud_idx]=mesh;
    m_finished_cloud_idx = m_working_cloud_idx;
    m_working_cloud_idx = (m_working_cloud_idx + 1) % NUM_CLOUDS_BUFFER;
    m_is_modified = true;

}

bool DataLoaderCloudRos::is_loader_thread_alive(){
    // std::cout << " is thread alive is " << m_is_thread_running;
    //HACK asfuck because for some reason pybind cannot read correctly the bool that I am returning
    // if(!m_is_thread_running){
        // delete this;
    // }

    return m_is_thread_running;
}

std::shared_ptr<easy_pbr::Mesh> DataLoaderCloudRos::get_cloud(){
    m_is_modified=false;
    return m_clouds_buffer[m_finished_cloud_idx];
}

// void DataLoaderCloudRos::apply_transform(Eigen::MatrixXd& V, const Eigen::Affine3d& trans){
//     for (size_t i = 0; i < V.rows(); i++) {
//         if(!V.row(i).isZero()){
//             V.row(i)=trans.linear()*V.row(i).transpose() + trans.translation();
//         }
//      }
// }

/*Grabs a point cloud and lays down the points in a row by row manner starting from the top-left. */
Eigen::MatrixXd DataLoaderCloudRos::pcl2eigen(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {

    int num_points = cloud->width*cloud->height;
    // std::cout << " creating nr of points " << num_points << std::endl;

    Eigen::MatrixXd V(num_points,3);  //xyz
    V.setZero();

    for (size_t x_idx = 0; x_idx < cloud->width; x_idx++) {
        for (size_t y_idx = 0; y_idx < cloud->height; y_idx++) {
            unsigned int idx = y_idx + x_idx * cloud->height;
            int insertion_idx=x_idx + y_idx * cloud->width; //inserts the points row by row (and not column by column) and going from left to right is you're looking from the sensor position
            // std::cout << "insertion_idx is  " << insertion_idx << '\n';
            if (!std::isnan(cloud->points[idx].x) && !std::isnan(cloud->points[idx].y) && !std::isnan(cloud->points[idx].z)) {
                //insert the point with a different structure so that we have the points not in a cloud of 1800x16 and starting from the upper right but rather of size 16x1800 and starting from bottom left
                // std::cout << "insertin row at " << insertion_idx  << '\n';

                Eigen::Vector3d pos;
                pos << cloud->points[idx].x, cloud->points[idx].y, cloud->points[idx].z;
                V.row(insertion_idx)(0)=pos(0);
                V.row(insertion_idx)(1)=pos(1);
                V.row(insertion_idx)(2)=pos(2);

            } else {
                //TODO better handling of invalid points
                V.row(insertion_idx).setConstant(0.0);
            }

        }
    }

    return V;

}


void DataLoaderCloudRos::read_pose_file(){
    std::ifstream infile( m_pose_file );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
    uint64_t scan_nr;
    uint64_t timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond quat;


    if(m_pose_file_format=="david_old"){
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            iss >> scan_nr >> timestamp
                >> position(0) >> position(1) >> position(2)
                >> quat.w() >> quat.x() >> quat.y() >> quat.z();

            timestamp*=m_timestamp_multiplier;
        //    std::cout << "input is \n" << scan_nr << " " << timestamp << "  \n Position is" << position << " " << quat.matrix()  << "\n";
            Eigen::Affine3d pose;
            pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
            pose.matrix().block<3,1>(0,3)=position;
            // if(m_do_timestamp_rounding_when_reading_file){
            //     timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
            // }
        // VLOG(2) << "recorded tmestamp is " << timestamp;
    //        VLOG(2) << "recorded scan_nr is " << scan_nr;
            m_worldROS_baselink_vec.push_back ( std::pair<double, Eigen::Affine3d>((double)timestamp,pose) );
        }


    }else if(m_pose_file_format=="david_new"){
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            iss >> timestamp
                >> position.x() >> position.y() >> position.z()
                >> quat.x() >> quat.y() >> quat.z() >> quat.w();

            timestamp*=m_timestamp_multiplier;
        //    std::cout << "input is \n" << " " << timestamp << "  \n Position is" << position << " " << quat.matrix()  << "\n";
            Eigen::Affine3d pose;
            pose.setIdentity();
            pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
            pose.matrix().block<3,1>(0,3)=position;
            // if(m_do_timestamp_rounding_when_reading_file){
            //     timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
            // }
        // VLOG(2) << "recorded tmestamp is " << timestamp;
    //        VLOG(2) << "recorded scan_nr is " << scan_nr;
            // Eigen::Affine3d pose_inv=pose.inverse();

            m_worldROS_baselink_vec.push_back ( std::pair<double, Eigen::Affine3d>((double)timestamp,pose) );
        }
    }else if(m_pose_file_format=="tum"){
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            iss >> timestamp
                >> position.x() >> position.y() >> position.z()
                >> quat.x() >> quat.y() >> quat.z() >> quat.w();

            Eigen::Affine3d pose;
            pose.setIdentity();
            pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
            pose.matrix().block<3,1>(0,3)=position;
       
            m_worldROS_baselink_vec.push_back ( std::pair<double, Eigen::Affine3d>((double)timestamp,pose) );
        }
    }else{
        LOG(FATAL) << "Not a known pose file format" << m_pose_file_format;
    }






}


void DataLoaderCloudRos::read_pose_file_vel2lasermap(){
    std::ifstream infile( "/media/rosu/Data/data/jan_razlaw/poses/laser_poses_lasermap_vel.txt" );
    if(!infile.is_open()){
        LOG(FATAL) << "Could not open pose file " << m_pose_file;
    }
    uint64_t scan_nr;
    uint64_t timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond quat;


    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >> timestamp
            >> position.x() >> position.y() >> position.z()
            >> quat.x() >> quat.y() >> quat.z() >> quat.w();

        timestamp*=m_timestamp_multiplier;
    //    std::cout << "input is \n" << " " << timestamp << "  \n Position is" << position << " " << quat.matrix()  << "\n";
        Eigen::Affine3d pose;
        pose.setIdentity();
        pose.matrix().block<3,3>(0,0)=quat.toRotationMatrix();
        pose.matrix().block<3,1>(0,3)=position;
        // if(m_do_timestamp_rounding_when_reading_file){
        //     timestamp=(uint64_t)std::round(timestamp/100000.0); ////TODO this is a dirty hack to reduce the discretization of time because the timestamps don't exactly match
        // }
    // VLOG(2) << "recorded tmestamp is " << timestamp;
//        VLOG(2) << "recorded scan_nr is " << scan_nr;
        // Eigen::Affine3d pose_inv=pose.inverse();

        m_lasermap_vel_vec.push_back ( std::pair<double, Eigen::Affine3d>((double)timestamp,pose) );
    }






}


bool DataLoaderCloudRos::get_pose_at_timestamp(Eigen::Affine3d& pose, double& deviation_ms, const double timestamp){

    // auto got = m_worldROS_baselink_map.find (timestamp);
    // std::cout << std::fixed << "querying" << timestamp << '\n';

    //return the closest one
    uint64_t closest_idx=-1;
    double smallest_timestamp_diff=std::numeric_limits<double>::max();
    double smallest_timestamp_diff_no_abs=std::numeric_limits<double>::max();
    for (size_t i = 0; i < m_worldROS_baselink_vec.size(); i++) {
        double recorded_timestamp=m_worldROS_baselink_vec[i].first;
        // Eigen::Affine3d pose=m_worldROS_baselink_vec[i].second;
        // std::cout << "comparing recorded_timestamp to timestmp \n" << recorded_timestamp << "\n" << timestamp << '\n';
        double diff=fabs((double)timestamp- (double)recorded_timestamp);
        if (  diff < smallest_timestamp_diff){
            closest_idx=i;
            smallest_timestamp_diff=diff;
            smallest_timestamp_diff_no_abs=(double)timestamp - (double)recorded_timestamp;
        }
    }
    if ( smallest_timestamp_diff > 0 && m_exact_time ){
        LOG(WARNING) << "time difference for pose is way too large! " << (smallest_timestamp_diff/1e6) << "s." << '\n';
        return false;
    }
    // std::cout << "smallest_timestamp_diff is " << smallest_timestamp_diff << '\n';
    // std::cout << "closest_idx is " << closest_idx << '\n';
    // std::cout << std::fixed << "closest_timestmap is " << m_worldROS_baselink_vec[closest_idx].first << '\n';
    // std::cout << "smallest_timestamp_diff_no_abs is " << smallest_timestamp_diff_no_abs << '\n';
    deviation_ms=(double)(smallest_timestamp_diff_no_abs)/1e6;
    // std::cout << "deviation_ms is " << deviation_ms << '\n';
    pose=m_worldROS_baselink_vec[closest_idx].second;
    // std::cout << "pose is \n " << pose.matrix() << '\n';
    return true;


}

bool DataLoaderCloudRos::get_pose_vel2lasermap_at_timestamp(Eigen::Affine3d& pose, double& deviation_ms, const double timestamp){

    // auto got = m_worldROS_baselink_map.find (timestamp);
    // std::cout << std::fixed << "querying" << timestamp << '\n';

    //return the closest one
    uint64_t closest_idx=-1;
    double smallest_timestamp_diff=std::numeric_limits<double>::max();
    double smallest_timestamp_diff_no_abs=std::numeric_limits<double>::max();
    for (size_t i = 0; i < m_lasermap_vel_vec.size(); i++) {
        double recorded_timestamp=m_lasermap_vel_vec[i].first;
        // Eigen::Affine3d pose=m_lasermap_vel_vec[i].second;
        // std::cout << "comparing recorded_timestamp to timestmp \n" << recorded_timestamp << "\n" << timestamp << '\n';
        double diff=fabs((double)timestamp- (double)recorded_timestamp);
        if (  diff < smallest_timestamp_diff){
            closest_idx=i;
            smallest_timestamp_diff=diff;
            smallest_timestamp_diff_no_abs=(double)timestamp - (double)recorded_timestamp;
        }
    }
    if ( smallest_timestamp_diff > 0 && m_exact_time ){
        LOG(WARNING) << "time difference for pose is way too large! " << (smallest_timestamp_diff/1e6) << "s." << '\n';
        return false;
    }
    // std::cout << "smallest_timestamp_diff is " << smallest_timestamp_diff << '\n';
    // std::cout << "closest_idx is " << closest_idx << '\n';
    // std::cout << std::fixed << "closest_timestmap is " << m_lasermap_vel_vec[closest_idx].first << '\n';
    // std::cout << "smallest_timestamp_diff_no_abs is " << smallest_timestamp_diff_no_abs << '\n';
    deviation_ms=(double)(smallest_timestamp_diff_no_abs)/1e6;
    // std::cout << "deviation_ms is " << deviation_ms << '\n';
    pose=m_lasermap_vel_vec[closest_idx].second;
    // std::cout << "pose is \n " << pose.matrix() << '\n';
    return true;


}


void DataLoaderCloudRos::create_transformation_matrices(){
    m_tf_baselink_vel.setIdentity();
    Eigen::Vector3d baselink_vel_t(-0.000, -0.000, -0.177);

    //TODO the quaternion didn't quite work here
    Eigen::AngleAxisd rollAngle(-3.142, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd yawAngle(0.0, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd pitchAngle(-1.614, Eigen::Vector3d::UnitZ());
    Eigen::Quaternion<double> baselink_vel_quat = pitchAngle * yawAngle * rollAngle;

    m_tf_baselink_vel.matrix().block<3,3>(0,0)=baselink_vel_quat.toRotationMatrix();
    m_tf_baselink_vel.matrix().block<3,1>(0,3)=baselink_vel_t;



    m_tf_worldGL_worldROS.setIdentity();
    Eigen::Matrix3d worldGL_worldROS_rot;
    worldGL_worldROS_rot = Eigen::AngleAxisd(-0.5*M_PI, Eigen::Vector3d::UnitX());
    m_tf_worldGL_worldROS.matrix().block<3,3>(0,0)=worldGL_worldROS_rot;
}