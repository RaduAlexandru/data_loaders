#include "data_loaders/DataTransformer.h"

//c++


//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>



//my stuff 
#include "easy_pbr/Mesh.h"
// #include "data_loaders/utils/MiscUtils.h"
#include "RandGenerator.h"
#include "ColorMngr.h"
#include "numerical_utils.h"

// using namespace er::utils;
using namespace easy_pbr::utils;


DataTransformer::DataTransformer(const configuru::Config& config):
    m_rand_gen(new RandGenerator)
{

    init_params(config);

}

void DataTransformer::init_params(const Config& transformer_config){

    //read all the parameters
    // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    // Config loader_config=cfg["loader_semantic_kitti"];

    m_random_translation_xyz_magnitude=transformer_config["random_translation_xyz_magnitude"];
    m_random_translation_xz_magnitude=transformer_config["random_translation_xz_magnitude"];
    m_rotation_y_max_angle=transformer_config["rotation_y_max_angle"];
    m_random_stretch_xyz_magnitude=transformer_config["random_stretch_xyz_magnitude"];
    m_adaptive_subsampling_falloff_start=transformer_config["adaptive_subsampling_falloff_start"];
    m_adaptive_subsampling_falloff_end=transformer_config["adaptive_subsampling_falloff_end"];
    m_random_subsample_percentage=transformer_config["random_subsample_percentage"];
    m_random_mirror_x=transformer_config["random_mirror_x"];
    m_random_mirror_z=transformer_config["random_mirror_z"];
    m_random_rotation_90_degrees_y=transformer_config["random_rotation_90_degrees_y"];

    m_hsv_jitter=transformer_config["hsv_jitter"];
    m_chance_of_xyz_noise = transformer_config["chance_of_xyz_noise"];
    m_xyz_noise_stddev=transformer_config["xyz_noise_stddev"];

}

MeshSharedPtr DataTransformer::transform(MeshSharedPtr& mesh){

    // Mesh transformed_mesh=mesh;

    //adaptive subsampling
    if(m_adaptive_subsampling_falloff_end!=0.0){
        CHECK(m_adaptive_subsampling_falloff_start<m_adaptive_subsampling_falloff_end) << " The falloff for the adaptive subsampling start should be lower than the end. For example we start at 0 meters and we end at 60m. The start is " << m_adaptive_subsampling_falloff_start << " adn the end is " << m_adaptive_subsampling_falloff_end;
        std::vector<bool> marked_to_be_removed(mesh->V.rows(), false);
        for(int i=0; i<mesh->V.rows(); i++){
            float dist=mesh->V.row(i).norm();
            float prob_to_remove= map(dist, m_adaptive_subsampling_falloff_start, m_adaptive_subsampling_falloff_end, 0.5, 0.0 ); //the closer verts have a high prob to be removed and the further away ones have one that is close to 0
            float r_val = m_rand_gen->rand_float(0.0, 1.0);
            if(r_val < prob_to_remove) { //the r_val will have no chance in going very low so it will not remove the points with prob_to_remove close to 0.0
                marked_to_be_removed[i]=true;
            }
        }
        mesh->remove_marked_vertices(marked_to_be_removed, false);
    }

    if(m_random_subsample_percentage!=0.0){
        float prob_of_death=m_random_subsample_percentage;
        int vertices_marked_for_removal=0;
        std::vector<bool> is_vertex_to_be_removed(mesh->V.rows(), false);
        for(int i = 0; i < mesh->V.rows(); i++){
            float random= m_rand_gen->rand_float(0.0, 1.0);
            if(random<prob_of_death){
                is_vertex_to_be_removed[i]=true;
                vertices_marked_for_removal++;
            }
        }
        mesh->remove_marked_vertices(is_vertex_to_be_removed, false);
    }

    if(m_random_translation_xyz_magnitude!=0.0){
        float translation_strength=m_random_translation_xyz_magnitude;
        Eigen::Affine3d tf;
        tf.setIdentity();
        tf.translation().x()=m_rand_gen->rand_float(-1.0, 1.0)*translation_strength;
        tf.translation().y()=m_rand_gen->rand_float(-1.0, 1.0)*translation_strength;
        tf.translation().z()=m_rand_gen->rand_float(-1.0, 1.0)*translation_strength;
        mesh->transform_vertices_cpu(tf);
    }

    if(m_random_translation_xz_magnitude!=0.0){
        float translation_strength=m_random_translation_xz_magnitude;
        Eigen::Affine3d tf;
        tf.setIdentity();
        tf.translation().x()=m_rand_gen->rand_float(-1.0, 1.0)*translation_strength;
        tf.translation().z()=m_rand_gen->rand_float(-1.0, 1.0)*translation_strength;
        mesh->transform_vertices_cpu(tf);
    }



    if(m_random_stretch_xyz_magnitude!=0.0){
        float s=m_random_stretch_xyz_magnitude;
        float stretch_factor_x=1.0 + m_rand_gen->rand_float(-s, s);
        float stretch_factor_y=1.0 + m_rand_gen->rand_float(-s, s);
        float stretch_factor_z=1.0 + m_rand_gen->rand_float(-s, s);
        mesh->V.col(0)*=stretch_factor_x;
        mesh->V.col(1)*=stretch_factor_y;
        mesh->V.col(2)*=stretch_factor_z;
    }

    //random rotation in y
    if(m_rotation_y_max_angle!=0){
        Eigen::Affine3d tf;
        tf.setIdentity();
        Eigen::Matrix3d tf_rot;
        float rand_angle_degrees=m_rand_gen->rand_float(0.0, m_rotation_y_max_angle);
        // float rand_angle_radians=degrees2radians(rand_angle_degrees);
        float rand_angle_radians=rand_angle_degrees * M_PI / 180.0;
        tf_rot = Eigen::AngleAxisd(rand_angle_radians, Eigen::Vector3d::UnitY());
        tf.matrix().block<3,3>(0,0)=tf_rot;
        mesh->transform_vertices_cpu(tf);
    }

    //random mirror along the yz plane will negate the x coordinate
    if(m_random_mirror_x){
        bool do_flip=m_rand_gen->rand_bool(0.5); //50/50 will do a flip
        if(do_flip){
            mesh->V.col(0)=-mesh->V.col(0);
        }
    }
    //random mirror along the xy plane will negate the z coordinate
    if(m_random_mirror_z){
        bool do_flip=m_rand_gen->rand_bool(0.5); //50/50 will do a flip
        if(do_flip){
            mesh->V.col(2)=-mesh->V.col(2);
        }
    }

    if(m_random_rotation_90_degrees_y){
        Eigen::Affine3d tf;
        tf.setIdentity();
        Eigen::Matrix3d tf_rot;
        //we rotate 90 degrees, or 180, or 270 randomly
        int nr_times=m_rand_gen->rand_int(0, 3);
        float rand_angle_degrees=90*nr_times;
        float rand_angle_radians=rand_angle_degrees * M_PI / 180.0;
        tf_rot = Eigen::AngleAxisd(rand_angle_radians, Eigen::Vector3d::UnitY());
        tf.matrix().block<3,3>(0,0)=tf_rot;
        mesh->transform_vertices_cpu(tf);
    }

    if (!m_hsv_jitter.isZero() && mesh->C.size()){
        //get a random jitter with those params
        Eigen::Vector3d hsv_noise;
        hsv_noise << m_rand_gen->rand_float(-m_hsv_jitter.x(), m_hsv_jitter.x() ), m_rand_gen->rand_float( -m_hsv_jitter.y(), m_hsv_jitter.y()  ), m_rand_gen->rand_float( -m_hsv_jitter.z(), m_hsv_jitter.z()  );
        for(int i=0; i<mesh->C.rows(); i++){
            Eigen::Vector3d color_rgb=mesh->C.row(i);
            Eigen::Vector3d hsv=rgb2hsv(color_rgb );
            hsv+=hsv_noise;
            hsv.x()= wrap(hsv.x(), 360.0); //hue is a 360 degree circle so a wrap is better than a clamp
            hsv.y()= clamp(hsv.y(), 0.0, 1.0);
            hsv.z()= clamp(hsv.z(), 0.0, 1.0);
            Eigen::Vector3d rgb= hsv2rgb(hsv);
            mesh->C.row(i)=rgb;       
        }
    }

    bool do_xyz_noise=m_rand_gen->rand_bool(m_chance_of_xyz_noise);
    if(do_xyz_noise){
        if (!m_xyz_noise_stddev.isZero()){
            for(int i = 0; i < mesh->V.rows(); i++){
                mesh->V(i,0)+=m_rand_gen->rand_normal_float(0.0, m_xyz_noise_stddev(0));
                mesh->V(i,1)+=m_rand_gen->rand_normal_float(0.0, m_xyz_noise_stddev(1));
                mesh->V(i,2)+=m_rand_gen->rand_normal_float(0.0, m_xyz_noise_stddev(2));
            }
        }
    }


    return mesh;

}

