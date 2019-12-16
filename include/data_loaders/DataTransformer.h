//eigen 
#include <Eigen/Core>

#include "easy_pbr/Mesh.h"

#include <configuru.hpp>


class RandGenerator;

class DataTransformer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataTransformer(const configuru::Config& config_file);

    Mesh transform(const Mesh& mesh);

private:

    void init_params(const configuru::Config& config_file);

    //objects 
    std::shared_ptr<RandGenerator> m_rand_gen;

    //params
    float m_random_translation_xyz_magnitude;
    float m_random_translation_xz_magnitude;
    float m_rotation_y_max_angle;
    float m_random_stretch_xyz_magnitude;
    float m_adaptive_subsampling_falloff_start;
    float m_adaptive_subsampling_falloff_end;
    float m_random_subsample_percentage;
    bool m_random_mirror_x;
    bool m_random_mirror_z;
    bool m_random_rotation_90_degrees_y;


    //internal

};