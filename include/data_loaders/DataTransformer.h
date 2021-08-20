//eigen
#include <Eigen/Core>

#include <memory>

#include <configuru.hpp>

namespace radu { namespace utils{
    class RandGenerator;
}}

namespace easy_pbr{
    class Mesh;
}

class DataTransformer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataTransformer(const configuru::Config& config_file);

    std::shared_ptr<easy_pbr::Mesh> transform(std::shared_ptr<easy_pbr::Mesh>& mesh);

    //params
    Eigen::Vector3f m_random_translation_xyz_magnitude;
    float m_rotation_x_max_angle;
    float m_rotation_y_max_angle;
    float m_rotation_z_max_angle;
    Eigen::Vector3f m_random_stretch_xyz_magnitude;
    float m_adaptive_subsampling_falloff_start;
    float m_adaptive_subsampling_falloff_end;
    float m_random_subsample_percentage;
    bool m_random_mirror_x;
    bool m_random_mirror_y;
    bool m_random_mirror_z;
    bool m_random_rotation_90_degrees_y;

    Eigen::Vector3f m_hsv_jitter;
    float m_chance_of_xyz_noise;
    Eigen::Vector3f m_xyz_noise_stddev;

private:

    void init_params(const configuru::Config& config_file);

    //objects
    std::shared_ptr<radu::utils::RandGenerator> m_rand_gen;




    //internal

};
