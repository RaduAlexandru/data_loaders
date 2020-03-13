//eigen 
#include <Eigen/Core>

#include <memory>

#include <configuru.hpp>


class RandGenerator;
class Mesh;

class DataTransformer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DataTransformer(const configuru::Config& config_file);

    std::shared_ptr<Mesh> transform(std::shared_ptr<Mesh>& mesh);

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

    Eigen::Vector3f m_hsv_jitter;
    float m_chance_of_xyz_noise;
    Eigen::Vector3f m_xyz_noise_stddev;


    //internal

};