#pragma once

//c++
#include <iostream>
#include <memory>

// #include <glad/glad.h> // Initialize with gladLoadGL()
// Include glfw3.h after our OpenGL definitions
// #include <GLFW/glfw3.h>

//opencv
// #include "opencv2/opencv.hpp"

//dir watcher
#ifdef WITH_DIR_WATCHER
    #include "dir_watcher/dir_watcher.hpp"
#endif

//gl
#include "Texture2D.h"
#include "Shader.h"

// #include <nlohmann/json_fwd.hpp>
#include "nlohmann/json.hpp"


//forward declarations
class Viewer;
class Recorder;
class RandGenerator;
class MeshGL;

//in order to dissalow building on the stack and having only ptrs https://stackoverflow.com/a/17135547
class SyntheticGenerator;

class SyntheticGenerator: public std::enable_shared_from_this<SyntheticGenerator>
{
public:
    //https://stackoverflow.com/questions/29881107/creating-objects-only-as-shared-pointers-through-a-base-class-create-method
    template <class ...Args>
    static std::shared_ptr<SyntheticGenerator> create( Args&& ...args ){
        return std::shared_ptr<SyntheticGenerator>( new SyntheticGenerator(std::forward<Args>(args)...) );
        // return std::make_shared<SyntheticGenerator>( std::forward<Args>(args)... );
    }
    ~SyntheticGenerator();
    


private:
    SyntheticGenerator(const std::string& config_file, const std::shared_ptr<Viewer>& view); // we put the constructor as private so as to dissalow creating the object on the stack because we want to only used shared ptr for it
    // SyntheticGenerator(const std::string& config_file);

    std::shared_ptr<Viewer> m_view;
    std::shared_ptr<Recorder> m_recorder;
    std::shared_ptr<RandGenerator> m_rand_gen;

    gl::Texture2D m_balloon_outline_tex;
    gl::Texture2D m_copter_blob_tex;
    gl::Shader m_detect_balloon_shader;
    gl::Shader m_detect_copter_shader;
    std::shared_ptr<MeshGL> m_fullscreen_quad; //we store it here because we precompute it and then we use for composing the final image after the deffered geom pass
    int m_iter_write;
    Eigen::Vector4f m_last_copter_bb_detected; //bounding box of the last detected copter. Contains x,y,width,height
    Eigen::Vector4f m_last_ball_bb_detected; //bounding box of the last detected copter. Contains x,y,width,height
    cv::Mat m_bb_mat; //for showing the bounding box of the copter detection
    cv::Mat m_ball_bb_mat; //for showing the bounding box of the copter detection
    nlohmann::json m_json_file; //for writing the copter detections into a coco like dataset
    Eigen::Vector3f m_camera_prev_direction;


    #ifdef WITH_DIR_WATCHER
        emilib::DelayedDirWatcher dir_watcher;
    #endif
    

    //params
    bool m_write_with_random_idx; //recorder will either write files with a random idx or with a incrementing one
    std::string m_rgb_output_path;
    std::string m_gt_output_path;


    void init_params(const std::string config_file);
    void compile_shaders();
    void init_opengl();
    void hotload_shaders();
    void install_callbacks(const std::shared_ptr<Viewer>& view); //installs some callbacks that will be called by the viewer after it finishes an update

    //pre draw callbacks
    void randomize_cam(Viewer& view);
    void randomize_balloon(Viewer& view);
    void randomize_copter(Viewer& view);
    void randomize_net(Viewer& view);

    //post draw callbacks
    void detect_balloon(Viewer& view);
    void detect_copter(Viewer& view);
    void detect_ball(Viewer& view);
    void record_data(Viewer& view);
    void write_to_json(Viewer& view);
};
