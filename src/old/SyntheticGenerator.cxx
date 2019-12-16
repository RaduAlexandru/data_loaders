#include "mbzirc_challenge_1/SyntheticGenerator.h"

#include <fstream>

//loguru
#define LOGURU_NO_DATE_TIME 1
#define LOGURU_NO_UPTIME 1
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>

//My stuff 
#include "UtilsGL.h"
#include "easy_pbr/Viewer.h"
#include "easy_pbr/Recorder.h"
#include "easy_pbr/Gui.h"
#include "easy_pbr/MeshGL.h"
#include "easy_pbr/Mesh.h"
#include "easy_pbr/Scene.h"
#include "easy_pbr/Camera.h"
#include "RandGenerator.h"
#include "ColorMngr.h"
#include "opencv_utils.h"
#include "numerical_utils.h"

//Add this header after we add all opengl stuff because we need the profiler to have glFinished defined
#define ENABLE_GL_PROFILING 1
#include "Profiler.h" 


//configuru
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>
using namespace configuru;

//boost
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

using namespace easy_pbr::utils;
using json = nlohmann::json;

// SyntheticGenerator::SyntheticGenerator(const std::string& config_file):
SyntheticGenerator::SyntheticGenerator(const std::string& config_file, const std::shared_ptr<Viewer>& view):
    #ifdef WITH_DIR_WATCHER 
        dir_watcher(std::string(PROJECT_SOURCE_DIR)+"/shaders/",5),
    #endif
    m_view(view),
    m_fullscreen_quad(MeshGL::create()),
    m_recorder(new Recorder()),
    m_iter_write(0)
    {
        m_last_copter_bb_detected.setZero();
        m_last_ball_bb_detected.setZero();
        m_camera_prev_direction.setZero();

        init_params(config_file);
        compile_shaders(); 
        init_opengl();                     
        install_callbacks(view);
}

SyntheticGenerator::~SyntheticGenerator(){

    //add the categories
    //balloon first and then copter
    json balloon_category;
    balloon_category["supercategory"]="detection";
    balloon_category["id"]=1;
    balloon_category["name"]="balloon";
    json copter_category;
    copter_category["supercategory"]="detection";
    copter_category["id"]=2;
    copter_category["name"]="copter";
    json ball_category;
    ball_category["supercategory"]="detection";
    ball_category["id"]=3;
    ball_category["name"]="ball";
    m_json_file["categories"].push_back(balloon_category);
    m_json_file["categories"].push_back(copter_category);
    m_json_file["categories"].push_back(ball_category);
    

    std::string path=(fs::path(m_rgb_output_path)/"coco_format_dataset.json").string();
    VLOG(1) << "Writing json file to disk in: "<< path;
    if (!fs::exists(m_rgb_output_path)){
        fs::create_directories(m_rgb_output_path);
    }
    std::ofstream file( path);
    file << m_json_file;

}

void SyntheticGenerator::init_params(const std::string config_file){

    //read all the parameters
    // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
    std::string config_file_abs;
    if (fs::path(config_file).is_relative()){
        config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
    }else{
        config_file_abs=config_file;
    }

    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config synth_config=cfg["synthetic_generator"];
    m_write_with_random_idx = synth_config["write_with_random_idx"];
    m_rgb_output_path = (std::string)synth_config["rgb_output_path"];
    m_gt_output_path = (std::string)synth_config["gt_output_path"];
    m_gt_output_path = (std::string)synth_config["gt_output_path"];

    // m_show_gui = synth_config["show_gui"];
    unsigned int seed=0;
    m_rand_gen=std::make_shared<RandGenerator> (seed);

}


void SyntheticGenerator::compile_shaders(){
       
    m_detect_balloon_shader.compile( std::string(PROJECT_SOURCE_DIR)+"/shaders/detect_balloon_vert.glsl", std::string(PROJECT_SOURCE_DIR)+"/shaders/detect_balloon_frag.glsl" ) ;
    m_detect_copter_shader.compile( std::string(PROJECT_SOURCE_DIR)+"/shaders/detect_copter_vert.glsl", std::string(PROJECT_SOURCE_DIR)+"/shaders/detect_copter_frag.glsl" ) ;
}

void SyntheticGenerator::init_opengl(){
    //create a fullscreen quad which we will use for composing the final image after the deffrred render pass
    m_fullscreen_quad->m_core->create_full_screen_quad();
    GL_C( m_fullscreen_quad->upload_to_gpu() );

}

void SyntheticGenerator::hotload_shaders(){
    #ifdef WITH_DIR_WATCHER
        std::vector<std::string> changed_files=dir_watcher.poll_files();
        if(changed_files.size()>0){
            compile_shaders();
        }
    #endif
}

void SyntheticGenerator::install_callbacks(const std::shared_ptr<Viewer>& view){


    //pre
    view->add_callback_pre_draw( [this]( Viewer& v ) -> void{ this->randomize_cam(v); }  );
    view->add_callback_pre_draw( [this]( Viewer& v ) -> void{ this->randomize_copter(v); }  );
    view->add_callback_pre_draw( [this]( Viewer& v ) -> void{ this->randomize_balloon(v); }  );
    view->add_callback_pre_draw( [this]( Viewer& v ) -> void{ this->randomize_net(v); }  );


    //post
    view->add_callback_post_draw( [this]( Viewer& v ) -> void{ this->detect_balloon(v); }  );
    view->add_callback_post_draw( [this]( Viewer& v ) -> void{ this->detect_copter(v); }  );
    view->add_callback_post_draw( [this]( Viewer& v ) -> void{ this->detect_ball(v); }  );
    view->add_callback_post_draw( [this]( Viewer& v ) -> void{ this->record_data(v); }  );
}

void SyntheticGenerator::randomize_cam(Viewer& view){
    if (!view.m_first_draw){
        //copy the default cam and assign the camera to the modified one (the default camera stays untouched)
        std::shared_ptr<Camera> new_cam = std::make_shared<Camera>(*(view.m_default_camera) );
        // Eigen::Quaternionf q=Eigen::Quaternionf::UnitRandom();

        //push away by some random dist
        // new_cam->push_away( m_rand_gen->rand_float(0.1, 10.0) ); //a value of 1 means no movement ( lower than 1 means closer and higher than 1 means further)
        // new_cam->push_away( m_rand_gen->rand_float(0.5, 2.0) ); //a value of 1 means no movement ( lower than 1 means closer and higher than 1 means further)
        new_cam->push_away( 0.07 ); //a value of 1 means no movement ( lower than 1 means closer and higher than 1 means further)
        // new_cam->push_away( m_rand_gen->rand_float(0.5, 5.0) ); //a value of 1 means no movement ( lower than 1 means closer and higher than 1 means further)
        
        // rotate around Y axis of the world (the vector 0,1,0)
        Eigen::Quaternionf q;
        Eigen::Vector3f axis_y;
        axis_y << 0,1,0; 
        Eigen::Quaternionf q_y = Eigen::Quaternionf( Eigen::AngleAxis<float>( m_rand_gen->rand_float(0, 2*3.14) ,  axis_y.normalized() ) );
        q_y.normalize(); 
        // //rotate around x axis of the camera coordinate
        Eigen::Vector3f axis_x;
        axis_x = view.m_default_camera->cam_axes().col(0);
        Eigen::Quaternionf q_x = Eigen::Quaternionf( Eigen::AngleAxis<float>( m_rand_gen->rand_float(-0.1*3.14, 0.3*3.14),  axis_x.normalized() ) ) ;
        q_x.normalize();
        q=q_y*q_x;
        new_cam->orbit(q);

        // //move thelookat vector a bit
        // float rand_trans_scale=500;
        // float rand_x=m_rand_gen->rand_float(-rand_trans_scale, rand_trans_scale);
        // float rand_y=m_rand_gen->rand_float(-rand_trans_scale, rand_trans_scale);
        // float rand_z=m_rand_gen->rand_float(-rand_trans_scale, rand_trans_scale);
        // Eigen::Vector3f lookat_displacement;
        // lookat_displacement << rand_x, rand_y, rand_z;
        // Eigen::Vector3f new_lookat=new_cam->lookat() + lookat_displacement;
        // new_cam->set_lookat(new_lookat);


        // rotate around Y axis of the world (the vector 0,1,0)
        m_camera_prev_direction=new_cam->direction(); //save the direction the camera is pointing into before we rotate it inplace. Will be useful when pushing object along the direction
        Eigen::Quaternionf q_rc;
        Eigen::Quaternionf q_rc_y = Eigen::Quaternionf( Eigen::AngleAxis<float>( m_rand_gen->rand_float(-0.4, 0.4) ,  axis_y.normalized() ) );
        Eigen::Quaternionf q_rc_x = Eigen::Quaternionf( Eigen::AngleAxis<float>( m_rand_gen->rand_float(-0.4, 0.4) ,  axis_x.normalized() ) );
        q_rc=q_rc_y*q_rc_x;
        new_cam->rotate(q_rc);



        // //exposure
        new_cam->m_exposure=m_rand_gen->rand_float(0.05, 15.0);

        view.m_camera=new_cam;


    }

}


void SyntheticGenerator::randomize_balloon(Viewer& view){
    if (!view.m_first_draw){
        //transform balloon if it exists
        bool exists=Scene::does_mesh_with_name_exist("balloon");
        if (exists){
            MeshSharedPtr balloon=Scene::get_mesh_with_name("balloon");

            //move the copter because that is the one that has the herarchy and it has the pole as child
            Eigen::Vector3d displacement_to_center=view.m_camera->position().cast<double>() - balloon->m_model_matrix.translation();
            balloon->translate_model_matrix( displacement_to_center  );
            float mov=m_rand_gen->rand_float(1.0, 20);
            //the displacement is along the direction of the camera BEFORE we rotated it in place. Therefore the balloon should now end up on the side of the image somewhere
            Eigen::Vector3f displacement;
            if (m_camera_prev_direction.isZero()){
                displacement= view.m_camera->direction()*mov; 
            }else{
                displacement= m_camera_prev_direction*mov; 
            }
            balloon->translate_model_matrix( displacement.cast<double>()  );



            MeshSharedPtr new_mesh=std::make_shared<Mesh>( balloon->clone() );
            balloon->m_vis.m_is_visible=false;
            new_mesh->m_vis.m_roughness+=m_rand_gen->rand_float(0.0, 0.3);
            // new_mesh->m_vis.m_solid_color=random_color(m_rand_gen).cast<float>();
            //make the color a random gray value
            float gray_val=m_rand_gen->rand_float(0.7,1.0);
            new_mesh->m_vis.m_solid_color << gray_val, gray_val, gray_val;
            new_mesh->m_vis.m_is_visible=true;
            new_mesh->m_force_vis_update=true;

            // //random squish and stretch 
            float s=0.1;
            float stretch_factor_x=1.0 + m_rand_gen->rand_float(-s, s);
            float stretch_factor_y=1.0 + m_rand_gen->rand_float(-s, s);
            float stretch_factor_z=1.0 + m_rand_gen->rand_float(-s, s);
            new_mesh->V.col(0)*=stretch_factor_x;
            new_mesh->V.col(1)*=stretch_factor_y;
            new_mesh->V.col(2)*=stretch_factor_z;

            // //transflate towards and away from the camera
            // float mov=m_rand_gen->rand_float(1.0, 15);
            // Eigen::Vector3f displacement= view.m_camera->direction()*mov; 
            // new_mesh->m_model_matrix.translation()=view.m_camera->position().cast<double>() + displacement.cast<double>();

            //randomly set the new mesh to not visible so we have some negative trianing samples as wekk
            // new_mesh->m_vis.m_is_visible=m_rand_gen->rand_bool(0.7);

            Scene::show(new_mesh, "balloon_transformed");
        }
    }
}

void SyntheticGenerator::randomize_copter(Viewer& view){
    if (!view.m_first_draw){
        //transform balloon if it exists
        bool exists=Scene::does_mesh_with_name_exist("copter");
        if (exists){
            MeshSharedPtr copter=Scene::get_mesh_with_name("copter");

            //move the copter because that is the one that has the herarchy
            Eigen::Vector3d displacement_to_center=view.m_camera->position().cast<double>() - copter->m_model_matrix.translation();
            // VLOG(1) << "displacement is " << displacement_to_center;
            copter->translate_model_matrix( displacement_to_center  );
            float mov=m_rand_gen->rand_float(1.0, 15);
            //the displacement is along the direction of the camera BEFORE we rotated it in place. Therefore the balloon should now end up on the side of the image somewhere
            Eigen::Vector3f displacement;
            if (m_camera_prev_direction.isZero()){
                displacement= view.m_camera->direction()*mov; 
            }else{
                displacement= m_camera_prev_direction*mov; 
            }
            copter->translate_model_matrix( displacement.cast<double>()  );


            MeshSharedPtr new_mesh=std::make_shared<Mesh>( copter->clone() );
            copter->m_vis.m_is_visible=false;
            new_mesh->m_vis.m_roughness+=m_rand_gen->rand_float(-0.2, 0.2);
            // make the color slightly sway from the black we had added intially
            Eigen::Vector3f color_noise;
            color_noise << m_rand_gen->rand_float(-0.2,0.2), m_rand_gen->rand_float(-0.2,0.2), m_rand_gen->rand_float(-0.2,0.2);
            new_mesh->m_vis.m_solid_color += color_noise;
            // clamp the color in case we moved it too much
            new_mesh->m_vis.m_solid_color.x()=clamp(new_mesh->m_vis.m_solid_color.x(),0.0f, 1.0f);
            new_mesh->m_vis.m_solid_color.y()=clamp(new_mesh->m_vis.m_solid_color.y(),0.0f, 1.0f);
            new_mesh->m_vis.m_solid_color.z()=clamp(new_mesh->m_vis.m_solid_color.z(),0.0f, 1.0f);
            new_mesh->m_vis.m_is_visible=true;
            new_mesh->m_force_vis_update=true;

            // //random squish and stretch 
            // float s=0.3;
            // float stretch_factor_x=1.0 + m_rand_gen->rand_float(-s, s);
            // float stretch_factor_y=1.0 + m_rand_gen->rand_float(-s, s);
            // float stretch_factor_z=1.0 + m_rand_gen->rand_float(-s, s);
            // new_mesh->V.col(0)*=stretch_factor_x;
            // new_mesh->V.col(1)*=stretch_factor_y;
            // new_mesh->V.col(2)*=stretch_factor_z;

            // // //random translate 
            // float scale=new_mesh->get_scale();
            // float dist=view.m_camera->dist_to_lookat();
            // float rand_trans_scale=scale*dist*1.0;
            // float rand_x=m_rand_gen->rand_float(-rand_trans_scale, rand_trans_scale);
            // float rand_y=m_rand_gen->rand_float(-rand_trans_scale, rand_trans_scale);
            // float rand_z=m_rand_gen->rand_float(-rand_trans_scale, rand_trans_scale);
            // Eigen::Affine3d tf;
            // tf.setIdentity();
            // tf.translation() << rand_x, rand_y, rand_z;
            // new_mesh->apply_transform(tf, true);

            //transflate towards and away from the camera
            // Eigen::Vector3d displacement=view.m_camera->position().cast<double>() - copter->m_model_matrix.translation();
            // if (!view.m_camera){
                // LOG(FATAL) << "why is there no camera";
            // }
            // VLOG(1) << "cam position is " << view.m_camera->position().cast<double>();
            // VLOG(1) << "cam is " << m_view->m_camera;
            // VLOG(1) << "cam position is " << m_view->m_camera->model_matrix();
            // new_mesh->
            // new_mesh->m_model_matrix.translation()=view.m_camera->position().cast<double>() + displacement.cast<double>();

            //randomly set the new mesh to not visible so we have some negative trianing samples as wekk
            // new_mesh->m_vis.m_is_visible=m_rand_gen->rand_bool(0.7);


            Scene::show(new_mesh, "copter_transformed");






            //rotate propelers 
            Eigen::Vector3d axis;
            MeshSharedPtr copter_tf=Scene::get_mesh_with_name("copter_transformed");
            axis=copter_tf->m_model_matrix.linear().col(1); //axis upon which the propellers rotate is the y axis of the copter so one pointing towards up

            bool prop_1_exists=Scene::does_mesh_with_name_exist("propeller1");
            if (prop_1_exists){
                MeshSharedPtr prop=Scene::get_mesh_with_name("propeller1");
                prop->rotate_model_matrix_local(axis, m_rand_gen->rand_float(0.0, 360.0));
            }

            bool prop_2_exists=Scene::does_mesh_with_name_exist("propeller2");
            if (prop_2_exists){
                MeshSharedPtr prop=Scene::get_mesh_with_name("propeller2");
                prop->rotate_model_matrix_local(axis, m_rand_gen->rand_float(0.0, 360.0));
            }

            bool prop_3_exists=Scene::does_mesh_with_name_exist("propeller3");
            if (prop_3_exists){
                MeshSharedPtr prop=Scene::get_mesh_with_name("propeller3");
                prop->rotate_model_matrix_local(axis, m_rand_gen->rand_float(0.0, 360.0));
            } 

            bool prop_4_exists=Scene::does_mesh_with_name_exist("propeller4");
            if (prop_4_exists){
                MeshSharedPtr prop=Scene::get_mesh_with_name("propeller4");
                prop->rotate_model_matrix_local(axis, m_rand_gen->rand_float(0.0, 360.0));
            }

            bool prop_5_exists=Scene::does_mesh_with_name_exist("propeller5");
            if (prop_5_exists){
                MeshSharedPtr prop=Scene::get_mesh_with_name("propeller5");
                prop->rotate_model_matrix_local(axis, m_rand_gen->rand_float(0.0, 360.0));
            } 

            bool prop_6_exists=Scene::does_mesh_with_name_exist("propeller6");
            if (prop_6_exists){
                MeshSharedPtr prop=Scene::get_mesh_with_name("propeller6");
                prop->rotate_model_matrix_local(axis, m_rand_gen->rand_float(0.0, 360.0));
            } 



            //rotate pole
            Eigen::Vector3d axis_x, axis_y;
            axis_x=copter_tf->m_model_matrix.linear().col(0);
            axis_y=copter_tf->m_model_matrix.linear().col(2);

            bool pole_exists=Scene::does_mesh_with_name_exist("pole");
            if (pole_exists){
                MeshSharedPtr pole=Scene::get_mesh_with_name("pole");
                //apply rotation matrix so that it the vector from copter to ball ends up being in mostly the same direction as the bottom dir of the copter
                float dot=1.0;
                do {
                    pole->rotate_model_matrix_local(axis_x, m_rand_gen->rand_float(-30.0, 30.0));
                    pole->rotate_model_matrix_local(axis_y, m_rand_gen->rand_float(-30.0, 30.0));
                    MeshSharedPtr ball=Scene::get_mesh_with_name("ball");
                    Eigen::Vector3d copter_down=-copter_tf->m_model_matrix.linear().col(1);
                    Eigen::Vector3d copter2ball=(ball->m_model_matrix.translation() - copter_tf->m_model_matrix.translation() ).normalized();
                    dot=copter_down.dot(copter2ball) ;
                }while(dot<0.5);

            }



        }
    }

   


}

void SyntheticGenerator::randomize_net(Viewer& view){
    if (!view.m_first_draw){
        //transform balloon if it exists
        bool exists=Scene::does_mesh_with_name_exist("net");
        if (exists){
            MeshSharedPtr net=Scene::get_mesh_with_name("net");

            //translate so that the center is at the lookat point, rotate there, then move

            //modify the model_matrix so that is coincides with the rotation of the camera
            net->m_model_matrix.linear()=Eigen::Affine3f(view.m_camera->model_matrix()).cast<double>().linear();
            net->m_model_matrix.translation()=view.m_camera->lookat().cast<double>();
            //rotate by 90 degrees with respect to y axis of the world so that it faces the camera. 
            // Eigen::Quaternionf align = Eigen::Quaternionf( Eigen::AngleAxis<float>( 90 * M_PI/180.0,  Eigen::Vector3f::UnitY() ) );
            // net->m_model_matrix.linear()=align.toRotationMatrix().cast<double>()* net->m_model_matrix.linear();
            // net->rotate_model_matrix_local(Eigen::Vector3d::UnitY(), 90);

            //rotate the net a bit 
            Eigen::Quaternionf rot;

            // rotate around Y axis of the world (the vector 0,1,0)
            Eigen::Vector3f axis_y;
            axis_y << 0,1,0; 
            Eigen::Quaternionf q_y = Eigen::Quaternionf( Eigen::AngleAxis<float>( m_rand_gen->rand_float(-0.8,0.8),  axis_y.normalized() ) );
            //rotate around the x axis of the cam
            Eigen::Vector3f axis_x;
            axis_x =view.m_camera->cam_axes().col(0); 
            Eigen::Quaternionf q_x = Eigen::Quaternionf( Eigen::AngleAxis<float>( m_rand_gen->rand_float(-0.3,0.3),  axis_x.normalized() ) );
             //rotate around the z axis of the cam
            Eigen::Vector3f axis_z;
            axis_z =view.m_camera->cam_axes().col(2); 
            Eigen::Quaternionf q_z = Eigen::Quaternionf( Eigen::AngleAxis<float>( m_rand_gen->rand_float(-0.3,0.3),  axis_z.normalized() ) );

            rot=q_z*q_x*q_y;

            // net->m_model_matrix.linear()=net->m_model_matrix.linear()*rot.toRotationMatrix().cast<double>();
            net->m_model_matrix.linear()=rot.toRotationMatrix().cast<double>()* net->m_model_matrix.linear();

             //transflate towards and away from the camera
            bool balloon_exists=Scene::does_mesh_with_name_exist("balloon_transformed");
            bool copter_exists=Scene::does_mesh_with_name_exist("copter_transformed");
            float min_mov=10;
            if (copter_exists){
                MeshSharedPtr copter=Scene::get_mesh_with_name("copter_transformed");
                min_mov =  (view.m_camera->position().cast<double>() - copter->m_model_matrix.translation()).norm();
            }
            if (balloon_exists){
                MeshSharedPtr balloon=Scene::get_mesh_with_name("balloon_transformed");
                min_mov =  (view.m_camera->position().cast<double>() - balloon->m_model_matrix.translation()).norm();
            }
            // min_mov*=1.5;
            // min_mov=clamp(min_mov, 0.0f, 25.0f);
            float mov=m_rand_gen->rand_float(min_mov*1.5, min_mov*1.5+10);
            Eigen::Vector3f displacement= view.m_camera->direction()*mov; 
            net->m_model_matrix.translation()=view.m_camera->position().cast<double>() + displacement.cast<double>();

            // //randomly set the new mesh to not visible so we have some negative trianing samples as wekk
            net->m_vis.m_is_visible=m_rand_gen->rand_bool(0.7);
            
        }
    }

}

// void SyntheticGenerator::detect_balloon(std::shared_ptr<Viewer>& view){
void SyntheticGenerator::detect_balloon(Viewer& view){

    bool exists=Scene::does_mesh_with_name_exist("balloon_transformed");
    if(exists){

        TIME_START("detect_balloon");

        // //create a final image the same size as the framebuffer
        // m_balloon_outline_tex.allocate_or_resize(GL_R8, GL_RED, GL_UNSIGNED_BYTE, view.m_gbuffer.width(), view.m_gbuffer.height() );
        int outline_subsample=2; //the detection network may need to have the outline at a lower res
        m_balloon_outline_tex.allocate_or_resize(GL_R8, GL_RED, GL_UNSIGNED_BYTE, view.m_gbuffer.width()/outline_subsample, view.m_gbuffer.height()/outline_subsample );

        // //dont perform depth checking nor write into the depth buffer 
        glDepthMask(false);
        glDisable(GL_DEPTH_TEST);
        glViewport(0.0f , 0.0f, view.m_gbuffer.width()/outline_subsample, view.m_gbuffer.height()/outline_subsample );

        gl::Shader& shader= m_detect_balloon_shader;

        // Set attributes that the vao will pulll from buffers
        GL_C( m_fullscreen_quad->vao.vertex_attribute(shader, "position", m_fullscreen_quad->V_buf, 3) );
        GL_C( m_fullscreen_quad->vao.vertex_attribute(shader, "uv", m_fullscreen_quad->UV_buf, 2) );
        m_fullscreen_quad->vao.indices(m_fullscreen_quad->F_buf); //Says the indices with we refer to vertices, this gives us the triangles
        
        
        //  //shader setup
        GL_C( shader.use() );
        // shader.bind_texture(view.m_gbuffer.tex_with_name("depth_gtex"),"depth_tex");
        shader.bind_texture(view.m_gbuffer.tex_with_name("mesh_id_gtex"),"mesh_id_tex");

        shader.draw_into(m_balloon_outline_tex, "out_color");

        // // draw
        m_fullscreen_quad->vao.bind(); 
        glDrawElements(GL_TRIANGLES, m_fullscreen_quad->m_core->F.size(), GL_UNSIGNED_INT, 0);
        TIME_END("detect_balloon");

        // //restore the state
        glDepthMask(true);
        glEnable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);


        // VIEW image 
        // view.m_gui->show_gl_texture(view.m_final_tex.get_tex_id(), "final_tex", true);
        view.m_gui->show_gl_texture(m_balloon_outline_tex.tex_id(), "balloon_outline_tex", true);

    }


}
void SyntheticGenerator::detect_copter(Viewer& view){

    bool exists=Scene::does_mesh_with_name_exist("copter_transformed");
    // bool exists=Scene::does_mesh_with_name_exist("copter");
    if(exists){

        TIME_START("detect_copter");

        // //create a final image the same size as the framebuffer
        m_copter_blob_tex.allocate_or_resize(GL_R8, GL_RED, GL_UNSIGNED_BYTE, view.m_gbuffer.width(), view.m_gbuffer.height() );

        //matrices setuo
        Eigen::Vector2f viewport;
        viewport << view.m_gbuffer.width(), view.m_gbuffer.height();
        MeshSharedPtr mesh = Scene::get_mesh_with_name("copter_transformed");
        Eigen::Matrix4f M=mesh->m_model_matrix.cast<float>().matrix();
        Eigen::Matrix4f V = view.m_camera->view_matrix();
        Eigen::Matrix4f P = view.m_camera->proj_matrix(viewport);
        Eigen::Matrix4f MVP = P*V*M;

        //get the middle point of the copter mesh
        Eigen::Vector3f centroid_object=mesh->centroid().cast<float>(); //centroid of the mesh in the local object coordinate
        Eigen::Vector3f centroid_world= (M*centroid_object.homogeneous()).head(3);
        Eigen::Vector3f centroid_projected=view.m_camera->project(centroid_world, V, P, viewport);
        //project all the vertices
        Eigen::MatrixXf V_projected_2d(mesh->V.rows(),3);
        for(int i=0; i<mesh->V.rows(); i++){
            Eigen::Vector3f point_world=(M*Eigen::Vector3d(mesh->V.row(i)).cast<float>().homogeneous()).head(3);
            Eigen::Vector3f point_projected= view.m_camera->project(point_world, V, P, viewport);
            V_projected_2d.row(i)=point_projected.head(2);
        }
        //get the min of all these projected points (in 2D as we don't care about dealing with a depth buffer) and we get the difference from the furthest point in 2D towards the center in 2D
        Eigen::Vector2f min_point = V_projected_2d.colwise().minCoeff(); 
        Eigen::Vector2f max_point = V_projected_2d.colwise().maxCoeff(); 
        Eigen::Vector2f diff_screen=min_point-centroid_projected.head(2);
        float size_in_pixels=std::sqrt(diff_screen.dot(diff_screen));
        size_in_pixels*=0.5;
        if (!mesh->m_vis.m_is_visible){
            size_in_pixels=0.0;
        }


        // //dont perform depth checking nor write into the depth buffer 
        glDepthMask(false);
        glDisable(GL_DEPTH_TEST);
        glViewport(0.0f , 0.0f, view.m_gbuffer.width(), view.m_gbuffer.height() );

        gl::Shader& shader= m_detect_copter_shader;

        // Set attributes that the vao will pulll from buffers
        GL_C( m_fullscreen_quad->vao.vertex_attribute(shader, "position", m_fullscreen_quad->V_buf, 3) );
        GL_C( m_fullscreen_quad->vao.vertex_attribute(shader, "uv", m_fullscreen_quad->UV_buf, 2) );
        m_fullscreen_quad->vao.indices(m_fullscreen_quad->F_buf); //Says the indices with we refer to vertices, this gives us the triangles
        
        
        //  //shader setup
        GL_C( shader.use() );
        // shader.bind_texture(view.m_gbuffer.tex_with_name("depth_gtex"),"depth_tex");
        // shader.bind_texture(view.m_gbuffer.tex_with_name("mesh_id_gtex"),"mesh_id_tex");
        // shader.uniform_4x4(MVP, "MVP");
        shader.uniform_float(size_in_pixels, "size_in_pixels");
        shader.uniform_v3_float(centroid_projected.head(3), "copter_centroid_projected");

        shader.draw_into(m_copter_blob_tex, "out_color");

        // // draw
        m_fullscreen_quad->vao.bind(); 
        glDrawElements(GL_TRIANGLES, m_fullscreen_quad->m_core->F.size(), GL_UNSIGNED_INT, 0);
        TIME_END("detect_balloon");

        // //restore the state
        glDepthMask(true);
        glEnable(GL_DEPTH_TEST);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);


        // VIEW image 
        // view.m_gui->show_gl_texture(view.m_final_tex.get_tex_id(), "final_tex", true);
        view.m_gui->show_gl_texture(m_copter_blob_tex.tex_id(), "copter_blob_tex", true);


        TIME_END("detect_copter");


        //detect also the bounding box and draw it
        // VLOG(1) << "min_point " << min_point;
        // VLOG(1) << "max_point " << max_point;
        m_bb_mat= cv::Mat(cv::Size(view.m_gbuffer.width(), view.m_gbuffer.height()), CV_32FC1);
        m_bb_mat=0;
        float bb_x, bb_y, bb_width, bb_height;
        bb_x=min_point.x();
        bb_y= view.m_gbuffer.height()-max_point.y(); //we get the max point in y becasue the max is actually the min because opencv considers another origin in the image
        bb_width=max_point.x()-min_point.x();
        bb_height=max_point.y()-min_point.y();
        if (mesh->m_vis.m_is_visible){
            cv::Rect rect(bb_x, bb_y, bb_width, bb_height);
            cv::rectangle(m_bb_mat, rect, cv::Scalar(255, 0, 0), 10);
        }

        Gui::show(m_bb_mat,"m_bb_mat");
        m_last_copter_bb_detected << bb_x, bb_y, bb_width, bb_height;

        if (!mesh->m_vis.m_is_visible){
            m_last_copter_bb_detected.setZero();
        }


    }


}

void SyntheticGenerator::detect_ball(Viewer& view){

    bool exists=Scene::does_mesh_with_name_exist("ball");
    // bool exists=Scene::does_mesh_with_name_exist("copter");
    if(exists){

        TIME_START("detect_ball");

        // //create a final image the same size as the framebuffer
        // m_copter_blob_tex.allocate_or_resize(GL_R8, GL_RED, GL_UNSIGNED_BYTE, view.m_gbuffer.width(), view.m_gbuffer.height() );

        //matrices setuo
        Eigen::Vector2f viewport;
        viewport << view.m_gbuffer.width(), view.m_gbuffer.height();
        MeshSharedPtr mesh = Scene::get_mesh_with_name("ball");
        Eigen::Matrix4f M=mesh->m_model_matrix.cast<float>().matrix();
        Eigen::Matrix4f V = view.m_camera->view_matrix();
        Eigen::Matrix4f P = view.m_camera->proj_matrix(viewport);
        Eigen::Matrix4f MVP = P*V*M;

        //project all the vertices
        Eigen::MatrixXf V_projected_2d(mesh->V.rows(),3);
        for(int i=0; i<mesh->V.rows(); i++){
            Eigen::Vector3f point_world=(M*Eigen::Vector3d(mesh->V.row(i)).cast<float>().homogeneous()).head(3);
            Eigen::Vector3f point_projected= view.m_camera->project(point_world, V, P, viewport);
            V_projected_2d.row(i)=point_projected.head(2);
        }
        //get the min of all these projected points (in 2D as we don't care about dealing with a depth buffer) and we get the difference from the furthest point in 2D towards the center in 2D
        Eigen::Vector2f min_point = V_projected_2d.colwise().minCoeff(); 
        Eigen::Vector2f max_point = V_projected_2d.colwise().maxCoeff(); 




        //detect also the bounding box and draw it
        // VLOG(1) << "min_point " << min_point;
        // VLOG(1) << "max_point " << max_point;
        m_ball_bb_mat= cv::Mat(cv::Size(view.m_gbuffer.width(), view.m_gbuffer.height()), CV_32FC1);
        m_ball_bb_mat=0;
        float bb_x, bb_y, bb_width, bb_height;
        bb_x=min_point.x();
        bb_y= view.m_gbuffer.height()-max_point.y(); //we get the max point in y becasue the max is actually the min because opencv considers another origin in the image
        bb_width=max_point.x()-min_point.x();
        bb_height=max_point.y()-min_point.y();

        //chekc if the ball is within the borders of the image
        bool ball_inside_image=true;
        // if (bb_x > view.m_gbuffer.width() || bb_x<0.0 || bb_y< 0.0)

        //clamps the bounding box so that it is inside the image
        // if(bb_x>view.m_gbuffer.width())
        // bb_x=clamp(bb_x, 0.0f, (float)view.m_gbuffer.width());
        if (bb_x< 0.0){ //exited to the left
            bb_width -= std::fabs(bb_x);
            bb_x=0.0;
        }
        if (bb_x+bb_width > view.m_gbuffer.width()){
            bb_width=view.m_gbuffer.width()-bb_x;
        }
        if (bb_y< 0.0){ //exited from the top
            bb_height -= std::fabs(bb_y);
            bb_y=0.0;
        }
        bb_y=clamp(bb_y, 0.0f, (float)view.m_gbuffer.height());
        if (bb_y+bb_height > view.m_gbuffer.height()){
            bb_height=view.m_gbuffer.height()-bb_y;
        }

        //if the width and height are zero then we have colapsed the bounding box to nothing
        if (bb_width<0.001 || bb_height<0.001){
            ball_inside_image=false;
        }



        if (mesh->m_vis.m_is_visible && ball_inside_image){
            cv::Rect rect(bb_x, bb_y, bb_width, bb_height);
            cv::rectangle(m_ball_bb_mat, rect, cv::Scalar(255, 0, 0), 10);
        }

        Gui::show(m_ball_bb_mat,"m_ball_bb_mat");
        m_last_ball_bb_detected << bb_x, bb_y, bb_width, bb_height;

        if (!mesh->m_vis.m_is_visible || !ball_inside_image){
            m_last_ball_bb_detected.setZero();
        }
        // VLOG(1) << m_last_ball_bb_detected.transpose();

        TIME_END("detect_ball");

    }


}

void SyntheticGenerator::record_data(Viewer& view){

    // VLOG(1) << "recoridng";

    //copter
    int idx;
    if (m_write_with_random_idx){
        idx=m_rand_gen->rand_int(0,99999999);
    }else{
        idx=m_iter_write;
    }
    m_iter_write++;

    //skip the first few recording so that the user has some time to resize the window
    if (m_iter_write<50){
        return;
    }

    // m_recorder->record(view.m_composed_tex, std::to_string(idx)+".jpeg", m_rgb_output_path);
    m_recorder->write_without_buffering(view.m_composed_tex, std::to_string(idx)+".jpeg", m_rgb_output_path);
    bool recorded_copter_gt=false;
    bool recorded_balloon_gt=false;
    if(m_copter_blob_tex.storage_initialized()){
        // recorded_copter_gt=m_recorder->record(m_copter_blob_tex, std::to_string(idx)+".jpeg", m_gt_output_path);
        m_recorder->write_without_buffering(m_copter_blob_tex, std::to_string(idx)+".jpeg", m_gt_output_path); //we write without buffering, otherwise the m_last_bb_detected wont correspond with the 3 frames delayed PBO that the record function would write
        // cv::imwrite( "/media/rosu/Data/phd/c_ws/src/mbzirc_2020/challenge_1_synthetic_data/recordings/copter_bb/"+std::to_string(idx)+".jpeg", m_bb_mat );
        // cv::imwrite( "/media/rosu/Data/phd/c_ws/src/mbzirc_2020/challenge_1_synthetic_data/recordings/copter_bb/"+std::to_string(idx)+"ball_.jpeg", m_ball_bb_mat );
        recorded_copter_gt=true;
    }
    if(m_balloon_outline_tex.storage_initialized()){
        // recorded_balloon_gt=m_recorder->record(m_balloon_outline_tex, std::to_string(idx)+".jpeg", m_gt_output_path);
        m_recorder->write_without_buffering(m_balloon_outline_tex, std::to_string(idx)+".jpeg", m_gt_output_path);
        recorded_balloon_gt=true;
    }


    //record json mostly following the guide in https://www.youtube.com/watch?v=h6s61a_pqfM

    //make a json object for the image ONLY FOR THE COPTER
    if(recorded_copter_gt){
        json im;
        im["id"]=idx;
        im["width"]=m_bb_mat.cols;
        im["height"]=m_bb_mat.rows;
        im["file_name"]=std::to_string(idx)+".jpeg";
        m_json_file["images"].push_back(im);

        //make annotation for this image
        if (!m_last_copter_bb_detected.isZero()){
            json annotation;
            annotation["id"]=idx; //not really correct but this field will probably nto get used either way
            annotation["category_id"]=2; //this is the category for the copter
            annotation["iscrowd"]=0;
            annotation["image_id"]=idx;
            annotation["bbox"]={m_last_copter_bb_detected[0], m_last_copter_bb_detected[1], m_last_copter_bb_detected[2], m_last_copter_bb_detected[3]};
            m_json_file["annotations"].push_back(annotation);
        }
        if (!m_last_ball_bb_detected.isZero()){
            json annotation;
            annotation["id"]=idx; //not really correct but this field will probably nto get used either way
            annotation["category_id"]=3; //this is the category for the ball
            annotation["iscrowd"]=0;
            annotation["image_id"]=idx;
            annotation["bbox"]={m_last_ball_bb_detected[0], m_last_ball_bb_detected[1], m_last_ball_bb_detected[2], m_last_ball_bb_detected[3]};
            m_json_file["annotations"].push_back(annotation);
        }
       


    }


}

void SyntheticGenerator::write_to_json(Viewer& view){


    // m_json_file["images"].push_back("foo");

}
