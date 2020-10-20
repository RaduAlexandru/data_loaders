#include "data_loaders/PyBridge.h"

// #include <torch/extension.h>
// #include "torch/torch.h"
// #include "torch/csrc/utils/pybind.h"

// #include "pybind_casters/pybind11_opencv.hpp"
// #include "pybind_casters/ndarray_converter.h"

//my stuff 
#include "data_loaders/DataLoaderShapeNetPartSeg.h"
#include "data_loaders/DataLoaderVolRef.h"
#include "data_loaders/DataLoaderStanford3DScene.h"
#include "data_loaders/DataLoaderImg.h"
#include "data_loaders/DataLoaderSemanticKitti.h"
#include "data_loaders/DataLoaderPhenorob.h"
#include "data_loaders/DataLoaderScanNet.h"
#include "easy_pbr/Mesh.h"
#include "easy_pbr/LabelMngr.h"

#ifdef WITH_ROS
    #include "data_loaders/DataLoaderImgRos.h"
    #include "data_loaders/DataLoaderCloudRos.h"
    #include "data_loaders/RosBagPlayer.h"
#endif


// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// PYBIND11_MAKE_OPAQUE(std::vector<int>); //to be able to pass vectors by reference to functions and have things like push back actually work 
// PYBIND11_MAKE_OPAQUE(std::vector<float>, std::allocator<float> >);

namespace py = pybind11;

using namespace easy_pbr;


PYBIND11_MODULE(dataloaders, m) {

    // NDArrayConverter::init_numpy();

    // py::module::import("easypbr");
    // py::object mesh_py = (py::object) py::module::import("easypbr").attr("Mesh");
    // py::print(Mesh.attr("path"));
    // py::object mesh=py::module::import("easypbr").attr("Mesh");;
    // py::class_<cv::Mat> (m, "Mat")
    // ;

    // //Frame
    // py::class_<Frame> (m, "Frame")
    // .def(py::init<>())
    // // .def_readwrite("rgb_32f", &Frame::rgb_32f) //not possible in pybind. You would need to wrap the opencv into a matrix type or soemthing like that
    // .def("create_frustum_mesh", &Frame::create_frustum_mesh, py::arg("scale_multiplier") = 1.0)
    // .def("rotate_y_axis", &Frame::rotate_y_axis )
    // .def("backproject_depth", &Frame::backproject_depth )
    // .def("assign_color", &Frame::assign_color )
    // .def_readwrite("rgb_8u", &Frame::rgb_8u )
    // .def_readwrite("rgb_32f", &Frame::rgb_32f )
    // // #ifdef WITH_TORCH
    //     // .def("rgb2tensor", &Frame::rgb2tensor )
    //     // .def("tensor2rgb", &Frame::tensor2rgb )
    // // #endif
    // ;
 
    //DataLoader ShapeNetPartSeg
    py::class_<DataLoaderShapeNetPartSeg> (m, "DataLoaderShapeNetPartSeg")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderShapeNetPartSeg::start )
    .def("get_cloud", &DataLoaderShapeNetPartSeg::get_cloud )
    .def("has_data", &DataLoaderShapeNetPartSeg::has_data ) 
    .def("is_finished", &DataLoaderShapeNetPartSeg::is_finished ) 
    .def("is_finished_reading", &DataLoaderShapeNetPartSeg::is_finished_reading ) 
    .def("reset", &DataLoaderShapeNetPartSeg::reset ) 
    .def("nr_samples", &DataLoaderShapeNetPartSeg::nr_samples ) 
    .def("label_mngr", &DataLoaderShapeNetPartSeg::label_mngr ) 
    .def("set_mode_train", &DataLoaderShapeNetPartSeg::set_mode_train ) 
    .def("set_mode_test", &DataLoaderShapeNetPartSeg::set_mode_test ) 
    .def("set_mode_validation", &DataLoaderShapeNetPartSeg::set_mode_validation ) 
    .def("get_object_name", &DataLoaderShapeNetPartSeg::get_object_name ) 
    .def("set_object_name", &DataLoaderShapeNetPartSeg::set_object_name ) 
    ;

    py::class_<DataLoaderVolRef> (m, "DataLoaderVolRef")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderVolRef::start )
    .def("get_color_frame", &DataLoaderVolRef::get_color_frame )
    .def("get_depth_frame", &DataLoaderVolRef::get_depth_frame )
    .def("has_data", &DataLoaderVolRef::has_data ) 
    .def("is_finished", &DataLoaderVolRef::is_finished ) 
    .def("is_finished_reading", &DataLoaderVolRef::is_finished_reading ) 
    .def("reset", &DataLoaderVolRef::reset ) 
    .def("nr_samples", &DataLoaderVolRef::nr_samples ) 
    .def("closest_color_frame", &DataLoaderVolRef::closest_color_frame ) 
    .def("closest_depth_frame", &DataLoaderVolRef::closest_depth_frame ) 
    .def("load_only_from_idxs", &DataLoaderVolRef::load_only_from_idxs ) 
    .def("set_shuffle", &DataLoaderVolRef::set_shuffle ) 
    ;

    py::class_<DataLoaderStanford3DScene> (m, "DataLoaderStanford3DScene")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderStanford3DScene::start )
    .def("get_color_frame", &DataLoaderStanford3DScene::get_color_frame )
    .def("get_depth_frame", &DataLoaderStanford3DScene::get_depth_frame )
    .def("has_data", &DataLoaderStanford3DScene::has_data ) 
    .def("is_finished", &DataLoaderStanford3DScene::is_finished ) 
    .def("is_finished_reading", &DataLoaderStanford3DScene::is_finished_reading ) 
    .def("reset", &DataLoaderStanford3DScene::reset ) 
    .def("nr_samples", &DataLoaderStanford3DScene::nr_samples ) 
    // .def("closest_color_frame", &DataLoaderStanford3DScene::closest_color_frame ) 
    // .def("closest_depth_frame", &DataLoaderStanford3DScene::closest_depth_frame ) 
    ;

    py::class_<DataLoaderImg> (m, "DataLoaderImg")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderImg::start )
    .def("get_frame_for_cam", &DataLoaderImg::get_frame_for_cam )
    .def("get_nr_cams", &DataLoaderImg::get_nr_cams )
    .def("has_data_for_cam", &DataLoaderImg::has_data_for_cam ) 
    .def("has_data_for_all_cams", &DataLoaderImg::has_data_for_all_cams ) 
    .def("is_finished", &DataLoaderImg::is_finished ) 
    .def("is_finished_reading", &DataLoaderImg::is_finished_reading ) 
    .def("reset", &DataLoaderImg::reset ) 
    .def("nr_samples_for_cam", &DataLoaderImg::nr_samples_for_cam ) 
    ;

    //DataLoaderSemanticKitti
    py::class_<DataLoaderSemanticKitti> (m, "DataLoaderSemanticKitti")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderSemanticKitti::start )
    .def("get_cloud", &DataLoaderSemanticKitti::get_cloud, R"EOS( get_cloud. )EOS" )
    .def("has_data", &DataLoaderSemanticKitti::has_data )
    .def("is_finished", &DataLoaderSemanticKitti::is_finished ) 
    .def("is_finished_reading", &DataLoaderSemanticKitti::is_finished_reading ) 
    .def("reset", &DataLoaderSemanticKitti::reset ) 
    .def("nr_samples", &DataLoaderSemanticKitti::nr_samples ) 
    .def("label_mngr", &DataLoaderSemanticKitti::label_mngr ) 
    .def("set_mode_train", &DataLoaderSemanticKitti::set_mode_train ) 
    .def("set_mode_test", &DataLoaderSemanticKitti::set_mode_test ) 
    .def("set_mode_validation", &DataLoaderSemanticKitti::set_mode_validation ) 
    .def("set_sequence", &DataLoaderSemanticKitti::set_sequence ) 
    // .def("set_adaptive_subsampling", &DataLoaderSemanticKitti::set_adaptive_subsampling ) 
    ;

    //DataLoaderPhenorob
    py::class_<DataLoaderPhenorob> (m, "DataLoaderPhenorob")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderPhenorob::start )
    .def("get_cloud", &DataLoaderPhenorob::get_cloud, R"EOS( get_cloud. )EOS" )
    .def("has_data", &DataLoaderPhenorob::has_data )
    .def("is_finished", &DataLoaderPhenorob::is_finished ) 
    .def("is_finished_reading", &DataLoaderPhenorob::is_finished_reading ) 
    .def("reset", &DataLoaderPhenorob::reset ) 
    .def("nr_samples", &DataLoaderPhenorob::nr_samples ) 
    .def("label_mngr", &DataLoaderPhenorob::label_mngr ) 
    .def("set_nr_clouds_to_skip", &DataLoaderPhenorob::set_nr_clouds_to_skip ) 
    .def("set_nr_clouds_to_read", &DataLoaderPhenorob::set_nr_clouds_to_read ) 
    ;

    py::class_<DataLoaderScanNet> (m, "DataLoaderScanNet")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderScanNet::start )
    .def("get_cloud", &DataLoaderScanNet::get_cloud )
    .def("has_data", &DataLoaderScanNet::has_data ) 
    .def("is_finished", &DataLoaderScanNet::is_finished ) 
    .def("is_finished_reading", &DataLoaderScanNet::is_finished_reading ) 
    .def("reset", &DataLoaderScanNet::reset ) 
    .def("nr_samples", &DataLoaderScanNet::nr_samples ) 
    .def("label_mngr", &DataLoaderScanNet::label_mngr ) 
    .def("set_mode_train", &DataLoaderScanNet::set_mode_train ) 
    .def("set_mode_test", &DataLoaderScanNet::set_mode_test ) 
    .def("set_mode_validation", &DataLoaderScanNet::set_mode_validation ) 
    .def("write_for_evaluating_on_scannet_server", &DataLoaderScanNet::write_for_evaluating_on_scannet_server ) 
    ;

    #ifdef WITH_ROS
        py::class_<DataLoaderImgRos> (m, "DataLoaderImgRos")
        .def(py::init<const std::string>())
        .def("get_frame_for_cam", &DataLoaderImgRos::get_frame_for_cam )
        .def("nr_cams", &DataLoaderImgRos::nr_cams ) 
        .def("has_data_for_all_cams", &DataLoaderImgRos::has_data_for_all_cams ) 
        .def("has_data_for_cam", &DataLoaderImgRos::has_data_for_cam ) 
        .def("is_loader_thread_alive", &DataLoaderImgRos::is_loader_thread_alive ) 
        ;

        py::class_<DataLoaderCloudRos> (m, "DataLoaderCloudRos")
        .def(py::init<const std::string>())
        .def("has_data", &DataLoaderCloudRos::has_data )
        .def("get_cloud", &DataLoaderCloudRos::get_cloud )
        ;

        py::class_<RosBagPlayer, std::shared_ptr<RosBagPlayer> > (m, "RosBagPlayer")
        .def_static("create",  &RosBagPlayer::create<const std::string>  ) 
        .def("start", &RosBagPlayer::start )
        .def("play", &RosBagPlayer::play )
        .def("pause", &RosBagPlayer::pause )
        .def("reset", &RosBagPlayer::reset ) 
        .def("is_paused", &RosBagPlayer::is_paused ) 
        .def("is_finished", &RosBagPlayer::is_finished ) 
        .def("kill", &RosBagPlayer::kill ) 
        ;

    #endif

    //  py::class_<Mesh, std::shared_ptr<Mesh>> (m, "Mesh")
    // .def(py::init<>())
    // .def(py::init<std::string>())
    // .def("load_from_file", &Mesh::load_from_file )
    // .def("save_to_file", &Mesh::save_to_file )
    // .def("clone", &Mesh::clone )
    // .def("add", &Mesh::add )
    // .def("is_empty", &Mesh::is_empty )
    // .def("create_box_ndc", &Mesh::create_box_ndc )
    // .def("create_floor", &Mesh::create_floor )
    // .def_readwrite("id", &Mesh::id)
    // .def_readwrite("name", &Mesh::name)
    // .def_readwrite("m_width", &Mesh::m_width)
    // .def_readwrite("m_height", &Mesh::m_height)
    // .def_readwrite("m_vis", &Mesh::m_vis)
    // .def_readwrite("m_force_vis_update", &Mesh::m_force_vis_update)
    // .def_readwrite("V", &Mesh::V)
    // .def_readwrite("F", &Mesh::F)
    // .def_readwrite("C", &Mesh::C)
    // .def_readwrite("E", &Mesh::E)
    // .def_readwrite("D", &Mesh::D)
    // .def_readwrite("NF", &Mesh::NF)
    // .def_readwrite("NV", &Mesh::NV)
    // .def_readwrite("UV", &Mesh::UV)
    // .def_readwrite("V_tangent_u", &Mesh::V_tangent_u)
    // .def_readwrite("V_lenght_v", &Mesh::V_length_v)
    // .def_readwrite("L_pred", &Mesh::L_pred)
    // .def_readwrite("L_gt", &Mesh::L_gt)
    // .def_readwrite("I", &Mesh::I)
    // .def_readwrite("m_label_mngr", &Mesh::m_label_mngr )
    // .def_readwrite("m_min_max_y_for_plotting", &Mesh::m_min_max_y_for_plotting )
    // .def("translate_model_matrix", &Mesh::translate_model_matrix )
    // .def("rotate_model_matrix", &Mesh::rotate_model_matrix )
    // .def("rotate_model_matrix_local", &Mesh::rotate_model_matrix_local )
    // // .def("rotate_x_axis", &Mesh::rotate_x_axis )
    // // .def("rotate_y_axis", &Mesh::rotate_y_axis )
    // .def("random_subsample", &Mesh::random_subsample )
    // // .def("move_in_x", &Mesh::move_in_x )
    // // .def("move_in_y", &Mesh::move_in_y )
    // // .def("move_in_z", &Mesh::move_in_z )
    // .def("add_child", &Mesh::add_child )
    // ;


}