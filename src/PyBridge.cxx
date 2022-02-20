#include "data_loaders/PyBridge.h"


#ifdef WITH_TORCH
    #include <torch/extension.h>
    #include "torch/torch.h"
    #include "torch/csrc/utils/pybind.h"
#endif

// #include "pybind_casters/pybind11_opencv.hpp"
// #include "pybind_casters/ndarray_converter.h"

//my stuff
#include "data_loaders/DataLoaderShapeNetPartSeg.h"
#include "data_loaders/DataLoaderShapeNetImg.h"
#include "data_loaders/DataLoaderVolRef.h"
#include "data_loaders/DataLoaderStanford3DScene.h"
#include "data_loaders/DataLoaderImg.h"
#include "data_loaders/DataLoaderSemanticKitti.h"
#include "data_loaders/DataLoaderPheno4D.h"
#include "data_loaders/DataLoaderPhenorobCP1.h"
#include "data_loaders/DataLoaderScanNet.h"
#include "data_loaders/DataLoaderNerf.h"
#include "data_loaders/DataLoaderEasyPBR.h"
#include "data_loaders/DataLoaderColmap.h"
#include "data_loaders/DataLoaderSRN.h"
#include "data_loaders/DataLoaderDTU.h"
#include "data_loaders/DataLoaderDeepVoxels.h"
#include "data_loaders/DataLoaderLLFF.h"
//fb
#include "data_loaders/fb/DataLoaderBlenderFB.h"
#ifdef WITH_TORCH
    #include "data_loaders/fb/DataLoaderUSCHair.h"
#endif
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

    //DataLoaderShapeNetImg
    py::class_<DataLoaderShapeNetImg> (m, "DataLoaderShapeNetImg")
    .def(py::init<const std::string>())
    .def("get_random_frame", &DataLoaderShapeNetImg::get_random_frame )
    .def("get_frame_at_idx", &DataLoaderShapeNetImg::get_frame_at_idx )
    .def("start_reading_next_scene", &DataLoaderShapeNetImg::start_reading_next_scene )
    .def("finished_reading_scene", &DataLoaderShapeNetImg::finished_reading_scene )
    .def("has_data", &DataLoaderShapeNetImg::has_data )
    .def("is_finished", &DataLoaderShapeNetImg::is_finished )
    .def("reset", &DataLoaderShapeNetImg::reset )
    .def("nr_samples", &DataLoaderShapeNetImg::nr_samples )
    ;

    py::class_<DataLoaderVolRef> (m, "DataLoaderVolRef")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderVolRef::start )
    .def("get_frame_at_idx", &DataLoaderVolRef::get_frame_at_idx )
    .def("get_depth_frame_at_idx", &DataLoaderVolRef::get_depth_frame_at_idx )
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
    .def("set_overfit", &DataLoaderVolRef::set_overfit )
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

    //DataLoaderPheno4D
    py::class_<DataLoaderPheno4D> (m, "DataLoaderPheno4D")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderPheno4D::start )
    .def("get_cloud", &DataLoaderPheno4D::get_cloud, R"EOS( get_cloud. )EOS" )
    .def("get_cloud_with_idx", &DataLoaderPheno4D::get_cloud_with_idx )
    .def("has_data", &DataLoaderPheno4D::has_data )
    .def("is_finished", &DataLoaderPheno4D::is_finished )
    .def("is_finished_reading", &DataLoaderPheno4D::is_finished_reading )
    .def("reset", &DataLoaderPheno4D::reset )
    .def("nr_samples", &DataLoaderPheno4D::nr_samples )
    .def("label_mngr", &DataLoaderPheno4D::label_mngr )
    .def("set_plant_nr", &DataLoaderPheno4D::set_plant_nr )
    .def("set_nr_plants_to_skip", &DataLoaderPheno4D::set_nr_plants_to_skip )
    .def("set_nr_plants_to_read", &DataLoaderPheno4D::set_nr_plants_to_read )
    .def("set_nr_days_to_skip", &DataLoaderPheno4D::set_nr_days_to_skip )
    .def("set_nr_days_to_read", &DataLoaderPheno4D::set_nr_days_to_read )
    .def("set_day", &DataLoaderPheno4D::set_day )
    .def("set_do_augmentation", &DataLoaderPheno4D::set_do_augmentation )
    .def("set_segmentation_method", &DataLoaderPheno4D::set_segmentation_method )
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

    //DataLoaderNerf
    py::class_<DataLoaderNerf> (m, "DataLoaderNerf")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderNerf::start )
    .def("has_data", &DataLoaderNerf::has_data )
    .def("get_next_frame", &DataLoaderNerf::get_next_frame )
    .def("get_all_frames", &DataLoaderNerf::get_all_frames )
    .def("get_frame_at_idx", &DataLoaderNerf::get_frame_at_idx )
    .def("get_random_frame", &DataLoaderNerf::get_random_frame )
    .def("get_closest_frame", &DataLoaderNerf::get_closest_frame )
    .def("get_close_frames", &DataLoaderNerf::get_close_frames )
    // .def("compute_frame_weights", &DataLoaderNerf::compute_frame_weights )
    .def("is_finished", &DataLoaderNerf::is_finished )
    .def("reset", &DataLoaderNerf::reset )
    .def("nr_samples", &DataLoaderNerf::nr_samples )
    .def("subsample_factor", &DataLoaderNerf::subsample_factor )
    .def("set_mode_train", &DataLoaderNerf::set_mode_train )
    .def("set_mode_test", &DataLoaderNerf::set_mode_test )
    .def("set_mode_validation", &DataLoaderNerf::set_mode_validation )
    ;

    //DataLoaderEasyPBR
    py::class_<DataLoaderEasyPBR> (m, "DataLoaderEasyPBR")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderEasyPBR::start )
    .def("has_data", &DataLoaderEasyPBR::has_data )
    .def("get_next_frame", &DataLoaderEasyPBR::get_next_frame )
    .def("get_all_frames", &DataLoaderEasyPBR::get_all_frames )
    .def("get_frame_at_idx", &DataLoaderEasyPBR::get_frame_at_idx )
    .def("get_random_frame", &DataLoaderEasyPBR::get_random_frame )
    .def("get_closest_frame", &DataLoaderEasyPBR::get_closest_frame )
    .def("get_close_frames", &DataLoaderEasyPBR::get_close_frames )
    // .def("compute_frame_weights", &DataLoaderNerf::compute_frame_weights )
    .def("loaded_scene_mesh", &DataLoaderEasyPBR::loaded_scene_mesh )
    .def("get_scene_mesh", &DataLoaderEasyPBR::get_scene_mesh )
    .def("is_finished", &DataLoaderEasyPBR::is_finished )
    .def("reset", &DataLoaderEasyPBR::reset )
    .def("nr_samples", &DataLoaderEasyPBR::nr_samples )
    .def("set_mode_train", &DataLoaderEasyPBR::set_mode_train )
    .def("set_mode_test", &DataLoaderEasyPBR::set_mode_test )
    .def("set_mode_validation", &DataLoaderEasyPBR::set_mode_validation )
    .def("set_shuffle", &DataLoaderEasyPBR::set_shuffle )
    ;


    //DataLoaderColmap
    py::class_<DataLoaderColmap> (m, "DataLoaderColmap")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderColmap::start )
    .def("has_data", &DataLoaderColmap::has_data )
    .def("get_next_frame", &DataLoaderColmap::get_next_frame )
    .def("get_frame_at_idx", &DataLoaderColmap::get_frame_at_idx )
    .def("get_random_frame", &DataLoaderColmap::get_random_frame )
    .def("get_closest_frame", &DataLoaderColmap::get_closest_frame )
    .def("get_close_frames", &DataLoaderColmap::get_close_frames )
    .def("is_finished", &DataLoaderColmap::is_finished )
    .def("reset", &DataLoaderColmap::reset )
    .def("nr_samples", &DataLoaderColmap::nr_samples )
    .def("set_mode_train", &DataLoaderColmap::set_mode_train )
    .def("set_mode_test", &DataLoaderColmap::set_mode_test )
    .def("set_mode_validation", &DataLoaderColmap::set_mode_validation )
    .def("set_mode_all", &DataLoaderColmap::set_mode_all )
    ;

    //DataLoaderSRN
    py::class_<DataLoaderSRN> (m, "DataLoaderSRN")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderSRN::start )
    .def("get_random_frame", &DataLoaderSRN::get_random_frame )
    .def("get_frame_at_idx", &DataLoaderSRN::get_frame_at_idx )
    .def("start_reading_next_scene", &DataLoaderSRN::start_reading_next_scene )
    .def("finished_reading_scene", &DataLoaderSRN::finished_reading_scene )
    .def("has_data", &DataLoaderSRN::has_data )
    .def("is_finished", &DataLoaderSRN::is_finished )
    .def("reset", &DataLoaderSRN::reset )
    .def("nr_samples", &DataLoaderSRN::nr_samples )
    .def("set_mode_train", &DataLoaderSRN::set_mode_train )
    .def("set_mode_test", &DataLoaderSRN::set_mode_test )
    .def("set_mode_validation", &DataLoaderSRN::set_mode_validation )
    ;

    //DataLoaderDTU
    py::class_<DataLoaderDTU> (m, "DataLoaderDTU")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderDTU::start )
    .def("get_random_frame", &DataLoaderDTU::get_random_frame )
    .def("get_frame_at_idx", &DataLoaderDTU::get_frame_at_idx )
    .def("start_reading_next_scene", &DataLoaderDTU::start_reading_next_scene )
    .def("finished_reading_scene", &DataLoaderDTU::finished_reading_scene )
    .def("has_data", &DataLoaderDTU::has_data )
    .def("is_finished", &DataLoaderDTU::is_finished )
    .def("reset", &DataLoaderDTU::reset )
    .def("nr_samples", &DataLoaderDTU::nr_samples )
    .def("nr_scenes", &DataLoaderDTU::nr_scenes )
    .def("set_restrict_to_scan_idx", &DataLoaderDTU::set_restrict_to_scan_idx )
    .def("set_mode_train", &DataLoaderDTU::set_mode_train )
    .def("set_mode_test", &DataLoaderDTU::set_mode_test )
    .def("set_mode_validation", &DataLoaderDTU::set_mode_validation )
    ;

    //DataLoaderDeepVoxels
    py::class_<DataLoaderDeepVoxels> (m, "DataLoaderDeepVoxels")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderDeepVoxels::start )
    .def("has_data", &DataLoaderDeepVoxels::has_data )
    .def("get_next_frame", &DataLoaderDeepVoxels::get_next_frame )
    .def("get_all_frames", &DataLoaderDeepVoxels::get_all_frames )
    .def("get_frame_at_idx", &DataLoaderDeepVoxels::get_frame_at_idx )
    .def("get_random_frame", &DataLoaderDeepVoxels::get_random_frame )
    .def("get_closest_frame", &DataLoaderDeepVoxels::get_closest_frame )
    .def("get_close_frames", &DataLoaderDeepVoxels::get_close_frames )
    // .def("compute_frame_weights", &DataLoaderNerf::compute_frame_weights )
    .def("is_finished", &DataLoaderDeepVoxels::is_finished )
    .def("reset", &DataLoaderDeepVoxels::reset )
    .def("nr_samples", &DataLoaderDeepVoxels::nr_samples )
    .def("set_mode_train", &DataLoaderDeepVoxels::set_mode_train )
    .def("set_mode_test", &DataLoaderDeepVoxels::set_mode_test )
    .def("set_mode_validation", &DataLoaderDeepVoxels::set_mode_validation )
    ;


    //DataLoaderLLFF
    py::class_<DataLoaderLLFF> (m, "DataLoaderLLFF")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderLLFF::start )
    .def("has_data", &DataLoaderLLFF::has_data )
    .def("get_next_frame", &DataLoaderLLFF::get_next_frame )
    .def("get_frame_at_idx", &DataLoaderLLFF::get_frame_at_idx )
    .def("get_random_frame", &DataLoaderLLFF::get_random_frame )
    .def("get_closest_frame", &DataLoaderLLFF::get_closest_frame )
    .def("get_close_frames", &DataLoaderLLFF::get_close_frames )
    .def("is_finished", &DataLoaderLLFF::is_finished )
    .def("reset", &DataLoaderLLFF::reset )
    .def("nr_samples", &DataLoaderLLFF::nr_samples )
    .def("set_mode_train", &DataLoaderLLFF::set_mode_train )
    .def("set_mode_test", &DataLoaderLLFF::set_mode_test )
    .def("set_mode_validation", &DataLoaderLLFF::set_mode_validation )
    .def("set_mode_all", &DataLoaderLLFF::set_mode_all )
    ;

    py::class_<DataLoaderPhenorobCP1> (m, "DataLoaderPhenorobCP1")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderPhenorobCP1::start )
    .def("has_data", &DataLoaderPhenorobCP1::has_data )
    .def("get_scan_with_idx", &DataLoaderPhenorobCP1::get_scan_with_idx )
    .def("load_mesh", &DataLoaderPhenorobCP1::load_mesh )
    // .def("get_next_frame", &DataLoaderPhenorobCP1::get_next_frame )
    // .def("get_all_frames", &DataLoaderPhenorobCP1::get_all_frames )
    // .def("get_frame_at_idx", &DataLoaderPhenorobCP1::get_frame_at_idx )
    // .def("get_random_frame", &DataLoaderPhenorobCP1::get_random_frame )
    // .def("get_closest_frame", &DataLoaderPhenorobCP1::get_closest_frame )
    // .def("get_close_frames", &DataLoaderPhenorobCP1::get_close_frames )
    // .def("compute_frame_weights", &DataLoaderNerf::compute_frame_weights )
    .def("dataset_path", &DataLoaderPhenorobCP1::dataset_path )
    .def("scan_date", &DataLoaderPhenorobCP1::scan_date )
    .def("rgb_pose_file", &DataLoaderPhenorobCP1::rgb_pose_file )
    .def("is_finished", &DataLoaderPhenorobCP1::is_finished )
    .def("reset", &DataLoaderPhenorobCP1::reset )
    .def("nr_scans", &DataLoaderPhenorobCP1::nr_scans )
    .def("loaded_dense_cloud", &DataLoaderPhenorobCP1::loaded_dense_cloud )
    .def("set_mode_train", &DataLoaderPhenorobCP1::set_mode_train )
    .def("set_mode_test", &DataLoaderPhenorobCP1::set_mode_test )
    .def("set_mode_validation", &DataLoaderPhenorobCP1::set_mode_validation )
    .def("set_mode_all", &DataLoaderPhenorobCP1::set_mode_all )
    ;
    py::class_<PRCP1Scan, std::shared_ptr<PRCP1Scan> > (m, "PRCP1Scan")
    .def("nr_blocks", &PRCP1Scan::nr_blocks )
    .def("get_block_with_idx", &PRCP1Scan::get_block_with_idx )
    .def("name", &PRCP1Scan::name )
    ;
    py::class_<PRCP1Block, std::shared_ptr<PRCP1Block> > (m, "PRCP1Block")
    .def("nr_frames", &PRCP1Block::nr_frames )
    .def("get_rgb_frame_with_idx", &PRCP1Block::get_rgb_frame_with_idx )
    .def("get_photoneo_frame", &PRCP1Block::get_photoneo_frame )
    .def("get_photoneo_mesh", &PRCP1Block::get_photoneo_mesh )
    .def("get_dense_cloud", &PRCP1Block::get_dense_cloud )
    .def("get_sparse_cloud", &PRCP1Block::get_sparse_cloud )
    .def("name", &PRCP1Block::name )
    ;



    //fb
    //DataLoaderBlender
    py::class_<DataLoaderBlenderFB> (m, "DataLoaderBlenderFB")
    .def(py::init<const std::string>())
    .def("start", &DataLoaderBlenderFB::start )
    .def("has_data", &DataLoaderBlenderFB::has_data )
    .def("get_next_frame", &DataLoaderBlenderFB::get_next_frame )
    .def("get_all_frames", &DataLoaderBlenderFB::get_all_frames )
    .def("get_frame_at_idx", &DataLoaderBlenderFB::get_frame_at_idx )
    .def("get_random_frame", &DataLoaderBlenderFB::get_random_frame )
    .def("get_closest_frame", &DataLoaderBlenderFB::get_closest_frame )
    .def("get_close_frames", &DataLoaderBlenderFB::get_close_frames )
    // .def("compute_frame_weights", &DataLoaderNerf::compute_frame_weights )
    .def("is_finished", &DataLoaderBlenderFB::is_finished )
    .def("reset", &DataLoaderBlenderFB::reset )
    .def("nr_samples", &DataLoaderBlenderFB::nr_samples )
    .def("set_mode_train", &DataLoaderBlenderFB::set_mode_train )
    .def("set_mode_test", &DataLoaderBlenderFB::set_mode_test )
    .def("set_mode_validation", &DataLoaderBlenderFB::set_mode_validation )
    ;



    #ifdef WITH_TORCH
        //DataLoaderUSCHair
        py::class_<DataLoaderUSCHair> (m, "DataLoaderUSCHair")
        .def(py::init<const std::string>())
        .def("start", &DataLoaderUSCHair::start )
        .def("get_hair", &DataLoaderUSCHair::get_hair )
        // .def("get_cloud", &DataLoaderUSCHair::get_cloud, R"EOS( get_cloud. )EOS" )
        .def("get_mesh_head", &DataLoaderUSCHair::get_mesh_head )
        .def("get_mesh_scalp", &DataLoaderUSCHair::get_mesh_scalp )
        .def("has_data", &DataLoaderUSCHair::has_data )
        .def("is_finished", &DataLoaderUSCHair::is_finished )
        .def("is_finished_reading", &DataLoaderUSCHair::is_finished_reading )
        .def("reset", &DataLoaderUSCHair::reset )
        .def("nr_samples", &DataLoaderUSCHair::nr_samples )
        .def("set_mode_train", &DataLoaderUSCHair::set_mode_train )
        .def("set_mode_test", &DataLoaderUSCHair::set_mode_test )
        .def("set_mode_validation", &DataLoaderUSCHair::set_mode_validation )
        .def("get_random_roots", &DataLoaderUSCHair::get_random_roots )
        ;
        //USCHair
        py::class_<USCHair, std::shared_ptr<USCHair> > (m, "USCHair")
        .def_readwrite("full_hair_cloud", &USCHair::full_hair_cloud)
        .def_readwrite("strand_meshes", &USCHair::strand_meshes)
        // .def_readwrite("points", &USCHair::points)
        // .def_readwrite("points_tensor", &USCHair::points_tensor)
        // .def_readwrite("per_point_strand_idx", &USCHair::per_point_strand_idx)
        .def_readwrite("uv_roots", &USCHair::uv_roots)
        .def_readwrite("tbn_roots_tensor", &USCHair::tbn_roots_tensor)
        .def_readwrite("position_roots", &USCHair::position_roots)
        // .def_readwrite("strand_lengths", &USCHair::strand_lengths)
        // .def_readwrite("per_strand_R_rodri_canonical_scalp", &USCHair::per_strand_R_rodri_canonical_scalp)
        // .def_readwrite("per_strand_dir_along", &USCHair::per_strand_dir_along)
        // .def_readwrite("full_hair_cumulative_strand_length", &USCHair::full_hair_cumulative_strand_length)
        // .def_readwrite("per_point_rotation_next_cur_tensor", &USCHair::per_point_rotation_next_cur_tensor)
        // .def_readwrite("per_point_delta_dist_tensor", &USCHair::per_point_delta_dist_tensor)
        .def_readwrite("per_point_direction_to_next_tensor", &USCHair::per_point_direction_to_next_tensor)
        // .def_readwrite("per_strand_R_rodri_across_canonical", &USCHair::per_strand_R_rodri_across_canonical)
        // .def_readwrite("per_strand_across_canonical_weight", &USCHair::per_strand_across_canonical_weight)
        // .def_readwrite("per_strand_dir_across", &USCHair::per_strand_dir_across)
        ;
    #endif



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
        .def("is_loader_thread_alive", &DataLoaderCloudRos::is_loader_thread_alive )
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
