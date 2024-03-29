#!/usr/bin/env python3

import os
import numpy as np
import sys
try:
  import torch
  import torch.nn.functional as F
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
except ImportError:
    pass
from easypbr  import *
from dataloaders import *
from collections import namedtuple

# np.set_printoptions(threshold=sys.maxsize)
np.random.seed(0)


#Just to have something close to the macros we have in c++
def profiler_start(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.start(name)
def profiler_end(name):
    if(Profiler.is_profiling_gpu()):
        torch.cuda.synchronize()
    Profiler.end(name)
TIME_START = lambda name: profiler_start(name)
TIME_END = lambda name: profiler_end(name)





config_file="test_loader.cfg"

config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
view=Viewer.create(config_path) #first because it needs to init context


def test_volref():
    loader=DataLoaderVolRef(config_path)
    loader.start()


    while True:
        if(loader.has_data()  ):

            first=True

            #volref
            # print("got frame")
            frame_color=loader.get_color_frame()
            frame_depth=loader.get_depth_frame()


            Gui.show(frame_color.rgb_32f, "rgb")

            rgb_with_valid_depth=frame_color.rgb_with_valid_depth(frame_depth)
            Gui.show(rgb_with_valid_depth, "rgb_valid")


            frustum_mesh=frame_depth.create_frustum_mesh(0.02)
            frustum_mesh.m_vis.m_line_width=3
            frustum_name="frustum"
            Scene.show(frustum_mesh, frustum_name)
            # Scene.show(frustum_mesh, "frustum_"+str(frame_color.frame_idx) )

            cloud=frame_depth.depth2world_xyz_mesh()
            frame_color.assign_color(cloud) #project the cloud into this frame and creates a color matrix for it
            Scene.show(cloud, "cloud")



        if loader.is_finished():
            # print("resetting")
            loader.reset()

        view.update()


def test_img():
    loader=DataLoaderImg(config_path)
    loader.start()

    while True:
        for cam_idx in range(loader.get_nr_cams()):
            if(loader.has_data_for_cam(cam_idx)):
                # print("data")
                frame=loader.get_frame_for_cam(cam_idx)
                Gui.show(frame.rgb_32f, "rgb")
                #get tensor
                rgb_tensor=frame.rgb2tensor()
                rgb_tensor=rgb_tensor.to("cuda")


        view.update()

def test_semantickitti():
    loader=DataLoaderSemanticKitti(config_path)
    loader.start()

    while True:
        if(loader.has_data()):
            cloud=loader.get_cloud()
            Scene.show(cloud, "cloud")

            # if cloud.V.size(0)==125620:
                # print("found")
        if loader.is_finished():
            loader.reset()

        view.update()

def test_scannet():
    loader=DataLoaderScanNet(config_path)
    loader.start()

    while True:
        if(loader.has_data()):
            cloud=loader.get_cloud()
            Scene.show(cloud, "cloud")


        view.update()

def test_stanford3dscene():
    loader=DataLoaderStanford3DScene(config_path)
    loader.start()

    while True:
        if(loader.has_data() ):

            print("got frame")
            frame_color=loader.get_color_frame()
            frame_depth=loader.get_depth_frame()


            Gui.show(frame_color.rgb_32f, "rgb")

            rgb_with_valid_depth=frame_color.rgb_with_valid_depth(frame_depth)
            Gui.show(rgb_with_valid_depth, "rgb_valid")

            frustum_mesh=frame_color.create_frustum_mesh(0.1)
            frustum_mesh.m_vis.m_line_width=3
            frustum_name="frustum"
            Scene.show(frustum_mesh, frustum_name)

            cloud=frame_depth.backproject_depth()
            frame_color.assign_color(cloud) #project the cloud into this frame and creates a color matrix for it
            # Scene.show(cloud, "cloud"+str(nr_cloud))
            Scene.show(cloud, "cloud")

            #show look dir
            look_dir_mesh=Mesh()
            look_dir=frame_color.look_dir()
            look_dir_mesh.V=[
                look_dir
            ]
            look_dir_mesh.m_vis.m_show_points=True
            Scene.show(look_dir_mesh, "look_dir_mesh")

        view.update()

def test_shapenet_img():
    loader=DataLoaderShapeNetImg(config_path)

    i=0

    while True:
        if(loader.finished_reading_scene() ):
            frame=loader.get_random_frame()

            if i%1==0:
                loader.start_reading_next_scene()

            if frame.is_shell:
                frame.load_images()

            Gui.show(frame.rgb_32f, "rgb")
            Gui.show(frame.mask, "mask")
            Gui.show(frame.depth, "depth")
            frustum=frame.create_frustum_mesh(0.1)
            # Scene.show(frustum, "frustum"+ str(frame.frame_idx) )
            Scene.show(frustum, "frustum" )

            # cloud=frame.depth2world_xyz_mesh()
            # cloud=frame.assign_color(cloud)
            # Scene.show(cloud, "cloud")

            i+=1


        if loader.is_finished():
            print("resetting")
            loader.reset()

        view.update()

def test_nerf():
    loader=DataLoaderNerf(config_path)
    loader.set_mode_train()
    # loader.set_mode_test()
    loader.start()

    while True:
        if(loader.has_data() ):

            # print("got frame")
            frame=loader.get_next_frame()
            # print("frame width and height is ", frame.width, " ", frame.height)


            Gui.show(frame.rgb_32f, "rgb")


            frustum_mesh=frame.create_frustum_mesh(0.02)
            frustum_mesh.m_vis.m_line_width=1
            Scene.show(frustum_mesh, "frustum_"+str(frame.frame_idx) )

            # cloud=frame.depth2world_xyz_mesh()
            # Scene.show(cloud, "cloud")

        if loader.is_finished():
            print("resetting")
            loader.reset()
            print("scene scale is ", Scene.get_scale())

        view.update()

def test_pheno4d():
    loader=DataLoaderPheno4D(config_path)
    loader.start()
    loader.set_do_augmentation(True)

    while True:
        if(loader.has_data() ):
            # print("has data")

            cloud=loader.get_cloud()

            Scene.show(cloud, "cloud" )

        if loader.is_finished():
            # print("resetting")
            loader.reset()

        view.update()

def test_cloud_ros():
    bag=RosBagPlayer.create(config_path)
    loader=DataLoaderCloudRos(config_path)

    while loader.is_loader_thread_alive():
        if(loader.has_data() ):
            cloud=loader.get_cloud()

            Scene.show(cloud, "cloud" )

        view.update()

def test_colmap():
    loader=DataLoaderColmap(config_path)
    loader.start()

    while True:
        if(loader.has_data() ):

            # print("got frame")
            frame=loader.get_next_frame()
            # print("loaded frame with width and height ", frame.width, " ", frame.height)


            Gui.show(frame.rgb_32f, "rgb")

            # print("frame k is ", frame.K)
            frustum_mesh=frame.create_frustum_mesh(0.02)
            frustum_mesh.m_vis.m_line_width=1
            Scene.show(frustum_mesh, "frustum_"+str(frame.frame_idx) )

            # cloud=frame.depth2world_xyz_mesh()
            # Scene.show(cloud, "cloud")

        if loader.is_finished():
            print("resetting")
            loader.reset()
            print("scene scale is ", Scene.get_scale())

        view.update()

def test_easypbr():
    loader=DataLoaderEasyPBR(config_path)
    loader.set_mode_train()
    loader.start()

    if loader.loaded_scene_mesh():
        mesh=loader.get_scene_mesh()
        Scene.show(mesh,"mesh")

    while True:
        if(loader.has_data() ):

            # print("got frame")
            frame=loader.get_next_frame()
            # print("frame width and height is ", frame.width, " ", frame.height)

            # if frame.frame_idx==20:
            Gui.show(frame.rgb_32f, "rgb")
            Gui.show(frame.mask.to_cv8u(), "mask")
            # mask_t=mat2tensor(frame.mask, True)
            # print("mask_t min max", mask_t.min(), mask_t.max())


            frustum_mesh=frame.create_frustum_mesh(0.05)
            # frustum_mesh=frame.create_frustum_mesh(0.2)
            frustum_mesh.m_vis.m_line_width=1
            if frame.frame_idx==1:
                frustum_mesh.m_vis.m_line_color=[0.0, 1.0, 0.0]
            Scene.show(frustum_mesh, "frustum_"+str(frame.frame_idx) )
            # mesh=Scene.get_mesh_with_name("frustum_"+str(frame.frame_idx))
            # frustum_mesh.m_vis.m_line_width=1
            # frustum_mesh.m_vis.m_line_color=[0.0, 1.0, 0.0]

            # #make all frustums red except the current one
            # for i in range(200):
            #     if i==frame.frame_idx:
            #         continue
            #     if Scene.does_mesh_with_name_exist("frustum_"+str(i)):
            #         mesh=Scene.get_mesh_with_name("frustum_"+str(i))
            #         frustum_mesh.m_vis.m_line_width=1
            #         frustum_mesh.m_vis.m_line_color=[1.0, 0.0, 0.0]

            # cloud=frame.depth2world_xyz_mesh()
            # Scene.show(cloud, "cloud")

        if loader.is_finished():
            print("resetting")
            loader.reset()
            print("scene scale is ", Scene.get_scale())

        view.update()

def test_srn():
    loader=DataLoaderSRN(config_path)
    loader.set_mode_train()
    # loader.set_mode_test()
    loader.start()

    i=0

    while True:
        if(loader.finished_reading_scene() ):
            frame=loader.get_random_frame()

            if i%1==0:
                loader.start_reading_next_scene()

            if frame.is_shell:
                frame.load_images()

            Gui.show(frame.rgb_32f, "rgb")
            frustum=frame.create_frustum_mesh(0.02)
            Scene.show(frustum, "frustum"+ str(frame.frame_idx) )
            # Scene.show(frustum, "frustum" )

            # cloud=frame.depth2world_xyz_mesh()
            # cloud=frame.assign_color(cloud)
            # Scene.show(cloud, "cloud")

            i+=1


        if loader.is_finished():
            print("resetting")
            loader.reset()

        view.update()


def test_dtu():
    loader=DataLoaderDTU(config_path)
    # loader.set_mode_train()
    # loader.set_mode_test()
    loader.set_mode_all()
    loader.start()


    #to check if the meshes fit within the unit sphere
    sphere=Mesh()
    sphere.create_sphere([0,0,0],0.5)
    # sphere_sdf.upsample(4, True)
    sphere.m_vis.m_show_mesh=False
    sphere.m_vis.m_show_wireframe=True
    Scene.show(sphere,"sphere")


    #get gt mesh from dtu
    # gt_mesh=Mesh("/media/rosu/Data/data/dtu/data_prepared_for_gt/Points/stl/stl024_total.ply")
    gt_mesh=Mesh("/media/rosu/Data/data/dtu/data_prepared_for_gt/Points/stl/stl105_total.ply")
    tf_easypbr_dtu=loader.get_tf_easypbr_dtu()
    gt_mesh.transform_model_matrix(tf_easypbr_dtu.to_double())
    gt_mesh.apply_model_matrix_to_cpu(True)
    Scene.show(gt_mesh, "gt_mesh")

    # view.m_camera.from_string(" 107.104 -8.58596   478.92  0.918689 0.0670844  0.178054 -0.346129  96.2678 -34.0564  505.822 60 39.8026 3.98026e+06")
    view.m_camera.from_string("4.61049 3.34609 2.97043 -0.237639  0.443042  0.123049 0.855627  0.548705  0.352442 0.0998282 60 0.00419346 419.346")

    i=0
    first_time=True

    while True:
        if(loader.finished_reading_scene() ):
            frame=loader.get_random_frame()


            TIME_START("loadtotensor")
            # gt_rgb=mat2tensor(frame.rgb_32f, True).to("cuda")
            TIME_END("loadtotensor")

            TIME_START("get_tensor")
            if frame.has_extra_field("has_gpu_tensors"):
                print("has gpu tensors")
                rgb_tensor=frame.get_extra_field_tensor("rgb_32f_tensor")
                mask_tensor=frame.get_extra_field_tensor("mask_tensor")
                #show
                Gui.show(tensor2mat(rgb_tensor).rgb2bgr(), "rgb_tensor")
                Gui.show(tensor2mat(mask_tensor), "mask_tensor")
            TIME_END("get_tensor")


            # if i%1==0:
                # loader.start_reading_next_scene()

            print("frame rgb path is ", frame.rgb_path)

            if frame.is_shell:
                frame.load_images()

            print("frame.tf_cam_world", frame.tf_cam_world.matrix())

            # print("frame.frame_idx", frame.frame_idx)
            # if frame.frame_idx==0:
                # print("frame.K is ", frame.K)

            if first_time:
                frame_subsampled=frame.subsample(32.0, True)
                Gui.show(frame_subsampled.rgb_32f, "rgb", interpolation="nearest")
            Gui.show(frame.mask, "mask")
            frustum=frame.create_frustum_mesh(20)
            # frustum.apply_model_matrix_to_cpu(True)
            Scene.show(frustum, "frustum"+ str(frame.frame_idx) )
            # Scene.show(frustum, "frustum" )

            # loader.start_reading_next_scene()
            # while True:
            #     if(loader.finished_reading_scene()):
            #         break

            #color the gt mesh with the color from a frame
            # if frame.frame_idx==33 and first_time:
            # if first_time:
            #     frame.assign_color(gt_mesh)
            #     Scene.show(gt_mesh, "gt_mesh")
            #     first_time=False

            # if frame.frame_idx==0:
            #     tf_cam_world=frame.tf_cam_world
            #     tf_world_cam=tf_cam_world.inverse()
            #     print("tf_cam_world ", tf_cam_world.matrix())
            #     print("tf_world_cam ", tf_world_cam.matrix())


            first_time=False 

            i+=1



        if loader.is_finished():
            print("resetting")
            loader.reset()

        view.update()

def test_blended_mvs():
    #has the same loader as DTU because the Neus paper uses the same format
    loader=DataLoaderDTU(config_path)
    loader.set_mode_train()
    loader.set_dataset_path("/media/rosu/Data/data/neus_data/data_BlendedMVS")
    # loader.set_restrict_to_scene_name("bmvs_dog")
    # loader.set_restrict_to_scene_name("bmvs_stone")
    # loader.set_restrict_to_scene_name("bmvs_sculpture")
    # loader.set_restrict_to_scene_name("bmvs_jade")
    loader.set_restrict_to_scene_name("bmvs_bear")
    loader.set_load_mask(True)
    loader.set_rotate_scene_x_axis_degrees(180)
    # loader.set_mode_validation() #test set actually doesnt exist and we actually use the validation one
    # loader.set_restrict_to_scan_idx(5)
    loader.start()

    i=0

    while True:
        if(loader.finished_reading_scene() ):
            frame=loader.get_random_frame()

            # if i%1==0:
                # loader.start_reading_next_scene()

            print("frame rgb path is ", frame.rgb_path)

            if frame.is_shell:
                frame.load_images()

            # print("frame.frame_idx", frame.frame_idx)
            # if frame.frame_idx==0:
                # print("frame.K is ", frame.K)
            
            if not frame.mask.empty():
                Gui.show(frame.mask, "mask")

            Gui.show(frame.rgb_32f, "rgb")
            frustum=frame.create_frustum_mesh(0.02)
            Scene.show(frustum, "frustum"+ str(frame.frame_idx) )
            # Scene.show(frustum, "frustum" )

            # loader.start_reading_next_scene()
            # while True:
            #     if(loader.finished_reading_scene()):
            #         break

            i+=1


        if loader.is_finished():
            print("resetting")
            loader.reset()

        view.update()


def test_deep_voxels():
    loader=DataLoaderDeepVoxels(config_path)
    loader.set_mode_train()
    # loader.set_mode_test()
    loader.start()

    while True:
        if(loader.has_data() ):

            # print("got frame")
            frame=loader.get_next_frame()
            # print("frame width and height is ", frame.width, " ", frame.height)


            Gui.show(frame.rgb_32f, "rgb")


            frustum_mesh=frame.create_frustum_mesh(0.01)
            frustum_mesh.m_vis.m_line_width=1
            Scene.show(frustum_mesh, "frustum_"+str(frame.frame_idx) )

            # cloud=frame.depth2world_xyz_mesh()
            # Scene.show(cloud, "cloud")

        if loader.is_finished():
            print("resetting")
            loader.reset()
            print("scene scale is ", Scene.get_scale())

        view.update()

def test_llff():
    loader=DataLoaderLLFF(config_path)
    loader.set_mode_train()
    loader.start()

    while True:
        if(loader.has_data() ):

            # print("got frame")
            frame=loader.get_next_frame()
            print("loaded frame with width and height ", frame.width, " ", frame.height)
            print("frame k is ", frame.K)


            Gui.show(frame.rgb_32f, "rgb")

            # print("frame k is ", frame.K)
            frustum_mesh=frame.create_frustum_mesh(0.001)
            frustum_mesh.m_vis.m_line_width=1
            Scene.show(frustum_mesh, "frustum_"+str(frame.frame_idx) )

            # cloud=frame.depth2world_xyz_mesh()
            # Scene.show(cloud, "cloud")

            near = frame.get_extra_field_float("min_near")
            far = frame.get_extra_field_float("max_far")
            print("near is ", near)
            print("far is ", far)
            # frame.add_extra_field("test", 54.0)
            # roundback= frame.get_extra_field_float("test")
            # print("roundback is ", roundback)

        if loader.is_finished():
            print("resetting")
            loader.reset()
            print("scene scale is ", Scene.get_scale())

        view.update()

def test_blender_fb():
    loader=DataLoaderBlenderFB(config_path)
    # loader.set_mode_train()
    # loader.set_mode_test()
    loader.start()

    while True:
        if(loader.has_data() ):

            # print("got frame")
            frame=loader.get_next_frame()
            # print("frame width and height is ", frame.width, " ", frame.height)


            Gui.show(frame.rgb_32f, "rgb")
            # Gui.show(frame.rgb_8u, "rgb8u")


            frustum_mesh=frame.create_frustum_mesh(0.05)
            frustum_mesh.m_vis.m_line_width=1
            # frustum_mesh.m_is_dirty=True
            Scene.show(frustum_mesh, "frustum_"+str(frame.frame_idx) )

            # cloud=frame.depth2world_xyz_mesh()
            # Scene.show(cloud, "cloud")

            if frame.has_extra_field("orientation_mat"):
                orientation_mat = frame.get_extra_field_mat("orientation_mat")
                variance_mat = frame.get_extra_field_mat("variance_mat")
                Gui.show(orientation_mat,"orientation_mat")
                Gui.show(variance_mat,"variance_mat")

        if loader.is_finished():
            # print("resetting")
            loader.reset()
            # print("scene scale is ", Scene.get_scale())

        view.update()

def test_usc_hair():
    loader=DataLoaderUSCHair(config_path)
    loader.set_mode_train()
    # loader.set_mode_test()
    loader.start()

    while True:
        if(loader.has_data()):
            # print("HAS DATA--------")
            # pass
            # hair=loader.get_hair()
            # cloud=hair.full_hair_cloud
            cloud=loader.get_hair().full_hair_cloud.clone()
            Scene.show(cloud, "cloud")

            #show also the head
            # head=loader.get_mesh_head()
            # Scene.show(head, "head")

            #show also the scalp
            scalp=loader.get_mesh_scalp()
            Scene.show(scalp, "scalp")

            if cloud.has_extra_field("strand_idx"):
                strand_idx=cloud.get_extra_field_matrixXi("strand_idx")
                # print("strand idx", strand_idx)

            if cloud.has_extra_field("uv_roots"):
                uv_roots=cloud.get_extra_field_matrixXd("uv_roots")
                # print("uv_roots", uv_roots)

        if loader.is_finished():
            loader.reset()

        view.update()


def test_phenorob_cp1():
    loader=DataLoaderPhenorobCP1(config_path)
    loader.set_mode_all()
    loader.start()

    show_backprojected_depth=False
    show_backprojected_depth_along_ray=False

    #make a sphere of radius 0.7 at origin because this is kinda what we use for instant ngp
    sphere=Mesh()
    sphere.create_sphere([0,0,0], 0.49)
    sphere.m_vis.m_show_mesh=False
    sphere.m_vis.m_show_wireframe=True
    Scene.show(sphere,"sphere")



    # frames_train=[]
    # max_width=0
    # max_height=0
    # CropStruct = namedtuple("CropStruct", "start_x start_y crop_width crop_height")
    # list_of_true_crops=[]
    # for i in range(loader.nr_samples()):
    #     frame=loader.get_frame_at_idx(i)
    #     if frame.is_shell:
    #         frame.load_images()
    #     cam=Camera()
    #     coords_2d_center=frame.project([0,0,0]) #project the center (the dataloader already ensures that the plant in interest is at the center)
    #     #project also along the x and y axis of the camera to get the bounds of the sphere in camera coords
    #     cam.from_frame(frame)
    #     cam_axes=cam.cam_axes()
    #     x_axis=cam_axes[:,0]
    #     y_axis=cam_axes[:,1]
    #     radius=0.7
    #     coords_2d_x_positive=frame.project(x_axis*radius)
    #     coords_2d_x_negative=frame.project(-x_axis*radius)
    #     coords_2d_y_positive=frame.project(y_axis*radius)
    #     coords_2d_y_negative=frame.project(-y_axis*radius)
    #     #for cropping
    #     start_x=int(coords_2d_x_negative[0])
    #     start_y=int(coords_2d_y_positive[1])
    #     width=int(coords_2d_x_positive[0]-coords_2d_x_negative[0])
    #     height=int(coords_2d_y_negative[1]-coords_2d_y_positive[1])
    #     print("start_x",start_x)
    #     print("start_y",start_y)
    #     print("width",width)
    #     print("height",height)
    #     start_x, start_y, width, height=frame.get_valid_crop(start_x, start_y, width, height)
    #     frame=frame.crop(start_x, start_y, width, height, True)
    #     # frames_train.append(frame)
    #     # Gui.show(frame.rgb_32f, " frame "+str(i) ) 
    #     if frame.width>max_width:
    #         max_width=frame.width
    #     if frame.height>max_height:
    #         max_height=frame.height
    #     true_crop = CropStruct(start_x, start_y, width, height)
    #     list_of_true_crops.append(true_crop)
    # print("max_width",max_width)
    # print("max_height",max_height)
    # #upsampel the true crops so that they are still inside the image but they all have the same width and height so that we can use the tensorreel
    # frames_train_cropped_equal=[]
    # for i in range(loader.nr_samples()):
    #     frame=loader.get_frame_at_idx(i)
    #     if frame.is_shell:
    #         frame.load_images()
    #     true_crop=list_of_true_crops[i]
    #     start_x, start_y, width, height=frame.enlarge_crop_to_size(true_crop.start_x, true_crop.start_y, true_crop.crop_width, true_crop.crop_height, max_width, max_height) #crops then all to the same size
    #     # print("enlarged crop has x,y, width and height ", start_x, start_y, width, height)
    #     # print("will now crop a frame of size width and height ", frame.width, frame.height)
    #     # print("true crop", true_crop)
    #     frame=frame.crop(start_x, start_y, width, height, True)
    #     print("frame shouls all have equal size now ", frame.width, " ", frame.height)
    #     frames_train_cropped_equal.append(frame)
    #     Gui.show(frame.rgb_32f, " frame "+str(i) ) 







    def map_range_tensor( input_val, input_start, input_end,  output_start,  output_end):
        # input_clamped=torch.clamp(input_val, input_start, input_end)
        # input_clamped=max(input_start, min(input_end, input_val))
        input_clamped=torch.clamp(input_val, input_start, input_end)
        return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)

    print("nr days ", loader.nr_days())
    #ITER days
    for d_idx in range(loader.nr_days()):
        day=loader.get_day_with_idx(d_idx)
        print("loading day with idx", d_idx)

        #ITER scans
        for s_idx in range(day.nr_scans()):
            scan=day.get_scan_with_idx(s_idx)
            print("loading scan with idx", s_idx)

            #ITER blocks
            for b_idx in range(scan.nr_blocks()):
                block=scan.get_block_with_idx(b_idx)


                #show all the images from the first block only
                if b_idx==0:
                    print("loading block with idx", b_idx)
                    for f_idx in range(block.nr_frames()):
                        frame=block.get_rgb_frame_with_idx(f_idx)
                        if frame.is_shell:
                            frame.load_images()
                        #create a frustum fro the RGB frames
                        frustum_mesh=frame.create_frustum_mesh(0.2, True, 256)
                        # frustum_mesh=frame.create_frustum_mesh(0.01, True, 256)
                        frustum_mesh.m_vis.m_line_width=1
                        Scene.show(frustum_mesh, "frustum_"+str(frame.cam_id) )


                        #show the visible points
                        # if f_idx==0 and frame.has_extra_field("visible_points"):
                        if False:
                            visible_points=frame.get_extra_field_mesh("visible_points")
                            visible_points=loader.load_mesh(visible_points)
                            visible_points.apply_model_matrix_to_cpu(True)
                            visible_points.recalculate_min_max_height()
                            Scene.show(visible_points, "visible_points_"+str(f_idx))
                            #show the frame0 together with the depth of the visible points
                            Gui.show(frame.rgb_32f, "rgb_"+str(f_idx))
                            #get the visible points in cam coordinates, compute the depth, and then splat it
                            visible_points_cam=visible_points.clone()
                            visible_points_cam.transform_model_matrix( frame.tf_cam_world.to_double() )
                            visible_points_cam.apply_model_matrix_to_cpu(True)
                            visible_points_cam_t=torch.from_numpy(visible_points_cam.V).float()
                            visible_points_depth=visible_points_cam_t.norm(dim=1,keepdim=True)
                            print("visible_points_depth", visible_points_depth.shape)
                            visible_points_depth_np=visible_points_depth.cpu().float().numpy()
                            depth_visible_points_mat=frame.naive_splat(visible_points, visible_points_depth_np)
                            Gui.show(depth_visible_points_mat, "depth_visible_points_mat")
                        

                        #show the depth if it exists
                        if f_idx==0 and not frame.depth.empty() and show_backprojected_depth:
                            sfm_depth_backproj=frame.depth2world_xyz_mesh()
                            sfm_depth_backproj.m_vis.m_point_color=[0.7, 0.3, 0.3]
                            Scene.show(sfm_depth_backproj, "sfm_depth_backproj_"+str(f_idx))
                            depth_tensor=mat2tensor(frame.depth, False)
                            depth_tensor_original=depth_tensor
                            depth_tensor=map_range_tensor(depth_tensor, 8.0, 16.0, 0.0, 1.0)
                            depth_tensor=depth_tensor.repeat(1,3,1,1)
                            Gui.show(tensor2mat(depth_tensor), "depth", frame.rgb_32f, "rgb")
                            depth_tensor_small=(   torch.logical_and(depth_tensor_original<5.0, depth_tensor_original!=0  )  )*1.0
                            Gui.show(tensor2mat(depth_tensor_small), "depth_tensor_small")
                            Gui.show(tensor2mat(depth_tensor_original), "depth", tensor2mat(depth_tensor_small), "depth_tensor_small")

                        #show the distance_along_ray if it exists
                        if f_idx==0 and not frame.depth_along_ray.empty() and show_backprojected_depth_along_ray:
                        # if not frame.depth_along_ray.empty() and show_backprojected_depth_along_ray:
                            depth_along_ray_mat=frame.depth_along_ray
                            Gui.show(depth_along_ray_mat, "depth_along_ray_mat" )
                            #BACKPROJECT also this depth to check if its correct
                            ######################create rays and dirs
                            x_coord= torch.arange(frame.width).view(-1, 1, 1).repeat(1,frame.height, 1)+0.5 #width x height x 1
                            y_coord= torch.arange(frame.height).view(1, -1, 1).repeat(frame.width, 1, 1)+0.5 #width x height x 1
                            ones=torch.ones(frame.width, frame.height).view(frame.width, frame.height, 1)
                            points_2D=torch.cat([x_coord, y_coord, ones],2).transpose(0,1).reshape(-1,3).cuda()
                            K_inv=torch.from_numpy( np.linalg.inv(frame.K) ).to("cuda").float()
                            #get from screen to cam coords
                            pixels_selected_screen_coords_t=points_2D.transpose(0,1) #3xN
                            pixels_selected_cam_coords=torch.matmul(K_inv,pixels_selected_screen_coords_t).transpose(0,1)
                            #multiply at various depths
                            nr_rays=pixels_selected_cam_coords.shape[0]
                            pixels_selected_cam_coords=pixels_selected_cam_coords.view(nr_rays, 3)
                            #get from cam_coords to world_coords
                            tf_world_cam=frame.tf_cam_world.inverse()
                            R=torch.from_numpy( tf_world_cam.linear() ).to("cuda").float()
                            t=torch.from_numpy( tf_world_cam.translation() ).to("cuda").view(1,3).float()
                            pixels_selected_world_coords=torch.matmul(R, pixels_selected_cam_coords.transpose(0,1).contiguous() ).transpose(0,1).contiguous()  + t
                            #get direction
                            ray_dirs = pixels_selected_world_coords-t
                            ray_dirs=F.normalize(ray_dirs, p=2, dim=1)
                            #ray_origins
                            ray_origins=t.repeat(nr_rays,1)
                            #################get the depth for each pixel
                            # tex=img2tex(img) #hwc
                            depth_along_ray_tensor=mat2tensor(depth_along_ray_mat,False).cuda()
                            tex=depth_along_ray_tensor.permute(0,2,3,1).squeeze(0) #hwc
                            tex_channels=tex.shape[2]
                            tex_flattened=tex.view(-1,tex_channels)        
                            gt_depth_along_ray_full=tex_flattened
                            gt_ray_end_full = ray_origins + ray_dirs * gt_depth_along_ray_full.view(-1,1)
                            ################show points
                            pred_points_cpu=gt_ray_end_full.contiguous().view(-1,3).detach().double().cpu().numpy()
                            pred_strands_mesh=Mesh()
                            pred_strands_mesh.V=pred_points_cpu
                            pred_strands_mesh.m_vis.m_show_points=True
                            Scene.show(pred_strands_mesh, "points_depth_along_ray"+str(f_idx))
                                                    



                        




                        #get the right stereo pair if it exists
                        if ( frame.has_right_stereo_pair() ):
                            frame_right=frame.right_stereo_pair();
                            print("cam ", frame.cam_id, " has right pair ", frame_right.cam_id)
                            if frame_right.is_shell:
                                frame_right.load_images()

                            
                            # #if this is the camera on the top fo robot, we need to rotate them to create a stereo pair, rotate the left frame 90 degrees and the right one 270
                            # if frame.cam_id==13:
                            #     print("rotating top cameras")
                            #     frame=frame.rotate_clockwise_90()
                            #     frame_right=frame_right.rotate_clockwise_90()
                            #     frame_right=frame_right.rotate_clockwise_90()
                            #     frame_right=frame_right.rotate_clockwise_90()
                            #     frame.set_right_stereo_pair(frame_right)
                            #     # #show the frames
                            #     frustum_mesh=frame.create_frustum_mesh(0.05, True, 256)
                            #     frustum_mesh.m_vis.m_line_width=1
                            #     Scene.show(frustum_mesh, "frustumleft" )
                            #     frustum_mesh=frame_right.create_frustum_mesh(0.05, True, 256)
                            #     frustum_mesh.m_vis.m_line_width=1
                            #     Scene.show(frustum_mesh, "frustumright" )
                            #     # TODO currently someone bumped their head into the top camera because the images don't look the same as when it was calibrated


                            #show the left and right camera images                           
                            # Gui.show(frame.rgb_32f, str(frame.cam_id)+"_left", frame_right.rgb_32f, str(frame_right.cam_id)+"_right")
                            
                            #rectify
                            # pair=frame.rectify_stereo_pair(0)
                            # frame_left_rectified=pair[0]
                            # frame_right_rectified=pair[1]
                            # baseline=pair[2]
                            # Q=pair[3]
                            # Gui.show(frame_left_rectified.rgb_32f, str(frame.cam_id)+"_rectleft", frame_right_rectified.rgb_32f, str(frame_right.cam_id)+"_rectright")
                            # print("baseline is ", baseline)


                            # #Show the rectified frame for the top one
                            # if frame.cam_id==13:
                            #     frustum_mesh=frame_left_rectified.create_frustum_mesh(0.05, True, 256)
                            #     Scene.show(frustum_mesh, "rect_fr_left" )
                            #     frustum_mesh=frame_right_rectified.create_frustum_mesh(0.05, True, 256)
                            #     Scene.show(frustum_mesh, "rect_fr_right" )



                #load the dense cloud for this block
                if loader.loaded_dense_cloud():
                    dense_cloud=block.get_dense_cloud()
                    dense_cloud=loader.load_mesh(dense_cloud)
                    # dense_cloud.load_from_file(dense_cloud.m_disk_path)
                    print("dense_cloud", dense_cloud.model_matrix.matrix() )
                    dense_cloud.apply_model_matrix_to_cpu(True)
                    dense_cloud.recalculate_min_max_height()
                    Scene.show(dense_cloud, "dense_cloud_"+str(b_idx)+"_"+str(d_idx))
                    # Scene.show(dense_cloud, "dense_cloud_")


            print("test_loader showng only one scan")
            break

        # print("test_loader showng only one day")
        # break
        if d_idx==2:
            print("test_loader showng only two days")
            # break





    while True:
        # view.m_camera.from_frame(frame1, True)

        view.update()

def test_multiface():

    def linear2color_corr(img, dim: int = -1):
        """Applies ad-hoc 'color correction' to a linear RGB Mugsy image along
        color channel `dim` and returns the gamma-corrected result."""

        if dim == -1:
            dim = len(img.shape) - 1

        gamma = 2.0
        black = 3.0 / 255.0
        color_scale = [1.4, 1.1, 1.6]

        assert img.shape[dim] == 3
        if dim == -1:
            dim = len(img.shape) - 1
        scale = np.array(color_scale).reshape([3 if i == dim else 1 for i in range(img.ndim)])
        img = img * scale / 1.1
        return np.clip(
            (((1.0 / (1 - black)) * 0.95 * np.clip(img - black, 0, 2)) ** (1.0 / gamma))
            - 15.0 / 255.0,
            0,
            2,
        )


    subject_id=7
    loader=DataLoaderMultiFace(config_path, subject_id)
    loader.set_mode_all()
    loader.start()
    nr_samples=loader.nr_samples()

    mesh_head=loader.get_mesh_head()
    Scene.show(mesh_head,"mesh_head")


    #make a sphere of radius 0.5 at origin because this is kinda what we use for instant ngp
    sphere=Mesh()
    sphere.create_sphere([0,0,0], 0.5)
    sphere.m_vis.m_show_mesh=False
    sphere.m_vis.m_show_wireframe=True
    Scene.show(sphere,"sphere")


    while True:
        if(loader.has_data() ):

        #     # print("got frame")
            frame=loader.get_next_frame()
            if frame.is_shell:
                frame.load_images()
        #     # print("frame width and height is ", frame.width, " ", frame.height)


            #change the image to the corrected one
            rgb_32f=frame.rgb_32f
            rgb_t=mat2tensor(rgb_32f,True)
            rgb_t=linear2color_corr(rgb_t, dim=1).clamp(0, 1).float()
            frame.rgb_32f=tensor2mat(rgb_t).rgb2bgr()

        #     # if frame.frame_idx==20:
            Gui.show(frame.rgb_32f, "rgb")
        #     Gui.show(frame.mask.to_cv8u(), "mask")
        #     # mask_t=mat2tensor(frame.mask, True)
        #     # print("mask_t min max", mask_t.min(), mask_t.max())

            # print("frame.tf_cam_world", frame.tf_cam_world.matrix())


            # frustum_mesh=frame.create_frustum_mesh(scale_multiplier=0.35, near_multiplier=0.001, far_multiplier=2.5)
            frustum_mesh=frame.create_frustum_mesh(scale_multiplier=0.3, near_multiplier=0.001, far_multiplier=2.5)

            #squish the frame in the z direction
            tf_world_obj=frustum_mesh.model_matrix.clone()
            tf_obj_world=tf_world_obj.inverse()
            frustum_mesh.transform_model_matrix(tf_obj_world)
            frustum_mesh.apply_model_matrix_to_cpu(True)
            V=frustum_mesh.V.copy()
            V[:,2:3]=V[:,2:3]*0.3
            frustum_mesh.V=V
            frustum_mesh.transform_model_matrix(tf_world_obj)
            

            frustum_mesh.m_vis.m_show_mesh=False
            frustum_mesh.m_vis.m_line_width=2
            # frustum_mesh.m_vis.m_show_lines=False
            frustum_mesh.m_vis.m_line_color=[0.2,0.2,0.2]
            # frustum_mesh.apply_model_matrix_to_cpu(True)
            Scene.show(frustum_mesh, "frustum_"+str(frame.cam_id) )
          

        if loader.is_finished():
        #     print("resetting")
            loader.reset()
        #     print("scene scale is ", Scene.get_scale())

        view.update()




# test_volref()
# test_img()
# test_img_ros()
# test_cloud_ros()
# test_semantickitti()
# test_scannet()
# test_stanford3dscene()
# test_shapenet_img()
# test_nerf()
# test_pheno4d()
# test_colmap()
# test_easypbr()
# test_srn()
# test_dtu()
# test_blended_mvs()
# test_deep_voxels()
# test_llff()
# test_blender_fb()
# test_usc_hair()
test_phenorob_cp1()
# test_multiface()
