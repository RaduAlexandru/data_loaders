#!/usr/bin/env python3.6

import os
import numpy as np
import sys
try:
  import torch
except ImportError:
    pass
from easypbr  import *
from dataloaders import *
# np.set_printoptions(threshold=sys.maxsize)
np.random.seed(0)


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
    # loader.set_mode_test()
    loader.start()

    while True:
        if(loader.has_data() ):

            # print("got frame")
            frame=loader.get_next_frame()
            # print("frame width and height is ", frame.width, " ", frame.height)

            # if frame.frame_idx==20:
            Gui.show(frame.rgb_32f, "rgb")


            # frustum_mesh=frame.create_frustum_mesh(0.02)
            frustum_mesh=frame.create_frustum_mesh(0.2)
            frustum_mesh.m_vis.m_line_width=1
            if frame.frame_idx==20:
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
    loader.set_mode_train()
    # loader.set_mode_validation() #test set actually doesnt exist and we actually use the validation one
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

            Gui.show(frame.rgb_32f, "rgb")
            frustum=frame.create_frustum_mesh(0.02)
            Scene.show(frustum, "frustum"+ str(frame.frame_idx) )
            # Scene.show(frustum, "frustum" )

            loader.start_reading_next_scene()
            while True:
                if(loader.finished_reading_scene()):
                    break

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

def test_llff():
    loader=DataLoaderLLFF(config_path)
    loader.set_mode_train()
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



    #get frame
    for i in range(14):
        frame=loader.get_block_with_idx(0).get_rgb_frame_at_idx(i)
        cam_id=frame.cam_id
        # if cam_id in[0,1,10]:
        if cam_id in[0,1,2]:
        # if True:
            if frame.is_shell:
                frame.load_images()
            frustum_mesh=frame.create_frustum_mesh(0.05, True, 1024)
            frustum_mesh.m_vis.m_line_width=1
            Scene.show(frustum_mesh, "frustum_"+str(frame.cam_id) )

            print("frame cam id is ", frame.cam_id)
            print("frame tf_cam_world is ", frame.tf_cam_world.matrix())

    frame0=loader.get_block_with_idx(0).get_rgb_frame_at_idx(0)
    frame1=loader.get_block_with_idx(0).get_rgb_frame_at_idx(1)
    frame0.load_images()
    frame1.load_images()
    # line=f,rame0.line_through_pixel(256,256,1000)
    # epipolar_image=frame1.draw_projected_line(line)

    line_p0=frame0.pos_in_world().astype(np.float64)
    # line_p1=frame0.unproject(4132.291660151611, 2813.56775866408, 1) #4132.291660151611, 2813.56775866408
    # line_p1=frame0.unproject(3083, frame0.height-986, 1.0) #4132.291660151611, 2813.56775866408
    line_p1=frame0.unproject(3083, 986, 1.0) #4132.291660151611, 2813.56775866408
    line=Mesh()
    line.create_line_strip_from_points([line_p0, line_p1])
    Scene.show(line, "line")

    epipolar_image=frame1.draw_projected_line(line_p0, line_p1, 1)
    Gui.show(epipolar_image, "epipolar_image")



    #test projecting and unprojecting
    # p_u=frame0.unproject(3083, 986, 1.0) 
    # p_u_roundback=frame0.project(p_u) 
    # print("puroundback ", p_u_roundback)


    while True:
        # view.m_camera.from_frame(frame1, True)



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
# test_deep_voxels()
# test_llff()
# test_blender_fb()
# test_usc_hair()
test_phenorob_cp1()
