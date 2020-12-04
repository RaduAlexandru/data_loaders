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

config_file="test_loader.cfg"

config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
view=Viewer.create(config_path) #first because it needs to init context



def test_volref():
    loader=DataLoaderVolRef(config_path)
    loader.start()

    while True:
        if(loader.has_data() ): 

            #volref 
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

            cloud=frame_depth.depth2world_xyz_mesh()
            frame_color.assign_color(cloud) #project the cloud into this frame and creates a color matrix for it
            Scene.show(cloud, "cloud")
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

            # if i==0:
            # if i==35:
            # if (frame.frame_idx==35):
            # if (frame.frame_idx==34):
            # if (frame.frame_idx==34 or frame.frame_idx==35):
            if True:
                loader.start_reading_next_scene()

                Gui.show(frame.rgb_32f, "rgb")
                Gui.show(frame.mask, "mask")
                Gui.show(frame.depth, "depth")
                frustum=frame.create_frustum_mesh(0.1)
                # Scene.show(frustum, "frustum"+ str(frame.frame_idx) )
                Scene.show(frustum, "frustum" )

                cloud=frame.depth2world_xyz_mesh()
                cloud=frame.assign_color(cloud)
                Scene.show(cloud, "cloud")

            i+=1


        if loader.is_finished():
            print("resetting")
            loader.reset()

        view.update()

def test_nerf():
    loader=DataLoaderNerf(config_path)
    loader.set_mode_train()
    loader.start()

    while True:
        if(loader.has_data() ): 

            # print("got frame")
            frame=loader.get_next_frame()


            Gui.show(frame.rgb_32f, "rgb")


            frustum_mesh=frame.create_frustum_mesh(0.2)
            frustum_mesh.m_vis.m_line_width=1
            Scene.show(frustum_mesh, "frustum_"+str(frame.frame_idx) )

            # cloud=frame.depth2world_xyz_mesh()
            # Scene.show(cloud, "cloud")
        
        if loader.is_finished():
            print("resetting")
            loader.reset()

        view.update()

def test_phenorob():
    loader=DataLoaderPhenorob(config_path)
    loader.start()

    while True:
        if(loader.has_data() ): 

            cloud=loader.get_cloud()

            Scene.show(cloud, "cloud" )
        
        if loader.is_finished():
            print("resetting")
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


# test_volref()
# test_img()
# test_img_ros()
# test_cloud_ros()
# test_semantickitti()
# test_scannet()
# test_stanford3dscene()
# test_shapenet_img()
# test_nerf()
test_phenorob()



