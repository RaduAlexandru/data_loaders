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


def show_3D_points(points_3d_tensor, color=None):
    mesh=Mesh()
    mesh.V=points_3d_tensor.detach().double().reshape((-1, 3)).cpu().numpy()

    if color is not None:
        color_channels_last=color.permute(0,2,3,1).detach() # from n,c,h,w to N,H,W,C
        color_channels_last=color_channels_last.view(-1,3).contiguous()
        # color_channels_last=color_channels_last.permute() #from bgr to rgb
        color_channels_last=torch.index_select(color_channels_last, 1, torch.LongTensor([2,1,0]).cuda() ) #switch the columns so that we grom from bgr to rgb
        mesh.C=color_channels_last.detach().double().reshape((-1, 3)).cpu().numpy()
        mesh.m_vis.set_color_pervertcolor()

    mesh.m_vis.m_show_points=True
    # Scene.show(mesh, name)

    return mesh


def ndc_rays (H , W , fx, fy , near , rays_o , rays_d ):
    # Shift ray origins to near plane
    t = -( near + rays_o [... , 2]) / rays_d [... , 2]
    rays_o = rays_o + t[... , None ] * rays_d
    # Projection
    o0 = -1./(W/( 2.* fx ) ) * rays_o [... , 0] / rays_o [... , 2]
    o1 = -1./(H/( 2.* fy ) ) * rays_o [... , 1] / rays_o [... , 2]
    o2 = 1. + 2. * near / rays_o [... , 2]
    d0 = -1./(W/( 2.* fx ) ) * ( rays_d [... , 0]/ rays_d [... , 2] - \
    rays_o [... , 0]/ rays_o [... , 2])
    d1 = -1./(H/( 2.* fy ) ) * ( rays_d [... , 1]/ rays_d [... , 2] - \
    rays_o [... , 1]/ rays_o [... , 2])
    d2 = -2. * near / rays_o [... , 2]
    # print("o0", o0.shape)
    # print("d0", d0.shape)
    # rays_o = tf . stack ([o0 ,o1 , o2], -1)
    # rays_d = tf . stack ([d0 ,d1 , d2], -1)
    rays_o = torch.cat([ o0.unsqueeze(1), o1.unsqueeze(1), o2.unsqueeze(1)    ], 1)
    # print("rays_o", rays_o.shape)
    rays_d = torch.cat([ d0.unsqueeze(1), d1.unsqueeze(1), d2.unsqueeze(1)    ], 1)
    return rays_o , rays_d



def test_volref():
    loader=DataLoaderVolRef(config_path)
    loader.start()

    first=False

    while True:
        if(loader.has_data() and not first ): 
        # if(loader.has_data()  ): 

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
            #move the frame in front a bit so that the clouds actually starts at zero bcause currently we have part of it in the negative z and part in the positive z

            # near =0.001
            # far = 9999

            near =0.2
            far = 9999

            show_ndc = False
            if show_ndc:
                V=cloud.V.copy()
                V[:,2:3] -=2
                cloud.V=V
                Scene.show(cloud, "cloud")

                #make the clouds into NDC
                cloud.remove_vertices_at_zero()
                # xyz2NDC = intrinsics_to_opengl_proj(frame_color.K, frame_color.width, frame_color.height, 0.01, 10)
                # n=0.001 #near
                # f=10 #far
                # # r= frame_color.width #right
                # # t= frame_color.height #top
                # r= 1.0 #right
                # t= 1.0
                # xyz2NDC = np.matrix([
                #                 [n/r, 0, 0, 0], 
                #                 [0, n/t, 0, 0], 
                #                 [0, 0, -(f+n)/(f-n), 2*f*n/(f-n) ], 
                #                 [0, 0, -1, 0], 
                #                 ])
                # max_size = np.maximum(frame_color.width, frame_color.height)
                xyz2NDC = intrinsics_to_opengl_proj(frame_depth.K, frame_depth.width, frame_depth.height, near, far)
                # xyz2NDC = intrinsics_to_opengl_proj(frame_color.K, frame_color.width/max_size, frame_color.height/max_size, 0.001, 10)
                V_xyzw = np.c_[ V, np.ones(cloud.V.shape[0]) ]
                V_clip = np.matmul(V_xyzw, xyz2NDC)
                print("V_clip has min max", np.min(V_clip), " ", np.max(V_clip)  )
                V_ndc= V_clip[:,0:3]/(V_clip[:,3:4])
                V_ndc /= frame_color.width # get it from the [0,frame.width] to [0,1]
                #flip y and z
                # V_ndc[:, 1] = -V_ndc[:, 1] 
                # V_ndc[:, 2] = -V_ndc[:, 2] 
                # V_ndc[:, 0] = -V_ndc[:, 0] 
                print("V_ndc has min max", np.min(V_ndc), " ", np.max(V_ndc)  )
                print("V_ndc, xy has min max", np.min(V_ndc[:,0:2]), " ", np.max(V_ndc[:,0:2])  )
                cloudNDC=Mesh()
                cloudNDC.V=V_ndc
                cloudNDC.C = cloud.C.copy()
                cloudNDC.m_vis.m_show_points=True
                cloudNDC.m_vis.set_color_pervertcolor()
                # Scene.show(cloudNDC, "cloudNDC")


                # #go from NDC back to original cloud
                # NDC2xyz = np.linalg.inv(xyz2NDC)
                # V_ndc = V_ndc*frame_color.width
                # V_ndc_xyzw = np.c_[ V_ndc, np.ones(cloud.V.shape[0]) ]
                # V_xyzw= np.matmul(V_ndc_xyzw, NDC2xyz) 
                # V_xyz= V_xyzw[:,0:3]/(V_xyzw[:,3:4])
                # roundback = Mesh()
                # roundback.V=V_xyz
                # roundback.C = roundback.C.copy()
                # roundback.m_vis.m_show_points=True
                # roundback.m_vis.set_color_pervertcolor()
                # Scene.show(roundback, "roundback")


            #get rays and show them
            Scene.show(cloud, "cloud")
            ray_dirs_mesh=frame_depth.pixels2dirs_mesh()
            # ray_dirs=ray_dirs_mesh.V.copy()
            ray_dirs=torch.from_numpy(ray_dirs_mesh.V.copy()).float()
            depth_per_pixel =   torch.ones([frame_depth.height* frame_depth.width, 1], dtype=torch.float32) 
            depth_per_pixel.fill_(near)
            camera_center=torch.from_numpy( frame_depth.pos_in_world() )
            camera_center=camera_center.view(1,3)
            points3D = camera_center + depth_per_pixel*ray_dirs #N,3,H,W
            rays_vis = show_3D_points(points3D)
            rays_vis.NV = ray_dirs.detach().double().reshape((-1, 3)).cpu().numpy()
            rays_vis.m_vis.m_show_normals=True
            Scene.show(rays_vis, "rays" )

            #make te origins and direcitons similar to what nerf does in here, at the end is pytorch code ndc_derivation.pdf
            rays_o = points3D
            rays_d = ray_dirs
            ndc_origins, ndc_dirs = ndc_rays (frame_depth.height , frame_depth.width , frame_depth.K[0,0], frame_depth.K[1,1], near, rays_o , rays_d )
            NDC_rays_vis = show_3D_points(ndc_origins)
            NDC_rays_vis.NV = ndc_dirs.detach().double().reshape((-1, 3)).cpu().numpy()
            NDC_rays_vis.m_vis.m_show_normals=True
            Scene.show(NDC_rays_vis, "NDC_rays_vis" )



            



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

def test_phenorob():
    loader=DataLoaderPhenorob(config_path)
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

            near = frame.get_extra_field_float("near")
            print("near is ", near)
            # frame.add_extra_field("test", 54.0)
            # roundback= frame.get_extra_field_float("test")
            # print("roundback is ", roundback)
        
        if loader.is_finished():
            print("resetting")
            loader.reset()
            print("scene scale is ", Scene.get_scale())

        view.update()


test_volref()
# test_img()
# test_img_ros()
# test_cloud_ros()
# test_semantickitti()
# test_scannet()
# test_stanford3dscene()
# test_shapenet_img()
# test_nerf()
# test_phenorob()
# test_colmap()
# test_easypbr()
# test_srn()
# test_dtu()
# test_deep_voxels()
# test_llff()



