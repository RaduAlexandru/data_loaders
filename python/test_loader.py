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


            frustum_mesh=frame.create_frustum_mesh(0.02)
            # frustum_mesh=frame.create_frustum_mesh(0.2)
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

    def map_range_tensor( input_val, input_start, input_end,  output_start,  output_end):
        # input_clamped=torch.clamp(input_val, input_start, input_end)
        # input_clamped=max(input_start, min(input_end, input_val))
        input_clamped=torch.clamp(input_val, input_start, input_end)
        return output_start + ((output_end - output_start) / (input_end - input_start)) * (input_clamped - input_start)


    for s_idx in range(loader.nr_scans()):
        scan=loader.get_scan_with_idx(s_idx)
        for b_idx in range(scan.nr_blocks()):
            block=scan.get_block_with_idx(b_idx)
            #show all the images from the first block only
            if b_idx==0:
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
                    if f_idx==0 and frame.has_extra_field("visible_points"):
                        visible_points=frame.get_extra_field_mesh("visible_points")
                        loader.load_mesh(visible_points)
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
                        # frame.naive_splat(visible_points, visible_points_depth_np)

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




            # load the photoneo frame from this block
            if loader.dataset_type()=="kalibr":
                photoneo_frame=block.get_photoneo_frame()
                photoneo_frame.load_images()
                frustum_mesh=photoneo_frame.create_frustum_mesh(0.05, True, 256)
                frustum_mesh.m_vis.m_line_width=1
                Scene.show(frustum_mesh, "photoneo_frustum_"+str(photoneo_frame.cam_id) )
                #load photoneo cloud
                photoneo_mesh=block.get_photoneo_mesh()
                loader.load_mesh(photoneo_mesh)
                # photoneo_mesh.load_from_file(photoneo_mesh.m_disk_path)
                #show the confidence 
                # Gui.show(photoneo_frame.confidence, "confidence_photoneo_"+str(photoneo_frame.cam_id))
                #show depth
                # Gui.show(photoneo_frame.depth.normalize_range(), "depth_photoneo_"+str(photoneo_frame.cam_id))
                #show both depth and confidence
                Gui.show(photoneo_frame.depth.normalize_range(), "depth_photoneo_"+str(photoneo_frame.cam_id), photoneo_frame.confidence, "confidence_photoneo_"+str(photoneo_frame.cam_id))
                #color the first cloud
                if b_idx==0:
                    frame0=loader.get_scan_with_idx(0).get_block_with_idx(0).get_rgb_frame_with_idx(0)
                    frame0.load_images()
                    photoneo_mesh=frame0.assign_color(photoneo_mesh)
                    print("frame 0 K is ", frame0.K)
                    print("width", frame0.width)
                Scene.show(photoneo_mesh, "photoneo_mesh_"+str(b_idx))
                # backproject depth
                # photoneo_depth_backproj=photoneo_frame.depth2world_xyz_mesh()
                # photoneo_depth_backproj.m_vis.m_point_color=[0.7, 0.3, 0.3]
                # Scene.show(photoneo_depth_backproj, "photoneo_depth_backproj_"+str(b_idx))


            #load the dense cloud for this block
            if loader.loaded_dense_cloud():
                dense_cloud=block.get_dense_cloud()
                loader.load_mesh(dense_cloud)
                # dense_cloud.load_from_file(dense_cloud.m_disk_path)
                print("dense_cloud", dense_cloud.model_matrix.matrix() )
                dense_cloud.apply_model_matrix_to_cpu(True)
                dense_cloud.recalculate_min_max_height()
                Scene.show(dense_cloud, "dense_cloud_"+str(b_idx))







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
test_easypbr()
# test_srn()
# test_dtu()
# test_deep_voxels()
# test_llff()
# test_blender_fb()
# test_usc_hair()
# test_phenorob_cp1()
