#!/usr/bin/env python3.6

import sys
import os
import time
# easy_pbr_path= os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../build/_deps/easy_pbr-build')
# sys.path.append( easy_pbr_path )
# MBZIRC_CH1_path= os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../build')
# sys.path.append( MBZIRC_CH1_path )
from easypbr  import *
from mbzirc1  import *

config_file="render_copter.cfg"

#COPTER----------
# copter=Mesh("/media/rosu/Data/data/copters/mavic_pro_metric.ply")
copter=Mesh("/media/rosu/Data/data/copters/m600-lowres_legsup_nogps_straighter_arms.ply")
# copter=Mesh("/media/rosu/Data/data/3d_objs/ballon_allformats/ballon_no_floor.ply")
copter.m_vis.m_solid_color=[0.1, 0.1, 0.1]
copter.m_vis.m_roughness=0.75
copter.id=2
Scene.show(copter, "copter")

#propeller
propeller1=Mesh("/media/rosu/Data/data/copters/propeller.ply")
propeller1.m_vis.m_solid_color=[0.1, 0.1, 0.1]
propeller1.m_vis.m_roughness=0.75
propeller1.rotate_model_matrix([1.0, 0.0, 0.0], 90)
propeller1.translate_model_matrix([0.53, 0.13, 0.3])
# propeller.rotate_model_matrix_local([0.0, 1.0, 0.0], 30)
propeller1.id=2
Scene.show(propeller1, "propeller1")
#propeller2
propeller2=propeller1.clone()
propeller2.rotate_model_matrix([0.0, 1.0, 0.0], 60)
Scene.show(propeller2, "propeller2")
#propeller3
propeller3=propeller2.clone()
propeller3.rotate_model_matrix([0.0, 1.0, 0.0], 60)
Scene.show(propeller3, "propeller3")
#propeller4
propeller4=propeller3.clone()
propeller4.rotate_model_matrix([0.0, 1.0, 0.0], 60)
Scene.show(propeller4, "propeller4")
#propeller5
propeller5=propeller4.clone()
propeller5.rotate_model_matrix([0.0, 1.0, 0.0], 60)
Scene.show(propeller5, "propeller5")
#propeller6
propeller6=propeller5.clone()
propeller6.rotate_model_matrix([0.0, 1.0, 0.0], 60)
Scene.show(propeller6, "propeller6")


##net 
net=Mesh("/media/rosu/Data/data/mbzirc/net2.obj")
net.m_vis.m_solid_color=[0.0, 0.0, 0.0]
net.m_vis.m_roughness=0.75
Scene.show(net, "net")

#ball
ball=Mesh("/media/rosu/Data/data/3d_objs/sphere2.obj")
ball.m_vis.m_solid_color=[1.0, 0.0, 0.0]
ball.m_vis.m_roughness=0.75
ball.translate_model_matrix([0.0, -1.45, 0.0])
ball.id=3
Scene.show(ball, "ball")

#make pole
pole=Mesh("/media/rosu/Data/data/copters/pole2.obj")
pole.m_vis.m_solid_color=[0.1, 0.1, 0.1]
pole.m_vis.m_roughness=0.7
Scene.show(pole, "pole")
#put also the transformer pole which will be the one actually rotating. and having the ball as a child. We make like this because it's easier on the hierarchy
# pole_transformed=pole.clone()
# pole.m_vis.m_is_visible=False
# Scene.show(pole_transformed, "pole_transformed")

#hide grid 
grid_floor=Scene.get_mesh_with_name("grid_floor")
grid_floor.m_vis.m_is_visible=False

#hierarchy 
ball.name="ball"
copter.add_child(pole)
# copter.add_child(ball)
pole.add_child(ball)
copter.add_child(propeller1)
copter.add_child(propeller2)
copter.add_child(propeller3)
copter.add_child(propeller4)
copter.add_child(propeller5)
copter.add_child(propeller6)


config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
view=Viewer.create(config_path) #first because it needs to init context

synth=SyntheticGenerator.create(config_path, view)

iter=0
max_nr_imgs=5000
environment_maps=[]
environment_maps.append("/media/rosu/Data/data/sibl/Desert_Highway/Road_to_MonumentValley_Ref.hdr")
environment_maps.append("/media/rosu/Data/data/sibl/new/GrandCanyon_C_YumaPoint/GCanyon_C_YumaPoint_3k.hdr")
environment_maps.append("/media/rosu/Data/data/sibl/new/Helipad_Afternoon/LA_Downtown_Afternoon_Fishing_3k.hdr")
environment_maps.append("/media/rosu/Data/data/sibl/new/Helipad_GoldenHour/LA_Downtown_Helipad_GoldenHour_3k.hdr")
environment_maps.append("/media/rosu/Data/data/sibl/new/MonValley_DirtRoad/MonValley_G_DirtRoad_3k.hdr")
environment_maps.append("/media/rosu/Data/data/sibl/new/MonValley_Lookout/MonValley_A_LookoutPoint_2k.hdr")
environment_maps.append("/media/rosu/Data/data/sibl/new/Serpentine_Valley/Serpentine_Valley_3k.hdr")
environment_maps.append("/media/rosu/Data/data/sibl/new/Shiodome_Stairs/10-Shiodome_Stairs_3k.hdr")
environment_maps.append("/media/rosu/Data/data/sibl/new/Tropical_Beach/Tropical_Beach_3k.hdr")
environment_maps.append("/media/rosu/Data/data/sibl/new/Wooden_Door/WoodenDoor_Ref.hdr")
map_idx=0
change_environment_map_every=max_nr_imgs/len(environment_maps)
print("changing map every", change_environment_map_every)

while True:
    iter+=1
    if(iter< 50):
        continue

    #change the environment is necessary
    if(iter%change_environment_map_every==0):
        view.load_environment_map(environment_maps[map_idx])
        map_idx+=1

    view.update()
    if(iter>=max_nr_imgs):
        break

#wait until all writing got disk has finished
time.sleep( 6 )


