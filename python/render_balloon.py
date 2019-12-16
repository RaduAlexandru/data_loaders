#!/usr/bin/env python3.6

import sys
import os
from easypbr  import *
from mbzirc1  import *

config_file="render_balloon.cfg"


# BALLOON------------
# balloon=Mesh("/media/rosu/Data/data/3d_objs/ballon_allformats/ballon_no_floor_no_string_metric2.ply")
balloon=Mesh("/media/rosu/Data/data/3d_objs/ballon_allformats/balloon_sphere.obj")
balloon.m_vis.m_solid_color=[1.0, 1.0, 1.0]
balloon.m_vis.m_roughness=0.5
balloon.translate_model_matrix([0.0, 2.5, 0.0])
balloon.id=1
Scene.show(balloon, "balloon")


#baloon pole
pole=Mesh("/media/rosu/Data/data/3d_objs/ballon_allformats/balloon_pole.obj")
pole.m_vis.m_solid_color=[0.1, 0.1, 0.1]
pole.m_vis.m_roughness=0.75
pole.id=0
Scene.show(pole, "pole")

##net 
net=Mesh("/media/rosu/Data/data/mbzirc/net2.obj")
net.m_vis.m_solid_color=[0.0, 0.0, 0.0]
net.m_vis.m_roughness=0.75
Scene.show(net, "net")

#hide grid 
grid_floor=Scene.get_mesh_with_name("grid_floor")
grid_floor.m_vis.m_is_visible=False

#hierarchy 
balloon.add_child(pole)



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



# iter=0
# max_nr_imgs=5000

# while True:
#     iter+=1
#     if(iter< 50):
#         continue

#     view.update()


#     if(iter>=max_nr_imgs):
#         break




