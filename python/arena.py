#!/usr/bin/env python3.6

import sys
import os
easy_pbr_path= os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../build/_deps/easy_pbr-build')
sys.path.append( easy_pbr_path )
MBZIRC_CH1_path= os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../build')
sys.path.append( MBZIRC_CH1_path )
from easypbr  import *
from MBZIRC_CH1  import *

config_file="render_copter.cfg"

#COPTER----------
# copter=Mesh("/media/rosu/Data/phd/mbzirc/mavic_pro_model/Mavic/model_fuse_metric.ply")
copter=Mesh("/media/rosu/Data/data/copters/m600-lowres_legsup.ply")
# copter=Mesh("/media/rosu/Data/data/3d_objs/ballon_allformats/ballon_no_floor.ply")
copter.m_vis.m_solid_color=[0.0, 0.0, 0.0]
copter.m_vis.m_roughness=0.75
# copter.rotate_x_axis(-90)
# copter.move_in_x(-20)
# copter.move_in_z(-20)
# copter.move_in_y(5.4)
copter.id=2
Scene.show(copter, "copter")

# # BALLOON------------
# balloon=Mesh("/media/rosu/Data/data/3d_objs/ballon_allformats/ballon_no_floor_no_string_metric2.ply")
# balloon.m_vis.m_solid_color=[1.0, 0.0, 0.0]
# balloon.m_vis.m_roughness=0.4
# balloon.rotate_x_axis(-90)
# # copter.move_in_x(-20)
# # copter.move_in_z(-20)
# # copter.move_in_y(5.4)
# balloon.id=1
# Scene.show(balloon, "balloon")

# #WALLS-----------
# walls=Mesh()
# wall_orig=Mesh("/media/rosu/Data/phd/c_ws/src/mbzirc_challenge_1/net_3.ply")
# wall_orig.m_vis.m_solid_color=[0.0, 0.0, 0.0]
# wall_orig.m_vis.m_roughness=0.75
# walls.m_vis=wall_orig.m_vis #the full walls should be of black color too
# wall_orig.rotate_x_axis(-90) #fix the blender up direction
#     #wall_1
# wall_1=wall_orig.clone()
# walls.add(wall_1)
#     #wall_2
# wall_2=wall_orig.clone()
# wall_2.rotate_y_axis(90)
# walls.add(wall_2)
#     #wall_3
# wall_3=wall_orig.clone()
# wall_3.move_in_x(-40)
# walls.add(wall_3)
#     #wall_4 
# wall_4=wall_orig.clone()
# wall_4.rotate_y_axis(90)
# wall_4.move_in_z(-40)
# walls.add(wall_4)
# Scene.show(walls, "wall")

# #TOWERS
# towers=Mesh()
# tower_orig=Mesh("/media/rosu/Data/phd/c_ws/src/mbzirc_challenge_1/tower_2.ply")
# tower_orig.m_vis.m_solid_color=[0.0, 0.0, 0.0]
# tower_orig.m_vis.m_roughness=0.75
# towers.m_vis=tower_orig.m_vis #the full walls should be of black color too
# tower_orig.rotate_x_axis(-90) #fix the blender up direction
#     #tower_1
# tower_1=tower_orig.clone()
# towers.add(tower_1)
#     #tower_2
# tower_2=tower_orig.clone()
# tower_2.move_in_z(-40)
# towers.add(tower_2)
#     #tower_3
# tower_3=tower_orig.clone()
# tower_3.move_in_x(-40)
# towers.add(tower_3)
#     #tower_4
# tower_4=tower_orig.clone()
# tower_4.move_in_z(-40)
# tower_4.move_in_x(-40)
# towers.add(tower_4)
# Scene.show(towers, "towers")

# #SUPPORT
# supports=Mesh()
# support_orig=Mesh("/media/rosu/Data/phd/c_ws/src/mbzirc_challenge_1/support_2.ply")
# support_orig.m_vis.m_solid_color=[0.0, 0.0, 0.0]
# support_orig.m_vis.m_roughness=0.75
# supports.m_vis=support_orig.m_vis #the full walls should be of black color too
# support_orig.rotate_x_axis(-90) #fix the blender up direction
#     #support_1
# support_1=support_orig.clone()
# supports.add(support_1)
#     #support_2
# support_2=support_orig.clone()
# support_2.move_in_z(-40)
# supports.add(support_2)
#     #support_3
# support_3=support_orig.clone()
# support_3.move_in_x(-40)
# supports.add(support_3)
#     #support_4
# support_4=support_orig.clone()
# support_4.move_in_z(-40)
# support_4.move_in_x(-40)
# supports.add(support_4)
# Scene.show(supports, "supports")

# #FLOOR
# floor=Mesh()
# floor.create_floor(0, 1)
# floor.V=floor.V*6
# floor.move_in_x(-20)
# floor.move_in_z(-20)
# floor.m_vis.m_solid_color=[0.5, 0.5, 0.5]
# floor.m_vis.m_roughness=0.75
# Scene.show(floor, "floor")






view=Viewer.create(config_file) #first because it needs to init context

# synth=SyntheticGenerator.create(config_file, view)


while True:
    view.update()
    # synth.detect_balloon(view)
    # synth.detect_copter(view)

    # record(synth.tex,"/med")
    # record(synth.tex, "/)
    # record(.final_fbo.)



