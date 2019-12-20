#!/usr/bin/env python3.6

import os
import numpy as np
import sys
from easypbr  import *
from dataloaders import *
np.set_printoptions(threshold=sys.maxsize)

config_file="test_loader.cfg"

config_path=os.path.join( os.path.dirname( os.path.realpath(__file__) ) , '../config', config_file)
view=Viewer.create(config_path) #first because it needs to init context
# loader=DataLoaderShapeNetPartSeg(config_path)
loader=DataLoaderVolRef(config_path)
loader.start()


while True:
    if(loader.has_data() ): 
        # cloud=loader.get_cloud()
        # Scene.show(cloud,"cloud")

        #volref 
        print("got frame")
        frame_color=loader.get_color_frame()
        frame_depth=loader.get_depth_frame()


        # Gui.show_rgb(frame_color, "rgb")
        # Gui.show(frame_color.rgb_32f, "rgb")
        # rgb_8u=frame_color.rgb_8u
        # help(rgb_8u)
        Gui.show(frame_color.rgb_32f, "rgb")
        # help(rgb_32f)
        # sys.exit("exit")

        frustum_mesh=frame_color.create_frustum_mesh(0.1)
        frustum_mesh.m_vis.m_line_width=3
        frustum_name="frustum"
        Scene.show(frustum_mesh, frustum_name)

        cloud=frame_depth.backproject_depth()
        frame_color.assign_color(cloud) #project the cloud into this frame and creates a color matrix for it
        # Scene.show(cloud, "cloud"+str(nr_cloud))
        Scene.show(cloud, "cloud")
    view.update()

