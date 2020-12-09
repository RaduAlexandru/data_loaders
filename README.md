# DataLoaders

This contains various data loaders for a series of datasets:

- [ScanNet] : Point clouds of rooms with annotations for 20 classes (bed, furnite, wall, etc.)
- [SemanticKitti] : Point clouds from a car driving in urban scenario annotated with 19 classes.
- [ShapeNetSem] : Point clouds of various objects (airplance, motorbike) with part-based annotations
- [VolumetricRefinement] : RGB-D images used in the work of Zollhöfer et al: Shading-based Refinement on Volumetric Signed Distance Functions
- [Pheno4D] : Point clouds of maize and tomato plants together with instance segmentation annotations

Example of usage and loading of the data can be found in python/test_loader.py
Each loader is controlled by a config file which can be found in config/test_loader.cfg

### Build and install: 
To build and install the example, you must have first installed [EasyPBR]. Afterwards the dataloader can be cloned and compiled with
```sh
$ git clone --recursive https://github.com/RaduAlexandru/data_loaders
$ cd data_loaders
$ make
```
Optionally, if you have Robot Operating System (ROS), you can clone data_loaders in your catkin workspace and you will ahve access to dataloaders for ROS bags and ROS topics.
```sh
$ cd YOUR_ROS_WORKSPACE/src
$ git clone --recursive https://github.com/RaduAlexandru/data_loaders
$ cd data_loaders
$ make
```



   [ScanNet]: <http://www.scan-net.org/>
   [SemanticKitti]: <http://semantic-kitti.org/>
   [ShapeNetSem]: <https://www.shapenet.org/>
   [VolumetricRefinement]: <http://graphics.stanford.edu/projects/vsfs/>
   [EasyPBR]: <https://github.com/RaduAlexandru/easy_pbr>
   [Pheno4D]: <TODO>
