# DataLoaders

This contains various data loaders for a series of datasets:

- [ScanNet] : Point clouds of rooms with annotations for 20 classes (bed, furnite, wall, etc.)
- [SemanticKitti] : Point clouds from a car driving in urban scenario annotated with 19 classes.
- [ShapeNetSem] : Point clouds of various objects (airplance, motorbike) with part-based annotations
- [VolumetricRefinement] : RGB-D images used in the work of Zollhöfer et al: Shading-based Refinement on Volumetric Signed Distance Functions

Example of usage and loading of the data can be found in python/test_loader.py

### Build and install: 
To build and install the example, you must have first installed [EasyPBR]. Afterwards the dataloader requires to be cloned and build inside a ROS workspace
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
