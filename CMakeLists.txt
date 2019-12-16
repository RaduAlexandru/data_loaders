cmake_minimum_required(VERSION 2.8 FATAL_ERROR) # we need at least 3.11 because we use the FetchContent feature to manage dependencies and at least 3.14 because we use FetchContent_MakeAvailable

project(data_loaders)


###   VARIABLES   ##############################################################
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -Wall -O3") #we need c++17 because this solves alignment issues with eigen http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1409
set(CMAKE_CXX_STANDARD 17) #needed also by the pybind so it can build itself in c++14 mode so we can use py:overload_cast
# set(FETCHCONTENT_QUIET NO)


####   GLOBAL OPTIONS   ###https://stackoverflow.com/questions/15201064/cmake-conditional-preprocessor-define-on-code


######   PACKAGES   ############################################################
find_package(catkin REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(EasyPBR REQUIRED)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/pybind11)
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs cv_bridge pcl_ros image_transport )


###   INCLUDES   #########################################################
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/deps)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EASYPBR_INCLUDE_DIR})
catkin_package( INCLUDE_DIRS include  CATKIN_DEPENDS roscpp std_msgs cv_bridge pcl_ros image_transport )


###   SOURCES   #################################################################
set(MY_SRC  
    # ${PROJECT_SOURCE_DIR}/src/Frame.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderVolRef.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderImgRos.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderScanNet.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderShapeNetPartSeg.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataTransformer.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderSemanticKitti.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderStanfordIndoor.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderToyExample.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderCloudRos.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderRueMonge.cxx
    # ${PROJECT_SOURCE_DIR}/src/RosBagPlayer.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderPNG.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderModelNet40.cxx
)


###   SET ALL THE GLOBAL OPTIONS #########################################


#lib 
add_library( dataloaders_cpp SHARED ${MY_SRC}  ${DEPS_SRC}  )


##pybind
pybind11_add_module(dataloaders ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )

###   EXECUTABLE   #######################################
# add_executable(run_data_loaders ${PROJECT_SOURCE_DIR}/src/main.cxx  )


###   LIBS   ###############################################
set(LIBS ${LIBS} ${catkin_LIBRARIES} ${EASYPBR_LIBRARY} ${OpenCV_LIBS})


target_link_libraries(dataloaders_cpp ${LIBS} )
target_link_libraries(dataloaders PRIVATE dataloaders_cpp )
# target_link_libraries(run_data_loaders PRIVATE data_loaders_cpp  )


#definitions for cmake variables that are necesarry during runtime
target_compile_definitions(dataloaders_cpp PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}") #point to the cmakelist folder of the easy_pbr
target_compile_definitions(dataloaders_cpp PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}") # points to the CMakeList folder of whichever project included easy_pbr


