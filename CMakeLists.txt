# cmake_minimum_required(VERSION 2.8 FATAL_ERROR) # we need at least 3.11 because we use the FetchContent feature to manage dependencies and at least 3.14 because we use FetchContent_MakeAvailable
cmake_minimum_required(VERSION 2.8...3.22)

project(data_loaders)


###   VARIABLES   ##############################################################
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -O3 -Wno-reorder") #we need c++17 because this solves alignment issues with eigen http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1409
set(CMAKE_CXX_STANDARD 17) #we need c++17 because this solves alignment issues with eigen http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1409
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)






# libigl
# libigl options: choose between header only and compiled static library
# Header-only is preferred for small projects. For larger projects the static build
# considerably reduces the compilation times
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         OFF)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           OFF)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          OFF)
option(LIBIGL_WITH_PNG               "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  OFF)
option(LIBIGL_WITH_XML               "Use XML"            OFF)




# set(PYTHON_VERSION 3.6 CACHE STRING "" FORCE)


# #pybind package
# #setting a python version to 3.6 for pybind so that the bindings work in 3.6 https://github.com/pybind/pybind11/issues/748
# unset(PYTHONINTERP_FOUND CACHE)
# unset(PYTHON_EXECUTABLE CACHE)
# unset(PYTHON_VERSION_STRING CACHE)
# unset(PYTHON_VERSION_MAJOR CACHE)
# unset(PYTHON_VERSION_MINOR CACHE)
# unset(PYTHON_VERSION_PATCH CACHE)
# unset(PYBIND11_INCLUDE_DIR CACHE)
# unset(PYTHON_INCLUDE_DIRS CACHE)
# unset(PYTHON_LIBRARIES CACHE)
# unset(PYTHON_MODULE_PREFIX CACHE)
# unset(PYTHON_MODULE_EXTENSION CACHE)
# unset(PYTHON_LIBRARY CACHE)
# set(PYBIND11_PYTHON_VERSION 3.6 CACHE STRING "" FORCE)
# message("PYBIND11_PYTHON_VERSION is" , ${PYBIND11_PYTHON_VERSION})



####   GLOBAL OPTIONS   ###https://stackoverflow.com/questions/15201064/cmake-conditional-preprocessor-define-on-code


######   PACKAGES   ############################################################
find_package(catkin QUIET)
set(catkin_FOUND 0 )
message("catkin found is ${catkin_FOUND}" )
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(LIBIGL REQUIRED)
find_package(EasyPBR REQUIRED)
# add_subdirectory(${PROJECT_SOURCE_DIR}/deps/pybind11)
if(${catkin_FOUND})
	find_package(catkin REQUIRED COMPONENTS roscpp std_msgs cv_bridge pcl_ros image_transport tf2_ros tf2_eigen)
endif()
#try to compile with pytorch if you can
# set(PYTORCH_PATH "/opt/pytorch")
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ";${PYTORCH_PATH};")
# get and append paths for finding dep
execute_process( #do it like this https://github.com/facebookresearch/hanabi_SAD/blob/6e4ed590f5912fcb99633f4c224778a3ba78879b/rela/CMakeLists.txt#L10
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
#sometimes we want to use libtorch because the pytorch one has a different abi
# set(TorchPath "/home/rosu/work/ws/libtorch-cxx11-abi-shared-with-deps-1.9.0+cu111/libtorch")
# set(TorchPath "/home/rosu/work/ws/libtorch-cxx11-abi-shared-with-deps-1.9.1+cu111/libtorch")
# message("TORCH path-------------------", ${TorchPath})
if(TorchPath STREQUAL "")
    set(TORCH_FOUND False)
else()
    list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
    find_package(Torch)
endif()
# set(TORCH_FOUND False)






add_subdirectory( ${EASYPBR_SRC_PATH}/deps/pybind11 ./../../build/pybind11 EXCLUDE_FROM_ALL)
#pybind attempt 3
pybind11_add_module(dataloaders ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/yaml-cpp)


###   INCLUDES   #########################################################
include_directories(${catkin_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/deps)
include_directories(${CMAKE_SOURCE_DIR}/deps/cnpy)
include_directories(${CMAKE_SOURCE_DIR}/deps/libigl/include)
include_directories(${CMAKE_SOURCE_DIR}/extern)
include_directories(${CMAKE_SOURCE_DIR}/deps/yaml-cpp/include )
# include_directories(${CMAKE_SOURCE_DIR}/deps/pybind11/include)
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${EASYPBR_INCLUDE_DIR})
if(${TORCH_FOUND})
    include_directories(${TORCH_INCLUDE_DIRS})
endif()
if(${catkin_FOUND})
    catkin_package( INCLUDE_DIRS include  CATKIN_DEPENDS roscpp std_msgs cv_bridge pcl_ros image_transport tf2_ros tf2_eigen )
endif()


###   SOURCES   #################################################################
set(MY_SRC
    # ${PROJECT_SOURCE_DIR}/src/Frame.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderVolRef.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderImgRos.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderScanNet.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderShapeNetPartSeg.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderShapeNetImg.cxx
    ${PROJECT_SOURCE_DIR}/src/DataTransformer.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderSemanticKitti.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderPheno4D.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderPhenorobCP1.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderStanfordIndoor.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderToyExample.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderCloudRos.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderRueMonge.cxx
    # ${PROJECT_SOURCE_DIR}/src/RosBagPlayer.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderImg.cxx
    # ${PROJECT_SOURCE_DIR}/src/DataLoaderModelNet40.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderStanford3DScene.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderNerf.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderEasyPBR.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderColmap.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderSRN.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderDTU.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderDeepVoxels.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderLLFF.cxx
    ${PROJECT_SOURCE_DIR}/src/DataLoaderMultiFace.cxx
    ${PROJECT_SOURCE_DIR}/src/MiscDataFuncs.cxx
    #fb
    ${PROJECT_SOURCE_DIR}/src/fb/DataLoaderBlenderFB.cxx
)
if(${TORCH_FOUND})
    set(MY_SRC ${MY_SRC}
        ${PROJECT_SOURCE_DIR}/src/fb/DataLoaderUSCHair.cxx
    )
endif()
set(DEPS_SRC
    ${PROJECT_SOURCE_DIR}/deps/json11/json11.cpp
    ${PROJECT_SOURCE_DIR}/deps/cnpy/cnpy.cpp
    # ${PROJECT_SOURCE_DIR}/extern/pybind_casters/ndarray_converter.cpp
    )
if (${catkin_FOUND} )
    set(MY_SRC ${MY_SRC}
        ${PROJECT_SOURCE_DIR}/src/DataLoaderImgRos.cxx
        ${PROJECT_SOURCE_DIR}/src/DataLoaderCloudRos.cxx
        ${PROJECT_SOURCE_DIR}/src/RosBagPlayer.cxx
    )
endif()
#loaders that require torch
if(${TORCH_FOUND})
    set(MY_SRC ${MY_SRC}
        ${PROJECT_SOURCE_DIR}/src/fb/DataLoaderUSCHair.cxx
        ${PROJECT_SOURCE_DIR}/src/TensorReel.cxx
    )
endif()


#disable warnings for yaml-cpp and cnpy
set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/deps/cnpy/cnpy.cpp
  PROPERTIES
  COMPILE_FLAGS "-w"
)
set_target_properties(yaml-cpp PROPERTIES COMPILE_FLAGS "-w")


###   SET ALL THE GLOBAL OPTIONS #########################################
if(${catkin_FOUND})
    message("USING ROS")
    add_definitions(-DWITH_ROS)
else()
    message("NOT USING ROS")
endif()


#lib
add_library( dataloaders_cpp SHARED ${MY_SRC}  ${DEPS_SRC}  )







if(${TORCH_FOUND})
    message("USING TORCH")
    target_compile_definitions(dataloaders_cpp PUBLIC WITH_TORCH)
else()
    message("NOT USING TORCH")
endif()




##pybind
# pybind11_add_module(dataloaders ${PROJECT_SOURCE_DIR}/src/PyBridge.cxx )

###   EXECUTABLE   #######################################
# add_executable(run_data_loaders ${PROJECT_SOURCE_DIR}/src/main.cxx  )


###   LIBS   ###############################################
# message("easypbr lib is ", ${EASYPBR_LIBRARY})
set(LIBS ${LIBS} igl::core  ${catkin_LIBRARIES} ${EASYPBR_LIBRARY} ${OpenCV_LIBS}  yaml-cpp  )
if(${TORCH_FOUND})
    set(LIBS ${LIBS} ${TORCH_LIBRARIES} )
    #torch 1.5.0 and above mess with pybind and we therefore need to link against libtorch_python.so also
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
    if(TORCH_PYTHON_LIBRARY)
        message(STATUS "Linking to torch_python_library")
        set(LIBS ${LIBS} ${TORCH_PYTHON_LIBRARY} )
    endif()
endif()



###   TINY PROCESS LIBRARY   #########################################
set (TINY_PROCESS_LIB_SOURCES   ${CMAKE_SOURCE_DIR}/deps/tiny-process-library/process.cpp ${CMAKE_SOURCE_DIR}/deps/tiny-process-library/process_unix.cpp )
add_library( TINY_PROCESS_LIB ${TINY_PROCESS_LIB_SOURCES} )
target_link_libraries(TINY_PROCESS_LIB ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(TINY_PROCESS_LIB PROPERTIES LINKER_LANGUAGE C)
set(LIBS ${LIBS} TINY_PROCESS_LIB)


target_link_libraries(dataloaders_cpp ${LIBS} )
target_link_libraries(dataloaders PRIVATE dataloaders_cpp )
# target_link_libraries(run_data_loaders PRIVATE data_loaders_cpp  )


#definitions for cmake variables that are necesarry during runtime
target_compile_definitions(dataloaders_cpp PRIVATE PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}") #point to the cmakelist folder of the easy_pbr
target_compile_definitions(dataloaders_cpp PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}") # points to the CMakeList folder of whichever project included easy_pbr
