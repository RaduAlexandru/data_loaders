if(LIBIGL_FOUND)
    return()
endif()

find_path(LIBIGL_INCLUDE_DIR igl/readOBJ.h
    HINTS
        ENV LIBIGL
        ENV LIBIGLROOT
        ENV LIBIGL_ROOT
        ENV LIBIGL_DIR
    PATHS
        ${CMAKE_SOURCE_DIR}/deps/libigl/include
    PATH_SUFFIXES include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBIGL
    "\nlibigl not found --- You can add it as a submodule it using:\n\tgit add submodule https://github.com/libigl/libigl.git deps/libigl"
    LIBIGL_INCLUDE_DIR)
mark_as_advanced(LIBIGL_INCLUDE_DIR)

list(APPEND CMAKE_MODULE_PATH "${LIBIGL_INCLUDE_DIR}/../cmake")
include(libigl)
