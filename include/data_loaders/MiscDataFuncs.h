#pragma once


//eigen
#include <Eigen/Core>

#ifdef WITH_TORCH
    #include "data_loaders/TensorReel.h"
#endif

//my stuff
#include "easy_pbr/Frame.h"



class MiscDataFuncs
{
public:
    MiscDataFuncs();

    #ifdef WITH_TORCH
        static TensorReel frames2tensors(const std::vector< easy_pbr::Frame >& frames);
    #endif
    

private:


    //internal

};

