#pragma once

#include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it



class TensorReel
{
public:
    TensorReel();

    //we need this in order to decide if we access the mask_reel. We can't use the tensor.defined because it will alreayd give an error when we do the accessor when calling the cuda kernel
    bool has_mask; 

    torch::Tensor rgb_reel; 
    torch::Tensor mask_reel; 
    torch::Tensor K_reel; 
    torch::Tensor tf_cam_world_reel; 
    torch::Tensor tf_world_cam_reel; 


private:

    
}; 
