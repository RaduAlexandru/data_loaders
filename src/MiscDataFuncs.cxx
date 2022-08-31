#include "data_loaders/MiscDataFuncs.h"

#ifdef WITH_TORCH
    #include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it
#endif


//configuru
// #define CONFIGURU_WITH_EIGEN 1
// #define CONFIGURU_IMPLICIT_CONVERSIONS 1
// #include <configuru.hpp>
// using namespace configuru;

//loguru
#define LOGURU_REPLACE_GLOG 1
#include <loguru.hpp>



//my stuff

// using namespace er::utils;
using namespace radu::utils;
using namespace easy_pbr;


MiscDataFuncs::MiscDataFuncs(){


}



#ifdef WITH_TORCH

    //function to get all the frames into gpu tensors which allows for faster sampling of rays later
    TensorReel MiscDataFuncs::frames2tensors(const std::vector< Frame >& frames){
        CHECK(!frames.empty()) << "Frames is empty";

        //we assume all the images have the same size so that they can be packed into a tensor of size BCHW where B is the nr of images
        int first_w=frames[0].width;
        int first_h=frames[0].height;
        for (size_t i=0; i<frames.size(); i++){
            if(frames[i].width!= first_w || frames[i].height!=first_h){
                LOG(FATAL) << "Frames vector contains images that are not the same size. Found image at idx " << i << " with width and height " << frames[i].width << " " << frames[i].height << " when the first image in the vector has width and height " << first_w << " " << first_h << ". Please use frame.crop to get a frame that has all the same size";
            }
        }
        

        //make tensors for K(as 3x3 matrix) and poses as 4x4 matrices
        torch::Tensor K_reel = torch::empty({ (int)frames.size(), 3,3 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        torch::Tensor tf_cam_world_reel = torch::empty({ (int)frames.size(), 4,4 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        torch::Tensor tf_world_cam_reel = torch::empty({ (int)frames.size(), 4,4 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        //Make a reel to contain all the image batches
        torch::Tensor rgb_reel = torch::empty({ (int)frames.size(), 3, first_h, first_w }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

        bool has_mask=false;
        torch::Tensor mask_reel=torch::empty({ 1, 1, 1, 1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        if (!frames[0].mask.empty()){
            mask_reel = torch::empty({ (int)frames.size(), 1, first_h, first_w }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
            has_mask=true;
        }

        //load all the images and K and poses
        for (size_t i=0; i<frames.size(); i++){ 
            

            //load rgb
            torch::Tensor rgb_32f_tensor=mat2tensor(frames[i].rgb_32f, true).to("cuda");
            rgb_reel.slice(0, i, i+1)=rgb_32f_tensor;

            //load mask
            if (has_mask){
                torch::Tensor mask_tensor=mat2tensor(frames[i].mask, false).to("cuda");
                //get the mask as only 1 channel
                mask_tensor=mask_tensor.slice(1, 0, 1); //dim,start,end
                mask_reel.slice(0, i, i+1)=mask_tensor;
            }


            //load K
            torch::Tensor K_tensor=eigen2tensor(frames[i].K);
            K_reel.slice(0, i, i+1)=K_tensor.unsqueeze(0);

            //load pose
            torch::Tensor tf_cam_world_tensor=eigen2tensor(frames[i].tf_cam_world.matrix());
            tf_cam_world_reel.slice(0, i, i+1)=tf_cam_world_tensor.unsqueeze(0);

            //load pose inverse
            torch::Tensor tf_world_cam_tensor=eigen2tensor(frames[i].tf_cam_world.inverse().matrix());
            tf_world_cam_reel.slice(0, i, i+1)=tf_world_cam_tensor.unsqueeze(0);


                
        }




        TensorReel reel;
        reel.K_reel=K_reel;
        reel.tf_cam_world_reel=tf_cam_world_reel;
        reel.tf_world_cam_reel=tf_world_cam_reel;
        reel.rgb_reel=rgb_reel;
        reel.mask_reel=mask_reel;
        reel.has_mask=has_mask;

        return reel;

    }

#endif



