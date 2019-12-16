#include "mbzirc_challenge_1/PyBridge.h"

// #include <torch/extension.h>
// #include "torch/torch.h"
// #include "torch/csrc/utils/pybind.h"

//my stuff 
#include "mbzirc_challenge_1/SyntheticGenerator.h"
#include "easy_pbr/Viewer.h"


// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// PYBIND11_MAKE_OPAQUE(std::vector<int>); //to be able to pass vectors by reference to functions and have things like push back actually work 
// PYBIND11_MAKE_OPAQUE(std::vector<float>, std::allocator<float> >);

namespace py = pybind11;




PYBIND11_MODULE(mbzirc1, m) {
 
    //Viewer
    py::class_<SyntheticGenerator, std::shared_ptr<SyntheticGenerator>> (m, "SyntheticGenerator")
    // .def(py::init<const std::shared_ptr<Viewer>>())
    .def_static("create",  &SyntheticGenerator::create<const std::string&, const std::shared_ptr<Viewer>& > ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def_static("create",  &SyntheticGenerator::create<const std::string > ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def("detect_balloon", &SyntheticGenerator::detect_balloon )
    // .def("detect_copter", &SyntheticGenerator::detect_copter )
    ;


}