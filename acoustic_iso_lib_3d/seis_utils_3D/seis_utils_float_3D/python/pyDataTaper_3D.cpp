#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "dataTaper_3D.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyDataTaper_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<dataTaper_3D, std::shared_ptr<dataTaper_3D>>(clsGeneric,"dataTaper_3D")

      // Constructor for offset + end of trace tapering
      .def(py::init<float,float,float,std::string,float,float,std::shared_ptr<SEP::hypercube>,std::shared_ptr<float2DReg>,std::shared_ptr<float3DReg>>(), "Initialize a dataTaper_3D for offset muting")

      // Constructor for time + end of trace tapering
      .def(py::init<float,float,float,float,std::string,std::string,float,float,std::shared_ptr<SEP::hypercube>,std::shared_ptr<float2DReg>,std::shared_ptr<float3DReg>>(), "Initialize a dataTaper_3D for time muting")

      // Constructor for time + end of trace tapering
      .def(py::init<float,float,float,float,std::string,std::string,float,float,float,std::string,float,float,std::shared_ptr<SEP::hypercube>,std::shared_ptr<float2DReg>,std::shared_ptr<float3DReg>>(), "Initialize a dataTaper_3D for time muting")

      // Constructor for end of trace tapering
      .def(py::init<float,float,std::shared_ptr<SEP::hypercube>>(), "Initialize a dataTaper_3D for end of trace tapering")

      .def("forward", (void (dataTaper_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dataTaper_3D::forward, "Forward")

      .def("adjoint", (void (dataTaper_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dataTaper_3D::adjoint, "Adjoint")

      .def("getTaperMaskTime_3D", (std::shared_ptr<float3DReg> (dataTaper_3D::*)()) &dataTaper_3D::getTaperMaskTime_3D, "getTaperMaskTime_3D")

      .def("getTaperMaskOffset_3D", (std::shared_ptr<float2DReg> (dataTaper_3D::*)()) &dataTaper_3D::getTaperMaskOffset_3D, "getTaperMaskOffset_3D")

      .def("getTaperMaskEndTrace_3D", (std::shared_ptr<float1DReg> (dataTaper_3D::*)()) &dataTaper_3D::getTaperMaskEndTrace_3D, "getTaperMaskEndTrace_3D")

  ;
}
