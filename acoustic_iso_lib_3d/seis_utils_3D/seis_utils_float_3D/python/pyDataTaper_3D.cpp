#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "dataTaper_3D.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyDataTaper, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<dataTaper_3D, std::shared_ptr<dataTaper_3D>>(clsGeneric,"dataTaper_3D")

      .def(py::init<float,float,float,float,std::string,int,float,float,float,int,std::shared_ptr<SEP::hypercube>,float,int>(), "Initialize a dataTaper_3D for time and offset muting")

      .def(py::init<float,float,float,float,std::shared_ptr<SEP::hypercube>,std::string,int,float>(), "Initialize a dataTaper_3D for time muting")

      .def(py::init<float,float,float,std::shared_ptr<SEP::hypercube>,int,float,int>(), "Initialize a dataTaper_3D for offset muting")

      .def(py::init<std::shared_ptr<SEP::hypercube>,float>(), "Initialize a dataTaper_3D for only end of trace muting")

      .def("forward", (void (dataTaper_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dataTaper_3D::forward, "Forward")

      .def("adjoint", (void (dataTaper_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dataTaper_3D::adjoint, "Adjoint")

      .def("getTaperMask_3D", (std::shared_ptr<float3DReg> (dataTaper_3D::*)()) &dataTaper_3D::getTaperMask_3D, "getTaperMask")

      .def("getTaperMaskTime_3D", (std::shared_ptr<float3DReg> (dataTaper_3D::*)()) &dataTaper_3D::getTaperMaskTime_3D, "getTaperMaskTime")

      .def("getTaperMaskOffset_3D", (std::shared_ptr<float3DReg> (dataTaper::*)()) &dataTaper_3D::getTaperMaskOffset_3D, "getTaperMaskOffset")

  ;
}
