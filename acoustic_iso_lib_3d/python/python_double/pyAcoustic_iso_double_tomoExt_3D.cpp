#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "tomoExtShotsGpu_3D.h"

namespace py = pybind11;
using namespace SEP;

//Definition of Born operator
PYBIND11_MODULE(pyAcoustic_iso_double_tomoExt_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<tomoExtShotsGpu_3D, std::shared_ptr<tomoExtShotsGpu_3D>>(clsGeneric,"tomoExtShotsGpu_3D")

      .def(py::init<std::shared_ptr<SEP::double3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<double2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<double5DReg>>(), "Initialize a tomoExtShotsGpu_3D")

      .def("forward", (void (tomoExtShotsGpu_3D::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &tomoExtShotsGpu_3D::forward, "Forward")

      .def("adjoint", (void (tomoExtShotsGpu_3D::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &tomoExtShotsGpu_3D::adjoint, "Adjoint")

      .def("setVel_3D",(void (tomoExtShotsGpu_3D::*)(std::shared_ptr<double3DReg>)) &tomoExtShotsGpu_3D::setVel_3D,"Function to set background velocity")

      .def("getWavefield1_3D",(std::shared_ptr<double4DReg> (tomoExtShotsGpu_3D::*)(int iWavefield)) &tomoExtShotsGpu_3D::getWavefield1_3D,"Function to get wfld #1")

      .def("getWavefield2_3D",(std::shared_ptr<double4DReg> (tomoExtShotsGpu_3D::*)(int iWavefield)) &tomoExtShotsGpu_3D::getWavefield2_3D,"Function to get wfld #2")

      .def("dotTest",(bool (tomoExtShotsGpu_3D::*)(const bool, const double)) &tomoExtShotsGpu_3D::dotTest,"Dot-Product Test")
;
}
