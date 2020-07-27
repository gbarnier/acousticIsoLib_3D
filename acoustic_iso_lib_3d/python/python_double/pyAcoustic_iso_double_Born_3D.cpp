#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "BornShotsGpu_3D.h"

namespace py = pybind11;
using namespace SEP;

//Definition of Born operator
PYBIND11_MODULE(pyAcoustic_iso_double_Born_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<BornShotsGpu_3D, std::shared_ptr<BornShotsGpu_3D>>(clsGeneric,"BornShotsGpu_3D")
      .def(py::init<std::shared_ptr<SEP::double3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<SEP::double2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>>(), "Initialize a BornShotsGpu_3D")

      .def(py::init<std::shared_ptr<SEP::double3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<SEP::double2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::vector<std::shared_ptr<SEP::hypercube>>, std::shared_ptr<SEP::int1DReg>, std::shared_ptr<SEP::int1DReg>, int, int, std::vector<int>, std::vector<int>>(), "Initialize a BornShotsGpu_3D with Ginsu")

      .def("forward", (void (BornShotsGpu_3D::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &BornShotsGpu_3D::forward, "Forward")

      .def("adjoint", (void (BornShotsGpu_3D::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &BornShotsGpu_3D::adjoint, "Adjoint")

      .def("setVel_3D",(void (BornShotsGpu_3D::*)(std::shared_ptr<double3DReg>)) &BornShotsGpu_3D::setVel_3D,"Function to set background velocity")

      .def("setVel_3D",(void (BornShotsGpu_3D::*)(std::shared_ptr<double3DReg>)) &BornShotsGpu_3D::setVel_3D,"Function to set background velocity")

      // .def("getSrcWavefield_3D",(std::shared_ptr<double4DReg> (BornShotsGpu_3D::*)(int iWavefield)) &BornShotsGpu_3D::getSrcWavefield_3D,"Function to get src wfld")

      .def("dotTest",(bool (BornShotsGpu_3D::*)(const bool, const double)) &BornShotsGpu_3D::dotTest,"Dot-Product Test")
;
}
