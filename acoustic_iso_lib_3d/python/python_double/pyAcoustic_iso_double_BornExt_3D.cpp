#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "BornExtShotsGpu_3D.h"

namespace py = pybind11;
using namespace SEP;

//Definition of Born operator
PYBIND11_MODULE(pyAcoustic_iso_double_BornExt_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<BornExtShotsGpu_3D, std::shared_ptr<BornExtShotsGpu_3D>>(clsGeneric,"BornExtShotsGpu_3D")

      .def(py::init<std::shared_ptr<SEP::double3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<SEP::double2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>>(), "Initialize a BornExtShotsGpu_3D")

      .def(py::init<std::shared_ptr<SEP::double3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<SEP::double2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::vector<std::shared_ptr<SEP::hypercube>>, std::shared_ptr<SEP::int1DReg>, std::shared_ptr<SEP::int1DReg>, int, int, std::vector<int>, std::vector<int>>(), "Initialize a BornExtShotsGpu_3D with Ginsu")

      .def("forward", (void (BornExtShotsGpu_3D::*)(const bool, const std::shared_ptr<double5DReg>, std::shared_ptr<double3DReg>)) &BornExtShotsGpu_3D::forward, "Forward")

      .def("adjoint", (void (BornExtShotsGpu_3D::*)(const bool, const std::shared_ptr<double5DReg>, std::shared_ptr<double3DReg>)) &BornExtShotsGpu_3D::adjoint, "Adjoint")

      .def("setVel_3D",(void (BornExtShotsGpu_3D::*)(std::shared_ptr<double3DReg>)) &BornExtShotsGpu_3D::setVel_3D,"Function to set background velocity")

      .def("deallocatePinnedBornExtGpu_3D",(void (BornExtShotsGpu_3D::*)()) &BornExtShotsGpu_3D::deallocatePinnedBornExtGpu_3D,"Function to deallocate the pinned memory where the source wavefields are stored")

      .def("dotTest",(bool (BornExtShotsGpu_3D::*)(const bool, const double)) &BornExtShotsGpu_3D::dotTest,"Dot-Product Test")
;
}
