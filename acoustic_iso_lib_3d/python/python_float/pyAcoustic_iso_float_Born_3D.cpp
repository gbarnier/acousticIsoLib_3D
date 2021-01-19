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
PYBIND11_MODULE(pyAcoustic_iso_float_Born_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<BornShotsGpu_3D, std::shared_ptr<BornShotsGpu_3D>>(clsGeneric,"BornShotsGpu_3D")

      // Normal constructor
      .def(py::init<std::shared_ptr<SEP::float3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<SEP::float2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>>(), "Initialize a BornShotsGpu_3D")

      // Normal constructor
      .def(py::init<std::shared_ptr<SEP::float3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<SEP::float2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<wavefieldVector_3D>>(), "Initialize a BornShotsGpu_3D for FWIME")

      // Constructor for Ginsu
      .def(py::init<std::shared_ptr<SEP::float3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<SEP::float2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::vector<std::shared_ptr<SEP::hypercube>>, std::shared_ptr<SEP::int1DReg>, std::shared_ptr<SEP::int1DReg>, int, int, std::vector<int>, std::vector<int>>(), "Initialize a BornShotsGpu_3D with Ginsu")

      // Constructor for Ginsu
      .def(py::init<std::shared_ptr<SEP::float3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<SEP::float2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::vector<std::shared_ptr<SEP::hypercube>>, std::shared_ptr<SEP::int1DReg>, std::shared_ptr<SEP::int1DReg>, int, int, std::vector<int>, std::vector<int>, std::shared_ptr<wavefieldVector_3D>>(), "Initialize a BornShotsGpu_3D for FWIME with Ginsu")

      .def("forward", (void (BornShotsGpu_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &BornShotsGpu_3D::forward, "Forward")

      .def("adjoint", (void (BornShotsGpu_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &BornShotsGpu_3D::adjoint, "Adjoint")

      .def("setVel_3D",(void (BornShotsGpu_3D::*)(std::shared_ptr<float3DReg>)) &BornShotsGpu_3D::setVel_3D,"Function to set background velocity")

      .def("deallocatePinnedBornGpu_3D",(void (BornShotsGpu_3D::*)()) &BornShotsGpu_3D::deallocatePinnedBornGpu_3D,"Function to deallocate the pinned memory where the source wavefields are stored")

      .def("dotTest",(bool (BornShotsGpu_3D::*)(const bool, const float)) &BornShotsGpu_3D::dotTest,"Dot-Product Test")

;
}
