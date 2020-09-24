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
PYBIND11_MODULE(pyAcoustic_iso_float_tomoExt_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<tomoExtShotsGpu_3D, std::shared_ptr<tomoExtShotsGpu_3D>>(clsGeneric,"tomoExtShotsGpu_3D")

      // Normal constructor
      .def(py::init<std::shared_ptr<SEP::float3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<float2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<float5DReg>>(), "Initialize a tomoExtShotsGpu_3D")

      // Ginsu constructor
      .def(py::init<std::shared_ptr<SEP::float3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<float2DReg>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::shared_ptr<float5DReg>, std::vector<std::shared_ptr<SEP::hypercube>>, std::shared_ptr<SEP::int1DReg>, std::shared_ptr<SEP::int1DReg>, int, int, std::vector<int>, std::vector<int>>(), "Initialize a tomoExtShotsGpu_3D")

      .def("forward", (void (tomoExtShotsGpu_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &tomoExtShotsGpu_3D::forward, "Forward")

      .def("adjoint", (void (tomoExtShotsGpu_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &tomoExtShotsGpu_3D::adjoint, "Adjoint")

      .def("setVel_3D",(void (tomoExtShotsGpu_3D::*)(std::shared_ptr<float3DReg>)) &tomoExtShotsGpu_3D::setVel_3D,"Function to set background velocity")

      .def("setExtReflectivity_3D",(void (tomoExtShotsGpu_3D::*)(std::shared_ptr<float3DReg>)) &tomoExtShotsGpu_3D::setExtReflectivity_3D,"Function to set background velocity")

      .def("deallocatePinnedTomoExtGpu_3D",(void (tomoExtShotsGpu_3D::*)()) &tomoExtShotsGpu_3D::deallocatePinnedTomoExtGpu_3D,"Function to deallocate the pinned memory where the source wavefields are stored")

      .def("dotTest",(bool (tomoExtShotsGpu_3D::*)(const bool, const float)) &tomoExtShotsGpu_3D::dotTest,"Dot-Product Test")
;
}
