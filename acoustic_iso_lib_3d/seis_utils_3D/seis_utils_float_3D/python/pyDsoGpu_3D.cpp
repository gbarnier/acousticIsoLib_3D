#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "pyDsoGpu_3D.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyDsoGpu_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<pyDsoGpu_3D, std::shared_ptr<pyDsoGpu_3D>>(clsGeneric,"pyDsoGpu_3D")

      .def(py::init<int,int,int,int,float>(), "Initialize a DSO operator")

      .def("forward", (void (pyDsoGpu_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &pyDsoGpu_3D::forward, "Forward")

      .def("adjoint", (void (pyDsoGpu_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &pyDsoGpu_3D::adjoint, "Adjoint")

  ;
}
