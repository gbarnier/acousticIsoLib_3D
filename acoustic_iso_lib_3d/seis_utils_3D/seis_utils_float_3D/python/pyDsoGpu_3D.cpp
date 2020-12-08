#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "dsoGpu_3D.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyDsoGpu_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<dsoGpu_3D, std::shared_ptr<dsoGpu_3D>>(clsGeneric,"dsoGpu_3D")

      .def(py::init<int,int,int,int,int,int,float>(), "Initialize a DSO operator")

      .def("forward", (void (dsoGpu_3D::*)(const bool, const std::shared_ptr<float5DReg>, std::shared_ptr<float5DReg>)) &dsoGpu_3D::forward, "Forward")

      .def("adjoint", (void (dsoGpu_3D::*)(const bool, const std::shared_ptr<float5DReg>, std::shared_ptr<float5DReg>)) &dsoGpu_3D::adjoint, "Adjoint")

  ;
}
