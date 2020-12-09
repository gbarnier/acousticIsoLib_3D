#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "traceNorm_3D.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyTraceNorm_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<traceNorm_3D, std::shared_ptr<traceNorm_3D>>(clsGeneric,"traceNorm_3D")

      .def(py::init<std::shared_ptr<float3DReg>,float>(), "Initialize traceNorm operator")

      .def("forward", (void (traceNorm_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &traceNorm_3D::forward, "Forward")

  ;

	py::class_<traceNormJac_3D, std::shared_ptr<traceNormJac_3D>>(clsGeneric,"traceNormJac_3D")

      .def(py::init<std::shared_ptr<float3DReg>,float>(), "Initialize traceNormJac_3D operator")

      .def("forward", (void (traceNormJac_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &traceNormJac_3D::forward, "Forward")

			.def("adjoint", (void (traceNormJac_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &traceNormJac_3D::adjoint, "Adjoint")

			.def("setData", (void (traceNormJac_3D::*)(std::shared_ptr<float3DReg>)) &traceNormJac_3D::setData, "setData")

  ;
}
