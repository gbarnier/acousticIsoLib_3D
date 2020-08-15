#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "interpBSpline_3D.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyInterpBSpline_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<interpBSpline_3D, std::shared_ptr<interpBSpline_3D>>(clsGeneric,"interpBSpline_3D")
      .def(py::init<int,int,int,std::shared_ptr<float1DReg>,std::shared_ptr<float1DReg>,std::shared_ptr<float1DReg>,axis,axis,axis,int,int,int,int,float,float,float,int,int,int>(), "Initialize a interpBSpline_3D")

      .def("forward", (void (interpBSpline_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &interpBSpline_3D::forward, "Forward")

      .def("adjoint", (void (interpBSpline_3D::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &interpBSpline_3D::adjoint, "Adjoint")

      .def("getZMeshModel", (std::shared_ptr<float1DReg> (interpBSpline_3D::*)()) &interpBSpline_3D::getZMeshModel, "getZMeshModel")

      .def("getXMeshModel", (std::shared_ptr<float1DReg> (interpBSpline_3D::*)()) &interpBSpline_3D::getXMeshModel, "getXMeshModel")

      .def("getYMeshModel", (std::shared_ptr<float1DReg> (interpBSpline_3D::*)()) &interpBSpline_3D::getYMeshModel, "getYMeshModel")

      .def("getZMeshData", (std::shared_ptr<float1DReg> (interpBSpline_3D::*)()) &interpBSpline_3D::getZMeshData, "getZMeshData")

      .def("getXMeshData", (std::shared_ptr<float1DReg> (interpBSpline_3D::*)()) &interpBSpline_3D::getXMeshData, "getXMeshData")

      .def("getYMeshData", (std::shared_ptr<float1DReg> (interpBSpline_3D::*)()) &interpBSpline_3D::getYMeshData, "getYMeshData")

      .def("getZControlPoints", (std::shared_ptr<float1DReg> (interpBSpline_3D::*)()) &interpBSpline_3D::getZControlPoints, "getZControlPoints")

      .def("getXControlPoints", (std::shared_ptr<float1DReg> (interpBSpline_3D::*)()) &interpBSpline_3D::getXControlPoints, "getXControlPoints")

      .def("getYControlPoints", (std::shared_ptr<float1DReg> (interpBSpline_3D::*)()) &interpBSpline_3D::getYControlPoints, "getYControlPoints")

  ;
}
