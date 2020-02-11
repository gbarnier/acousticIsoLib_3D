#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "nonlinearPropShotsGpu_3D.h"

namespace py = pybind11;
using namespace SEP;

// Definition of Device object and non-linear propagator
PYBIND11_MODULE(pyAcoustic_iso_double_nl_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<deviceGpu_3D, std::shared_ptr<deviceGpu_3D>>(clsGeneric, "deviceGpu_3D")

      .def(py::init<const std::shared_ptr<SEP::double1DReg>, const std::shared_ptr<SEP::double1DReg>, const std::shared_ptr<SEP::double1DReg>, const std::shared_ptr<double3DReg>, int &, int, double, double, double, std::string, int>(), "Initialize a deviceGPU_3D object using location, velocity, and nt")

      .def(py::init<const int &, const int &, const int &, const int &, const int &, const int &, const int &, const int &, const int &, const std::shared_ptr<double3DReg>, int &, int, int, int, int, std::string, int>(), "Initlialize a deviceGPU_3D object using sampling in z, x and y axes, velocity, and nt")

  ;

  py::class_<nonlinearPropShotsGpu_3D, std::shared_ptr<nonlinearPropShotsGpu_3D>>(clsGeneric,"nonlinearPropShotsGpu_3D")

      .def(py::init<std::shared_ptr<SEP::double3DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu_3D>>, std::vector<std::shared_ptr<deviceGpu_3D>>>(), "Initialize a nonlinearPropShotsGpu_3D")

      .def("forward", (void (nonlinearPropShotsGpu_3D::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu_3D::forward, "Forward")

      .def("adjoint", (void (nonlinearPropShotsGpu_3D::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu_3D::adjoint, "Adjoint")

      .def("setVel_3D",(void (nonlinearPropShotsGpu_3D::*)(std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu_3D::setVel_3D,"Function to set background velocity")

      .def("dotTest",(bool (nonlinearPropShotsGpu_3D::*)(const bool, const double)) &nonlinearPropShotsGpu_3D::dotTest,"Dot-Product Test")

;

}
