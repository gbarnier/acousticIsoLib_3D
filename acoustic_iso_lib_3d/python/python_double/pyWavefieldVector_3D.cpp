#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "wavefieldVector_3D.h"

namespace py = pybind11;

// Definition of wavefield vector object
PYBIND11_MODULE(pyWavefieldVector_3D, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<wavefieldVector_3D, std::shared_ptr<wavefieldVector_3D>>(clsGeneric, "wavefieldVector_3D")

      .def(py::init<>(), "Initialize a wavefieldVector_3D")
  ;
}
