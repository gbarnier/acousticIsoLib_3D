cmake_minimum_required(VERSION 2.8)
project(pyAcoustic_iso_double_3D)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# include folder for ACOUSTIC_ISO_DOUBLE library
include_directories(../../acoustic_iso_double/include)


# This is a CMake example for Python
find_package(pybind11)
if (pybind11_FOUND)
  #PYBIND11 library headers
  include_directories(${PYTHON_INCLUDE_DIR})
  include_directories(${pybind11_INCLUDE_DIR})
  include_directories(${PYTHON_INCLUDE_PATH})
  #library headers
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../acoustic_isotropic_operators")

  find_package(PythonLibs)
  include_directories(${PYTHON_INCLUDE_PATH})

  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  include_directories(${GENERIC_EXPORT_INC_DIRS})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  #SEP LIBRARY DIRECTORIES
  link_directories(${Generic-cpp_BINARY_DIR}
				 ${hypercube_BINARY_DIR}
				 ${buffers-lib_BINARY_DIR}
				 ${json_BINARY_DIR}
				 ${sepVector-lib_BINARY_DIR}
				 ${SEPlib_LIBDIR}
  )

  set(LIBS genericCpp buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

  #Compiling and adding module to library
  # pybind11_add_module(pyAcoustic_iso_double_nl MODULE pyAcoustic_iso_double_nl.cpp)
  # target_link_libraries(pyAcoustic_iso_double_nl PRIVATE ACOUSTIC_ISO_DOUBLE_NL ${LIBS})

	#Installing library
	# install( TARGETS pyAcoustic_iso_double_nl DESTINATION lib/python) #Module for non-linear operator

	#Installing Python executables
  # install( FILES nonlinearPythonDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)


else(pybind11_FOUND)
  MESSAGE("WARNING!Did not find pybid11. Not compiling ${PROJECT-NAME}")
endif(pybind11_FOUND)
