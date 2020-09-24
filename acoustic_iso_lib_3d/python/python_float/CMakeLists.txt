cmake_minimum_required(VERSION 2.8)
project(pyAcoustic_iso_float_3D)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# include folder for ACOUSTIC_ISO_DOUBLE library
include_directories(../../acoustic_iso_float/include)

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

# Compiling and adding module to library
pybind11_add_module(pyAcoustic_iso_float_nl_3D MODULE pyAcoustic_iso_float_nl_3D.cpp)
target_link_libraries(pyAcoustic_iso_float_nl_3D PRIVATE ACOUSTIC_ISO_FLOAT_NL_3D ${LIBS})
pybind11_add_module(pyAcoustic_iso_float_Born_3D MODULE pyAcoustic_iso_float_Born_3D.cpp)
target_link_libraries(pyAcoustic_iso_float_Born_3D PRIVATE ACOUSTIC_ISO_FLOAT_BORN_3D ${LIBS})
pybind11_add_module(pyAcoustic_iso_float_BornExt_3D MODULE pyAcoustic_iso_float_BornExt_3D.cpp)
target_link_libraries(pyAcoustic_iso_float_BornExt_3D PRIVATE ACOUSTIC_ISO_FLOAT_BORN_EXT_3D ${LIBS})
pybind11_add_module(pyAcoustic_iso_float_tomoExt_3D MODULE pyAcoustic_iso_float_tomoExt_3D.cpp)
target_link_libraries(pyAcoustic_iso_float_tomoExt_3D PRIVATE ACOUSTIC_ISO_FLOAT_TOMO_EXT_3D ${LIBS})

# Installing library
install( TARGETS pyAcoustic_iso_float_nl_3D DESTINATION lib/python) # Module for nonlinear operator
install( TARGETS pyAcoustic_iso_float_Born_3D DESTINATION lib/python) # Module for nonlinear operator
install( TARGETS pyAcoustic_iso_float_BornExt_3D DESTINATION lib/python) # Module for nonlinear operator
install( TARGETS pyAcoustic_iso_float_tomoExt_3D DESTINATION lib/python) # Module for nonlinear operator
install( FILES Acoustic_iso_float_3D.py DESTINATION lib/python)
install( FILES inversionUtilsFloat_3D.py DESTINATION lib/python)

# Installing Python executables
install( FILES nonlinearPythonFloatMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES BornPythonFloatMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES BornExtPythonFloatMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES tomoExtPythonFloatMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES makeIrregGeometryFloat_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES lsrtmFloatMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES fwiFloatMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
