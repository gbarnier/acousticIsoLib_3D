cmake_minimum_required(VERSION 2.8)
project(seis_utils_float_3D)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(PythonMains
    waveletFloatMain_3D.py
)

install(FILES ${PythonModules} DESTINATION ${PYTHON_DIR})
install(FILES ${PythonMains} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

# PYBIND11 library headers
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})
# Adding include files
include_directories(../include)

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

set(LIBS genericCpp sepVector buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

# Compiling and adding module to library
pybind11_add_module(pyInterpBSpline_3D MODULE pyInterpBSpline_3D.cpp)
target_link_libraries(pyInterpBSpline_3D PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})

# Installing library
set(PYTHON_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
install( TARGETS pyInterpBSpline_3D DESTINATION ${PYTHON_DIR})
install( FILES interpBSplineModule_3D.py DESTINATION ${PYTHON_DIR})
install( FILES maskGradientModule_3D.py DESTINATION ${PYTHON_DIR})
install( FILES dataTaperModule_3D.py DESTINATION ${PYTHON_DIR})

# Installing Python executables
install( FILES interpBSplineIterMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES interpBSplineInvMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES maskGradientMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES dataTaperMain_3D.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)