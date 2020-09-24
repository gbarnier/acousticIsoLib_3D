cmake_minimum_required(VERSION 2.8)
project(ACOUSTIC_ISO_FLOAT_3D-lib)


#Adding include files
include_directories(include)

#Header files to include during installation
set(INC_DIR include)
set(HEADERS 			${INC_DIR}/deviceGpu_3D.h
						${INC_DIR}/fdParam_3D.h
						${INC_DIR}/nonlinearPropGpu_3D.h
						${INC_DIR}/nonlinearPropShotsGpu_3D.h
						${INC_DIR}/nonlinearShotsGpuFunctions_3D.h
						# ${INC_DIR}/BornGpu_3D.h
						# ${INC_DIR}/BornShotsGpu_3D.h
						# ${INC_DIR}/BornShotsGpuFunctions_3D.h
						${INC_DIR}/seismicOperator_3D.cpp
						${INC_DIR}/seismicOperator_3D.h
						${INC_DIR}/varDeclare_3D.h
)

#Files to be compiled
set(SRC_DIR src)

# Files for nonlinear operator library
set(SOURCE_NL 		${SRC_DIR}/nonlinearPropShotsGpu_3D.cpp
		   	  		${SRC_DIR}/nonlinearPropGpu_3D.cpp
					${SRC_DIR}/fdParam_3D.cpp
					${SRC_DIR}/deviceGpu_3D.cpp
					${SRC_DIR}/nonlinearShotsGpuFunctions_3D.cu
					${SRC_DIR}/interpTimeLinTbb_3D.cpp
)

# Files for Born operator library
set(SOURCE_BORN 	${SRC_DIR}/BornShotsGpu_3D.cpp
		   	  		${SRC_DIR}/BornGpu_3D.cpp
					${SRC_DIR}/fdParam_3D.cpp
					${SRC_DIR}/deviceGpu_3D.cpp
					${SRC_DIR}/BornShotsGpuFunctions_3D.cu
					${SRC_DIR}/interpTimeLinTbb_3D.cpp
					${SRC_DIR}/secondTimeDerivative_3D.cpp
)

# Files for Born extended operator library
set(SOURCE_BORN_EXT	${SRC_DIR}/BornExtShotsGpu_3D.cpp
		   	  		${SRC_DIR}/BornExtGpu_3D.cpp
					${SRC_DIR}/fdParam_3D.cpp
					${SRC_DIR}/deviceGpu_3D.cpp
					${SRC_DIR}/BornExtShotsGpuFunctions_3D.cu
					${SRC_DIR}/interpTimeLinTbb_3D.cpp
					${SRC_DIR}/secondTimeDerivative_3D.cpp
)

# Files for tomo extended operator library
set(SOURCE_TOMO_EXT	${SRC_DIR}/tomoExtShotsGpu_3D.cpp
		   	  		${SRC_DIR}/tomoExtGpu_3D.cpp
					${SRC_DIR}/fdParam_3D.cpp
					${SRC_DIR}/deviceGpu_3D.cpp
					${SRC_DIR}/tomoExtShotsGpuFunctions_3D.cu
					${SRC_DIR}/interpTimeLinTbb_3D.cpp
					${SRC_DIR}/secondTimeDerivative_3D.cpp
)

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
								 ${hypercube_BINARY_DIR}
								 ${buffers-lib_BINARY_DIR}
								 ${json_BINARY_DIR}
								 ${sepVector-lib_BINARY_DIR}
								 ${SEPlib_LIBDIR}
)

#Setting io libraries
set(LIBS genericCpp sepVector sep3d sep)

#Compiling the project library
add_library(ACOUSTIC_ISO_FLOAT_NL_3D ${SOURCE_NL})
target_link_libraries(ACOUSTIC_ISO_FLOAT_NL_3D ${LIBS} OpenMP::OpenMP_CXX)

add_library(ACOUSTIC_ISO_FLOAT_BORN_3D ${SOURCE_BORN})
target_link_libraries(ACOUSTIC_ISO_FLOAT_BORN_3D ${LIBS} OpenMP::OpenMP_CXX)

add_library(ACOUSTIC_ISO_FLOAT_BORN_EXT_3D ${SOURCE_BORN_EXT})
target_link_libraries(ACOUSTIC_ISO_FLOAT_BORN_EXT_3D ${LIBS} OpenMP::OpenMP_CXX)

add_library(ACOUSTIC_ISO_FLOAT_TOMO_EXT_3D ${SOURCE_TOMO_EXT})
target_link_libraries(ACOUSTIC_ISO_FLOAT_TOMO_EXT_3D ${LIBS} OpenMP::OpenMP_CXX)

# add_executable(padFileGpuMain_3D ${SRC_DIR}/padFileGpuMain_3D.cpp)
# target_link_libraries(padFileGpuMain_3D ${LIBS} OpenMP::OpenMP_CXX)

# Adding libray and header files to install destination
install(TARGETS ACOUSTIC_ISO_FLOAT_NL_3D DESTINATION lib)
install(TARGETS ACOUSTIC_ISO_FLOAT_BORN_3D DESTINATION lib)
install(TARGETS ACOUSTIC_ISO_FLOAT_BORN_EXT_3D DESTINATION lib)
install(TARGETS ACOUSTIC_ISO_FLOAT_TOMO_EXT_3D DESTINATION lib)

# install(TARGETS padFileGpuMain_3D DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)
