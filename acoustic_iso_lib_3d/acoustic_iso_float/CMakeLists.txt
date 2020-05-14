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
						${INC_DIR}/seismicOperator_3D.cpp
						${INC_DIR}/seismicOperator_3D.h
						${INC_DIR}/varDeclare_3D.h
)

#Files to be compiled
set(SRC_DIR src)
#Files for nonlinear operator library
set(SOURCE_NL 		${SRC_DIR}/nonlinearPropShotsGpu_3D.cpp
		   	  		${SRC_DIR}/nonlinearPropGpu_3D.cpp
					${SRC_DIR}/fdParam_3D.cpp
					${SRC_DIR}/deviceGpu_3D.cpp
					${SRC_DIR}/nonlinearShotsGpuFunctions_3D.cu
					${SRC_DIR}/interpTimeLinTbb_3D.cpp
)

#Setting io libraries
set(LIBS genericCpp sepVector sep3d sep)

#Compiling the project library
add_library(ACOUSTIC_ISO_FLOAT_NL_3D ${SOURCE_NL})
target_link_libraries(ACOUSTIC_ISO_FLOAT_NL_3D ${LIBS} OpenMP::OpenMP_CXX)

# Compiling the project executables
# Model padding program for GPU
add_executable(padFileGpuMain_3D ${SRC_DIR}/padFileGpuMain_3D.cpp)
target_link_libraries(padFileGpuMain_3D ${LIBS} OpenMP::OpenMP_CXX)

#Adding libray and header files to install destination
install(TARGETS ACOUSTIC_ISO_FLOAT_NL_3D DESTINATION lib)
install(TARGETS padFileGpuMain_3D DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)
