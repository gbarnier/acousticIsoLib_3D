#ifndef NONLINEAR_SHOTS_GPU_FUNCTIONS_3D_H
#define NONLINEAR_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/*********************************** Initialization **************************************/
// Gpu info
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumber);

// Initialization
void initNonlinearGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);

// Allocation
void allocateNonlinearGpu_3D(double *vel2Dtw2, int iGpu, int iGpuId);

// Deallocation
void deallocateNonlinearGpu_3D(int iGpu, int iGpuId);

// Initialization - Ginsu
void initNonlinearGinsuGpu_3D(double dz, double dx, double dy, int nts, double dts, int sub, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);

// Allocation - Ginsu
void allocateSetNonlinearGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, double alphaCos, double *vel2Dtw2, int iGpu, int iGpuId);

/*********************************** Nonlinear FWD **************************************/
// Forward
void propShotsFwdGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Forward - Ginsu
void propShotsFwdGinsuGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Damp test
void propShotsFwdGpu_3D_dampTest(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId, double *dampVolume);

// Forward - Free surface
void propShotsFwdFreeSurfaceGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Forward - Free surface - Ginsu
void propShotsFwdFreeSurfaceGinsuGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

/*********************************** Nonlinear ADJ **************************************/
// Adjoint - No Ginsu
void propShotsAdjGpu_3D(double *modelRegDtw, double *dataRegDtw, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Adjoint - Ginsu
void propShotsAdjGinsuGpu_3D(double *modelRegDtw, double *dataRegDtw, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Adjoint - Free surface
void propShotsAdjFreeSurfaceGpu_3D(double *modelRegDtw, double *dataRegDtw, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Adjoint - Free surface - Ginsu
void propShotsAdjFreeSurfaceGinsuGpu_3D(double *modelRegDtw, double *dataRegDtw, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

/****************************** Laplacian **********************************/
// void laplacianFwd_3d(double *model, double *data, int iGpu, int iGpuId);
// void laplacianAdj_3d(double *model, double *data, int iGpu, int iGpuId);
//
// void freeSurfaceDebugFwd(double *model, double *data, int iGpu, int iGpuId);
// void freeSurfaceDebugAdj(double *model, double *data, int iGpu, int iGpuId);

#endif
