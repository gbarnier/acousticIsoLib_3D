#ifndef NONLINEAR_SHOTS_GPU_FUNCTIONS_3D_H
#define NONLINEAR_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/*********************************** Initialization **************************************/
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumber);
void initNonlinearGpu_3D(float dz, float dx, float dy, int nz, int nx, int ny, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateNonlinearGpu_3D(float *vel2Dtw2, int iGpu, int iGpuId);
void deallocateNonlinearGpu_3D(int iGpu, int iGpuId);

/*********************************** Nonlinear FWD **************************************/
void propShotsFwdGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

void propShotsFwdGpu_3D_dampTest(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId, float *dampVolume);

void propShotsFwdFreeSurfaceGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

/*********************************** Nonlinear ADJ **************************************/
void propShotsAdjGpu_3D(float *modelRegDtw, float *dataRegDtw, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

void propShotsAdjFreeSurfaceGpu_3D(float *modelRegDtw, float *dataRegDtw, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

/****************************** Laplacian **********************************/
// void laplacianFwd_3d(float *model, float *data, int iGpu, int iGpuId);
// void laplacianAdj_3d(float *model, float *data, int iGpu, int iGpuId);
//
// void freeSurfaceDebugFwd(float *model, float *data, int iGpu, int iGpuId);
// void freeSurfaceDebugAdj(float *model, float *data, int iGpu, int iGpuId);

#endif
