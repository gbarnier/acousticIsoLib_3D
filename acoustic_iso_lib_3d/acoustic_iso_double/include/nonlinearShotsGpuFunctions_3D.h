#ifndef NONLINEAR_SHOTS_GPU_FUNCTIONS_3D_H
#define NONLINEAR_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/*********************************** Initialization **************************************/
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumber);
void initNonlinearGpu_3D(float dz, float dx, float dy, int nz, int nx, int ny, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateNonlinearGpu_3D(float *vel2Dtw2, int iGpu, int iGpuId);
void deallocateNonlinearGpu_3D(int iGpu, int iGpuId);

/*********************************** Nonlinear FWD **************************************/
void propShotsFwdGpu_3D(float *modelRegDtw, float *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);
void propShotsFwdGpuWavefield_3D(float *modelRegDtw, float *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);

/*********************************** Nonlinear ADJ **************************************/
/* Adjoint propagation -- Data recorded at fine scale */
void propShotsAdjGpu_3D(float *modelRegDtw, float *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);
void propShotsAdjGpuWavefield_3D(float *modelRegDtw, float *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);

#endif
