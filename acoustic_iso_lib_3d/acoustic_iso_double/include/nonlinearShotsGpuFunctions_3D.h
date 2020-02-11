#ifndef NONLINEAR_SHOTS_GPU_FUNCTIONS_3D_H
#define NONLINEAR_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/*********************************** Initialization **************************************/
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumber);
void initNonlinearGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateNonlinearGpu_3D(double *vel2Dtw2, int iGpu, int iGpuId);
void deallocateNonlinearGpu_3D(int iGpu, int iGpuId);

/*********************************** Nonlinear FWD **************************************/
void propShotsFwdGpu_3D(double *modelRegDtw, double *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);

/*********************************** Nonlinear ADJ **************************************/
void propShotsAdjGpu_3D(double *modelRegDtw, double *dataRegDtw, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);

/****************************** Laplacian **********************************/
void laplacianFwd_3d(double *model, double *data, int iGpu, int iGpuId);
void laplacianAdj_3d(double *model, double *data, int iGpu, int iGpuId);

#endif
