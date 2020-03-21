#ifndef TOMO_EXT_SHOTS_GPU_FUNCTIONS_3D_H
#define TOMO_EXT_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/* Parameter settings */
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initTomoExtGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt1, int nExt2, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc);
void allocateTomoExtShotsGpu_3D(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId);
void deallocateTomoExtShotsGpu_3D(int iGpu, int iGpuId);

/******************************************************************************/
/************************* Tomo extended forward ******************************/
/******************************************************************************/

/******************************* No free surface ******************************/
void tomoHxShotsFwdGpu_3D(double *model, double *dataRegDtw, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId);

/******************************************************************************/
/************************* Tomo extended adjoint ******************************/
/******************************************************************************/

/******************************* No free surface ******************************/
// void tomoTauShotsAdjGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

#endif
