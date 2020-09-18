#ifndef TOMO_EXT_SHOTS_GPU_FUNCTIONS_3D_H
#define TOMO_EXT_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/* Parameter settings */
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo);

void initTomoExtGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, std::string extension, int nExt1, int nExt2, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc);

void allocateTomoExtShotsGpu_3D(double *vel2Dtw2, double *reflectivityScale, double *extReflectivity, int iGpu, int iGpuId);

void allocatePinnedTomoExtGpu_3D(int nzWavefield, int nxWavefield, int nyWavefield, int ntsWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

void initTomoExtGinsuGpu_3D(double dz, double dx, double dy, int nts, double dts, int sub, int blockSize, double alphaCos, std::string extension, int nExt1, int nExt2, int nGpu, int iGpuId, int iGpuAlloc);

void allocateSetTomoExtGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, double alphaCos, double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId);

void deallocateTomoExtShotsGpu_3D(int iGpu, int iGpuId);

void deallocatePinnedTomoExtShotsGpu_3D(int iGpu, int iGpuId);

/******************************************************************************/
/************************* Tomo extended forward ******************************/
/******************************************************************************/

/******************************* No free surface ******************************/
// Subsurface offset
void tomoHxHyShotsFwdGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId);

// Time-lags
void tomoTauShotsFwdGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId);

/******************************** Free surface ********************************/
// Subsurface offset
void tomoHxHyFsShotsFwdGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId);

// Time-lags
void tomoTauFsShotsFwdGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId);

/******************************************************************************/
/************************* Tomo extended adjoint ******************************/
/******************************************************************************/

/******************************* No free surface ******************************/
// Subsurface offset
void tomoHxHyShotsAdjGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId, double *dev_dataRegDtsQcIn);

// Time-lags
void tomoTauShotsAdjGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId, double *dev_dataRegDtsQcIn);

/******************************* Free surface *********************************/
// Subsurface offset
void tomoHxHyFsShotsAdjGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId, double *dev_dataRegDtsQcIn);

// Time-lags
void tomoTauFsShotsAdjGpu_3D(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *wavefield1, double *wavefield2, int iGpu, int iGpuId, double *dev_dataRegDtsQcIn);

#endif
