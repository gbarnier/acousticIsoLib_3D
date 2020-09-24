#ifndef TOMO_EXT_SHOTS_GPU_FUNCTIONS_3D_H
#define TOMO_EXT_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/* Parameter settings */
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo);

void initTomoExtGpu_3D(float dz, float dx, float dy, int nz, int nx, int ny, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, std::string extension, int nExt1, int nExt2, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc);

void allocateTomoExtShotsGpu_3D(float *vel2Dtw2, float *reflectivityScale, float *extReflectivity, int iGpu, int iGpuId);

void allocatePinnedTomoExtGpu_3D(int nzWavefield, int nxWavefield, int nyWavefield, int ntsWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

void initTomoExtGinsuGpu_3D(float dz, float dx, float dy, int nts, float dts, int sub, int blockSize, float alphaCos, std::string extension, int nExt1, int nExt2, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc);

void allocateSetTomoExtGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, float alphaCos, float *vel2Dtw2, float *reflectivityScale, float *extReflectivity, int iGpu, int iGpuId);

void deallocateTomoExtShotsGpu_3D(int iGpu, int iGpuId);

void deallocatePinnedTomoExtShotsGpu_3D(int iGpu, int iGpuId);

/******************************************************************************/
/************************* Tomo extended forward ******************************/
/******************************************************************************/

/******************************* No free surface ******************************/
// Time-lags
void tomoTauShotsFwdGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Time-lags + Ginsu
void tomoTauShotsFwdGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Subsurface offsets
void tomoHxHyShotsFwdGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Subsurface offsets + Ginsu
void tomoHxHyShotsFwdGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

/****************************** Free surface **********************************/
// Time-lags
void tomoTauShotsFwdFsGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Time-lags
void tomoTauShotsFwdFsGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Subsurface offsets
void tomoHxHyShotsFwdFsGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Subsurface offsets + Ginsu
void tomoHxHyShotsFwdFsGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

/******************************************************************************/
/************************* Tomo extended adjoint ******************************/
/******************************************************************************/

/******************************* No free surface ******************************/
// Time-lags
void tomoTauShotsAdjGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Time-lags + Ginsu
void tomoTauShotsAdjGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Subsurface offsets
void tomoHxHyShotsAdjGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Subsurface offsets + Ginsu
void tomoHxHyShotsAdjGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

/****************************** Free surface **********************************/
// Time-lags
void tomoTauShotsAdjFsGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Time-lags + Ginsu
void tomoTauShotsAdjFsGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Subsurface offsets
void tomoHxHyShotsAdjFsGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Subsurface offsets + Ginsu
void tomoHxHyShotsAdjFsGinsuGpu_3D(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

#endif
