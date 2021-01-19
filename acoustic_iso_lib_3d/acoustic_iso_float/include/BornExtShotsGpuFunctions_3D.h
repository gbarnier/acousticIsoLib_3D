#ifndef BORN_EXT_SHOTS_GPU_FUNCTIONS_3D_H
#define BORN_EXT_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/* Parameter settings */
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo);

void initBornExtGpu_3D(float dz, float dx, float dy, int nz, int nx, int ny, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, std::string extension, int nExt1, int nExt2, int nGpu, int iGpuId, int iGpuAlloc);

void allocateBornExtShotsGpu_3D(float *vel2Dtw2, float *reflectivityScale, int iGpu, int iGpuId);

void allocatePinnedBornExtGpu_3D(int nzWavefield, int nxWavefield, int nyWavefield, int ntsWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

// Allocate pinned memory for Fwime
void setPinnedBornExtGpuFwime_3D(float *wavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

void initBornExtGinsuGpu_3D(float dz, float dx, float dy, int nts, float dts, int sub, int blockSize, float alphaCos, std::string extension, int nExt1, int nExt2, int nGpu, int iGpuId, int iGpuAlloc);

void allocateSetBornExtGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, float alphaCos, float *vel2Dtw2, float *reflectivityScale, int iGpu, int iGpuId);

void deallocateBornExtShotsGpu_3D(int iGpu, int iGpuId);

void deallocatePinnedBornExtShotsGpu_3D(int iGpu, int iGpuId);

/******************************************************************************/
/************************* Born extended forward ******************************/
/******************************************************************************/

/******************************* No free surface ******************************/
// Time
void BornTauShotsFwdGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

// Time Ginsu
void BornTauShotsFwdGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

// Offset
void BornHxHyShotsFwdGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

// Offset Ginsu
void BornHxHyShotsFwdGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

/******************************* Free surface *********************************/
// Time
void BornTauFreeSurfaceShotsFwdGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

// Time Ginsu
void BornTauFreeSurfaceShotsFwdGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

// Offset
void BornHxHyFreeSurfaceShotsFwdGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

// Offset Ginsu
void BornHxHyFreeSurfaceShotsFwdGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

/******************************************************************************/
/************************* Born extended adjoint ******************************/
/******************************************************************************/

/******************************* No free surface ******************************/
// Time
void BornTauShotsAdjGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

// Time Ginsu
void BornTauShotsAdjGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

// Offset
void BornHxHyShotsAdjGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

// Offset Ginsu
void BornHxHyShotsAdjGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int slowSquare, int iGpu, int iGpuId);

/******************************* Free surface *********************************/
// Time
void BornTauFreeSurfaceShotsAdjGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int slowSquare, int nReceiversReg, int iGpu, int iGpuId);

// Time Ginsu
void BornTauFreeSurfaceShotsAdjGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int slowSquare, int nReceiversReg, int iGpu, int iGpuId);

// Offset
void BornHxHyFreeSurfaceShotsAdjGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int slowSquare, int nReceiversReg, int iGpu, int iGpuId);

// Offset Ginsu
void BornHxHyFreeSurfaceShotsAdjGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int slowSquare, int nReceiversReg, int iGpu, int iGpuId);

#endif
