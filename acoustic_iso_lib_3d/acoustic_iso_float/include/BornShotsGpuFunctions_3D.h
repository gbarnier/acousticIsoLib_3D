#ifndef BORN_SHOTS_GPU_FUNCTIONS_3D_H
#define BORN_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/* Parameter settings */
// GPU info
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo);

// Init normal
void initBornGpu_3D(float dz, float dx, float dy, int nz, int nx, int ny, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc);

// Allocate normal
void allocateBornShotsGpu_3D(float *vel2Dtw2, float *reflectivityScale, int iGpu, int iGpuId);

// Allocate normal pinned memory
void allocatePinnedBornGpu_3D(int nzWavefield, int nxWavefield, int nyWavefield, int ntsWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

// Allocate pinned memory for FWIME
void setPinnedBornGpuFwime_3D(float *wavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

// Init Ginsu
void initBornGinsuGpu_3D(float dz, float dx, float dy, int nts, float dts, int sub, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc);

// Allocate Ginsu
void allocateSetBornGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, float alphaCos, float *vel2Dtw2, float *reflectivityScale, int iGpu, int iGpuId);

// Deallocate
void deallocateBornShotsGpu_3D(int iGpu, int iGpuId);

// Deallocate pinned memory
void deallocatePinnedBornShotsGpu_3D(int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Born forward **********************************/
/******************************************************************************/
// Normal
void BornShotsFwdGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Ginsu
void BornShotsFwdGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Free surface
void BornShotsFwdFreeSurfaceGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Free surface + Ginsu
void BornShotsFwdFreeSurfaceGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Born adjoint **********************************/
/******************************************************************************/
void BornShotsAdjGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

void BornShotsAdjGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

void BornShotsAdjFreeSurfaceGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

void BornShotsAdjFreeSurfaceGinsuGpu_3D(float *model, float *dataRegDts, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// void BornShotsAdjNoStreamGpu_3D(float *model, float *dataRegDtw, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, float *srcWavefield, int iGpu, int iGpuId);

// /*********************************** Debug ************************************/
// // void imagingFwd_zLoop(float *model, float *data, int iGpu, int iGpuId);
// // void imagingFwd_yLoop(float *model, float *data, int iGpu, int iGpuId);
//
// void BornShotsFwdGpu_3D_Threads(float *model, float *dataRegDtw, float *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, float *srcWavefield, int iGpu, int iGpuId);


#endif
