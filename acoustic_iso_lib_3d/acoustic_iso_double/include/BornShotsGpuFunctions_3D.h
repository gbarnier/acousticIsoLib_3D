#ifndef BORN_SHOTS_GPU_FUNCTIONS_3D_H
#define BORN_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/* Parameter settings */
// GPU info
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo);

// Init normal
void initBornGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);

// Allocate normal
void allocateBornShotsGpu_3D(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId);

// Allocate normal pinned memory
void allocatePinnedBornGpu_3D(int nzWavefield, int nxWavefield, int nyWavefield, int ntsWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

// Allocate pinned memory for FWIME
void setPinnedBornGpuFwime_3D(double *wavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

// Init Ginsu
void initBornGinsuGpu_3D(double dz, double dx, double dy, int nts, double dts, int sub, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);

// Allocate Ginsu
void allocateSetBornGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, double alphaCos, double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId);

// Deallocate
void deallocateBornShotsGpu_3D(int iGpu, int iGpuId);

// Deallocate pinned memory
void deallocatePinnedBornShotsGpu_3D(int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Born forward **********************************/
/******************************************************************************/
// Normal
void BornShotsFwdGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Ginsu
void BornShotsFwdGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Free surface
void BornShotsFwdFreeSurfaceGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// Free surface + Ginsu
void BornShotsFwdFreeSurfaceGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Born adjoint **********************************/
/******************************************************************************/
void BornShotsAdjGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

void BornShotsAdjGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

void BornShotsAdjFreeSurfaceGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

void BornShotsAdjFreeSurfaceGinsuGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId);

// void BornShotsAdjNoStreamGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

// /*********************************** Debug ************************************/
// // void imagingFwd_zLoop(double *model, double *data, int iGpu, int iGpuId);
// // void imagingFwd_yLoop(double *model, double *data, int iGpu, int iGpuId);
//
// void BornShotsFwdGpu_3D_Threads(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);


#endif
