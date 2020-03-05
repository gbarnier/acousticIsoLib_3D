#ifndef BORN_EXT_SHOTS_GPU_FUNCTIONS_3D_H
#define BORN_EXT_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/* Parameter settings */
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initBornExtGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt1, int nExt2, int nGpu, int iGpuId, int iGpuAlloc);
void allocateBornExtShotsGpu_3D(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId);
void deallocateBornExtShotsGpu_3D(int iGpu, int iGpuId);

/************************************** Born FWD ****************************************/
// void BornTimeShotsFwdGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

void BornHxShotsFwdGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

// void BornTimeShotsFwdFreeSurfaceGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);
//
// void BornOffsetShotsFwdFreeSurfaceGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

/************************************** Born ADJ ****************************************/
void BornHxShotsAdjGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

#endif
