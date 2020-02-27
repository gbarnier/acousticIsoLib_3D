#ifndef BORN_SHOTS_GPU_FUNCTIONS_3D_H
#define BORN_SHOTS_GPU_FUNCTIONS_3D_H 1
#include <vector>

/* Parameter settings */
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initBornGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateBornShotsGpu_3D(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId);
void deallocateBornShotsGpu_3D(int iGpu, int iGpuId);

/************************************** Born FWD ****************************************/
void BornShotsFwdGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

void BornShotsFwdFreeSurfaceGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

void BornShotsFwdNoStreamGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

/************************************** Born ADJ ****************************************/
void BornShotsAdjGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

void BornShotsAdjFreeSurfaceGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

void BornShotsAdjNoStreamGpu_3D(double *model, double *dataRegDtw, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefield, int iGpu, int iGpuId);

/*********************************** Debug ************************************/
void imagingFwd_zLoop(double *model, double *data, int iGpu, int iGpuId);
void imagingFwd_yLoop(double *model, double *data, int iGpu, int iGpuId);


#endif
