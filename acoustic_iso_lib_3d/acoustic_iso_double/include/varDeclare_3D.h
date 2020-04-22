#ifndef VAR_DECLARE_3D_H
#define VAR_DECLARE_3D_H 1

#include <math.h>
#define BLOCK_SIZE_Z 16
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE 16
#define BLOCK_SIZE_EXT 8
#define BLOCK_SIZE_DATA 128
#define FAT 4

#define PI_CUDA M_PI // Import the number "Pi" from the math library
#define PAD_MAX 200 // Maximum number of points for padding (on one side)
#define SUB_MAX 100 // Maximum subsampling value for time

#define min2(v1,v2) (((v1)<(v2))?(v1):(v2)) /* Minimum function */
#define max2(v1,v2) (((v1)>(v2))?(v1):(v2)) /* Minimum function */

// Laplacian coefficients
#define COEFF_SIZE 13
#define C0  0
#define CZ1 1
#define CX1 2
#define CY1 3
#define CZ2 4
#define CX2 5
#define CY2 6
#define CZ3 7
#define CX3 8
#define CY3 9
#define CZ4 10
#define CX4 11
#define CY4 12
#define C_C00(d) (8.0/(5.0*(d)*(d)))
#define get_coeffs(d1,d2,d3) {-1025.0/576.0*(C_C00(d1)+C_C00(d2)+C_C00(d3)), C_C00(d1), C_C00(d2), C_C00(d3),-C_C00(d1)/8.0,-C_C00(d2)/8.0,-C_C00(d3)/8.0,C_C00(d1)/63.0,C_C00(d2)/63.0,C_C00(d3)/63.0,-C_C00(d1)/896.0,-C_C00(d2)/896.0,-C_C00(d3)/896.0}

/******************************************************************************/
/*************************** Declaration on device ****************************/
/******************************************************************************/
// Device function
__device__ int min4(int v1,int v2,int v3,int v4){return min2(min2(v1,v2),min2(v3,v4));}

// Constant memory
__constant__ double dev_coeff[COEFF_SIZE];
__constant__ int dev_nTimeInterpFilter; // Time interpolation filter length
__constant__ int dev_hTimeInterpFilter; // Time interpolation filter half-length
__constant__ double dev_timeInterpFilter[2*(SUB_MAX+1)]; // Time interpolation filter stored in constant memory

__constant__ int dev_nts; // Number of time steps at the coarse time sampling on Device
__constant__ int dev_ntw; // Number of time steps at the fine time sampling on Device
__constant__ int dev_nz; // nz on Device
__constant__ int dev_nx; // nx on Device
__constant__ int dev_ny; // ny on Device
__constant__ long long dev_yStride; // nz * nx on Device
__constant__ unsigned long long dev_nModel; // nz * nx * ny on Device
__constant__ unsigned long long dev_nModelExt; // nz * nx * ny * nExt1 * nExt2 on Device
__constant__ unsigned long long dev_nVel; // nz * nx * ny on Device
__constant__ unsigned long long dev_extStride; // nExt1 * nz * nx * ny on Device
__constant__ int dev_sub; // Subsampling in time
__constant__ int dev_nExt1, dev_nExt2; // Length of extension axis
__constant__ int dev_hExt1, dev_hExt2; // Half-length of extension axis

__constant__ int dev_nSourcesReg; // Nb of source grid points
__constant__ int dev_nReceiversReg; // Nb of receiver grid points

__constant__ double dev_alphaCos; // Decay coefficient
__constant__ int dev_minPad; // Minimum padding length
__constant__ double dev_cosDampingCoeff[PAD_MAX]; // Padding array
__constant__ double dev_cSide;
__constant__ double dev_cCenter;

// Global memory
long long **dev_sourcesPositionReg; // Array containing the positions of the sources on the regular grid
long long **dev_receiversPositionReg; // Array containing the positions of the receivers on the regular grid
double **dev_p0, **dev_p1, **dev_p2, **dev_temp1, **dev_p1_temp; // Temporary slices for stepping
double **dev_vel2Dtw2; // Precomputed scaling v^2 * dtw^2

// Nonlinear modeling
double **dev_modelRegDtw; // Model for nonlinear propagation (wavelet)
double **dev_dataRegDts; // Data on device at coarse time-sampling (converted to regular grid)
double **dev_dataRegDtsQc;
double *dev_wavefieldDts; // Source wavefield

// Born
double **dev_ssLeft, **dev_ssRight, **dev_ssTemp1; // Temporary slices for stepping for Born
double **dev_sourcesSignals, **dev_reflectivityScale, **dev_extReflectivity;
double **dev_modelBorn, **dev_modelBornExt, **dev_modelTomo;
double **dev_pLeft, **dev_pRight, **dev_pTemp, **dev_pTempTau;
double **dev_pDt0, **dev_pDt1, **dev_pDt2, **dev_pDtTemp, **dev_pWavefieldSliceDt2;

// Streams
double **pin_wavefieldSlice, **pin_wavefieldSlice1, **pin_wavefieldSlice2, **dev_pStream, **dev_pSourceWavefield, **dev_pRecWavefield;
double ***dev_pSourceWavefieldTau;
cudaStream_t *compStream, *transferStream, *topStream;
cudaStream_t *transferStreamH2D, *transferStreamD2H;

// Events
cudaEvent_t eventTopFreeSurface, eventBodyFreeSurface, compStreamDone;

// Debug
double **dev_modelDebug, **dev_dataDebug;

/******************************************************************************/
/**************************** Declaration on host *****************************/
/******************************************************************************/
int host_nz; // Includes padding + FAT
int host_nx;
int host_ny;
long long host_yStride;
unsigned long long host_nModel;
unsigned long long host_nModelExt;
unsigned long long host_nVel;
unsigned long long host_extStride;
double host_dz;
double host_dx;
double host_dy;
int host_nts;
double host_dts;
int host_ntw;
int host_sub;
int host_nExt1, host_nExt2; // Length of extended axis
int host_hExt1, host_hExt2; // Half-length of extended axis
double host_cSide, host_cCenter; // Coefficients for the second-order time derivative
int host_leg1, host_leg2;
std::string host_extension;

#endif
