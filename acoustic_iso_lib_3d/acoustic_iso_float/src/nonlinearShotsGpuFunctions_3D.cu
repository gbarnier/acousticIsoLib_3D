#include <cstring>
#include <iostream>
#include "nonlinearShotsGpuFunctions_3D.h"
#include "varDeclare_3D.h"
#include "kernelsGpu_3D.cu"
#include "kernelsGinsuGpu_3D.cu"
#include "cudaErrors_3D.cu"
#include <vector>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <assert.h>

/****************************************************************************************/
/******************************* Set GPU propagation parameters *************************/
/****************************************************************************************/
// GPU info
bool getGpuInfo_3D(std::vector<int> gpuList, int info, int deviceNumberInfo){

	int nDevice, driver;
	cudaGetDeviceCount(&nDevice);

	if (info == 1){

		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "---------------------------- INFO FOR GPU# " << deviceNumberInfo << " ----------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;

		// Number of devices
		std::cout << "Number of requested GPUs: " << gpuList.size() << std::endl;
		std::cout << "Number of available GPUs: " << nDevice << std::endl;
		std::cout << "Id of requested GPUs: ";
		for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
			if (iGpu<gpuList.size()-1){std::cout << gpuList[iGpu] << ", ";}
 			else{ std::cout << gpuList[iGpu] << std::endl;}
		}

		// Driver version
		std::cout << "Cuda driver version: " << cudaDriverGetVersion(&driver) << std::endl;

		// Get properties
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop,deviceNumberInfo);

		// Display
		std::cout << "Name: " << dprop.name << std::endl;
		std::cout << "Total global memory: " << dprop.totalGlobalMem/(1024*1024*1024) << " [GB] " << std::endl;
		std::cout << "Shared memory per block: " << dprop.sharedMemPerBlock/1024 << " [kB]" << std::endl;
		std::cout << "Number of register per block: " << dprop.regsPerBlock << std::endl;
		std::cout << "Warp size: " << dprop.warpSize << " [threads]" << std::endl;
		std::cout << "Maximum pitch allowed for memory copies in bytes: " << dprop.memPitch/(1024*1024*1024) << " [GB]" << std::endl;
		std::cout << "Maximum threads per block: " << dprop.maxThreadsPerBlock << std::endl;
		std::cout << "Maximum block dimensions: " << "(" << dprop.maxThreadsDim[0] << ", " << dprop.maxThreadsDim[1] << ", " << dprop.maxThreadsDim[2] << ")" << std::endl;
		std::cout << "Maximum grid dimensions: " << "(" << dprop.maxGridSize[0] << ", " << dprop.maxGridSize[1] << ", " << dprop.maxGridSize[2] << ")" << std::endl;
		std::cout << "Total constant memory: " << dprop.totalConstMem/1024 << " [kB]" << std::endl;
		std::cout << "Number of streaming multiprocessors on device: " << dprop.multiProcessorCount << std::endl;
		if (dprop.deviceOverlap == 1) {std::cout << "Device can simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.deviceOverlap != 1) {std::cout << "Device cannot simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.canMapHostMemory == 1) { std::cout << "Device can map host memory" << std::endl; }
		if (dprop.canMapHostMemory != 1) { std::cout << "Device cannot map host memory" << std::endl; }
		if (dprop.concurrentKernels == 1) {std::cout << "Device can support concurrent kernel" << std::endl;}
		if (dprop.concurrentKernels != 1) {std::cout << "Device cannot support concurrent kernel execution" << std::endl;}

		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

	// Check that the number of requested GPU is less or equal to the total number of available GPUs
	if (gpuList.size()>nDevice) {
		std::cout << "**** ERROR [getGpuInfo_3D]: Number of requested GPU greater than available GPUs ****" << std::endl;
		return false;
	}

	// Check that the GPU numbers in the list are between 0 and nGpu-1
	for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
		if (gpuList[iGpu]<0 || gpuList[iGpu]>nDevice-1){
			std::cout << "**** ERROR [getGpuInfo_3D]: One of the element of the GPU Id list is not a valid GPU Id number ****" << std::endl;
			return false;
		}
	}

	return true;
}

// Init normal
void initNonlinearGpu_3D(float dz, float dx, float dy, int nz, int nx, int ny, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nz = nz;
	host_nx = nx;
    host_ny = ny;
	host_nModel = nz * nx * ny;
	host_yStride = nz * nx;
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;
	host_minPad = minPad;

	/************************* ALLOCATE ARRAYS OF ARRAYS **********************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new float*[nGpu];
		dev_p1 = new float*[nGpu];
		dev_temp1 = new float*[nGpu];

		// Data and model
		dev_modelRegDtw = new float*[nGpu];
		dev_dataRegDts = new float*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new float*[nGpu];

		// Damping slice
		dev_dampingSlice = new float*[nGpu];

		// Debug model and data
		dev_modelDebug = new float*[nGpu];
		dev_dataDebug = new float*[nGpu];

		// Compute and transfer stream
		topStream = new cudaStream_t[nGpu];
		compStream = new cudaStream_t[nGpu];

	}

	/********************* COMPUTE LAPLACIAN COEFFICIENTS *********************/
	// Compute coefficients for 8th order central finite difference Laplacian
	float host_coeff[COEFF_SIZE] = get_coeffs((float)dz,(float)dx,(float)dy); // Stored on host

	/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****" << std::endl;
		throw std::runtime_error("");
	}

	// Allocate and fill time interpolation filter
	float interpFilter[nInterpFilter];
	for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
		interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
		interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
		interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
		interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
	}

	/************************* COMPUTE COSINE DAMPING COEFFICIENTS **********************/
	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Padding value is too high ****" << std::endl;
		throw std::runtime_error("");
	}
	float cosDampingCoeff[minPad];

	// Cosine padding
	for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
		float arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		cosDampingCoeff[iFilter-FAT] = arg;
		// std::cout << "Damp array gpu [" << iFilter-FAT << "] = " << arg << std::endl;
	}

	// Check that the block size is consistent between parfile and "varDeclare_3D.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare.h file ****" << std::endl;
		throw std::runtime_error("");
	}

	/**************************** COPY TO CONSTANT MEMORY *******************************/
	// Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_coeff, host_coeff, COEFF_SIZE*sizeof(float), 0, cudaMemcpyHostToDevice)); // Copy Laplacian coefficients to device

	// Time interpolation filter
	cuda_call(cudaMemcpyToSymbol(dev_nTimeInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
	cuda_call(cudaMemcpyToSymbol(dev_hTimeInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
	cuda_call(cudaMemcpyToSymbol(dev_timeInterpFilter, interpFilter, nInterpFilter*sizeof(float), 0, cudaMemcpyHostToDevice)); // Filter

	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(float), 0, cudaMemcpyHostToDevice)); // Array for damping
	cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(float), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula
	cuda_call(cudaMemcpyToSymbol(dev_minPad, &minPad, sizeof(int), 0, cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nz, &nz, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ny, &ny, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_yStride, &host_yStride, sizeof(long long), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nModel, &host_nModel, sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}

// Allocate normal
void allocateNonlinearGpu_3D(float *vel2Dtw2, int iGpu, int iGpuId){

	// Get GPU number
	cudaSetDevice(iGpuId);

	// Scaled velocity
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nModel*sizeof(float))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nModel*sizeof(float), cudaMemcpyHostToDevice));

	// Allocate time slices on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nModel*sizeof(float))); // Allocate time slices on device (for the stepper)
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nModel*sizeof(float)));
}

// Init Ginsu
void initNonlinearGinsuGpu_3D(float dz, float dx, float dy, int nts, float dts, int sub, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;

	/************************ ALLOCATE ARRAYS OF ARRAYS ***********************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new float*[nGpu];
		dev_p1 = new float*[nGpu];
		dev_temp1 = new float*[nGpu];

		// Data and model
		dev_modelRegDtw = new float*[nGpu];
		dev_dataRegDts = new float*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new float*[nGpu];

		// Damping slice
		dev_dampingSlice = new float*[nGpu];

		// Debug model and data
		dev_modelDebug = new float*[nGpu];
		dev_dataDebug = new float*[nGpu];

		// Compute and transfer stream
		topStream = new cudaStream_t[nGpu];
		compStream = new cudaStream_t[nGpu];

	}

	/*********************** COMPUTE LAPLACIAN COEFFICIENTS *******************/
	// Compute coefficients for 8th order central finite difference Laplacian
	float host_coeff[COEFF_SIZE] = get_coeffs((float)dz,(float)dx,(float)dy); // Stored on host

	/********************** COMPUTE TIME-INTERPOLATION FILTER *****************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (nGpu>N_GPU_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Number of requested GPUs exceeds number allowed for constant memory storage ****" << std::endl;
		throw std::runtime_error("");
	}

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****" << std::endl;
		throw std::runtime_error("");
	}

	// Allocate and fill time interpolation filter
	float interpFilter[nInterpFilter];
	for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
		interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
		interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
		interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
		interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
	}

	// Check that the block size is consistent between parfile and "varDeclare_3D.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare.h file ****" << std::endl;
		throw std::runtime_error("");
	}

	/**************************** COPY TO CONSTANT MEMORY *******************************/
	// Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_coeff, host_coeff, COEFF_SIZE*sizeof(float), 0, cudaMemcpyHostToDevice)); // Copy Laplacian coefficients to device

	// Time interpolation filter
	cuda_call(cudaMemcpyToSymbol(dev_nTimeInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
	cuda_call(cudaMemcpyToSymbol(dev_hTimeInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
	cuda_call(cudaMemcpyToSymbol(dev_timeInterpFilter, interpFilter, nInterpFilter*sizeof(float), 0, cudaMemcpyHostToDevice)); // Filter

	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(float), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula

	// Time parameters
	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}

// Allocate Ginsu
void allocateSetNonlinearGinsuGpu_3D(int nz, int nx, int ny, int minPad, int blockSize, float alphaCos, float *vel2Dtw2, int iGpu, int iGpuId){

	// Set GPU
	cudaSetDevice(iGpuId);

	// Host variables
	host_nz_ginsu[iGpu] = nz;
	host_nx_ginsu[iGpu] = nx;
    host_ny_ginsu[iGpu] = ny;
	host_nModel_ginsu[iGpu] = nz * nx * ny;
	host_yStride_ginsu[iGpu] = nz * nx;
	host_minPad_ginsu[iGpu] = minPad;

	/******************** COMPUTE COSINE DAMPING COEFFICIENTS *****************/
	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Padding value is too high ****" << std::endl;
		throw std::runtime_error("");
	}

	// Allocate array to store damping coefficients on host
	float host_cosDampingCoeffGinsuTemp[host_minPad_ginsu[iGpu]];

	// Compute array coefficients
	for (int iFilter=FAT; iFilter<FAT+host_minPad_ginsu[iGpu]; iFilter++){
		float arg = M_PI / (1.0 * host_minPad_ginsu[iGpu]) * 1.0 * (host_minPad_ginsu[iGpu]-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		host_cosDampingCoeffGinsuTemp[iFilter-FAT] = arg;
	}

	// Check that the block size is consistent between parfile and "varDeclare_3D.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare.h file ****" << std::endl;
		throw std::runtime_error("");
	}

	/********************** COPY TO CONSTANT MEMORY ***************************/
	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeffGinsuConstant, &host_cosDampingCoeffGinsuTemp, host_minPad_ginsu[iGpu]*sizeof(float), iGpu*PAD_MAX*sizeof(float), cudaMemcpyHostToDevice)); // Array for damping
	cuda_call(cudaMemcpyToSymbol(dev_minPad_ginsu, &host_minPad_ginsu[iGpu], sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nz_ginsu, &nz, sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_nx_ginsu, &nx, sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_ny_ginsu, &ny, sizeof(int), iGpu*sizeof(int), cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_yStride_ginsu, &host_yStride_ginsu[iGpu], sizeof(long long), iGpu*sizeof(long long), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nModel_ginsu, &host_nModel_ginsu[iGpu], sizeof(unsigned long long), iGpu*sizeof(unsigned long long), cudaMemcpyHostToDevice));

	// Copy to global memory
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nModel_ginsu[iGpu]*sizeof(float))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nModel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Allocate time slices on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nModel_ginsu[iGpu]*sizeof(float))); // Allocate time slices on device (for the stepper)
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nModel_ginsu[iGpu]*sizeof(float)));
}

// Deallocate
void deallocateNonlinearGpu_3D(int iGpu, int iGpuId){
	cudaSetDevice(iGpuId); // Set device number on GPU cluster
	cuda_call(cudaFree(dev_vel2Dtw2[iGpu])); // Deallocate scaled velocity
	cuda_call(cudaFree(dev_p0[iGpu]));
	cuda_call(cudaFree(dev_p1[iGpu]));
}

/****************************************************************************************/
/******************************* Nonlinear forward propagation **************************/
/****************************************************************************************/
void propShotsFwdGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate output on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize output on device

	// Time slices
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(float)));
	// cuda_call(cudaMemset(dev_dampingSlice[iGpu], 0, host_nModel*sizeof(float)));

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Loop over coarse time samples
	for (long long its = 0; its < host_nts-1; its++){

		// Loop over sub loop
		for (long long it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			long long itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSourceLinear_3D<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract and interpolate data
			kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	// Deallocate
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsFwdGinsuGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate output on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize output on device

	// Time slices
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(float)));

	// Laplacian grid and blocks
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Loop over coarse time samples
	for (long long its = 0; its < host_nts-1; its++){

		// Loop over sub loop
		for (long long it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			long long itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGinsuGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu));

			// Inject source
			kernel_exec(injectSourceLinear_3D<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu));

			// Extract and interpolate data
			kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	// Deallocate
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsFwdFreeSurfaceGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate output on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize output on device

	// Time slices
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(float)));

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Start propagation

	// Loop over coarse time samples
	for (long long its = 0; its < host_nts-1; its++){
		// Loop over sub loop
		for (long long it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			long long itw = its * host_sub + it2;

			// Apply free surface conditions for Laplacian
			kernel_exec(setFreeSurfaceConditionFwdGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p1[iGpu]));

			// Step forward in time
			kernel_exec(stepFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSourceLinear_3D<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract and interpolate data
			kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	// Deallocate
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsFwdFreeSurfaceGinsuGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate output on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize output on device

	// Time slices
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(float)));

	// Laplacian grid and blocks
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny_ginsu[iGpu]-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Start propagation
	// Loop over coarse time samples
	for (long long its = 0; its < host_nts-1; its++){
		// Loop over sub loop
		for (long long it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			long long itw = its * host_sub + it2;

			// Apply free surface conditions for Laplacian
			kernel_exec(setFreeSurfaceConditionFwdGinsuGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p1[iGpu], iGpu));

			// Step forward in time
			kernel_exec(stepFwdGinsuGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu));

			// Inject source
			kernel_exec(injectSourceLinear_3D<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32>>>(dev_p1[iGpu], dev_p0[iGpu], iGpu));

			// Extract and interpolate data
			kernel_exec(recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	// Deallocate
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

/****************************************************************************************/
/******************************* Nonlinear adjoint propagation **************************/
/****************************************************************************************/
void propShotsAdjGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate model on device
  	cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(float))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(float)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Loop over coarse time samples
	for (int its = host_nts-2; its > -1; its--){
		// Loop over sub loop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward in time
			kernel_exec(stepAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdge_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract model
			kernel_exec(recordSource_3D<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float), cudaMemcpyDeviceToHost));

	// Deallocate all slices
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsAdjGinsuGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate model on device
  	cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(float))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(float)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Loop over coarse time samples
	for (int its = host_nts-2; its > -1; its--){
		// Loop over sub loop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward in time
			kernel_exec(stepAdjGinsuGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu));

			// Inject data
			kernel_exec(interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeGinsu_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu));

			// Extract model
			kernel_exec(recordSource_3D<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float), cudaMemcpyDeviceToHost));

	// Deallocate all slices
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsAdjFreeSurfaceGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&topStream[iGpu]);
	cudaStreamCreate(&compStream[iGpu]);

	// Create to synchornize top/body for free surface computation
	cudaEventCreate(&eventTopFreeSurface);
	cudaEventCreate(&eventBodyFreeSurface);
	cudaEventCreate(&compStreamDone);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate model on device
  	cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(float))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(float)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Loop over coarse time samples
	for (int its = host_nts-2; its > -1; its--){
		// Loop over sub loop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step backward in time
			kernel_exec(stepAdjFreeSurfaceGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdgeFreeSurface_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract model
			kernel_exec(recordSource_3D<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float), cudaMemcpyDeviceToHost));

	// Deallocate all slices
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void propShotsAdjFreeSurfaceGinsuGpu_3D(float *modelRegDtw, float *dataRegDts, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&topStream[iGpu]);
	cudaStreamCreate(&compStream[iGpu]);

	// Create to synchornize top/body for free surface computation
	cudaEventCreate(&eventTopFreeSurface);
	cudaEventCreate(&eventBodyFreeSurface);
	cudaEventCreate(&compStreamDone);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Model
  	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate model on device
  	cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(float))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel_ginsu[iGpu]*sizeof(float)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx_ginsu[iGpu]-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny_ginsu[iGpu]-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocksize = 32
	int nblockx32 = (host_nz_ginsu[iGpu]-2*FAT+32-1) / 32;
	int nblocky32 = (host_nx_ginsu[iGpu]-2*FAT+32-1) / 32;
	dim3 dimGrid32(nblockx32, nblocky32);
	dim3 dimBlock32(32, 32);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Loop over coarse time samples
	for (int its = host_nts-2; its > -1; its--){
		// Loop over sub loop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step backward in time
			kernel_exec(stepAdjFreeSurfaceGinsuGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu));

			// Inject data
			kernel_exec(interpLinearInjectData_3D<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdgeFreeSurfaceGinsu_32_3D<<<dimGrid32, dimBlock32>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu));

			// Extract model
			kernel_exec(recordSource_3D<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(float), cudaMemcpyDeviceToHost));

	// Deallocate all slices
    cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}
