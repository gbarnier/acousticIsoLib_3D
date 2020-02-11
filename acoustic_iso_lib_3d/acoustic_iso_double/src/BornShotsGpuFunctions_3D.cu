#include <cstring>
#include <iostream>
#include "BornShotsGpuFunctions_3D.h"
#include "varDeclare_3D.h"
#include "kernelsGpu_3D.cu"
#include "cudaErrors_3D.cu"
#include <vector>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

/****************************************************************************************/
/******************************* Set GPU propagation parameters *************************/
/****************************************************************************************/
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

void initBornGpu_3D(double dz, double dx, double dy, int nz, int nx, int ny, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

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

	/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new double*[nGpu];
		dev_p1 = new double*[nGpu];
		dev_temp1 = new double*[nGpu];

		dev_pLeft = new double*[nGpu];
		dev_pRight = new double*[nGpu];
		dev_pTemp = new double*[nGpu];

		// Data and model
		dev_dataRegDts = new double*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new long long*[nGpu];
		dev_receiversPositionReg = new long long*[nGpu];

        // Sources signal
		dev_sourcesSignals = new double*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new double*[nGpu];

        // Reflectivity scaling
		dev_reflectivityScale = new double*[nGpu];

        // Reflectivity
		dev_modelBorn = new double*[nGpu];

		// Debug model and data
		dev_modelDebug = new double*[nGpu];
		dev_dataDebug = new double*[nGpu];

        // Streams
		compStream = new cudaStream_t[nGpu];
		transferStream = new cudaStream_t[nGpu];
		pin_wavefieldSlice = new double*[nGpu];
		dev_pStream = new double*[nGpu];
		dev_pSourceWavefield = new double*[nGpu];



	}

	/**************************** COMPUTE LAPLACIAN COEFFICIENTS ************************/
	double host_coeff[COEFF_SIZE] = get_coeffs((double)dz,(double)dx,(double)dy); // Stored on host
    // for (int iCoeff=0; iCoeff<COEFF_SIZE; iCoeff++){
	// 	std::cout << "Coeff [" << iCoeff << "] = " << host_coeff[iCoeff]<< std::endl;
    // }

	/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
	// Time interpolation filter length / half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Subsampling parameter for time interpolation is too high ****" << std::endl;
		assert (1==2);
	}

	// Allocate and fill time interpolation filter
	double interpFilter[nInterpFilter];
	for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
		interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
		interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
		interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
		interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
	}

	// std::cout << "Size of time filter" << 2*(SUB_MAX+1) << std::endl;
	// for (int i = 0; i < nInterpFilter; i++){
	// 	std::cout << "interpFilter[" << i << "] =" << interpFilter[i] << std::endl;
	// }

	/************************* COMPUTE COSINE DAMPING COEFFICIENTS **********************/
	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Padding value is too high ****" << std::endl;
		assert (1==2);
	}
	double cosDampingCoeff[minPad];

	// Cosine padding
	for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
		double arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		cosDampingCoeff[iFilter-FAT] = arg;
	}

	// Check that the block size is consistent between parfile and "varDeclare.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR [nonlinearShotsGpuFunctions_3D]: Blocksize value from parfile does not match value from varDeclare file ****" << std::endl;
		assert (1==2);
	}

	/**************************** COPY TO CONSTANT MEMORY *******************************/
	// Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_coeff, host_coeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice)); // Copy Laplacian coefficients to device

	// Time interpolation filter
	cuda_call(cudaMemcpyToSymbol(dev_nTimeInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
	cuda_call(cudaMemcpyToSymbol(dev_hTimeInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
	cuda_call(cudaMemcpyToSymbol(dev_timeInterpFilter, interpFilter, nInterpFilter*sizeof(double), 0, cudaMemcpyHostToDevice)); // Filter

	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(double), 0, cudaMemcpyHostToDevice)); // Array for damping
	cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(double), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula
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

void allocateBornShotsGpu_3D(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId){

	// Get GPU number
	cudaSetDevice(iGpuId);

	// Scaled velocity
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nz*host_nx*host_ny*sizeof(double))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

    // Reflectivity scale
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nz*host_nx*host_ny*sizeof(double))); // Allocate scaling for reflectivity
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice)); //

	// Allocate time slices on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nz*host_nx*host_ny*sizeof(double))); // Allocate time slices on device (for the stepper)
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pLeft[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
    cuda_call(cudaMalloc((void**) &dev_pRight[iGpu], host_nz*host_nx*host_ny*sizeof(double)));

    // Reflectivity model
    cuda_call(cudaMalloc((void**) &dev_modelBorn[iGpu], host_nz*host_nx*host_ny*sizeof(double)));

	// Allocate pinned memory on host
	cuda_call(cudaHostAlloc((void**) &pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaHostAllocDefault));

	// Allocate the slice where we store the wavefield slice before transfering it to the host's pinned memory
	cuda_call(cudaMalloc((void**) &dev_pStream[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_pSourceWavefield[iGpu], host_nz*host_nx*host_ny*sizeof(double)));

}

void deallocateBornShotsGpu_3D(int iGpu, int iGpuId){
	cudaSetDevice(iGpuId);
	cuda_call(cudaFree(dev_vel2Dtw2[iGpu]));
    cuda_call(cudaFree(dev_reflectivityScale[iGpu]));
	cuda_call(cudaFree(dev_p0[iGpu]));
	cuda_call(cudaFree(dev_p1[iGpu]));
    cuda_call(cudaFree(dev_pLeft[iGpu]));
    cuda_call(cudaFree(dev_pRight[iGpu]));
    cuda_call(cudaFree(dev_modelBorn[iGpu]));
	cuda_call(cudaFreeHost(pin_wavefieldSlice[iGpu]));
}

/******************************************************************************/
/****************************** Born forward **********************************/
/******************************************************************************/
void BornShotsFwdGpu_3D(double *model, double *dataRegDts, double *sourcesSignals, long long *sourcesPositionReg, int nSourcesReg, long long *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Create streams
	cudaStreamCreate(&compStream[iGpu]);
	cudaStreamCreate(&transferStream[iGpu]);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Sources geometry + signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(long long)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(long long), cudaMemcpyHostToDevice));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nModel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));

	// Initialize time-slices for transfer to host's pinned memory
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nModel*sizeof(double)));

	// Initialize pinned memory
	cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nModel*sizeof(double));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocks data recording
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// QC shits
	std::cout << "Number of blocks in z-drection for Laplacian:" << nblockx << std::endl;
	std::cout << "Number of blocks in x-drection for Laplacian:" << nblocky << std::endl;
	std::cout << "Number of blocks in x-drection for Laplacian:" << nblocky << std::endl;
	std::cout << "Block size for Laplacian:" << BLOCK_SIZE_Z << std::endl;
	std::cout << "Number of blocks for data extraction:" << nblockData << std::endl;
	std::cout << "Block size for data extraction:" << BLOCK_SIZE_DATA << std::endl;

	/********************** Source wavefield computation **********************/

	// cudaEvent_t start,stop;
	// cudaEventCreate(&start);
	// float ms;
	// cudaEventCreate(&stop);
	// cudaEventRecord(start, 0);

	for (int its = 0; its < host_nts-1; its++){
		std::cout << "its #1 = " << its << std::endl;
		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesReg, 0, compStream[iGpu]>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pLeft and dev_pRight
			interpFineToCoarseSlice<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], its, it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
		std::cout << "its #2 = " << its << std::endl;
		/* Note: At that point pLeft [its] is ready to be transfered back to host */
		// Synchronize [transfer] (make sure the temporary device array dev_pStream has been transfered to host)
		cuda_call(cudaStreamSynchronize(transferStream[iGpu])); // Blocks host until all issued cuda calls in transfer stream are completed
		// At that point, the value of pStream has been transfered back to host pinned memory
		std::cout << "its #3 = " << its << std::endl;
		// Asynchronous copy of dev_pLeft => dev_pStream [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
		// At the same time, request CPU to memcpy the pin_wavefieldSlice to wavefield [its-1] [host]
		std::cout << "its #4 = " << its << std::endl;
		if (its>0) {
			// Standard library
			std::memcpy(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));

			// Using HostToHost
			// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(its-1)*host_nModel, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));

		}
		// The CPU has to wait until the memcpy from pinned -> RAM is done to launch next command
		// Synchronize [compute] (make sure the copy from dev_pLeft => dev_pStream is done)
		cuda_call(cudaStreamSynchronize(compStream[iGpu]));
		// Asynchronous transfer of pStream => pin [its] [transfer]
		// Launch the transfer while we compute the next coarse time sample
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost, transferStream[iGpu]));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStream[iGpu])); // Reinitialize dev_pRight to zero (because of the += in the kernel)
	}

	// At this point, pLeft contains the value of the wavefield at the last time sample, nts-1
	// The CPU has stored the wavefield values ranging from 0,...,nts-3

	// Wait until pStream (which contains the wavefield at nts-2) has transfered value to pinned memory
	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pLeft[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));

	// In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(srcWavefieldDts+(host_nts-2)*host_nModel, pin_wavefieldSlice[iGpu], host_nModel*sizeof(double));
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-2)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));

	// Wait until pLeft -> pStream is done
	cuda_call(cudaStreamSynchronize(compStream[iGpu]));

	// At this point, pStream contains the value of the wavefield at nts-1
	// Transfer pStream -> pinned
	cuda_call(cudaMemcpy(pin_wavefieldSlice[iGpu], dev_pStream[iGpu], host_nModel*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(srcWavefieldDts+(host_nts-1)*host_nModel,pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double)); // Copy pinned array to wavefield array for the last sample [nts-1] [host]
	// cuda_call(cudaMemcpyAsync(srcWavefieldDts+(host_nts-1)*host_nz*host_nx*host_ny, pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));

	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&ms, start, stop);
	// std::cout << "Duration: " << ms/1000 << " [s]" << std::endl;

	/********************** Scattered wavefield computation *******************/

	// Reset the time slices to zero
	// cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));
  	// cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));
	// cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));
	// cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));
  	// cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));
	// cudaMemset(pin_wavefieldSlice[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double));
	//
	// // Copy model to device
	// cuda_call(cudaMemcpy(dev_modelBorn[iGpu], model, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device
	//
	// // Allocate and initialize data
  	// cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	// cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device
	//
	// // Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
	// // Should this be done on the CPU to avoid allocating an additional time-slice on the GPU?
	// scaleReflectivity_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]);

	/************************ Streaming stuff starts **************************/

	// Copy wavefield time-slice its = 0: RAM -> pinned -> dev_pStream -> dev_pSourceWavefield
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nz*host_nx*host_ny*sizeof(double));
	// // cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
	//
	// // Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	// imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);
	//
	// // Copy new slice from RAM -> pinned for time its=1 -> transfer to pStream
	// std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+host_nz*host_nx*host_ny, host_nz*host_nx*host_ny*sizeof(double));
	// cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToDevice, transferStream[iGpu]));
	// cuda_call(cudaStreamSynchronize(transferStream[iGpu]));

	// At that point:
	// dev_pSourceWavefield contains wavefield at its=1
	// pin_wavefieldSlice and dev_pStream are free to be used
	// dev_pLeft (secondary source at its = 0) is computed

	// Start propagating scattered wavefield
	// for (int its = 0; its < host_nts-1; its++){
	//
	// 	if (its < host_nts-2){
	// 		// Copy wavefield slice its+2 from RAM to pinned
	// 		std::memcpy(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nz*host_nx*host_ny, host_nz*host_nx*host_ny*sizeof(double)); // -> this should be done with transfer stream
	// 		// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], srcWavefieldDts+(its+2)*host_nz*host_nx*host_ny, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToHost, transferStream[iGpu]));
	// 		// Start asynchronous transfer from pinned -> pStream for time-slice its+2
	// 		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice, transferStream[iGpu]));
	// 	}
	//
	// 	// Compute secondary source for first coarse time index (its+1) with compute stream
	// 	// Propagate with compute stream
	// 	imagingFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_modelBorn[iGpu], dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
	//
	// 	for (int it2 = 1; it2 < host_sub+1; it2++){
	//
	// 		// Step forward
	// 		stepFwdGpu_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
	//
	// 		// Inject secondary source sample itw-1
	// 		injectSecondarySource_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);
	//
	// 		// Damp wavefields
	// 		dampCosineEdge_3D<<<dimGrid, dimBlock, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu]);
	//
	// 		// Extract data
	// 		recordLinearInterpData_3D<<<nblockData, BLOCK_SIZE_DATA, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]);
	//
	// 		// Switch pointers
	// 		dev_temp1[iGpu] = dev_p0[iGpu];
	// 		dev_p0[iGpu] = dev_p1[iGpu];
	// 		dev_p1[iGpu] = dev_temp1[iGpu];
	// 		dev_temp1[iGpu] = NULL;
	//
	// 	}
	//
	// 	// Wait until the transfer from pinned -> pStream is completed
	// 	cuda_call(cudaStreamSynchronize(transferStream[iGpu]));
	//
	// 	// At that point, dev_pStream contains the wavefield at its+2
	// 	// Copy wavefield value at its+2 from pStream -> pSourceWavefield
	// 	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToDevice, compStream[iGpu]));
	//
	// 	// Switch pointers for secondary source
	// 	dev_pTemp[iGpu] = dev_pLeft[iGpu];
	// 	dev_pLeft[iGpu] = dev_pRight[iGpu];
	// 	dev_pRight[iGpu] = dev_pTemp[iGpu];
	// 	dev_pTemp[iGpu] = NULL;
	// 	// cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double), compStream[iGpu]));
	//
	// }
	//
	// // Copy data back to host
	// cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
	//
	// /*************************** Memory deallocation **************************/
	// // Deallocate the array for sources/receivers' positions
    // cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    // cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	// cuda_call(cudaFree(dev_dataRegDts[iGpu]));
	//
	// // Calls that should be moved from here when debugging is done
    // cuda_call(cudaStreamDestroy(compStream[iGpu]));
    // cuda_call(cudaStreamDestroy(transferStream[iGpu]));

}

/******************************************************************************/
/****************************** Benchmark *************************************/
/******************************************************************************/
// Benchmark for imaging condition
void imagingFwd_zLoop(double *model, double *data, int iGpu, int iGpuId){

	// Set device number
	cudaSetDevice(iGpuId);

	// Allocate and copy model to dev_modelDebug
	cuda_call(cudaMalloc((void**) &dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_modelDebug[iGpu], model, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate, copy data to dev_dataDebug and initialize it
	cuda_call(cudaMalloc((void**) &dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemset(dev_dataDebug[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));

	// Allocate slice for testing imaging kernel
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_p1[iGpu], model, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocks for wavefield extraction
	int nblockyModel = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockzModel = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGridModel(nblockyModel, nblockzModel);
	dim3 dimBlockModel(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	float ms;
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for (int iRep=0; iRep<10000; iRep++){
		kernel_exec(imagingFwdGpu_3D_zLoop<<<dimGridModel, dimBlockModel>>>(dev_modelDebug[iGpu], dev_dataDebug[iGpu], dev_p1[iGpu]));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "duration for z-loop: " << ms/1000 << " [s]" << std::endl;

	// Copy data back to host
	cuda_call(cudaMemcpy(data, dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToHost));
	std::cout << "Done zloop" << std::endl;
}

void imagingFwd_yLoop(double *model, double *data, int iGpu, int iGpuId){

	// Set device number
	cudaSetDevice(iGpuId);

	// Allocate and copy model to dev_modelDebug
	cuda_call(cudaMalloc((void**) &dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_modelDebug[iGpu], model, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate, copy data to dev_dataDebug and initialize it
	cuda_call(cudaMalloc((void**) &dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemset(dev_dataDebug[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));

	// Allocate slice for testing imaging kernel
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_p1[iGpu], model, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Blocks for Laplacian
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Blocks for wavefield extraction
	int nblockyModel = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockzModel = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGridModel(nblockyModel, nblockzModel);
	dim3 dimBlockModel(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	float ms;
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Timer
	// std::clock_t start;
	// double duration;
	// start = std::clock();

	for (int iRep=0; iRep<10000; iRep++){
		kernel_exec(imagingFwdGpu_3D_yLoop<<<dimGrid, dimBlock>>>(dev_modelDebug[iGpu], dev_dataDebug[iGpu], dev_p1[iGpu]));
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms, start, stop);
	std::cout << "duration for y-loop: " << ms/1000 << " [s]" << std::endl;

	// duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	// duration*=1000;


	// Copy data back to host
	cuda_call(cudaMemcpy(data, dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToHost));
	std::cout << "Done yloop" << std::endl;
}