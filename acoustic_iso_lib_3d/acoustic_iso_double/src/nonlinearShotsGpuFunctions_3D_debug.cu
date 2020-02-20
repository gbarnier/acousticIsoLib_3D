// Debug Laplacian
void laplacianFwd_3d(double *modelDebug, double *dataDebug, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Model
	cuda_call(cudaMalloc((void**) &dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_modelDebug[iGpu], modelDebug, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
	cuda_call(cudaMalloc((void**) &dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemset(dev_dataDebug[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));

	// std::cout << "Inside Laplacian cuda forward 1" << std::endl;

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Apply forward Laplacian
	kernel_exec(LaplacianFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_modelDebug[iGpu], dev_dataDebug[iGpu], dev_vel2Dtw2[iGpu]));

	// std::cout << "Inside Laplacian cuda forward 2" << std::endl;

	// Copy data back to host
	cuda_call(cudaMemcpy(dataDebug, dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToHost));

}

void laplacianAdj_3d(double *modelDebug, double *dataDebug, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);

	// Model
	cuda_call(cudaMalloc((void**) &dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));

	// Data
	cuda_call(cudaMalloc((void**) &dev_dataDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double)));
	cuda_call(cudaMemcpy(dev_dataDebug[iGpu], dataDebug, host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyHostToDevice));

	// Initialize model to zero
	cuda_call(cudaMemset(dev_modelDebug[iGpu], 0, host_nz*host_nx*host_ny*sizeof(double)));

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);

	// Apply adjoint Laplacian
	kernel_exec(LaplacianAdjGpu_3D<<<dimGrid, dimBlock>>>(dev_modelDebug[iGpu], dev_dataDebug[iGpu],dev_vel2Dtw2[iGpu]));
	// kernel_exec(LaplacianFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_dataDebug[iGpu], dev_modelDebug[iGpu], dev_vel2Dtw2[iGpu]));

	// Copy data back to host
	cuda_call(cudaMemcpy(modelDebug, dev_modelDebug[iGpu], host_nz*host_nx*host_ny*sizeof(double), cudaMemcpyDeviceToHost));

}

void freeSurfaceDebugFwd(double *model, double *data, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);


	cuda_call(cudaMemcpy(dev_p0[iGpu], model, host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	kernel_exec(setFreeSurfaceConditionFwdGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p0[iGpu]));
	kernel_exec(derivFwdVelGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_vel2Dtw2[iGpu]));
	// kernel_exec(derivFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));
	// kernel_exec(derivFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));
	cuda_call(cudaMemcpy(data, dev_p1[iGpu],host_nModel*sizeof(double), cudaMemcpyDeviceToHost));

}

void freeSurfaceDebugAdj(double *model, double *data, int iGpu, int iGpuId) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpuId);
	cudaStreamCreate(&compStreamTop[iGpu]);
	cudaStreamCreate(&compStreamBody[iGpu]);
	cuda_call(cudaMemcpy(dev_p0[iGpu], data, host_nModel*sizeof(double), cudaMemcpyHostToDevice));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nModel*sizeof(double)));


	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE_Z;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE_X;
	int nblockz = (host_ny-2*FAT+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimGridTop(1, nblocky);
	dim3 dimGridBody(nblockx-1, nblocky);
	dim3 dimBlock(BLOCK_SIZE_Z, BLOCK_SIZE_X);
	dim3 dimGridFreeSurface(nblocky, nblockz);
	dim3 dimBlockFreeSurface(BLOCK_SIZE_X, BLOCK_SIZE_Y);

	// kernel_exec(setFreeSurfaceToZero<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p0[iGpu]));
	// derivTopAdjGpu_3D<<<dimGridTop, dimBlock, 0, compStreamTop[iGpu]>>>(dev_p1[iGpu], dev_p0[iGpu]);
	// derivBodyAdjGpu_3D<<<dimGridBody, dimBlock, 0, compStreamBody[iGpu]>>>(dev_p1[iGpu], dev_p0[iGpu]);
	kernel_exec(derivAdjVelGpu_3D<<<dimGrid, dimBlock>>>(dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
	// kernel_exec(derivFwdGpu_3D<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));
	// kernel_exec(setFreeSurfaceConditionAdjGpu_3D<<<dimGridFreeSurface, dimBlockFreeSurface>>>(dev_p1[iGpu]));

	cuda_call(cudaMemcpy(model, dev_p1[iGpu],host_nModel*sizeof(double), cudaMemcpyDeviceToHost));

}
