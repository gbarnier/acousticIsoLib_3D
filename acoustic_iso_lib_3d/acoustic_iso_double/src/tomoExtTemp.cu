/******************************** Leg 1 ***************************************/
// Source -> reflectivity -> model -> data
// void computeTomoLeg1HxHyFwd_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_dataRegDtsIn, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){
//
// 	/**************************************************************************/
// 	/*************************** First part of leg #1 *************************/
// 	/**************************************************************************/
//
// 	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(double));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(double));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);
	cuda_call(cudaStreamSynchronize(compStreamIn)); // ?

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(double));
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStreamIn));
			// Wait until pStream is ready to be updated
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load wavefield slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

		// Asynchronous copy of dev_pDt1 => dev_pDt0 [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			// Standard library
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));

		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(double), compStreamIn)); // Reinitialize dev_pRight to zero (because of the += in the kernel)

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

	// In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));
//
// 	/**************************************************************************/
// 	/*************************** First part of leg #1 *************************/
// 	/**************************************************************************/
//
// 	// Scatter wavefield2 on model perturbation
//
// 	// Reset the time slices to zero
// 	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
// 	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
//
// 	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
// 	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(double));
// 	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));
//
// 	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
// 	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);
//
// 	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
// 	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+host_nVel, host_nVel*sizeof(double));
// 	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
// 	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));
//
// 	// At that point:
// 	// dev_pSourceWavefield contains wavefield at its=1
// 	// pin_wavefieldSlice and dev_pStream are free to be used
// 	// dev_pLeft (secondary source at its = 0) is computed
//
// 	// Start propagating scattered wavefield
// 	for (int its = 0; its < host_nts-1; its++){
//
// 		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
// 		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));
//
// 		if (its < host_nts-2){
// 			// Copy wavefield slice its+2 from RAM > dev_pStream
// 			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(its+2)*host_nVel, host_nVel*sizeof(double)); // -> this should be done with transfer stream
// 			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], wavefield2+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStreamH2DIn));
// 			cuda_call(cudaStreamSynchronize(compStreamIn));
// 			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
// 		}
//
// 		// Compute secondary source for first coarse time index (its+1) with compute stream
// 		imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);
//
// 		for (int it2 = 1; it2 < host_sub+1; it2++){
//
// 			// Step forward
// 			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);
//
// 			// Inject secondary source sample itw-1
// 			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);
//
// 			// Damp wavefields
// 			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);
//
// 			// Extract data
// 			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionRegIn);
//
// 			// Switch pointers
// 			dev_temp1[iGpu] = dev_p0[iGpu];
// 			dev_p0[iGpu] = dev_p1[iGpu];
// 			dev_p1[iGpu] = dev_temp1[iGpu];
// 			dev_temp1[iGpu] = NULL;
//
// 		}
//
// 		// Switch pointers for secondary source
// 		dev_pTemp[iGpu] = dev_pLeft[iGpu];
// 		dev_pLeft[iGpu] = dev_pRight[iGpu];
// 		dev_pRight[iGpu] = dev_pTemp[iGpu];
// 		dev_pTemp[iGpu] = NULL;
// 		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStreamIn));
//
// 		// Wait until the transfer from pinned -> pStream is completed
// 		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));
// 	}
//
// }

void computeTomoLeg1HxHyFwd_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_dataRegDtsIn, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #1 *************************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));
	cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(double));
	cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(double));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);
	cuda_call(cudaStreamSynchronize(compStreamIn)); // ?

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(double));
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStreamIn));
			// Wait until pStream is ready to be updated
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load wavefield slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2);

			// Extract data
			// recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

		// Asynchronous copy of dev_pDt1 => dev_pDt0 [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		// Save wavefield2 from pin -> RAM for its-1
		if (its>0) {
			// Standard library
			std::memcpy(wavefield2+(its-1)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));

		}

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel*sizeof(double), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel*sizeof(double), compStreamIn)); // Reinitialize dev_pRight to zero (because of the += in the kernel)

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

	// In the meantime, copy value of wavefield at nts-2 from pinned memory to RAM
	std::memcpy(wavefield2+(host_nts-2)*host_nVel, pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));

	// Wait until pDt1 -> pDt0 is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy pinned -> RAM
	std::memcpy(wavefield2+(host_nts-1)*host_nVel,pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double));

	/**************************************************************************/
	/*************************** Second part of leg #1 ************************/
	/**************************************************************************/

	// Scatter wavefield2 on model perturbation

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel*sizeof(double)));

	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu]);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// At that point:
	// dev_pSourceWavefield contains wavefield at its=1
	// pin_wavefieldSlice and dev_pStream are free to be used
	// dev_pLeft (secondary source at its = 0) is computed

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield2+(its+2)*host_nVel, host_nVel*sizeof(double)); // -> this should be done with transfer stream
			// cuda_call(cudaMemcpyAsync(pin_wavefieldSlice[iGpu], wavefield2+(its+2)*host_nVel, host_nVel*sizeof(double), cudaMemcpyHostToHost, transferStreamH2DIn));
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu]);

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionRegIn);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nModel*sizeof(double), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));
	}

}

// Source -> reflectivity -> model <- data
void computeTomoLeg1HxHyAdj_3D(double *dev_modelTomoIn, double *wavefield1, double *wavefield2, double *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn, double *dev_dataRegDtsQcIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel*sizeof(double)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(pin_wavefieldSlice1[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(pin_wavefieldSlice2[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel*sizeof(double)));
	cuda_call(cudaMemset(dev_pRecWavefield[iGpu], 0, host_nVel*sizeof(double)));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
		}
	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu]);
	cuda_call(cudaStreamSynchronize(compStreamIn)); // ?

	// Copy source wavefield slice from RAM -> pinned for time its = 1 -> transfer to pStream
	std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+host_nVel, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Copy receiver wavefield slice from RAM -> pinned for time its = 0 -> transfer to pDt0
	std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2, host_nVel*sizeof(double));
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy source wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel*sizeof(double), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){

			// Copy wavefield1 slice its+2 from RAM -> pin
			std::memcpy(pin_wavefieldSlice1[iGpu], wavefield1+(its+2)*host_nVel, host_nVel*sizeof(double));

			// Wait until dev_pStream is ready to be used
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load wavefield slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Copy wavefield2 slice its+1 from RAM -> pin
		std::memcpy(pin_wavefieldSlice2[iGpu], wavefield2+(its+1)*host_nVel, host_nVel*sizeof(double));

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its+1
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel*sizeof(double), cudaMemcpyHostToDevice, transferStreamH2DIn));

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFd_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu]);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]);

			// Inject secondary source sample itw-1
			injectSecondarySource_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1);

			// Damp wavefields
			dampCosineEdge_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu]);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsQcIn, its, it2, dev_receiversPositionRegIn);

			// Spread energy to dev_pLeft and dev_pRight
			interpFineToCoarseSlice_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_t
