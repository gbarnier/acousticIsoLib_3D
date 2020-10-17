/******************************************************************************/
/**************** Auxiliary functions - Normal propagation ********************/
/******************************************************************************/

/***************************** Common parts ***********************************/
// Source wavefield with an additional second time derivative
void computeTomoSrcWfldDt2Ginsu_3D(float *dev_sourcesIn, long long *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn){

    float *dummySliceRight;
	dummySliceRight = new float[host_nVel_ginsu[iGpu]];

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pSourceWavefield[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Compute coarse source wavefield sample at its = 0
	int its = 0;

	// Loop within the first two values of its (coarse time grid)
	for (int it2 = 1; it2 < host_sub+1; it2++){

		// Compute fine time-step index
		int itw = its * host_sub + it2;

		// Step forward
		stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

		// Inject source
		injectSourceLinear_3D<<<1, nSourcesRegIn, 0, compStreamIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionsRegIn);

		// Damp wavefields
		dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

		// Spread energy to dev_pLeft and dev_pRight
		interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

		// Switch pointers
		dev_temp1[iGpu] = dev_p0[iGpu];
		dev_p0[iGpu] = dev_p1[iGpu];
		dev_p1[iGpu] = dev_temp1[iGpu];
		dev_temp1[iGpu] = NULL;
	}

	// Copy pDt1 (its = 0)
	cuda_call(cudaMemcpyAsync(dev_pDt1[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Switch pointers
	dev_pTemp[iGpu] = dev_pLeft[iGpu];
	dev_pLeft[iGpu] = dev_pRight[iGpu];
	dev_pRight[iGpu] = dev_pTemp[iGpu];
	dev_pTemp[iGpu] = NULL;
	cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

	/************************** Main loop (its > 0) ***************************/
	for (int its = 1; its < host_nts-1; its++){

		// Loop within two values of its (coarse time grid)
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
		    stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject source
			injectSourceLinear_3D<<<1, nSourcesRegIn, 0, compStreamIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionsRegIn);

			// Damp wavefields
		    dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

            // Spread energy to dev_pLeft and dev_pRight
    		interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Copy source wavefield value at its into pDt2
		cuda_call(cudaMemcpyAsync(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Compute second-order time-derivative of source wavefield at its-1
	    srcWfldSecondTimeDerivativeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu], iGpu);
		cuda_call(cudaStreamSynchronize(compStreamIn));

        //////////////////////////////// Debug /////////////////////////////////
    	// cuda_call(cudaMemcpy(dummySliceRight, dev_pSourceWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
        // std::cout << "Source Ginsu = " << its << std::endl;
    	// std::cout << "its = " << its << std::endl;
    	// std::cout << "Min value dev_pSourceWavefield = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
    	// std::cout << "Max value dev_pSourceWavefield = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
    	////////////////////////////////////////////////////////////////////////

		// Wait for pStream to be free
		cuda_call(cudaStreamSynchronize(transferStreamIn));

		// Copy second time derivative of source wavefield at its-1 to pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pSourceWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Wait for pStream to be ready
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Copy second time derivative of source wavefield from device -> pinned memory for time sample its-1
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu]+(its-1)*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost, transferStreamIn));

		// Switch pointers
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Switch pointers for time derivative
		dev_pDtTemp[iGpu] = dev_pDt0[iGpu];
		dev_pDt0[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;

	}

	// Copy source wavefield at nts-1 into pDt2
	cuda_call(cudaMemcpyAsync(dev_pDt2[iGpu], dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute second-order time-derivative of source wavefield at nts-2
	srcWfldSecondTimeDerivativeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt0[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu], iGpu);

	// Transfer dev_pSourceWavefield (second-order time-derivative of source wavefield at nts-2) to pinned memory
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu]+(host_nts-2)*host_nVel_ginsu[iGpu], dev_pSourceWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost, compStreamIn));

	// Reset pDt0 and compute second-order time-derivative at nts-1
	cuda_call(cudaMemsetAsync(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));
	srcWfldSecondTimeDerivativeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pSourceWavefield[iGpu], dev_pDt1[iGpu], dev_pDt2[iGpu], dev_pDt0[iGpu], iGpu);

	// Transfer dev_pSourceWavefield (second-order time-derivative of source wavefield at nts-1) to pinned memory
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice1[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pSourceWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost, compStreamIn));

	//////////////////////////////// Debug /////////////////////////////////
	// cuda_call(cudaMemcpy(dummySliceRight, dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = " << its << std::endl;
	// std::cout << "Min value source wavefield = " << *std::min_element(pin_wavefieldSlice1[iGpu],pin_wavefieldSlice1[iGpu]+host_nts*host_nVel_ginsu[iGpu]) << std::endl;
	// std::cout << "Max value source wavefield = " << *std::max_element(pin_wavefieldSlice1[iGpu],pin_wavefieldSlice1[iGpu]+host_nts*host_nVel_ginsu[iGpu]) << std::endl;
	////////////////////////////////////////////////////////////////////////

}

// Receiver wavefield
void computeTomoRecWfldGinsu_3D(float *dev_dataRegDtsIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamIn){

    float *dummySliceRight;
	dummySliceRight = new float[host_nVel_ginsu[iGpu]];

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject data
			interpLinearInjectData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_dataRegDtsIn, dev_p0[iGpu], its, it2, dev_receiversPositionRegIn);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Wait until pStream has been transfered
		cuda_call(cudaStreamSynchronize(transferStreamIn));

		// Copy pRight (contains wavefield at its+1) into pStream
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Wait until pStream has been updated
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Transfer pStream -> pin (at its+1)
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu]+(its+1)*host_nVel_ginsu[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost, transferStreamIn));

        //////////////////////////////// Debug /////////////////////////////////
        // cuda_call(cudaMemcpy(dummySliceRight, dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
        // std::cout << "Receiver Ginsu = " << its << std::endl;
        // std::cout << "its = " << its << std::endl;
        // std::cout << "Min value dev_pStream = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
        // std::cout << "Max value dev_pStream = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
        ////////////////////////////////////////////////////////////////////////

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

	}

	// Wait until pStream has been transfered
	cuda_call(cudaStreamSynchronize(transferStreamIn));

 	// Transfer pStream -> pin (at its=0)
	cuda_call(cudaMemcpy(pin_wavefieldSlice2[iGpu], dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));

}

/************************** Forward - Time-lags *******************************/
// Source -> reflectivity -> model -> data
void computeTomoLeg1TauFwdGinsu_3D(float *dev_modelTomoIn, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){


	float *dummySliceRight;
	dummySliceRight = new float[host_nVel_ginsu[iGpu]];

	/**************************************************************************/
	/*************************** First part of leg #1 *************************/
	/*************** Source -> reflectivity -> scattered wavefield ************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel_ginsu[iGpu]*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel_ginsu[iGpu]*sizeof(float))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu]+iExt*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	/****************************** its = 0 ***********************************/
	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Launch transfer
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(2*host_hExt1+1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
	}

    //////////////////////////////// Debug /////////////////////////////////
    // cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "Part 1 Ginsu" << std::endl;
    // std::cout << "its = " << its << std::endl;
    // std::cout << "Min value dev_pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
    // std::cout << "Max value dev_pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
    ////////////////////////////////////////////////////////////////////////

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	// Otherwise, transfer slice its = 1 -> pSourceWavefieldTau
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	/****************************** Main loops ********************************/

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of the propagation
		if (its < 2*host_hExt1-1) {

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Launch transfer
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		// Last part of the propagation
		else {

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

        //////////////////////////////// Debug /////////////////////////////////
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
        // std::cout << "Part 1 Ginsu" << std::endl;
        // std::cout << "its = " << its << std::endl;
		// std::cout << "Min value dev_pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		// std::cout << "Max value dev_pRight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		////////////////////////////////////////////////////////////////////////

		for (int it2 = 1; it2 < host_sub+1; it2++){

            // Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

            // Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

            // Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Wait until the scattered wavefield has been transfered to pin
		cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

		// Asynchronous copy of dev_pDt1 => dev_pDt0 [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		//////////////////////////////// Debug /////////////////////////////////
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
        // std::cout << "Part 1 Ginsu" << std::endl;
        // std::cout << "its = " << its << std::endl;
		// std::cout << "Min value dev_pDt0 = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		// std::cout << "Max value dev_pDt0 = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		////////////////////////////////////////////////////////////////////////

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu]+its*host_nVel_ginsu[iGpu], dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
		}

		if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][0];
			for (int iExt1=0; iExt1<4*host_hExt1; iExt1++){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1+1];
			}
			dev_pSourceWavefieldTau[iGpu][4*host_hExt1] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until the transfer from pinned -> pStream is completed
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}
	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Wait until pDt1 -> pinned wavefield is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	//////////////////////////////// Debug /////////////////////////////////
	// cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = " << its << std::endl;
	// std::cout << "Min value dev_pRight = " << *std::min_element(pin_wavefieldSlice2[iGpu],pin_wavefieldSlice2[iGpu]+host_nVel_ginsu[iGpu]*host_nts) << std::endl;
	// std::cout << "Max value dev_pRight = " << *std::max_element(pin_wavefieldSlice2[iGpu],pin_wavefieldSlice2[iGpu]+host_nVel_ginsu[iGpu]*host_nts) << std::endl;
	////////////////////////////////////////////////////////////////////////

	/**************************************************************************/
	/*************************** Second part of leg #1 ************************/
	/***************** Scattered wavefield -> model -> data *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice2[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], iGpu);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+(its+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		//////////////////////////////// Debug /////////////////////////////////
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value dev_pRight = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		// std::cout << "Max value dev_pRight = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		////////////////////////////////////////////////////////////////////////

		for (int it2 = 1; it2 < host_sub+1; it2++){

            // Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

            // Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

        //////////////////////////////// Debug /////////////////////////////////
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pLeft[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
        // std::cout << "Part 2 Ginsu" << std::endl;
        // std::cout << "its = " << its << std::endl;
		// std::cout << "Min value dev_pLeft = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		// std::cout << "Max value dev_pLeft = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		////////////////////////////////////////////////////////////////////////

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

// Source -> model -> reflectivity -> data
void computeTomoLeg2TauFwdGinsu_3D(float *dev_modelTomoIn, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #2 *************************/
	/***************** Source -> model -> scattered wavefield *****************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy source wavefield time-slice its = 0: pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], iGpu);

	// Transfer new slice from pinned -> pStream for time its = 1
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(its+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

            // Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

            // Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

            // Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Wait until the scattered wavefield has been transfered to pin
		cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

		// Asynchronous copy of dev_pDt1 => dev_pDt0 [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu]+its*host_nVel_ginsu[iGpu], dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Wait until pDt1 -> pinned wavefield is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	//////////////////////////////// Debug /////////////////////////////////
	// cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = " << its << std::endl;
	// std::cout << "Min value dev_pRight = " << *std::min_element(pin_wavefieldSlice2[iGpu],pin_wavefieldSlice2[iGpu]+host_nVel_ginsu[iGpu]*host_nts) << std::endl;
	// std::cout << "Max value dev_pRight = " << *std::max_element(pin_wavefieldSlice2[iGpu],pin_wavefieldSlice2[iGpu]+host_nVel_ginsu[iGpu]*host_nts) << std::endl;
	////////////////////////////////////////////////////////////////////////

	/**************************************************************************/
	/*************************** Second part of leg #2 ************************/
	/*************** Scattered wavefield -> reflectivity -> data **************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Initialize source wavefield slices
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel_ginsu[iGpu]*sizeof(float))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){
			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice2[iGpu]+iExt*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	/****************************** its = 0 ***********************************/
	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Launch transfer slice 2*host_hExt1+1 from pinned -> device
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+(2*host_hExt1+1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	/****************************** Main loops ********************************/

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of the propagation
		if (its < 2*host_hExt1-1) {

			// Transfer slice from pinned -> device
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Transfer slice (its+2)+2*host_hExt1 from pinned -> device
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		// Last part of the propagation
		else {

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

            // Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStream[iGpu]>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

            // Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStream[iGpu]>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}

		if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][0];
			for (int iExt1=0; iExt1<4*host_hExt1; iExt1++){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1+1];
			}
			dev_pSourceWavefieldTau[iGpu][4*host_hExt1] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until the transfer from pinned -> pStream is completed
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}
	}

	// Wait for all jobs on GPU to be done
	cuda_call(cudaDeviceSynchronize());
}

/************************** Forward - Offsets *********************************/
// Source -> reflectivity -> model -> data
void computeTomoLeg1HxHyFwdGinsu_3D(float *dev_modelTomoIn, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	// float *dummySliceRight;
	// dummySliceRight = new float[host_nVel_ginsu[iGpu]];

	/**************************************************************************/
	/*************************** First part of leg #1 *************************/
	/*************** Source -> reflectivity -> scattered wavefield ************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
		}
	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFdGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Launch transfer of wavefield2 slice its+2 by transfering from host to device
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(its+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFdGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu], iGpu);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Wait until the scattered wavefield has been transfered to pin
		cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

		// Asynchronous copy of dev_pDt1 => dev_pDt0 [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		//////////////////////////////// Debug /////////////////////////////////
		// cuda_call(cudaMemcpy(dummySliceRight, dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
		// std::cout << "its = " << its << std::endl;
		// std::cout << "Min value dev_pDt0 = " << *std::min_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		// std::cout << "Max value dev_pDt0 = " << *std::max_element(dummySliceRight,dummySliceRight+host_nVel_ginsu[iGpu]) << std::endl;
		////////////////////////////////////////////////////////////////////////

		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu]+its*host_nVel_ginsu[iGpu], dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Wait until pDt1 -> pinned wavefield is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	/**************************************************************************/
	/*************************** Second part of leg #1 ************************/
	/***************** Scattered wavefield -> model -> data *******************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice2[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], iGpu);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+(its+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

// Source -> model -> reflectivity -> data
void computeTomoLeg2HxHyFwdGinsu_3D(float *dev_modelTomoIn, float *dev_dataRegDtsIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int nBlockDataIn, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, cudaStream_t transferStreamD2HIn){

	/**************************************************************************/
	/*************************** First part of leg #2 *************************/
	/******************** Source -> model -> scattered wavefield **************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	imagingFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], iGpu);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			cuda_call(cudaStreamSynchronize(compStreamIn));
			// Copy wavefield1 slice its+2 from RAM > dev_pStream
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(its+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Compute secondary source for first coarse time index (its+1) with compute stream
		imagingFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Wait until the scattered wavefield has been transfered to pin
		cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

		// Asynchronous copy of dev_pDt1 => dev_pDt0 [its] [compute]
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Wait until pDt0 is ready to be transfered
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from device -> host of wavefield2 at its
		cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu]+its*host_nVel_ginsu[iGpu], dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost, transferStreamD2HIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}

	// Wait until the scattered wavefield has been transfered to pin
	cuda_call(cudaStreamSynchronize(transferStreamD2HIn));

	// Load pLeft to pStream (value of wavefield at nts-1)
	cuda_call(cudaMemcpyAsync(pin_wavefieldSlice2[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Wait until pDt1 -> pinned wavefield is done
	cuda_call(cudaStreamSynchronize(compStreamIn));

	//////////////////////////////// Debug /////////////////////////////////////
	// cuda_call(cudaMemcpy(dummySliceRight, dev_pRight[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost));
	// std::cout << "its = " << its << std::endl;
	// std::cout << "Min value dev_pRight = " << *std::min_element(pin_wavefieldSlice2[iGpu],pin_wavefieldSlice2[iGpu]+host_nVel_ginsu[iGpu]*host_nts) << std::endl;
	// std::cout << "Max value dev_pRight = " << *std::max_element(pin_wavefieldSlice2[iGpu],pin_wavefieldSlice2[iGpu]+host_nVel_ginsu[iGpu]*host_nts) << std::endl;
	////////////////////////////////////////////////////////////////////////////

	/**************************************************************************/
	/*************************** Second part of leg #2 ************************/
	/*************** Scattered wavefield -> reflectivity -> data **************/
	/**************************************************************************/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy wavefield2 time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice2[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
		}
	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFdGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Copy new slice from RAM -> pinned for time its = 1 -> transfer to pStream
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){
			// Copy wavefield slice its+2 from RAM > dev_pStream
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+(its+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFdGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu], iGpu);

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Extract data
			recordLinearInterpData_3D<<<nBlockDataIn, BLOCK_SIZE_DATA, 0, compStreamIn>>>(dev_p0[iGpu], dev_dataRegDtsIn, its, it2, dev_receiversPositionRegIn);

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
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

		// Wait until the transfer from pinned -> pStream is completed
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	}
}

/************************* Adjoint - Time-lags ********************************/
// Source -> reflectivity -> model <- data
void computeTomoLeg1TauAdjGinsu_3D(float *dev_modelTomoIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	// cuda_call(cudaMemset(dev_pStream[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=0; iExt<4*host_hExt1+1; iExt++){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel_ginsu[iGpu]*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel_ginsu[iGpu]*sizeof(float))); // Useless

		// Load the source time-slices from its = 0,...,2*hExt1 (included)
		if (iExt < 2*host_hExt1+1){
			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice1[iGpu]+iExt*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	/****************************** its = 0 ***********************************/
	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = 0
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Transfer slice 2*host_hExt1+1 from pinned -> device
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(2*host_hExt1+1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = 0
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iSlice = its - 2 * (iExt-host_hExt1);
		imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice 2*host_hExt1+1 only if hExt1 > 0
	// Otherwise, transfer slice its = 1 -> pSourceWavefieldTau
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	// Copy receiver wavefield slice from pinned -> device for time its = 0 -> transfer to pDt0
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	/****************************** Main loops ********************************/

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Lower bound for imaging condition at its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its+1
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of the propagation
		if (its < 2*host_hExt1-1) {

			// Launch transfer from pinned -> device
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = (its+1) - 2 * (iExt-host_hExt1);
				imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}

		}

		// Middle part of the propagation (load the new slice and change pointers)
		else if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Transfer slice (its+2)+2*host_hExt1 from pinned -> device
			cuda_call(cudaStreamSynchronize(compStreamIn));
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(its+2*host_hExt1+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Compute imaging condition for its + 1 while the slice (its+2)+2*host_hExt1 is being transfered
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				// Index of the wavefield slice corresponding to iExt
				int iSlice = 4*host_hExt1 - 2*iExt;
				imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		// Last part of the propagation
		else {

			// Imaging condition for its+1
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){

				int iSlice = its + 2 + 6*host_hExt1 - 2*iExt - host_nts;
				imagingTauFwdGinsuGpu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], iExt, iGpu);
			}
		}

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its+1
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu]+(its+1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		/////////////////////////////// QC /////////////////////////////////////
		// std::cout << "its = " << its << std::endl;
		// float *dummySlice, *dummyModel ;
		// dummySlice = new float[host_nVel_ginsu[iGpu]];
		// dummyModel = new float[host_nVel_ginsu[iGpu]];
		// cudaMemcpy(dummySlice, dev_pRecWavefield[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost);
		// std::cout << "dev_pRecWavefield[iGpu] = " << *std::min_element(dummySlice,dummySlice+host_nVel_ginsu[iGpu]) << std::endl;
		// std::cout << "dev_pRecWavefield[iGpu] = " << *std::max_element(dummySlice,dummySlice+host_nVel_ginsu[iGpu]) << std::endl;
		// cudaMemcpy(dummyModel, dev_pDt1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToHost);
		// std::cout << "dev_pDt1[iGpu] = " << *std::min_element(dummyModel,dummyModel+host_nVel_ginsu[iGpu]) << std::endl;
		// std::cout << "dev_pDt1[iGpu] = " << *std::max_element(dummyModel,dummyModel+host_nVel_ginsu[iGpu]) << std::endl;
		////////////////////////////////////////////////////////////////////////

		// Apply imaging condition at its
		imagingAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu], iGpu);

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemsetAsync(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemsetAsync(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));

		if (its < 2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy new wavefield slice
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][its+2*host_hExt1+2], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
		}

		if (its >= 2*host_hExt1-1 && its < host_nts-2*host_hExt1-2) {

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][0];
			for (int iExt1=0; iExt1<4*host_hExt1; iExt1++){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1+1];
			}
			dev_pSourceWavefieldTau[iGpu][4*host_hExt1] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until the transfer from pinned -> pStream is completed
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][4*host_hExt1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}
	}

	// Copy receiver wavefield value at nts-1 from pDt0 -> pRecWavefield
	cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute imaging condition at its = nts-1
	imagingAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu], iGpu);

}

// Source -> model <- reflectivity <- data
void computeTomoLeg2TauAdjGinsu_3D(float *dev_modelTomoIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Allocate time-slices from 0,...,4*hExt1 (included)
	for (int iExt=4*host_hExt1; iExt>-1; iExt--){

		// Allocate source wavefield slice
		// cuda_call(cudaMalloc((void**) &dev_pSourceWavefieldTau[iGpu][iExt], host_nVel_ginsu[iGpu]*sizeof(float)));
		cuda_call(cudaMemset(dev_pSourceWavefieldTau[iGpu][iExt], 0, host_nVel_ginsu[iGpu]*sizeof(float))); // Useless

		// Load the source time-slices from its = 4*hExt1,...,2*hExt1 (included)
		if (iExt > 2*host_hExt1-1){

			// Transfer from pinned -> GPU
			cuda_call(cudaMemcpy(dev_pSourceWavefieldTau[iGpu][iExt], pin_wavefieldSlice2[iGpu]+(host_nts-1+iExt-4*host_hExt1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	// The last time slice loaded from the receiver wavefield is nts-1-2*hExt1
	// The index of the temporary wavefield for this slice is 2*host_hExt1

	/****************************** its = nts-1 *******************************/

	// Declare upper and lower bounds
	int iExtMin, iExtMax;

	// Do first imaging condition for its = nts-1
	int its = host_nts-1;
	iExtMin = -its/2;
	iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;
	iExtMax = (host_nts-1-its)/2;
	iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

	// Transfer slice nts-2-2*host_hExt1 from RAM to pStream
	cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+(host_nts-2-2*host_hExt1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

	// Imaging condition for its = nts-1
	for (int iExt=iExtMin; iExt<iExtMax; iExt++){

		int iSlice = 2*host_hExt1 - host_nts + 1 + its + 2*iExt;
		imagingTauTomoAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt, iGpu);
	}

	// Wait until the transfer to pStream has been done
	cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

	// Transfer slice nts-2-2*host_hExt1 only if hExt1 > 0
	// Otherwise, transfer slice its = nts-2 -> pSourceWavefieldTau[0]
	if (host_hExt1 > 0){
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][2*host_hExt1-1], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	} else {
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));
	}

	// At this point:
	// - The imaging condition at its = nts-1 is done (secondary source in pRight)
	// - Time-slice its = nts-2-2*host_hExt1 is loaded into dev_pSourceWavefieldTau[iGpu][2*host_hExt1-1]
	// - The imaging at its = nts-2 is ready
	/****************************** Main loop *********************************/

	// Start propagating adjoint wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Lower bound for imaging condition at its
		iExtMin = -its/2;
		iExtMin = std::max(iExtMin, -host_hExt1) + host_hExt1;

		// Upper bound for imaging condition at its
		iExtMax = (host_nts-1-its)/2;
		iExtMax = std::min(iExtMax, host_hExt1) + host_hExt1 + 1;

		// First part of adjoint propagation
		if (its > host_nts-2*host_hExt1-1){

			// Wait until compStream has done copying wavefield value from pStream -> dev_pSourceWavefield
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Copy slice its-2*host_hExt-1 from pinned -> device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+(its-2*host_hExt1-1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = 2*host_hExt1 - host_nts + 1 + its + 2*iExt;
				imagingTauTomoAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt, iGpu);
			}
			// At this point, the secondary source has been computed at
			// its = nts-1 and its = nts-2
			// So we can propagate the adjoint scattered wavefield from nts-1 to nts-2

		// Middle part of adjoint propagation
		} else if (its <= host_nts-2*host_hExt1-1 && its >= 2*host_hExt1+1){

			// Wait until compStream has done copying wavefield value from pStream -> dev_pSourceWavefield
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Copy slice its-2*host_hExt-1 from pinned -> device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice2[iGpu]+(its-2*host_hExt1-1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

			// Imaging condition for its
			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = 2*iExt;
				imagingTauTomoAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt, iGpu);
			}

		// Last part of adjoint propagation
		} else {

			for (int iExt=iExtMin; iExt<iExtMax; iExt++){
				int iSlice = its + 2*(iExt-host_hExt1);
				imagingTauTomoAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pSourceWavefieldTau[iGpu][iSlice], dev_extReflectivityIn, iExt, iGpu);
			}
		}

        // Load source wavefield at its+1 from host -> device
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu]+(its+1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

		// Start subloop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2+1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

        // Wait until source wavefield slice has been copied into dev_pSourceWavefield
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

		// Apply imaging condition at its+1
		imagingAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		// cuda_call(cudaMemsetAsync(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float), compStreamIn));
		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

		// First part of adjoint propagation
		if (its > host_nts-2*host_hExt1-1) {

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Compute index on the temporary receiver wavefield array
			int iSlice = 2*host_hExt1-host_nts+its;

			// Copy new wavefield slice from pStream -> pSourceWavefieldTau
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][iSlice], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Middle part of adjoint propagation
		} else if (its <= host_nts-2*host_hExt1-1 && its >= 2*host_hExt1+1){

			// Switch wavefield pointers
			dev_pTempTau[iGpu] = dev_pSourceWavefieldTau[iGpu][4*host_hExt1];
			for (int iExt1=4*host_hExt1; iExt1>0; iExt1--){
				dev_pSourceWavefieldTau[iGpu][iExt1] = dev_pSourceWavefieldTau[iGpu][iExt1-1];
			}
			dev_pSourceWavefieldTau[iGpu][0] = dev_pTempTau[iGpu];
			dev_pTempTau[iGpu] = NULL;

			// Wait until pStream is ready
			cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

			// Copy pStream -> device
			cuda_call(cudaMemcpyAsync(dev_pSourceWavefieldTau[iGpu][0], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		}

	}

	// Load source wavefield for its = 0
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Apply imaging condition at its = 0
	imagingAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu], iGpu);

}

/************************* Adjoint - Offsets **********************************/
// Source -> reflectivity -> model <- data
void computeTomoLeg1HxHyAdjGinsu_3D(float *dev_modelTomoIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy source wavefield time-slice its = 0: RAM -> pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpy(dev_pSourceWavefield[iGpu], pin_wavefieldSlice1[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Do first fwd imaging condition for its = 0 (after that, secondary source at its = 0 is done)
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pLeft[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
		}
	}

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFdGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_vel2Dtw2[iGpu], iGpu);
	cuda_call(cudaStreamSynchronize(compStreamIn));

	// Copy source wavefield slice from RAM -> pinned for time its = 1 -> transfer to pStream
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Copy receiver wavefield slice from RAM -> pinned for time its = 0 -> transfer to pDt0
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Copy source wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its < host_nts-2){

			// Wait until dev_pStream is ready to be used
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Load wavefield slice its+2 by transfering from host to device
			cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(its+2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Launch transfer from pin -> dev_pDt0 for receiver wavefield at its+1
		cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu]+(its+1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

		// Compute secondary source for first coarse time index (its+1) with compute stream
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_extReflectivityIn, dev_pRight[iGpu], dev_pSourceWavefield[iGpu], ihx, iExt1, ihy, iExt2, iGpu);
			}
		}

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFdGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_vel2Dtw2[iGpu], iGpu);

		// Start subloop
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			stepFwdGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2-1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// At this point, pDt1 contains the value of the scattered wavefield at its
		// The imaging condition can be done for its

		// Apply imaging condition at its
		imagingAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu], iGpu);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	}

	// Copy receiver wavefield value at nts-1 from pDt0 -> pRecWavefield
	cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute imaging condition at its = nts-1
	imagingAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt1[iGpu], dev_pRecWavefield[iGpu], iGpu);

}

// Source -> model <- reflectivity <- data
void computeTomoLeg2HxHyAdjGinsu_3D(float *dev_modelTomoIn, float *dev_extReflectivityIn, long long *dev_receiversPositionRegIn, dim3 dimGridIn, dim3 dimBlockIn, dim3 dimGrid32In, dim3 dimBlock32In, int iGpu, cudaStream_t compStreamIn, cudaStream_t transferStreamH2DIn, int nBlockDataIn){

	/************* Compute scattered wavefield and imaging condition **********/

	// Reset the time slices to zero
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pRight[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt0[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));
	cuda_call(cudaMemset(dev_pDt2[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	// Copy receiver wavefield time-slice its = nts-1
	// From RAM -> pinned -> dev_pSourceWavefield
	cuda_call(cudaMemcpy(dev_pRecWavefield[iGpu], pin_wavefieldSlice2[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	scaleSecondarySourceFdGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRecWavefield[iGpu], dev_vel2Dtw2[iGpu], iGpu);

	// Compute secondary source for its = nts-1
	for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
		long long iExt2 = ihy + host_hExt2;
		for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
			long long iExt1 = ihx + host_hExt1;
			imagingHxHyTomoAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRight[iGpu], dev_pRecWavefield[iGpu], dev_extReflectivityIn, ihx, iExt1, ihy, iExt2, iGpu);
		}
	}

	// Copy receiver wavefield slice from RAM -> pinned for time nts-2 -> transfer to pStream
	cuda_call(cudaMemcpy(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu]+(host_nts-2)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Copy source wavefield slice from RAM -> pinned for time its = nts-1 -> transfer to pDt0
	cuda_call(cudaMemcpy(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+(host_nts-1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice));

	// Start propagating scattered first scattered wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Copy receiver wavefield value at its from pDt0 -> pRecWavefield
		cuda_call(cudaMemcpyAsync(dev_pRecWavefield[iGpu], dev_pDt0[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		// Copy receiver wavefield value at its+1 from pStream -> pSourceWavefield
		cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

		if (its > 0){

			// Wait until dev_pStream is ready to be used
			cuda_call(cudaStreamSynchronize(compStreamIn));

			// Copy receiver wavefield slice its-1 from pinned -> device
			cuda_call(cudaMemcpyAsync(dev_pDt0[iGpu], pin_wavefieldSlice2[iGpu]+(its-1)*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));
		}

		// Wait until dev_pDt0 is ready to be used
		cuda_call(cudaStreamSynchronize(compStreamIn));

		// Copy source wavefield slice its from pinned -> device
		cuda_call(cudaMemcpyAsync(dev_pStream[iGpu], pin_wavefieldSlice1[iGpu]+its*host_nVel_ginsu[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyHostToDevice, transferStreamH2DIn));

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		scaleSecondarySourceFdGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pRecWavefield[iGpu], dev_vel2Dtw2[iGpu], iGpu);

		// Compute secondary source for its + 1
		for (int ihy = -host_hExt2; ihy <= host_hExt2; ihy++){
			long long iExt2 = ihy + host_hExt2;
			for (int ihx = -host_hExt1; ihx <= host_hExt1; ihx++){
				long long iExt1 = ihx + host_hExt1;
				imagingHxHyTomoAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRecWavefield[iGpu], dev_extReflectivityIn, ihx, iExt1, ihy, iExt2, iGpu);
			}
		}

		// Start subloop
		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint
			stepAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu], iGpu);

			// Inject secondary source sample itw-1
			injectSecondarySourceGinsu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_pLeft[iGpu], dev_pRight[iGpu], dev_p0[iGpu], it2+1, iGpu);

			// Damp wavefields
			dampCosineEdgeGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_p0[iGpu], dev_p1[iGpu], iGpu);

			// Spread energy to dev_pDt1 and dev_pDt2
			interpFineToCoarseSliceGinsu_32_3D<<<dimGrid32In, dimBlock32In, 0, compStreamIn>>>(dev_pDt1[iGpu], dev_pDt2[iGpu], dev_p0[iGpu], it2, iGpu);

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its+1
		imagingAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu], iGpu);

		// Wait until transfer stream has finished copying slice its from pinned -> pStream
		cuda_call(cudaStreamSynchronize(transferStreamH2DIn));

		// Switch pointers for secondary source
		dev_pTemp[iGpu] = dev_pRight[iGpu];
		dev_pRight[iGpu] = dev_pLeft[iGpu];
		dev_pLeft[iGpu] = dev_pTemp[iGpu];
		dev_pTemp[iGpu] = NULL;
		cuda_call(cudaMemset(dev_pLeft[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

		// Switch pointers for the scattered wavefield
		dev_pDtTemp[iGpu] = dev_pDt2[iGpu];
		dev_pDt2[iGpu] = dev_pDt1[iGpu];
		dev_pDt1[iGpu] = dev_pDtTemp[iGpu];
		dev_pDtTemp[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_pDt1[iGpu], 0, host_nVel_ginsu[iGpu]*sizeof(float)));

	}

	// Copy receiver wavefield value at its = 0 from pStream -> pSourceWavefield
	cuda_call(cudaMemcpyAsync(dev_pSourceWavefield[iGpu], dev_pStream[iGpu], host_nVel_ginsu[iGpu]*sizeof(float), cudaMemcpyDeviceToDevice, compStreamIn));

	// Compute imaging condition at its = 0
	imagingAdjGinsuGpu_3D<<<dimGridIn, dimBlockIn, 0, compStreamIn>>>(dev_modelTomoIn, dev_pDt2[iGpu], dev_pSourceWavefield[iGpu], iGpu);

}
