#include "BornExtGpu_3D.h"

BornExtGpu_3D::BornExtGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParam_3D = std::make_shared<fdParam_3D>(vel, par, "BornExt"); // Fd parameter object
	_timeInterp_3D = std::make_shared<interpTimeLinTbb_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_ots, _fdParam_3D->_sub); // Time interpolation object
	_secTimeDer = std::make_shared<secondTimeDerivative_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts); // Second time derivative object
	_iGpu = iGpu; // Gpu number
	_nGpu = nGpu; // Number of requested GPUs
	_iGpuId = iGpuId;
	_ginsu = _fdParam_3D->_par->getInt("ginsu");

	// Initialize GPU
	if (_ginsu == 0){
		initBornExtGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_extension, _fdParam_3D->_nExt1, _fdParam_3D->_nExt2, _nGpu, _iGpuId, iGpuAlloc);
	} else {
		initBornExtGinsuGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_extension, _fdParam_3D->_nExt1, _fdParam_3D->_nExt2, _nGpu, _iGpuId, iGpuAlloc);
	}
}

void BornExtGpu_3D::setBornExtGinsuGpu_3D(std::shared_ptr<SEP::hypercube> velHyperGinsu, int xPadMinusGinsu, int xPadPlusGinsu, int ixGinsu, int iyGinsu, int iGpu, int iGpuId){

	// Update Ginsu parameters from fdParam
	_fdParam_3D->setFdParamGinsu_3D(velHyperGinsu, xPadMinusGinsu, xPadPlusGinsu, ixGinsu, iyGinsu);
	_iGpu = iGpu;
	_iGpuId = iGpuId;

	// Initialize GPU
	allocateSetBornExtGinsuGpu_3D(_fdParam_3D->_nzGinsu, _fdParam_3D->_nxGinsu, _fdParam_3D->_nyGinsu, _fdParam_3D->_minPadGinsu, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_vel2Dtw2Ginsu, _fdParam_3D->_reflectivityScaleGinsu, _iGpu, _iGpuId);

}

bool BornExtGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::float5DReg> model, const std::shared_ptr<SEP::float2DReg> data) const {
	if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam_3D->checkParfileConsistencyTime_3D(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam_3D->checkParfileConsistencySpace_3D(model, "Model file") != true) {return false;}; // Check model space axes
	return true;
}

void BornExtGpu_3D::forward(const bool add, const std::shared_ptr<float5DReg> model, std::shared_ptr<float2DReg> data) const {

	if (!add) data->scale(0.0);

	/* Allocation */
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam_3D->_nts, _nReceiversReg));
	dataRegDts->scale(0.0);

	/* Launch Born extended forward */

	// No free surface
	if (_fdParam_3D->_freeSurface != 1){

		// Time-lags extension
		if (_fdParam_3D->_extension == "time") {

			// No Ginsu
			if (_ginsu==0){
				BornTauShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			// Ginsu
			} else {
				BornTauShotsFwdGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
		}
		// Subsurface offset extension
		else if (_fdParam_3D->_extension == "offset") {

			// No Ginsu
			if (_ginsu==0){
				BornHxHyShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			// Ginsu
			} else {
				BornHxHyShotsFwdGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}

		} else {
			std::cout << "**** ERROR [BornExtGpu_3D]: Please specify the type of extension (time of offset) ****" << std::endl; throw std::runtime_error("");
		}
	// Free surface modeling
	} else {

		// Time-lags extension
		if (_fdParam_3D->_extension == "time") {

			// No Ginsu
			if (_ginsu==0){
				BornTauFreeSurfaceShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
			// Ginsu
			else {
				BornTauFreeSurfaceShotsFwdGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
		}

		// Subsurface offsets
		else if (_fdParam_3D->_extension == "offset") {
			// No Ginsu
			if (_ginsu==0){
				BornHxHyFreeSurfaceShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
			// Ginsu
			else {
				BornHxHyFreeSurfaceShotsFwdGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
		}
		else {
			std::cout << "**** ERROR [BornExtGpu_3D]: Please specify the type of extension (time of offset) ****" << std::endl; throw std::runtime_error("");
		}
	}

	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);

}
void BornExtGpu_3D::adjoint(const bool add, std::shared_ptr<float5DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<float5DReg> modelTemp = model->clone();
	modelTemp->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Launch Born extended adjoint */

	// No free surface
	if (_fdParam_3D->_freeSurface != 1){

		// Time-lags extension
		if (_fdParam_3D->_extension == "time") {

			// No ginsu
			if (_ginsu==0){
				BornTauShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
			// Ginsu
			else {
				BornTauShotsAdjGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}

		// Subsurface offsets
		} else if (_fdParam_3D->_extension == "offset") {

			// No ginsu
			if (_ginsu==0){
				BornHxHyShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
			// Ginsu
			else {
				BornHxHyShotsAdjGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
		} else {
			std::cout << "**** ERROR [BornExtGpu_3D]: Please specify the type of extension (time of offset) ****" << std::endl; throw std::runtime_error("");
		}

	// Free surface modeling
	} else {

		// Time-lags extension
		if (_fdParam_3D->_extension == "time") {

			// No ginsu
			if (_ginsu==0) {
				BornTauFreeSurfaceShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);

			// Ginsu
			} else {
				BornTauFreeSurfaceShotsAdjGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
		}

		// Subsurface offsets
		else if (_fdParam_3D->_extension == "offset") {
			// No ginsu
			if (_ginsu==0){
				BornHxHyFreeSurfaceShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
			// Ginsu
			else {
				BornHxHyFreeSurfaceShotsAdjGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			}
		} else {
			std::cout << "**** ERROR [BornExtGpu_3D]: Please specify the type of extension (time of offset) ****" << std::endl; throw std::runtime_error("");
		}
	}

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);
}
