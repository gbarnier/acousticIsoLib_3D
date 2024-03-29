#include "tomoExtGpu_3D.h"

tomoExtGpu_3D::tomoExtGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::shared_ptr<SEP::float5DReg> extReflectivity, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	// Finite-difference parameters
	_fdParam_3D = std::make_shared<fdParam_3D>(vel, par, "tomoExt");
	_timeInterp_3D = std::make_shared<interpTimeLinTbb_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_ots, _fdParam_3D->_sub);
	_secTimeDer = std::make_shared<secondTimeDerivative_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts);
	_extReflectivity = extReflectivity;
	_leg1 = par->getInt("leg1", 1);
	_leg2 = par->getInt("leg2", 1);
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;
	_ginsu = _fdParam_3D->_par->getInt("ginsu");
	_slowSquare = par->getInt("slowSquare");

	// Check parfile consistency for source and extended reflectivity
	if (_fdParam_3D->checkParfileConsistencySpace_3D(_extReflectivity, "Extended reflectivity file") != true) {
		throw std::runtime_error("**** ERROR [tomoExtGpu_3D]: Extended reflectivity not consistent with parfile ****");
	};

	// Initialize GPU
	if (_ginsu == 0){
		initTomoExtGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_extension, _fdParam_3D->_nExt1, _fdParam_3D->_nExt2, _leg1, _leg2, _nGpu, _iGpuId, iGpuAlloc);
	} else {
		initTomoExtGinsuGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_extension, _fdParam_3D->_nExt1, _fdParam_3D->_nExt2, _leg1, _leg2, _nGpu, _iGpuId, iGpuAlloc);
	}
}

void tomoExtGpu_3D::setTomoExtGinsuGpu_3D(std::shared_ptr<SEP::hypercube> velHyperGinsu, int xPadMinusGinsu, int xPadPlusGinsu, int ixGinsu, int iyGinsu, int iGpu, int iGpuId){

	// Update Ginsu parameters from fdParam
	_fdParam_3D->setFdParamGinsu_3D(velHyperGinsu, xPadMinusGinsu, xPadPlusGinsu, ixGinsu, iyGinsu, _extReflectivity);
	_iGpu = iGpu;
	_iGpuId = iGpuId;

	// Initialize GPU
	allocateSetTomoExtGinsuGpu_3D(_fdParam_3D->_nzGinsu, _fdParam_3D->_nxGinsu, _fdParam_3D->_nyGinsu, _fdParam_3D->_minPadGinsu, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_vel2Dtw2Ginsu, _fdParam_3D->_reflectivityScaleGinsu, _fdParam_3D->_extReflectivityGinsu, _iGpu, _iGpuId);

}

bool tomoExtGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::float3DReg> model, const std::shared_ptr<SEP::float2DReg> data) const {
	if (_fdParam_3D->checkParfileConsistencyTime_3D(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam_3D->checkParfileConsistencySpace_3D(model, "Model file") != true) {return false;}; // Check model space axes
	return true;
}

void tomoExtGpu_3D::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float2DReg> data) const {

	if (!add) data->scale(0.0);
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam_3D->_nts, _nReceiversReg));

	/* Tomo extended forward */

	// No free surface
	if (_fdParam_3D->_freeSurface != 1){

		// Time-lags extension
		if (_fdParam_3D->_extension == "time") {

			// No Ginsu
			if (_ginsu==0){
	            tomoTauShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);

			// Ginsu
			} else {
	            tomoTauShotsFwdGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _fdParam_3D->_extReflectivityGinsu, _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
			}
		}

		// Subsurface offset extension
		else if (_fdParam_3D->_extension == "offset") {

			// No Ginsu
			if (_ginsu==0){
				// std::cout << "fwd 1" << std::endl;
				// std::cout << "slowSquare = " << _slowSquare << std::endl;
				// std::cout << "model max" << model->max() << std::endl;
				// std::cout << "model max" << model->min() << std::endl;
	            tomoHxHyShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
				// std::cout << "dataRegDts max" << dataRegDts->max() << std::endl;
				// std::cout << "dataRegDts max" << dataRegDts->min() << std::endl;
			// Ginsu
			} else {
	            tomoHxHyShotsFwdGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _fdParam_3D->_extReflectivityGinsu, _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
			}
		}
		else {
			std::cout << "**** ERROR [tomoExtGpu_3D]: Please specify the type of extension ****" << std::endl;
			assert(1==2);
		}

	// Free surface
	} else {

		// Time-lags extension
		if (_fdParam_3D->_extension == "time") {

			// No Ginsu
			if (_ginsu==0){
            	tomoTauShotsFwdFsGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);

			// Ginsu
			} else {
            	tomoTauShotsFwdFsGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _fdParam_3D->_extReflectivityGinsu, _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
			}
		}

		// Subsurface offset extension
		else if (_fdParam_3D->_extension == "offset"){

			// No Ginsu
			if (_ginsu==0){
            	tomoHxHyShotsFwdFsGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);

			// Ginsu
			} else {
            	tomoHxHyShotsFwdFsGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _fdParam_3D->_extReflectivityGinsu, _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
			}
		}
		else {
			std::cout << "**** ERROR [tomoExtGpu_3D]: Please specify the type of extension (time of offset) ****" << std::endl; throw std::runtime_error("");
		}
	}

	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);
}

void tomoExtGpu_3D::adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);

	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<float3DReg> modelTemp = model->clone(); // We need to create a temporary model for "add"
	modelTemp->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Tomo extended adjoint */

	// No free surface
	if (_fdParam_3D->_freeSurface != 1){

		// Time-lags extension
		if (_fdParam_3D->_extension == "time") {

			// No Ginsu
			if (_ginsu==0){
				tomoTauShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);

			// Ginsu
			} else {
				tomoTauShotsAdjGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _fdParam_3D->_extReflectivityGinsu, _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
			}

		// Subsurface offset extension
		} else if (_fdParam_3D->_extension == "offset") {

			// No Ginsu
			if (_ginsu==0){
				// std::cout << "adj 1" << std::endl;
				// std::cout << "slowSquare = " << _slowSquare << std::endl;
				// std::cout << "data max" << dataRegDts->max() << std::endl;
				// std::cout << "data max" << dataRegDts->min() << std::endl;
				tomoHxHyShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
				// std::cout << "model max" << modelTemp->max() << std::endl;
				// std::cout << "model max" << modelTemp->min() << std::endl;

			// Ginsu
			} else {
				// std::cout << "Time 2" << std::endl;
				tomoHxHyShotsAdjGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _fdParam_3D->_extReflectivityGinsu, _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
			}
		}
		else {
			std::cout << "**** ERROR [tomoExtGpu_3D]: Please specify the type of extension (time of offset) ****" << std::endl; throw std::runtime_error("");
		}

	// Free surface
	} else {

		// Time-lags extension
		if (_fdParam_3D->_extension == "time") {
			// std::cout << "Time 1" << std::endl;
			// No Ginsu
			if (_ginsu==0){
				// std::cout << "Time 2" << std::endl;
				tomoTauShotsAdjFsGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);

			// Ginsu
			} else {
				// std::cout << "Time 3" << std::endl;
				tomoTauShotsAdjFsGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _fdParam_3D->_extReflectivityGinsu, _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
			}
		// Subsurface offset extension
		} else if (_fdParam_3D->_extension == "offset") {
			// std::cout << "Offset 1" << std::endl;
			// No Ginsu
			if (_ginsu==0){
				// std::cout << "adj 1" << std::endl;
				// std::cout << "slowSquare = " << _slowSquare << std::endl;
				// std::cout << "data max" << dataRegDts->max() << std::endl;
				// std::cout << "data max" << dataRegDts->min() << std::endl;
				tomoHxHyShotsAdjFsGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
				// std::cout << "model max" << modelTemp->max() << std::endl;
				// std::cout << "model max" << modelTemp->min() << std::endl;

			// Ginsu
			} else {
				// std::cout << "Offset2 1" << std::endl;
				tomoHxHyShotsAdjFsGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _fdParam_3D->_extReflectivityGinsu, _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _slowSquare, _iGpu, _iGpuId);
			}
 		} else {
			std::cout << "**** ERROR [tomoExtGpu_3D]: Please specify the type of extension (time of offset) ****" << std::endl; throw std::runtime_error("");
		}
	}

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);
}
