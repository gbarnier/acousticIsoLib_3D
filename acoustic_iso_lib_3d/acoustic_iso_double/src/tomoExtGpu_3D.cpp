#include "tomoExtGpu_3D.h"

tomoExtGpu_3D::tomoExtGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::shared_ptr<SEP::double5DReg> extReflectivity, std::shared_ptr<SEP::double4DReg> wavefield1, std::shared_ptr<SEP::double4DReg> wavefield2, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	// Finite-difference parameters
	_fdParam_3D = std::make_shared<fdParam_3D>(vel, par);
	_timeInterp_3D = std::make_shared<interpTimeLinTbb_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_ots, _fdParam_3D->_sub);
	_secTimeDer = std::make_shared<secondTimeDerivative_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts);
	_extReflectivity = extReflectivity;
	_leg1 = par->getInt("leg1", 1);
	_leg2 = par->getInt("leg2", 1);
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;

	// Check parfile consistency for source and extended reflectivity
	if (_fdParam_3D->checkParfileConsistencySpace_3D(_extReflectivity, "Extended reflectivity file") != true) {assert(1==2);};

	// Allocate the source wavefield on the RAM
	_wavefield1 = wavefield1; // Point to wavefield
	_wavefield2 = wavefield2; // Point to wavefield
	unsigned long long int _wavefieldSize = _fdParam_3D->_zAxis.n * _fdParam_3D->_xAxis.n * _fdParam_3D->_yAxis.n;
	_wavefieldSize = _wavefieldSize * _fdParam_3D->_nts*sizeof(double) / (1024*1024*1024);

	// Initialize GPU
	initTomoExtGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx,  _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_extension, _fdParam_3D->_nExt1, _fdParam_3D->_nExt2, _leg1, _leg2, _nGpu, _iGpuId, iGpuAlloc);
}

bool tomoExtGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double2DReg> data) const {
	if (_fdParam_3D->checkParfileConsistencyTime_3D(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam_3D->checkParfileConsistencySpace_3D(model, "Model file") != true) {return false;}; // Check model space axes
	return true;
}

void tomoExtGpu_3D::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double2DReg> data) const {

	if (!add) data->scale(0.0);
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));

	/* Tomo extended forward */
	if (_fdParam_3D->_freeSurface != 1){
		if (_fdParam_3D->_extension == "time") {
			// std::cout << "Time-lags, max val model before = " << model->max() << std::endl;
			// std::cout << "Time-lags, min val model before = " << model->min() << std::endl;
			// std::cout << "Time-lags, max val data before = " << dataRegDts->max() << std::endl;
			// std::cout << "Time-lags, min val data before = " << dataRegDts->min() << std::endl;

            tomoTauShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId);
			// std::cout << "Time-lags, max val model after = " << model->max() << std::endl;
			// std::cout << "Time-lags, min val model after = " << model->min() << std::endl;
			// std::cout << "Time-lags, max val data after = " << dataRegDts->max() << std::endl;
			// std::cout << "Time-lags, min val data after = " << dataRegDts->min() << std::endl;
		}
		else if (_fdParam_3D->_extension == "offset") {
			// std::cout << "Offsets, max val model before = " << model->max() << std::endl;
			// std::cout << "Offsets, min val model before = " << model->min() << std::endl;
			// std::cout << "Offsets, max val data before = " << dataRegDts->max() << std::endl;
			// std::cout << "Offsets, min val data before = " << dataRegDts->min() << std::endl;
			// std::cout << "Wavefield1 before max = " << _wavefield1->max() << std::endl;
			// std::cout << "Wavefield1 before min = " << _wavefield1->min() << std::endl;

			// for (int i=0; i<_nSourcesReg; i++){
			// 	std::cout << "Position for source[" << i << "] = " << _sourcesPositionReg[i] << std::endl;
			// }

            tomoHxHyShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId);



			// std::cout << "Wavefield1 after max = " << _wavefield1->max() << std::endl;
			// std::cout << "Wavefield1 after min = " << _wavefield1->min() << std::endl;
			// std::cout << "Offsets, max val data after = " << dataRegDts->max() << std::endl;
			// std::cout << "Offsets, min val data after = " << dataRegDts->min() << std::endl;
			// std::cout << "Offsets, max val model after = " << model->max() << std::endl;
			// std::cout << "Offsets, min val model after = " << model->min() << std::endl;


		}
		else {
			std::cout << "**** ERROR [tomoExtGpu_3D]: Please specify the type of extension ****" << std::endl;
			assert(1==2);
		}

	} else {
		if (_fdParam_3D->_extension == "time") {
            tomoTauFsShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId);
		}
		else if (_fdParam_3D->_offsetType == "hxhy"){
            tomoHxHyFsShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId);
		}
		else {
			std::cout << "**** ERROR [tomoExtGpu_3D]: Please specify the type of extension ****" << std::endl;
			assert(1==2);
		}
	}

	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);
}

void tomoExtGpu_3D::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double2DReg> data) const {

	if (!add) model->scale(0.0);
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<double2DReg> dataRegDtsQc(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<double3DReg> modelTemp = model->clone(); // We need to create a temporary model for "add"
	modelTemp->scale(0.0);
	dataRegDtsQc->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Tomo extended adjoint */
	if (_fdParam_3D->_freeSurface != 1){

		if (_fdParam_3D->_extension == "time") {
			tomoTauShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId, dataRegDtsQc->getVals());
		}
		else if (_fdParam_3D->_extension == "offset") {
			tomoHxHyShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId, dataRegDtsQc->getVals());
		}
		else {
			std::cout << "**** ERROR [tomoExtGpu_3D]: Please specify the type of extension ****" << std::endl;
			assert(1==2);
		}
	}
	else {
		if (_fdParam_3D->_extension == "time") {
			tomoTauFsShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId, dataRegDtsQc->getVals());
		}
		else if (_fdParam_3D->_extension == "offset") {
			tomoHxHyFsShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId, dataRegDtsQc->getVals());
		}
		else {
			std::cout << "**** ERROR [tomoExtGpu_3D]: Please specify the type of extension ****" << std::endl;
			assert(1==2);
		}
	}

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);
}
