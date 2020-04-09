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

	// Allocate the source wavefield on the RAM
	_wavefield1 = wavefield1; // Point to wavefield
	_wavefield2 = wavefield2; // Point to wavefield
	unsigned long long int _wavefieldSize = _fdParam_3D->_zAxis.n * _fdParam_3D->_xAxis.n * _fdParam_3D->_yAxis.n;
	_wavefieldSize = _wavefieldSize * _fdParam_3D->_nts*sizeof(double) / (1024*1024*1024);

	// Initialize GPU
	initTomoExtGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx,  _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_nExt1, _fdParam_3D->_nExt2, _leg1, _leg2, _nGpu, _iGpuId, iGpuAlloc);
}

bool tomoExtGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double2DReg> data) const {
	if (_fdParam_3D->checkParfileConsistencyTime_3D(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam_3D->checkParfileConsistencySpace_3D(model, "Model file") != true) {return false;}; // Check model space axes
	if (_fdParam_3D->checkParfileConsistencySpace_3D(_extReflectivity, "Extended reflectivity file") != true) {return false;}; // Check extended reflectivity axes
	return true;
}

void tomoExtGpu_3D::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double2DReg> data) const {

	if (!add) data->scale(0.0);
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));

	/* Tomo extended forward */
	if (_fdParam_3D->_freeSurface != 1){
		if (_fdParam_3D->_extension == "time") {
            tomoTauShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId);
		}
		if (_fdParam_3D->_extension == "offset") {
            tomoHxHyShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId);

			// for (int iRec=0; iRec<_nReceiversReg; iRec++){
			// 	for (int its=0; its<_fdParam_3D->_nts; its++){
			// 		sum += (*dataRegDts->_mat)[iRec][its]*(*dataRegDts->_mat)[iRec][its];
			// 	}
			// }
		}
		// }
	} else {
		// if (_fdParam_3D->_extension == "time") {
        //     tomoTauFreeSurfaceShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		// }
		// if (_fdParam_3D->_extension == "offset") {
		// 	if (_fdParam_3D->_offsetType == "hx"){
	    //         tomoHxFreeSurfaceShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		// 	}
		// 	if (_fdParam_3D->_offsetType == "hxhy"){
	    //         tomoHxHyFreeSurfaceShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		// 	}
		// }
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
		if (_fdParam_3D->_extension == "offset") {
			tomoHxHyShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _extReflectivity->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield1->getVals(), _wavefield2->getVals(), _iGpu, _iGpuId, dataRegDtsQc->getVals());
		}
	}
	else {
		// if (_fdParam_3D->_extension == "time") {
        //     tomoTauFreeSurfaceShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		// }
		// if (_fdParam_3D->_extension == "offset") {
		// 	if (_fdParam_3D->_offsetType == "hx"){
	    //         tomoHxFreeSurfaceShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		// 	}
		// 	if (_fdParam_3D->_offsetType == "hxhy"){
	    //         tomoHxHyFreeSurfaceShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		// 	}
		// }
	}

	// double sum;
	// sum=0;
	// std::cout << "Max val data in Adj" << dataRegDtsQc->max() << std::endl;
	// std::cout << "Min val data in Adj" << dataRegDtsQc->min() << std::endl;
	// for (int iRec=0; iRec<_nReceiversReg; iRec++){
	// 	for (int its=0; its<_fdParam_3D->_nts; its++){
	// 		sum += (*dataRegDtsQc->_mat)[iRec][its]*(*dataRegDtsQc->_mat)[iRec][its];
	// 	}
	// }
	// std::cout << "Sum in adj = " << sum << std::endl;

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);
}
