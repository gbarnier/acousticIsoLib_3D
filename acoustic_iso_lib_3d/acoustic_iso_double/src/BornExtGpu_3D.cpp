#include "BornExtGpu_3D.h"

BornExtGpu_3D::BornExtGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::shared_ptr<SEP::double4DReg> srcWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParam_3D = std::make_shared<fdParam_3D>(vel, par); // Fd parameter object
	_timeInterp_3D = std::make_shared<interpTimeLinTbb_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_ots, _fdParam_3D->_sub); // Time interpolation object
	_secTimeDer = std::make_shared<secondTimeDerivative_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts); // Second time derivative object
	_iGpu = iGpu; // Gpu number
	_nGpu = nGpu; // Number of requested GPUs
	_iGpuId = iGpuId;

	// Allocate the source wavefield on the RAM
	_srcWavefield = srcWavefield; // Point to wavefield
	unsigned long long int _wavefieldSize = _fdParam_3D->_zAxis.n * _fdParam_3D->_xAxis.n * _fdParam_3D->_yAxis.n;
	_wavefieldSize = _wavefieldSize * _fdParam_3D->_nts*sizeof(double) / (1024*1024*1024);

	// Initialize GPU
	initBornExtGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_nExt1, _fdParam_3D->_nExt2, _nGpu, _iGpuId, iGpuAlloc);

}

bool BornExtGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::double5DReg> model, const std::shared_ptr<SEP::double2DReg> data) const {
	if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam_3D->checkParfileConsistencyTime_3D(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam_3D->checkParfileConsistencySpace_3D(model, "Model file") != true) {return false;}; // Check model space axes
	return true;
}

void BornExtGpu_3D::forward(const bool add, const std::shared_ptr<double5DReg> model, std::shared_ptr<double2DReg> data) const {

	if (!add) data->scale(0.0);

	/* Allocation */
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));
	dataRegDts->scale(0.0);

	// Scale the model by 2.0/v^3
	std::shared_ptr<double5DReg> modelTemp;
	// modelTemp = model->clone();

	// #pragma omp parallel for
	// for (int iExt2=0; iExt2<_fdParam_3D->_nExt2; iExt2++){
	// 	for (int iExt1=0; iExt1<_fdParam_3D->_nExt1; iExt1++){
	// 		for (int iy=_fdParam_3D->_fat; iy<_fdParam_3D->_ny-_fdParam_3D->_fat; iy++){
	// 			for (int ix=_fdParam_3D->_fat; ix<_fdParam_3D->_nx-_fdParam_3D->_fat; ix++){
	// 				for (int iz=_fdParam_3D->_fat; iz<_fdParam_3D->_nz-_fdParam_3D->_fat; iz++){
	// 					(*modelTemp->_mat)[iExt2][iExt1][iy][ix][iz] *= 2.0 / ( (*_fdParam_3D->_vel->_mat)[iy][ix][iz] * (*_fdParam_3D->_vel->_mat)[iy][ix][iz] * (*_fdParam_3D->_vel->_mat)[iy][ix][iz] );
	// 				}
	// 			}
	// 		}
	// 	}
	// }
	// long long nModelExt = _fdParam_3D->_nz * _fdParam_3D->_nx * _fdParam_3D->_ny * _fdParam_3D->_nExt1 * _fdParam_3D->_nExt2;

	/* Launch Born extended forward */
	if (_fdParam_3D->_freeSurface != 1){
		if (_fdParam_3D->_extension == "time") {
				BornTauShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
		}
		if (_fdParam_3D->_extension == "offset") {
			if (_fdParam_3D->_offsetType == "hx"){
				BornHxShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			}
			else if (_fdParam_3D->_offsetType == "hxhy"){
				BornHxHyShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			}
			else {
				std::cout << "**** ERROR [BornExtGpu_3D]: Please specify the type of subsurface offset extension ****" << std::endl;
				assert(1==2);
			}
		}
	} else {
		if (_fdParam_3D->_extension == "time") {
				BornTauFreeSurfaceShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
		}
		if (_fdParam_3D->_extension == "offset") {
			if (_fdParam_3D->_offsetType == "hx"){
				BornHxFreeSurfaceShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			}
			else if (_fdParam_3D->_offsetType == "hxhy"){
				BornHxHyFreeSurfaceShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			}
			else {
				std::cout << "**** ERROR [BornExtGpu_3D]: Please specify the type of subsurface offset extension ****" << std::endl;
				assert(1==2);
			}
		}
	}

	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);

}
void BornExtGpu_3D::adjoint(const bool add, std::shared_ptr<double5DReg> model, const std::shared_ptr<double2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<double5DReg> modelTemp = model->clone();
	modelTemp->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Launch Born extended adjoint */
	if (_fdParam_3D->_freeSurface != 1){

		// Time-lags + no free surface
		if (_fdParam_3D->_extension == "time") {
			BornTauShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
		}

		// Horizontal subsurface offsets + no free surface
		if (_fdParam_3D->_extension == "offset") {
			if (_fdParam_3D->_offsetType == "hx"){
				BornHxShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			}
			else if (_fdParam_3D->_offsetType == "hxhy"){
				BornHxHyShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			}
			else {
				std::cout << "**** ERROR [BornExtGpu_3D]: Please specify the type of subsurface offset extension ****" << std::endl;
				assert(1==2);
			}
		}

	} else {

		// Time-lags + free surface
		if (_fdParam_3D->_extension == "time") {
			BornTauFreeSurfaceShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
		}

		// Horizontal subsurface offsets + free surface
		if (_fdParam_3D->_extension == "offset") {
			if (_fdParam_3D->_offsetType == "hx"){
				BornHxFreeSurfaceShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			}
			else if (_fdParam_3D->_offsetType == "hxhy"){
				BornHxHyFreeSurfaceShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			}
			else {
				std::cout << "**** ERROR [BornExtGpu_3D]: Please specify the type of subsurface offset extension ****" << std::endl;
				assert(1==2);
			}
		}
	}

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);

}
