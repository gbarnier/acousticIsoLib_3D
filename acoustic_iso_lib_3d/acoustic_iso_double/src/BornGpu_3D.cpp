#include "BornGpu_3D.h"

BornGpu_3D::BornGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::shared_ptr<SEP::double4DReg> srcWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParam_3D = std::make_shared<fdParam_3D>(vel, par); // Fd parameter object
	_timeInterp_3D = std::make_shared<interpTimeLinTbb_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_ots, _fdParam_3D->_sub); // Time interpolation object
	_secTimeDer = std::make_shared<secondTimeDerivative_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts); // Second time derivative object
	_iGpu = iGpu; // Gpu number
	_nGpu = nGpu; // Number of requested GPUs
	_iGpuId = iGpuId;
	_ginsu = _fdParam_3D->_par->getInt("ginsu");

	// Allocate the source wavefield on the RAM
	_srcWavefield = srcWavefield; // Point to wavefield
	unsigned long long int _wavefieldSize = _fdParam_3D->_zAxis.n * _fdParam_3D->_xAxis.n * _fdParam_3D->_yAxis.n;
	_wavefieldSize = _wavefieldSize * _fdParam_3D->_nts*sizeof(double) / (1024*1024*1024);

	// Initialize GPU
	if (_ginsu == 0){
		initBornGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);
	} else {
		initBornGinsuGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);
	}
}

void BornGpu_3D::setBornGinsuGpu_3D(std::shared_ptr<SEP::hypercube> velHyperGinsu, int xPadMinusGinsu, int xPadPlusGinsu, int iGpu, int iGpuId){

	// Update Ginsu parameters from fdParam
	_fdParam_3D->setFdParamGinsu_3D(velHyperGinsu, xPadMinusGinsu, xPadPlusGinsu);
	_iGpu = iGpu;
	_iGpuId = iGpuId;

	// Initialize GPU
	allocateSetBornGinsuGpu_3D(_fdParam_3D->_nzGinsu, _fdParam_3D->_nxGinsu, _fdParam_3D->_nyGinsu, _fdParam_3D->_minPadGinsu, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_vel2Dtw2Ginsu, _fdParam_3D->_reflectivityScaleGinsu, _iGpu, _iGpuId);
}

bool BornGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double2DReg> data) const {
	if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam_3D->checkParfileConsistencyTime_3D(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam_3D->checkParfileConsistencySpace_3D(model, "Model file") != true) {return false;}; // Check model space axes
	return true;
}

void BornGpu_3D::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double2DReg> data) const {

	if (!add) data->scale(0.0);

	/* Allocation */
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));
	dataRegDts->scale(0.0);

	// std::cout << "Ginsu = " << _ginsu << std::endl;

	/* Launch Born forward */
	if (_fdParam_3D->_freeSurface != 1){
		if (_ginsu == 0){
			// std::cout << "Inside normal = " << std::endl;
			BornShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			// BornShotsFwdGpu_3D_Threads(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId); //TESTING thread for pin memory copy
		} else {
			// std::cout << "Inside Ginsu = " << std::endl;
			// std::cout << "max model Born before = " << model->max() << std::endl;
			// std::cout << "min model Born before = " << model->min() << std::endl;
			BornShotsFwdGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			// std::cout << "max data Born after = " << dataRegDts->max() << std::endl;
			// std::cout << "min data Born after = " << dataRegDts->min() << std::endl;
		}
	} else {
		if (_ginsu == 0){
			BornShotsFwdFreeSurfaceGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
		} else {
			BornShotsFwdFreeSurfaceGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
		}
	}

	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);

}

void BornGpu_3D::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<double3DReg> modelTemp = model->clone();
	modelTemp->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	if (_fdParam_3D->_freeSurface != 1){
		if (_ginsu ==0){
			BornShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
		} else {
			// std::cout << "Inside Ginsu = " << std::endl;
			// std::cout << "max data Born before = " << data->max() << std::endl;
			// std::cout << "min data Born before = " << data->min() << std::endl;
			BornShotsAdjGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
			// std::cout << "max model Born after = " << modelTemp->max() << std::endl;
			// std::cout << "min model Born after = " << modelTemp->min() << std::endl;
		}
	} else {
		if (_ginsu ==0){
			BornShotsAdjFreeSurfaceGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
		} else {
			BornShotsAdjFreeSurfaceGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);
		}
	}

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);

}
