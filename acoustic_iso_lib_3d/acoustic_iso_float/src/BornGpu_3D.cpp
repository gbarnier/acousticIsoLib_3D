#include "BornGpu_3D.h"

BornGpu_3D::BornGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParam_3D = std::make_shared<fdParam_3D>(vel, par, "Born"); // Fd parameter object
	_timeInterp_3D = std::make_shared<interpTimeLinTbb_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_ots, _fdParam_3D->_sub); // Time interpolation object
	_secTimeDer = std::make_shared<secondTimeDerivative_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts); // Second time derivative object
	_iGpu = iGpu; // Gpu number
	_nGpu = nGpu; // Number of requested GPUs
	_iGpuId = iGpuId;
	_ginsu = _fdParam_3D->_par->getInt("ginsu");

	// Initialize GPU
	if (_ginsu == 0){
		initBornGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);
	} else {
		initBornGinsuGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);
	}
}

void BornGpu_3D::setBornGinsuGpu_3D(std::shared_ptr<SEP::hypercube> velHyperGinsu, int xPadMinusGinsu, int xPadPlusGinsu, int ixGinsu, int iyGinsu, int iGpu, int iGpuId){

	// Update Ginsu parameters from fdParam
	_fdParam_3D->setFdParamGinsu_3D(velHyperGinsu, xPadMinusGinsu, xPadPlusGinsu, ixGinsu, iyGinsu);
	_iGpu = iGpu;
	_iGpuId = iGpuId;

	// Initialize GPU
	allocateSetBornGinsuGpu_3D(_fdParam_3D->_nzGinsu, _fdParam_3D->_nxGinsu, _fdParam_3D->_nyGinsu, _fdParam_3D->_minPadGinsu, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _fdParam_3D->_vel2Dtw2Ginsu, _fdParam_3D->_reflectivityScaleGinsu, _iGpu, _iGpuId);
}

bool BornGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::float3DReg> model, const std::shared_ptr<SEP::float2DReg> data) const {
	if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam_3D->checkParfileConsistencyTime_3D(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam_3D->checkParfileConsistencySpace_3D(model, "Model file") != true) {return false;}; // Check model space axes
	return true;
}

void BornGpu_3D::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float2DReg> data) const {

	if (!add) data->scale(0.0);

	/* Allocation */
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam_3D->_nts, _nReceiversReg));
	dataRegDts->scale(0.0);

	/* Launch Born forward */
	if (_fdParam_3D->_freeSurface != 1){
		if (_ginsu == 0){
			BornShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);

		} else {

			BornShotsFwdGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
		}
	} else {
		if (_ginsu == 0){
			BornShotsFwdFreeSurfaceGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
		} else {
			BornShotsFwdFreeSurfaceGinsuGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
		}
	}

	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);

}

void BornGpu_3D::adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<float3DReg> modelTemp = model->clone();
	modelTemp->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	if (_fdParam_3D->_freeSurface != 1){
		if (_ginsu == 0){
			BornShotsAdjGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
		} else {
			BornShotsAdjGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
		}
	} else {
		if (_ginsu == 0){
			// std::cout << "Inside born adjoint [before]" << std::endl;
			// std::cout << "dataRegDts min before = " << dataRegDts->min() << std::endl;
			// std::cout << "dataRegDts max before = " << dataRegDts->max() << std::endl;
			BornShotsAdjFreeSurfaceGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
			// std::cout << "modelTemp min after = " << modelTemp->min() << std::endl;
			// std::cout << "modelTemp max after = " << modelTemp->max() << std::endl;
			// std::cout << "Inside born adjoint [after]" << std::endl;
		} else {
			BornShotsAdjFreeSurfaceGinsuGpu_3D(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
		}
	}

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);

}
