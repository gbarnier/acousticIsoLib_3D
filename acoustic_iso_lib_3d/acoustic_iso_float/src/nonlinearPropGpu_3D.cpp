#include <vector>
#include <ctime>
#include "nonlinearPropGpu_3D.h"

nonlinearPropGpu_3D::nonlinearPropGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParam_3D = std::make_shared<fdParam_3D>(vel, par);
 	_timeInterp_3D = std::make_shared<interpTimeLinTbb_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_ots, _fdParam_3D->_sub);
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;

	// Initialize GPU
	initNonlinearGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);
}

bool nonlinearPropGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float2DReg> data) const{

	if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam_3D->checkParfileConsistencyTime_3D(model,1, "Model file") != true) {return false;}; // Check model time axis

	return true;
}

void nonlinearPropGpu_3D::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const {

	if (!add) data->scale(0.0);

 	std::clock_t start;
    float duration;

	/* Allocation */
	std::shared_ptr<float2DReg> modelRegDts(new float2DReg(_fdParam_3D->_nts, _nSourcesReg));
	std::shared_ptr<float2DReg> modelRegDtw(new float2DReg(_fdParam_3D->_ntw, _nSourcesReg));
	std::shared_ptr<float2DReg> dataRegDtw(new float2DReg(_fdParam_3D->_ntw, _nReceiversReg));
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam_3D->_nts, _nReceiversReg));

	/* Interpolate model (seismic source) to regular grid */
	_sources->adjoint(false, modelRegDts, model);

	/* Scale model by dtw^2 * vel^2 * dSurface */
	scaleSeismicSource_3D(_sources, modelRegDts, _fdParam_3D);

	/* Interpolate model from coarse to fine time sampling */
	_timeInterp_3D->forward(false, modelRegDts, modelRegDtw);

	/* Propagate */
	if (_fdParam_3D->_freeSurface != 1){
		if (_fdParam_3D->_par->getInt("dampTest") == 0){
			propShotsFwdGpu_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg,_iGpu, _iGpuId);
		} else {
			std::cout << "Damping test" << std::endl;
			propShotsFwdGpu_3D_dampTest(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg,_iGpu, _iGpuId, _fdParam_3D->_dampVolume->getVals());
		}
	} else {
		// std::cout << "FWD with free surface" << std::endl;
		propShotsFwdFreeSurfaceGpu_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
	}

	/* Interpolate to irregular grid */
	_receivers->forward(true, dataRegDts, data);

}

void nonlinearPropGpu_3D::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<float2DReg> modelRegDtw(new float2DReg(_fdParam_3D->_ntw, _nSourcesReg));
	std::shared_ptr<float2DReg> modelRegDts(new float2DReg(_fdParam_3D->_nts, _nSourcesReg));

	float sum1, sum2, sum3;

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Propagate */
	if (_fdParam_3D->_freeSurface != 1){
		propShotsAdjGpu_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);
	} else{
		propShotsAdjFreeSurfaceGpu_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _iGpu, _iGpuId);

	}
	/* Interpolate to coarse time-sampling */
	_timeInterp_3D->adjoint(false, modelRegDts, modelRegDtw);

	/* Scale model by dtw^2 * vel^2 * dSurface */
	scaleSeismicSource_3D(_sources, modelRegDts, _fdParam_3D);

	/* Interpolate to irregular grid */
	_sources->forward(true, modelRegDts, model);

}