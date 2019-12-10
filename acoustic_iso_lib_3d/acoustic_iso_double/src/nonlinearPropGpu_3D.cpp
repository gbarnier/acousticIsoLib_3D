#include <vector>
#include <ctime>
#include "nonlinearPropGpu_3D.h"

nonlinearPropGpu_3D::nonlinearPropGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParam = std::make_shared<fdParam_3D>(vel, par);
 	_timeInterp = std::make_shared<interpTimeLinTbb_3D>(_fdParam->_nts, _fdParam->_dts, _fdParam->_ots, _fdParam->_sub);
	setAllWavefields_3D(par->getInt("saveWavefield", 0));
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;

	// Initialize GPU
	initNonlinearGpu_3D(_fdParam->_dz, _fdParam->_dx, _fdParam->_dy, _fdParam->_nz, _fdParam->_nx, _fdParam->_ny, _fdParam->_nts, _fdParam->_dts, _fdParam->_sub, _fdParam->_minPad, _fdParam->_blockSize, _fdParam->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);
}

void nonlinearPropGpu_3D::setAllWavefields_3D(int wavefieldFlag){
	_wavefield = setWavefield_3D(wavefieldFlag);
}

bool nonlinearPropGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::double2DReg> model, const std::shared_ptr<SEP::double2DReg> data) const{

	if (_fdParam->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam->checkParfileConsistencyTime_3D(model,1, "Model file") != true) {return false;}; // Check model time axis

	return true;
}

void nonlinearPropGpu_3D::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const {

	if (!add) data->scale(0.0);

 	std::clock_t start;
    double duration;

	/* Allocation */
	std::shared_ptr<double2DReg> modelRegDts(new double2DReg(_fdParam->_nts, _nSourcesReg));
	std::shared_ptr<double2DReg> modelRegDtw(new double2DReg(_fdParam->_ntw, _nSourcesReg));
	std::shared_ptr<double2DReg> dataRegDtw(new double2DReg(_fdParam->_ntw, _nReceiversReg));
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam->_nts, _nReceiversReg));

	/* Interpolate model (seismic source) to regular grid */
	_sources->adjoint(false, modelRegDts, model);

	/* Scale model by dtw^2 * vel^2 * dSurface */
	scaleSeismicSource_3D(_sources, modelRegDts, _fdParam);

	/* Interpolate to fine time-sampling */
	_timeInterp->forward(false, modelRegDts, modelRegDtw);

	/* Propagate */
	if (_saveWavefield == 0) {
		propShotsFwdGpu_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);
    } else {
		propShotsFwdGpuWavefield_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);
	}

	/* Interpolate to irregular grid */
	_receivers->forward(true, dataRegDts, data);

}

void nonlinearPropGpu_3D::adjoint(const bool add, std::shared_ptr<double1DReg> model, const std::shared_ptr<double2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam->_nts, _nReceiversReg));
	std::shared_ptr<double2DReg> modelRegDtw(new double2DReg(_fdParam->_ntw, _nSourcesReg));
	std::shared_ptr<double2DReg> modelRegDts(new double2DReg(_fdParam->_nts, _nSourcesReg));

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Propagate */
	if (_saveWavefield == 0) {
		propShotsAdjGpu_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);
	} else {
		propShotsAdjGpuWavefield_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);
	}

	/* Scale adjoint wavefield */
	if (_saveWavefield == 1){
		#pragma omp parallel for collapse(4)
		for (int its=0; its<_fdParam->_nts; its++){
			for (int iy=0; iy<_fdParam->_ny; iy++){
                for (int ix=0; ix<_fdParam->_nx; ix++){
                    for (int iz=0; iz<_fdParam->_nz; iz++){
                        (*_wavefield->_mat)[its][iy][ix][iz] *= _fdParam->_dtw*_fdParam->_dtw*(*_fdParam->_vel->_mat)[iy][ix][iz]*(*_fdParam->_vel->_mat)[iy][ix][iz];
                    }
                }
            }
        }
    }

	/* Interpolate to coarse time-sampling */
	_timeInterp->adjoint(false, modelRegDts, modelRegDtw);

	/* Scale model by dtw^2 * vel^2 * dSurface */
	scaleSeismicSource_3D(_sources, modelRegDts, _fdParam);

	/* Interpolate to irregular grid */
	_sources->forward(true, modelRegDts, model);

}
