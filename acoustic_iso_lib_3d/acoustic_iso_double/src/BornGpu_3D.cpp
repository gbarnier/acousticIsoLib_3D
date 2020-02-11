#include "BornGpu_3D.h"

BornGpu_3D::BornGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::shared_ptr<SEP::double4DReg> srcWavefield, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

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
	std::cout << "Wavefield size = " << _wavefieldSize << " [GB]" << std::endl;
	std::cout << "nts = " << _srcWavefield->getHyper()->getAxis(4).n << std::endl;
	std::cout << "ny = " << _srcWavefield->getHyper()->getAxis(3).n << std::endl;
	std::cout << "nx = " << _srcWavefield->getHyper()->getAxis(2).n << std::endl;
	std::cout << "nz = " << _srcWavefield->getHyper()->getAxis(1).n << std::endl;

	// Initialize GPU
	initBornGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);
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

	std::cout << "Min model before: " << model->min() << std::endl;
	std::cout << "Max model before: " << model->max() << std::endl;

	/* Launch Born forward */
	BornShotsFwdGpu_3D(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _iGpu, _iGpuId);

	std::cout << "Min data after: " << dataRegDts->min() << std::endl;
	std::cout << "Max data after: " << dataRegDts->max() << std::endl;

	// for (int iy = 0; iy < _fdParam_3D->_ny; iy++){
    //     for (int ix = 0; ix < _fdParam_3D->_nx; ix++){
    //     	for (int iz = 0; iz < _fdParam_3D->_nz; iz++){
    //             (*model->_mat)[iy][ix][iz] = 2.0;
    //         }
    //     }
    // }

	// std::shared_ptr<double3DReg> data_zLoop, data_yLoop;
	// data_zLoop = model->clone();
	// data_yLoop = model->clone();
	//
	// std::cout << "Min model before z-loop=" << model->min() << std::endl;
	// std::cout << "Max model before z-loop=" << model->max() << std::endl;
	// imagingFwd_yLoop(model->getVals(), data_zLoop->getVals(), _iGpu, _iGpuId);
	// std::cout << "Min data_zLoop=" << data_zLoop->min() << std::endl;
	// std::cout << "Max data_zLoop=" << data_zLoop->max() << std::endl;
	//
	// std::cout << "Min model before y-loop=" << model->min() << std::endl;
	// std::cout << "Max model before y-loop=" << model->max() << std::endl;
	// imagingFwd_zLoop(model->getVals(), data_yLoop->getVals(), _iGpu, _iGpuId);
	// std::cout << "Min data_yLoop=" << data_yLoop->min() << std::endl;
	// std::cout << "Max data_yLoop=" << data_yLoop->max() << std::endl;
	//
	// for (int iy = 0; iy < _fdParam_3D->_ny; iy++){
    //     for (int ix = 0; ix < _fdParam_3D->_nx; ix++){
    //     	for (int iz = 0; iz < _fdParam_3D->_nz; iz++){
    //             (*data_yLoop->_mat)[iy][ix][iz] = (*data_yLoop->_mat)[iy][ix][iz] - (*data_zLoop->_mat)[iy][ix][iz];
    //         }
    //     }
    // }
	// std::cout << "Minval difference =" << data_yLoop->min() << std::endl;
	// std::cout << "Maxval difference =" << data_yLoop->max() << std::endl;

	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);

	std::cout << "Min data after 2: " << dataRegDts->min() << std::endl;
	std::cout << "Max data after 2: " << dataRegDts->max() << std::endl;

}
void BornGpu_3D::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<double3DReg> modelTemp = model->clone();
	modelTemp->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Launch Born adjoint */
	// BornShotsAdjGpu(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu, _iGpuId);

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);

}
