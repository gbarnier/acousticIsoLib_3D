#include <vector>
#include <ctime>
#include "nonlinearPropGpu_3D.h"

nonlinearPropGpu_3D::nonlinearPropGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	_fdParam_3D = std::make_shared<fdParam_3D>(vel, par);
 	_timeInterp_3D = std::make_shared<interpTimeLinTbb_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_ots, _fdParam_3D->_sub);
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;
	_iGpuAlloc = iGpuAlloc; // Can remove that

	// Initialize GPU
	initNonlinearGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);

	// Create a Laplacian operator
	// std::cout << "Creating Laplacian" << std::endl;
	// std::shared_ptr<double3DReg> velLaplacian;
	// velLaplacian = _fdParam_3D->_vel->clone();
	// // Create vel with ones
	// for (int iy=4; iy<_fdParam_3D->_ny-4;iy++){
	// 	for (int ix=4; ix<_fdParam_3D->_nx-4;ix++){
	// 		for (int iz=4; iz<_fdParam_3D->_nz-4;iz++){
	// 			(*velLaplacian->_mat)[iy][ix][iz] = 1.0;
	// 		}
	// 	}
	// }
	// // for (int iy=0; iy<_fdParam_3D->_ny;iy++){
	// // 	for (int ix=0; ix<_fdParam_3D->_nx;ix++){
	// // 		for (int iz=0; iz<_fdParam_3D->_nz;iz++){
	// // 			(*velLaplacian->_mat)[iy][ix][iz] = 1.0;
	// // 		}
	// // 	}
	// // }

	// std::cout << "Maxval velLaplacian" << velLaplacian->max() << std::endl;
	// std::cout << "Minval velLaplacian" << velLaplacian->min() << std::endl;
	//
	// _laplacianObj = std::make_shared<laplacianGpu_3D>(velLaplacian, _fdParam_3D->_par, _nGpu, _iGpu, _iGpuId, _iGpuAlloc);
	// // std::shared_ptr<laplacianGpu_3D> laplacianObj(new laplacianGpu_3D(_fdParam_3D->_vel, _fdParam_3D->_par, _nGpu, _iGpu, _iGpuId, _iGpuAlloc));
	// _laplacianObj->setDomainRange(_fdParam_3D->_vel, _fdParam_3D->_vel);
	// std::cout << "Done creating Laplacian" << std::endl;

}

bool nonlinearPropGpu_3D::checkParfileConsistency_3D(const std::shared_ptr<SEP::double2DReg> model, const std::shared_ptr<SEP::double2DReg> data) const{

	if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam_3D->checkParfileConsistencyTime_3D(model,1, "Model file") != true) {return false;}; // Check model time axis

	return true;
}

void nonlinearPropGpu_3D::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const {

	if (!add) data->scale(0.0);

 	std::clock_t start;
    double duration;

	/* Allocation */
	std::shared_ptr<double2DReg> modelRegDts(new double2DReg(_fdParam_3D->_nts, _nSourcesReg));
	std::shared_ptr<double2DReg> modelRegDtw(new double2DReg(_fdParam_3D->_ntw, _nSourcesReg));
	std::shared_ptr<double2DReg> dataRegDtw(new double2DReg(_fdParam_3D->_ntw, _nReceiversReg));
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));

	// Perform dot product test

	// std::cout << "Doing dot product test of sources" << std::endl;
	_sources->setDomainRange(modelRegDts, model);
	// _sources->dotTest(true);
	// std::cout << "Finished dot product test of sources" << std::endl;

	/* Interpolate model (seismic source) to regular grid */
	_sources->adjoint(false, modelRegDts, model);

	/* Scale model by dtw^2 * vel^2 * dSurface */
	// scaleSeismicSource_3D(_sources, modelRegDts, _fdParam_3D); // Try without this one

	/* Interpolate to fine time-sampling */
	// std::cout << "Doing dot product test of time interpolation" << std::endl;
	_timeInterp_3D->setDomainRange(modelRegDts, modelRegDtw);
	// _timeInterp_3D->dotTest(true);
	// std::cout << "Doing dot product test of time interpolation" << std::endl;

	// std::cout << "Doing dot product test of Laplacian" << std::endl;
	// _laplacianObj->dotTest(true);
	// _laplacianObj->dotTest(true);
	// _laplacianObj->dotTest(true);
	// std::cout << "Finished doing dot product test of Laplacian" << std::endl;
	// exit(0);

	_timeInterp_3D->forward(false, modelRegDts, modelRegDtw);

	/* Propagate */
	propShotsFwdGpu_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);

	/* Interpolate to irregular grid */
	_receivers->forward(true, dataRegDts, data);
	// _receivers->setDomainRange(modelRegDts, model);
	// _receivers->dotTest(true);


}

void nonlinearPropGpu_3D::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam_3D->_nts, _nReceiversReg));
	std::shared_ptr<double2DReg> modelRegDtw(new double2DReg(_fdParam_3D->_ntw, _nSourcesReg));
	std::shared_ptr<double2DReg> modelRegDts(new double2DReg(_fdParam_3D->_nts, _nSourcesReg));

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Propagate */
	propShotsAdjGpu_3D(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);

	/* Interpolate to coarse time-sampling */
	_timeInterp_3D->adjoint(false, modelRegDts, modelRegDtw);

	/* Scale model by dtw^2 * vel^2 * dSurface */
	// scaleSeismicSource_3D(_sources, modelRegDts, _fdParam_3D);

	/* Interpolate to irregular grid */
	_sources->forward(true, modelRegDts, model);

}
