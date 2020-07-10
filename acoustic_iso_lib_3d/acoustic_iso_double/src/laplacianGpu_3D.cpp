#include <vector>
#include <ctime>
#include "laplacianGpu_3D.h"

laplacianGpu_3D::laplacianGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){
	// std::cout << "Inside Laplacian constructor 1" << std::endl;
	_fdParam_3D = std::make_shared<fdParam_3D>(vel, par);
 	_timeInterp_3D = std::make_shared<interpTimeLinTbb_3D>(_fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_ots, _fdParam_3D->_sub);
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;

	// Initialize GPU
	initNonlinearGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);

}

void laplacianGpu_3D::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) data->scale(0.0);

	// std::cout << "Laplacian forward... " << std::endl;
    std::shared_ptr<double3DReg> dataTemp;
    dataTemp=data->clone();
	std::shared_ptr<double3DReg> modelTemp;
	modelTemp=model->clone();
	for (int iy=0; iy<4;iy++){
		for (int ix=0; ix<_fdParam_3D->_nx;ix++){
			for (int iz=0; iz<_fdParam_3D->_nz;iz++){
				(*modelTemp->_mat)[iy][ix][iz] = 0.0;
				(*modelTemp->_mat)[iy+_fdParam_3D->_ny-4][ix][iz] = 0.0;
			}
		}
	}
	for (int iy=0; iy<_fdParam_3D->_ny;iy++){
		for (int ix=0; ix<4;ix++){
			for (int iz=0; iz<_fdParam_3D->_nz;iz++){
				(*modelTemp->_mat)[iy][ix][iz] = 0.0;
				(*modelTemp->_mat)[iy][ix+_fdParam_3D->_nx-4][iz] = 0.0;
			}
		}
	}
	for (int iy=0; iy<_fdParam_3D->_ny;iy++){
		for (int ix=0; ix<_fdParam_3D->_nx;ix++){
			for (int iz=0; iz<4;iz++){
				(*modelTemp->_mat)[iy][ix][iz] = 0.0;
				(*modelTemp->_mat)[iy][ix][iz+_fdParam_3D->_nz-4] = 0.0;
			}
		}
	}

	laplacianFwd_3d(modelTemp->getVals(), dataTemp->getVals(), _iGpu, _iGpuId);
    data->scaleAdd(dataTemp, 1.0, 1.0);

}

void laplacianGpu_3D::adjoint(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) model->scale(0.0);

    std::shared_ptr<double3DReg> modelTemp;
    modelTemp=model->clone();

	std::shared_ptr<double3DReg> dataTemp;
	dataTemp=data->clone();
	for (int iy=0; iy<4;iy++){
		for (int ix=0; ix<_fdParam_3D->_nx;ix++){
			for (int iz=0; iz<_fdParam_3D->_nz;iz++){
				(*dataTemp->_mat)[iy][ix][iz] = 0.0;
				(*dataTemp->_mat)[iy+_fdParam_3D->_ny-4][ix][iz] = 0.0;
			}
		}
	}
	for (int iy=0; iy<_fdParam_3D->_ny;iy++){
		for (int ix=0; ix<4;ix++){
			for (int iz=0; iz<_fdParam_3D->_nz;iz++){
				(*dataTemp->_mat)[iy][ix][iz] = 0.0;
				(*dataTemp->_mat)[iy][ix+_fdParam_3D->_nx-4][iz] = 0.0;
			}
		}
	}
	for (int iy=0; iy<_fdParam_3D->_ny;iy++){
		for (int ix=0; ix<_fdParam_3D->_nx;ix++){
			for (int iz=0; iz<4;iz++){
				(*dataTemp->_mat)[iy][ix][iz] = 0.0;
				(*dataTemp->_mat)[iy][ix][iz+_fdParam_3D->_nz-4] = 0.0;
			}
		}
	}

	// std::cout << "Laplacian adj... " << std::endl;
	// std::cout << "Maxval data ADJ " << dataTemp->max() << std::endl;
	// std::cout << "Minval data ADJ " << dataTemp->min() << std::endl;
	// laplacianFwd_3d(dataTemp->getVals(), modelTemp->getVals(), _iGpu, _iGpuId);
	// laplacianAdj_3d(modelTemp->getVals(), dataTemp->getVals(), _iGpu, _iGpuId);
	// std::cout << "Maxval model ADJ" << modelTemp->max() << std::endl;
	// std::cout << "Minval model ADJ" << modelTemp->min() << std::endl;
    model->scaleAdd(modelTemp, 1.0, 1.0);

}
