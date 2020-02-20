#include <vector>
#include <ctime>
#include "freeSurfaceDebugOp.h"

freeSurfaceDebugOp::freeSurfaceDebugOp(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	// _fdParam_3D = std::make_shared<fdParam_3D>(vel, par);
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;
	_iGpuAlloc = iGpuAlloc; // Can remove that
	_vel = vel;
	// Initialize GPU
	// initNonlinearGpu_3D(_fdParam_3D->_dz, _fdParam_3D->_dx, _fdParam_3D->_dy, _fdParam_3D->_nz, _fdParam_3D->_nx, _fdParam_3D->_ny, _fdParam_3D->_nts, _fdParam_3D->_dts, _fdParam_3D->_sub, _fdParam_3D->_minPad, _fdParam_3D->_blockSize, _fdParam_3D->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);

	// for (int iy=0; iy<_vel->getHyper()->getAxis(3).n; iy++){
	// 	for (int ix=0; ix<_vel->getHyper()->getAxis(2).n; ix++){
	// 		for (int iz=0; iz<_vel->getHyper()->getAxis(1).n; iz++){
	// 			if ( (*_vel->_mat)[iy][ix][iz] > 0) {
	// 				(*_vel->_mat)[iy][ix][iz] = 1.0;
	// 			}
	// 		}
	// 	}
	// }
	std::shared_ptr<double3DReg> dimTemp;
	int nzModel = _vel->getHyper()->getAxis(1).n;
	int nxModel = _vel->getHyper()->getAxis(2).n;
	int nyModel = _vel->getHyper()->getAxis(3).n;
	std::shared_ptr<hypercube> dimTempHyper(new hypercube(nzModel, nxModel, nyModel));
	dimTemp = std::make_shared<double3DReg>(dimTempHyper);
	setDomainRange(dimTemp, dimTemp);

}

bool freeSurfaceDebugOp::checkParfileConsistency_3D(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double3DReg> data) const{

	// if (_fdParam_3D->checkParfileConsistencyTime_3D(data, 1, "Data file") != true) {return false;} // Check data time axis
	// if (_fdParam_3D->checkParfileConsistencyTime_3D(model,1, "Model file") != true) {return false;}; // Check model time axis

	return true;
}

void freeSurfaceDebugOp::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) data->scale(0.0);

	std::shared_ptr<double3DReg> dataTemp, modelTemp;
	std::shared_ptr<hypercube> hyperTemp(new hypercube(_vel->getHyper()->getAxis(1).n, _vel->getHyper()->getAxis(2).n, _vel->getHyper()->getAxis(3).n));
	modelTemp = std::make_shared<double3DReg>(hyperTemp);
	dataTemp = std::make_shared<double3DReg>(hyperTemp);
	modelTemp->scale(0.0);
	dataTemp->scale(0.0);

	for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
			for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
				(*modelTemp->_mat)[iy][ix][iz] = (*model->_mat)[iy][ix][iz];
			}
		}
	}

	double sum1, sum2;
	sum1=0;
	sum2=0;

	for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
			sum1 += (*modelTemp->_mat)[iy][ix][4];
		}
	}

	std::cout << "sum1 forward= " << sum1 << std::endl;

	for (int iy=0; iy<4; iy++){
		for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
			for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
				(*modelTemp->_mat)[iy][ix][iz] = 0.0;
				(*modelTemp->_mat)[iy+modelTemp->getHyper()->getAxis(3).n-4][ix][iz] = 0.0;
			}
		}
	}

	for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<4; ix++){
			for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
				(*modelTemp->_mat)[iy][ix][iz] = 0.0;
				(*modelTemp->_mat)[iy][ix+modelTemp->getHyper()->getAxis(2).n-4][iz] = 0.0;
			}
		}
	}

	for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
			for (int iz=0; iz<4; iz++){
				(*modelTemp->_mat)[iy][ix][iz] = 0.0;
				(*modelTemp->_mat)[iy][ix][iz+modelTemp->getHyper()->getAxis(1).n-4] = 0.0;
			}
		}
	}


	// std::cout << "model norm forward:" << model->norm(2) << std::endl;
	freeSurfaceDebugFwd(modelTemp->getVals(), dataTemp->getVals(), _iGpu, _iGpuId);

	// for (int iy=0; iy<4; iy++){
	// 	for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
	// 		for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
	// 			(*dataTemp->_mat)[iy][ix][iz] = 0.0;
	// 			(*dataTemp->_mat)[iy+modelTemp->getHyper()->getAxis(3).n-4][ix][iz] = 0.0;
	// 		}
	// 	}
	// }
	//
	// for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
	// 	for (int ix=0; ix<4; ix++){
	// 		for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
	// 			(*dataTemp->_mat)[iy][ix][iz] = 0.0;
	// 			(*dataTemp->_mat)[iy][ix+modelTemp->getHyper()->getAxis(2).n-4][iz] = 0.0;
	// 		}
	// 	}
	// }
	//
	// for (int iy=0; iy<dataTemp->getHyper()->getAxis(3).n; iy++){
	// 	for (int ix=0; ix<dataTemp->getHyper()->getAxis(2).n; ix++){
	// 		for (int iz=0; iz<4; iz++){
	// 			(*dataTemp->_mat)[iy][ix][iz] = 0.0;
	// 			(*dataTemp->_mat)[iy][ix][iz+modelTemp->getHyper()->getAxis(1).n-4] = 0.0;
	// 		}
	// 	}
	// }

	for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
			sum2 += (*dataTemp->_mat)[iy][ix][4];
		}
	}
	std::cout << "sum2 forward= " << sum2 << std::endl;

	for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
			for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
				(*data->_mat)[iy][ix][iz] += (*dataTemp->_mat)[iy][ix][iz];
			}
		}
	}

	// std::cout << "dataTemp norm forward:" << dataTemp->norm(2) << std::endl;
	// data->scaleAdd(dataTemp,1.,1.);
	// std::cout << "data norm forward:" << data->norm(2) << std::endl;
}

void freeSurfaceDebugOp::adjoint(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) model->scale(0.0);

	std::shared_ptr<double3DReg> dataTemp, modelTemp;
	std::shared_ptr<hypercube> hyperTemp(new hypercube(_vel->getHyper()->getAxis(1).n, _vel->getHyper()->getAxis(2).n, _vel->getHyper()->getAxis(3).n));
	modelTemp = std::make_shared<double3DReg>(hyperTemp);
	dataTemp = std::make_shared<double3DReg>(hyperTemp);
	modelTemp->scale(0.0);
	dataTemp->scale(0.0);

	for (int iy=4; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
			for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
				(*dataTemp->_mat)[iy][ix][iz] = (*data->_mat)[iy][ix][iz];
			}
		}
	}

	double sum1, sum2;
	sum1=0;
	sum2=0;

	for (int iy=0; iy<dataTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<dataTemp->getHyper()->getAxis(2).n; ix++){
			sum1 += (*dataTemp->_mat)[iy][ix][4];
		}
	}

	std::cout << "sum1 adjoint= " << sum1 << std::endl;
	for (int iy=0; iy<4; iy++){
		for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
			for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
				(*dataTemp->_mat)[iy][ix][iz] = 0.0;
				(*dataTemp->_mat)[iy+modelTemp->getHyper()->getAxis(3).n-4][ix][iz] = 0.0;
			}
		}
	}

	for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<4; ix++){
			for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
				(*dataTemp->_mat)[iy][ix][iz] = 0.0;
				(*dataTemp->_mat)[iy][ix+modelTemp->getHyper()->getAxis(2).n-4][iz] = 0.0;
			}
		}
	}

	for (int iy=0; iy<dataTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<dataTemp->getHyper()->getAxis(2).n; ix++){
			(*dataTemp->_mat)[iy][ix][4] = 0.0;
			for (int iz=0; iz<4; iz++){
				(*dataTemp->_mat)[iy][ix][iz] = 0.0;
				(*dataTemp->_mat)[iy][ix][iz+modelTemp->getHyper()->getAxis(1).n-4] = 0.0;
			}
		}
	}
	// for (int iy=0; iy<dataTemp->getHyper()->getAxis(3).n; iy++){
	// 	for (int ix=0; ix<dataTemp->getHyper()->getAxis(2).n; ix++){
	// 			(*dataTemp->_mat)[iy][ix][4] = 0.0;
	// 	}
	// }
	freeSurfaceDebugAdj(modelTemp->getVals(), dataTemp->getVals(), _iGpu, _iGpuId);

	// for (int iy=0; iy<4; iy++){
	// 	for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
	// 		for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
	// 			(*modelTemp->_mat)[iy][ix][iz] = 0.0;
	// 			(*modelTemp->_mat)[iy+modelTemp->getHyper()->getAxis(3).n-4][ix][iz] = 0.0;
	// 		}
	// 	}
	// }
	//
	// for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
	// 	for (int ix=0; ix<4; ix++){
	// 		for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
	// 			(*modelTemp->_mat)[iy][ix][iz] = 0.0;
	// 			(*modelTemp->_mat)[iy][ix+modelTemp->getHyper()->getAxis(2).n-4][iz] = 0.0;
	// 		}
	// 	}
	// }
	//
	// for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
	// 	for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
	// 		for (int iz=0; iz<4; iz++){
	// 			(*modelTemp->_mat)[iy][ix][iz] = 0.0;
	// 			(*modelTemp->_mat)[iy][ix][iz+modelTemp->getHyper()->getAxis(1).n-4] = 0.0;
	// 		}
	// 	}
	// }
	//
	for (int iy=0; iy<dataTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<dataTemp->getHyper()->getAxis(2).n; ix++){
			sum2 += (*modelTemp->_mat)[iy][ix][4];
		}
	}
	std::cout << "sum2 adjoint= " << sum2 << std::endl;

	for (int iy=0; iy<modelTemp->getHyper()->getAxis(3).n; iy++){
		for (int ix=0; ix<modelTemp->getHyper()->getAxis(2).n; ix++){
			for (int iz=0; iz<modelTemp->getHyper()->getAxis(1).n; iz++){
				(*model->_mat)[iy][ix][iz] += (*modelTemp->_mat)[iy][ix][iz];
			}
		}
	}

	// std::cout << "modelTemp norm adjoint:" << modelTemp->norm(2) << std::endl;
	// model->scaleAdd(modelTemp,1.,1.);
	// std::cout << "model norm adjoint :" << model->norm(2) << std::endl;
}
