#include <float3DReg.h>
#include "dsoGpu_3D.h"
#include <vector>
#include <omp.h>

using namespace SEP;

dsoGpu_3D::dsoGpu_3D(int nz, int nx, int ny, int nExt1, int nExt2, int fat, float zeroShift){

    _nz = nz;
    _nx = nx;
    _ny = ny;
    _nExt1 = nExt1;
    _nExt2 = nExt2;
	if (_nExt1 % 2 == 0) {std::cout << "**** [dsoGpu_3D] ERROR: Length of extended axis #1 must be an uneven number ****" << std::endl; throw std::runtime_error(""); }
    if (_nExt2 % 2 == 0) {std::cout << "**** [dsoGpu_3D] ERROR: Length of extended axis #2 must be an uneven number ****" << std::endl; throw std::runtime_error(""); }
    _hExt1 = (_nExt1-1) / 2;
    _hExt2 = (_nExt2-1) / 2;
    _zeroShift = zeroShift;
    _fat = fat;
}

void dsoGpu_3D::forward(const bool add, const std::shared_ptr<float5DReg> model, std::shared_ptr<float5DReg> data) const {

	if (!add) data->scale(0.0);

    for (int iExt2=0; iExt2<_nExt2; iExt2++){
        float weight2 = 1.0*(std::abs(iExt2-_hExt2)); // Compute distance from physical axis #2
        for (int iExt1=0; iExt1<_nExt1; iExt1++){
            float weight1 = 1.0*(std::abs(iExt1-_hExt1)); // Compute distance from physical axis #1
            weight1 = std::sqrt(weight2*weight2+weight1*weight1) + std::abs(_zeroShift); // Compute weight for subsurface point
            #pragma omp parallel for collapse(3)
            for (int iy=_fat; iy<_ny-_fat; iy++){
                for (int ix=_fat; ix<_nx-_fat; ix++){
                    for (int iz=_fat; iz<_nz-_fat; iz++){
                        (*data->_mat)[iExt2][iExt1][iy][ix][iz] += weight1 * (*model->_mat)[iExt2][iExt1][iy][ix][iz];
                    }
                }
            }
        }
    }

    // #pragma omp parallel for collapse(5)
    // for (int iExt2=0; iExt2<_nExt2; iExt2++){
    //     for (int iExt1=0; iExt1<_nExt1; iExt1++){
    //         for (int iy=_fat; iy<_ny-_fat; iy++){
    //             for (int ix=_fat; ix<_nx-_fat; ix++){
    //                 for (int iz=_fat; iz<_nz-_fat; iz++){
    //                     float weight2 = 1.0*(std::abs(iExt2-_hExt2)); // Compute distance from physical axis #2
    //                     float weight1 = 1.0*(std::abs(iExt1-_hExt1)); // Compute distance from physical axis #1
    //                     weight1 = std::sqrt(weight2*weight2+weight1*weight1) + std::abs(_zeroShift); // Compute weight for subsurface point
    //                     (*data->_mat)[iExt2][iExt1][iy][ix][iz] += weight1 * (*model->_mat)[iExt2][iExt1][iy][ix][iz];
    //                 }
    //             }
    //         }
    //     }
    // }

}

void dsoGpu_3D::adjoint(const bool add, std::shared_ptr<float5DReg> model, const std::shared_ptr<float5DReg> data) const {

	if (!add) model->scale(0.0);

    for (int iExt2=0; iExt2<_nExt2; iExt2++){
        float weight2 = 1.0*(std::abs(iExt2-_hExt2)); // Compute distance from physical axis #2
        for (int iExt1=0; iExt1<_nExt1; iExt1++){
            float weight1 = 1.0*(std::abs(iExt1-_hExt1)); // Compute distance from physical axis #1
            weight1 = std::sqrt(weight2*weight2+weight1*weight1) + std::abs(_zeroShift); // Compute weight for subsurface point
            #pragma omp parallel for collapse(3)
            for (int iy=_fat; iy<_ny-_fat; iy++){
                for (int ix=_fat; ix<_nx-_fat; ix++){
                    for (int iz=_fat; iz<_nz-_fat; iz++){
                        (*model->_mat)[iExt2][iExt1][iy][ix][iz] += weight1 * (*data->_mat)[iExt2][iExt1][iy][ix][iz];
                    }
                }
            }
        }
    }
}
