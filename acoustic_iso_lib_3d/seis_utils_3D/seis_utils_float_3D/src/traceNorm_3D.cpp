#include <float3DReg.h>
#include "traceNorm_3D.h"
#include <math.h>

using namespace SEP;

//Constructor for trace normalization
traceNorm_3D::traceNorm_3D(std::shared_ptr<float3DReg> model, float epsilonTraceNorm){

	// Getting data-space sizes
	_epsilonTraceNorm = epsilonTraceNorm;
	_nShot = model->getHyper()->getAxis(3).n;
	_nRec = model->getHyper()->getAxis(2).n;
	_nts = model->getHyper()->getAxis(1).n;

}


void traceNorm_3D::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const{

	if (!add) data->zero();

	#pragma omp parallel for collapse(2)
	for (int iShot = 0; iShot < _nShot; iShot++){
		for (int iRec = 0; iRec < _nRec; iRec++){
			//Computing norm of trace
			float trace_norm = 0.0;
			for (int its = 0; its < _nts; its++){
				trace_norm += (*model->_mat)[iShot][iRec][its]*(*model->_mat)[iShot][iRec][its];
			}
			trace_norm = 1.0/(sqrt(trace_norm)+_epsilonTraceNorm);
			//Normalizating traces
			for (int its = 0; its < _nts; its++){
				(*data->_mat)[iShot][iRec][its] += (*model->_mat)[iShot][iRec][its]*trace_norm;
			}
		}
	}

}

traceNormJac_3D::traceNormJac_3D(std::shared_ptr<float3DReg> predDat, float epsilonTraceNorm){

	// Getting data-space sizes
	_epsilonTraceNorm = epsilonTraceNorm;
	_predDat = predDat; //Predicted data
	_nShot = predDat->getHyper()->getAxis(3).n;
	_nRec = predDat->getHyper()->getAxis(2).n;
	_nts = predDat->getHyper()->getAxis(1).n;

}

void traceNormJac_3D::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const{

	if (!add) data->zero();

	#pragma omp parallel for collapse(2)
	for (int iShot = 0; iShot < _nShot; iShot++){
		for (int iRec = 0; iRec < _nRec; iRec++){
			//Computing norm of trace
			float trace_norm = 0.0;
			float xCorr = 0.0;
			for (int its = 0; its < _nts; its++){
				trace_norm += (*_predDat->_mat)[iShot][iRec][its]*(*_predDat->_mat)[iShot][iRec][its];
				xCorr += (*_predDat->_mat)[iShot][iRec][its]*(*model->_mat)[iShot][iRec][its];
			}
			float trace_norm_inv = 1.0/sqrt(trace_norm);
			float trace_norm_eps =  1.0/(sqrt(trace_norm)+_epsilonTraceNorm);
			float trace_norm_cube =  trace_norm_inv * trace_norm_eps * trace_norm_eps;
			//Normalizating traces
			for (int its = 0; its < _nts; its++){
				(*data->_mat)[iShot][iRec][its] += (*model->_mat)[iShot][iRec][its]*trace_norm_eps-(*_predDat->_mat)[iShot][iRec][its]*xCorr*trace_norm_cube;
			}
		}
	}

}

void traceNormJac_3D::adjoint(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const{
	// Self-adjoint operator
	forward(add,data,model);

}
