#ifndef TRACE_NORM_3D_H
#define TRACE_NORM_3D_H 1

#include "float3DReg.h"
#include "operator.h"
#include <string>

using namespace SEP;

class traceNorm_3D : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:

		float _epsilonTraceNorm;
		int _nShot, _nRec, _nts;

	public:

		// Constructor for offset tapering only
		traceNorm_3D(std::shared_ptr<float3DReg> model, float epsilonTraceNorm);
		/* Destructor */
		~traceNorm_3D(){};

  	/* Forward / Adjoint */
  	virtual void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
 		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const{
				// Non-linear operator!
				throw std::runtime_error("Error! Trace Normalization is a non-linear operator!");
		};


};

class traceNormJac_3D : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:

		float _epsilonTraceNorm;
		int _nShot, _nRec, _nts;
		std::shared_ptr<float3DReg> _predDat;


	public:

		// Constructor for offset tapering only
		traceNormJac_3D(std::shared_ptr<float3DReg> predDat, float epsilonTraceNorm);
		/* Destructor */
		~traceNormJac_3D(){};

  	/* Forward / Adjoint */
  	virtual void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
 		virtual void adjoint(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;

		void setData(std::shared_ptr<float3DReg> predDat){_predDat = predDat;};


};

#endif
