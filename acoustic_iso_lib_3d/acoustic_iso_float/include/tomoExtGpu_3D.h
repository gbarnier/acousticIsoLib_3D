#ifndef TOMO_EXT_GPU_3D_H
#define TOMO_EXT_GPU_3D_H 1

#include <string>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "float2DReg.h"
#include "float3DReg.h"
#include "float5DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "seismicOperator_3D.h"
#include "interpTimeLinTbb_3D.h"
#include "secondTimeDerivative_3D.h"
#include "tomoExtShotsGpuFunctions_3D.h"

using namespace SEP;

class tomoExtGpu_3D : public seismicOperator_3D<SEP::float3DReg, SEP::float2DReg> {

	private:

		int _leg1, _leg2;
		int _wavefieldSize;
		std::shared_ptr<float5DReg> _extReflectivity;

	public:

		/* Overloaded constructors */
		tomoExtGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::shared_ptr<float5DReg> extReflectivity, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		/* Mutators */
		void setTomoExtGinsuGpu_3D(std::shared_ptr<SEP::hypercube> velHyperGinsu, int xPadMinusGinsu, int xPadPlusGinsu, int ixGinsu, int iyGinsu, int iGpu, int iGpuId);
		void setExtReflectivity_3D(std::shared_ptr<float5DReg> extReflectivity){ _extReflectivity=extReflectivity;}

		/* Quality control */
		bool checkParfileConsistency_3D(const std::shared_ptr<SEP::float3DReg> model, const std::shared_ptr<SEP::float2DReg> data) const;

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float2DReg> data) const;

		/* Accessor */
		std::shared_ptr<float5DReg> getExtReflectivity_3D(){ return _extReflectivity;}

		/* Destructor */
		~tomoExtGpu_3D(){};

};

#endif
