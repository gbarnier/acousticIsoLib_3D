#ifndef BORN_GPU_3D_H
#define BORN_GPU_3D_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "float2DReg.h"
#include "float3DReg.h"
#include "float4DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "seismicOperator_3D.h"
#include "interpTimeLinTbb_3D.h"
#include "secondTimeDerivative_3D.h"
#include "BornShotsGpuFunctions_3D.h"

using namespace SEP;

class BornGpu_3D : public seismicOperator_3D<SEP::float3DReg, SEP::float2DReg> {

	private:

		int _rtm;
		int _wavefieldSize;

	public:

		/* Overloaded constructors */
		BornGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		/* Mutator */
		void setBornGinsuGpu_3D(std::shared_ptr<SEP::hypercube> velHyperGinsu, int 	xPadMinusGinsu, int xPadPlusGinsu, int ixGinsu, int iyGinsu, int iGpu, int iGpuId);

		/* QC */
		bool checkParfileConsistency_3D(const std::shared_ptr<SEP::float3DReg> model, const std::shared_ptr<SEP::float2DReg> data) const;

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float2DReg> data) const;

		/* Destructor */
		~BornGpu_3D(){};

};

#endif
