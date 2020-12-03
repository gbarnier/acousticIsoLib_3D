#ifndef BORN_EXT_GPU_3D_H
#define BORN_EXT_GPU_3D_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "double2DReg.h"
#include "double3DReg.h"
#include "double4DReg.h"
#include "double5DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "seismicOperator_3D.h"
#include "interpTimeLinTbb_3D.h"
#include "secondTimeDerivative_3D.h"
#include "BornExtShotsGpuFunctions_3D.h"

using namespace SEP;

class BornExtGpu_3D : public seismicOperator_3D<SEP::double5DReg, SEP::double2DReg> {

	private:

		int _slowSquare;
		int _wavefieldSize;

	public:

		/* Overloaded constructors */
		BornExtGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		/* Mutator */
		void setBornExtGinsuGpu_3D(std::shared_ptr<SEP::hypercube> velHyperGinsu, int xPadMinusGinsu, int xPadPlusGinsu, int ixGinsu, int iyGinsu, int iGpu, int iGpuId);

		/* QC */
		bool checkParfileConsistency_3D(const std::shared_ptr<SEP::double5DReg> model, const std::shared_ptr<SEP::double2DReg> data) const;

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<double5DReg> model, std::shared_ptr<double2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double5DReg> model, const std::shared_ptr<double2DReg> data) const;

		/* Destructor */
		~BornExtGpu_3D(){};

};

#endif
