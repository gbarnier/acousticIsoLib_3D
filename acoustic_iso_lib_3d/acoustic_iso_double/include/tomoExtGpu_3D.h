#ifndef TOMO_EXT_GPU_3D_H
#define TOMO_EXT_GPU_3D_H 1

#include <string>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "double2DReg.h"
#include "double3DReg.h"
#include "double5DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "seismicOperator_3D.h"
#include "interpTimeLinTbb_3D.h"
#include "secondTimeDerivative_3D.h"
#include "tomoExtShotsGpuFunctions_3D.h"

using namespace SEP;

class tomoExtGpu_3D : public seismicOperator_3D<SEP::double3DReg, SEP::double2DReg> {

	private:

		int _leg1, _leg2;
		std::shared_ptr<double4DReg> _wavefield1, _wavefield2;
		std::shared_ptr<double5DReg> _extReflectivity;

	public:

		/* Overloaded constructors */
		tomoExtGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::shared_ptr<double5DReg> extReflectivity, std::shared_ptr<SEP::double4DReg> wavefield1, std::shared_ptr<SEP::double4DReg> wavefield2, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		/* Mutators */
		void setExtReflectivity(std::shared_ptr<double5DReg> extReflectivity){ _extReflectivity=extReflectivity;}

		/* Quality control */
		bool checkParfileConsistency_3D(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double2DReg> data) const;

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double2DReg> data) const;

		/* Destructor */
		~tomoExtGpu_3D(){};

		/* Accessors */
		std::shared_ptr<double4DReg> getWavefield1_3D() { return _wavefield1; }
		std::shared_ptr<double4DReg> getWavefield2_3D() { return _wavefield2; }
		std::shared_ptr<double5DReg> getExtReflectivity() { return _extReflectivity; }


};

#endif
