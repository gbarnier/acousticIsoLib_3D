#ifndef NL_PROP_GPU_3D_H
#define NL_PROP_GPU_3D_H 1

#include <string>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "double2DReg.h"
#include "double3DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "seismicOperator_3D.h"
#include "interpTimeLinTbb_3D.h"
#include "nonlinearShotsGpuFunctions_3D.h"
#include "freeSurfaceDebugOp.h"

using namespace SEP;

class nonlinearPropGpu_3D : public seismicOperator_3D<SEP::double2DReg, SEP::double2DReg> {

	public:

		/* Overloaded constructors */
		nonlinearPropGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		/* Mutators */
		void setAllWavefields_3D(int wavefieldFlag);

		/* QC */
		bool checkParfileConsistency_3D(const std::shared_ptr<SEP::double2DReg> model, const std::shared_ptr<SEP::double2DReg> data) const;

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const;

		/* Destructor */
		~nonlinearPropGpu_3D(){};

		/* Variable */
		std::shared_ptr<double2DReg> _dataDtw;

		/* Accessor */
		// int _iGpuAlloc;
		// std::shared_ptr<freeSurfaceDebugOp> _freeSurfaceDebugOpObj;

};

#endif
