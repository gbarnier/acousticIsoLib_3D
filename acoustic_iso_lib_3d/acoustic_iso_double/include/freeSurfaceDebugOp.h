#ifndef FREE_SURFACE_OP_H
#define FREE_SURFACE_OP_H 1

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

using namespace SEP;

class freeSurfaceDebugOp : public seismicOperator_3D<SEP::double3DReg, SEP::double3DReg> {

	public:

		// int _nzVelSmall, _nxVelSmall, _nyVelSmall, _nzVelBig, _nxVelBig, _nyVelBig;

		/* Overloaded constructors */
		freeSurfaceDebugOp(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		/* Mutators */
		void setAllWavefields_3D(int wavefieldFlag){};

		/* QC */
		bool checkParfileConsistency_3D(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double3DReg> data) const;

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const;

		/* Destructor */
		~freeSurfaceDebugOp(){};
		std::shared_ptr<double3DReg> _vel;
		int _iGpu, _nGpu, iGpuId, _iGpuAlloc;

};

#endif
