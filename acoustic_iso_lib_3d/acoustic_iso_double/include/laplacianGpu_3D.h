#ifndef LAPLACIAN_GPU_3D_H
#define LAPLACIAN_GPU_3D_H 1

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

class laplacianGpu_3D : public seismicOperator_3D<SEP::double3DReg, SEP::double3DReg> {

	// private:

		// std::shared_ptr<double4DReg> _wavefield;

	public:

		/* Overloaded constructors */
		laplacianGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const;

		/* Destructor */
		~laplacianGpu_3D(){};

		// Virtual functions
		void setAllWavefields_3D(int wavefieldFlag){};
		bool checkParfileConsistency_3D(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double3DReg> data) const{};
};

#endif
