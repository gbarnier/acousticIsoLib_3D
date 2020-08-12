#ifndef NL_PROP_SHOTS_GPU_3D_H
#define NL_PROP_SHOTS_GPU_3D_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "float2DReg.h"
#include "float3DReg.h"
#include "float4DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "operator.h"

using namespace SEP;

class nonlinearPropShotsGpu_3D : public Operator<SEP::float2DReg, SEP::float3DReg> {

	private:
		int _nShot, _nGpu, _info, _deviceNumberInfo, _iGpuAlloc, _ginsu;
		std::shared_ptr<SEP::float3DReg> _vel;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu_3D>> _sourcesVector, _receiversVector;
		std::vector<int> _gpuList;
		// std::shared_ptr<float3DReg> _dampVolumeShots;
		std::shared_ptr<fdParam_3D> _fdParamDampShots_3D;
		std::vector<std::shared_ptr<SEP::hypercube>> _velHyperVectorGinsu;
		std::shared_ptr<SEP::int1DReg> _xPadMinusVectorGinsu, _xPadPlusVectorGinsu;
		std::vector<int> _ixVectorGinsu, _iyVectorGinsu;

	public:

		/* Overloaded constructor */
		nonlinearPropShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector);

		/* Overloaded constructor Ginsu */
		nonlinearPropShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::vector<std::shared_ptr<SEP::hypercube>> velHyperVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadMinusVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadPlusVectorGinsu, std::vector<int> ixVectorGinsu, std::vector<int> iyVectorGinsu);

		/* Destructor */
		~nonlinearPropShotsGpu_3D(){};

		/* Create Gpu list */
		void createGpuIdList_3D();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float3DReg> data) const;

		/* Mutator */
		void setVel_3D(std::shared_ptr<SEP::float3DReg> vel){_vel = vel;}

		// Debugging
		std::shared_ptr<float3DReg> getDampVolumeShots_3D() {
			_fdParamDampShots_3D = std::make_shared<fdParam_3D>(_vel, _par);
			return _fdParamDampShots_3D->_dampVolume;
		}
};

#endif
