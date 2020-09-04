#ifndef BORN_EXT_SHOTS_GPU_3D_H
#define BORN_EXT_SHOTS_GPU_3D_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "float2DReg.h"
#include "float3DReg.h"
#include "float4DReg.h"
#include "float5DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "operator.h"
#include "BornExtShotsGpuFunctions_3D.h"


using namespace SEP;

class BornExtShotsGpu_3D : public Operator<SEP::float5DReg, SEP::float3DReg> {

	private:
		int _nShot, _nGpu, _iGpuAlloc, _ginsu;
		int _info, _deviceNumberInfo;
		std::shared_ptr<SEP::float3DReg> _vel;
		std::shared_ptr<SEP::float2DReg> _sourcesSignals;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu_3D>> _sourcesVector, _receiversVector;
		std::vector<int> _gpuList;
		std::shared_ptr <hypercube> _srcWavefieldHyper;
		std::vector<std::shared_ptr<SEP::hypercube>> _velHyperVectorGinsu;
		std::shared_ptr<SEP::int1DReg> _xPadMinusVectorGinsu, _xPadPlusVectorGinsu;
		std::vector<int> _ixVectorGinsu, _iyVectorGinsu;

	public:

		/* Overloaded constructors */
		BornExtShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::float2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector);

		BornExtShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::float2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::vector<std::shared_ptr<SEP::hypercube>> velHyperVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadMinusVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadPlusVectorGinsu, int nxMaxGinsu, int nyMaxGinu, std::vector<int> ixVectorGinsu, std::vector<int> iyVectorGinsu);;

		/* Destructor */
		~BornExtShotsGpu_3D(){};

		/* Create Gpu list */
		void createGpuIdList_3D();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<float5DReg> model, std::shared_ptr<float3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float5DReg> model, const std::shared_ptr<float3DReg> data) const;

		/* Mutators */
		void setVel_3D(std::shared_ptr<SEP::float3DReg> vel){ _vel = vel; }
		void deallocatePinnedBornExtGpu_3D();
};

#endif
