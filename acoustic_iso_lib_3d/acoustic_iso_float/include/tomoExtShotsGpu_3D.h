#ifndef TOMO_EXT_SHOTS_GPU_3D_H
#define TOMO_EXT_SHOTS_GPU_3D_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "float3DReg.h"
#include "float4DReg.h"
#include "float5DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "seismicOperator_3D.h"
#include "interpTimeLinTbb_3D.h"
#include "secondTimeDerivative_3D.h"
#include "tomoExtShotsGpuFunctions_3D.h"

using namespace SEP;

class tomoExtShotsGpu_3D : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:
		int _nShot, _nGpu, _iGpuAlloc, _ginsu;
		int _info, _deviceNumberInfo;
		std::shared_ptr<SEP::float3DReg> _vel;
		std::shared_ptr<SEP::float2DReg> _sourcesSignals;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu_3D>> _sourcesVector, _receiversVector;
		std::shared_ptr<SEP::float5DReg> _extReflectivity;
		std::shared_ptr <hypercube> _srcWavefieldHyper;
		std::vector<int> _gpuList;
		std::vector<std::shared_ptr<SEP::hypercube>> _velHyperVectorGinsu;
		std::shared_ptr<SEP::int1DReg> _xPadMinusVectorGinsu, _xPadPlusVectorGinsu;
		std::vector<int> _ixVectorGinsu, _iyVectorGinsu;
		int _fwime;
		std::vector<float*> _pinWavefieldVec;

	public:

		/* Overloaded constructors */
		tomoExtShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::float2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::shared_ptr<SEP::float5DReg> reflectivityExt);

		tomoExtShotsGpu_3D(std::shared_ptr<SEP::float3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::float2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::shared_ptr<SEP::float5DReg> reflectivityExt, std::vector<std::shared_ptr<SEP::hypercube>> velHyperVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadMinusVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadPlusVectorGinsu, int nxMaxGinsu, int nyMaxGinu, std::vector<int> ixVectorGinsu, std::vector<int> iyVectorGinsu);

		/* Destructor */
		~tomoExtShotsGpu_3D(){};

		/* Create Gpu list */
		void createGpuIdList_3D();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

		/* Mutators */
		void setVel_3D(std::shared_ptr<SEP::float3DReg> vel){ _vel = vel; }
		void setExtReflectivity_3D(std::shared_ptr<SEP::float5DReg> extReflectivity){ _extReflectivity = extReflectivity; }

		/* Deallocate pinned memory */
		void deallocatePinnedTomoExtGpu_3D();

		/* Accessors */
		std::vector<float*> getPinWavefieldVec(){return _pinWavefieldVec;}
};

#endif