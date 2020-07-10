#ifndef BORN_SHOTS_GPU_3D_H
#define BORN_SHOTS_GPU_3D_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "double2DReg.h"
#include "double3DReg.h"
#include "double4DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "operator.h"
#include "BornShotsGpuFunctions_3D.h"


using namespace SEP;

class BornShotsGpu_3D : public Operator<SEP::double3DReg, SEP::double3DReg> {

	private:
		int _nShot, _nGpu, _iGpuAlloc, _ginsu;
		int _info, _deviceNumberInfo;
		std::shared_ptr<SEP::double3DReg> _vel;
		std::shared_ptr<SEP::double2DReg> _sourcesSignals;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu_3D>> _sourcesVector, _receiversVector;
		std::vector<std::shared_ptr<SEP::double4DReg>> _srcWavefieldVector;
		std::vector<int> _gpuList;
		std::shared_ptr <hypercube> _srcWavefieldHyper;
		std::vector<std::shared_ptr<SEP::hypercube>> _velHyperVectorGinsu;
		std::shared_ptr<SEP::int1DReg> _xPadMinusVectorGinsu, _xPadPlusVectorGinsu;

	public:

		/* Overloaded constructor */
		BornShotsGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::double2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::vector<std::shared_ptr<SEP::double4DReg>> srcWavefieldVector);

		/* Overloaded constructor for Ginsu */
		BornShotsGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::double2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::vector<std::shared_ptr<SEP::hypercube>> velHyperVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadMinusVectorGinsu, std::shared_ptr<SEP::int1DReg> xPadPlusVectorGinsu, int nxMaxGinsu, int nyMaxGinu, std::vector<std::shared_ptr<SEP::double4DReg>> srcWavefieldVector);

		/* Destructor */
		~BornShotsGpu_3D(){};

		/* Create Gpu list */
		void createGpuIdList_3D();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const;

		/* Mutators */
		void setVel_3D(std::shared_ptr<SEP::double3DReg> vel){ _vel = vel; }

		/* Accessor */
		// iSrcWavefield corresponds to the wavefield for iGpu #iWavefield
		std::shared_ptr<double4DReg> getSrcWavefield_3D(int iSrcWavefield) {
			if ( iSrcWavefield < 0 || iSrcWavefield > _nGpu-1){
				std::cout << "**** ERROR [BornShotsGpu_3D]: Please provide a valid ID for the wavefield to be saved ****" << std::endl;
				assert(1==2);
			}
			return _srcWavefieldVector[iSrcWavefield];
		}

};

#endif
