#ifndef TOMO_EXT_SHOTS_GPU_3D_H
#define TOMO_EXT_SHOTS_GPU_3D_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "double3DReg.h"
#include "double4DReg.h"
#include "double5DReg.h"
#include "ioModes.h"
#include "deviceGpu_3D.h"
#include "fdParam_3D.h"
#include "seismicOperator_3D.h"
#include "interpTimeLinTbb_3D.h"
#include "secondTimeDerivative_3D.h"
#include "tomoExtShotsGpuFunctions_3D.h"

using namespace SEP;

class tomoExtShotsGpu_3D : public Operator<SEP::double3DReg, SEP::double3DReg> {

	private:
		int _nShot, _nGpu, _iGpuAlloc;
		int _saveWavefield, _wavefieldShotNumber, _info, _deviceNumberInfo;
		std::shared_ptr<SEP::double3DReg> _vel;
		std::shared_ptr<SEP::double2DReg> _sourcesSignals;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu_3D>> _sourcesVector, _receiversVector;
		std::vector<std::shared_ptr<SEP::double4DReg>> _wavefieldVector1, _wavefieldVector2;
		std::shared_ptr<SEP::double5DReg> _extReflectivity;
		std::vector<int> _gpuList;
		std::shared_ptr <hypercube> _wavefieldHyper;

	public:

		/* Overloaded constructors */
		tomoExtShotsGpu_3D(std::shared_ptr<SEP::double3DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu_3D>> sourcesVector, std::shared_ptr<SEP::double2DReg> sourcesSignals, std::vector<std::shared_ptr<deviceGpu_3D>> receiversVector, std::shared_ptr<SEP::double5DReg> reflectivityExt);

		/* Destructor */
		~tomoExtShotsGpu_3D(){};

		/* Create Gpu list */
		void createGpuIdList_3D();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const;

		/* Accessor */
		// iWavefield1 corresponds to the wavefield for iGpu #iWavefield
		std::shared_ptr<double4DReg> getWavefield1_3D(int iWavefield1) {
			if ( iWavefield1 < 0 || iWavefield1 > _nGpu-1){
				std::cout << "**** ERROR [tomoExtShotsGpu_3D]: Please provide a valid ID for the wavefield to be saved ****" << std::endl;
				assert(1==2);
			}
			return _wavefieldVector1[iWavefield1];
		}
		std::shared_ptr<double4DReg> getWavefield2_3D(int iWavefield2) {
			if ( iWavefield2 < 0 || iWavefield2 > _nGpu-1){
				std::cout << "**** ERROR [tomoExtShotsGpu_3D]: Please provide a valid ID for the wavefield to be saved ****" << std::endl;
				assert(1==2);
			}
			return _wavefieldVector2[iWavefield2];
		}

		/* Mutators */
		void setVel_3D(std::shared_ptr<SEP::double3DReg> vel){ _vel = vel; }
		void setExtReflectivity(std::shared_ptr<SEP::double5DReg> extReflectivity){ _extReflectivity = extReflectivity; }

};

#endif
