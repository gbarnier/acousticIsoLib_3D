#ifndef DEVICE_GPU_3D_H
#define DEVICE_GPU_3D_H 1

#include "ioModes.h"
#include "double1DReg.h"
#include "double2DReg.h"
#include "double3DReg.h"
#include "operator.h"
#include <vector>

using namespace SEP;

class deviceGpu_3D : public Operator<SEP::double2DReg, SEP::double2DReg> {

	private:

		/* Spatial interpolation */
		std::shared_ptr<double3DReg> _vel;
		std::shared_ptr<double1DReg> _zCoord, _xCoord, _yCoord;
		std::vector<long long> _gridPointIndexUnique; // Array containing all the positions of the excited grid points - each grid point is unique
		std::map<int, int> _indexMap;
		std::map<int, int>::iterator _iteratorIndexMap;
		double *_weight;
		double _oz, _dz, _ox, _dx, _oy, _dy;
		int *_gridPointIndex;
		int _nDeviceIrreg, _nDeviceReg, _nt, _nz, _nx, _ny;
		int _nzSmall, _nxSmall, _nySmall;
		int _fat, _zPadMinus, _zPadPlus, _xPadMinus, _xPadPlus, _yPad;
		int _dipole, _zDipoleShift, _xDipoleShift, _yDipoleShift;
		int _hFilter1d, _nFilter1d, _nFilter3d, _nFilter3dDipole;
		std::string _interpMethod;
		std::shared_ptr<paramObj> _par;

	public:

		// Constructor #1: Provide positions of acquisition devices in km
		// Acquisition devices do not need to be placed on grid points
		deviceGpu_3D(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord, const std::shared_ptr<double1DReg> yCoord, const std::shared_ptr<double3DReg> vel, int &nt, std::shared_ptr<paramObj> par, int dipole=0, double zDipoleShift=0, double xDipoleShift=0, double yDipoleShift=0, std::string interpMethod="linear", int hFilter1d=1);

		// Constructor #2: Provide regular array of acquisition device by o, d, n
		// Assumes acquisition devices are placed on grid points
		deviceGpu_3D(const int &nzDevice, const int &ozDevice, const int &dzDevice, const int &nxDevice, const int &oxDevice, const int &dxDevice, const int &nyDevice, const int &oyDevice, const int &dyDevice, const std::shared_ptr<double3DReg> vel, int &nt, std::shared_ptr<paramObj> par, int dipole=0, int zDipoleShift=0, int xDipoleShift=0, int yDipoleShift=0, std::string interpMethod="linear", int hFilter1d=1);

		// FWD / ADJ
		void forward(const bool add, const std::shared_ptr<double2DReg> signalReg, std::shared_ptr<double2DReg> signalIrreg) const;
		void adjoint(const bool add, std::shared_ptr<double2DReg> signalReg, const std::shared_ptr<double2DReg> signalIrreg) const;

		// Destructor
		~deviceGpu_3D(){};

		// Other functions
  		void checkOutOfBounds(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord, const std::shared_ptr<double1DReg> yCoord, const std::shared_ptr<double3DReg> vel); // For constructor #1
		void checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const int &nyDevice, const int &oyDevice, const int &dyDevice, const std::shared_ptr<double3DReg> vel); // For constructor #3
  		void convertIrregToReg();
		void calcLinearWeights();
		void calcSincWeights();

  		long long *getRegPosUnique(){ return _gridPointIndexUnique.data(); }
  		int *getRegPos(){ return _gridPointIndex; }
  		int getNt(){ return _nt; }
  		int getNDeviceReg(){ return _nDeviceReg; }
  		int getNDeviceIrreg(){ return _nDeviceIrreg; }
  		double * getWeights() { return _weight; }
  		int getSizePosUnique(){ return _gridPointIndexUnique.size(); }
};

#endif
