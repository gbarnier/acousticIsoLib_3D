#ifndef FD_PARAM_3D_H
#define FD_PARAM_3D_H 1

#include <string>
#include <double1DReg.h>
#include "double2DReg.h"
#include "double3DReg.h"
#include "double5DReg.h"
#include "ioModes.h"
#include <iostream>

using namespace SEP;

class fdParam_3D{

 	public:

		// Constructor
  		fdParam_3D(const std::shared_ptr<double3DReg> vel, const std::shared_ptr<paramObj> par, std::string seismicOpType);

        // Ginsu mutator
  		void setFdParamGinsu_3D(std::shared_ptr<SEP::hypercube> velHyperGinsu, int xPadMinusGinsu, int xPadPlusGinsu, int ixGinsu, int iyGinsu);

        // Ginsu mutator for tomo
  		void setFdParamGinsu_3D(std::shared_ptr<SEP::hypercube> velHyperGinsu, int xPadMinusGinsu, int xPadPlusGinsu, int ixGinsu, int iyGinsu, const std::shared_ptr<double5DReg> extReflectivity);

        // Destructor
  		~fdParam_3D();

		// QC stuff
		bool checkParfileConsistencyTime_3D(const std::shared_ptr<double2DReg> seismicTraces, int timeAxisIndex, std::string fileToCheck) const;
		bool checkParfileConsistencySpace_3D(const std::shared_ptr<double3DReg> model, std::string fileToCheck) const;
		bool checkParfileConsistencySpace_3D(const std::shared_ptr<double5DReg> modelExt, std::string fileToCheck) const;

		bool checkFdStability_3D(double courantMax=0.45);
		bool checkFdDispersion_3D(double dispersionRatioMin=3.0);
		bool checkModelSize_3D(); // Make sure the domain size (without the FAT) is a multiple of the dimblock size
		void getInfo_3D();

		// Variables
		std::shared_ptr<paramObj> _par;
		std::shared_ptr<double3DReg> _vel, _smallVel;
		axis _timeAxisCoarse, _timeAxisFine, _zAxis, _xAxis, _yAxis, _extAxis1, _extAxis2;

        // Damping volume
        std::shared_ptr<double3DReg> _dampVolume;
        std::shared_ptr<double1DReg> _dampArray;

		double *_vel2Dtw2, *_reflectivityScale;
		double _errorTolerance;
		double _minVel, _maxVel, _minDzDxDy, _maxDzDxDy;
		int _nts, _sub, _ntw;
		double _ots, _dts, _otw, _dtw, _oExt1, _oExt2, _dExt1, _dExt2;
		double _Courant, _dispersionRatio;
		int _nz, _nx, _ny, _nExt1, _nExt2, _hExt1, _hExt2, _nzSmall, _nxSmall, _nySmall;
		int _zPadMinus, _zPadPlus, _xPadMinus, _xPadPlus, _yPadMinus, _yPadPlus, _zPad, _xPad, _yPad, _minPad;
		double _dz, _dx, _dy, _oz, _ox, _oy, _fMax;
		int _saveWavefield, _blockSize, _fat, _freeSurface, _splitTopBody;
		double _alphaCos;
		std::string _extension;
        std::string _seismicOpType; // Nonlinear, Born, BornExt, tomoExt

        // Ginsu parameters
        int _ginsu;
        double *_vel2Dtw2Ginsu, *_reflectivityScaleGinsu, *_extReflectivityGinsu;
        int _nzGinsu, _nxGinsu, _nyGinsu;
        int _zPadMinusGinsu, _zPadPlusGinsu, _xPadMinusGinsu, _xPadPlusGinsu, _yPadMinusGinsu, _yPadPlusGinsu, _zPadGinsu, _xPadGinsu, _yPadGinsu, _minPadGinsu;
        double _ozGinsu, _dzGinsu, _oxGinsu, _dxGinsu, _oyGinsu, _dyGinsu;
        int _izGinsu, _ixGinsu, _iyGinsu;
        axis _zAxisGinsu, _xAxisGinsu, _yAxisGinsu;

};

#endif
