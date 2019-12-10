#include <string>
#include <double2DReg.h>
#include <double3DReg.h>
#include <double5DReg.h>
#include "fdParam_3D.h"
#include <math.h>
#include <iomanip>
#include <iostream>
using namespace SEP;

fdParam_3D::fdParam_3D(const std::shared_ptr<double3DReg> vel, const std::shared_ptr<paramObj> par) {

	_vel = vel;
	_par = par;

	/***** Coarse time-sampling *****/
	_nts = _par->getInt("nts");
	_dts = _par->getFloat("dts",0.0);
	_ots = _par->getFloat("ots", 0.0);
	_sub = _par->getInt("sub");
	_timeAxisCoarse = axis(_nts, _ots, _dts);

	/***** Fine time-sampling *****/
	_ntw = (_nts - 1) * _sub + 1;
	_dtw = _dts / double(_sub);
	_otw = _ots;
	_timeAxisFine = axis(_ntw, _otw, _dtw);

	/***** Vertical axis *****/
	_nz = _par->getInt("nz");
	_zPadPlus = _par->getInt("zPadPlus");
	_zPadMinus = _par->getInt("zPadMinus");
	_zPad = std::min(_zPadMinus, _zPadPlus);
	_dz = _par->getFloat("dz",-1.0);
	_oz = _vel->getHyper()->getAxis(1).o;
	_zAxis = axis(_nz, _oz, _dz);

	/***** Horizontal x-axis *****/
	_nx = _par->getInt("nx");
	_xPadPlus = _par->getInt("xPadPlus");
	_xPadMinus = _par->getInt("xPadMinus");
	_xPad = std::min(_xPadMinus, _xPadPlus);
	_dx = _par->getFloat("dx",-1.0);
	_ox = _vel->getHyper()->getAxis(2).o;
	_xAxis = axis(_nx, _ox, _dx);

    /***** Horizontal y-axis *****/
    _ny = _par->getInt("ny");
	_yPad = _par->getInt("yPad");
	_dy = _par->getFloat("dy",-1.0);
	_oy = _vel->getHyper()->getAxis(3).o;
	_yAxis = axis(_ny, _oy, _dy);

	/***** Extended axes *********/
	_nExt1 = _par->getInt("nExt1", 1);
    _nExt2 = _par->getInt("nExt2", 1);
	if (_nExt1 % 2 == 0) {std::cout << "**** ERROR [fdParam_3D]: Length of extended axis #1 must be an uneven number ****" << std::endl; assert(1==2); }
    if (_nExt2 % 2 == 0) {std::cout << "**** ERROR [fdParam_3D]: Length of extended axis #2 must be an uneven number ****" << std::endl; assert(1==2); }
	_hExt1 = (_nExt1-1)/2;
    _hExt2 = (_nExt2-1)/2;
	_extension = par->getString("extension", "none");

    // Time-lag extension
    if (_extension=="time"){
        // Axis #1
        _oExt1 = -_dts*_hExt1;
        _dExt1 = _dts;
        _extAxis1 = axis(_nExt1, _oExt1, _dExt1);
        // Axis #2
        _oExt2 = -_dts*_hExt2;
        _dExt2 = _dts;
        _extAxis2 = axis(_nExt2, _oExt2, _dExt2);
    }
    // Subsurface offset extension
    else if (_extension=="offset"){
        // x-axis
		_oExt1 = -_dx*_hExt1;
		_dExt1 = _dx;
		_extAxis1 = axis(_nExt1, _oExt1, _dExt1);
        // y-axis
        _oExt2 = -_dy*_hExt2;
		_dExt2 = _dy;
		_extAxis2 = axis(_nExt2, _oExt2, _dExt2);

	} else {
        // x-axis
		_oExt1 = 0.0;
		_dExt1 = 1.0;
		_extAxis1 = axis(_nExt1, _oExt1, _dExt1);
        // y-axis
        _oExt2 = 0.0;
		_dExt2 = 1.0;
		_extAxis2 = axis(_nExt, _oExt2, _dExt2);
    }

	/***** Other parameters *****/
	_fMax = _par->getFloat("fMax",1000.0);
	_blockSize = _par->getInt("blockSize");
	_fat = _par->getInt("fat");
	_minPad = std::min(_zPad, _xPad, _yPad);
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_alphaCos = par->getFloat("alphaCos", 0.99);
	_errorTolerance = par->getFloat("errorTolerance", 0.000001);

	/***** QC *****/
	assert(checkParfileConsistencySpace_3D(_vel, "Velocity file")); // Parfile - velocity file consistency
	assert(checkFdStability_3D());
	assert(checkFdDispersion_3D());
	assert(checkModelSize_3D());

	/***** Scaling for propagation *****/
	// v^2 * dtw^2
	_vel2Dtw2 = new double[_nz * _nx * _ny * sizeof(double)];
	#pragma omp parallel for collapse(3)
    for (int iy = 0; iy < _ny; iy++){
        for (int ix = 0; ix < _nx; ix++){
            for (int iz = 0; iz < _nz; iz++) {
                int i1 = iy * _nz * _nx + ix * _nz + iz;
                _vel2Dtw2[i1] = (*_vel->_mat)[iy][ix][iz] * (*_vel->_mat)[iy][ix][iz] * _dtw * _dtw;
            }
        }
    }

	/*********** Maybe put an "if" statement for nonlinear propagation *********/
	// Compute reflectivity scaling
	_reflectivityScale = new double[_nz * _nx * _ny * sizeof(double)];
	#pragma omp parallel for collapse(3)
    for (int iy = 0; iy < _ny; iy++){
        for (int ix = 0; ix < _nx; ix++){
            for (int iz = 0; iz < _nz; iz++) {
                int i1 = iy * _nz * _nx + ix * _nz + iz;
                _reflectivityScale[i1] = 2.0 / ( (*_vel->_mat)[iy][ix][iz] * (*_vel->_mat)[iy][ix][iz] * (*_vel->_mat)[iy][ix][iz] );
            }
        }
    }

void fdParam_3D::getInfo_3D(){

		std::cout << " " << std::endl;
		std::cout << "*******************************************************************" << std::endl;
		std::cout << "************************ FD PARAMETERS INFO ***********************" << std::endl;
		std::cout << "*******************************************************************" << std::endl;
		std::cout << " " << std::endl;

		// Coarse time sampling
		std::cout << "------------------------ Coarse time sampling ---------------------" << std::endl;
		std::cout << std::fixed;
		std::cout << std::setprecision(3);
		std::cout << "nts = " << _nts << " [samples], dts = " << _dts << " [s], ots = " << _ots << " [s]" << std::endl;
		std::cout << std::setprecision(1);
		std::cout << "Nyquist frequency = " << 1.0/(2.0*_dts) << " [Hz]" << std::endl;
		std::cout << "Maximum frequency from seismic source = " << _fMax << " [Hz]" << std::endl;
		std::cout << std::setprecision(3);
		std::cout << "Total recording time = " << (_nts-1) * _dts << " [s]" << std::endl;
		std::cout << "Subsampling = " << _sub << std::endl;
		std::cout << " " << std::endl;

		// Coarse time sampling
		std::cout << "------------------------ Fine time sampling -----------------------" << std::endl;
		std::cout << "ntw = " << _ntw << " [samples], dtw = " << _dtw << " [s], otw = " << _otw << " [s]" << std::endl;
		std::cout << " " << std::endl;

		// Vertical spatial sampling
		std::cout << "--------------------------- Vertical axis -------------------------" << std::endl;
		std::cout << std::setprecision(2);
		std::cout << "nz = " << _nz-2*_fat-_zPadMinus-_zPadPlus << " [samples], dz = " << _dz << " [km], oz = " << _oz+(_fat+_zPadMinus)*_dz << " [km]" << std::endl;
		std::cout << "Model thickness (area of interest) = " << _oz+(_fat+_zPadMinus)*_dz+(_nz-2*_fat-_zPadMinus-_zPadPlus-1)*_dz << " [km]" << std::endl;
		std::cout << "Top padding = " << _zPadMinus << " [samples], bottom padding = " << _zPadPlus << " [samples]" << std::endl;
		std::cout << "nz (padded) = " << _nz << " [samples], oz (padded) = " << _oz << " [km]" << std::endl;
		std::cout << "Model thickness (padding+fat) = " << _oz+(_nz-1)*_dz << " [km]" << std::endl;
		std::cout << " " << std::endl;

		// Horizontal spatial sampling
		std::cout << "-------------------------- Horizontal x-axis ----------------------" << std::endl;
		std::cout << std::setprecision(2);
		std::cout << "nx = " << _nx-2*_fat-_xPadMinus-_xPadPlus << " [samples], dx = " << _dx << " [km], ox = " << _ox+(_fat+_xPadMinus)*_dx << " [km]" << std::endl;
		std::cout << "Model width in x-direction (area of interest) = " << _ox+(_fat+_xPadMinus)*_dx+(_nx-2*_fat-_xPadMinus-_xPadPlus-1)*_dx << " [km]" << std::endl;
		std::cout << "Left padding = " << _xPadMinus << " [samples], right padding = " << _xPadPlus << " [samples]" << std::endl;
		std::cout << "nx (padded) = " << _nx << " [samples], ox (padded) = " << _ox << " [km]" << std::endl;
		std::cout << "Model width (padding+fat) = " << _ox+(_nx-1)*_dx << " [km]" << std::endl;
		std::cout << " " << std::endl;

		// Horizontal spatial sampling
		std::cout << "-------------------------- Horizontal y-axis ----------------------" << std::endl;
		std::cout << std::setprecision(2);
		std::cout << "ny = " << _ny-2*_fat-2*_yPad << " [samples], dy = " << _dy << " [km], ox = " << _oy+(_fat+_yPad)*_dy << " [km]" << std::endl;
		std::cout << "Model width in y-drection (area of interest) = " << _oy+(_fat+_yPad)*_dy+(_ny-2*_fat-2*_yPad-1)*_dy << " [km]" << std::endl;
		std::cout << "Left padding = " << _yPad << " [samples], right padding = " << _yPad << " [samples]" << std::endl;
		std::cout << "ny (padded) = " << _ny << " [samples], oy (padded) = " << _oy << " [km]" << std::endl;
		std::cout << "Model width (padding+fat) = " << _oy+(_ny-1)*_dy << " [km]" << std::endl;
		std::cout << " " << std::endl;

		// Extended axis
		if ( _extension=="time" ){
			std::cout << std::setprecision(3);
			std::cout << "-------------------- Extended axis #1: time-lags ---------------------" << std::endl;
			std::cout << "nTau = " << _hExt << " [samples], dTau= " << _dExt << " [s], oTau = " << _oExt << " [s]" << std::endl;
			std::cout << "Total extension length nTau = " << _nExt << " [samples], which corresponds to " << _nExt*_dExt << " [s]" << std::endl;
			std::cout << " " << std::endl;
		}

		if ( _extension=="offset" ){
			std::cout << std::setprecision(2);
			std::cout << "---------- Extended x-axis: horizontal subsurface offsets -----------" << std::endl;
			std::cout << "nxOffset = " << _hExt1 << " [samples], dxOffset= " << _dExt1 << " [km], oxOffset = " << _oExt1 << " [km]" << std::endl;
			std::cout << "Total extension in x-direction length nxOffset = " << _nExt1 << " [samples], which corresponds to " << _nExt1*_dExt1 << " [km]" << std::endl;
			std::cout << " " << std::endl;

            std::cout << "---------- Extended y-axis: horizontal subsurface offsets -----------" << std::endl;
			std::cout << "nyOffset = " << _hExt2 << " [samples], dyOffset= " << _dExt2 << " [km], oyOffset = " << _oExt2 << " [km]" << std::endl;
			std::cout << "Total extension in y-direction length nyOffset = " << _nExt2 << " [samples], which corresponds to " << _nExt2*_dExt2 << " [km]" << std::endl;
			std::cout << " " << std::endl;
		}

		// GPU FD parameters
		std::cout << "---------------------- GPU kernels parameters ---------------------" << std::endl;
		std::cout << "Block size in z-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cout << "Block size in x-direction = " << _blockSize << " [threads/block]" << std::endl;
        std::cout << "Block size in y-direction = " << _blockSize << " [threads/block]" << std::endl;
		std::cout << "Halo size for Laplacian 8th order [FAT] = " << _fat << " [samples]" << std::endl;
		std::cout << " " << std::endl;

		// Stability and dispersion
		std::cout << "---------------------- Stability and dispersion -------------------" << std::endl;
		std::cout << std::setprecision(2);
		std::cout << "Courant number = " << _Courant << " [-]" << std::endl;
		std::cout << "Dispersion ratio = " << _dispersionRatio << " [points/min wavelength]" << std::endl;
		std::cout << "Minimum velocity value = " << _minVel << " [km/s]" << std::endl;
		std::cout << "Maximum velocity value = " << _maxVel << " [km/s]" << std::endl;
		std::cout << std::setprecision(1);
		std::cout << "Maximum frequency without dispersion = " << _minVel/(3.0*std::max(_dz, _dx, _dy)) << " [Hz]" << std::endl;
		std::cout << " " << std::endl;
		std::cout << "*******************************************************************" << std::endl;
		std::cout << " " << std::endl;
		std::cout << std::scientific; // Reset to scientific formatting notation
		std::cout << std::setprecision(6); // Reset the default formatting precision
}

bool fdParam_3D::checkFdStability_3D(double CourantMax){
	_maxVel = _vel->max();
	_minDzDxDy = std::min(_dz, _dx, _dy);
	_Courant = _maxVel * _dtw / _minDzDxDy;
	if (_Courant > CourantMax){
		std::cout << "**** ERROR [fdParam_3D]: Courant is too big: " << _Courant << " ****" << std::endl;
		std::cout << "Max velocity value: " << _maxVel << std::endl;
		std::cout << "Dtw: " << _dtw << " [s]" << std::endl;
		std::cout << "Min (dz, dx, dy): " << _minDzDxDy << " [km]" << std::endl;
		return false;
	}
	return true;
}

bool fdParam_3D::checkFdDispersion_3D(double dispersionRatioMin){

	_minVel = _vel->min();
	_maxDzDxDy = std::max(_dz, _dx, _dy);
	_dispersionRatio = _minVel / (_fMax*_maxDzDxDy);

	if (_dispersionRatio < dispersionRatioMin){
		std::cout << "**** ERROR [fdParam_3D]: Dispersion ratio is too small: " << _dispersionRatio <<  " > " << dispersionRatioMin << " ****" << std::endl;
		std::cout << "Min velocity value = " << _minVel << " [km/s]" << std::endl;
		std::cout << "Max (dz, dx, dy) = " << _maxDzDxDy << " [km]" << std::endl;
		std::cout << "Max frequency = " << _fMax << " [Hz]" << std::endl;
		return false;
	}
	return true;
}

bool fdParam_3D::checkModelSize_3D(){
	if ( (_nz-2*_fat) % _blockSize != 0) {
		std::cout << "**** ERROR [fdParam_3D]: nz-2xFAT not a multiple of block size ****" << std::endl;
		return false;
	}
	if ((_nx-2*_fat) % _blockSize != 0) {
		std::cout << "**** ERROR [fdParam_3D]: nx-2xFAT not a multiple of block size ****" << std::endl;
		return false;
	}

	// Not needed for the slow axis
    // if ((_ny-2*_fat) % _blockSize != 0) {
	// 	std::cout << "**** ERROR: ny not a multiple of block size ****" << std::endl;
	// 	return false;
	// }

	return true;
}

bool fdParam_3D::checkParfileConsistencyTime_3D(const std::shared_ptr<double2DReg> seismicTraces, int timeAxisIndex, std::string fileToCheck) const {
	if (_nts != seismicTraces->getHyper()->getAxis(timeAxisIndex).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: nts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dts - seismicTraces->getHyper()->getAxis(timeAxisIndex).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: dts not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ots - seismicTraces->getHyper()->getAxis(timeAxisIndex).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: ots not consistent with parfile ****" << std::endl; return false;}
	return true;
}

// check consistency of velocity model
bool fdParam_3D::checkParfileConsistencySpace_3D(const std::shared_ptr<double3DReg> model, std::string fileToCheck) const {

	// Vertical axis
	if (_nz != model->getHyper()->getAxis(1).n) {std::cout << "**** ["<< fileToCheck << "] ERROR [fdParam_3D]: nz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dz - model->getHyper()->getAxis(1).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: dz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oz - model->getHyper()->getAxis(1).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: oz not consistent with parfile ****" << std::endl; return false;}

	// Horizontal x-axis
	if (_nx != model->getHyper()->getAxis(2).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: nx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dx - model->getHyper()->getAxis(2).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: dx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ox - model->getHyper()->getAxis(2).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: ox not consistent with parfile ****" << std::endl; return false;}

	// Horizontal y-axis
	if (_ny != model->getHyper()->getAxis(3).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: ny not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dy - model->getHyper()->getAxis(3).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: dy not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oy - model->getHyper()->getAxis(3).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: oy not consistent with parfile ****" << std::endl; return false;}

	return true;
}

// check consistency of extended image
bool fdParam_3D::checkParfileConsistencySpace_3D(const std::shared_ptr<double5DReg> modelExt, std::string fileToCheck) const {

	// Vertical z-axis
	if (_nz != modelExt->getHyper()->getAxis(1).n) {std::cout << "**** ["<< fileToCheck << "] ERROR [fdParam_3D]: nz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dz - modelExt->getHyper()->getAxis(1).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: dz not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oz - modelExt->getHyper()->getAxis(1).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: oz not consistent with parfile ****" << std::endl; return false;}

	// Horizontal x-axis
	if (_nx != modelExt->getHyper()->getAxis(2).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: nx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dx - modelExt->getHyper()->getAxis(2).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: dx not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_ox - modelExt->getHyper()->getAxis(2).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: ox not consistent with parfile ****" << std::endl; return false;}

	// Horizontal y-axis
	if (_ny != modelExt->getHyper()->getAxis(3).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: ny not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_dy - modelExt->getHyper()->getAxis(3).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: dy not consistent with parfile ****" << std::endl; return false;}
	if ( std::abs(_oy - modelExt->getHyper()->getAxis(3).o) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: oy not consistent with parfile ****" << std::endl; return false;}

	// Extended axis #1
	if (_nExt1 != modelExt->getHyper()->getAxis(4).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: nExt #1 not consistent with parfile ****" << std::endl; return false;}
	if (_nExt1>1) {
		if ( std::abs(_dExt1 - modelExt->getHyper()->getAxis(4).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: dExt #1 not consistent with parfile ****" << std::endl; return false;}
		if ( std::abs(_oExt1 - modelExt->getHyper()->getAxis(4).o) > _errorTolerance ) { std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: oExt #1 not consistent with parfile ****" << std::endl; return false;}
	}

	// Extended axis #2
	if (_nExt2 != modelExt->getHyper()->getAxis(5).n) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: nExt #2 not consistent with parfile ****" << std::endl; return false;}
	if (_nExt2>1) {
		if ( std::abs(_dExt2 - modelExt->getHyper()->getAxis(5).d) > _errorTolerance ) {std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: dExt #2 not consistent with parfile ****" << std::endl; return false;}
		if ( std::abs(_oExt2 - modelExt->getHyper()->getAxis(5).o) > _errorTolerance ) { std::cout << "**** [" << fileToCheck << "] ERROR [fdParam_3D]: oExt #2 not consistent with parfile ****" << std::endl; return false;}
	}

	return true;
}

fdParam_3D::~fdParam_3D(){
	// Deallocate _vel2Dtw2 on host
	delete [] _vel2Dtw2;
	_vel2Dtw2 = NULL;
}
