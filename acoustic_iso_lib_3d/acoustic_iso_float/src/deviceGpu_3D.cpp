#include <float1DReg.h>
#include <float2DReg.h>
#include <float3DReg.h>
#include <iostream>
#include "deviceGpu_3D.h"
#include <boost/math/special_functions/sinc.hpp>
#include <boost/math/special_functions/cos_pi.hpp>
#include <boost/math/distributions/normal.hpp>
#include <math.h>
#include <vector>


// Constructor #1 -- Only for irregular geometry
deviceGpu_3D::deviceGpu_3D(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<float1DReg> yCoord, const std::shared_ptr<float3DReg> vel, int &nt, std::shared_ptr<paramObj> par, int dipole, float zDipoleShift, float xDipoleShift, float yDipoleShift, std::string interpMethod, int hFilter1d){

	// Get domain dimensions
	_vel = vel;
	_par = par;
	_oz = vel->getHyper()->getAxis(1).o;
	_dz = vel->getHyper()->getAxis(1).d;
	_nz = vel->getHyper()->getAxis(1).n;
	_ox = vel->getHyper()->getAxis(2).o;
	_dx = vel->getHyper()->getAxis(2).d;
	_nx = vel->getHyper()->getAxis(2).n;
	_oy = vel->getHyper()->getAxis(3).o;
	_dy = vel->getHyper()->getAxis(3).d;
	_ny = vel->getHyper()->getAxis(3).n;
	_fat = _par->getInt("fat");
	_zPadMinus = _par->getInt("zPadMinus");
	_zPadPlus = _par->getInt("zPadPlus");
	_xPadMinus = _par->getInt("xPadMinus");
	_xPadPlus = _par->getInt("xPadPlus");
	_yPad = _par->getInt("yPad");
	_errorTolerance = par->getFloat("errorTolerance", 1e-4);

	// Get positions of aquisition devices + other parameters
	_zCoord = zCoord;
	_xCoord = xCoord;
    _yCoord = yCoord;
	_dipole = dipole; // Default value is 0
	_zDipoleShift = zDipoleShift; // Default value is 0
	_xDipoleShift = xDipoleShift; // Default value is 0
	_yDipoleShift = yDipoleShift; // Default value is 0
	checkOutOfBounds(_zCoord, _xCoord, _yCoord); // Check that none of the acquisition devices are out of bounds
	_hFilter1d = hFilter1d; // Half-length of the filter for each dimension. For sinc, the filter in each direction z, x, y has the same length
	_interpMethod = interpMethod; // Default is linear
	_nDeviceIrreg = _zCoord->getHyper()->getAxis(1).n; // Nb of devices on irregular grid
	_nt = nt;

	// Check that for the linear interpolation case, the filter half-length is 1
	if (_interpMethod == "linear" && _hFilter1d != 1){
		std::cout << "**** ERROR [deviceGpu_3D]: Half-length of interpolation filter should be set to 1 for linear interpolation ****" << std::endl;
		throw std::runtime_error("");
	}

	// Compute the total number of points on the grid for each axis that will be involved in the interpolation
	_nFilter1d = 2*_hFilter1d; // Filter length for each dimension. For sinc, we have "_hFilter" number of points on each side
	_nFilter3d = _nFilter1d*_nFilter1d*_nFilter1d; // Total number of points involved in the interpolation in 3D

	// Dipole case: we use twice as many points
	if (_dipole == 1){
		_nFilter3dDipole = _nFilter3d;
		_nFilter3d = _nFilter3d*2;
	}

	_gridPointIndex = new int[_nFilter3d*_nDeviceIrreg]; // 1d-array containing the index of all the grid points (on the regular grid) that will be used in the interpolation. The indices are not unique
	_weight = new float[_nFilter3d*_nDeviceIrreg]; // Weights corresponding to the points on the grid stored in _gridPointIndex

	// Compute list of all grid points used in the interpolation
	if (_interpMethod == "linear"){
		calcLinearWeights();
	} else if (_interpMethod == "sinc"){
		calcSincWeights();
	} else {
		std::cerr << "**** ERROR [deviceGpu_3D]: Space interpolation method not supported ****" << std::endl;
	}

	// Convert the list -> Create a new list with unique indices of the regular grid points involved in the interpolation
	convertIrregToReg();

}

// Constructor #2 -- Only for regular geometry
// This contructor only allows linear interpolation
deviceGpu_3D::deviceGpu_3D(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const int &nyDevice, const int &oyDevice, const int &dyDevice, const std::shared_ptr<float3DReg> vel, int &nt, std::shared_ptr<paramObj> par, int dipole, int zDipoleShift, int xDipoleShift, int yDipoleShift, std::string interpMethod, int hFilter1d){

	_vel = vel;
	_nt = nt;
	_par = par;
	_dipole = dipole;
	_zDipoleShift = zDipoleShift;
	_xDipoleShift = xDipoleShift;
	_yDipoleShift = yDipoleShift;
	_hFilter1d = hFilter1d; // Half-length of the filter for each dimension (for sinc, the filter in each direction has the same length)
	_interpMethod = interpMethod;
	// std::cout << "Before check out of bounds" << std::endl;
	// std::cout << "After check out of bounds" << std::endl;
	_nDeviceIrreg = nzDevice * nxDevice * nyDevice; // Nb of devices on irregular grid
	_oz = vel->getHyper()->getAxis(1).o;
	_dz = vel->getHyper()->getAxis(1).d;
	_nz = vel->getHyper()->getAxis(1).n;
	_ox = vel->getHyper()->getAxis(2).o;
	_dx = vel->getHyper()->getAxis(2).d;
	_nx = vel->getHyper()->getAxis(2).n;
	_oy = vel->getHyper()->getAxis(3).o;
	_dy = vel->getHyper()->getAxis(3).d;
	_ny = vel->getHyper()->getAxis(3).n;
	_fat = _par->getInt("fat");
	_zPadMinus = _par->getInt("zPadMinus");
	_zPadPlus = _par->getInt("zPadPlus");
	_xPadMinus = _par->getInt("xPadMinus");
	_xPadPlus = _par->getInt("xPadPlus");
	_yPad = _par->getInt("yPad");

	checkOutOfBounds(nzDevice, ozDevice, dzDevice, nxDevice, oxDevice, dxDevice, nyDevice, oyDevice, dyDevice);
	// Check that the user is not asking for sinc interpolation
	if (_interpMethod != "linear"){
		std::cout << "**** ERROR [deviceGpu_3D]: The constructor used only supports linear spatial interpolation ****" << std::endl;
		throw std::runtime_error("");
	}
	// Check that for the linear interpolation case, the filter half-length is 1
	if (_hFilter1d != 1){
		std::cout << "**** ERROR [deviceGpu_3D]: Half-length of interpolation filter should be 1 for linear interpolation ****" << std::endl;
		throw std::runtime_error("");
	}

	// Compute the total number of points on the grid for each axis that will be involved in the interpolation
	_nFilter1d = 2*_hFilter1d;
	_nFilter3d = _nFilter1d*_nFilter1d*_nFilter1d; // Total number of points involved in the interpolation
	if (_dipole == 1){_nFilter3d=_nFilter3d*2;} // Dipole case
	_gridPointIndex = new int[_nFilter3d*_nDeviceIrreg]; // Index of all the neighboring points of each device (non-unique) on the regular "1D" grid
	_weight = new float[_nFilter3d*_nDeviceIrreg]; // Weights for spatial interpolation

	int iDevice = -1;

    for (int iy = 0; iy < nyDevice; iy++) {
        int iyDevice = oyDevice + iy * dyDevice; // y-position of device on FD grid

        for (int ix = 0; ix < nxDevice; ix++) {
            int ixDevice = oxDevice + ix * dxDevice; // x-position of device on FD grid

            for (int iz = 0; iz < nzDevice; iz++) {
                int izDevice = ozDevice + iz * dzDevice; // z-position of device on FD grid
                iDevice++;
                int i1 = iDevice * _nFilter3d;

    			// Top front left
    			_gridPointIndex[i1] = iyDevice * _nz * _nx + ixDevice * _nz + izDevice;
    			_weight[i1] = 1.0;

    			// Bottom front left
    			_gridPointIndex[i1+1] = _gridPointIndex[i1] + 1;
    			_weight[i1+1] = 0.0;

    			// Top front right
    			_gridPointIndex[i1+2] = _gridPointIndex[i1] + _nz;
    			_weight[i1+2] = 0.0;

    			// Bottom font right
    			_gridPointIndex[i1+3] = _gridPointIndex[i1] + _nz + 1;
    			_weight[i1+3] = 0.0;

                // Top back left
        		_gridPointIndex[i1+4] = _gridPointIndex[i1] + _nz *_nx;
        		_weight[i1+4] = 0.0;

        		// Bottom back left
        		_gridPointIndex[i1+5] = _gridPointIndex[i1] + _nz *_nx + 1;
        		_weight[i1+5] = 0.0;

            	// Top back right
        		_gridPointIndex[i1+6] = _gridPointIndex[i1] + _nz *_nx + _nz;
        		_weight[i1+6] = 0.0;

            	// Bottom back right
        		_gridPointIndex[i1+7] = _gridPointIndex[i1] + _nz *_nx + _nz + 1;
        		_weight[i1+7] = 0.0;

				if (_dipole == 1){

					// Top front left (dipole point)
					_gridPointIndex[i1+8] = (iyDevice + yDipoleShift) * _nz * _nx + (ixDevice + xDipoleShift) * _nz + izDevice + zDipoleShift;
					_weight[i1+8] = -1.0;

					// Bottom front left (dipole point)
					_gridPointIndex[i1+9] = _gridPointIndex[i1+8] + 1;
					_weight[i1+9] = 0.0;

					// Top front right (dipole point)
					_gridPointIndex[i1+10] = _gridPointIndex[i1+8] + _nz;
					_weight[i1+10] = 0.0;

					// Bottom front right (dipole point)
					_gridPointIndex[i1+11] = _gridPointIndex[i1+8] + _nz + 1;
					_weight[i1+11] = 0.0;

					// Top back left (dipole point)
					_gridPointIndex[i1+12] = _gridPointIndex[i1+8] + _nz * _nx;
					_weight[i1+12] = 0.0;

					// Bottom back left (dipole point)
					_gridPointIndex[i1+13] = _gridPointIndex[i1+8] + _nz * _nx + 1;
					_weight[i1+13] = 0.0;

					// Top back right (dipole point)
					_gridPointIndex[i1+14] = _gridPointIndex[i1+8] + _nz * _nx + _nz;
					_weight[i1+14] = 0.0;

					// Bottom back right (dipole point)
					_gridPointIndex[i1+15] = _gridPointIndex[i1+8] + _nz * _nx + _nz + 1;
					_weight[i1+15] = 0.0;
				}
			}
		}
	}
	convertIrregToReg();

	// std::cout << "_nDeviceIrreg" << _nDeviceIrreg << std::endl;
	// int sizeGridPointIndex = _nFilter3d*_nDeviceIrreg;
	// for (int i=0; i<sizeGridPointIndex; i++){
	// 	std::cout << "gridPointIndex [" << i << "] =" << _gridPointIndex[i] << std::endl;
	// }

	// std::cout << "_nDeviceRegUnique" << _nDeviceReg << std::endl;
	// for (int i=0; i<_nDeviceReg; i++){
	// 	std::cout << "gridPointIndexUnique [" << i << "] =" << _gridPointIndexUnique[i] << std::endl;
	// }

	// std::cout << "Done constructor" << std::endl;

}

// Update spatial interpolation parameters for Ginsu
void deviceGpu_3D::setDeviceGpuGinsu_3D(const std::shared_ptr<SEP::hypercube> velHyperGinsu, const int xPadMinusGinsu, const int xPadPlusGinsu){

	// Get domain dimensions
	_oz = velHyperGinsu->getAxis(1).o;
	_dz = velHyperGinsu->getAxis(1).d;
	_nz = velHyperGinsu->getAxis(1).n;
	_ox = velHyperGinsu->getAxis(2).o;
	_dx = velHyperGinsu->getAxis(2).d;
	_nx = velHyperGinsu->getAxis(2).n;
	_oy = velHyperGinsu->getAxis(3).o;
	_dy = velHyperGinsu->getAxis(3).d;
	_ny = velHyperGinsu->getAxis(3).n;
	_xPadMinus = xPadMinusGinsu;
	_xPadPlus = xPadPlusGinsu;

	std::cout << "Device Ginsu, _oz = " << _oz << std::endl;
	std::cout << "Device Ginsu, _dz = " << _dz << std::endl;
	std::cout << "Device Ginsu, _nz = " << _nz << std::endl;
	std::cout << "Device Ginsu, _ox = " << _ox << std::endl;
	std::cout << "Device Ginsu, _dx = " << _dx << std::endl;
	std::cout << "Device Ginsu, _nx = " << _nx << std::endl;
	std::cout << "Device Ginsu, _oy = " << _oy << std::endl;
	std::cout << "Device Ginsu, _dy = " << _dy << std::endl;
	std::cout << "Device Ginsu, _ny = " << _ny << std::endl;

	// std::cout << "_xPadMinus = " << _xPadMinus << std::endl;
	// std::cout << "_xPadPlus = " << _xPadPlus << std::endl;
	// std::cout << "_nz = " << _nz << std::endl;
	// std::cout << "_nx = " << _nx << std::endl;
	// std::cout << "_ny = " << _ny << std::endl;
	// std::cout << "_nDeviceIrreg = " << _nDeviceIrreg << std::endl;
	// std::cout << "_nDeviceReg = " << _nDeviceReg << std::endl;

	checkOutOfBounds(_zCoord, _xCoord, _yCoord); // Check that none of the acquisition devices are out of bounds

	// Compute list of all grid points used in the interpolation
	if (_interpMethod == "linear"){
		calcLinearWeights();
	} else if (_interpMethod == "sinc"){
		calcSincWeights();
	} else {
		std::cerr << "**** ERROR [deviceGpu_3D]: Space interpolation method not supported ****" << std::endl;
	}

	// std::cout << "_nDeviceIrreg before = " << _nDeviceIrreg << std::endl;
	// std::cout << "_nDeviceReg before = " << _nDeviceReg << std::endl;

	// Convert the list -> Create a new list with unique indices of the regular grid points involved in the interpolation
	convertIrregToReg();

	// std::cout << "_nDeviceIrreg after = " << _nDeviceIrreg << std::endl;
	// std::cout << "_nDeviceReg after = " << _nDeviceReg << std::endl;

}

void deviceGpu_3D::convertIrregToReg() {

	/* (1) Create map where:
		- Key = excited grid point index (points are unique)
		- Value = signal trace number
		(2) Create a vector containing the indices of the excited grid points
	*/

	_nDeviceReg = 0; // Initialize the number of regular devices to zero
	_gridPointIndexUnique.clear(); // Initialize to empty vector
	_indexMap.clear(); // Clear the map to reinitialize the unique device positions array

	for (long long iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over gridPointIndex array
		for (long long iFilter = 0; iFilter < _nFilter3d; iFilter++){
			long long i1 = iDevice * _nFilter3d + iFilter;

			// If the grid point is not already in the list
			if (_indexMap.count(_gridPointIndex[i1]) == 0) {
				// std::cout << "Adding" << std::endl;
				_nDeviceReg++; // Increment the number of (unique) grid points excited by the signal
				_indexMap[_gridPointIndex[i1]] = _nDeviceReg - 1; // Add the pair to the map
				_gridPointIndexUnique.push_back(_gridPointIndex[i1]); // Append vector containing all unique grid point index
			}
		}
	}
}

void deviceGpu_3D::forward(const bool add, const std::shared_ptr<float2DReg> signalReg, std::shared_ptr<float2DReg> signalIrreg) const {

	/* FORWARD: Go from REGULAR grid -> IRREGULAR grid */
	if (!add) signalIrreg->scale(0.0);

	std::shared_ptr<float2D> d = signalIrreg->_mat;
	std::shared_ptr<float2D> m = signalReg->_mat;

	for (long long iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over device
		for (long long iFilter = 0; iFilter < _nFilter3d; iFilter++){ // Loop over neighboring points on regular grid
			long long i1 = iDevice * _nFilter3d + iFilter;
			long long i2 = _indexMap.find(_gridPointIndex[i1])->second;
			for (long long it = 0; it < _nt; it++){
				(*d)[iDevice][it] += _weight[i1] * (*m)[i2][it];
			}
		}
	}
}

void deviceGpu_3D::adjoint(const bool add, std::shared_ptr<float2DReg> signalReg, const std::shared_ptr<float2DReg> signalIrreg) const {

	/* ADJOINT: Go from IRREGULAR grid -> REGULAR grid */
	if (!add) signalReg->scale(0.0);
	std::shared_ptr<float2D> d = signalIrreg->_mat;
	std::shared_ptr<float2D> m = signalReg->_mat;

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over acquisition devices' positions
		for (int iFilter = 0; iFilter < _nFilter3d; iFilter++){ // Loop over neighboring points on regular grid
			int i1 = iDevice * _nFilter3d + iFilter; // Grid point index
			int i2 = _indexMap.find(_gridPointIndex[i1])->second; // Get trace number for signalReg
			for (int it = 0; it < _nt; it++){
				(*m)[i2][it] += _weight[i1] * (*d)[iDevice][it];
			}
		}
	}
}

void deviceGpu_3D::checkOutOfBounds(const std::shared_ptr<float1DReg> zCoord, const std::shared_ptr<float1DReg> xCoord, const std::shared_ptr<float1DReg> yCoord){

	int nDevice = zCoord->getHyper()->getAxis(1).n;
	_nzSmall = _nz - 2 * _fat - _zPadMinus - _zPadPlus;
	_nxSmall = _nx - 2 * _fat - _xPadMinus - _xPadPlus;
	_nySmall = _ny - 2 * _fat - 2 * _yPad;

	float zMin = _oz + (_fat + _zPadMinus) * _dz;
	float xMin = _ox + (_fat + _xPadMinus) * _dx;
	float yMin = _oy + (_fat + _yPad) * _dy;

	float zMax = zMin + (_nzSmall - 1) * _dz;
	float xMax = xMin + (_nxSmall - 1) * _dx;
    float yMax = yMin + (_nySmall - 1) * _dy;

	for (int iDevice = 0; iDevice < nDevice; iDevice++){
		if ( (*zCoord->_mat)[iDevice] - zMax > _errorTolerance || (*zCoord->_mat)[iDevice] - zMin < -_errorTolerance || (*xCoord->_mat)[iDevice] - xMax > _errorTolerance || (*xCoord->_mat)[iDevice] - xMin < -_errorTolerance || (*yCoord->_mat)[iDevice] - yMax > _errorTolerance || (*yCoord->_mat)[iDevice] - yMin < -_errorTolerance ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of the acquisition devices is out of bounds ****" << std::endl;
			std::cout << "iDevice = " << iDevice << std::endl;
			std::cout << "zMin = " << zMin << std::endl;
			std::cout << "zMax = " << zMax << std::endl;
			std::cout << "xMin = " << xMin << std::endl;
			std::cout << "xMax = " << xMax << std::endl;
			std::cout << "yMin = " << yMin << std::endl;
			std::cout << "yMax = " << yMax << std::endl;
			std::cout << "zCoord = " << (*zCoord->_mat)[iDevice] << std::endl;
			std::cout << "xCoord = " << (*xCoord->_mat)[iDevice] << std::endl;
			std::cout << "yCoord = " << (*yCoord->_mat)[iDevice] << std::endl;
			std::cout << "(*zCoord->_mat)[iDevice] - zMin = " << (*zCoord->_mat)[iDevice] - zMin << std::endl;
			std::cout << "-_errorTolerance = " << -_errorTolerance << std::endl;
			std::cout << "-------------------------------" << std::endl;
			throw std::runtime_error("");
		}
	}
}

void deviceGpu_3D::checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const int &nyDevice, const int &oyDevice, const int &dyDevice){

	_nzSmall = _nz - 2 * _fat - _zPadMinus - _zPadPlus;
	_nxSmall = _nx - 2 * _fat - _xPadMinus - _xPadPlus;
	_nySmall = _ny - 2 * _fat - 2 * _yPad;

	// ozDevice, oxDevice and oyDevice are the indices on the grid (laready shifted by fat + padMinus)
	int ozDeviceNoShift = ozDevice - _fat - _zPadMinus + 1;
	int oxDeviceNoShift = oxDevice - _fat - _xPadMinus + 1;
	int oyDeviceNoShift = oyDevice - _fat - _yPad + 1;

	int zIntMax = ozDeviceNoShift + (nzDevice - 1) * dzDevice;
	int xIntMax = oxDeviceNoShift + (nxDevice - 1) * dxDevice;
	int yIntMax = oyDeviceNoShift + (nyDevice - 1) * dyDevice;

	// std::cout << "ozDevice = " << ozDevice << std::endl;
	// std::cout << "oxDevice = " << oxDevice << std::endl;
	// std::cout << "oyDevice = " << oyDevice << std::endl;
	// std::cout << "zIntMax = " << zIntMax << std::endl;
	// std::cout << "xIntMax = " << xIntMax << std::endl;
	// std::cout << "yIntMax = " << yIntMax << std::endl;
	// std::cout << "nzSmall = " << _nzSmall << std::endl;
	// std::cout << "nxSmall = " << _nxSmall << std::endl;
	// std::cout << "nySmall = " << _nySmall << std::endl;

	// if ( (zIntMax > _nzSmall) || (xIntMax > _nxSmall) || (yIntMax > _nySmall) || (ozDevice < 0) || (oxDevice < 0) || (oyDevice < 0) ){
	// 	std::cout << "**** ERROR [deviceGpu_3D]: One of the acquisition devices is out of bounds ****" << std::endl;
	// 	throw std::runtime_error("");
	// }

	if ( (zIntMax > _nzSmall) || (ozDevice < 0) ){
		std::cout << "zIntMax = " << zIntMax << std::endl;
		std::cout << "_nzSmall = " << _nzSmall << std::endl;
		std::cout << "ozDevice = " << ozDevice << std::endl;
		std::cout << "**** ERROR [deviceGpu_3D]: One of the acquisition devices is out of bounds in the z-direction ****" << std::endl;
		throw std::runtime_error("");
	}

	if ( (xIntMax > _nxSmall) || (oxDevice < 0) ){
		std::cout << "**** ERROR [deviceGpu_3D]: One of the acquisition devices is out of bounds in the x-direction ****" << std::endl;
		std::cout << "xIntMax = " << xIntMax << std::endl;
		std::cout << "_nxSmall = " << _nxSmall << std::endl;
		std::cout << "oxDevice = " << oxDevice << std::endl;
		throw std::runtime_error("");
	}

	if ( (yIntMax > _nySmall) || (oyDevice < 0) ){
		std::cout << "**** ERROR [deviceGpu_3D]: One of the acquisition devices is out of bounds in the y-direction ****" << std::endl;
		throw std::runtime_error("");
	}

}

// Compute weights for linear interpolation
void deviceGpu_3D::calcLinearWeights(){

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {

		// Find the 8 neighboring points for all devices and compute the weights for the spatial interpolation
		int i1 = iDevice * _nFilter3d;
		float wz = ( (*_zCoord->_mat)[iDevice] - _oz ) / _dz;
		float wx = ( (*_xCoord->_mat)[iDevice] - _ox ) / _dx;
        float wy = ( (*_yCoord->_mat)[iDevice] - _oy ) / _dy;
		int zReg = wz; // z-coordinate on regular grid
		wz = wz - zReg;
		wz = 1.0 - wz;
		int xReg = wx; // x-coordinate on regular grid
		wx = wx - xReg;
		wx = 1.0 - wx;
        int yReg = wy; // y-coordinate on regular grid
        wy = wy - yReg;
        wy = 1.0 - wy;

		// Check for the y-axis
		if ( (yReg < _fat + _yPad) || ( (yReg + 1 >= _ny - _fat - _yPad) && (wy < 1.0-_errorTolerance) ) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the y-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}
		// Check for the x-axis

		//////////////////////////// Debug shit ////////////////////////////////
		// std::cout << "iDevice = " << iDevice << std::endl;
		// std::cout << "wx = " << wx << std::endl;
		// std::cout << "xReg = " << xReg << std::endl;
		// std::cout << "_nx - _fat - _xPadPlus = " << _nx - _fat - _xPadPlus << std::endl;
		////////////////////////////////////////////////////////////////////////

		if ( (xReg < _fat + _xPadMinus) || ( (xReg + 1 >= _nx - _fat - _xPadPlus) && (wx < 1.0-_errorTolerance) ) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the x-axis is out of bounds ****" << std::endl;

			////////////////////////// Debug shit //////////////////////////////
			// if ( (*_xCoord->_mat)[iDevice] < _ox){
			// 	std::cout << "(*_xCoord->_mat)[iDevice] - _ox" << (*_xCoord->_mat)[iDevice] < _ox << std::end;
			// 	std::cout << "Device out of bounds" << std::end;
			// }

			std::cout << "iDevice = " << iDevice << std::endl;
			std::cout << "(*_xCoord->_mat)[iDevice] = " << (*_xCoord->_mat)[iDevice] << std::endl;
			float wxBefore = ( (*_xCoord->_mat)[iDevice] - _ox ) / _dx;
			std::cout << "wx before = " << wxBefore << std::endl;
			int xRegTest = wxBefore;
			std::cout << "xRegTest = " << xRegTest << std::endl;
			int wxAfter = wxBefore - xRegTest;
			std::cout << "wxAfter = " << wxAfter << std::endl;
			int wxFinal = 1.0 - wxAfter;
			std::cout << "wxFinal = " << wxFinal << std::endl;
			std::cout << "_fat = " << _fat << std::endl;
			std::cout << "_xPadMinus = " << _xPadMinus << std::endl;
			std::cout << "xRegTest = " << xRegTest << std::endl;
			std::cout << "_fat + _xPadMinus = " << _fat + _xPadMinus << std::endl;
			// std::cout << "_nx = " << _nx << std::endl;
			// std::cout << "_xPadPlus = " << _xPadPlus << std::endl;
			// std::cout << "_nx - _fat - _xPadPlus = " << _nx - _fat - _xPadPlus << std::endl;
			////////////////////////////////////////////////////////////////////

			throw std::runtime_error("");
		}
		// Check for the z-axis
		if ( (zReg < _fat + _zPadMinus) || ( (zReg + 1 >= _nz - _fat - _zPadPlus) && (wz < 1.0-_errorTolerance) ) ) {
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the z-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}

		// Top front left
		_gridPointIndex[i1] = yReg * _nz * _nx + xReg * _nz + zReg; // Index of this point for a 1D array representation
		_weight[i1] = wz * wx * wy;

		// Bottom front left
		_gridPointIndex[i1+1] = _gridPointIndex[i1] + 1;
		_weight[i1+1] = (1.0 - wz) * wx * wy;

		// Top front right
		_gridPointIndex[i1+2] = _gridPointIndex[i1] + _nz;
		_weight[i1+2] = wz * (1.0 - wx) * wy;

		// Bottom front right
		_gridPointIndex[i1+3] = _gridPointIndex[i1] + _nz + 1;
		_weight[i1+3] = (1.0 - wz) * (1.0 - wx) * wy;

        // Top back left
        _gridPointIndex[i1+4] = _gridPointIndex[i1] + _nz *_nx;
        _weight[i1+4] = wz * wx * (1.0 - wy);

        // Bottom back left
        _gridPointIndex[i1+5] = _gridPointIndex[i1] + _nz *_nx + 1;
        _weight[i1+5] = (1.0 - wz) * wx * (1.0 - wy);

        // Top back right
        _gridPointIndex[i1+6] = _gridPointIndex[i1] + _nz *_nx + _nz;
        _weight[i1+6] = wz * (1.0 - wx) * (1.0 - wy);

        // Bottom back right
        _gridPointIndex[i1+7] = _gridPointIndex[i1] + _nz *_nx + _nz + 1;
        _weight[i1+7] = (1.0 - wz) * (1.0 - wx) * (1.0 - wy);

		// Case where we use a dipole or the seismic device
		if (_dipole == 1){

			// Find the 8 neighboring points for all devices dipole points and compute the weights for the spatial interpolation
			float wzDipole = ( (*_zCoord->_mat)[iDevice] + _zDipoleShift - _oz ) / _dz;
			float wxDipole = ( (*_xCoord->_mat)[iDevice] + _xDipoleShift - _ox ) / _dx;
            float wyDipole = ( (*_yCoord->_mat)[iDevice] + _yDipoleShift - _oy ) / _dz;
			int zRegDipole = wzDipole; // z-coordinate on regular grid
			wzDipole = wzDipole - zRegDipole;
			wzDipole = 1.0 - wzDipole;
			int xRegDipole = wxDipole; // x-coordinate on regular grid
			wxDipole = wxDipole - xRegDipole;
			wxDipole = 1.0 - wxDipole;
            int yRegDipole = wyDipole; // y-coordinate on regular grid
			wyDipole = wyDipole - yRegDipole;
			wyDipole = 1.0 - wyDipole;

			// Check for the y-axis
			if ( (yRegDipole < _fat + _yPad) || (yRegDipole + 1 >= _ny - _fat - _yPad) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the y-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}
			// Check for the x-axis
			if ( (xRegDipole < _fat + _xPadMinus) || (xRegDipole + 1 >= _nx - _fat - _xPadPlus) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the x-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}
			// Check for the z-axis
			if ( (zRegDipole < _fat + _zPadMinus) || (zRegDipole + 1 >= _nz - _fat - _zPadPlus) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the z-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}

			// Top front left (dipole point)
			_gridPointIndex[i1+8] = yRegDipole * _nz * _nx + xRegDipole * _nz + zRegDipole; // Index of this point for a 1D array representation
			_weight[i1+8] = (-1.0) * wzDipole * wxDipole * wyDipole;

			// Bottom front left (dipole point)
			_gridPointIndex[i1+9] = _gridPointIndex[i1+8] + 1;
			_weight[i1+9] = (-1.0) * (1.0 - wzDipole) * wxDipole * wyDipole;

			// Top front right (dipole point)
			_gridPointIndex[i1+10] = _gridPointIndex[i1+8] + _nz;
			_weight[i1+10] = (-1.0) * wzDipole * (1.0 - wxDipole) * wyDipole;

			// Bottom right (dipole point)
			_gridPointIndex[i1+11] = _gridPointIndex[i1+8] + _nz + 1;
			_weight[i1+11] = (-1.0) * (1.0 - wzDipole) * (1.0 - wxDipole) * wyDipole;

            // Top back left (dipole point)
            _gridPointIndex[i1+12] = _gridPointIndex[i1+8] + _nz * _nx;
			_weight[i1+12] = (-1.0) * wzDipole * wxDipole * (1.0 - wyDipole);

            // Bottom back left (dipole point)
            _gridPointIndex[i1+13] = _gridPointIndex[i1+8] + _nz * _nx + 1;
			_weight[i1+13] = (-1.0) * (1.0 - wzDipole) * wxDipole * (1.0 - wyDipole);

            // Top back right (dipole point)
            _gridPointIndex[i1+14] = _gridPointIndex[i1+8] + _nz * _nx + _nz;
			_weight[i1+14] = (-1.0) * wzDipole * (1.0 - wxDipole) * (1.0 - wyDipole);

            // Bottom back right (dipole point)
            _gridPointIndex[i1+15] = _gridPointIndex[i1+8] + _nz * _nx + _nz + 1;
			_weight[i1+15] = (-1.0) * (1.0 - wzDipole) * (1.0 - wxDipole) * (1.0 - wyDipole);

		}
	}
}

// Compute weights for sinc interpolation
void deviceGpu_3D::calcSincWeights(){

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {

		int i1 = iDevice * _nFilter3d;

		// Compute position of the acquisition device [km]
		float zIrreg = (*_zCoord->_mat)[iDevice];
		float xIrreg = (*_xCoord->_mat)[iDevice];
		float yIrreg = (*_yCoord->_mat)[iDevice];
		float zReg = (zIrreg - _oz) / _dz;
		float xReg = (xIrreg - _ox) / _dx;
		float yReg = (yIrreg - _oy) / _dy;

		// Index of top left grid point closest to the acquisition device
		int zRegInt = zReg;
		int xRegInt = xReg;
		int yRegInt = yReg;

		// Check that none of the points used in the interpolation are out of bounds
		if ( (yRegInt-_hFilter1d+1 < 0) || (yRegInt+_hFilter1d+1 < _ny) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation on the y-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}
		if ( (xRegInt-_hFilter1d+1 < 0) || (xRegInt+_hFilter1d+1 < _nx) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation on the x-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}
		if ( (zRegInt-_hFilter1d+1 < 0) || (zRegInt+_hFilter1d+1 < _nz) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation on the z-axis is out of bounds ****" << std::endl;
			throw std::runtime_error("");
		}

		// Loop over grid points involved in the interpolation
		for (int iy = 0; iy < _nFilter1d; iy++){
			for (int ix = 0; ix < _nFilter1d; ix++){
				for (int iz = 0; iz < _nFilter1d; iz++){

					// Compute grid point position
					float yCur = (yRegInt+iy-_hFilter1d+1) * _dy + _oy;
					float xCur = (xRegInt+ix-_hFilter1d+1) * _dx + _ox;
					float zCur = (zRegInt+iz-_hFilter1d+1) * _dz + _oz;

					// Compute argument for the sinc function
					float wy = (yIrreg-yCur)/_dy;
					float wx = (xIrreg-xCur)/_dx;
					float wz = (zIrreg-zCur)/_dz;

					// Compute global index of grid point used in the interpolation (on the main FD grid)
					int iPointInterp = _nz*_nx*(yRegInt+iy-_hFilter1d+1) + _nz*(xRegInt+ix-_hFilter1d+1) + (zRegInt+iz-_hFilter1d+1);

					// Compute index in the array that contains the positions of the grid points involved in the interpolation
					int iGridPointIndex = i1+iy*_nFilter1d*_nFilter1d+ix*_nFilter1d+iz;

					// Compute global index
					_gridPointIndex[iGridPointIndex] = iPointInterp;

					// Compute weight associated with that point
					_weight[iGridPointIndex] = boost::math::sinc_pi(M_PI*wz)*boost::math::sinc_pi(M_PI*wx)*boost::math::sinc_pi(M_PI*wy);

				}
			}
		}

		if (_dipole == 1){

			int i1Dipole = i1 + _nFilter3dDipole;

			// Compute position of the acquisition device [km] for the second pole
			float zIrregDipole = (*_zCoord->_mat)[iDevice]+_zDipoleShift;
			float xIrregDipole = (*_xCoord->_mat)[iDevice]+_xDipoleShift;
			float yIrregDipole = (*_yCoord->_mat)[iDevice]+_yDipoleShift;
			float zRegDipole = (zIrregDipole - _oz) / _dz;
			float xRegDipole = (xIrregDipole - _ox) / _dx;
			float yRegDipole = (yIrregDipole - _oy) / _dy;

			// Index of top left grid point closest to the acquisition device (corner of the voxel where the device lies that has the smallest index)
			int zRegDipoleInt = zRegDipole;
			int xRegDipoleInt = xRegDipole;
			int yRegDipoleInt = yRegDipole;

			// Check that none of the points used in the interpolation are out of bounds
			if ( (yRegDipoleInt-_hFilter1d+1 < 0) || (yRegDipoleInt+_hFilter1d+1 < _ny) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation for the negative dipole on the y-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}
			if ( (xRegDipoleInt-_hFilter1d+1 < 0) || (xRegDipoleInt+_hFilter1d+1 < _nx) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation for the negative dipole on the x-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}
			if ( (zRegDipoleInt-_hFilter1d+1 < 0) || (zRegDipoleInt+_hFilter1d+1 < _nz) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation for the negative dipole on the z-axis is out of bounds ****" << std::endl;
				throw std::runtime_error("");
			}

			// Loop over grid points involved in the interpolation
			for (int iy = 0; iy < _nFilter1d; iy++){
				for (int ix = 0; ix < _nFilter1d; ix++){
					for (int iz = 0; iz < _nFilter1d; iz++){

						// Compute grid point position
						float yCurDipole = (yRegDipoleInt+iy-_hFilter1d+1) * _dy + _oy;
						float xCurDipole = (xRegDipoleInt+ix-_hFilter1d+1) * _dx + _ox;
						float zCurDipole = (zRegDipoleInt+iz-_hFilter1d+1) * _dz + _oz;

						// Compute argument for the sinc function
						float wyDipole = (yIrregDipole-yCurDipole)/_dy;
						float wxDipole = (xIrregDipole-xCurDipole)/_dx;
						float wzDipole = (zIrregDipole-zCurDipole)/_dz;

						// Compute global index of grid point used in the interpolation (on the main FD grid) for the other pole
						int iPointInterpDipole = _nz*_nx*(yRegDipoleInt+iy-_hFilter1d+1) + _nz*(xRegDipoleInt+ix-_hFilter1d+1) + (zRegDipoleInt+iz-_hFilter1d+1);

						// Compute index in the array that contains the positions of the grid points (non-unique) involved in the interpolation
						int iGridPointIndexDipole = i1Dipole+iy*_nFilter1d*_nFilter1d+ix*_nFilter1d+iz;

						// Compute global index
						_gridPointIndex[iGridPointIndexDipole] = iPointInterpDipole;

						// Compute weight associated with that point
						_weight[iGridPointIndexDipole] = boost::math::sinc_pi(M_PI*wzDipole)*boost::math::sinc_pi(M_PI*wxDipole)*boost::math::sinc_pi(M_PI*wyDipole);

					}
				}
			}

		}

	}
}

void deviceGpu_3D::printRegPosUnique(){
	std::cout << "Size unique = " << getSizePosUnique() << std::endl;
	std::cout << "getNDeviceIrreg = " << getNDeviceIrreg() << std::endl;
	std::cout << "getSizePosUnique = " << getSizePosUnique() << std::endl;
	for (int iDevice=0; iDevice<getSizePosUnique(); iDevice++){
		std::cout << "Position for device #" << iDevice << " = " << _gridPointIndexUnique[iDevice] << std::endl;
	}
}
