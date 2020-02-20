#include <double1DReg.h>
#include <double2DReg.h>
#include <double3DReg.h>
#include <iostream>
#include "deviceGpu_3D.h"
#include <boost/math/special_functions/sinc.hpp>
#include <boost/math/special_functions/cos_pi.hpp>
#include <boost/math/distributions/normal.hpp>
#include <math.h>
#include <vector>

// Constructor #1 -- OOnly for irregular geometry
deviceGpu_3D::deviceGpu_3D(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord, const std::shared_ptr<double1DReg> yCoord, const std::shared_ptr<double3DReg> vel, int &nt, int dipole, double zDipoleShift, double xDipoleShift, double yDipoleShift, std::string interpMethod, int hFilter1d){

	// Get domain dimensions
	_vel = vel;
	_oz = vel->getHyper()->getAxis(1).o;
	_dz = vel->getHyper()->getAxis(1).d;
	_nz = vel->getHyper()->getAxis(1).n;
	_ox = vel->getHyper()->getAxis(2).o;
	_dx = vel->getHyper()->getAxis(2).d;
	_nx = vel->getHyper()->getAxis(2).n;
	_oy = vel->getHyper()->getAxis(3).o;
	_dy = vel->getHyper()->getAxis(3).d;
	_ny = vel->getHyper()->getAxis(3).n;

	// Get positions of aquisition devices + other parameters
	_zCoord = zCoord;
	_xCoord = xCoord;
    _yCoord = yCoord;

	_dipole = dipole; // Default value is 0
	_zDipoleShift = zDipoleShift; // Default value is 0
	_xDipoleShift = xDipoleShift; // Default value is 0
	_yDipoleShift = yDipoleShift; // Default value is 0
	checkOutOfBounds(_zCoord, _xCoord, _yCoord, _vel); // Check that none of the acquisition devices are out of bounds
	_hFilter1d = hFilter1d; // Half-length of the filter for each dimension. For sinc, the filter in each direction z, x, y has the same length
	_interpMethod = interpMethod; // Default is linear
	_nDeviceIrreg = _zCoord->getHyper()->getAxis(1).n; // Nb of devices on irregular grid
	_nt = nt;

	// Check that for the linear interpolation case, the filter half-length is 1
	if (interpMethod == "linear" && _hFilter1d != 1){
		std::cout << "**** ERROR [deviceGpu_3D]: Half-length of interpolation filter should be set to 1 for linear interpolation ****" << std::endl;
		assert (1==2);
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
	_weight = new double[_nFilter3d*_nDeviceIrreg]; // Weights corresponding to the points on the grid stored in _gridPointIndex

	// Compute list of all grid points used in the interpolation
	if (interpMethod == "linear"){
		calcLinearWeights();
	} else if (interpMethod == "sinc"){
		calcSincWeights();
	} else {
		std::cerr << "**** ERROR [deviceGpu_3D]: Space interpolation method not supported ****" << std::endl;
	}
	// Convert the list -> Create a new list with unique indices of the regular grid points involved in the interpolation
	convertIrregToReg();

}

// Constructor #2 -- Only for regular geometry
// This contructor only allows linear interpolation
deviceGpu_3D::deviceGpu_3D(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const int &nyDevice, const int &oyDevice, const int &dyDevice, const std::shared_ptr<double3DReg> vel, int &nt, int dipole, int zDipoleShift, int xDipoleShift, int yDipoleShift, std::string interpMethod, int hFilter1d){

	_vel = vel;
	_nt = nt;
	_dipole = dipole;
	_zDipoleShift = zDipoleShift;
	_xDipoleShift = xDipoleShift;
	_yDipoleShift = yDipoleShift;
	_hFilter1d = hFilter1d; // Half-length of the filter for each dimension (for sinc, the filter in each direction has the same length)
	_interpMethod = interpMethod;
	checkOutOfBounds(nzDevice, ozDevice, dzDevice, nxDevice, oxDevice, dxDevice, nyDevice, oyDevice, dyDevice, vel);
	_nDeviceIrreg = nzDevice * nxDevice * nyDevice; // Nb of devices on irregular grid
	int _nz = vel->getHyper()->getAxis(1).n;
    int _nx = vel->getHyper()->getAxis(2).n;

	// Debug
    // int ny = vel->getHyper()->getAxis(3).n;
	// std::cout << "nz" << _nz << std::endl;
	// std::cout << "nx" << _nx << std::endl;
	// std::cout << "ny" << ny << std::endl;
	// std::cout << "nzDevice" << nzDevice << std::endl;
	// std::cout << "ozDevice" << ozDevice << std::endl;
	// std::cout << "dzDevice" << dzDevice << std::endl;
	// std::cout << "nxDevice" << nxDevice << std::endl;
	// std::cout << "oxDevice" << oxDevice << std::endl;
	// std::cout << "dxDevice" << dxDevice << std::endl;
	// std::cout << "nyDevice" << nyDevice << std::endl;
	// std::cout << "oyDevice" << oyDevice << std::endl;
	// std::cout << "dyDevice" << dyDevice << std::endl;
	// End debug

	// Check that the user is not asking for sinc interpolation
	if (interpMethod != "linear"){
		std::cout << "**** ERROR [deviceGpu_3D]: The constructor used only supports linear spatial interpolation ****" << std::endl;
		assert (1==2);
	}
	// Check that for the linear interpolation case, the filter half-length is 1
	if (_hFilter1d != 1){
		std::cout << "**** ERROR [deviceGpu_3D]: Half-length of interpolation filter should be 1 for linear interpolation ****" << std::endl;
		assert (1==2);
	}

	// Compute the total number of points on the grid for each axis that will be involved in the interpolation
	_nFilter1d = 2*_hFilter1d;
	_nFilter3d = _nFilter1d*_nFilter1d*_nFilter1d; // Total number of points involved in the interpolation
	if (_dipole == 1){_nFilter3d=_nFilter3d*2;} // Dipole case
	_gridPointIndex = new int[_nFilter3d*_nDeviceIrreg]; // Index of all the neighboring points of each device (non-unique) on the regular "1D" grid
	_weight = new double[_nFilter3d*_nDeviceIrreg]; // Weights for spatial interpolation

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

void deviceGpu_3D::convertIrregToReg() {

	/* (1) Create map where:
		- Key = excited grid point index (points are unique)
		- Value = signal trace number
		(2) Create a vector containing the indices of the excited grid points
	*/

	_nDeviceReg = 0; // Initialize the number of regular devices to zero
	_gridPointIndexUnique.clear(); // Initialize to empty vector

	for (long long iDevice = 0; iDevice < _nDeviceIrreg; iDevice++){ // Loop over gridPointIndex array
		for (long long iFilter = 0; iFilter < _nFilter3d; iFilter++){
			long long i1 = iDevice * _nFilter3d + iFilter;

			// If the grid point is not already in the list
			if (_indexMap.count(_gridPointIndex[i1]) == 0) {
				_nDeviceReg++; // Increment the number of (unique) grid points excited by the signal
				_indexMap[_gridPointIndex[i1]] = _nDeviceReg - 1; // Add the pair to the map
				_gridPointIndexUnique.push_back(_gridPointIndex[i1]); // Append vector containing all unique grid point index
			}
		}
	}
}

void deviceGpu_3D::forward(const bool add, const std::shared_ptr<double2DReg> signalReg, std::shared_ptr<double2DReg> signalIrreg) const {

	/* FORWARD: Go from REGULAR grid -> IRREGULAR grid */
	if (!add) signalIrreg->scale(0.0);

	std::shared_ptr<double2D> d = signalIrreg->_mat;
	std::shared_ptr<double2D> m = signalReg->_mat;

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

void deviceGpu_3D::adjoint(const bool add, std::shared_ptr<double2DReg> signalReg, const std::shared_ptr<double2DReg> signalIrreg) const {

	/* ADJOINT: Go from IRREGULAR grid -> REGULAR grid */
	if (!add) signalReg->scale(0.0);
	std::shared_ptr<double2D> d = signalIrreg->_mat;
	std::shared_ptr<double2D> m = signalReg->_mat;

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

void deviceGpu_3D::checkOutOfBounds(const std::shared_ptr<double1DReg> zCoord, const std::shared_ptr<double1DReg> xCoord, const std::shared_ptr<double1DReg> yCoord, const std::shared_ptr<double3DReg> vel){

	int nDevice = zCoord->getHyper()->getAxis(1).n;
	double zMax = vel->getHyper()->getAxis(1).o + (vel->getHyper()->getAxis(1).n - 1) * vel->getHyper()->getAxis(1).d;
	double xMax = vel->getHyper()->getAxis(2).o + (vel->getHyper()->getAxis(2).n - 1) * vel->getHyper()->getAxis(2).d;
    double yMax = vel->getHyper()->getAxis(3).o + (vel->getHyper()->getAxis(3).n - 1) * vel->getHyper()->getAxis(3).d;

	for (int iDevice = 0; iDevice < nDevice; iDevice++){
		if ( ((*zCoord->_mat)[iDevice] >= zMax) || ((*xCoord->_mat)[iDevice] >= xMax) || ((*yCoord->_mat)[iDevice] >= yMax) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of the acquisition devices is out of bounds ****" << std::endl;
			assert (1==2);
		}
	}
}

void deviceGpu_3D::checkOutOfBounds(const int &nzDevice, const int &ozDevice, const int &dzDevice , const int &nxDevice, const int &oxDevice, const int &dxDevice, const int &nyDevice, const int &oyDevice, const int &dyDevice, const std::shared_ptr<double3DReg> vel){

	double zIntMax = ozDevice + (nzDevice - 1) * dzDevice;
	double xIntMax = oxDevice + (nxDevice - 1) * dxDevice;
	double yIntMax = oyDevice + (nyDevice - 1) * dyDevice;
	if ( (zIntMax >= _vel->getHyper()->getAxis(1).n) || (xIntMax >= _vel->getHyper()->getAxis(2).n) || (yIntMax >= _vel->getHyper()->getAxis(3).n) ){
		std::cout << "**** ERROR [deviceGpu_3D]: One of the device is out of bounds ****" << std::endl;
		assert (1==2);
	}
}

// Compute weights for linear interpolation
void deviceGpu_3D::calcLinearWeights(){

	for (int iDevice = 0; iDevice < _nDeviceIrreg; iDevice++) {

		// Find the 8 neighboring points for all devices and compute the weights for the spatial interpolation
		int i1 = iDevice * _nFilter3d;
		double wz = ( (*_zCoord->_mat)[iDevice] - _vel->getHyper()->getAxis(1).o ) / _vel->getHyper()->getAxis(1).d;
		double wx = ( (*_xCoord->_mat)[iDevice] - _vel->getHyper()->getAxis(2).o ) / _vel->getHyper()->getAxis(2).d;
        double wy = ( (*_yCoord->_mat)[iDevice] - _vel->getHyper()->getAxis(3).o ) / _vel->getHyper()->getAxis(3).d;
		int zReg = wz; // z-coordinate on regular grid
		wz = wz - zReg;
		wz = 1.0 - wz;
		int xReg = wx; // x-coordinate on regular grid
		wx = wx - xReg;
		wx = 1.0 - wx;
        int yReg = wy; // y-coordinate on regular grid
        wy = wy - yReg;
        wy = 1.0 - wy;

		// Check that none of the points used in the interpolation are out of bounds
		if ( (yReg-_hFilter1d+1 < 0) || (yReg+_hFilter1d+1 < _ny) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the y-axis is out of bounds ****" << std::endl;
			assert (1==2);
		}
		if ( (xReg-_hFilter1d+1 < 0) || (xReg+_hFilter1d+1 < _nx) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the x-axis is out of bounds ****" << std::endl;
			assert (1==2);
		}
		if ( (zReg-_hFilter1d+1 < 0) || (zReg+_hFilter1d+1 < _nz) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation on the z-axis is out of bounds ****" << std::endl;
			assert (1==2);
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
			double wzDipole = ( (*_zCoord->_mat)[iDevice] + _zDipoleShift - _vel->getHyper()->getAxis(1).o ) / _vel->getHyper()->getAxis(1).d;
			double wxDipole = ( (*_xCoord->_mat)[iDevice] + _xDipoleShift - _vel->getHyper()->getAxis(2).o ) / _vel->getHyper()->getAxis(2).d;
            double wyDipole = ( (*_yCoord->_mat)[iDevice] + _yDipoleShift - _vel->getHyper()->getAxis(3).o ) / _vel->getHyper()->getAxis(3).d;
			int zRegDipole = wzDipole; // z-coordinate on regular grid
			wzDipole = wzDipole - zRegDipole;
			wzDipole = 1.0 - wzDipole;
			int xRegDipole = wxDipole; // x-coordinate on regular grid
			wxDipole = wxDipole - xRegDipole;
			wxDipole = 1.0 - wxDipole;
            int yRegDipole = wyDipole; // y-coordinate on regular grid
			wyDipole = wyDipole - yRegDipole;
			wyDipole = 1.0 - wyDipole;

			// Check that none of the points used in the interpolation are out of bounds
			if ( (yRegDipole-_hFilter1d+1 < 0) || (yRegDipole+_hFilter1d+1 < _ny) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation for the negative pole on the y-axis is out of bounds ****" << std::endl;
				assert (1==2);
			}
			if ( (xRegDipole-_hFilter1d+1 < 0) || (xRegDipole+_hFilter1d+1 < _nx) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation for the negative pole on the x-axis is out of bounds ****" << std::endl;
				assert (1==2);
			}
			if ( (zRegDipole-_hFilter1d+1 < 0) || (zRegDipole+_hFilter1d+1 < _nz) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the linear interpolation for the negative pole on the z-axis is out of bounds ****" << std::endl;
				assert (1==2);
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
		double zIrreg = (*_zCoord->_mat)[iDevice];
		double xIrreg = (*_xCoord->_mat)[iDevice];
		double yIrreg = (*_yCoord->_mat)[iDevice];
		double zReg = (zIrreg - _oz) / _dz;
		double xReg = (xIrreg - _ox) / _dx;
		double yReg = (yIrreg - _oy) / _dy;

		// Index of top left grid point closest to the acquisition device
		int zRegInt = zReg;
		int xRegInt = xReg;
		int yRegInt = yReg;

		// Check that none of the points used in the interpolation are out of bounds
		if ( (yRegInt-_hFilter1d+1 < 0) || (yRegInt+_hFilter1d+1 < _ny) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation on the y-axis is out of bounds ****" << std::endl;
			assert (1==2);
		}
		if ( (xRegInt-_hFilter1d+1 < 0) || (xRegInt+_hFilter1d+1 < _nx) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation on the x-axis is out of bounds ****" << std::endl;
			assert (1==2);
		}
		if ( (zRegInt-_hFilter1d+1 < 0) || (zRegInt+_hFilter1d+1 < _nz) ){
			std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation on the z-axis is out of bounds ****" << std::endl;
			assert (1==2);
		}

		// Loop over grid points involved in the interpolation
		for (int iy = 0; iy < _nFilter1d; iy++){
			for (int ix = 0; ix < _nFilter1d; ix++){
				for (int iz = 0; iz < _nFilter1d; iz++){

					// Compute grid point position
					double yCur = (yRegInt+iy-_hFilter1d+1) * _dy + _oy;
					double xCur = (xRegInt+ix-_hFilter1d+1) * _dx + _ox;
					double zCur = (zRegInt+iz-_hFilter1d+1) * _dz + _oz;

					// Compute argument for the sinc function
					double wy = (yIrreg-yCur)/_dy;
					double wx = (xIrreg-xCur)/_dx;
					double wz = (zIrreg-zCur)/_dz;

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
			double zIrregDipole = (*_zCoord->_mat)[iDevice]+_zDipoleShift;
			double xIrregDipole = (*_xCoord->_mat)[iDevice]+_xDipoleShift;
			double yIrregDipole = (*_yCoord->_mat)[iDevice]+_yDipoleShift;
			double zRegDipole = (zIrregDipole - _oz) / _dz;
			double xRegDipole = (xIrregDipole - _ox) / _dx;
			double yRegDipole = (yIrregDipole - _oy) / _dy;

			// Index of top left grid point closest to the acquisition device (corner of the voxel where the device lies that has the smallest index)
			int zRegDipoleInt = zRegDipole;
			int xRegDipoleInt = xRegDipole;
			int yRegDipoleInt = yRegDipole;

			// Check that none of the points used in the interpolation are out of bounds
			if ( (yRegDipoleInt-_hFilter1d+1 < 0) || (yRegDipoleInt+_hFilter1d+1 < _ny) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation for the negative dipole on the y-axis is out of bounds ****" << std::endl;
				assert (1==2);
			}
			if ( (xRegDipoleInt-_hFilter1d+1 < 0) || (xRegDipoleInt+_hFilter1d+1 < _nx) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation for the negative dipole on the x-axis is out of bounds ****" << std::endl;
				assert (1==2);
			}
			if ( (zRegDipoleInt-_hFilter1d+1 < 0) || (zRegDipoleInt+_hFilter1d+1 < _nz) ){
				std::cout << "**** ERROR [deviceGpu_3D]: One of grid points used in the sinc interpolation for the negative dipole on the z-axis is out of bounds ****" << std::endl;
				assert (1==2);
			}

			// Loop over grid points involved in the interpolation
			for (int iy = 0; iy < _nFilter1d; iy++){
				for (int ix = 0; ix < _nFilter1d; ix++){
					for (int iz = 0; iz < _nFilter1d; iz++){

						// Compute grid point position
						double yCurDipole = (yRegDipoleInt+iy-_hFilter1d+1) * _dy + _oy;
						double xCurDipole = (xRegDipoleInt+ix-_hFilter1d+1) * _dx + _ox;
						double zCurDipole = (zRegDipoleInt+iz-_hFilter1d+1) * _dz + _oz;

						// Compute argument for the sinc function
						double wyDipole = (yIrregDipole-yCurDipole)/_dy;
						double wxDipole = (xIrregDipole-xCurDipole)/_dx;
						double wzDipole = (zIrregDipole-zCurDipole)/_dz;

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
