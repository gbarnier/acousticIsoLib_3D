#include <iostream>
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"

using namespace SEP;

int main(int argc, char **argv) {

	// IO bullshit
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();

	// Model
	std::shared_ptr <genericRegFile> modelFile = io->getRegFile("model",usageIn);
 	std::shared_ptr<SEP::hypercube> modelHyper = modelFile->getHyper();
	std::shared_ptr<SEP::float3DReg> model(new SEP::float3DReg(modelHyper));
	modelFile->readFloatStream(model);

	// Model parameters
	long long nz = model->getHyper()->getAxis(1).n;
	long long nx = model->getHyper()->getAxis(2).n;
	long long ny = model->getHyper()->getAxis(3).n;

	// Parfile
	int zPad = par->getInt("zPad");
	int xPad = par->getInt("xPad");
    int yPad = par->getInt("yPad");
	int fat = par->getInt("fat", 4);
	int blockSize = par->getInt("blockSize", 16);
	int freeSurface = par->getInt("freeSurface", 0);

	// Compute size of zPadPlus
	int zPadPlus;
	long long nzNew;
	long long nzNewTotal;
	if (freeSurface == 1){
		long long nzTotal = zPad + nz;
		float ratioz = float(nzTotal) / float(blockSize);
		ratioz = ceilf(ratioz);
		long long nbBlockz = ratioz;
		zPadPlus = nbBlockz * blockSize - nz;
		nzNew = zPadPlus + nz;
		nzNewTotal = nzNew + 2*fat;
		zPad = 0;
	} else {
		long long nzTotal = zPad * 2 + nz;
		float ratioz = float(nzTotal) / float(blockSize);
		ratioz = ceilf(ratioz);
		long long nbBlockz = ratioz;
		zPadPlus = nbBlockz * blockSize - nz - zPad;
		nzNew = zPad + zPadPlus + nz;
		nzNewTotal = nzNew + 2*fat;
	}

	// Compute size of xPadPlus
	int xPadPlus;
	long long nxTotal = xPad * 2 + nx;
	float ratiox = float(nxTotal) / float(blockSize);
	ratiox = ceilf(ratiox);
	long long nbBlockx = ratiox;
	xPadPlus = nbBlockx * blockSize - nx - xPad;
	long long nxNew = xPad + xPadPlus + nx;
	long long nxNewTotal = nxNew + 2*fat;

	// Compute size on y-direction
	// No need to make it a multiple of the block size
	int yPadPlus;
	long long nyNew = 2 * yPad + ny;
	yPadPlus = yPad;
	long long nyNewTotal = nyNew + 2*fat;

	// Compute parameters
	float dz = modelHyper->getAxis(1).d;
	float oz = modelHyper->getAxis(1).o - (fat + zPad) * dz;
	float dx = modelHyper->getAxis(2).d;
	float ox = modelHyper->getAxis(2).o - (fat + xPad) * dx;
	float dy = modelHyper->getAxis(3).d;
	float oy = modelHyper->getAxis(3).o - (fat + yPad) * dy;

	// Data
	axis zAxis = axis(nzNewTotal, oz, dz);
	axis xAxis = axis(nxNewTotal, ox, dx);
	axis yAxis = axis(nyNewTotal, oy, dy);
 	std::shared_ptr<SEP::hypercube> dataHyper(new hypercube(zAxis, xAxis, yAxis));
 	std::shared_ptr<SEP::float3DReg> data(new SEP::float3DReg(dataHyper));
	std::shared_ptr <genericRegFile> dataFile = io->getRegFile("data",usageOut);
	dataFile->setHyper(dataHyper);
	dataFile->writeDescription();
	data->scale(0.0);

	// Central part on the y-axis
	for (long long iy=0; iy<ny; iy++){

		// Copy central part
		for (long long ix=0; ix<nx; ix++){
			for (long long iz=0; iz<nz; iz++){
				(*data->_mat)[iy+fat+yPad][ix+fat+xPad][iz+fat+zPad] = (*model->_mat)[iy][ix][iz];
			}
		}

		// Vertical direction
		for (long long ix=0; ix<nx; ix++){
			// Top central part
			for (long long iz=0; iz<zPad; iz++){
				(*data->_mat)[iy+fat+yPad][ix+fat+xPad][iz+fat] = (*model->_mat)[iy][ix][0];
			}
			// Bottom central part
			for (long long iz=0; iz<zPadPlus; iz++){
			(*data->_mat)[iy+fat+yPad][ix+fat+xPad][iz+fat+zPad+nz] = (*model->_mat)[iy][ix][nz-1];
			}
		}
		// Left part
		for (long long ix=0; ix<xPad; ix++){
			for (long long iz=0; iz<nzNew; iz++) {
				(*data->_mat)[iy+fat+yPad][ix+fat][iz+fat] = (*data->_mat)[iy+fat+yPad][xPad+fat][iz+fat];
			}
		}

		// Right part
		for (long long ix=0; ix<xPadPlus; ix++){
			for (long long iz=0; iz<nzNew; iz++){
				(*data->_mat)[iy+fat+yPad][ix+fat+nx+xPad][iz+fat] = (*data->_mat)[iy+fat+yPad][fat+xPad+nx-1][iz+fat];
			}
		}
	}

	// Padding on y-axis
	for (long long iy=0; iy<yPad; iy++){
		for (long long ix=fat; ix<nxNewTotal-fat; ix++){
			for (long long iz=fat; iz<nzNewTotal-fat; iz++){
				(*data->_mat)[iy+fat][ix][iz] = (*data->_mat)[fat+yPad][ix][iz]; // Front part
				(*data->_mat)[nyNewTotal-fat-yPad+iy][ix][iz] = (*data->_mat)[fat+yPad+ny-1][ix][iz]; // Back part
			}
		}
	}

	// Write model
	dataFile->writeFloatStream(data);

	// Display info
	std::cout << " " << std::endl;
	std::cout << "------------------------ Model padding program --------------------" << std::endl;
	std::cout << "Original nz = " << nz << " [samples]" << std::endl;
	std::cout << "Original nx = " << nx << " [samples]" << std::endl;
	std::cout << "Original ny = " << ny << " [samples]" << std::endl;
	std::cout << " " << std::endl;
	if (freeSurface == 1){
		std::cout << "zPadMinus = " << zPad << " [samples] => Model designed with a free surface at the top" << std::endl;
	} else {
		std::cout << "zPadMinus = " << zPad << " [samples]" << std::endl;
	}
	std::cout << "zPadPlus = " << zPadPlus << " [samples]" << std::endl;
	std::cout << "xPadMinus = " << xPad << " [samples]" << std::endl;
	std::cout << "xPadPlus = " << xPadPlus << " [samples]" << std::endl;
	std::cout << "yPadMinus = " << yPad << " [samples]" << std::endl;
	std::cout << "yPadPlus = " << yPad << " [samples]" << std::endl;
	std::cout << " " << std::endl;
	std::cout << "blockSize = " << blockSize << " [samples]" << std::endl;
	std::cout << "FAT = " << fat << " [samples]" << std::endl;
	std::cout << " " << std::endl;
	std::cout << "New nz = " << nzNewTotal << " [samples including padding and FAT]" << std::endl;
	std::cout << "New nx = " << nxNewTotal << " [samples including padding and FAT]" << std::endl;
	std::cout << "New ny = " << nyNewTotal << " [samples including padding and FAT]" << std::endl;
	std::cout << "-------------------------------------------------------------------" << std::endl;
	std::cout << " " << std::endl;
	return 0;

}
