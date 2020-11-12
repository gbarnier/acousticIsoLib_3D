#!/usr/bin/env python3
"""
generatePSF_3D.py model= zPadMinus= zPadPlus= xPadMinus= xPadPlus= yPad= fat= nPSF_= PSF_file=

Generate model vector containing point-spread functions given the provided number of PSFs per axis
"""
import numpy as np
import pyVector as pyVec
import sys
import genericIO


if __name__ == "__main__":
	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

	parObject=genericIO.io(params=sys.argv)
	# Read input and output files
	modelFile=parObject.getString("model")
	PSFile=parObject.getString("PSF_file")
	model = genericIO.defaultIO.getVector(modelFile, ndims=5).zero()
	nz = model.getHyper().getAxis(1).n
	nx = model.getHyper().getAxis(2).n
	ny = model.getHyper().getAxis(3).n
	nExt1 = model.getHyper().getAxis(4).n
	nExt2 = model.getHyper().getAxis(5).n
	# Getting padding to place PSF in the modeling area
	zPadMinus = parObject.getInt("zPadMinus")
	zPadPlus = parObject.getInt("zPadPlus")
	xPadMinus = parObject.getInt("xPadMinus")
	xPadPlus = parObject.getInt("xPadPlus")
	yPad = parObject.getInt("yPad")
	fat = parObject.getInt("fat")
	# Getting number of PSFs on each axis
	nPSF1 = parObject.getInt("nPSF1")
	nPSF2 = parObject.getInt("nPSF2")
	nPSF3 = parObject.getInt("nPSF3")
	nPSF4 = parObject.getInt("nPSF4",1)
	nPSF5 = parObject.getInt("nPSF5",1)

	if nPSF4 != 1 and nExt1 == 1 or nPSF5 != 1 and nExt2 == 1:
		raiseValueError("ERROR! nPSF4 and PSF5 must be 1 for non-extended images")
	if nPSF4 == 1 and nExt1 != 1 or nPSF5 == 1 and nExt2 != 1:
		raiseValueError("ERROR! nPSF4 and PSF5 must be different than 1 for extended images")
	if nPSF1 <= 0 or nPSF2 <= 0 or nPSF3 <= 0 or nPSF4 <= 0 or nPSF5 <= 0:
		raiseValueError("ERROR! nPSF must be positive")

	# Computing non-padded model size
	nz_nopad = nz - 2*fat - zPadMinus - zPadPlus
	nx_nopad = nx - 2*fat - xPadMinus - xPadPlus
	ny_nopad = ny - 2*fat - 2*yPad
	PSF1_pos = np.linspace(zPadMinus+fat, zPadMinus+fat+nz_nopad, nPSF1).astype(np.int)
	PSF2_pos = np.linspace(xPadMinus+fat, xPadMinus+fat+nx_nopad, nPSF2).astype(np.int)
	PSF3_pos = np.linspace(yPad+fat, yPad+fat+ny_nopad, nPSF3).astype(np.int)
	PSF4_pos = np.linspace(0, nExt1-1, nPSF4).astype(np.int)
	PSF5_pos = np.linspace(0, nExt2-1, nPSF5).astype(np.int)
	zz,xx,yy,ee1,ee2 = np.meshgrid(PSF1_pos,PSF2_pos,PSF3_pos,PSF4_pos,PSF5_pos,indexing='ij',sparse=True)
	model.getNdArray()[ee2,ee1,yy,xx,zz] = 1.0
	model.writeVec(PSFile)
