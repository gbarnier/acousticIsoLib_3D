#!/usr/bin/env python3
"""
interpPSF_3D.py model= zPadMinus= zPadPlus= xPadMinus= xPadPlus= yPad= fat= nPSF_= interpPSF=

Linear interpolation of the point-spread functions
"""
import numpy as np
import pyVector as pyVec
import sys
import genericIO
from scipy.interpolate import RegularGridInterpolator


if __name__ == "__main__":
	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

	parObject=genericIO.io(params=sys.argv)
	# Read input and output files
	modelFile=parObject.getString("model")
	interpFile=parObject.getString("interpPSF")
	model = genericIO.defaultIO.getVector(modelFile, ndims=5)
	modelNd = model.getNdArray()
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
	if nExt1 == 1 and nExt2 == 1:
		yy,xx,zz = np.meshgrid(PSF3_pos,PSF2_pos,PSF1_pos,indexing='ij',sparse=True)
		PSF_values = modelNd[0,0,yy,xx,zz]
		points = (PSF3_pos,PSF2_pos,PSF1_pos)
		yy1,xx1,zz1 = np.meshgrid(np.arange(ny),np.arange(nx),np.arange(nz),indexing='ij',sparse=True)
		modelMesh = (yy1,xx1,zz1)
	else:
		PSF4_pos = np.linspace(0, nExt1-1, nPSF4).astype(np.int)
		PSF5_pos = np.linspace(0, nExt2-1, nPSF5).astype(np.int)
		ee2,ee1,yy,xx,zz = np.meshgrid(PSF5_pos,PSF4_pos,PSF3_pos,PSF2_pos,PSF1_pos,indexing='ij',sparse=True)
		PSF_values = modelNd[ee2,ee1,yy,xx,zz]
		points = (PSF5_pos,PSF4_pos,PSF3_pos,PSF2_pos,PSF1_pos)
		ee21,ee11,yy1,xx1,zz1 = np.meshgrid(np.arange(nExt2),np.arange(nExt1),np.arange(ny),np.arange(nx),np.arange(nz),indexing='ij',sparse=True)
		modelMesh = (ee21,ee11,yy1,xx1,zz1)
	# Removing negative values
	PSF_values[PSF_values < 0.0] = 0.0

	f = RegularGridInterpolator(points, PSF_values, bounds_error=False)
	modelNd[:] = f(modelMesh)
	# Filling the padding
	modelNd[:,:,0:yPad+fat,:,:] = np.reshape(modelNd[:,:,yPad+fat,:,:],(nExt2,nExt1,1,nx,nz))
	modelNd[:,:,-(yPad+fat):,:,:] = np.reshape(modelNd[:,:,-(yPad+fat),:,:],(nExt2,nExt1,1,nx,nz))
	modelNd[:,:,:,0:xPadMinus+fat,:] = np.reshape(modelNd[:,:,:,xPadMinus+fat,:],(nExt2,nExt1,ny,1,nz))
	modelNd[:,:,:,-(xPadPlus+fat):,:] = np.reshape(modelNd[:,:,:,-(xPadPlus+fat),:],(nExt2,nExt1,ny,1,nz))
	modelNd[:,:,:,:,0:zPadMinus+fat] = np.reshape(modelNd[:,:,:,:,zPadMinus+fat],(nExt2,nExt1,ny,nx,1))
	modelNd[:,:,:,:,-(zPadPlus+fat):] = np.reshape(modelNd[:,:,:,:,-(zPadPlus+fat)],(nExt2,nExt1,ny,nx,1))
	model.writeVec(interpFile)
