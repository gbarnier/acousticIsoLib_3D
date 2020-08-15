#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import dataTaperModule_3D
import sys

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Read model (seismic data that you wish to mute/taper)
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Initialize operator
	t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth,streamers=dataTaperModule_3D.dataTaperInit_3D(sys.argv)

	# Instanciate operator
	dataTaperOb=dataTaperModule_3D.datTaper_3D(model,model,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,model.getHyper(),time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth,streamers)

	# Get tapering mask
	maskFile=parObject.getString("mask","noMaskFile")
	if (maskFile != "noMaskFile"):
		taperMask=dataTaperOb.getTaperMask()
		genericIO.defaultIO.writeVector(maskFile,taperMask)

	# Write data
	data=SepVector.getSepVector(model.getHyper())
	dataTaperOb.forward(False,model,data)
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)
