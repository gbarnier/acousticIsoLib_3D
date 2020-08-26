#!/usr/bin/env python3
"""
Data tapering code
EG Optimzation (2020)
Ettore Biondi and Guillaume Barnier

USAGE EXAMPLE:
	dataTaperMain_3D.py model=rawData.H data=taperedData.H maskOffsetFile=maskOffset.H maskEndTraceFile=maskEndOfTrace.H maskTimeFile=maskTime.H receiverGeomFile=recGeom.H sourceGeomFile=sourceGeom.H offset=1 maxOffset=6.0 taperWidthOffset=0.5 offsetMuting=far time=1 t0=0.5 velMute=1.4 taperWidthTime=0.5 moveout=linear timeMuting=early taperEndTraceWidth=0.5

INPUT PARAMETERS:
	- model  = format: [nts,nRec,nShot] - string; Raw shot gathers [nts,nRec,nShot]

	- sourceGeomFile = [no default] - string; Source geometry file

	- receiverGeomFile = [no default] - string; Receiver geometry file

OUTPUT PARAMETERS:
	- data  = format: [nts,nRec,nShot] - string; Tapered shot gathers [nts,nRec,nShot]

	- maskOffsetFile = format: [nReceiver,nShot] - string; Mask for the offset muting

	- maskTimeFile = format: [nts,nReceiver,nShot] - string; Mask for the time muting

	- maskEndTraceFile = format: [nts] - string; Mask for the end of trace muting

	######################################
	#### PARAMETERS FOR OFFSET MUTING ####
	######################################

	- offset = [0] - int; Should be set to 1 to apply an offset muting to the raw shot gathers

	- offsetMuting = [no default] - string; should be set to "near" (to mute near offsets) or "far" (to mute far offset)

	- maxOffset = [0.0] - float - [km];
		1) If offsetMuting="near", no tapering is applied for offsets greater than abs(maxOffset)
		2) If offsetMuting="far", no tapering is applied for offsets smaller than abs(maxOffset)

	- taperWidthOffset = [0.0] - float - [km]; Offset range for for which the cosine tapering is applied (offset = abs(receiver_position-source_postion))
		1) For taperWidthOffset = "near", a weight of 0 is applied to traces whose receivers are located within a radius of maxOffset-taperWidthOffset from the source
		1) For taperWidthOffset = "far", a weight of 0 is applied to traces whose receivers are located outside of a radius of abs(maxOffset)+abs(taperWidthOffset) from the source

	- expOffset = [2.0] - float; Exponent for the cosine tapering function (must be an even number)

	######################################
	#### PARAMETERS FOR TIME MUTING ######
	######################################

	- time = [0] - int; Should be set to 1 to apply an offset muting to the raw shot gathers

	- timeMuting = [no default] - string; should be set to "early" (to mute early arrivals) or "late" (to mute late arrivals)

	- moveout = [no default] - string; set to "linear" or "hyperbolic"
		1) If moveout="linear", cutoff time is given by t_cutoff = t0 + abs(source_position - receiver_position) / velMute
		2) If moveout="hyperbolic", cutoff time is given by t_cutoff = sqrt(t0^2 + (source_position - receiver_position)^2 / velMute^2

	- t0 = [0.0] - float - [s]; Time cutoff at zero offset (i.e., where source_position = receiver_position)

	- velMute = [0.0] - float - [s]; Velocity computed for moveout and time cutoff

	- taperWidthTime = [0.0] - float - [s];
		1) If taperWidthTime = "early":
			* A weight of 0 is applied to times smaller than t_cutoff - taperWidthTime
			* A weight of 1 is applied to times larger than t_cutoff
			* A cosine taper is applied to times larger than t_cutoff - taperWidthTime and smaller than t_cutoff
		2) If taperWidthTime = "late":
			* A weight of 0 is applied to times larger than t_cutoff + taperWidthTime
			* A weight of 1 is applied to times samller than t_cutoff
			* A cosine taper is applied to times larger than t_cutoff and t_cutoff + taperWidthTime

	- expTime = [2.0] - float; Exponent for the cosine tapering function (must be an even number)

	######################################
	# PARAMETERS FOR END OF TRACE MUTING #
	######################################
	taperEndTraceWidth = [0] - float - [s]; If taperEndTraceWidth > 0, a cosing taper is applied to mute the end of each trace. The tapering beging at t_max - taperEndTraceWidth and is zero at t_max

"""

import genericIO
import SepVector
import Hypercube
import numpy as np
import dataTaperModule_3D
import sys

if __name__ == '__main__':

	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Read model (seismic data that you wish to mute/taper)
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Initialize operator
	t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,time,offset,sourceGeometry,receiverGeometry=dataTaperModule_3D.dataTaperInit_3D(sys.argv)

	# Instanciate operator
	dataTaperOb=dataTaperModule_3D.dataTaper(model,model,t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,time,offset,model.getHyper(),sourceGeometry,receiverGeometry)

	# Get offset tapering mask
	maskOffsetFile=parObject.getString("maskOffsetFile","noMaskOffsetFile")
	if (maskOffsetFile != "noMaskOffsetFile" and offset==1):
		taperMaskOffset=dataTaperOb.getTaperMaskOffset_3D()
		genericIO.defaultIO.writeVector(maskOffsetFile,taperMaskOffset)

	# Get time tapering mask
	maskTimeFile=parObject.getString("maskTimeFile","noMaskTimeFile")
	if (maskTimeFile != "noMaskTimeFile" and time==1):
		taperMaskTime=dataTaperOb.getTaperMaskTime_3D()
		genericIO.defaultIO.writeVector(maskTimeFile,taperMaskTime)

	# Get end of trace tapering mask
	maskEndTraceFile=parObject.getString("maskEndTraceFile","noMaskEndTraceFile")
	if (maskEndTraceFile != "noMaskEndTraceFile" and taperEndTraceWidth>0.0):
		taperMaskEndTrace=dataTaperOb.getTaperMaskEndTrace_3D()
		genericIO.defaultIO.writeVector(maskEndTraceFile,taperMaskEndTrace)

	# Write data
	data=SepVector.getSepVector(model.getHyper())
	dataTaperOb.forward(False,model,data)
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)
