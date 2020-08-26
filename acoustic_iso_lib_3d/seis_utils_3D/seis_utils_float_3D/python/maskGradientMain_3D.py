#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import maskGradientModule_3D
import sys

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Initialize operator
	vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile,bathymetryFile=maskGradientModule_3D.maskGradientInit_3D(sys.argv)

	# Read model
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Instanciate operator
	maskGradientOp=maskGradientModule_3D.maskGradient_3D(vel,vel,vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile,bathymetryFile)

	# Get tapering mask and write to output
	maskFile=parObject.getString("mask","noMaskForOutput")
	mask=maskGradientOp.getMask()
	if (maskFile != "noMaskForOutput"):
		genericIO.defaultIO.writeVector(maskFile,mask)

	# Apply forward operator and write output data
	data=SepVector.getSepVector(model.getHyper())
	maskGradientOp.adjoint(False,data,model)
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)
