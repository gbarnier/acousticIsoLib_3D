#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import traceNormModule_3D
import sys
import time
from numpy import linalg as LA

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Read model
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Read epsilon stabilization
	epsilonTraceNorm=parObject.getFloat("epsilonTraceNorm",1e-10)

	# Create data
	data=model.clone()

	# Instanciate PhaseOnlyXk object
	traceNormOp=traceNormModule_3D.traceNorm_3D(data,data,epsilonTraceNorm)

	# Apply normalization
	traceNormOp.forward(False,model,data)

	# Write data
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)

	# help(nlOp.linTest)
