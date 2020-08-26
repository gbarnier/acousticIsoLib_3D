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

	# Create data
	data=model.clone()

	# Create PhaseOnlyXkJac object and run forward
	predData,epsilonTraceNorm=traceNormModule_3D.traceNormDerivInit_3D(sys.argv)
	traceNormDerivOp=traceNormModule_3D.traceNormDeriv_3D(predData,epsilonTraceNorm)
	traceNormDerivOp.forward(False,model,data)

	# Run dot-product test
	# traceNormDerivOp.dotTest(True)

	# Write data
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)
