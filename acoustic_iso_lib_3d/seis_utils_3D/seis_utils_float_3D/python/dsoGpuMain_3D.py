#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import dsoGpuModule_3D
# import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)
	adj=parObject.getInt("adj",0)

	nz,nx,ny,nExt1,nExt2,fat,zeroShift=dsoGpuModule_3D.dsoGpuInit_3D(sys.argv)

	# Forward
	if (adj==0):

		# Read model (on coarse grid)
		modelFile=parObject.getString("model")
		model=genericIO.defaultIO.getVector(modelFile,ndims=5)

		modelDouble=SepVector.getSepVector(model.getHyper(),storage="dataDouble",ndims=5)
		modelDoubleNp=modelDouble.getNdArray()
		modelNp=model.getNdArray()
		modelDoubleNp[:]=modelNp

		# Create data
		dataDouble=modelDouble.clone()

		# Create DSO object
		dsoOp=dsoGpuModule_3D.dsoGpu_3D(modelDouble,dataDouble,nz,nx,ny,nExt1,nExt2,fat,zeroShift)

		# Testing dot-product test of the operator
		if (parObject.getInt("dpTest",0) == 1):
			dsoOp.dotTest(True)
			dsoOp.dotTest(True)
			dsoOp.dotTest(True)
			quit(0)

		# Run forward
		t0 = time.time()
		dsoOp.forward(False,modelDouble,dataDouble)
		t1 = time.time()
		total = t1-t0
		print("total time for DSO = ",total)
		# Write data
		dataFile=parObject.getString("data","noDataFile")
		dataFloat=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat",ndims=5)
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataFloatNp[:]=dataDoubleNp
		genericIO.defaultIO.writeVector(dataFile,dataFloat)

	# Adjoint
	else:

		# Read data
		dataFile=parObject.getString("data","noDataFile")
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=5)
		dataDouble=SepVector.getSepVector(dataFloat.getHyper(),storage="dataDouble",ndims=5)
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataDoubleNp[:]=dataFloatNp

		# Create model
		modelDouble=dataDouble.clone()

		# Create DSO object
		dsoOp=dsoGpuModule_3D.dsoGpu_3D(modelDouble,dataDouble,nz,nx,ny,nExt1,nExt2,fat,zeroShift)

		if (parObject.getInt("dpTest",0) == 1):
			dsoOp.dotTest(True)
			dsoOp.dotTest(True)
			dsoOp.dotTest(True)
			quit(0)

		# Run adjoint
		dsoOp.adjoint(False,modelDouble,dataDouble)

		# Write model
		modelFloat=SepVector.getSepVector(modelDouble.getHyper(),storage="dataFloat")
		modelFloatNp=modelFloat.getNdArray()
		modelDoubleNp=modelDouble.getNdArray()
		modelFloatNp[:]=modelDoubleNp
		modelFile=parObject.getString("model","noModelFile")
		genericIO.defaultIO.writeVector(modelFile,modelFloat)
