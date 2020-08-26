#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_double_3D
import numpy as np
import time
import sys

if __name__ == '__main__':

	# Initialize operator
	modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,dataHyperForOutput=Acoustic_iso_double_3D.BornExtOpInitDouble_3D(sys.argv)

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu = Acoustic_iso_double_3D.buildGeometryGinsu_3D(parObject,velDouble,sourcesVector,receiversVector)

	# Construct nonlinear operator object
	if (parObject.getInt("ginsu", 0) == 0):
		BornExtOp=Acoustic_iso_double_3D.BornExtShotsGpu_3D(modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector)
	else:
		BornExtOp=Acoustic_iso_double_3D.BornExtShotsGpu_3D(modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)

	# Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		# print("Main 1")
		BornExtOp.dotTest(True)
		BornExtOp.dotTest(True)
		BornExtOp.dotTest(True)
		quit(0)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("------------- Running Python Born extended forward 3D -------------")
		print("--------------------- Double precision Python code ----------------")
		print("-------------------------------------------------------------------\n")

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise ValueError("**** ERROR [BornExtPythonDoubleMain_3D]: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise ValueError("**** ERROR [BornExtPythonDoubleMain_3D]: User did not provide data file name ****\n")

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=5)
		modelDouble=SepVector.getSepVector(modelFloat.getHyper(),storage="dataDouble",ndims=5)
		modelDoubleNp=modelDouble.getNdArray()
		modelFloatNp=modelFloat.getNdArray()
		modelDoubleNp[:]=modelFloatNp

		# Apply forward
		BornExtOp.forward(False,modelDouble,dataDouble)

		# Write data
		dataFloat=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat",ndims=3)
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataFloatNp[:]=dataDoubleNp
		genericIO.defaultIO.writeVector(dataFile,dataFloat)

		# Deallocate pinned memory
		BornExtOp.deallocatePinnedBornExtGpu_3D()

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("------------- Running Python Born extended adjoint 3D -------------")
		print("--------------------- Double precision Python code ----------------")
		print("-------------------------------------------------------------------\n")

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

		# Check that data was provided
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
		    print("**** ERROR: User did not provide data file ****\n")
		    quit()

		# Read data
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataDoubleNp[:]=dataFloatNp

		# Apply adjoint
		BornExtOp.adjoint(False,modelDouble,dataDouble)

		# Write model
		modelFloat=SepVector.getSepVector(modelDouble.getHyper(),storage="dataFloat")
		modelFloatNp=modelFloat.getNdArray()
		modelDoubleNp=modelDouble.getNdArray()
		modelFloatNp[:]=modelDoubleNp
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
		    print("**** ERROR: User did not provide model file name ****\n")
		    quit()
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

		# Deallocate pinned memory
		BornExtOp.deallocatePinnedBornExtGpu_3D()

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
