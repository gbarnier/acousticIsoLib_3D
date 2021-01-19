#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float_3D
import numpy as np
import time
import sys

if __name__ == '__main__':

	# Initialize operator
	modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,dataHyperForOutput=Acoustic_iso_float_3D.BornExtOpInitFloat_3D(sys.argv)

	if (parObject.getInt("fwime", 0) == 1):
		_,_,_,_,_,_,_,reflectivityExtFloat,_=Acoustic_iso_float_3D.tomoExtOpInitFloat_3D(sys.argv)
	else:
		wavefieldVecObj=None

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu = Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,velFloat,sourcesVector,receiversVector)

	# Normal constructor
	if (parObject.getInt("ginsu", 0) == 0):
		if (parObject.getInt("fwime",0)==1):
			tomoExtOp=Acoustic_iso_float_3D.tomoExtShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityExtFloat)
			wavefieldVecObj=tomoExtOp.getWavefieldVector()
		BornExtOp=Acoustic_iso_float_3D.BornExtShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,wavefieldVecFlag=wavefieldVecObj)
	else:
		if (parObject.getInt("fwime",0)==1):
			tomoExtOp=Acoustic_iso_float_3D.tomoExtShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityExtFloat,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)
			wavefieldVecObj=tomoExtOp.getWavefieldVector()
		BornExtOp=Acoustic_iso_float_3D.BornExtShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu,wavefieldVecFlag=wavefieldVecObj)

	# Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		BornExtOp.dotTest(True)
		BornExtOp.dotTest(True)
		BornExtOp.dotTest(True)
		quit(0)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("------------- Running Python Born extended forward 3D -------------")
		print("--------------------- Single precision Python code ----------------")
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

		# Apply forward
		BornExtOp.forward(False,modelFloat,dataFloat)

		# Write data
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

		# Apply adjoint
		BornExtOp.adjoint(False,modelFloat,dataFloat)

		# Write model
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			print("**** ERROR [BornPythonFloatMain_3D]: User did not provide model file name ****\n")
			quit()
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

		# Deallocate pinned memory
		BornExtOp.deallocatePinnedBornExtGpu_3D()

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
