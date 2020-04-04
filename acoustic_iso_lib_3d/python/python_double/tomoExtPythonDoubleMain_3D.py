#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import pyOperator as pyOp
import Acoustic_iso_double_3D
import numpy as np
import time
import sys

if __name__ == '__main__':

	# Initialize operator
	modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,reflectivityExt,dataHyperForOutput=Acoustic_iso_double_3D.tomoExtOpInitDouble_3D(sys.argv)

	# Construct tomo extended operator object
	tomoExtOp=Acoustic_iso_double_3D.tomoExtShotsGpu_3D(modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,reflectivityExt)

	# Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		# print("Main 1")
		tomoExtOp.dotTest(True)
		tomoExtOp.dotTest(True)
		tomoExtOp.dotTest(True)
		quit(0)

	if (parObject.getInt("saveWavefield1",0) == 1):		
		wavefield1File=parObject.getString("wavefield1File","noWavefield1File")
		if (wavefield1File == "noWavefield1File"):
			raise ValueError("**** ERROR [tomoExtPythonDoubleMain_3D]: User asked to save wavefield #1 but did not provide a file name ****\n")

		iWavefield=parObject.getInt("iWavefield",0)
		print("**** [tomoExtPythonDoubleMain_3D]: User has requested to save source wavefield #%d ****\n"%(iWavefield))

	if (parObject.getInt("saveWavefield2",0) == 1):
		wavefield2File=parObject.getString("wavefield2File","noWavefield2File")
		if (wavefield2File == "noWavefield2File"):
			raise ValueError("**** ERROR [tomoExtPythonDoubleMain_3D]: User asked to save wavefield #2 but did not provide a file name ****\n")

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("------------- Running Python tomo extended forward 3D -------------")
		print("--------------------- Double precision Python code ----------------")
		print("-------------------------------------------------------------------\n")

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise ValueError("**** ERROR [tomoExtPythonDoubleMain_3D]: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise ValueError("**** ERROR [tomoExtPythonDoubleMain_3D]: User did not provide data file name ****\n")

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)
		modelDouble=SepVector.getSepVector(modelFloat.getHyper(),storage="dataDouble",ndims=3)
		modelDoubleNp=modelDouble.getNdArray()
		modelFloatNp=modelFloat.getNdArray()
		modelDoubleNp[:]=modelFloatNp

		# Apply forward
		tomoExtOp.forward(False,modelDouble,dataDouble)

		# Write data
		dataFloat=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat",ndims=3)
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataFloatNp[:]=dataDoubleNp
		genericIO.defaultIO.writeVector(dataFile,dataFloat)

		# Saving wavefield 1
		if (parObject.getInt("saveWavefield1",0) == 1):
			wavefield1Double = tomoExtOp.getWavefield1_3D(iWavefield)
			wavefield1Float=SepVector.getSepVector(wavefield1Double.getHyper())
			wavefield1DoubleNp=wavefield1Double.getNdArray()
			wavefield1FloatNp=wavefield1Float.getNdArray()
			wavefield1FloatNp[:]=wavefield1DoubleNp
			genericIO.defaultIO.writeVector(wavefield1File,wavefield1Float)

		# Saving wavefield 2
		if (parObject.getInt("saveWavefield2",0) == 1):
			wavefield2Double = tomoExtOp.getWavefield1_3D(iWavefield)
			wavefield2Float=SepVector.getSepVector(wavefield2Double.getHyper())
			wavefield2DoubleNp=wavefield2Double.getNdArray()
			wavefield2FloatNp=wavefield2Float.getNdArray()
			wavefield2FloatNp[:]=wavefield2DoubleNp
			genericIO.defaultIO.writeVector(wavefield2File,wavefield2Float)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("---------------- Running Python tomo extended adjoint -------------")
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise ValueError("**** ERROR [tomoExtPythonDoubleMain_3D]: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise ValueError("**** ERROR [tomoExtPythonDoubleMain_3D]: User did not provide data file name ****\n")

		# Read data
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataDoubleNp[:]=dataFloatNp

		# Apply adjoint
		tomoExtOp.adjoint(False,modelDouble,dataDouble)

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

		# Saving wavefield 1
		if (parObject.getInt("saveWavefield1",0) == 1):
			wavefield1Double = tomoExtOp.getWavefield1_3D(iWavefield)
			wavefield1Float=SepVector.getSepVector(wavefield1Double.getHyper())
			wavefield1DoubleNp=wavefield1Double.getNdArray()
			wavefield1FloatNp=wavefield1Float.getNdArray()
			wavefield1FloatNp[:]=wavefield1DoubleNp
			genericIO.defaultIO.writeVector(wavefield1File,wavefield1Float)

		# Saving wavefield 2
		if (parObject.getInt("saveWavefield2",0) == 1):
			wavefield2Double = tomoExtOp.getWavefield1_3D(iWavefield)
			wavefield2Float=SepVector.getSepVector(wavefield2Double.getHyper())
			wavefield2DoubleNp=wavefield2Double.getNdArray()
			wavefield2FloatNp=wavefield2Float.getNdArray()
			wavefield2FloatNp[:]=wavefield2DoubleNp
			genericIO.defaultIO.writeVector(wavefield2File,wavefield2Float)


		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
