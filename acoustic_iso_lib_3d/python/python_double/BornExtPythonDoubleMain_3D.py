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

	# QC data size
	# print("model nDim=",modelDouble.getHyper().getNdim())
	# print("model n1=",modelDouble.getHyper().axes[0].n)
	# print("model n2=",modelDouble.getHyper().axes[1].n)
	# print("model n3=",modelDouble.getHyper().axes[2].n)
	# print("model n4=",modelDouble.getHyper().axes[3].n)
	# print("model n5=",modelDouble.getHyper().axes[4].n)
	# print("data nDim=",data.getHyper().getNdim())
	# print("data n1=",data.getHyper().axes[0].n)
	# print("data n2=",data.getHyper().axes[1].n)
	# print("data n3=",data.getHyper().axes[2].n)

	# Construct nonlinear operator object
	BornExtOp=Acoustic_iso_double_3D.BornExtShotsGpu_3D(modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector)

	# Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		# print("Main 1")
		BornExtOp.dotTest(True)
		BornExtOp.dotTest(True)
		BornExtOp.dotTest(True)
		quit(0)

	if (parObject.getInt("saveSrcWavefield",0) == 1):
		srcWavefieldFile=parObject.getString("srcWavefieldFile","noSrcWavefieldFile")
		if (srcWavefieldFile == "noSrcWavefieldFile"):
			raise ValueError("**** ERROR [BornExtPythonDoubleMain_3D]: User asked to save source wavefield but did not provide a file name ****\n")

		iSrcWavefield=parObject.getInt("iSrcWavefield",0)
		print("**** [BornExtPythonDoubleMain_3D]: User has requested to save source wavefield #%d ****\n"%(iSrcWavefield))

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

		# Saving source wavefield
		if (parObject.getInt("saveSrcWavefield",0) == 1):
			srcWavefieldDouble = BornExtOp.getSrcWavefield_3D(iSrcWavefield)
			srcWavefieldFloat=SepVector.getSepVector(srcWavefieldDouble.getHyper())
			srcWavefieldDoubleNp=srcWavefieldDouble.getNdArray()
			srcWavefieldFloatNp=srcWavefieldFloat.getNdArray()
			srcWavefieldFloatNp[:]=srcWavefieldDoubleNp
			genericIO.defaultIO.writeVector(srcWavefieldFile,srcWavefieldFloat)

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

		# Saving source wavefield
		if (parObject.getInt("saveSrcWavefield",0) == 1):
			srcWavefieldDouble = BornExtOp.getSrcWavefield_3D(iSrcWavefield)
			srcWavefieldFloat=SepVector.getSepVector(srcWavefieldDouble.getHyper())
			srcWavefieldDoubleNp=srcWavefieldDouble.getNdArray()
			srcWavefieldFloatNp=srcWavefieldFloat.getNdArray()
			srcWavefieldFloatNp[:]=srcWavefieldDoubleNp
			genericIO.defaultIO.writeVector(srcWavefieldFile,srcWavefieldFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
