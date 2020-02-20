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
	modelDouble,dataDouble,velDouble,parObject,sourcesVector,receiversVector,dataHyperForOutput=Acoustic_iso_double_3D.nonlinearOpInitDouble_3D(sys.argv)

	# Construct nonlinear operator object
	nonlinearOp=Acoustic_iso_double_3D.nonlinearPropShotsGpu_3D(modelDouble,dataDouble,velDouble,parObject,sourcesVector,receiversVector)

	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		nonlinearOp.dotTest(True)
		nonlinearOp.dotTest(True)
		nonlinearOp.dotTest(True)
		quit(0)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("----------------- Running Python nonlinear forward 3D -------------")
		print("-------------------- Double precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
		    raise ValueError("**** ERROR [nonlinearPythonDoubleMain_3D]: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
		    raise ValueError("**** ERROR [nonlinearPythonDoubleMain_3D]: User did not provide data file name ****\n")

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=2)
		# Check that the length of the second axis of the model are consistent
		if (modelDouble.getHyper().axes[1].n != modelFloat.getHyper().axes[1].n):
			raise ValueError("**** ERROR [nonlinearPythonDoubleMain_3D]: Length of axis#2 for model file (n2=%d) is not consistent with length from parameter file (n2=%d) ****\n" %(modelFloat.getHyper().axes[1].n,modelDouble.getHyper().axes[1].n))
		modelDMat=modelDouble.getNdArray()
		modelSMat=modelFloat.getNdArray()
		modelDMat[:]=modelSMat

		# Apply forward
		nonlinearOp.forward(False,modelDouble,dataDouble)

		# QC data size
		# print("dataDouble nDim=",dataDouble.getHyper().getNdim())
		# print("dataDouble n1=",dataDouble.getHyper().axes[0].n)
		# print("dataDouble n2=",dataDouble.getHyper().axes[1].n)
		# print("dataDouble n3=",dataDouble.getHyper().axes[2].n)
		#
		# print("dataHyperForOutput nDim",dataHyperForOutput.getNdim())
		# print("dataHyperForOutput n1",dataHyperForOutput.axes[0].n)
		# print("dataHyperForOutput n2",dataHyperForOutput.axes[1].n)
		# print("dataHyperForOutput n3",dataHyperForOutput.axes[2].n)
		# print("dataHyperForOutput n4",dataHyperForOutput.axes[3].n)
		# print("dataHyperForOutput n5",dataHyperForOutput.axes[4].n)
		# print("dataHyperForOutput n6",dataHyperForOutput.axes[5].n)
		# print("dataHyperForOutput n7",dataHyperForOutput.axes[6].n)

		# Write data
		dataFloat=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat")
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataFloatNp[:]=dataDoubleNp
		genericIO.defaultIO.writeVector(dataFile,dataFloat)
		# if dataHyperForOutput.getNdim() == 7:
		# 	dataTest=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat")
		# 	io=genericIO.defaultIO
		# 	fle=io.getRegFile("toto.H",fromHyper=dataHyperForOutput)
		# 	fle.writeWindow(dataTest)
		# 	# fileObj=genericIO.regFile(ioM=genericIO.io,tag=dataFile,fromHyper=dataHyperForOutput,usage="output")
		# 	# fileObj.writeWindow(dataFloat)
		# else:
		# 	genericIO.defaultIO.writeVector(dataFile,dataFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("----------------- Running Python nonlinear adjoint ----------------")
		print("-------------------- Double precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

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
		nonlinearOp.adjoint(False,modelDouble,dataDouble)

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

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
