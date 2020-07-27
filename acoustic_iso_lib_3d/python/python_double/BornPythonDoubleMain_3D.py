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
	modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,dataHyperForOutput=Acoustic_iso_double_3D.BornOpInitDouble_3D(sys.argv)

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu = Acoustic_iso_double_3D.buildGeometryGinsu_3D(parObject,velDouble,sourcesVector,receiversVector)

	# Construct Born operator object
	if (parObject.getInt("ginsu", 0) == 0):
		BornOp=Acoustic_iso_double_3D.BornShotsGpu_3D(modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector)
	else:
		print("Ginsu main")
		BornOp=Acoustic_iso_double_3D.BornShotsGpu_3D(modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)

	# Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		# print("Main 1")
		BornOp.dotTest(True)
		BornOp.dotTest(True)
		BornOp.dotTest(True)
		quit(0)

	if (parObject.getInt("saveSrcWavefield",0) == 1):
		srcWavefieldFile=parObject.getString("srcWavefieldFile","noSrcWavefieldFile")
		if (srcWavefieldFile == "noSrcWavefieldFile"):
			raise ValueError("**** ERROR [BornPythonDoubleMain_3D]: User asked to save source wavefield but did not provide a file name ****\n")

		iSrcWavefield=parObject.getInt("iSrcWavefield",0)
		print("**** [BornPythonDoubleMain_3D]: User has requested to save source wavefield #%d ****\n"%(iSrcWavefield))

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("-------------------- Running Python Born forward 3D ---------------")
		print("--------------------- Double precision Python code ----------------")
		print("-------------------------------------------------------------------\n")

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise ValueError("**** ERROR [BornPythonDoubleMain_3D]: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise ValueError("**** ERROR [BornPythonDoubleMain_3D]: User did not provide data file name ****\n")

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile)
		modelDouble=SepVector.getSepVector(modelFloat.getHyper(),storage="dataDouble")
		modelDoubleNp=modelDouble.getNdArray()
		modelFloatNp=modelFloat.getNdArray()
		modelDoubleNp[:]=modelFloatNp

		# Apply forward
		t0 = time.time()
		BornOp.forward(False,modelDouble,dataDouble)
		t1 = time.time()
		total = t1-t0
		print("Time for Born forward = ", total)
		# Write data
		dataFloat=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat")
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataFloatNp[:]=dataDoubleNp
		# genericIO.defaultIO.writeVector(dataFile,dataFloat)

		if dataHyperForOutput.getNdim() == 7:
			ioMod=genericIO.defaultIO.cppMode
			fileObj=genericIO.regFile(ioM=ioMod,tag=dataFile,fromHyper=dataHyperForOutput,usage="output")
			fileObj.writeWindow(dataFloat)
		else:
			genericIO.defaultIO.writeVector(dataFile,dataFloat)

		# Saving source wavefield
		if (parObject.getInt("saveSrcWavefield",0) == 1):
			srcWavefieldDouble = BornOp.getSrcWavefield_3D(iSrcWavefield)
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
		print("-------------------- Running Python Born adjoint 3D ---------------")
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
		t0 = time.time()
		BornOp.adjoint(False,modelDouble,dataDouble)
		t1 = time.time()
		total = t1-t0
		print("Time for Born adjoint = ", total)

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
			srcWavefieldDouble = BornOp.getSrcWavefield_3D(iSrcWavefield)
			srcWavefieldFloat=SepVector.getSepVector(srcWavefieldDouble.getHyper())
			srcWavefieldDoubleNp=srcWavefieldDouble.getNdArray()
			srcWavefieldFloatNp=srcWavefieldFloat.getNdArray()
			srcWavefieldFloatNp[:]=srcWavefieldDoubleNp
			genericIO.defaultIO.writeVector(srcWavefieldFile,srcWavefieldFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
