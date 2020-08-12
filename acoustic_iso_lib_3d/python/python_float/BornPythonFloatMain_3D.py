#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float_3D
import numpy as np
import time
import sys

if __name__ == '__main__':

	print("Here 0")

	# Initialize operator
	modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,dataHyperForOutput=Acoustic_iso_float_3D.BornOpInitFloat_3D(sys.argv)

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu = Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,velFloat,sourcesVector,receiversVector)

	# Construct Born operator object
	if (parObject.getInt("ginsu", 0) == 0):
		BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector)
	else:
		BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)

	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		BornOp.dotTest(True)
		BornOp.dotTest(True)
		BornOp.dotTest(True)
		quit(0)

	if (parObject.getInt("saveSrcWavefield",0) == 1):
		srcWavefieldFile=parObject.getString("srcWavefieldFile","noSrcWavefieldFile")
		if (srcWavefieldFile == "noSrcWavefieldFile"):
			raise ValueError("**** ERROR [BornPythonFloatMain_3D]: User asked to save source wavefield but did not provide a file name ****\n")

		iSrcWavefield=parObject.getInt("iSrcWavefield",0)
		print("**** [BornPythonFloatMain_3D]: User has requested to save source wavefield #%d ****\n"%(iSrcWavefield))

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("-------------------- Running Python Born forward 3D ---------------")
		print("--------------------- Single precision Python code ----------------")
		print("-------------------------------------------------------------------\n")

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise ValueError("**** ERROR [BornPythonFloatMain_3D]: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise ValueError("**** ERROR [BornPythonFloatMain_3D]: User did not provide data file name ****\n")

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile)

		# Apply forward
		t0 = time.time()
		BornOp.forward(False,modelFloat,dataFloat)
		t1 = time.time()
		total = t1-t0
		print("Time for Born forward = ", total)

		# Write data
		if dataHyperForOutput.getNdim() == 7:
			ioMod=genericIO.defaultIO.cppMode
			fileObj=genericIO.regFile(ioM=ioMod,tag=dataFile,fromHyper=dataHyperForOutput,usage="output")
			fileObj.writeWindow(dataFloat)
		else:
			genericIO.defaultIO.writeVector(dataFile,dataFloat)

		# Saving source wavefield
		if (parObject.getInt("saveSrcWavefield",0) == 1):
			srcWavefieldFloat = BornOp.getSrcWavefield_3D(iSrcWavefield)
			genericIO.defaultIO.writeVector(srcWavefieldFile,srcWavefieldFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("-------------------- Running Python Born adjoint 3D ---------------")
		print("--------------------- Single precision Python code ----------------")
		print("-------------------------------------------------------------------\n")

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

		# Check that data was provided
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
		    print("**** ERROR [BornPythonFloatMain_3D]: User did not provide data file ****\n")
		    quit()

		# Read data for irregular geometry
		if (dataHyperForOutput.getNdim() == 3):
			dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Problem here - needs to be fixed by Bob (add readble 7 dimension hypercubes)
		else:
			dataFloatTemp=genericIO.defaultIO.getVector(dataFile,ndims=7)
			dataFloatTempNp=dataFloatTemp.getNdArray()
			dataFloatNp=dataFloat.getNdArray()
			dataFloatNp.flat[:]=dataFloatTempNp

		# Apply adjoint
		t0 = time.time()
		BornOp.adjoint(False,modelFloat,dataFloat)
		t1 = time.time()
		total = t1-t0
		print("Time for Born adjoint = ", total)

		# Write model
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
		    print("**** ERROR [BornPythonFloatMain_3D]: User did not provide model file name ****\n")
		    quit()
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

		# Saving source wavefield
		if (parObject.getInt("saveSrcWavefield",0) == 1):
			srcWavefieldFloat = BornOp.getSrcWavefield_3D(iSrcWavefield)
			genericIO.defaultIO.writeVector(srcWavefieldFile,srcWavefieldFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
