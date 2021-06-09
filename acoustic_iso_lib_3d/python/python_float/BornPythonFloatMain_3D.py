#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float_3D
import pyOperator as pyOp
import numpy as np
import time
import sys

#Dask-related modules
import pyDaskOperator as DaskOp

if __name__ == '__main__':

	#Getting parameter object
	parObject=genericIO.io(params=sys.argv)

	# Checking if Dask was requested
	client, nWrks = Acoustic_iso_float_3D.create_client(parObject)

	# Initialize operator
	modelFloat,dataFloat,velFloat,parObject1,sourcesVector,sourcesSignalsFloat,receiversVector,dataHyperForOutput,modelFloatLocal=Acoustic_iso_float_3D.BornOpInitFloat_3D(sys.argv, client)

	if (parObject.getInt("fwime", 0) == 1):
		_,_,_,_,_,_,_,reflectivityExtFloat,_=Acoustic_iso_float_3D.tomoExtOpInitFloat_3D(sys.argv, client)
	else:
		tomoExtOp=None

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		if client:
			raise NotImplementedError("Ginsu option, not supported for Dask interface yet")
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu = Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,velFloat,sourcesVector,receiversVector)

	# Construct Born operator object
	if (parObject.getInt("ginsu", 0) == 0):
		if client:
			iwrk = 0
			#Instantiating Dask Operator
			BornOp_args = [(modelFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsFloat.vecDask[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
			BornOp_kwargs = [dict() for iwrk in range(nWrks)]
			for iwrk in range(nWrks):
				BornOp_kwargs[iwrk].update({'tomoExtOp':tomoExtOp})
			BornOp = DaskOp.DaskOperator(client,Acoustic_iso_float_3D.BornShotsGpu_3D,BornOp_args,[1]*nWrks, op_kwargs=BornOp_kwargs)
			#Adding spreading operator and concatenating with Born operator (using modelFloatLocal)
			Sprd = DaskOp.DaskSpreadOp(client,modelFloatLocal,[1]*nWrks)
			BornOp = pyOp.ChainOperator(Sprd,BornOp)
		else:
			if (parObject.getInt("fwime",0)==1):
				tomoExtOp=Acoustic_iso_float_3D.tomoExtShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityExtFloat)
			BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,tomoExtOp=tomoExtOp)
	else:
		if (parObject.getInt("fwime",0)==1):
			tomoExtOp=Acoustic_iso_float_3D.tomoExtShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityExtFloat,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)
		BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu,tomoExtOp=tomoExtOp)

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
		modelFloatLocal=genericIO.defaultIO.getVector(modelFile)

		# Apply forward
		BornOp.forward(False,modelFloatLocal,dataFloat)

		# Write data
		if dataHyperForOutput is None:
			dataHyperForOutputDim = 1
		else:
			dataHyperForOutputDim = dataHyperForOutput.getNdim()
		if dataHyperForOutputDim == 7:
			ioMod=genericIO.defaultIO.cppMode
			fileObj=genericIO.regFile(ioM=ioMod,tag=dataFile,fromHyper=dataHyperForOutput,usage="output")
			fileObj.writeWindow(dataFloat)
		else:
			dataFloat.writeVec(dataFile)
			
		# Saving source wavefield
		if (parObject.getInt("saveSrcWavefield",0) == 1) and client is None:
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
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
		    print("**** ERROR [BornPythonFloatMain_3D]: User did not provide model file name ****\n")
		    quit()

		# Write data
		if dataHyperForOutput is None:
			dataHyperForOutputDim = 3
		else:
			dataHyperForOutputDim = dataHyperForOutput.getNdim()

		# Read data for irregular geometry
		if (dataHyperForOutputDim == 3):
			dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)
		# Problem here - needs to be fixed by Bob (add readble 7 dimension hypercubes)
		else:
			dataFloatTemp=genericIO.defaultIO.getVector(dataFile,ndims=7)
			dataFloatTempNp=dataFloatTemp.getNdArray()
			dataFloatNp=dataFloat.getNdArray()
			dataFloatNp.flat[:]=dataFloatTempNp

		if(client):
			#Chunking the data and spreading them across workers if dask was requested
			dataFloat = Acoustic_iso_float_3D.chunkData(dataFloat,BornOp.getRange())

		# Apply adjoint
		BornOp.adjoint(False,modelFloatLocal,dataFloat)

		# Write model
		modelFloatLocal.writeVec(modelFile)

		# Saving source wavefield
		if (parObject.getInt("saveSrcWavefield",0) == 1) and client is None:
			srcWavefieldFloat = BornOp.getSrcWavefield_3D(iSrcWavefield)
			genericIO.defaultIO.writeVector(srcWavefieldFile,srcWavefieldFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
