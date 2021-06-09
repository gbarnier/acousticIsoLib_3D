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
	modelFloat,dataFloat,velFloat,parObject1,sourcesVector,sourcesSignalsFloat,receiversVector,dataHyperForOutput,modelFloatLocal=Acoustic_iso_float_3D.BornExtOpInitFloat_3D(sys.argv, client)

	if (parObject.getInt("fwime", 0) == 1):
		_,_,_,_,_,_,_,reflectivityExtFloat,_=Acoustic_iso_float_3D.tomoExtOpInitFloat_3D(sys.argv, client)
	else:
		wavefieldVecObj=None

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		if client:
			raise NotImplementedError("Ginsu option, not supported for Dask interface yet")
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu = Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,velFloat,sourcesVector,receiversVector)

	# Normal constructor
	if (parObject.getInt("ginsu", 0) == 0):
		if client:
			iwrk = 0
			#Instantiating Dask Operator
			BornExtOp_args = [(modelFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsFloat.vecDask[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
			BornExtOp_kwargs = [dict() for iwrk in range(nWrks)]
			for iwrk in range(nWrks):
				BornExtOp_kwargs[iwrk].update({'wavefieldVecFlag':wavefieldVecObj})
			BornExtOp = DaskOp.DaskOperator(client,Acoustic_iso_float_3D.BornExtShotsGpu_3D,BornExtOp_args,[1]*nWrks, op_kwargs=BornExtOp_kwargs)
			#Adding spreading operator and concatenating with Born operator (using modelFloatLocal)
			Sprd = DaskOp.DaskSpreadOp(client,modelFloatLocal,[1]*nWrks)
			BornExtOp = pyOp.ChainOperator(Sprd,BornExtOp)
		else:
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
		modelFloatLocal=genericIO.defaultIO.getVector(modelFile,ndims=5)

		# Apply forward
		BornExtOp.forward(False,modelFloatLocal,dataFloat)

		# Write data
		dataFloat.writeVec(dataFile)

		if client is None:
			# Deallocate pinned memory
			BornExtOp.deallocatePinnedBornExtGpu_3D()

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("------------- Running Python Born extended adjoint 3D -------------")
		print("--------------------- Single precision Python code ----------------")
		print("-------------------------------------------------------------------\n")

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

		# Check that data was provided
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			print("**** ERROR: User did not provide data file ****\n")
			quit()
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			print("**** ERROR [BornPythonFloatMain_3D]: User did not provide model file name ****\n")
			quit()

		# Read data
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)

		if(client):
			#Chunking the data and spreading them across workers if dask was requested
			dataFloat = Acoustic_iso_float_3D.chunkData(dataFloat,BornExtOp.getRange())

		# Apply adjoint
		BornExtOp.adjoint(False,modelFloatLocal,dataFloat)

		# Write model
		modelFloatLocal.writeVec(modelFile)

		if client is None:
			# Deallocate pinned memory
			BornExtOp.deallocatePinnedBornExtGpu_3D()

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
