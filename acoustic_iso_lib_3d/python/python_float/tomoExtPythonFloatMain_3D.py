#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import pyOperator as pyOp
import Acoustic_iso_float_3D
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
	modelFloat,dataFloat,velFloat,parObject1,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityExtFloat,dataHyperForOutput,modelFloatLocal=Acoustic_iso_float_3D.tomoExtOpInitFloat_3D(sys.argv,client)

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		if client:
			raise NotImplementedError("Ginsu option, not supported for Dask interface yet")
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu = Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,velFloat,sourcesVector,receiversVector)

	# Construct tomo extended operator object
	if (parObject.getInt("ginsu", 0) == 0):
		if client:
			iwrk = 0
			#Instantiating Dask Operator
			tomoExtOp_args = [(modelFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsFloat.vecDask[iwrk],receiversVector[iwrk],reflectivityExtFloat.vecDask[iwrk]) for iwrk in range(nWrks)]
			tomoExtOp = DaskOp.DaskOperator(client,Acoustic_iso_float_3D.tomoExtShotsGpu_3D,tomoExtOp_args,[1]*nWrks)
			#Adding spreading operator and concatenating with Born operator (using modelFloatLocal)
			Sprd = DaskOp.DaskSpreadOp(client,modelFloatLocal,[1]*nWrks)
			tomoExtOp = pyOp.ChainOperator(Sprd,tomoExtOp)
		else:
			tomoExtOp=Acoustic_iso_float_3D.tomoExtShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityExtFloat)
	else:
		tomoExtOp=Acoustic_iso_float_3D.tomoExtShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityExtFloat,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)

	# Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		tomoExtOp.dotTest(True)
		tomoExtOp.dotTest(True)
		tomoExtOp.dotTest(True)
		quit(0)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("------------- Running Python tomo extended forward 3D -------------")
		print("--------------------- Single precision Python code ----------------")
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

		# Apply forward
		tomoExtOp.forward(False,modelFloatLocal,dataFloat)

		# Write data
		dataFloat.writeVec(dataFile)

		# Deallocate pinned memory
		if client is None:
			tomoExtOp.deallocatePinnedTomoExtGpu_3D()

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("------------- Running Python tomo extended adjoint 3D -------------")
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

		if(client):
			#Chunking the data and spreading them across workers if dask was requested
			dataFloat = Acoustic_iso_float_3D.chunkData(dataFloat,tomoExtOp.getRange())

		# Apply adjoint
		tomoExtOp.adjoint(False,modelFloatLocal,dataFloat)

		# Write model
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
		    print("**** ERROR: User did not provide model file name ****\n")
		    quit()
		modelFloatLocal.writeVec(modelFile)

		# Deallocate pinned memory
		if client is None:
			tomoExtOp.deallocatePinnedTomoExtGpu_3D()

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
