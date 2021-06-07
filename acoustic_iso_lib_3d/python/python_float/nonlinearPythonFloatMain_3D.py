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
	modelFloat,dataFloat,velFloat,parObject1,sourcesVector,receiversVector,dataHyperForOutput,modelFloatLocal=Acoustic_iso_float_3D.nonlinearOpInitFloat_3D(sys.argv,client)

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		if client:
			raise NotImplementedError("Ginsu option, not supported for Dask interface yet")
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,_,_ = Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,velFloat,sourcesVector,receiversVector)

	# Construct nonlinear operator object
	if (parObject.getInt("ginsu", 0) == 0):
		if client:
			#Instantiating Dask Operator
			nlOp_args = [(modelFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
			nonlinearOp = DaskOp.DaskOperator(client,Acoustic_iso_float_3D.nonlinearPropShotsGpu_3D,nlOp_args,[1]*nWrks)
			#Adding spreading operator and concatenating with non-linear operator (using modelFloatLocal)
			Sprd = DaskOp.DaskSpreadOp(client,modelFloatLocal,[1]*nWrks)
			nonlinearOp = pyOp.ChainOperator(Sprd,nonlinearOp)
		else:
			nonlinearOp=Acoustic_iso_float_3D.nonlinearPropShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,receiversVector)
	else:
		nonlinearOp=Acoustic_iso_float_3D.nonlinearPropShotsGpu_3D(modelFloat,dataFloat,velFloat,parObject,sourcesVector,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,ixVectorGinsu,iyVectorGinsu)

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
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
		    raise ValueError("**** ERROR [nonlinearPythonSingleMain_3D]: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
		    raise ValueError("**** ERROR [nonlinearPythonSingleMain_3D]: User did not provide data file name ****\n")

		# Read model
		modelFloatTemp=genericIO.defaultIO.getVector(modelFile,ndims=2)

		# Check that the length of the second axis of the model are consistent
		if (modelFloatTemp.getHyper().axes[1].n != modelFloatLocal.getHyper().axes[1].n):
			raise ValueError("**** ERROR [nonlinearPythonSingleMain_3D]: Length of axis #2 for model file (n2=%d) is not consistent with length from parameter file (n2=%d) ****\n" %(modelFloatTemp.getHyper().axes[1].n,modelFloatLocal.getHyper().axes[1].n))
		modelFloatTempNp=modelFloatTemp.getNdArray()
		modelFloatNp=modelFloatLocal.getNdArray()
		modelFloatNp[:]=modelFloatTempNp

		# Apply forward
		# t0 = time.time()
		nonlinearOp.forward(False,modelFloatLocal,dataFloat)
		# t1 = time.time()
		# total = t1-t0
		# print("--- [nonlinearPythonSingleMain_3D]: Time for nonlinear forward = ", total," ---")

		# Write data
		dataFloat.writeVec(dataFile)
		
		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("----------------- Running Python nonlinear adjoint ----------------")
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		print("Free surface = ",parObject.getInt("freeSurface",0))
		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- [nonlinearPythonSingleMain_3D]: Using a free surface condition for modeling ------------")

		# Check that data was provided
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
		    print("**** ERROR [nonlinearPythonSingleMain_3D]: User did not provide data file ****\n")
		    quit()

		
		# Read data for irregular geometry
		dataFile=parObject.getString("data")
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)
		if(client):
			#Chunking the data and spreading them across workers if dask was requested
			dataFloat = Acoustic_iso_float_3D.chunkData(dataFloat,nonlinearOp.getRange())

		# Apply adjoint
		nonlinearOp.adjoint(False,modelFloatLocal,dataFloat)

		# Write model
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
		    print("**** ERROR [nonlinearPythonSingleMain_3D]: User did not provide model file name ****\n")
		    quit()

		modelFloatLocal.writeVec(modelFile)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
