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
	modelFloat,dataFloat,velFloat,parObject,sourcesVector,receiversVector,dataHyperForOutput=Acoustic_iso_float_3D.nonlinearOpInitFloat_3D(sys.argv)

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,_,_ = Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,velFloat,sourcesVector,receiversVector)

	# Construct nonlinear operator object
	if (parObject.getInt("ginsu", 0) == 0):
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
		if (modelFloatTemp.getHyper().axes[1].n != modelFloat.getHyper().axes[1].n):
			raise ValueError("**** ERROR [nonlinearPythonSingleMain_3D]: Length of axis #2 for model file (n2=%d) is not consistent with length from parameter file (n2=%d) ****\n" %(modelFloatTemp.getHyper().axes[1].n,modelFloat.getHyper().axes[1].n))
		modelFloatTempNp=modelFloatTemp.getNdArray()
		modelFloatNp=modelFloat.getNdArray()
		modelFloatNp[:]=modelFloatTempNp

		# Apply forward
		# t0 = time.time()
		print("run #",i)
		nonlinearOp.forward(False,modelFloat,dataFloat)
		# t1 = time.time()
		# total = t1-t0
		# print("--- [nonlinearPythonSingleMain_3D]: Time for nonlinear forward = ", total," ---")

		# Write data
		genericIO.defaultIO.writeVector(dataFile,dataFloat)

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

		# Apply adjoint
		nonlinearOp.adjoint(False,modelFloat,dataFloat)

		# Write model
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
		    print("**** ERROR [nonlinearPythonSingleMain_3D]: User did not provide model file name ****\n")
		    quit()
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
