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
	model,data,vel,parObject,sourcesVector,receiversVector,dataHyperForOutput=Acoustic_iso_float_3D.nonlinearOpInitFloat_3D(sys.argv)

	# Construct nonlinear operator object
	nonlinearOp=Acoustic_iso_float_3D.nonlinearPropShotsGpu_3D(model,data,vel,parObject,sourcesVector,receiversVector)

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
		    raise ValueError("**** ERROR [nonlinearPythonFloatMain_3D]: User did not provide model file ****\n")
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
		    raise ValueError("**** ERROR [nonlinearPythonFloatMain_3D]: User did not provide data file name ****\n")

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=2)
		# Check that the length of the second axis of the model are consistent
		if (modelFloat.getHyper().axes[1].n != model.getHyper().axes[1].n):
			raise ValueError("**** ERROR [nonlinearPythonFloatMain_3D]: Length of axis#2 for model file (n2=%d) is not consistent with length from parameter file (n2=%d) ****\n" %(modelFloat.getHyper().axes[1].n,model.getHyper().axes[1].n))

		# Apply forward
		nonlinearOp.forward(False,modelFloat,data)

		# Write data
		genericIO.defaultIO.writeVector(dataFile,data)


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
			print("---------- Using a free surface condition for modeling ------------")

		# Check that data was provided
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
		    print("**** ERROR: User did not provide data file ****\n")
		    quit()

		# Read data
		data=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Apply adjoint
		nonlinearOp.adjoint(False,model,data)

		# Write model
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
		    print("**** ERROR: User did not provide model file name ****\n")
		    quit()
		genericIO.defaultIO.writeVector(modelFile,model)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")
