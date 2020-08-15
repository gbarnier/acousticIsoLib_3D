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

	# Initialize Ginsu
	if (parObject.getInt("ginsu", 0) == 1):
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,_,_ = Acoustic_iso_double_3D.buildGeometryGinsu_3D(parObject,velDouble,sourcesVector,receiversVector)

	# QC acquisition devices
	# nShot = len(sourcesVector)
	# print("Position devices after")
	# for iShot in range(nShot):
	# 	sourcesVector[iShot].printRegPosUnique()
	# 	receiversVector[iShot].printRegPosUnique()

	# print("-----------------------")
	# QC geometry coordinates
	# Shot 1
	# print("Shot 1, nzGinsu = ", velHyperVectorGinsu[0].axes[0].n)
	# print("Shot 1, ozGinsu = ", velHyperVectorGinsu[0].axes[0].o)
	# print("Shot 1, dzGinsu = ", velHyperVectorGinsu[0].axes[0].d)
	# print("Shot 1, nxGinsu = ", velHyperVectorGinsu[0].axes[1].n)
	# print("Shot 1, oxGinsu = ", velHyperVectorGinsu[0].axes[1].o)
	# print("Shot 1, dxGinsu = ", velHyperVectorGinsu[0].axes[1].d)
	# print("Shot 1, nyGinsu = ", velHyperVectorGinsu[0].axes[2].n)
	# print("Shot 1, oyGinsu = ", velHyperVectorGinsu[0].axes[2].o)
	# print("Shot 1, dyGinsu = ", velHyperVectorGinsu[0].axes[2].d)

	# Construct nonlinear operator object
	if (parObject.getInt("ginsu", 0) == 0):
		nonlinearOp=Acoustic_iso_double_3D.nonlinearPropShotsGpu_3D(modelDouble,dataDouble,velDouble,parObject,sourcesVector,receiversVector)
	else:
		print("Nonlinear main, ixVectorGinsu = ", ixVectorGinsu)
		print("Nonlinear main, iyVectorGinsu = ", iyVectorGinsu)
		nonlinearOp=Acoustic_iso_double_3D.nonlinearPropShotsGpu_3D(modelDouble,dataDouble,velDouble,parObject,sourcesVector,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,ixVectorGinsu,iyVectorGinsu)

		print("ixVectorGinsu = ", ixVectorGinsu)
		print("iyVectorGinsu = ", iyVectorGinsu)

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

		if (parObject.getInt("freeSurface",0) == 1):
			print("---------- Using a free surface condition for modeling ------------")

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

		print("Here 1")
		# Apply forward
		t0 = time.time()
		nonlinearOp.forward(False,modelDouble,dataDouble)
		t1 = time.time()
		total = t1-t0
		print("Time for nonlinear forward = ", total)
		# nonlinearOp.forward(False,modelDouble,dataDouble)
		# nonlinearOp.forward(False,modelDouble,dataDouble)
		# nonlinearOp.forward(False,modelDouble,dataDouble)
		# nonlinearOp.forward(False,modelDouble,dataDouble)
		# nonlinearOp.forward(False,modelDouble,dataDouble)

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
		# genericIO.defaultIO.writeVector(dataFile,dataFloat)

		if dataHyperForOutput.getNdim() == 7:
			ioMod=genericIO.defaultIO.cppMode
			fileObj=genericIO.regFile(ioM=ioMod,tag=dataFile,fromHyper=dataHyperForOutput,usage="output")
			fileObj.writeWindow(dataFloat)
		else:
			genericIO.defaultIO.writeVector(dataFile,dataFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("----------------- Running Python nonlinear adjoint ----------------")
		print("-------------------- Double precision Python code -----------------")
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
		# dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)
		dataFloat=genericIO.defaultIO.getVector(dataFile)
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		# Check if we have a regular acquisition geometry
		# if (dataHyperForOutput.getNdim() > 3):
		dataDoubleNp.flat[:]=dataFloatNp

		# Apply adjoint
		nonlinearOp.adjoint(False,modelDouble,dataDouble)
		# nonlinearOp.adjoint(False,modelDouble,dataDouble)
		# nonlinearOp.adjoint(False,modelDouble,dataDouble)
		# nonlinearOp.adjoint(False,modelDouble,dataDouble)
		# nonlinearOp.adjoint(False,modelDouble,dataDouble)
		# nonlinearOp.adjoint(False,modelDouble,dataDouble)
		# nonlinearOp.adjoint(False,modelDouble,dataDouble)
		# nonlinearOp.adjoint(False,modelDouble,dataDouble)

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
