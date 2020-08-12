#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Acoustic_iso_float_3D

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
from pyLinearSolver import LSQRsolver as LSQR
import pyProblem as Prblm
import inversionUtilsFloat_3D
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)
	solver=parObject.getString("solver","LCG")

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtilsFloat_3D.inversionInitFloat_3D(sys.argv)
	# Logger
	inv_log = logger(logFile)

	# Display information
	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------ Linearized waveform inversion ------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ Linearized waveform inversion --------------")
	if (parObject.getInt("freeSurface",0) == 1):
		print("---------- Using a free surface condition for modeling ------------")

	# Initialize Born
	modelInitFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,dataHyperForOutput=Acoustic_iso_float_3D.BornOpInitFloat_3D(sys.argv)
	print("dataHyperForOutput = ", dataHyperForOutput)
	print("data shape 1 = ",dataFloat.shape)

	# Check if Ginsu is required
	if (parObject.getInt("ginsu",0) == 1):
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu=Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,velFloat,sourcesVector,receiversVector)

	# Construct Born operator object - No Ginsu
	if (parObject.getInt("ginsu",0) == 0):
		BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelInitFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector)
	# With Ginsu
	else:
		BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelInitFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)

	# Create inversion operator
	invOp=BornOp

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInitFloat.set(0.0)
	else:
		modelInitFloat=genericIO.defaultIO.getVector(modelInitFile,ndims=3)

	# Data
	if dataHyperForOutput.getNdim() == 7:
		dataFile=parObject.getString("data")
		dataFloatTemp=genericIO.defaultIO.getVector(dataFile,ndim=7)
		dataFloatTempNp=dataFloatTemp.getNdArray()
		dataFloat.getNdArray()[:]=dataFloatTempNp

	else:
		dataFile=parObject.getString("data")
		dataFloatTemp=genericIO.defaultIO.getVector(dataFile,ndim=7)
		dataFloatTempNp=dataFloatTemp.getNdArray()
		dataFloat.getNdArray()[:]=dataFloatTempNp

	# No regularization
	invProb=Prblm.ProblemL2Linear(modelInitFloat,dataFloat,invOp)

	############################## Solver ######################################
	# Solver
	if solver == "LCG":
		Linsolver=LCG(stop,logger=inv_log)
	elif solver == "LSQR":
		Linsolver=LSQR(stop,logger=inv_log)
	else:
		raise ValueError("Unknown solver: %s"%(solver))
	Linsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	Linsolver.run(invProb,verbose=True)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
