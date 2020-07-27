#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Acoustic_iso_double_3D

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
from pyLinearSolver import LSQRsolver as LSQR
import pyProblem as Prblm
import inversionUtils
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	# Display information
	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------ Linearized waveform inversion --------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ Linearized waveform inversion --------------")
	if (parObject.getInt("freeSurface",0) == 1):
		print("---------- Using a free surface condition for modeling ------------")

	# Initialize Born
	modelInitDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,dataHyperForOutput=Acoustic_iso_double_3D.BornOpInitDouble_3D(sys.argv)

	# Check if Ginsu is required
	if (parObject.getInt("ginsu",0) == 1):
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu=Acoustic_iso_double_3D.buildGeometryGinsu_3D(parObject,velDouble,sourcesVector,receiversVector)

	# Construct Born operator object - No Ginsu
	if (parObject.getInt("ginsu",0) == 0):
		BornOp=Acoustic_iso_double_3D.BornShotsGpu_3D(modelInitDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector)
	# With Ginsu
	else:
		BornOp=Acoustic_iso_double_3D.BornShotsGpu_3D(modelInitDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)

	# Create inversion operator
	invOp=BornOp

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInitDouble.set(0.0)
	else:
		modelInitDouble=modelInitDouble.getNdArray()
		modelInitFloat=genericIO.defaultIO.getVector(modelInitFile,ndims=3)
		modelInitFloatNp=modelInitFloat.getNdArray()
		modelInitDouble[:]=modelInitFloatNp

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3,storage="dataDouble")

	# No regularization
	invProb=Prblm.ProblemL2Linear(modelInitDouble,data,invOp)

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
