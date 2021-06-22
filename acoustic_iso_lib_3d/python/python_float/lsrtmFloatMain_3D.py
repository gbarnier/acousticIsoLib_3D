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
import dataTaperModule_3D

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
from pyLinearSolver import LSQRsolver as LSQR
import pyProblem as Prblm
import inversionUtilsFloat_3D
from sys_util import logger

#Dask-related modules
import pyDaskOperator as DaskOp
import pyDaskVector

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)
	dataTaper=parObject.getInt("dataTaper",0)
	rawData=parObject.getInt("rawData",1)
	pyinfo=parObject.getInt("pyinfo",1)
	solver=parObject.getString("solver","LCG")

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtilsFloat_3D.inversionInitFloat_3D(sys.argv)
	# Logger
	inv_log = logger(logFile)

	# Checking if Dask was requested
	client, nWrks = Acoustic_iso_float_3D.create_client(parObject)

	# Display information
	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------ Linearized waveform inversion ------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ Linearized waveform inversion --------------")

	if (parObject.getInt("freeSurface",0) == 1):
		print("---- [lsrtmFloatMain_3D]: User has requested to use a free surface modeling ----")
		inv_log.addToLog("---- [lsrtmFloatMain_3D]: User has requested to use a free surface modeling ----")

	# Data tapering
	dataMask = None
	if (dataTaper==1):
		print("--- [lsrtmFloatMain_3D]: User has requested to use a data tapering mask for the data ---")
		inv_log.addToLog("--- [lsrtmFloatMain_3D]: User has requested to use a data tapering mask for the data ---")
		t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,time,offset,sourceGeometry,receiverGeometry,dataMask=dataTaperModule_3D.dataTaperInit_3D(sys.argv)

	# Initialize Born
	modelInitFloat,dataFloat,velFloat,parObject1,sourcesVector,sourcesSignalsFloat,receiversVector,dataHyperForOutput,modelFloatLocal=Acoustic_iso_float_3D.BornOpInitFloat_3D(sys.argv,client)

	# Check if Ginsu is required
	if (parObject.getInt("ginsu",0) == 1):
		if client:
			raise NotImplementedError("Ginsu option, not supported for Dask interface yet")
		print("---- [lsrtmFloatMain_3D]: User has requested to use a Ginsu modeling ----")
		inv_log.addToLog("---- [lsrtmFloatMain_3D]: User has requested to use a Ginsu modeling ----")
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu=Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,velFloat,sourcesVector,receiversVector)

	# Construct Born operator object - No Ginsu
	if (parObject.getInt("ginsu",0) == 0):
		if client:
			iwrk = 0
			#Instantiating Dask Operator
			BornOp_args = [(modelInitFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsFloat.vecDask[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
			BornOp = DaskOp.DaskOperator(client,Acoustic_iso_float_3D.BornShotsGpu_3D,BornOp_args,[1]*nWrks)
			#Adding spreading operator and concatenating with Born operator (using modelFloatLocal)
			Sprd = DaskOp.DaskSpreadOp(client,modelFloatLocal,[1]*nWrks)
			BornOp = pyOp.ChainOperator(Sprd,BornOp)
		else:
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
		modelFloatLocal.set(0.0)
		modelInitFloat = modelFloatLocal
	else:
		modelInitFloat=genericIO.defaultIO.getVector(modelInitFile,ndims=3)

	# Data
	dataFile=parObject.getString("data")

    # User provided a source/receiver geometry file
	dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)

	if(client):
			#Chunking the data and spreading them across workers if dask was requested
			dataFloat = Acoustic_iso_float_3D.chunkData(dataFloat,BornOp.getRange())
			if dataMask is not None:
				dataMask = Acoustic_iso_float_3D.chunkData(dataMask,BornOp.getRange())

	############################################################################

	# Data tapering
	if (dataTaper==1):
		if client:
			hypers = client.getClient().map(lambda x: x.getHyper(),dataFloat.vecDask,pure=False)
			dataTaper_args = [(dataFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,time,offset,hypers[iwrk],sourceGeometry,receiverGeometry,dataMask.vecDask[iwrk]) for iwrk in range(nWrks)]
			dataTaperOp = DaskOp.DaskOperator(client,dataTaperModule_3D.datTaper,dataTaper_args,[1]*nWrks)
		else:
			# Instantiate operator
			dataTaperOp=dataTaperModule_3D.dataTaper(dataFloat,dataFloat,t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,time,offset,dataFloat.getHyper(),sourceGeometry,receiverGeometry,dataMask)
		# If input data have not been tapered yet -> taper them
		if (rawData==1):
			if (pyinfo==1):
				print("---- [lsrtmFloatMain_3D]: User has required a data tapering/muting and has provided raw observed data -> applying tapering on raw observed data ----")
			inv_log.addToLog("---- [lsrtmFloatMain_3D]: User has required a data tapering/muting and has provided raw observed data -> applying tapering on raw observed data ----")
			dataTapered = dataFloat.clone()
			dataTaperOp.forward(False,dataFloat,dataTapered) # Apply tapering to the data
			dataFloat=dataTapered
		invOp=pyOp.ChainOperator(invOp,dataTaperOp)


	############################################################################

	# Diagonal Preconditioning
	PrecFile = parObject.getString("PrecFile","None")
	Precond = None
	if PrecFile != "None":
		if(pyinfo): print("--- Using diagonal preconditioning ---")
		inv_log.addToLog("--- Using diagonal preconditioning ---")
		PrecVec=genericIO.defaultIO.getVector(PrecFile,ndims=3)
		if not PrecVec.checkSame(modelInitFloat):
			raise ValueError("ERROR! Preconditioning diagonal inconsistent with model vector")
		Precond = pyOp.DiagonalOp(PrecVec)

	############################################################################

	# Model mask operator
	MaskFile = parObject.getString("ModMask","None")
	ModMask = None
	if MaskFile != "None":
		if(pyinfo): print("--- Using model mask ---")
		inv_log.addToLog("--- Using model mask ---")
		ModMask=genericIO.defaultIO.getVector(MaskFile)
		if not ModMask.checkSame(modelInitFloat):
			raise ValueError("ERROR! Model mask inconsistent with model vector")
		ModMask = pyOp.DiagonalOp(ModMask)
		invOp=pyOp.ChainOperator(ModMask,invOp)

	# No regularization
	invProb=Prblm.ProblemL2Linear(modelInitFloat,dataFloat,invOp,prec=Precond)

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
