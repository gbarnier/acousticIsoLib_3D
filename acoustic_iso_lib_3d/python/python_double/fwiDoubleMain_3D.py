#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Solver library
import pyOperator as pyOp
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as LBFGS
import pyProblem as Prblm
import pyStepper as Stepper
from pyStopper import BasicStopper as Stopper
from sys_util import logger
import inversionUtils

# Template for FWI workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)

	# Nonlinear solver
	solverType=parObject.getString("solver")
	stepper=parObject.getString("stepper","default")

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)

	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------------ Conventional FWI -------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------------ Conventional FWI -------------------------")

	############################# Initialization ###############################
	# FWI nonlinear operator
	modelFineInit,dataInit,wavelet,parObject1,sourcesVector,receiversVector,modelFineInitLocal=Acoustic_iso_float.nonlinearFwiOpInitFloat(sys.argv,client)

	# Born
	_,_,_,_,_,sourcesSignalsVector,_,_=Acoustic_iso_float.BornOpInitFloat(sys.argv,client)

	# Gradient mask
	if (gradientMask==1):
		print("--- Using gradient masking ---")
		velLocal,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile=maskGradientModule.maskGradientInit(sys.argv)

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	#Born operator pointer for inversion
	BornInvOp=BornOp

	if (gradientMask==1):
		maskGradientOp=maskGradientModule.maskGradient(modelFineInitLocal,modelFineInitLocal,velLocal,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile)
		BornInvOp=pyOp.ChainOperator(maskGradientOp,BornOp)
		gMask=maskGradientOp.getMask()

	# Conventional FWI
	if client:
		fwiInvOp=pyOp.NonLinearOperator(nonlinearFwiOp,BornInvOp,BornOpDask.set_background)
	else:
		fwiInvOp=pyOp.NonLinearOperator(nonlinearFwiOp,BornInvOp,BornOp.setVel)
	modelInit=modelFineInitLocal

	############################### Bounds #####################################
	minBoundVector,maxBoundVector=Acoustic_iso_float.createBoundVectors(parObject,modelInit)

	############################# Problem ######################################
	fwiProb=Prblm.ProblemL2NonLinear(modelInit,data,fwiInvOp,minBound=minBoundVector,maxBound=maxBoundVector)
	############################# Solver #######################################
	# Nonlinear conjugate gradient
	if (solverType=="nlcg"):
		nlSolver=NLCG(stop,logger=inv_log)
	# LBFGS
	elif (solverType=="lbfgs"):
		illumination_file=parObject.getString("illumination","noIllum")
		H0_Op = None
		if illumination_file != "noIllum":
			illumination=genericIO.defaultIO.getVector(illumination_file,ndims=2)
			H0_Op = pyOp.DiagonalOp(illumination)
		nlSolver=LBFGS(stop, H0=H0_Op, logger=inv_log)
	# Steepest descent
	elif (solverType=="sd"):
		nlSolver=NLCG(stop,beta_type="SD",logger=inv_log)

	############################# Stepper ######################################
	if (stepper == "parabolic"):
		nlSolver.stepper.eval_parab=True
	elif (stepper == "linear"):
		nlSolver.stepper.eval_parab=False
	elif (stepper == "parabolicNew"):
		print("New parabolic stepper")
		nlSolver.stepper = Stepper.ParabolicStepConst()

	####################### Manual initial step length #########################
	initStep=parObject.getFloat("initStep",-1.0)
	if (initStep>0):
		nlSolver.stepper.alpha=initStep

	nlSolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	nlSolver.run(fwiProb,verbose=info)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
