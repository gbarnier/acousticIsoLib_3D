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
# import interpBSplineModule

# Solver library
import pyOperator as pyOp
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as LBFGS
import pyProblem as Prblm
import pyStepper as Stepper
from pyStopper import BasicStopper as Stopper
from sys_util import logger
import inversionUtilsDouble_3D

# Template for FWI workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)

	# Nonlinear solver
	solverType=parObject.getString("solver")
	stepper=parObject.getString("stepper","default")
	gradientMask=parObject.getInt("gradientMask",0)
	spline=parObject.getInt("spline",0)

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtilsDouble_3D.inversionInitDouble_3D(sys.argv)

	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------------ Conventional FWI -------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------------ Conventional FWI -------------------------")

	############################# Initialization ###############################
	# Spline
	if (spline==1):
		modelCoarseInit,modelFineInit,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,fat=interpBSplineModule.bSpline3dInit(sys.argv)

	# Data tapering


	# FWI nonlinear operator
	modelFineInitDouble,dataDouble,sourcesSignalsDouble,parObject,sourcesVector,receiversVector,dataHyperForOutput=Acoustic_iso_double_3D.nonlinearFwiOpInitDouble_3D(sys.argv)

	# Born
	# _,_,_,_,_,_,_,_=Acoustic_iso_double_3D.BornOpInitDouble_3D(sys.argv)

	# Check if Ginsu is required
	if (parObject.getInt("ginsu",0) == 1):
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu=Acoustic_iso_double_3D.buildGeometryGinsu_3D(parObject,modelFineInitDouble,sourcesVector,receiversVector)

	# Gradient mask
	if (gradientMask==1):
		print("--- Initializing gradient mask ---")
		# velLocal,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile=maskGradientModule.maskGradientInit(sys.argv)

	############################### Read data ##################################
	# Data
	dataFile=parObject.getString("data")
	dataFloat=genericIO.defaultIO.getVector(dataFile)
	dataDoubleNp=dataDouble.getNdArray()
	dataFloatNp=dataFloat.getNdArray()

	# Check if we have a regular acquisition geometry
	# if (dataHyperForOutput.getNdim() > 3):
	#
	# 	nTime=dataHyperForOutput.axes[0].n
	# 	nRec=dataHyperForOutput.axes[1].n*dataHyperForOutput.axes[2].n*dataHyperForOutput.axes[3].n
	# 	nShot=dataHyperForOutput.axes[4].n*dataHyperForOutput.axes[5].n*dataHyperForOutput.axes[6].n

	dataDoubleNp.flat[:]=dataFloatNp

	############################# Instantiation ################################
	# No Ginsu
	if (parObject.getInt("ginsu",0) == 0):

		# Nonlinear
		nonlinearFwiOp=Acoustic_iso_double_3D.nonlinearFwiPropShotsGpu_3D(modelFineInitDouble,dataDouble,sourcesSignalsDouble,parObject,sourcesVector,receiversVector)

		# Born
		BornOp=Acoustic_iso_double_3D.BornShotsGpu_3D(modelFineInitDouble,dataDouble,modelFineInitDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector)

	# With Ginsu
	else:

		# Nonlinear
		nonlinearFwiOp=Acoustic_iso_double_3D.nonlinearFwiPropShotsGpu_3D(modelFineInitDouble,dataDouble,sourcesSignalsDouble,parObject,sourcesVector,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,ixVectorGinsu,iyVectorGinsu)

		# Born
		BornOp=Acoustic_iso_double_3D.BornShotsGpu_3D(modelFineInitDouble,dataDouble,modelFineInitDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)

	# Born operator pointer for inversion
	BornInvOp=BornOp

	if (gradientMask==1):
		print("--- Instantiating gradient mask ---")
		# maskGradientOp=maskGradientModule.maskGradient(modelFineInitLocal,modelFineInitLocal,velLocal,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile)
		# BornInvOp=pyOp.ChainOperator(maskGradientOp,BornOp)
		# gMask=maskGradientOp.getMask()

	# Conventional FWI
	fwiInvOp=pyOp.NonLinearOperator(nonlinearFwiOp,BornInvOp,BornOp.setVel_3D)
	modelInit=modelFineInitDouble

	############################### Bounds #####################################
	minBoundVector,maxBoundVector=Acoustic_iso_double_3D.createBoundVectors_3D(parObject,modelInit)

	############################# Problem ######################################
	fwiProb=Prblm.ProblemL2NonLinear(modelInit,dataDouble,fwiInvOp,minBound=minBoundVector,maxBound=maxBoundVector)

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
