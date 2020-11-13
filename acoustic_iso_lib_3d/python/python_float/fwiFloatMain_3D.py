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

# Seismic utils
import interpBSplineModule_3D
import maskGradientModule_3D
import dataTaperModule_3D
import traceNormModule_3D

# Solver library
import pyOperator as pyOp
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as LBFGS
import pyProblem as Prblm
import pyStepper as Stepper
from pyStopper import BasicStopper as Stopper
from sys_util import logger
import inversionUtilsFloat_3D

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
	dataTaper=parObject.getInt("dataTaper",0)
	traceNorm=parObject.getInt("traceNorm",0)
	regType=parObject.getString("reg","None")
	rawData=parObject.getInt("rawData",1)
	reg=0
	if (regType != "None"): reg=1

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtilsFloat_3D.inversionInitFloat_3D(sys.argv)

	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------------ Conventional FWI -------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------------ Conventional FWI -------------------------")

	############################# Initialization ###############################
	# Spline
	if (spline==1):
		print("---- [fwiFloatMain_3D]: User has requestd to use a SPLINE parametrization for the velocity model ----")
		inv_log.addToLog("---- [fwiFloatMain_3D]: User has requestd to use a SPLINE parametrization for the velocity model ----")
		modelCoarseInit,modelFineInit,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat=interpBSplineModule_3D.bSpline3dInit(sys.argv)
	else:
		print("---- [fwiFloatMain_3D]: User has requestd to use the FINITE-DIFFERENCE grid as a parametrization for the velocity model ----")
		inv_log.addToLog("---- [fwiFloatMain_3D]: User has requestd to use the FINITE-DIFFERENCE grid as a parametrization for the velocity model ----")

	# Trace normalization
	if (traceNorm==1):
		print("---- [fwiFloatMain_3D]: User has requestd to use a trace normalization operator on the data ----")
		inv_log.addToLog("--- [fwiFloatMain_3D]: User has requestd to use a trace normalization operator on the data ---")

	# Data tapering
	if (dataTaper==1):
		print("--- [fwiFloatMain_3D]: User has requestd to use a data tapering mask for the data ---")
		inv_log.addToLog("--- [fwiFloatMain_3D]: User has requestd to use a data tapering mask for the data ---")
		t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,tPow,time,offset,sourceGeometry,receiverGeometry,dataMask=dataTaperModule_3D.dataTaperInit_3D(sys.argv)

	# FWI nonlinear operator
	modelFineInitFloat,dataFloat,sourcesSignalsFloat,parObject,sourcesVector,receiversVector,dataHyperForOutput=Acoustic_iso_float_3D.nonlinearFwiOpInitFloat_3D(sys.argv)

	# Ginsu
	if (parObject.getInt("ginsu",0) == 1):
		print("--- [fwiFloatMain_3D]: User has requestd to use a Ginsu modeling ---")
		inv_log.addToLog("--- [fwiFloatMain_3D]: User has requestd to use a Ginsu modeling ---")
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu=Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,modelFineInitFloat,sourcesVector,receiversVector)
	else:
		print("--- [fwiFloatMain_3D]: User has NOT requestd to use a Ginsu modeling ---")
		inv_log.addToLog("--- [fwiFloatMain_3D]: User has NOT requestd to use a Ginsu modeling ---")

	# Gradient mask
	if (gradientMask==1):
		print("--- [fwiFloatMain_3D]: User has requestd to use a MASK for the gradient ---")
		inv_log.addToLog("--- [fwiFloatMain_3D]: User has requestd to use a MASK for the gradient ---")
		velDummy,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile,bathymetryFile=maskGradientModule_3D.maskGradientInit_3D(sys.argv)
	else:
		print("--- [fwiFloatMain_3D]: User has NOT requestd to use a MASK for the gradient ---")
		inv_log.addToLog("--- [fwiFloatMain_3D]: User has NOT requestd to use a MASK for the gradient ---")

	############################### Read files #################################
	# Coarse grid model for spline
	if (spline==1):
		modelCoarseInitFile=parObject.getString("modelCoarseInit")
		modelCoarseInit=genericIO.defaultIO.getVector(modelCoarseInitFile,ndims=3)

	# Data
	dataFile=parObject.getString("data")
	dataFloat=genericIO.defaultIO.getVector(dataFile)

	############################# Instantiation ################################
	# No Ginsu
	if (parObject.getInt("ginsu",0) == 0):
		# Nonlinear
		nonlinearFwiOp=Acoustic_iso_float_3D.nonlinearFwiPropShotsGpu_3D(modelFineInitFloat,dataFloat,sourcesSignalsFloat,parObject,sourcesVector,receiversVector)
		# Born
		BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelFineInitFloat,dataFloat,modelFineInitFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector)

	# With Ginsu
	else:
		# Nonlinear
		nonlinearFwiOp=Acoustic_iso_float_3D.nonlinearFwiPropShotsGpu_3D(modelFineInitFloat,dataFloat,sourcesSignalsFloat,parObject,sourcesVector,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,ixVectorGinsu,iyVectorGinsu)
		# Born
		BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelFineInitFloat,dataFloat,modelFineInitFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)

	# Born operator pointer for inversion
	BornInvOp=BornOp

	if (gradientMask==1):
		maskGradientOp=maskGradientModule_3D.maskGradient_3D(modelFineInitFloat,modelFineInitFloat,velDummy,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile,bathymetryFile)
		if reg == 0:
			# Chain mask if problem is not regularized
			BornInvOp=pyOp.ChainOperator(maskGradientOp,BornOp)
		gMask=maskGradientOp.getMask()

	# Conventional FWI
	fwiInvOp=pyOp.NonLinearOperator(nonlinearFwiOp,BornInvOp,BornOp.setVel_3D)
	modelInit=modelFineInitFloat

	# Spline
	if (spline==1):
		modelInit=modelCoarseInit
		splineOp=interpBSplineModule_3D.bSpline3d(modelCoarseInit,modelFineInitFloat,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp) # Create spline nonlinear operator
		fwiInvOp=pyOp.CombNonlinearOp(splineNlOp,fwiInvOp)

	# Trace normalization
	if (traceNorm==1):
		epsilonTraceNorm=parObject.getFloat("epsilonTraceNorm",1e-10)
		# Instantiate nonlinear forward
		traceNormOp=traceNormModule_3D.traceNorm_3D(dataFloat,dataFloat,epsilonTraceNorm)
		# Instantiate Jacobian
		traceNormDerivOp=traceNormModule_3D.traceNormDeriv_3D(dataFloat,epsilonTraceNorm)
		# Instantiate nonlinear operator
		traceNormNlFwiOp=pyOp.NonLinearOperator(traceNormOp,traceNormDerivOp,traceNormDerivOp.setData)
		# If input data has not been normalized yet -> normalize it
		if (rawData==1):
			if (pyinfo==1):
				print("---- [fwiFloatMain_3D]: User has required a trace normalization and has provided raw observed data -> applying trace normlization on raw observed data ----")
			inv_log.addToLog("---- [fwiFloatMain_3D]: User has required a trace normalization and has provided raw observed data -> applying trace normlization on raw observed data ----")
			# Apply normalization to data
			dataNormalized = dataFloat.clone()
			traceNormOp.forward(False,dataFloat,dataNormalized)
			dataFloat=dataNormalized
		fwiInvOp=pyOp.CombNonlinearOp(fwiInvOp,traceNormNlFwiOp)

	# Data tapering
	if (dataTaper==1):
		# Instantiate operator
		dataTaperOp=dataTaperModule_3D.dataTaper(dataFloat,dataFloat,t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,tPow,time,offset,dataFloat.getHyper(),sourceGeometry,receiverGeometry,dataMask)
		# If input data have not been tapered yet -> taper them
		if (rawData==1):
			if (pyinfo==1):
				print("---- [fwiFloatMain_3D]: User has required a data tapering/muting and has provided raw observed data -> applying tapering on raw observed data ----")
			inv_log.addToLog("---- [fwiFloatMain_3D]: User has required a data tapering/muting and has provided raw observed data -> applying tapering on raw observed data ----")
			dataTapered = dataFloat.clone()
			dataTaperOp.forward(False,dataFloat,dataTapered) # Apply tapering to the data
			dataFloat=dataTapered
		dataTaperNlOp=pyOp.NonLinearOperator(dataTaperOp,dataTaperOp) # Create dataTaper nonlinear operator
		fwiInvOp=pyOp.CombNonlinearOp(fwiInvOp,dataTaperNlOp)

	############################### Bounds #####################################
	minBoundVector,maxBoundVector=Acoustic_iso_float_3D.createBoundVectors_3D(parObject,modelInit)

	############################# Problem ######################################
	fwiProb=Prblm.ProblemL2NonLinear(modelInit,dataFloat,fwiInvOp,minBound=minBoundVector,maxBound=maxBoundVector)

	############################# Solver #######################################
	# Nonlinear conjugate gradient
	if (solverType=="nlcg"):
		nlSolver=NLCG(stop,logger=inv_log)

	# LBFGS
	elif (solverType=="lbfgs"):
		illumination_file=parObject.getString("illumination","noIllum")
		H0_Op = None
		if illumination_file != "noIllum":
			print("--- Using illumination as initial Hessian inverse ---")
			illumination=genericIO.defaultIO.getVector(illumination_file,ndims=3)
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
