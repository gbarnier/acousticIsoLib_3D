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
import dsoGpuModule_3D
import traceNormModule_3D

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as LBFGS
import pyProblem as Prblm
from sys_util import logger
import inversionUtilsFloat_3D

#Dask-related modules
import pyDaskOperator as DaskOp
import pyDaskVector

def call_add_spline(BornOp,SplineOp):
	"""Function to add spline to BornExtOp"""
	BornOp.add_spline_3D(SplineOp)
	return

def call_tomo_wavefield(Tomo_Op):
	"""Function to add spline to BornExtOp"""
	return Tomo_Op.getWavefieldVector()

# Template for FWIME workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)

	# Auxiliary operators
	spline=parObject.getInt("spline")
	dataTaper=parObject.getInt("dataTaper")
	traceNorm=parObject.getInt("traceNorm")
	gradientMask=parObject.getInt("gradientMask")
	print("gradientMask = ",gradientMask)
	rawData=parObject.getInt("rawData",1)
	# fwimeFlag=parObject.getInt("fwime",0)
	# if (fwimeFlag != 1):
	# 	raise ValueError("**** ERROR [fwimeFloatMain_3D]: Please set 'fwime' parameter value to 1 ****\n")

	# Regularization
	regType=parObject.getString("reg")
	reg=0
	if (regType != "None"): reg=1
	epsilonEval=parObject.getInt("epsilonEval",0)

	# Nonlinear solver
	nlSolverType=parObject.getString("nlSolver")
	evalParab=parObject.getInt("evalParab",1)

	# Initialize solvers
	stopNl,logFileNl,saveObjNl,saveResNl,saveGradNl,saveModelNl,invPrefixNl,bufferSizeNl,iterSamplingNl,restartFolderNl,flushMemoryNl,stopLin,logFileLin,saveObjLin,saveResLin,saveGradLin,saveModelLin,invPrefixLin,bufferSizeLin,iterSamplingLin,restartFolderLin,flushMemoryLin,epsilon,info=inversionUtilsFloat_3D.inversionVpInitFloat_3D(sys.argv)

	# Logger
	inv_log = logger(logFileNl)

	# Checking if Dask was requested
	client, nWrks = Acoustic_iso_float_3D.create_client(parObject)

	print("-------------------------------------------------------------------")
	print("------------------------------ FWIME ------------------------------")
	print("-------------------------------------------------------------------\n")

	############################################################################
	############################# Initialization ###############################
	############################################################################

	############################# Seismic utils ################################
	# Spline
	if (spline==1):
		print("---- [fwimeFloatMain_3D]: User has requestd to use a SPLINE parametrization for the velocity model ----")
		inv_log.addToLog("---- [fwimeFloatMain_3D]: User has requestd to use a SPLINE parametrization for the velocity model ----")
		modelCoarseInit,modelFineInitFloat,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat=interpBSplineModule_3D.bSpline3dInit(sys.argv)
	else:
		print("---- [fwimeFloatMain_3D]: User has requestd to use the FINITE-DIFFERENCE grid as a parametrization for the velocity model ----")
		inv_log.addToLog("---- [fwimeFloatMain_3D]: User has requestd to use the FINITE-DIFFERENCE grid as a parametrization for the velocity model ----")

	# Data tapering
	dataMask = None
	if (dataTaper==1):
		print("--- [fwimeFloatMain_3D]: User has requestd to use a data tapering mask for the data ---")
		inv_log.addToLog("--- [fwimeFloatMain_3D]: User has requestd to use a data tapering mask for the data ---")
		t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,time,offset,sourceGeometry,receiverGeometry,dataMask=dataTaperModule_3D.dataTaperInit_3D(sys.argv)

	# Trace normalization
	if (traceNorm==1):
		print("---- [fwimeFloatMain_3D]: User has requestd to use a trace normalization operator on the data ----")
		inv_log.addToLog("--- [fwimeFloatMain_3D]: User has requestd to use a trace normalization operator on the data ---")

	# Gradient mask
	if (gradientMask==1):
		print("--- [fwimeFloatMain_3D]: User has requestd to use a MASK for the gradient ---")
		inv_log.addToLog("--- [fwimeFloatMain_3D]: User has requestd to use a MASK for the gradient ---")
		velDummy,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile,bathymetryFile=maskGradientModule_3D.maskGradientInit_3D(sys.argv)
		print("gradientMaskFile: ", gradientMaskFile)
	else:
		print("--- [fwimeFloatMain_3D]: User has NOT requestd to use a MASK for the gradient ---")
		inv_log.addToLog("--- [fwimeFloatMain_3D]: User has NOT requestd to use a MASK for the gradient ---")

	############################# Seismic operators ############################
	# Nonlinear modeling operator
	modelFineInitFloat,dataFloat,sourcesSignalsFloat,parObject1,sourcesVector,receiversVector,dataHyperForOutput,modelFineInitFloatLocal=Acoustic_iso_float_3D.nonlinearFwiOpInitFloat_3D(sys.argv,client)

	# Born
	_,_,_,_,_,sourcesSignalsFloat,_,_,_=Acoustic_iso_float_3D.BornOpInitFloat_3D(sys.argv,client)

	# Born extended
	reflectivityExtInit,_,velFloat,_,_,_,_,_,reflectivityExtInitLocal=Acoustic_iso_float_3D.BornExtOpInitFloat_3D(sys.argv,client)

	# Tomo extended
	# _,_,_,_,_,_,_,_=Acoustic_iso_float_3D.tomoExtOpInitFloat_3D(sys.argv,client)

	# Dso
	nz,nx,ny,nExt1,nExt2,fat,zeroShift=dsoGpuModule_3D.dsoGpuInit_3D(sys.argv)

	################################# Ginsu ####################################
	# Ginsu
	if (parObject.getInt("ginsu",0) == 1):
		print("--- [fwiFloatMain_3D]: User has requestd to use a Ginsu modeling ---")
		if client:
			raise NotImplementedError("Ginsu option, not supported for Dask interface yet")
		inv_log.addToLog("--- [fwiFloatMain_3D]: User has requestd to use a Ginsu modeling ---")
		velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu=Acoustic_iso_float_3D.buildGeometryGinsu_3D(parObject,modelFineInitFloat,sourcesVector,receiversVector)
	else:
		print("--- [fwiFloatMain_3D]: User has NOT requestd to use a Ginsu modeling ---")
		inv_log.addToLog("--- [fwiFloatMain_3D]: User has NOT requestd to use a Ginsu modeling ---")

	############################# Read files ###################################
	# The initial model is read during the initialization of the nonlinear operator (no need to re-read it)
	# Except for the waveletFile
	# Seismic source
	# waveletFile=parObject.getString("sources")
	# sourcesSignalsFloat=genericIO.defaultIO.getVector(waveletFile,ndims=2)

	# Read initial extended reflectivity
	reflectivityExtInitFile=parObject.getString("reflectivityExtInit","None")
	if (reflectivityExtInitFile=="None"):
		reflectivityExtInitLocal.set(0.0)
	else:
		reflectivityExtInitLocal=genericIO.defaultIO.getVector(reflectivityExtInitFile,ndims=5)

	# Coarse-grid model
	modelInit=modelFineInitFloat
	if (spline==1):
		modelCoarseInitFile=parObject.getString("modelCoarseInit")
		modelCoarseInit=genericIO.defaultIO.getVector(modelCoarseInitFile,ndims=3)
		modelInit=modelCoarseInit
		if client:
			SprdCoarse = DaskOp.DaskSpreadOp(client,modelInit,[1]*nWrks)

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)
	
	############################################################################
	################################ Instanciation #############################
	############################################################################
	# No Ginsu
	if (parObject.getInt("ginsu",0) == 0):
		if client:
			#Spreading operator and concatenating with non-linear and born operators
			Sprd = DaskOp.DaskSpreadOp(client,modelFineInitFloatLocal,[1]*nWrks)
			SprdRefl = DaskOp.DaskSpreadOp(client,reflectivityExtInitLocal,[1]*nWrks)
			SprdMod = SprdCoarse if spline == 1 else Sprd
			# Tomo
			tomoExtOp_args = [(modelFineInitFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsFloat.vecDask[iwrk],receiversVector[iwrk],reflectivityExtInit.vecDask[iwrk]) for iwrk in range(nWrks)]
			tomoExtOpDask = DaskOp.DaskOperator(client,Acoustic_iso_float_3D.tomoExtShotsGpu_3D,tomoExtOp_args,[1]*nWrks,setbackground_func_name="setVel_3D",spread_op=Sprd,set_aux_name="setExtReflectivity_3D",spread_op_aux=SprdRefl)
			wavefieldVecObj = [client.getClient().submit(call_tomo_wavefield, tomo_Op, pure=False) for tomo_Op in tomoExtOp.dask_ops]
			tomoExtOp = pyOp.ChainOperator(Sprd,tomoExtOpDask)
			# Born extended
			BornExtOp_args = [(modelFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsFloat.vecDask[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
			BornExtOpDask = DaskOp.DaskOperator(client,Acoustic_iso_float_3D.BornExtShotsGpu_3D,BornExtOp_args,[1]*nWrks, op_kwargs=wavefieldVecObj,setbackground_func_name="setVel_3D",spread_op=SprdMod)
			BornExtOp = pyOp.ChainOperator(SprdRefl,BornExtOpDask)
			#Adding Spline to BornExt set_vel functions
			if (spline==1):
				#Spreading vectors for spline operator
				modelInitD = pyDaskVector.DaskVector(client,vectors=[modelInit]*nWrks)
				zSplineMeshD = pyDaskVector.DaskVector(client,vectors=[zSplineMesh]*nWrks)
				ySplineMeshD = pyDaskVector.DaskVector(client,vectors=[ySplineMesh]*nWrks)
				xSplineMeshD = pyDaskVector.DaskVector(client,vectors=[xSplineMesh]*nWrks)
				splineOp_args = [(modelInitD.vecDask[iwrk],modelFineInitFloat.vecDask[iwrk],zOrder,yOrder,xOrder,zSplineMeshD.vecDask[iwrk],ySplineMeshD.vecDask[iwrk],xSplineMeshD.vecDask[iwrk],zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat) for iwrk in range(nWrks)]
				splineOpD = DaskOp.DaskOperator(client,interpBSplineModule_3D.bSpline3d,splineOp_args,[1]*nWrks)
				add_spline_ftr = client.getClient().map(call_add_spline,BornExtOpDask.dask_ops,splineOpD.dask_ops,pure=False)
				daskD.wait(add_spline_ftr)
			# Nonlinear
			nonlinearFwdOp_args = [(modelFineInitFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],sourcesSignalsFloat.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
			nonlinearFwdOp = DaskOp.DaskOperator(client,Acoustic_iso_float_3D.nonlinearFwiPropShotsGpu_3D,nonlinearFwdOp_args,[1]*nWrks)
			nonlinearFwdOp = pyOp.ChainOperator(Sprd,nonlinearFwdOp)
			# Born
			BornOp_args = [(modelFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsFloat.vecDask[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
			BornOpDask = DaskOp.DaskOperator(client,Acoustic_iso_float_3D.BornShotsGpu_3D,BornOp_args,[1]*nWrks, op_kwargs=wavefieldVecObj, setbackground_func_name="setVel_3D",spread_op=Sprd)
			BornOp = pyOp.ChainOperator(Sprd,BornOpDask)
		else:
			# Tomo
			tomoExtOp=Acoustic_iso_float_3D.tomoExtShotsGpu_3D(modelFineInitFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityExtInit)
			wavefieldVecObj=tomoExtOp.getWavefieldVector()
			# Born extended
			BornExtOp=Acoustic_iso_float_3D.BornExtShotsGpu_3D(reflectivityExtInit,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,wavefieldVecFlag=wavefieldVecObj)
			# Nonlinear
			nonlinearFwdOp=Acoustic_iso_float_3D.nonlinearFwiPropShotsGpu_3D(modelFineInitFloat,dataFloat,sourcesSignalsFloat,parObject,sourcesVector,receiversVector)
			# Born
			BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelFineInitFloat,dataFloat,modelFineInitFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,wavefieldVecFlag=wavefieldVecObj)
	# Ginsu
	else:
		# Tomo
		tomoExtOp=Acoustic_iso_float_3D.tomoExtShotsGpu_3D(modelFineInitFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityExtInit,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)
		wavefieldVecObj=tomoExtOp.getWavefieldVector()
		# Born extended
		BornExtOp=Acoustic_iso_float_3D.BornExtShotsGpu_3D(reflectivityExtInit,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu,wavefieldVecFlag=wavefieldVecObj)
		# Nonlinear
		nonlinearFwdOp=Acoustic_iso_float_3D.nonlinearFwiPropShotsGpu_3D(modelFineInitFloat,dataFloat,sourcesSignalsFloat,parObject,sourcesVector,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,ixVectorGinsu,iyVectorGinsu)
		# Born
		BornOp=Acoustic_iso_float_3D.BornShotsGpu_3D(modelFineInitFloat,dataFloat,modelFineInitFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu,wavefieldVecFlag=wavefieldVecObj)
	# Born operator pointer for inversion
	BornInvOp=BornOp

	if (gradientMask==1):
		print("Main, gradientMaskFile: ",gradientMaskFile)
		maskGradientOp=maskGradientModule_3D.maskGradient_3D(modelFineInitFloat,modelFineInitFloat,velDummy,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile,bathymetryFile)
		BornInvOp=pyOp.ChainOperator(maskGradientOp,BornOp)
		gMask=maskGradientOp.getMask_3D()

	# g nonlinear (same as conventional FWI operator)
	if client:
		gOp=pyOp.NonLinearOperator(nonlinearFwdOp,BornInvOp,BornOpDask.set_background)
	else:
		gOp=pyOp.NonLinearOperator(nonlinearFwdOp,BornInvOp,BornOp.setVel_3D)
	gInvOp=gOp

	# Diagonal Preconditioning
	PrecFile = parObject.getString("PrecFile","None")
	Precond = None
	if PrecFile != "None":
		print("--- Using diagonal preconditioning ---")
		PrecVec=genericIO.defaultIO.getVector(PrecFile)
		if not PrecVec.checkSame(reflectivityExtInit):
			raise ValueError("ERROR! Preconditioning diagonal inconsistent with model vector")
		Precond = pyOp.DiagonalOp(PrecVec)

	# Spline
	if (spline==1):
		modelInit=modelCoarseInit
		splineOp=interpBSplineModule_3D.bSpline3d(modelCoarseInit,modelFineInitFloat,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp) # Create spline nonlinear operator

	if (spline==1) and client is not None:
			BornExtOp.add_spline_3D(splineOp)

	# Spreading data vectors
	if(client):
			#Chunking the data and spreading them across workers if dask was requested
			data = Acoustic_iso_float_3D.chunkData(data,BornOp.getRange())
			if dataMask is not None:
				dataMask = Acoustic_iso_float_3D.chunkData(dataMask,BornOp.getRange())

	# Trace normalization
	if (traceNorm==1):
		epsilonTraceNorm=parObject.getFloat("epsilonTraceNorm",1e-10)
		if client:
			traceNormOp_args = [(dataFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],epsilonTraceNorm) for iwrk in range(nWrks)]
			traceNormOp = DaskOp.DaskOperator(client,traceNormModule_3D.traceNorm_3D,traceNormOp_args,[1]*nWrks)
			traceNormDerivOp_args = [(dataFloat.vecDask[iwrk],epsilonTraceNorm) for iwrk in range(nWrks)]
			traceNormDerivOp = DaskOp.DaskOperator(client,traceNormModule_3D.traceNormDeriv_3D,traceNormDerivOp_args,[1]*nWrks,setbackground_func_name="setData")
			traceNormNlFwimeOp=pyOp.NonLinearOperator(traceNormOp,traceNormDerivOp,traceNormDerivOp.set_background)
		else:
			# Instantiate nonlinear forward
			traceNormOp=traceNormModule_3D.traceNorm_3D(dataFloat,dataFloat,epsilonTraceNorm)
			# Instantiate Jacobian
			traceNormDerivOp=traceNormModule_3D.traceNormDeriv_3D(dataFloat,epsilonTraceNorm)
			# Instantiate nonlinear operator
			traceNormNlFwimeOp=pyOp.NonLinearOperator(traceNormOp,traceNormDerivOp,traceNormDerivOp.setData)
		# If input data has not been normalized yet -> normalize it
		if (rawData==1):
			if (pyinfo==1):
				print("---- [fwimeFloatMain_3D]: User has required a trace normalization and has provided raw observed data -> applying trace normlization on raw observed data ----")
			inv_log.addToLog("---- [fwimeFloatMain_3D]: User has required a trace normalization and has provided raw observed data -> applying trace normlization on raw observed data ----")
			# Apply normalization to data
			dataNormalized = dataFloat.clone()
			traceNormOp.forward(False,dataFloat,dataNormalized)
			dataFloat=dataNormalized
		# fwiInvOp=pyOp.CombNonlinearOp(fwiInvOp,traceNormNlFwiOp)

	# Data taper
	if (dataTaper==1):
		if client:
			hypers = client.getClient().map(lambda x: x.getHyper(),dataFloat.vecDask,pure=False)
			dataTaper_args = [(dataFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,time,offset,hypers[iwrk],sourceGeometry,receiverGeometry,dataMask.vecDask[iwrk]) for iwrk in range(nWrks)]
			dataTaperOp = DaskOp.DaskOperator(client,dataTaperModule_3D.datTaper,dataTaper_args,[1]*nWrks,setbackground_func_name="setData")
		else:
			# Instantiate operator
			dataTaperOp=dataTaperModule_3D.dataTaper(dataFloat,dataFloat,t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,time,offset,dataFloat.getHyper(),sourceGeometry,receiverGeometry,dataMask)
		# If input data have not been tapered yet -> taper them
		if (rawData==1):
			if (pyinfo==1):
				print("---- [fwiFloatMain_3D]: User has required a data tapering/muting and has provided raw observed data -> applying tapering on raw observed data ----")
			inv_log.addToLog("---- [fwiFloatMain_3D]: User has required a data tapering/muting and has provided raw observed data -> applying tapering on raw observed data ----")
			dataTapered = dataFloat.clone()
			dataTaperOp.forward(False,dataFloat,dataTapered) # Apply tapering to the data
			dataFloat=dataTapered
		dataTaperNlOp=pyOp.NonLinearOperator(dataTaperOp,dataTaperOp) # Create dataTaper nonlinear operator

	# Concatenate operators
	if (spline==1):
		gInvOp=pyOp.CombNonlinearOp(splineNlOp,gInvOp)
	if (traceNorm==1):
		gInvOp=pyOp.CombNonlinearOp(gInvOp,traceNormNlFwimeOp)
	if (dataTaper==1):
		gInvOp=pyOp.CombNonlinearOp(gInvOp,dataTaperNlOp)

	# Instantiation of variable projection operator
	BornExtInvOp=BornExtOp
	if (spline==1):
		BornExtOp.add_spline_3D(splineOp)
	tomoExtInvOp=tomoExtOp

	# Concatenate operators
	if (gradientMask==1 and dataTaper==1):
		tomoTemp1=pyOp.ChainOperator(maskGradientOp,tomoExtInvOp)
		tomoExtInvOp=pyOp.ChainOperator(tomoTemp1,dataTaperOp)
		BornExtInvOp=pyOp.ChainOperator(BornExtInvOp,dataTaperOp)
	if (gradientMask==1 and dataTaper==0):
		tomoExtInvOp=pyOp.ChainOperator(maskGradientOp,tomoExtInvOp)
	if (gradientMask==0 and dataTaper==1):
		BornExtInvOp=pyOp.ChainOperator(BornExtInvOp,dataTaperOp)
		tomoExtInvOp=pyOp.ChainOperator(tomoExtInvOp,dataTaperOp)


	# Dso
	dsoOp=dsoGpuModule_3D.dsoGpu_3D(reflectivityExtInitLocal,reflectivityExtInitLocal,nz,nx,ny,nExt1,nExt2,fat,zeroShift)

	# h nonlinear
	hNonlinearDummyOp=pyOp.ZeroOp(modelFineInitFloatLocal,data)
	if client:
		hNonlinearOp=pyOp.NonLinearOperator(hNonlinearDummyOp,tomoExtInvOp,tomoExtOp.set_background)
	else:
		hNonlinearOp=pyOp.NonLinearOperator(hNonlinearDummyOp,tomoExtInvOp,tomoExtOp.setVel_3D) # We don't need the nonlinear fwd (the residuals are already computed in during the variable projection step)
	hNonlinearInvOp=hNonlinearOp
	if (spline == 1):
		hNonlinearInvOp=pyOp.CombNonlinearOp(splineNlOp,hNonlinearOp) # Combine everything

	# Variable projection operator for the data fitting term
	if client:
		vpOp=pyOp.VpOperator(hNonlinearInvOp,BornExtInvOp,BornExtOp.set_background,tomoExtOp.set_aux)
	else:
		vpOp=pyOp.VpOperator(hNonlinearInvOp,BornExtInvOp,BornExtOp.setVel_3D,tomoExtOp.setExtReflectivity_3D)

	# Regularization operators
	dsoNonlinearJacobian=pyOp.ZeroOp(modelInit,reflectivityExtInit)
	dsoNonlinearDummy=pyOp.ZeroOp(modelInit,reflectivityExtInit)
	dsoNonlinearOp=pyOp.NonLinearOperator(dsoNonlinearDummy,dsoNonlinearJacobian)

	# Variable projection operator for the regularization term
	vpRegOp=pyOp.VpOperator(dsoNonlinearOp,dsoOp,pyOp.dummy_set_background,pyOp.dummy_set_background)

	############################### solver #####################################
	# Initialize solvers
	stopNl,logFileNl,saveObjNl,saveResNl,saveGradNl,saveModelNl,invPrefixNl,bufferSizeNl,iterSamplingNl,restartFolderNl,flushMemoryNl,stopLin,logFileLin,saveObjLin,saveResLin,saveGradLin,saveModelLin,invPrefixLin,bufferSizeLin,iterSamplingLin,restartFolderLin,flushMemoryLin,epsilon,info=inversionUtilsFloat_3D.inversionVpInitFloat_3D(sys.argv)

	# linear solver
	linSolver=LCG(stopLin,logger=logger(logFileLin))
	linSolver.setDefaults(save_obj=saveObjLin,save_res=saveResLin,save_grad=saveGradLin,save_model=saveModelLin,prefix=invPrefixLin,iter_buffer_size=bufferSizeLin,iter_sampling=iterSamplingLin,flush_memory=flushMemoryLin)

	# Nonlinear solver
	if (nlSolverType=="nlcg"):
		nlSolver=NLCG(stopNl,logger=logger(logFileNl))
		if (evalParab==0):
			nlSolver.stepper.eval_parab=False
	elif(nlSolverType=="lbfgs"):
		illumination_file=parObject.getString("illumination","noIllum")
		H0_Op = None
		if illumination_file != "noIllum":
			print("--- Using illumination as initial Hessian inverse ---")
			illumination=genericIO.defaultIO.getVector(illumination_file,ndims=3)
			H0_Op = pyOp.DiagonalOp(illumination)
		# By default, Lbfgs uses MT stepper
		nlSolver=LBFGS(stopNl, H0=H0_Op, logger=logger(logFileNl))
	else:
		print("**** ERROR: User did not provide a nonlinear solver type ****")
		quit()

	# Manual step length for the nonlinear solver
	initStep=parObject.getFloat("initStep",-1)
	if (initStep>0):
		nlSolver.stepper.alpha=initStep

	nlSolver.setDefaults(save_obj=saveObjNl,save_res=saveResNl,save_grad=saveGradNl,save_model=saveModelNl,prefix=invPrefixNl,iter_buffer_size=bufferSizeNl,iter_sampling=iterSamplingNl,flush_memory=flushMemoryNl)

	############################### Bounds #####################################
	minBoundVector,maxBoundVector=Acoustic_iso_float_3D.createBoundVectors_3D(parObject,modelInit)

	######################### Variable projection problem ######################
	vpProb=Prblm.ProblemL2VpReg(modelInit,reflectivityExtInitLocal,vpOp,data,linSolver,gInvOp,h_op_reg=vpRegOp,epsilon=epsilon,minBound=minBoundVector,maxBound=maxBoundVector,prec=Precond)

	################################# Inversion ################################
	# nlSolver.run(vpProb,verbose=info)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")
