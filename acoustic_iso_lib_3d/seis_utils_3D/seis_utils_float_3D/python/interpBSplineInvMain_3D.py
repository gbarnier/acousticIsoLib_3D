#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import interpBSplineModule_3D
import numpy as np
import time
import sys
import os

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
import pyProblem as Prblm
import pyStopper as Stopper
from sys_util import logger

# Template for interpolation optimization (to find a coarse model parameters given a fine model)
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Necessary for multi-parameter inversion
	modelCheck = genericIO.defaultIO.getVector(parObject.getString("vel"))

	if modelCheck.getHyper().getNdim() == 4:
		# Initialize operator
		model,data,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat=interpBSplineModule_3D.bSplineIter3dInit(sys.argv)

		# Construct operator
		splineOp=interpBSplineModule_3D.bSplineIter3d(model,data,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=4)
	else:

		# Initialize operator
		model,data,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat=interpBSplineModule_3D.bSpline3dInit(sys.argv)

		# Construct operator
		splineOp=interpBSplineModule_3D.bSpline3d(model,data,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	# Read starting model
	modelStartFile=parObject.getString("modelStart","None")
	if (modelStartFile=="None"):
		modelStart=model
		modelStart.scale(0.0)
	else:
		modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=3)

	############################## Problem #####################################
	# Problem
	invProb=Prblm.ProblemL2Linear(modelStart,data,splineOp)

	############################## Solver ######################################
	# Stopper
	stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

	# Folder
	folder=parObject.getString("folder")
	if (os.path.isdir(folder)==False): os.mkdir(folder)
	prefix=parObject.getString("prefix","None")
	if (prefix=="None"): prefix=folder
	invPrefix=folder+"/"+prefix
	logFile=invPrefix+"_logFile"

	# Solver recording parameters
	iterSampling=parObject.getInt("iterSampling",1)
	bufferSize=parObject.getInt("bufferSize",-1)
	if (bufferSize<0): bufferSize=None

	# Solver
	LCGsolver=LCG(stop,logger=logger(logFile))
	LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling)

	# Run solver
	LCGsolver.run(invProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")
