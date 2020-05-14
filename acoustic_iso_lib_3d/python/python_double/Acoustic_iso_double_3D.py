# Bullshit stuff
import pyAcoustic_iso_double_nl_3D
import pyAcoustic_iso_double_Born_3D
import pyAcoustic_iso_double_BornExt_3D
import pyAcoustic_iso_double_tomoExt_3D
import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np
from pyAcoustic_iso_double_nl_3D import deviceGpu_3D

################################################################################
############################## Sources geometry ################################
################################################################################
# Build source geometry
def buildSourceGeometry_3D(parObject,vel):

	# Common parameters
	info = parObject.getInt("info",0)
	sourceGeomFile = parObject.getString("sourceGeomFile","None")
	nts = parObject.getInt("nts")
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getInt("zDipoleShift",0)
	xDipoleShift = parObject.getInt("xDipoleShift",0)
	yDipoleShift = parObject.getInt("yDipoleShift",0)
	sourcesVector=[] # Sources vector stores deviceGpu objects for each source
	spaceInterpMethod = parObject.getString("spaceInterpMethod","linear")

	# Get total number of shots
	nShot = parObject.getInt("nShot",-1)
	if (nShot==-1):
		raise ValueError("**** ERROR [buildSourceGeometry_3D]: User must provide the total number of shots ****\n")

	# Read parameters for spatial interpolation
	if (spaceInterpMethod == "linear"):
		hFilter1d = 1
	elif (spaceInterpMethod == "sinc"):
		hFilter1d = parObject.getInt("hFilter1d")
	else:
		raise ValueError("**** ERROR [buildSourceGeometry_3D]: Spatial interpolation method requested by user is not implemented ****\n")

	# Display information for user
	if (info == 1):
		print("**** [buildSourceGeometry_3D]: User has requested to display information ****\n")
		# Geometry file
		if (sourceGeomFile != "None"):
			print("**** [buildSourceGeometry_3D]: User has provided a geometry file for the sources' locations ****\n")
		else:
			print("**** [buildSourceGeometry_3D]: User has requested a regular geometry for the sources' locations ****\n")
		# Interpolation method
		if (spaceInterpMethod == "sinc"):
			print("**** [buildSourceGeometry_3D]: User has requested a sinc spatial interpolation method for the sources' signals injection/extraction ****\n")
		else:
			print("**** [buildSourceGeometry_3D]: User has requested a linear spatial interpolation method for the sources' signals injection/extraction ****\n")
		# Dipole injection/extraction
		if (dipole == 1):
			print("**** [buildSourceGeometry_3D]: User has requested a dipole source injection/extraction ****\n")
			print("**** [buildSourceGeometry_3D]: Dipole shift in z-direction: %d [samples], %f [km]  ****\n" %(zDipoleShift, zDipoleShift*vel.getHyper().axes[0].d))
			print("**** [buildSourceGeometry_3D]: Dipole shift in x-direction: %d [samples], %f [km]  ****\n" %(xDipoleShift, xDipoleShift*vel.getHyper().axes[1].d))
			print("**** [buildSourceGeometry_3D]: Dipole shift in y-direction: %d [samples], %f [km]  ****\n" %(yDipoleShift, yDipoleShift*vel.getHyper().axes[2].d))

	# Make sure that the user does not want sinc interpolation + regular geometry
	if ( (spaceInterpMethod == "sinc") and (sourceGeomFile != "None")):
		raise ValueError("**** ERROR [buildSourceGeometry_3D]: User can not request sinc spatial interpolation for a regular source geometry ****\n")

	# Irregular source geometry => Reading source geometry from file
	if(sourceGeomFile != "None"):

		# Set the flag to irregular
		# regSourceGeom = 0

		# Read geometry file
		# 2 axes:
		# First (fast) axis: shot index
		# Second (slow) axis: spatial coordinates
		sourceGeomVectorNd = genericIO.defaultIO.getVector(sourceGeomFile,ndims=2).getNdArray()

		# Create inputs for devceGpu_3D constructor
		# We assume point sources -> zCoordFloat is a 1D array of length 1
		zCoordDouble=SepVector.getSepVector(ns=[1],storage="dataDouble")
		xCoordDouble=SepVector.getSepVector(ns=[1],storage="dataDouble")
		yCoordDouble=SepVector.getSepVector(ns=[1],storage="dataDouble")

		# Check for consistency between number of shots and provided coordinates
		if (nShot != sourceGeomVectorNd.shape[1]):
			raise ValueError("**** ERROR [buildSourceGeometry_3D]: Number of shots from parfile (#shot=%s) not consistent with geometry file (#shots=%s)! ****\n" %(nShot,sourceGeomVectorNd.shape[1]))

		# Generate vector containing deviceGpu_3D objects
		for ishot in range(nShot):

			# Setting z, x and y-positions of the source for the given experiment
			zCoordDouble.set(sourceGeomVectorNd[2,ishot]) # n2, n1
			xCoordDouble.set(sourceGeomVectorNd[0,ishot])
			yCoordDouble.set(sourceGeomVectorNd[1,ishot])

			# Create a deviceGpu_3D for this source and append to sources vector
			sourcesVector.append(deviceGpu_3D(zCoordDouble.getCpp(), xCoordDouble.getCpp(), yCoordDouble.getCpp(), vel.getCpp(), nts, parObject.param, dipole, zDipoleShift, xDipoleShift, yDipoleShift, spaceInterpMethod, hFilter1d))

		# Generate hypercube with one axis which has the length of the number of shots
		shotAxis=Hypercube.axis(n=nShot,o=0.0,d=1.0)
		shotHyper=Hypercube.hypercube(axes=[shotAxis])

	# Reading regular source geometry from parameters
	# Assumes all acquisition devices are located on finite-difference grid points
	else:

		# Set the source geometry flag to "reg"
		# regSourceGeom = 1

		# z-axis
		nzShot = parObject.getInt("nzShot")
		ozShot = parObject.getInt("ozShot") # ozShot is in sample number (i.e., 1 means on the first grid point)
		dzShot = parObject.getInt("dzShot")

		# x-axis
		nxShot = parObject.getInt("nxShot")
		oxShot = parObject.getInt("oxShot")
		dxShot = parObject.getInt("dxShot")

		# y-axis
		nyShot = parObject.getInt("nyShot")
		oyShot = parObject.getInt("oyShot")
		dyShot = parObject.getInt("dyShot")

		# Position of first shot on each axis (convert to grid point index)
		ozShot = ozShot-1 + parObject.getInt("zPadMinus") + parObject.getInt("fat")
		oxShot = oxShot-1 + parObject.getInt("xPadMinus") + parObject.getInt("fat")
		oyShot = oyShot-1 + parObject.getInt("yPad") + parObject.getInt("fat")

		# Model coordinates + dimensions
		dz=vel.getHyper().axes[0].d
		oz=vel.getHyper().axes[0].o
		dx=vel.getHyper().axes[1].d
		ox=vel.getHyper().axes[1].o
		dy=vel.getHyper().axes[2].d
		oy=vel.getHyper().axes[2].o

		# Shot axes
		zShotAxis=Hypercube.axis(n=nzShot,o=oz+ozShot*dz,d=dzShot*dz)
		xShotAxis=Hypercube.axis(n=nxShot,o=ox+oxShot*dx,d=dxShot*dx)
		yShotAxis=Hypercube.axis(n=nyShot,o=oy+oxShot*dy,d=dyShot*dy)
		nShotTemp=nzShot*nxShot*nyShot
		# Check shot number consistency
		if (nShotTemp != nShot):
			raise ValueError("**** ERROR [buildSourceGeometry_3D]: Number of shots not consistent with source acquisition geometry (make sure nShot=nzShot*nxShot*nyShot) ****\n")

		# shotAxis=Hypercube.axis(n=nShot,o=0.0,d=1.0)

		# Generate hypercube
		shotHyper=Hypercube.hypercube(axes=[zShotAxis,xShotAxis,yShotAxis])
		# shotHyperIrreg=Hypercube.hypercube(axes=[shotAxis])

		# Simultaneous shots
		nzSource=1
		dzSource=1
		ozSource=ozShot
		nxSource=1
		dxSource=1
		oxSource=oxShot
		nySource=1
		dySource=1
		oySource=oyShot

		# Create vectors containing deviceGpu_3D objects
		for iyShot in range(nyShot):
			for ixShot in range(nxShot):
				for izShot in range(nzShot):
					sourcesVector.append(deviceGpu_3D(nzSource,ozSource,dzSource,nxSource,oxSource,dxSource,nySource,oySource,dySource,vel.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,"linear",hFilter1d))
					ozSource=ozSource+dzShot # Shift source in z-direction [samples]
				oxSource=oxSource+dxShot # Shift source in x-direction [samples]
			oySource=oySource+dyShot # Shift source in y-direction [samples]

	return sourcesVector,shotHyper

################################################################################
############################## Receivers geometry ##############################
################################################################################
# Build receivers geometry
def buildReceiversGeometry_3D(parObject,vel):

	# Common parameters
	info = parObject.getInt("info",0)
	receiverGeomFile = parObject.getString("receiverGeomFile","None")
	nts = parObject.getInt("nts")
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getInt("zDipoleShift",0)
	xDipoleShift = parObject.getInt("xDipoleShift",0)
	yDipoleShift = parObject.getInt("yDipoleShift",0)
	receiversVector=[]
	spaceInterpMethod = parObject.getString("spaceInterpMethod","linear")

	# Get total number of shots
	nShot = parObject.getInt("nShot",-1)
	if (nShot==-1):
		raise ValueError("**** ERROR [buildSourceGeometry_3D]: User must provide the total number of shots ****\n")

	# Check that user provides the number of receivers per shot (must be constant)
	nReceiverPerShot = parObject.getInt("nReceiverPerShot",-1)
	if (nReceiverPerShot == -1):
		raise ValueError("**** ERROR [buildReceiversGeometry_3D]: User must provide the total number of receivers per shot (this number must be the same for all shots) ****\n")

	# Read parameters for spatial interpolation
	if (spaceInterpMethod == "linear"):
		hFilter1d = 1
	elif (spaceInterpMethod == "sinc"):
		hFilter1d = parObject.getInt("hFilter1d")
	else:
		raise ValueError("**** ERROR [buildReceiversGeometry_3D]: Spatial interpolation method requested by user is not implemented ****\n")

	# Display information for user
	if (info == 1):
		# print("**** [buildReceiversGeometry_3D]: User has requested to display information ****\n")
		if (receiverGeomFile != "None"):
			print("**** [buildReceiversGeometry_3D]: User has provided a geometry file for the receivers' locations ****\n")
		else:
			print("**** [buildReceiversGeometry_3D]: User has requested a regular and constant geometry for the receivers' locations ****\n")
		if (spaceInterpMethod == "sinc"):
			print("**** [buildReceiversGeometry_3D]: User has requested a sinc spatial interpolation method for receivers' signals injection/extraction ****\n")
		else:
			print("**** [buildReceiversGeometry_3D]: User has requested a linear spatial interpolation method for receivers' signals injection/extraction ****\n")
		if (dipole == 1):
			print("**** [buildReceiversGeometry_3D]: User has requested a dipole data injection/extraction ****\n")
			print("**** [buildReceiversGeometry_3D]: Dipole shift in z-direction: %d [samples], %f [km]  ****\n" %(zDipoleShift, zDipoleShift*vel.getHyper().axes[0].d))
			print("**** [buildReceiversGeometry_3D]: Dipole shift in x-direction: %d [samples], %f [km]  ****\n" %(xDipoleShift, xDipoleShift*vel.getHyper().axes[1].d))
			print("**** [buildReceiversGeometry_3D]: Dipole shift in y-direction: %d [samples], %f [km]  ****\n" %(yDipoleShift, yDipoleShift*vel.getHyper().axes[2].d))

	# Check that the user does NOT require sinc interpolation + regular geometry
	if ( (spaceInterpMethod == "sinc") and (receiverGeomFile != "None")):
		raise ValueError("**** ERROR [buildReceiversGeometry_3D]: Code can not handle sinc spatial interpolation for a regular receiver geometry ****\n")

	# Irregular source geometry => Reading source geometry from file
	if(receiverGeomFile != "None"):

		# Read geometry file: 3 axes
		# First (fast) axis: spatial coordinates
		# Second axis: receiver index
		# !!! The number of receivers per shot must be constant !!!
		# Third axis: shot index
		receiverGeomVectorNd = genericIO.defaultIO.getVector(receiverGeomFile,ndims=3).getNdArray()

		# Check consistency with total number of shots
		if (nShot != receiverGeomVectorNd.shape[2]):
			raise ValueError("**** ERROR [buildReceiversGeometry_3D]: Number of shots from parfile (#shot=%s) not consistent with receivers' geometry file (#shots=%s) ****\n"%(nShot,receiverGeomVectorNd.shape[2]))

		# Read size of receivers' geometry file
		# Check consistency between the size of the receiver geometry file and the number of receivers per shot
		if(nReceiverPerShot != receiverGeomVectorNd.shape[1]):
			raise ValueError("**** ERROR [buildReceiversGeometry_3D]: Number of shots from parfile (#shot=%s) not consistent with receivers' geometry file (#shots=%s) ****\n"%(nShot,receiverGeomVectorNd.shape[1]))

		nShot = receiverGeomVectorNd.shape[2]
		if (nShot==1 and info==1):
				print("**** [buildReceiversGeometry_3D]: User has requested a constant geometry (over shots) for receivers ****\n")

		# Create inputs for devceiGpu_3D constructor
		zCoordDouble=SepVector.getSepVector(ns=[nReceiverPerShot],storage="dataDouble")
		xCoordDouble=SepVector.getSepVector(ns=[nReceiverPerShot],storage="dataDouble")
		yCoordDouble=SepVector.getSepVector(ns=[nReceiverPerShot],storage="dataDouble")

		zCoordDoubleNd=zCoordDouble.getNdArray()
		xCoordDoubleNd=xCoordDouble.getNdArray()
		yCoordDoubleNd=yCoordDouble.getNdArray()

		# If receiver geometry is constant -> use only one deviceGpu for the reveivers

		# Generate vector containing deviceGpu_3D objects
		for ishot in range(nShot):

			# Update the receiver's coordinates

			zCoordDoubleNd[:]=receiverGeomVectorNd[2,:,ishot]
			xCoordDoubleNd[:]=receiverGeomVectorNd[0,:,ishot]
			yCoordDoubleNd[:]=receiverGeomVectorNd[1,:,ishot]
			receiversVector.append(deviceGpu_3D(zCoordDouble.getCpp(), xCoordDouble.getCpp(), yCoordDouble.getCpp(), vel.getCpp(), nts, parObject.param, dipole, zDipoleShift, xDipoleShift, yDipoleShift, spaceInterpMethod, hFilter1d))

		# Generate hypercubes
		receiverAxis=Hypercube.axis(n=nReceiverPerShot,o=0.0,d=1.0)
		receiverHyper=Hypercube.hypercube(axes=[receiverAxis])

	else:

		# Reading regular receiver geometry from parameters
		# Assumes all acquisition devices are located on finite-difference grid points
		# Assumes constant receiver geometry (over shots)

		# Set the source geometry flag to "reg"
		# regReceiverGeom = 1

		# z-axis
		nzReceiver = parObject.getInt("nzReceiver")
		ozReceiver = parObject.getInt("ozReceiver") # ozShot is in sample number (i.e., 1 means on the first grid point)
		dzReceiver = parObject.getInt("dzReceiver")

		# x-axis
		nxReceiver = parObject.getInt("nxReceiver")
		oxReceiver = parObject.getInt("oxReceiver")
		dxReceiver = parObject.getInt("dxReceiver")

		# y-axis
		nyReceiver = parObject.getInt("nyReceiver")
		oyReceiver = parObject.getInt("oyReceiver")
		dyReceiver = parObject.getInt("dyReceiver")

		# Position of first receiver on each axis
		ozReceiver = ozReceiver-1 + parObject.getInt("zPadMinus") + parObject.getInt("fat")
		oxReceiver = oxReceiver-1 + parObject.getInt("xPadMinus") + parObject.getInt("fat")
		oyReceiver = oyReceiver-1 + parObject.getInt("yPad") + parObject.getInt("fat")

		# Model coordinates + dimensions
		dz=vel.getHyper().axes[0].d
		oz=vel.getHyper().axes[0].o
		dx=vel.getHyper().axes[1].d
		ox=vel.getHyper().axes[1].o
		dy=vel.getHyper().axes[2].d
		oy=vel.getHyper().axes[2].o

		# Shot axes
		zReceiverAxis=Hypercube.axis(n=nzReceiver,o=oz+ozReceiver*dz,d=dzReceiver*dz)
		xReceiverAxis=Hypercube.axis(n=nxReceiver,o=ox+oxReceiver*dx,d=dxReceiver*dx)
		yReceiverAxis=Hypercube.axis(n=nyReceiver,o=oy+oyReceiver*dy,d=dyReceiver*dy)
		nReceiver=nzReceiver*nxReceiver*nyReceiver
		if(nReceiverPerShot != nReceiver):
			raise ValueError("**** ERROR [buildReceiversGeometry_3D]: Number of receivers per shot from tag nReceiverPerShot (nReceiverPerShot=%s) not consistent product nReceiver = nzReceiver x nxReceiver x nyReceiver (nReceiver=%s) ****\n"%(nReceiverPerShot,nReceiver))
		receiverAxis=Hypercube.axis(n=nReceiver,o=0.0,d=1.0)

		# Generate hypercubes
		receiverHyper=Hypercube.hypercube(axes=[zReceiverAxis,xReceiverAxis,yReceiverAxis])
		# receiverHyperIrreg=Hypercube.hypercube(axes=[receiverAxis])

		# Create vector containing 1 deviceGpu_3D object
		receiversVector.append(deviceGpu_3D(nzReceiver,ozReceiver,dzReceiver,nxReceiver,oxReceiver,dxReceiver,nyReceiver,oyReceiver,dyReceiver,vel.getCpp(),nts,parObject.param,dipole,zDipoleShift,xDipoleShift,yDipoleShift,"linear",hFilter1d))

	return receiversVector,receiverHyper

################################################################################
############################ Nonlinear propagation #############################
################################################################################
def nonlinearOpInitDouble_3D(args):

	"""Function to correctly initialize nonlinear 3D operator
	   The function will return the necessary variables for operator construction
	"""
	# IO object
	parObject=genericIO.io(params=args)

	# Read flag to display information
	info = parObject.getInt("info",0)
	if (info == 1):
		print("**** [nonlinearOpInitDouble_3D]: User has requested to display information ****\n")

	# Read velocity and convert to double
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		raise ValueError("**** ERROR [nonlinearOpInitDouble_3D]: User did not provide velocity file ****\n")
	velFloat=genericIO.defaultIO.getVector(velFile)
	velDouble=SepVector.getSepVector(velFloat.getHyper(),storage="dataDouble")
	velDoubleNp=velDouble.getNdArray()
	velFloatNp=velFloat.getNdArray()
	velDoubleNp[:]=velFloatNp

	# Determine if the source time signature is constant over shots
	constantWavelet = parObject.getInt("constantWavelet",-1)
	if (info == 1 and constantWavelet == 1):
		print("**** [nonlinearOpInitDouble_3D]: Using the same seismic source time signature for all shots ****\n")
	if (info == 1 and constantWavelet == 0):
		print("**** [nonlinearOpInitDouble_3D]: Using different seismic source time signatures for each shot ****\n")
	if (constantWavelet != 0 and constantWavelet != 1):
		raise ValueError("**** ERROR [nonlinearOpInitDouble_3D]: User did not specify an acceptable value for tag constantWavelet (must be 0 or 1) ****\n")

	# Build sources/receivers geometry
	sourcesVector,shotHyper=buildSourceGeometry_3D(parObject,velDouble)
	receiversVector,receiverHyper=buildReceiversGeometry_3D(parObject,velDouble)

	# Compute the number of shots
	if (shotHyper.getNdim() > 1):
		# Case where we have a regular source geometry (the hypercube has 3 axes)
		nShot=shotHyper.axes[0].n*shotHyper.axes[1].n*shotHyper.axes[2].n
		zShotAxis=shotHyper.axes[0]
		xShotAxis=shotHyper.axes[1]
		yShotAxis=shotHyper.axes[2]
	else:
		# Case where we have an irregular geometry (the shot hypercube has one axis)
		nShot=shotHyper.axes[0].n

	# Create shot axis for the modeling
	shotAxis=Hypercube.axis(n=nShot)

	# Compute the number of receivers per shot (the number is constant for all shots)
	if (receiverHyper.getNdim() > 1):
		# Regular geometry
		nReceiver=receiverHyper.axes[0].n*receiverHyper.axes[1].n*receiverHyper.axes[2].n
		zReceiverAxis=receiverHyper.axes[0]
		xReceiverAxis=receiverHyper.axes[1]
		yReceiverAxis=receiverHyper.axes[2]
	else:
		# Irregular geometry
		nReceiver=receiverHyper.axes[0].n

	# Create receiver axis for the modeling
	receiverAxis=Hypercube.axis(n=nReceiver)

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate model
	# If we use a constant wavelet => allocate a 2D array where the second axis has a length of 1
	if (constantWavelet == 1):
		dummyAxis=Hypercube.axis(n=1)
		modelHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis])
		modelDouble=SepVector.getSepVector(modelHyper,storage="dataDouble")

	else:
		# If we do not use a constant wavelet
		# Allocate a 2D array where the second axis has a length of the total number of shots
		modelHyper=Hypercube.hypercube(axes=[timeAxis,shotAxis])
		modelDouble=SepVector.getSepVector(modelHyper,storage="dataDouble")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,shotAxis])
	dataDouble=SepVector.getSepVector(dataHyper,storage="dataDouble")

	# Create data hypercube for writing the data to disk
	# Regular geometry for both the sources and receivers
	if (shotHyper.getNdim()>1 and receiverHyper.getNdim()>1):
		dataHyperForOutput=Hypercube.hypercube(axes=[timeAxis,zReceiverAxis,xReceiverAxis,yReceiverAxis,zShotAxis,xShotAxis,yShotAxis])
	# Irregular geometry for both the sources and receivers
	if (shotHyper.getNdim()==1 and receiverHyper.getNdim()==1):
		dataHyperForOutput=dataHyper

	# Outputs
	return modelDouble,dataDouble,velDouble,parObject,sourcesVector,receiversVector,dataHyperForOutput

class nonlinearPropShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear 3D propagator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,receiversVector):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		self.pyOp = pyAcoustic_iso_double_nl_3D.nonlinearPropShotsGpu_3D(velocity,paramP.param,sourceVector,receiversVector)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double_nl_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double_nl_3D.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def setVel_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_double_nl_3D.ostream_redirect():
			self.pyOp.setVel_3D(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_double_nl_3D.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

	def getDampVolumeShots_3D(self):
		with pyAcoustic_iso_double_nl_3D.ostream_redirect():
			dampVolume = self.pyOp.getDampVolumeShots_3D()
			dampVolume = SepVector.floatVector(fromCpp=dampVolume)
		return dampVolume

################################################################################
################################### Born #######################################
################################################################################
def BornOpInitDouble_3D(args,client=None):
	"""Function to correctly initialize Born operator
	   The function will return the necessary variables for operator construction
	"""

	# IO object
	parObject=genericIO.io(params=args)

	# Read flag to display information
	info = parObject.getInt("info",0)
	if (info == 1):
		print("**** [BornOpInitDouble_3D]: User has requested to display information ****\n")

	# Read velocity and convert to double
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		raise ValueError("**** ERROR [BornOpInitDouble_3D]: User did not provide velocity file ****\n")
	velFloat=genericIO.defaultIO.getVector(velFile)
	velDouble=SepVector.getSepVector(velFloat.getHyper(),storage="dataDouble")
	velDoubleNp=velDouble.getNdArray()
	velFloatNp=velFloat.getNdArray()
	velDoubleNp[:]=velFloatNp

	# Determine if the source time signature is constant over shots
	constantWavelet = parObject.getInt("constantWavelet",-1)
	if (info == 1 and constantWavelet == 1):
		print("**** [BornOpInitDouble_3D]: Using the same seismic source time signature for all shots ****\n")
	if (info == 1 and constantWavelet == 0):
		print("**** [BornOpInitDouble_3D]: Using different seismic source time signatures for each shot ****\n")
	if (constantWavelet != 0 and constantWavelet != 1):
		raise ValueError("**** ERROR [BornOpInitDouble_3D]: User did not specify an acceptable value for tag constantWavelet (must be 0 or 1) ****\n")

	# Build sources/receivers geometry
	sourcesVector,shotHyper=buildSourceGeometry_3D(parObject,velDouble)
	receiversVector,receiverHyper=buildReceiversGeometry_3D(parObject,velDouble)

	# Compute the number of shots
	if (shotHyper.getNdim() > 1):
		# Case where we have a regular source geometry (the hypercube has 3 axes)
		nShot=shotHyper.axes[0].n*shotHyper.axes[1].n*shotHyper.axes[2].n
		zShotAxis=shotHyper.axes[0]
		xShotAxis=shotHyper.axes[1]
		yShotAxis=shotHyper.axes[2]
	else:
		# Case where we have an irregular geometry (the shot hypercube has one axis)
		nShot=shotHyper.axes[0].n

	# Create shot axis for the modeling
	shotAxis=Hypercube.axis(n=nShot)

	# Compute the number of receivers per shot (the number is constant for all shots for both regular/irregular geometries)
	if (receiverHyper.getNdim() > 1):
		# Regular geometry
		nReceiver=receiverHyper.axes[0].n*receiverHyper.axes[1].n*receiverHyper.axes[2].n
		zReceiverAxis=receiverHyper.axes[0]
		xReceiverAxis=receiverHyper.axes[1]
		yReceiverAxis=receiverHyper.axes[2]
	else:
		# Irregular geometry
		nReceiver=receiverHyper.axes[0].n

	# Create receiver axis for the modeling
	receiverAxis=Hypercube.axis(n=nReceiver)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate model
	modelDouble=SepVector.getSepVector(velDouble.getHyper(),storage="dataDouble")

	# Read velocity and convert to double
	sourcesSignalsFile=parObject.getString("sources","noSourcesFile")
	if (sourcesSignalsFile == "noSourcesFile"):
		raise IOError("**** ERROR [BornOpInitDouble_3D]: User did not provide seismic sources file ****\n")
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesSignalsFile,ndims=2)
	sourcesSignalsDouble=SepVector.getSepVector(sourcesSignalsFloat.getHyper(),storage="dataDouble")
	sourcesSignalsDoubleNp=sourcesSignalsDouble.getNdArray()
	sourcesSignalsFloatNp=sourcesSignalsFloat.getNdArray()
	sourcesSignalsDoubleNp[:]=sourcesSignalsFloatNp

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,shotAxis])
	dataDouble=SepVector.getSepVector(dataHyper,storage="dataDouble")

	# Create data hypercurbe for writing the data to disk
	# Regular geometry for both the sources and receivers
	if (shotHyper.getNdim()>1 and receiverHyper.getNdim()>1):
		dataHyperForOutput=Hypercube.hypercube(axes=[timeAxis,zReceiverAxis,xReceiverAxis,yReceiverAxis,zShotAxis,xShotAxis,yShotAxis])
	# Irregular geometry for both the sources and receivers
	if (shotHyper.getNdim()==1 and receiverHyper.getNdim==1):
		dataHyperForOutput=dataHyper

	return modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,dataHyperForOutput

class BornShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsDouble,receiversVector):
		# Domain = reflectivity
		# Range = Born data
		self.setDomainRange(domain,range)

		# print("domain nDim=",domain.getHyper().getNdim())
		# print("domain n1=",domain.getHyper().axes[0].n)
		# print("domain n2=",domain.getHyper().axes[1].n)
		# print("domain n3=",domain.getHyper().axes[2].n)
		#
		# print("range nDim=",range.getHyper().getNdim())
		# print("range n1=",range.getHyper().axes[0].n)
		# print("range n2=",range.getHyper().axes[1].n)
		# print("range n3=",range.getHyper().axes[2].n)

		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(sourcesSignalsDouble)):
			sourcesSignalsDouble = sourcesSignalsDouble.getCpp()
		self.pyOp = pyAcoustic_iso_double_Born_3D.BornShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsDouble,receiversVector)
		return

	def __str__(self):
		"""Name of the operator"""
		return " BornOp "

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double_Born_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		# QC data size
		# print("Inside adjoint")
		# print("model nDim=",model.getHyper().getNdim())
		# print("model n1=",model.getHyper().axes[0].n)
		# print("model n2=",model.getHyper().axes[1].n)
		# print("model n3=",model.getHyper().axes[2].n)
		# print("data nDim=",data.getHyper().getNdim())
		# print("data n1=",data.getHyper().axes[0].n)
		# print("data n2=",data.getHyper().axes[1].n)
		# print("data n3=",data.getHyper().axes[2].n)

		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double_Born_3D.ostream_redirect():
			# print("Just before")
			self.pyOp.adjoint(add,model,data)
		return

	def setVel_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_double_Born_3D.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_double_Born_3D.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

	def getSrcWavefield_3D(self,iWavefield):
		with pyAcoustic_iso_double_Born_3D.ostream_redirect():
			wfld = self.pyOp.getSrcWavefield_3D(iWavefield)
			wfld = SepVector.floatVector(fromCpp=wfld)
		return wfld

################################################################################
############################## Born extended ###################################
################################################################################
def BornExtOpInitDouble_3D(args,client=None):
	"""Function to correctly initialize Born extended operator
	   The function will return the necessary variables for operator construction
	"""

	# IO object
	parObject=genericIO.io(params=args)

	# Read flag to display information
	info = parObject.getInt("info",0)
	if (info == 1):
		print("**** [BornExtOpInitDouble_3D]: User has requested to display information ****\n")

	# Read velocity and convert to double
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		raise ValueError("**** ERROR [BornExtOpInitDouble_3D]: User did not provide velocity file ****\n")
	velFloat=genericIO.defaultIO.getVector(velFile)
	velDouble=SepVector.getSepVector(velFloat.getHyper(),storage="dataDouble")
	velDoubleNp=velDouble.getNdArray()
	velFloatNp=velFloat.getNdArray()
	velDoubleNp[:]=velFloatNp

	# Determine if the source time signature is constant over shots
	constantWavelet = parObject.getInt("constantWavelet",-1)
	if (info == 1 and constantWavelet == 1):
		print("**** [BornExtOpInitDouble_3D]: Using the same seismic source time signature for all shots ****\n")
	if (info == 1 and constantWavelet == 0):
		print("**** [BornExtOpInitDouble_3D]: Using different seismic source time signatures for each shot ****\n")
	if (constantWavelet != 0 and constantWavelet != 1):
		raise ValueError("**** ERROR [BornExtOpInitDouble_3D]: User did not specify an acceptable value for tag constantWavelet (must be 0 or 1) ****\n")

	# Build sources/receivers geometry
	sourcesVector,shotHyper=buildSourceGeometry_3D(parObject,velDouble)
	receiversVector,receiverHyper=buildReceiversGeometry_3D(parObject,velDouble)

	# Compute the number of shots
	if (shotHyper.getNdim() > 1):
		# Case where we have a regular source geometry (the hypercube has 3 axes)
		nShot=shotHyper.axes[0].n*shotHyper.axes[1].n*shotHyper.axes[2].n
		zShotAxis=shotHyper.axes[0]
		xShotAxis=shotHyper.axes[1]
		yShotAxis=shotHyper.axes[2]
	else:
		# Case where we have an irregular geometry (the shot hypercube has one axis)
		nShot=shotHyper.axes[0].n

	# Create shot axis for the modeling
	shotAxis=Hypercube.axis(n=nShot)

	# Compute the number of receivers per shot (the number is constant for all shots for both regular/irregular geometries)
	if (receiverHyper.getNdim() > 1):
		# Regular geometry
		nReceiver=receiverHyper.axes[0].n*receiverHyper.axes[1].n*receiverHyper.axes[2].n
		zReceiverAxis=receiverHyper.axes[0]
		xReceiverAxis=receiverHyper.axes[1]
		yReceiverAxis=receiverHyper.axes[2]
	else:
		# Irregular geometry
		nReceiver=receiverHyper.axes[0].n

	# Create receiver axis for the modeling
	receiverAxis=Hypercube.axis(n=nReceiver)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Read velocity and convert to double
	sourcesSignalsFile=parObject.getString("sources","noSourcesFile")
	if (sourcesSignalsFile == "noSourcesFile"):
		raise IOError("**** ERROR [BornExtOpInitDouble_3D]: User did not provide seismic sources file ****\n")
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesSignalsFile,ndims=2)
	sourcesSignalsDouble=SepVector.getSepVector(sourcesSignalsFloat.getHyper(),storage="dataDouble")
	sourcesSignalsDoubleNp=sourcesSignalsDouble.getNdArray()
	sourcesSignalsFloatNp=sourcesSignalsFloat.getNdArray()
	sourcesSignalsDoubleNp[:]=sourcesSignalsFloatNp

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,shotAxis])
	dataDouble=SepVector.getSepVector(dataHyper,storage="dataDouble")

	# Create data hypercurbe for writing the data to disk
	# Regular geometry for both the sources and receivers
	if (shotHyper.getNdim()>1 and receiverHyper.getNdim()>1):
		dataHyperForOutput=Hypercube.hypercube(axes=[timeAxis,zReceiverAxis,xReceiverAxis,yReceiverAxis,zShotAxis,xShotAxis,yShotAxis])
	# Irregular geometry for both the sources and receivers
	if (shotHyper.getNdim()==1 and receiverHyper.getNdim==1):
		dataHyperForOutput=dataHyper

	# Allocate model
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR [BornExtOpInitDouble_3D]: User did not provide extension type ****\n")
		quit()

	nExt1=parObject.getInt("nExt1", -1)
	if (nExt1 == -1):
		print("**** ERROR [BornExtOpInitDouble_3D]: User did not provide size of extension axis #1 ****\n")
		quit()
	if (nExt1%2 ==0):
		print("Length of extension axis #1 must be an uneven number")
		quit()

	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR [BornExtOpInitDouble_3D]: User did not provide extension type ****\n")
		quit()

	nExt2=parObject.getInt("nExt2", -1)
	if (nExt2 == -1):
		print("**** ERROR [BornExtOpInitDouble_3D]: User did not provide size of extension axis #2 ****\n")
		quit()
	if (nExt2%2 ==0):
		print("Length of extension axis #2 must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt1=parObject.getFloat("dts",-1.0)
		hExt1=(nExt1-1)/2
		oExt1=-dExt1*hExt1
		dExt2=parObject.getFloat("dts",-1.0)
		hExt2=(nExt2-1)/2
		oExt2=-dExt2*hExt2

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt1=parObject.getFloat("dx",-1.0)
		hExt1=(nExt1-1)/2
		oExt1=-dExt1*hExt1
		dExt2=parObject.getFloat("dy",-1.0)
		hExt2=(nExt2-1)/2
		oExt2=-dExt2*hExt2

	# Space axes
	zAxis=velFloat.getHyper().axes[0]
	xAxis=velFloat.getHyper().axes[1]
	yAxis=velFloat.getHyper().axes[2]
	ext1Axis=Hypercube.axis(n=nExt1,o=oExt1,d=dExt1) # Create extended axis
	ext2Axis=Hypercube.axis(n=nExt2,o=oExt2,d=dExt2) # Create extended axis

	modelDouble=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,yAxis,ext1Axis, ext2Axis]),storage="dataDouble")

	# print("model nDim=",modelDouble.getHyper().getNdim())
	# print("model n1=",modelDouble.getHyper().axes[0].n)
	# print("model n2=",modelDouble.getHyper().axes[1].n)
	# print("model n3=",modelDouble.getHyper().axes[2].n)
	# print("model n4=",modelDouble.getHyper().axes[3].n)
	# print("model n5=",modelDouble.getHyper().axes[4].n)

	return modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,dataHyperForOutput

class BornExtShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsDouble,receiversVector):
		# Domain = reflectivity
		# Range = Born data
		self.setDomainRange(domain,range)

		# print("domain nDim=",domain.getHyper().getNdim())
		# print("domain n1=",domain.getHyper().axes[0].n)
		# print("domain n2=",domain.getHyper().axes[1].n)
		# print("domain n3=",domain.getHyper().axes[2].n)
		#
		# print("range nDim=",range.getHyper().getNdim())
		# print("range n1=",range.getHyper().axes[0].n)
		# print("range n2=",range.getHyper().axes[1].n)
		# print("range n3=",range.getHyper().axes[2].n)

		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(sourcesSignalsDouble)):
			sourcesSignalsDouble = sourcesSignalsDouble.getCpp()
		self.pyOp = pyAcoustic_iso_double_BornExt_3D.BornExtShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsDouble,receiversVector)
		return

	def __str__(self):
		"""Name of the operator"""
		return " BornOp "

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double_BornExt_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		# QC data size
		# print("Inside adjoint")
		# print("model nDim=",model.getHyper().getNdim())
		# print("model n1=",model.getHyper().axes[0].n)
		# print("model n2=",model.getHyper().axes[1].n)
		# print("model n3=",model.getHyper().axes[2].n)
		# print("data nDim=",data.getHyper().getNdim())
		# print("data n1=",data.getHyper().axes[0].n)
		# print("data n2=",data.getHyper().axes[1].n)
		# print("data n3=",data.getHyper().axes[2].n)

		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double_BornExt_3D.ostream_redirect():
			# print("Just before")
			self.pyOp.adjoint(add,model,data)
		return

	def setVel_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_double_BornExt_3D.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_double_BornExt_3D.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

	def getSrcWavefield_3D(self,iWavefield):
		with pyAcoustic_iso_double_BornExt_3D.ostream_redirect():
			wfld = self.pyOp.getSrcWavefield_3D(iWavefield)
			wfld = SepVector.floatVector(fromCpp=wfld)
		return wfld

################################################################################
############################## Tomo extended ###################################
################################################################################
def tomoExtOpInitDouble_3D(args,client=None):
	"""Function to correctly initialize Born extended operator
	   The function will return the necessary variables for operator construction
	"""

	# IO object
	parObject=genericIO.io(params=args)

	# Read flag to display information
	info = parObject.getInt("info",0)
	if (info == 1):
		print("**** [BornExtOpInitDouble_3D]: User has requested to display information ****\n")

	############################## Velocity ####################################
	# Read velocity and convert to double
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		raise ValueError("**** ERROR [tomoExtOpInitDouble_3D]: User did not provide velocity file ****\n")
	velFloat=genericIO.defaultIO.getVector(velFile)
	velDouble=SepVector.getSepVector(velFloat.getHyper(),storage="dataDouble")
	velDoubleNp=velDouble.getNdArray()
	velFloatNp=velFloat.getNdArray()
	velDoubleNp[:]=velFloatNp

	############################## Sources #####################################
	# Determine if the source time signature is constant over shots
	constantWavelet = parObject.getInt("constantWavelet",-1)
	if (info == 1 and constantWavelet == 1):
		print("**** [tomoExtOpInitDouble_3D]: Using the same seismic source time signature for all shots ****\n")
	if (info == 1 and constantWavelet == 0):
		print("**** [tomoExtOpInitDouble_3D]: Using different seismic source time signatures for each shot ****\n")
	if (constantWavelet != 0 and constantWavelet != 1):
		raise ValueError("**** ERROR [tomoExtOpInitDouble_3D]: User did not specify an acceptable value for tag constantWavelet (must be 0 or 1) ****\n")

	# Build sources geometry
	sourcesVector,shotHyper=buildSourceGeometry_3D(parObject,velDouble)

	# Compute the number of shots
	if (shotHyper.getNdim() > 1):
		# Case where we have a regular source geometry (the hypercube has 3 axes)
		nShot=shotHyper.axes[0].n*shotHyper.axes[1].n*shotHyper.axes[2].n
		zShotAxis=shotHyper.axes[0]
		xShotAxis=shotHyper.axes[1]
		yShotAxis=shotHyper.axes[2]
	else:
		# Case where we have an irregular geometry (the shot hypercube has one axis)
		nShot=shotHyper.axes[0].n

	# Create shot axis for the modeling
	shotAxis=Hypercube.axis(n=nShot)

	# Read sources signals and convert to double
	sourcesSignalsFile=parObject.getString("sources","noSourcesFile")
	if (sourcesSignalsFile == "noSourcesFile"):
		raise IOError("**** ERROR [tomoExtOpInitDouble_3D]: User did not provide seismic sources file ****\n")
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesSignalsFile,ndims=2)
	sourcesSignalsDouble=SepVector.getSepVector(sourcesSignalsFloat.getHyper(),storage="dataDouble")
	sourcesSignalsDoubleNp=sourcesSignalsDouble.getNdArray()
	sourcesSignalsFloatNp=sourcesSignalsFloat.getNdArray()
	sourcesSignalsDoubleNp[:]=sourcesSignalsFloatNp

	############################## Receivers ###################################
	# Build receivers geometry
	receiversVector,receiverHyper=buildReceiversGeometry_3D(parObject,velDouble)

	# Compute the number of receivers per shot (the number is constant for all shots for both regular/irregular geometries)
	if (receiverHyper.getNdim() > 1):
		# Regular geometry
		nReceiver=receiverHyper.axes[0].n*receiverHyper.axes[1].n*receiverHyper.axes[2].n
		zReceiverAxis=receiverHyper.axes[0]
		xReceiverAxis=receiverHyper.axes[1]
		yReceiverAxis=receiverHyper.axes[2]
	else:
		# Irregular geometry
		nReceiver=receiverHyper.axes[0].n

	# Create receiver axis for the modeling
	receiverAxis=Hypercube.axis(n=nReceiver)

	############################## Axes ########################################
	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Space axes
	zAxis=velFloat.getHyper().axes[0]
	xAxis=velFloat.getHyper().axes[1]
	yAxis=velFloat.getHyper().axes[2]

	# Extended axes - QC
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR [tomoExtOpInitDouble_3D]: User did not provide extension type ****\n")
		quit()

	nExt1=parObject.getInt("nExt1", -1)
	if (nExt1 == -1):
		print("**** ERROR [tomoExtOpInitDouble_3D]: User did not provide size of extension axis #1 ****\n")
		quit()
	if (nExt1%2 ==0):
		print("Length of extension axis #1 must be an uneven number")
		quit()

	nExt2=parObject.getInt("nExt2", -1)
	if (nExt2 == -1):
		print("**** ERROR [tomoExtOpInitDouble_3D]: User did not provide size of extension axis #2 ****\n")
		quit()
	if (nExt2%2 ==0):
		print("Length of extension axis #2 must be an uneven number")
		quit()

	# Time-lag extension
	if (extension == "time"):
		dExt1=parObject.getFloat("dts",-1.0)
		hExt1=(nExt1-1)/2
		oExt1=-dExt1*hExt1
		dExt2=parObject.getFloat("dts",-1.0)
		hExt2=(nExt2-1)/2
		oExt2=-dExt2*hExt2

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt1=parObject.getFloat("dx",-1.0)
		hExt1=(nExt1-1)/2
		oExt1=-dExt1*hExt1
		dExt2=parObject.getFloat("dy",-1.0)
		hExt2=(nExt2-1)/2
		oExt2=-dExt2*hExt2

	extAxis1=Hypercube.axis(n=nExt1,o=oExt1,d=dExt1) # Create extended axis #1
	extAxis2=Hypercube.axis(n=nExt2,o=oExt2,d=dExt2) # Create extended axis #2

	######################## Extended reflectivity #############################
	# Read extended reflectivity and convert to double
	reflectivityFile=parObject.getString("reflectivity","noReflectivityFile")
	if (reflectivityFile == "noReflectivityFile"):
		raise ValueError("**** ERROR [tomoExtOpInitDouble_3D]: User did not provide reflectivity file ****\n")
	reflectivityFloat=genericIO.defaultIO.getVector(reflectivityFile,ndims=5)
	reflectivityDouble=SepVector.getSepVector(reflectivityFloat.getHyper(),storage="dataDouble")
	reflectivityDoubleNp=reflectivityDouble.getNdArray()
	reflectivityFloatNp=reflectivityFloat.getNdArray()
	reflectivityDoubleNp[:]=reflectivityFloatNp

	############################## Data ########################################
	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,shotAxis])
	dataDouble=SepVector.getSepVector(dataHyper,storage="dataDouble")

	# Create data hypercurbe for writing the data to disk
	# Regular geometry for both the sources and receivers
	if (shotHyper.getNdim()>1 and receiverHyper.getNdim()>1):
		dataHyperForOutput=Hypercube.hypercube(axes=[timeAxis,zReceiverAxis,xReceiverAxis,yReceiverAxis,zShotAxis,xShotAxis,yShotAxis])
	# Irregular geometry for both the sources and receivers
	if (shotHyper.getNdim()==1 and receiverHyper.getNdim==1):
		dataHyperForOutput=dataHyper

	############################## Model #######################################
	modelDouble=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,yAxis]),storage="dataDouble")

	############################## Output ######################################
	return modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsDouble,receiversVector,reflectivityDouble,dataHyperForOutput

class tomoExtShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for the extended tomographic operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsDouble,receiversVector,reflectivityExt):
		# Domain = reflectivity
		# Range = Born data
		self.setDomainRange(domain,range)

		# print("domain nDim=",domain.getHyper().getNdim())
		# print("domain n1=",domain.getHyper().axes[0].n)
		# print("domain n2=",domain.getHyper().axes[1].n)
		# print("domain n3=",domain.getHyper().axes[2].n)
		#
		# print("range nDim=",range.getHyper().getNdim())
		# print("range n1=",range.getHyper().axes[0].n)
		# print("range n2=",range.getHyper().axes[1].n)
		# print("range n3=",range.getHyper().axes[2].n)

		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(sourcesSignalsDouble)):
			sourcesSignalsDouble = sourcesSignalsDouble.getCpp()
		if("getCpp" in dir(reflectivityExt)):
			reflectivityExt = reflectivityExt.getCpp()
		self.pyOp = pyAcoustic_iso_double_tomoExt_3D.tomoExtShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsDouble,receiversVector,reflectivityExt)
		return

	def __str__(self):
		"""Name of the operator"""
		return " TomoOp "

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double_tomoExt_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		# QC data size
		# print("Inside adjoint")
		# print("model nDim=",model.getHyper().getNdim())
		# print("model n1=",model.getHyper().axes[0].n)
		# print("model n2=",model.getHyper().axes[1].n)
		# print("model n3=",model.getHyper().axes[2].n)
		# print("data nDim=",data.getHyper().getNdim())
		# print("data n1=",data.getHyper().axes[0].n)
		# print("data n2=",data.getHyper().axes[1].n)
		# print("data n3=",data.getHyper().axes[2].n)

		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double_tomoExt_3D.ostream_redirect():
			# print("Just before")
			self.pyOp.adjoint(add,model,data)
		return

	def setVel_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_double_tomoExt_3D.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_double_tomoExt_3D.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

	def getWavefield1_3D(self,iWavefield):
		with pyAcoustic_iso_double_tomoExt_3D.ostream_redirect():
			wfld = self.pyOp.getWavefield1_3D(iWavefield)
			wfld = SepVector.floatVector(fromCpp=wfld)
		return wfld

	def getWavefield2_3D(self,iWavefield):
		with pyAcoustic_iso_double_tomoExt_3D.ostream_redirect():
			wfld = self.pyOp.getWavefield2_3D(iWavefield)
			wfld = SepVector.floatVector(fromCpp=wfld)
		return wfld
