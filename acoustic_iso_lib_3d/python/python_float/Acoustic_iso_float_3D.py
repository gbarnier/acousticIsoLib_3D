# Bullshit stuff
import pyAcoustic_iso_float_nl_3D
import pyAcoustic_iso_float_Born_3D
import pyAcoustic_iso_float_BornExt_3D
import pyAcoustic_iso_float_tomoExt_3D
import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np
import math
from pyAcoustic_iso_float_nl_3D import deviceGpu_3D

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

		# Check for consistency between number of shots and provided coordinates
		if (nShot != sourceGeomVectorNd.shape[1]):
			raise ValueError("**** ERROR [buildSourceGeometry_3D]: Number of shots from parfile (#shot=%s) not consistent with geometry file (#shots=%s)! ****\n" %(nShot,sourceGeomVectorNd.shape[1]))

		# Generate vector containing deviceGpu_3D objects
		for ishot in range(nShot):

			# Create inputs for devceGpu_3D constructor
			# We assume point sources -> zCoordFloat is a 1D array of length 1
			zCoord=SepVector.getSepVector(ns=[1])
			xCoord=SepVector.getSepVector(ns=[1])
			yCoord=SepVector.getSepVector(ns=[1])

			# Setting z, x and y-positions of the source for the given experiment
			zCoord.set(sourceGeomVectorNd[2,ishot]) # n2, n1
			xCoord.set(sourceGeomVectorNd[0,ishot])
			yCoord.set(sourceGeomVectorNd[1,ishot])

			# Create a deviceGpu_3D for this source and append to sources vector
			sourcesVector.append(deviceGpu_3D(zCoord.getCpp(), xCoord.getCpp(), yCoord.getCpp(), vel.getCpp(), nts, parObject.param, dipole, zDipoleShift, xDipoleShift, yDipoleShift, spaceInterpMethod, hFilter1d))

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
		yShotAxis=Hypercube.axis(n=nyShot,o=oy+oyShot*dy,d=dyShot*dy)
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

		# If receiver geometry is constant -> use only one deviceGpu for the reveivers

		# Generate vector containing deviceGpu_3D objects
		for ishot in range(nShot):

			# Create inputs for devceiGpu_3D constructor
			zCoord=SepVector.getSepVector(ns=[nReceiverPerShot])
			xCoord=SepVector.getSepVector(ns=[nReceiverPerShot])
			yCoord=SepVector.getSepVector(ns=[nReceiverPerShot])

			zCoordNd=zCoord.getNdArray()
			xCoordNd=xCoord.getNdArray()
			yCoordNd=yCoord.getNdArray()

			# Update the receiver's coordinates
			zCoordNd[:]=receiverGeomVectorNd[2,:,ishot]
			xCoordNd[:]=receiverGeomVectorNd[0,:,ishot]
			yCoordNd[:]=receiverGeomVectorNd[1,:,ishot]
			receiversVector.append(deviceGpu_3D(zCoord.getCpp(), xCoord.getCpp(), yCoord.getCpp(), vel.getCpp(), nts, parObject.param, dipole, zDipoleShift, xDipoleShift, yDipoleShift, spaceInterpMethod, hFilter1d))

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
################################### Ginsu ######################################
################################################################################
# Geometry
def buildGeometryGinsu_3D(parObject, vel, sourcesVector, receiversVector):

	"""Function that creates a propagation geometry for each shot for Ginsu modeling
	   Input:
	   		- Parameter object
			- Full velocity model
			- Sources' and receivers' geometry vectors
	   Output:
	   		- Vector of hypercubes containing the Ginsu velocity models for each shot
			- 2 integer arrays containing the values of xPadMinusGinsu and xPadPlusGinsu for each shot
		Other:
			- Updates the sources' and receivers' geometry vectors for Ginsu parameters
	"""

	# Model coordinates + dimensions
	xPadMinus = parObject.getInt("xPadMinus")
	xPadPlus = parObject.getInt("xPadPlus")
	yPadMinus = parObject.getInt("yPad")
	yPadPlus = parObject.getInt("yPad")
	fat = parObject.getInt("fat")
	blockSize = parObject.getInt("blockSize")

	# z-axis
	zPadMinus = parObject.getInt("zPadMinus")
	zPadPlus = parObject.getInt("zPadPlus")
	nz = vel.getHyper().axes[0].n
	dz = vel.getHyper().axes[0].d
	oz = vel.getHyper().axes[0].o
	ozTrue = oz + (fat+zPadMinus) * dz # Origin of domain of interest x-coordinate
	zMaxTrue = oz + (nz-zPadPlus-fat-1) * dz # Max x-coordinate without padding
	print("nz big = ", nz)
	print("oz big = ", oz)
	print("dz big = ", dz)
	print("ozTrue = ", ozTrue)
	print("zMaxTrue = ", zMaxTrue)

	# x-axis
	nx = vel.getHyper().axes[1].n
	dx = vel.getHyper().axes[1].d
	ox = vel.getHyper().axes[1].o
	oxTrue = ox + (fat+xPadMinus) * dx # Origin of domain of interest x-coordinate
	xMaxTrue = ox + (nx-xPadPlus-fat-1) * dx # Max x-coordinate without padding

	# y-axis
	ny = vel.getHyper().axes[2].n
	dy = vel.getHyper().axes[2].d
	oy = vel.getHyper().axes[2].o
	oyTrue = oy + (fat+yPadMinus) * dy # Origin of domain of interest y-coordinate
	yMaxTrue = oy + (ny-yPadPlus-fat-1) * dy # Max y-coordinate without padding

	print("nx big = ", nx)
	print("ox big = ", ox)
	print("dx big = ", dx)
	print("xMax big = ", ox + (nx-1)*dx)
	print("oxTrue = ", oxTrue)
	print("xMaxTrue = ", xMaxTrue)

	print("ny big = ", ny)
	print("oy big = ", oy)
	print("dy big = ", dy)
	print("yMax big = ", oy + (ny-1)*dy)
	print("oyTrue = ", oyTrue)
	print("yMaxTrue = ", yMaxTrue)

	# Loop over shots and create a hypercube for the Ginsu velocity
	nShot = len(sourcesVector)

	# Create vector of hypercubes for each Ginsu velocity
	velHyperVectorGinsu=[]
	ixVectorGinsu=[]
	iyVectorGinsu=[]

	# Create sepVector array for the
	xPadMinusVectorGinsu=SepVector.getSepVector(ns=[nShot], storage="dataInt")
	xPadPlusVectorGinsu=SepVector.getSepVector(ns=[nShot], storage="dataInt")
	xPadMinusVectorGinsuNp = xPadMinusVectorGinsu.getNdArray()
	xPadPlusVectorGinsuNp = xPadPlusVectorGinsu.getNdArray()

	# Get buffer sizes
	xBufferGinsu = parObject.getFloat("xBufferGinsu")
	yBufferGinsu = parObject.getFloat("yBufferGinsu")

	if (xBufferGinsu < dx):
		raise ValueError("**** ERROR [buildGeometryGinsu_3D]: xBufferGinsu < dx ****\n")

	if (yBufferGinsu < dy):
		raise ValueError("**** ERROR [buildGeometryGinsu_3D]: yBufferGinsu < dy ****\n")

	# Declare the maximum nx and ny over all shots
	nxMaxGinsu = 0
	nyMaxGinsu = 0

	# Loop over shots
	for iShot in range(nShot):

		print("Shot #", iShot)
		# print("Position devices before")
		# sourcesVector[iShot].printRegPosUnique()
		# receiversVector[iShot].printRegPosUnique()

		# Get sources' positions [km]
		xCoordSource = sourcesVector[iShot].getXCoord()
		xCoordSource = SepVector.floatVector(fromCpp=xCoordSource)
		xCoordSourceNp = xCoordSource.getNdArray()
		yCoordSource = sourcesVector[iShot].getYCoord()
		yCoordSource = SepVector.floatVector(fromCpp=yCoordSource)
		yCoordSourceNp = yCoordSource.getNdArray()

		# Get receivers' positions [km]
		xCoordRec = receiversVector[iShot].getXCoord()
		xCoordRec = SepVector.floatVector(fromCpp=xCoordRec)
		yCoordRec = receiversVector[iShot].getYCoord()
		yCoordRec = SepVector.floatVector(fromCpp=yCoordRec)

		# print("xCoordSource max = ", xCoordSource.max())
		# print("xCoordSource min = ", xCoordSource.min())
		# print("yCoordSource max = ", yCoordSource.max())
		# print("yCoordSource min = ", yCoordSource.min())

		# print("xCoordRec max = ", xCoordRec.max())
		# print("xCoordRec min = ", xCoordRec.min())
		# print("yCoordRec max = ", yCoordRec.max())
		# print("yCoordRec min = ", yCoordRec.min())

		# Get max and min values for velocity bounds [km]
		xMax = max(xCoordSource.max(), xCoordRec.max()) + xBufferGinsu
		xMin = min(xCoordSource.min(), xCoordRec.min()) - xBufferGinsu
		yMax = max(yCoordSource.max(), yCoordRec.max()) + yBufferGinsu
		yMin = min(yCoordSource.min(), yCoordRec.min()) - yBufferGinsu

		# print("xMax before = ", xMax)
		# print("xMin before = ", xMin)
		# print("yMax before = ", yMax)
		# print("yMin before = ", yMin)

		# If the acquisition device is close to the edge, use the bound from the big velocity model
		xMax = min(xMax, xMaxTrue)
		xMin = max(xMin, oxTrue)
		yMax = min(yMax, yMaxTrue)
		yMin = max(yMin, oyTrue)

		# print("xMax = ", xMax)
		# print("xMin = ", xMin)
		# print("yMax = ", yMax)
		# print("yMin = ", yMin)

		# Compute Ginsu min/max in terms of grid index
		# print("(xMin-ox)/dx = ", (xMin-ox)/dx)
		ixMin = int((xMin-ox)/dx)
		# print("ixMin = ", ixMin)
		ixMax = int((xMax-ox)/dx+0.5)
		iyMin = int((yMin-oy)/dy)
		# print("iyMin = ", ixMin)
		iyMax = int((yMax-oy)/dy+0.5)
		nxGinsu = ixMax - ixMin + 1
		oxGinsu = ox + ixMin * dx
		nyGinsu = iyMax - iyMin + 1
		oyGinsu = oy + iyMin * dy

		# print("nxGinsu small = ", nxGinsu)
		# print("oxGinsu small = ", oxGinsu)
		# print("nyGinsu small = ", nyGinsu)
		# print("oyGinsu small = ", oyGinsu)

		# Compute xPadPlusGinsu
		nxTotalNoFat = 2*xPadMinus + nxGinsu
		ratiox = math.ceil(float(nxTotalNoFat) / float(blockSize))
		xPadPlusGinsu = ratiox * blockSize - nxGinsu - xPadMinus

		# Make sure that the new padPlus does not take you out of bounds
		ixMaxTotal = ixMax + xPadPlusGinsu + fat
		if (ixMaxTotal >= nx):
			# In that case, set xPadMinusGinsu
			# and xPadPlusGinsu to xPadMinus
			print("Weird case")
			print("ixMaxTotal = ", ixMaxTotal)
			print("nx = ", nx)
			xPadMinusGinsu = xPadPlusGinsu
			xPadPlusGinsu = xPadMinus
		else:
			xPadMinusGinsu = xPadMinus

		# Compute dimensions of vel Ginsu
		nxGinsu = nxGinsu + xPadMinusGinsu + xPadPlusGinsu + 2*fat
		oxGinsu = oxGinsu - (xPadMinusGinsu+fat) * dx
		nyGinsu = nyGinsu + yPadMinus + yPadPlus + 2*fat
		oyGinsu = oyGinsu - (yPadMinus+fat) * dy

		# Check if the velGinsu did not go out of bounds from the origin (if we switched the padding)
		if (oxGinsu < ox):
			oxGinsu = ox
			nxGinsu = nx
			xPadMinusGinsu = xPadMinus
			xPadPlusGinsu = xPadPlus

		# Check that the velGinsu is contained within the large velocity
		if (nxGinsu > nx):
			raise ValueError("**** ERROR [buildGeometryGinsu_3D]: nxGinsu > nx ****\n")
		if (oxGinsu < ox):
			raise ValueError("**** ERROR [buildGeometryGinsu_3D]: oxGinsu < ox ****\n")

		# Create hypercube for Ginsu velocity
		zAxisGinsu = vel.getHyper().axes[0]
		xAxisGinsu = Hypercube.axis(n=nxGinsu, o=oxGinsu, d=dx)
		yAxisGinsu = Hypercube.axis(n=nyGinsu, o=oyGinsu, d=dy)
		velHyperGinsu = Hypercube.hypercube(axes=[zAxisGinsu, xAxisGinsu, yAxisGinsu])

		xPadMinusVectorGinsuNp[iShot] = xPadMinusGinsu
		xPadPlusVectorGinsuNp[iShot] = xPadPlusGinsu

		print("xPadMinusGinsu = ", xPadMinusVectorGinsuNp[iShot])
		print("xPadPlusGinsu = ", xPadPlusVectorGinsuNp[iShot])

		print("nz Ginsu = ", velHyperGinsu.axes[0].n)
		print("oz Ginsu = ", velHyperGinsu.axes[0].o)
		print("dz Ginsu = ", velHyperGinsu.axes[0].d)
		print("nx Ginsu = ", velHyperGinsu.axes[1].n)
		print("ox Ginsu = ", velHyperGinsu.axes[1].o)
		print("dx Ginsu = ", velHyperGinsu.axes[1].d)
		print("ny Ginsu = ", velHyperGinsu.axes[2].n)
		print("oy Ginsu = ", velHyperGinsu.axes[2].o)
		print("dy Ginsu = ", velHyperGinsu.axes[2].d)

		izGinsu = (velHyperGinsu.axes[0].o-oz)/velHyperGinsu.axes[0].d
		ixGinsu = (velHyperGinsu.axes[1].o-ox)/velHyperGinsu.axes[1].d
		iyGinsu = (velHyperGinsu.axes[2].o-oy)/velHyperGinsu.axes[2].d
		ixGinsu = int(ixGinsu)
		iyGinsu = int(iyGinsu)

		ixVectorGinsu.append(ixGinsu)
		iyVectorGinsu.append(iyGinsu)

		print("iz Ginsu = ", izGinsu)
		print("ix Ginsu = ", ixGinsu)
		print("iy Ginsu = ", iyGinsu)

		print("New oz Ginsu = ", oz+izGinsu*velHyperGinsu.axes[0].d)
		print("New ox Ginsu = ", ox+ixGinsu*velHyperGinsu.axes[1].d)
		print("New oy Ginsu = ", oy+iyGinsu*velHyperGinsu.axes[2].d)

		# Compute nx and ny max among all shots for the wavefield allocation
		if (velHyperGinsu.axes[1].n > nxMaxGinsu):
			nxMaxGinsu = velHyperGinsu.axes[1].n

		if (velHyperGinsu.axes[2].n > nyMaxGinsu):
			nyMaxGinsu = velHyperGinsu.axes[2].n

		# Transform the stupid Hypercube into a f*** pyHypercube
		velHyperGinsu = velHyperGinsu.getCpp()

		# Append hyper
		velHyperVectorGinsu.append(velHyperGinsu)

		# Update source and receivers vectors with Ginsu parameters
		sourcesVector[iShot].setDeviceGpuGinsu_3D(velHyperGinsu, xPadMinusGinsu, xPadPlusGinsu)
		receiversVector[iShot].setDeviceGpuGinsu_3D(velHyperGinsu, xPadMinusGinsu, xPadPlusGinsu)

		print("Finished Shot #", iShot+1)
		print("------------------------")

	print("nxMaxGinsu = ", nxMaxGinsu)
	print("nyMaxGinsu = ", nyMaxGinsu)

	return velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,sourcesVector,receiversVector,ixVectorGinsu,iyVectorGinsu,nxMaxGinsu,nyMaxGinsu

################################################################################
############################# Useful functions #################################
################################################################################
# Gpu list to preallocate wavefields on the CPU
def getGpuNumber_3D(parObject):

	# Get number of Gpus
	nGpu = parObject.getInt("nGpu",-1)

	# Get number of shots
	nShot = parObject.getInt("nShot",-1)

	# If the user does not provide nGpu > 0 or a valid list -> break
	gpuList = parObject.getInts("iGpu", [-1])
	if (gpuList[0]<0 and nGpu<=0):
		raise ValueError("**** ERROR [getGpuNumber_3D]: Please provide a list of GPUs to be used ****\n")

	# If user does not provide a valid list but provides nGpu -> use id: 0,...,nGpu-1
	if (gpuList[0]<0 and nGpu>0):
		gpuList=[iGpu for iGpu in range(nGpu)]

	# If the user provides a list -> use that list and ignore nGpu for the parfile
	if (gpuList[0]>=0):
		# Get number of Gpu
		nGpu=len(gpuList)
		for iGpu in gpuList:
			if gpuList.count(iGpu) > 1:
				raise ValueError("**** ERROR [getGpuNumber_3D]: Please provide a correct list of GPUs to be used without duplicates ****")

	# Check that nGpu <= nShot
	if (nGpu > nShot):
		raise ValueError("**** ERROR [getGpuNumber_3D]: User required more GPUs than shots to be modeled ****\n")

	return nGpu,gpuList

# Bounds vectors
# Create bound vectors for FWI
def createBoundVectors_3D(parObject,model):

	# Get model dimensions
	nz=parObject.getInt("nz")
	nx=parObject.getInt("nx")
	ny=parObject.getInt("ny")
	fat=parObject.getInt("fat")
	spline=parObject.getInt("spline",0)
	if (spline==1): fat=0

	# Min bound
	minBoundVectorFile=parObject.getString("minBoundVector","noMinBoundVectorFile")
	if (minBoundVectorFile=="noMinBoundVectorFile"):
		minBound=parObject.getFloat("minBound")
		minBoundVector=model.clone()
		minBoundVector.zero()
		minBoundVector.getNdArray()[fat:ny-fat,fat:nx-fat,fat:nz-fat]=minBound

	else:
		minBoundVector=genericIO.defaultIO.getVector(minBoundVectorFile)

	# Max bound
	maxBoundVectorFile=parObject.getString("maxBoundVector","noMaxBoundVectorFile")
	if (maxBoundVectorFile=="noMaxBoundVectorFile"):
		maxBound=parObject.getFloat("maxBound")
		maxBoundVector=model.clone()
		maxBoundVector.zero()
		maxBoundVector.getNdArray()[fat:ny-fat,fat:nx-fat,fat:nz-fat]=maxBound

	else:
		maxBoundVector=genericIO.defaultIO.getVector(maxBoundVectorFile)

	return minBoundVector,maxBoundVector

################################################################################
############################ Nonlinear propagation #############################
################################################################################
def nonlinearOpInitFloat_3D(args):

	"""Function to correctly initialize nonlinear 3D operator
	   The function will return the necessary variables for operator construction
	"""
	# IO object
	parObject=genericIO.io(params=args)

	# Read flag to display information
	info = parObject.getInt("info",0)
	if (info == 1):
		print("**** [nonlinearOpInitFloat_3D]: User has requested to display information ****\n")

	# Read velocity and convert to float
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		raise ValueError("**** ERROR [nonlinearOpInitFloat_3D]: User did not provide velocity file ****\n")
	velFloat=genericIO.defaultIO.getVector(velFile)
	velFloatNp=velFloat.getNdArray()

	# Determine if the source time signature is constant over shots
	constantWavelet = parObject.getInt("constantWavelet",-1)
	if (info == 1 and constantWavelet == 1):
		print("**** [nonlinearOpInitFloat_3D]: Using the same seismic source time signature for all shots ****\n")
	if (info == 1 and constantWavelet == 0):
		print("**** [nonlinearOpInitFloat_3D]: Using different seismic source time signatures for each shot ****\n")
	if (constantWavelet != 0 and constantWavelet != 1):
		raise ValueError("**** ERROR [nonlinearOpInitFloat_3D]: User did not specify an acceptable value for tag constantWavelet (must be 0 or 1) ****\n")

	# Build sources/receivers geometry
	sourcesVector,shotHyper=buildSourceGeometry_3D(parObject,velFloat)
	receiversVector,receiverHyper=buildReceiversGeometry_3D(parObject,velFloat)

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
		model=SepVector.getSepVector(modelHyper)

	else:
		# If we do not use a constant wavelet
		# Allocate a 2D array where the second axis has a length of the total number of shots
		modelHyper=Hypercube.hypercube(axes=[timeAxis,shotAxis])
		model=SepVector.getSepVector(modelHyper)

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,shotAxis])
	data=SepVector.getSepVector(dataHyper)

	# Create data hypercube for writing the data to disk
	# Regular geometry for both the sources and receivers
	if (shotHyper.getNdim()>1 and receiverHyper.getNdim()>1):
		dataHyperForOutput=Hypercube.hypercube(axes=[timeAxis,zReceiverAxis,xReceiverAxis,yReceiverAxis,zShotAxis,xShotAxis,yShotAxis])
	# Irregular geometry for both the sources and receivers
	if (shotHyper.getNdim()==1 and receiverHyper.getNdim()==1):
		dataHyperForOutput=dataHyper

	# Outputs
	return model,data,velFloat,parObject,sourcesVector,receiversVector,dataHyperForOutput

class nonlinearPropShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear 3D propagator"""

	def __init__(self,*args):
		#Domain = source wavelet
		#Range = recorded data space
		domain = args[0]
		range = args[1]
		self.setDomainRange(domain,range)
		velocity = args[2]
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		paramP = args[3]
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		sourceVector = args[4]
		receiversVector = args[5]
		# Ginsu
		if (len(args) > 6):
			velHyperVectorGinsu = args[6]
			if("getCpp" in dir(velHyperVectorGinsu)):
				velHyperVectorGinsu = velHyperVectorGinsu.getCpp()
			xPadMinusVectorGinsu = args[7]
			if("getCpp" in dir(xPadMinusVectorGinsu)):
				xPadMinusVectorGinsu = xPadMinusVectorGinsu.getCpp()
			xPadPlusVectorGinsu = args[8]
			if("getCpp" in dir(xPadPlusVectorGinsu)):
				xPadPlusVectorGinsu = xPadPlusVectorGinsu.getCpp()
			ixVectorGinsu = args[9]
			iyVectorGinsu = args[10]
			print("Nonlinear: ixVectorGinsu = ", ixVectorGinsu)
			print("Nonlinear: iyVectorGinsu = ", iyVectorGinsu)
			self.pyOp = pyAcoustic_iso_float_nl_3D.nonlinearPropShotsGpu_3D(velocity,paramP.param,sourceVector,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,ixVectorGinsu,iyVectorGinsu)
		# No ginsu
		else:
			self.pyOp = pyAcoustic_iso_float_nl_3D.nonlinearPropShotsGpu_3D(velocity,paramP.param,sourceVector,receiversVector)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl_3D.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def setVel_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_nl_3D.ostream_redirect():
			self.pyOp.setVel_3D(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_nl_3D.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

################################################################################
################################# FWI ##########################################
################################################################################
def nonlinearFwiOpInitFloat_3D(args):

	"""Function to correctly initialize a nonlinear operator where the model is velocity
	   The function will return the necessary variables for operator construction
	"""
	# IO object
	parObject=genericIO.io(params=args)

	# Read flag to display information
	info = parObject.getInt("info",0)
	if (info == 1):
		print("**** [nonlinearOpInitDouble_3D]: User has requested to display information ****\n")

	# Read velocity and convert to double
	modelStartFile=parObject.getString("vel")
	modelStartFloat=genericIO.defaultIO.getVector(modelStartFile)

	# Determine if the source time signature is constant over shots
	constantWavelet = parObject.getInt("constantWavelet",-1)
	if (info == 1 and constantWavelet == 1):
		print("**** [nonlinearOpInitDouble_3D]: Using the same seismic source time signature for all shots ****\n")
	if (info == 1 and constantWavelet == 0):
		print("**** [nonlinearOpInitDouble_3D]: Using different seismic source time signatures for each shot ****\n")
	if (constantWavelet != 0 and constantWavelet != 1):
		raise ValueError("**** ERROR [nonlinearOpInitDouble_3D]: User did not specify an acceptable value for tag constantWavelet (must be 0 or 1) ****\n")

	# Build sources/receivers geometry
	sourcesVector,shotHyper=buildSourceGeometry_3D(parObject,modelStartFloat)
	receiversVector,receiverHyper=buildReceiversGeometry_3D(parObject,modelStartFloat)

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

	# Read sources' signals into float array
	sourcesSignalsFile=parObject.getString("sources")
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesSignalsFile,ndims=2)

	# Allocate data as a 3 dimensional array (even for regular geometry)
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,shotAxis])
	dataFloat=SepVector.getSepVector(dataHyper)

	# Create data hypercube for writing the data to disk
	# Regular geometry for both the sources and receivers
	if (shotHyper.getNdim()>1 and receiverHyper.getNdim()>1):
		dataHyperForOutput=Hypercube.hypercube(axes=[timeAxis,zReceiverAxis,xReceiverAxis,yReceiverAxis,zShotAxis,xShotAxis,yShotAxis])
	# Irregular geometry for both the sources and receivers
	if (shotHyper.getNdim()==1 and receiverHyper.getNdim()==1):
		dataHyperForOutput=dataHyper

	# Outputs
	return modelStartFloat,dataFloat,sourcesSignalsFloat,parObject,sourcesVector,receiversVector,dataHyperForOutput

class nonlinearFwiPropShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear 3D propagator"""

	# def __init__(self,domain,range,sourcesSignal,paramP,sourceVector,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,ginsu):
	def __init__(self,*args):

		# Domain = velocity model
		# Range = recorded data space
		domain = args[0]
		range = args[1]
		self.setDomainRange(domain,range)
		# Model (velocity)
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		# Source signal
		sourcesSginal = args[2]
		if("getCpp" in dir(sourcesSginal)):
			sourcesSginal = sourcesSginal.getCpp()
			self.sourcesSginal = sourcesSginal.clone()
		# Parfile
		paramP = args[3]
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		# Acquisition
		sourceVector = args[4]
		receiversVector = args[5]
		if (len(args) > 6):
			print("Ginsu constructor")
			velHyperVectorGinsu = args[6]
			if("getCpp" in dir(velHyperVectorGinsu)):
				velHyperVectorGinsu = velHyperVectorGinsu.getCpp()
			xPadMinusVectorGinsu = args[7]
			if("getCpp" in dir(xPadMinusVectorGinsu)):
				xPadMinusVectorGinsu = xPadMinusVectorGinsu.getCpp()
			xPadPlusVectorGinsu = args[8]
			if("getCpp" in dir(xPadPlusVectorGinsu)):
				xPadPlusVectorGinsu = xPadPlusVectorGinsu.getCpp()
			ixVectorGinsu = args[9]
			iyVectorGinsu = args[10]
			print("Fwi: ixVectorGinsu = ", ixVectorGinsu)
			print("Fwi: iyVectorGinsu = ", iyVectorGinsu)
			self.pyOp = pyAcoustic_iso_float_nl_3D.nonlinearPropShotsGpu_3D(domain,paramP.param,sourceVector,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,ixVectorGinsu,iyVectorGinsu)
		else:
			self.pyOp = pyAcoustic_iso_float_nl_3D.nonlinearPropShotsGpu_3D(domain,paramP.param,sourceVector,receiversVector)
		return

	def forward(self,add,model,data):

		self.setVel_3D(model)
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl_3D.ostream_redirect():
			# print("in fwd, before")
			self.pyOp.forward(add,self.sourcesSginal,data)
			# print("in fwd, after")
		return

	def setVel_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_nl_3D.ostream_redirect():
			self.pyOp.setVel_3D(vel)
		return

################################################################################
################################### Born #######################################
################################################################################
def BornOpInitFloat_3D(args):
	"""Function to correctly initialize Born operator
	   The function will return the necessary variables for operator construction
	"""

	# IO object
	parObject=genericIO.io(params=args)

	# Read flag to display information
	info = parObject.getInt("info",0)
	if (info == 1):
		print("**** [BornOpInitFloat_3D]: User has requested to display information ****\n")

	# Read velocity and convert to double
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		raise ValueError("**** ERROR [BornOpInitFloat_3D]: User did not provide velocity file ****\n")
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Determine if the source time signature is constant over shots
	constantWavelet = parObject.getInt("constantWavelet",-1)
	if (info == 1 and constantWavelet == 1):
		print("**** [BornOpInitDouble_3D]: Using the same seismic source time signature for all shots ****\n")
	if (info == 1 and constantWavelet == 0):
		print("**** [BornOpInitDouble_3D]: Using different seismic source time signatures for each shot ****\n")
	if (constantWavelet != 0 and constantWavelet != 1):
		raise ValueError("**** ERROR [BornOpInitDouble_3D]: User did not specify an acceptable value for tag constantWavelet (must be 0 or 1) ****\n")

	# Build sources/receivers geometry
	sourcesVector,shotHyper=buildSourceGeometry_3D(parObject,velFloat)
	receiversVector,receiverHyper=buildReceiversGeometry_3D(parObject,velFloat)

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
	modelFloat=SepVector.getSepVector(velFloat.getHyper())

	# Read velocity and convert to double
	sourcesSignalsFile=parObject.getString("sources","noSourcesFile")
	if (sourcesSignalsFile == "noSourcesFile"):
		raise IOError("**** ERROR [BornOpInitDouble_3D]: User did not provide seismic sources file ****\n")
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesSignalsFile,ndims=2)

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,shotAxis])
	dataFloat=SepVector.getSepVector(dataHyper)

	# Create data hypercurbe for writing the data to disk
	# Regular geometry for both the sources and receivers
	if (shotHyper.getNdim()>1 and receiverHyper.getNdim()>1):
		dataHyperForOutput=Hypercube.hypercube(axes=[timeAxis,zReceiverAxis,xReceiverAxis,yReceiverAxis,zShotAxis,xShotAxis,yShotAxis])
	# Irregular geometry for both the sources and receivers
	if (shotHyper.getNdim()==1 and receiverHyper.getNdim()==1):
		dataHyperForOutput=dataHyper

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,dataHyperForOutput

class BornShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born operator"""

	# def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsFloat,receiversVector):
	def __init__(self,*args,**kwargs):

		# Domain = reflectivity
		# Range = Born data
		domainOp = args[0]
		rangeOp = args[1]
		self.setDomainRange(domainOp,rangeOp)

		# Get tomo operator for Fwime
		tomoExtOp=kwargs.get("tomoExtOp",None)

		# Get wavefield dimensions
		zAxisWavefield = domainOp.getHyper().axes[0]
		xAxisWavefield = domainOp.getHyper().axes[1]
		yAxisWavefield = domainOp.getHyper().axes[2]
		timeAxisWavefield = rangeOp.getHyper().axes[0]

		# Velocity model
		velocity = args[2]
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()

		# Parameter object
		paramP = args[3]
		nGpu = getGpuNumber_3D(paramP)
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()

		# Source / receiver geometry
		sourceVector = args[4]
		receiversVector = args[6]

		# Source signal
		sourcesSignalsFloat = args[5]
		if("getCpp" in dir(sourcesSignalsFloat)):
			sourcesSignalsFloat = sourcesSignalsFloat.getCpp()

		# Declare wavefield vector
		wavefieldVector = []

		# Ginsu
		if (len(args) > 7):

			print("Ginsu constructor Iso")

			# Vel hypercube for Ginsu
			velHyperVectorGinsu = args[7]
			if("getCpp" in dir(velHyperVectorGinsu)):
				velHyperVectorGinsu = velHyperVectorGinsu.getCpp()

			# Padding for Ginsu
			xPadMinusVectorGinsu = args[8]
			if("getCpp" in dir(xPadMinusVectorGinsu)):
				xPadMinusVectorGinsu = xPadMinusVectorGinsu.getCpp()
			xPadPlusVectorGinsu = args[9]
			if("getCpp" in dir(xPadPlusVectorGinsu)):
				xPadPlusVectorGinsu = xPadPlusVectorGinsu.getCpp()

			# Maximum dimensions for model
			nxMaxGinsu = args[10]
			nyMaxGinsu = args[11]

			# Create hypercube for Ginsu wavefield
			xAxisWavefield = Hypercube.axis(n=nxMaxGinsu, o=0.0, d=1.0)
			yAxisWavefield = Hypercube.axis(n=nyMaxGinsu, o=0.0, d=1.0)
			hyperWavefield = Hypercube.hypercube(axes=[zAxisWavefield,xAxisWavefield,yAxisWavefield,timeAxisWavefield])
			print("Allocating Ginsu wavefields")
			nzWav = zAxisWavefield.n
			nxWav = xAxisWavefield.n
			nyWav = yAxisWavefield.n
			ntWav = timeAxisWavefield.n
			print("Size wavefield = ", nzWav*nxWav*nyWav*ntWav*4/(1024*1024*1024), " [GB]")
			ixVectorGinsu = args[12]
			iyVectorGinsu = args[13]

			# Allocate wavefield
			# for iGpu in range(nGpu):
			# 	newWavefield = SepVector.getSepVector(hyperWavefield,storage="dataDouble")
			# 	if("getCpp" in dir(newWavefield)):
			# 		newWavefield = newWavefield.getCpp()
			# 	wavefieldVector.append(newWavefield)
			# print("Done allocating Ginsu wavefields")
			# if("getCpp" in dir(wavefieldVector)):
			# 	wavefieldVector = wavefieldVector.getCpp()

			if (tomoExtOp == None):
				# Ginsu constructor
				self.pyOp = pyAcoustic_iso_float_Born_3D.BornShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)
			else:
				# Ginsu constructor for Fwime
				self.pyOp = pyAcoustic_iso_float_Born_3D.BornShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu,tomoExtOp.pyOp)
		else:
			# Create hypercube for wavefield
			hyperWavefield = Hypercube.hypercube(axes=[zAxisWavefield,xAxisWavefield,yAxisWavefield,timeAxisWavefield])
			# print("Allocating normal wavefields")
			nzWav = zAxisWavefield.n
			nxWav = xAxisWavefield.n
			nyWav = yAxisWavefield.n
			ntWav = timeAxisWavefield.n
			print("Size wavefield = ", nzWav*nxWav*nyWav*ntWav*8/(1024*1024*1024), " [GB]")
			# for iGpu in range(nGpu):
			# 	newWavefield = SepVector.getSepVector(hyperWavefield,storage="dataDouble")
			# 	if("getCpp" in dir(newWavefield)):
			# 		newWavefield = newWavefield.getCpp()
			# 	wavefieldVector.append(newWavefield)
			# print("Done allocating normal wavefields")
			# if("getCpp" in dir(wavefieldVector)):
			# 	wavefieldVector = wavefieldVector.getCpp()
			if (tomoExtOp == None):
				# Normal constructor
				self.pyOp = pyAcoustic_iso_float_Born_3D.BornShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector)
			else:
				# Normal constructor for Fwime
				print("Born constructor for FWIME")
				self.pyOp = pyAcoustic_iso_float_Born_3D.BornShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector,tomoExtOp.pyOp)
				print("Done Born constructor for FWIME")
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
		with pyAcoustic_iso_float_Born_3D.ostream_redirect():
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

		# print("Born adjoint before data max = ", data.max())
		# print("Born adjoint before data min = ", data.min())

		#Checking if getCpp is present
		# if("getCpp" in dir(model)):
		# 	model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_Born_3D.ostream_redirect():
			# print("Just before")
			# print("Before")
			self.pyOp.adjoint(add,model.getCpp(),data)
			# print("Born adjoint after model max = ", model.max())
			# print("Born adjoint after model min = ", model.min())
		return

	def setVel_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_Born_3D.ostream_redirect():
			self.pyOp.setVel_3D(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_Born_3D.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

	def getSrcWavefield_3D(self,iWavefield):
		with pyAcoustic_iso_float_Born_3D.ostream_redirect():
			wfld = self.pyOp.getSrcWavefield_3D(iWavefield)
			wfld = SepVector.floatVector(fromCpp=wfld)
		return wfld

################################################################################
############################## Born extended ###################################
################################################################################
def BornExtOpInitFloat_3D(args):
	"""Function to correctly initialize Born extended operator
	   The function will return the necessary variables for operator construction
	"""

	# IO object
	parObject=genericIO.io(params=args)

	# Read flag to display information
	info = parObject.getInt("info",0)
	if (info == 1):
		print("**** [BornExtOpInitFloat_3D]: User has requested to display information ****\n")

	# Read velocity and convert to double
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		raise ValueError("**** ERROR [BornExtOpInitFloat_3D]: User did not provide velocity file ****\n")
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Determine if the source time signature is constant over shots
	constantWavelet = parObject.getInt("constantWavelet",-1)
	if (info == 1 and constantWavelet == 1):
		print("**** [BornExtOpInitFloat_3D]: Using the same seismic source time signature for all shots ****\n")
	if (info == 1 and constantWavelet == 0):
		print("**** [BornExtOpInitFloat_3D]: Using different seismic source time signatures for each shot ****\n")
	if (constantWavelet != 0 and constantWavelet != 1):
		raise ValueError("**** ERROR [BornExtOpInitFloat_3D]: User did not specify an acceptable value for tag constantWavelet (must be 0 or 1) ****\n")

	# Build sources/receivers geometry
	sourcesVector,shotHyper=buildSourceGeometry_3D(parObject,velFloat)
	receiversVector,receiverHyper=buildReceiversGeometry_3D(parObject,velFloat)

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
		raise IOError("**** ERROR [BornExtOpInitFloat_3D]: User did not provide seismic sources file ****\n")
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesSignalsFile,ndims=2)

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,shotAxis])
	dataFloat=SepVector.getSepVector(dataHyper)

	########################## Delete once QC'ed ###############################
	# Create data hypercurbe for writing the data to disk
	# Regular geometry for both the sources and receivers

	if (shotHyper.getNdim()>1 and receiverHyper.getNdim()>1):
		dataHyperForOutput=Hypercube.hypercube(axes=[timeAxis,zReceiverAxis,xReceiverAxis,yReceiverAxis,zShotAxis,xShotAxis,yShotAxis])
	# Irregular geometry for both the sources and receivers
	if (shotHyper.getNdim()==1 and receiverHyper.getNdim()==1):
		dataHyperForOutput=dataHyper
	############################################################################

	# Allocate model
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR [BornExtOpInitFloat_3D]: User did not provide extension type ****\n")
		quit()

	nExt1=parObject.getInt("nExt1", -1)
	if (nExt1 == -1):
		print("**** ERROR [BornExtOpInitFloat_3D]: User did not provide size of extension axis #1 ****\n")
		quit()
	if (nExt1%2 ==0):
		print("Length of extension axis #1 must be an uneven number")
		quit()

	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR [BornExtOpInitFloat_3D]: User did not provide extension type ****\n")
		quit()

	nExt2=parObject.getInt("nExt2", -1)
	if (nExt2 == -1):
		print("**** ERROR [BornExtOpInitFloat_3D]: User did not provide size of extension axis #2 ****\n")
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

	modelFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,yAxis,ext1Axis,ext2Axis]))

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,dataHyperForOutput

class BornExtShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born extended operator"""

	def __init__(self,*args,**kwargs):

		# Domain = reflectivity
		# Range = Born data
		domain = args[0]
		range = args[1]
		self.setDomainRange(domain,range)

		tomoExtOp=kwargs.get("tomoExtOp",None)

		# Get wavefield dimensions
		zAxisWavefield = domain.getHyper().axes[0]
		xAxisWavefield = domain.getHyper().axes[1]
		yAxisWavefield = domain.getHyper().axes[2]
		timeAxisWavefield = range.getHyper().axes[0]

		# Velocity model
		velocity = args[2]
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()

		# Parameter object
		paramP = args[3]
		nGpu = getGpuNumber_3D(paramP)
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()

		# Source / receiver geometry
		sourceVector = args[4]
		receiversVector = args[6]

		# Source signal
		sourcesSignalsFloat = args[5]
		if("getCpp" in dir(sourcesSignalsFloat)):
			sourcesSignalsFloat = sourcesSignalsFloat.getCpp()

		# Declare wavefield vector
		wavefieldVector = []

		# Ginsu
		if (len(args) > 7):

			# Vel hypercube for Ginsu
			velHyperVectorGinsu = args[7]
			if("getCpp" in dir(velHyperVectorGinsu)):
				velHyperVectorGinsu = velHyperVectorGinsu.getCpp()

			# Padding for Ginsu
			xPadMinusVectorGinsu = args[8]
			if("getCpp" in dir(xPadMinusVectorGinsu)):
				xPadMinusVectorGinsu = xPadMinusVectorGinsu.getCpp()
			xPadPlusVectorGinsu = args[9]
			if("getCpp" in dir(xPadPlusVectorGinsu)):
				xPadPlusVectorGinsu = xPadPlusVectorGinsu.getCpp()

			# Maximum dimensions for model
			nxMaxGinsu = args[10]
			nyMaxGinsu = args[11]

			# Create hypercube for Ginsu wavefield
			xAxisWavefield = Hypercube.axis(n=nxMaxGinsu, o=0.0, d=1.0)
			yAxisWavefield = Hypercube.axis(n=nyMaxGinsu, o=0.0, d=1.0)
			hyperWavefield = Hypercube.hypercube(axes=[zAxisWavefield,xAxisWavefield,yAxisWavefield,timeAxisWavefield])
			nzWav = zAxisWavefield.n
			nxWav = xAxisWavefield.n
			nyWav = yAxisWavefield.n
			ntWav = timeAxisWavefield.n
			print("Size wavefield = ", nzWav*nxWav*nyWav*ntWav*4/(1024*1024*1024), " [GB]")
			ixVectorGinsu = args[12]
			iyVectorGinsu = args[13]

			if (tomoExtOp == None):
				# Ginsu constructor
				self.pyOp = pyAcoustic_iso_float_BornExt_3D.BornExtShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)
			else:
				# Ginsu constructor for Fwime
				self.pyOp = pyAcoustic_iso_float_BornExt_3D.BornExtShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu,tomoExtOp.pyOp)
		else:

			# Create hypercube for wavefield
			hyperWavefield = Hypercube.hypercube(axes=[zAxisWavefield,xAxisWavefield,yAxisWavefield,timeAxisWavefield])
			# print("Allocating normal wavefields")
			nzWav = zAxisWavefield.n
			nxWav = xAxisWavefield.n
			nyWav = yAxisWavefield.n
			ntWav = timeAxisWavefield.n
			print("Size wavefield = ", nzWav*nxWav*nyWav*ntWav*4/(1024*1024*1024), " [GB]")

			if (tomoExtOp == None):
				# Normal constructor
				self.pyOp = pyAcoustic_iso_float_BornExt_3D.BornExtShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector)
			else:
				print("BornExt constructor for FWIME")
				# Normal constructor for Fwime
				self.pyOp = pyAcoustic_iso_float_BornExt_3D.BornExtShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector,tomoExtOp.pyOp)
				print("Done BornExt constructor for FWIME")
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
		with pyAcoustic_iso_float_BornExt_3D.ostream_redirect():
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
		with pyAcoustic_iso_float_BornExt_3D.ostream_redirect():
			# print("Just before")
			self.pyOp.adjoint(add,model,data)
		return

	def add_spline_3D(self,Spline_op):
		"""
		   Adding spline operator to set background
		"""
		self.Spline_op = Spline_op
		self.tmp_fine_model = Spline_op.range.clone()
		return

	def setVel_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_BornExt_3D.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def deallocatePinnedBornExtGpu_3D(self):
		with pyAcoustic_iso_float_BornExt_3D.ostream_redirect():
			self.pyOp.deallocatePinnedBornExtGpu_3D()
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_BornExt_3D.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

################################################################################
############################## Tomo extended ###################################
################################################################################
def tomoExtOpInitFloat_3D(args):
	"""Function to correctly initialize Born extended operator
	   The function will return the necessary variables for operator construction
	"""

	# IO object
	parObject=genericIO.io(params=args)

	# Read flag to display information
	info = parObject.getInt("info",0)
	if (info == 1):
		print("**** [BornExtOpInitFloat_3D]: User has requested to display information ****\n")

	############################## Velocity ####################################
	# Read velocity and convert to double
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		raise ValueError("**** ERROR [tomoExtOpInitDouble_3D]: User did not provide velocity file ****\n")
	velFloat=genericIO.defaultIO.getVector(velFile)

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
	sourcesVector,shotHyper=buildSourceGeometry_3D(parObject,velFloat)

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

	############################## Receivers ###################################
	# Build receivers geometry
	receiversVector,receiverHyper=buildReceiversGeometry_3D(parObject,velFloat)

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
		print("**** ERROR [tomoExtOpInitFloat_3D]: User did not provide extension type ****\n")
		quit()

	nExt1=parObject.getInt("nExt1", -1)
	if (nExt1 == -1):
		print("**** ERROR [tomoExtOpInitFloat_3D]: User did not provide size of extension axis #1 ****\n")
		quit()
	if (nExt1%2 ==0):
		print("Length of extension axis #1 must be an uneven number")
		quit()

	nExt2=parObject.getInt("nExt2", -1)
	if (nExt2 == -1):
		print("**** ERROR [tomoExtOpInitFloat_3D]: User did not provide size of extension axis #2 ****\n")
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
		raise ValueError("**** ERROR [tomoExtOpInitFloat_3D]: User did not provide reflectivity file ****\n")
	reflectivityFloat=genericIO.defaultIO.getVector(reflectivityFile,ndims=5)

	############################## Data ########################################
	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,shotAxis])
	dataFloat=SepVector.getSepVector(dataHyper)

	# Create data hypercurbe for writing the data to disk
	# Regular geometry for both the sources and receivers
	if (shotHyper.getNdim()>1 and receiverHyper.getNdim()>1):
		dataHyperForOutput=Hypercube.hypercube(axes=[timeAxis,zReceiverAxis,xReceiverAxis,yReceiverAxis,zShotAxis,xShotAxis,yShotAxis])
	# Irregular geometry for both the sources and receivers
	if (shotHyper.getNdim()==1 and receiverHyper.getNdim()==1):
		dataHyperForOutput=dataHyper

	############################## Model #######################################
	modelFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,yAxis]))

	############################## Output ######################################
	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsFloat,receiversVector,reflectivityFloat,dataHyperForOutput

class tomoExtShotsGpu_3D(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for the extended tomographic operator"""

	# def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsDouble,receiversVector,reflectivityExt):
	def __init__(self,*args):
		# Domain = Velocity model perturbation
		# Range = Tomo data
		domain = args[0]
		range = args[1]
		self.setDomainRange(domain,range)

		# Get wavefield dimensions
		zAxisWavefield = domain.getHyper().axes[0]
		xAxisWavefield = domain.getHyper().axes[1]
		yAxisWavefield = domain.getHyper().axes[2]
		timeAxisWavefield = range.getHyper().axes[0]

		# Velocity model
		velocity = args[2]
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()

		# Parameter object
		paramP = args[3]
		nGpu = getGpuNumber_3D(paramP)
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()

		# Source / receiver geometry
		sourceVector = args[4]
		receiversVector = args[6]

		# Source signal
		sourcesSignalsFloat = args[5]
		if("getCpp" in dir(sourcesSignalsFloat)):
			sourcesSignalsFloat = sourcesSignalsFloat.getCpp()

		# Extended reflectivity
		extReflectivityFloat = args[7]
		if("getCpp" in dir(extReflectivityFloat)):
			extReflectivityFloat = extReflectivityFloat.getCpp()

		# Declare wavefield vector
		wavefieldVector = []

		# Ginsu
		if (len(args) > 8):

			# Vel hypercube for Ginsu
			velHyperVectorGinsu = args[8]
			if("getCpp" in dir(velHyperVectorGinsu)):
				velHyperVectorGinsu = velHyperVectorGinsu.getCpp()

			# Padding for Ginsu
			xPadMinusVectorGinsu = args[9]
			if("getCpp" in dir(xPadMinusVectorGinsu)):
				xPadMinusVectorGinsu = xPadMinusVectorGinsu.getCpp()
			xPadPlusVectorGinsu = args[10]
			if("getCpp" in dir(xPadPlusVectorGinsu)):
				xPadPlusVectorGinsu = xPadPlusVectorGinsu.getCpp()

			# Maximum dimensions for model
			nxMaxGinsu = args[11]
			nyMaxGinsu = args[12]

			# Create hypercube for Ginsu wavefield
			xAxisWavefield = Hypercube.axis(n=nxMaxGinsu, o=0.0, d=1.0)
			yAxisWavefield = Hypercube.axis(n=nyMaxGinsu, o=0.0, d=1.0)
			hyperWavefield = Hypercube.hypercube(axes=[zAxisWavefield,xAxisWavefield,yAxisWavefield,timeAxisWavefield])
			nzWav = zAxisWavefield.n
			nxWav = xAxisWavefield.n
			nyWav = yAxisWavefield.n
			ntWav = timeAxisWavefield.n
			print("Size wavefield = ", nzWav*nxWav*nyWav*ntWav*4/(1024*1024*1024), " [GB]")
			ixVectorGinsu = args[13]
			iyVectorGinsu = args[14]

			# Ginsu constructor
			self.pyOp = pyAcoustic_iso_float_tomoExt_3D.tomoExtShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector,extReflectivityFloat,velHyperVectorGinsu,xPadMinusVectorGinsu,xPadPlusVectorGinsu,nxMaxGinsu,nyMaxGinsu,ixVectorGinsu,iyVectorGinsu)

		# No Ginsu
		else:

			# Create hypercube for wavefield
			hyperWavefield = Hypercube.hypercube(axes=[zAxisWavefield,xAxisWavefield,yAxisWavefield,timeAxisWavefield])
			# print("Allocating normal wavefields")
			nzWav = zAxisWavefield.n
			nxWav = xAxisWavefield.n
			nyWav = yAxisWavefield.n
			ntWav = timeAxisWavefield.n
			print("Size wavefield = ", nzWav*nxWav*nyWav*ntWav*4/(1024*1024*1024), " [GB]")
			print("nzWav=",nzWav)
			print("nxWav=",nxWav)
			print("nyWav=",nyWav)
			print("ntWav=",ntWav)
			self.pyOp = pyAcoustic_iso_float_tomoExt_3D.tomoExtShotsGpu_3D(velocity,paramP.param,sourceVector,sourcesSignalsFloat,receiversVector,extReflectivityFloat)
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
		with pyAcoustic_iso_float_tomoExt_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_tomoExt_3D.ostream_redirect():
			# print("Just before")
			self.pyOp.adjoint(add,model,data)
		return

	def setVel_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_tomoExt_3D.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def setExtReflectivity_3D(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_tomoExt_3D.ostream_redirect():
			self.pyOp.setExtReflectivity_3D(vel)
		return

	def deallocatePinnedTomoExtGpu_3D(self):
		with pyAcoustic_iso_float_tomoExt_3D.ostream_redirect():
			self.pyOp.deallocatePinnedTomoExtGpu_3D()
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_tomoExt_3D.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result
