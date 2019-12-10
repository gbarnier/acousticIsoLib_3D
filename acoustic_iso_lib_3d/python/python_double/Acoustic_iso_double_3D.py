# Bullshit stuff
import pyAcoustic_iso_float_nl
import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np
from pyAcoustic_iso_double_nl_3D import deviceGpu_3D

################################################################################
############################ Acquisition geometry ##############################
################################################################################

############################### Sources' geometry ##############################
def buildSourceGeometry_3D(parObject,vel):

	#Common parameters
    info = parObject.getInt("info",0)
	sourceGeomFile = parObject.getString("sourceGeomFile","None")
	nts = parObject.getInt("nts")
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getInt("zDipoleShift",0)
	xDipoleShift = parObject.getInt("xDipoleShift",0)
	sourcesVector=[]
    spaceInterpMethod = parObject.getString("spaceInterpMethod","linear")

    if (info == 1):
        print("**** [buildSourceGeometry_3D]: User has requested to display information ****\n")
        if (sourceGeomFile != "None"):
            print("**** [buildSourceGeometry_3D]: User has provided a geometry file for the sources' locations ****\n")
        else:
            print("**** [buildSourceGeometry_3D]: User has requested a regular geometry for the sources' locations ****\n")
        if (spaceInterpMethod == "sinc"):
            print("**** [buildSourceGeometry_3D]: User has requested a sinc spatial interpolation method ****\n")
        else:
            print("**** [buildSourceGeometry_3D]: User has requested a linear spatial interpolation method ****\n")
        if (dipole == 1):
            print("**** [buildSourceGeometry_3D]: User has requested a dipole source injection/extraction ****\n")
            print("**** [buildSourceGeometry_3D]: Dipole shift in z-direction: %d [samples], %f [km]  ****\n" %(zDipoleShift, zDipoleShift*vel.getHyper().axes[0].d))
            print("**** [buildSourceGeometry_3D]: Dipole shift in x-direction: %d [samples], %f [km]  ****\n" %(xDipoleShift, xDipoleShift*vel.getHyper().axes[1].d))
            print("**** [buildSourceGeometry_3D]: Dipole shift in y-direction: %d [samples], %f [km]  ****\n" %(yDipoleShift, yDipoleShift*vel.getHyper().axes[2].d))

    # Check that the user does not want sinc interpolation + regular geometry
    if ( (spaceInterpMethod == "sinc") and (sourceGeomFile != "None")):
		print("**** ERROR [buildSourceGeometry_3D]: User can not request sinc spatial interpolation for a regular source geometry ****\n")
		quit()

	# Reading source geometry from file
	if(sourceGeomFile != "None"):
        nShot = parObject.getInt("nShot")
		sourceGeomVectorNd = genericIO.defaultIO.getVector(sourceGeomFile).getNdArray()
		zCoordFloat=SepVector.getSepVector(ns=[1])
		xCoordFloat=SepVector.getSepVector(ns=[1])
		#Check for consistency between number of shots and provided coordinates
		if(nShot != sourceGeomVectorNd.shape[1]):
			raise ValueError("ERROR [buildSourceGeometry_3D]: Number of shots (#shot=%s) not consistent with geometry file (#shots=%s)!"%(nShot,sourceGeomVectorNd.shape[1]))
		#Setting source geometry
		for ishot in range(nShot):
			#Setting z and x position of the source for the given experiment
			zCoordFloat.set(sourceGeomVectorNd[2,ishot])
			xCoordFloat.set(sourceGeomVectorNd[0,ishot])
			sourcesVector.append(deviceGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(), vel.getCpp(), nts, dipole, zDipoleShift, xDipoleShift))
		sourceAxis=Hypercube.axis(n=nShot,o=1.0,d=1.0)

    # Reading regular source geometry from parameters
    # Assumes all acquisition devices are located on finite-difference grid points
    else:

        # Shot number on each axis
        nxShot = parObject.getInt("nxShot")
        nyShot = parObject.getInt("nxShot")

        # Horizontal axes
        dx=vel.getHyper().axes[1].d
        ox=vel.getHyper().axes[1].o
        dy=vel.getHyper().axes[2].d
        oy=vel.getHyper().axes[2].o

        # Sources geometry
        nzSource=1
        dzSource=1
        nxSource=1
        dxSource=1
        nySource=1
        dySource=1
        ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
        oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
        oySource=parObject.getInt("ySource")-1+parObject.getInt("yPad")+parObject.getInt("fat")
        xSpacingShots=parObject.getInt("xSpacingShots")
        ySpacingShots=parObject.getInt("ySpacingShots")
        xSourceAxis=Hypercube.axis(n=nxShot,o=ox+oxSource*dx,d=xSpacingShots*dx)
        ySourceAxis=Hypercube.axis(n=nyShot,o=oy+oySource*dy,d=ySpacingShots*dy)

        # Setting source geometry
        for iyShot in range(nyShot):
            for ixShot in range(nxShot):
                sourcesVector.append(deviceGpu_3D(nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, nySource, oySource, dySource, vel.getCpp(), nts, dipole, zDipoleShift, xDipoleShift, yDipoleShift, "linear", 1))
                oxSource=oxSource+xSpacingShots # Shift source in x-direction
            oySource=oySource+spacingShots # Shift source in y-direction

	return sourcesVector,sourceAxis

############################### Receivers' geometry ############################
def buildReceiversGeometry(parObject,vel):

	# Horizontal axis
	dx=vel.getHyper().axes[1].d
	ox=vel.getHyper().axes[1].o
	nts = parObject.getInt("nts")
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getInt("zDipoleShift",2)
	xDipoleShift = parObject.getInt("xDipoleShift",0)

	nzReceiver=1
	ozReceiver=parObject.getInt("depthReceiver")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
	dzReceiver=1
	nxReceiver=parObject.getInt("nReceiver")
	oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
	dxReceiver=parObject.getInt("dReceiver")
	receiverAxis=Hypercube.axis(n=nxReceiver,o=ox+oxReceiver*dx,d=dxReceiver*dx)
	receiversVector=[]
	nRecGeom=1; # Constant receivers' geometry
	for iRec in range(nRecGeom):
		receiversVector.append(deviceGpu(nzReceiver,ozReceiver,dzReceiver,nxReceiver,oxReceiver,dxReceiver,vel.getCpp(),nts, dipole, zDipoleShift, xDipoleShift))

	return receiversVector,receiverAxis
