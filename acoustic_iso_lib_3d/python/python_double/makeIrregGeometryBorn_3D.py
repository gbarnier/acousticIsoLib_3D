#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_double_3D
import numpy as np
import time
import sys
import random

if __name__ == '__main__':

	parObject=genericIO.io(params=sys.argv)

	# Read velocity file
	velFile=parObject.getString("vel")
	vel=genericIO.defaultIO.getVector(velFile, ndims=3)

	# Model coordinates + dimensions
	nz=vel.getHyper().axes[0].n
	dz=vel.getHyper().axes[0].d
	oz=vel.getHyper().axes[0].o
	fz=oz+(nz-1)*dz
	nx=vel.getHyper().axes[1].n
	dx=vel.getHyper().axes[1].d
	ox=vel.getHyper().axes[1].o
	fx=ox+(nx-1)*dx
	ny=vel.getHyper().axes[2].n
	dy=vel.getHyper().axes[2].d
	oy=vel.getHyper().axes[2].o
	fy=oy+(ny-1)*dy

	print("ox = ", ox)
	print("fx = ", fx)
	print("oy = ", oy)
	print("fy = ", fy)

	############################### Sources ####################################
	# Manual inputs
	nShot = 1
	zSource = 0.1

	# Source geometry
	sourceAxis1 = Hypercube.axis(n=nShot, o=0.0, d=1.0)
	sourceAxis2 = Hypercube.axis(n=3, o=0.0, d=1.0)
	sourceHyper = Hypercube.hypercube(axes=[sourceAxis1, sourceAxis2])
	sourceGeom = SepVector.getSepVector(sourceHyper)
	sourceGeomNp = sourceGeom.getNdArray()
	sourceGeomNp[2, :] = zSource

	# First shot
	sourceGeomNp[0, 0] = 3 # x-position
	sourceGeomNp[1, 0] = 3 # y-position

	# Second shot
	# sourceGeomNp[0, 1] = 2.0 # x-position
	# sourceGeomNp[1, 1] = 3.4 # y-position
	#
	# # Third shot
	# sourceGeomNp[0, 2] = 6.0 # x-position
	# sourceGeomNp[1, 2] = 2.8 # y-position
	#
	# # Fourth shot
	# sourceGeomNp[0, 3] = 4.5 # x-position
	# sourceGeomNp[1, 3] = 4.0 # y-position
	#
	# # Fifth shot
	# sourceGeomNp[0, 4] = 3.0 # x-position
	# sourceGeomNp[1, 4] = 2.4# y-position

	print("SourceGeom = ", sourceGeomNp.shape)
	print("sourceGeom axis 0 = ", sourceGeom.getHyper().axes[0].n)
	print("sourceGeomNp axis 1 = ", sourceGeom.getHyper().axes[1].n)

	# Geometry file
	sourceFile=parObject.getString("sourceFile")
	genericIO.defaultIO.writeVector(sourceFile, sourceGeom)

	############################### Receivers ##################################
	nRec = 30
	zRec = 0.06

	# Receivers
	recAxis1 = Hypercube.axis(n=nShot, o=0.0, d=1.0) # Shot index
	recAxis2 = Hypercube.axis(n=nRec, o=0.0, d=1.0) # Receivers' index
	recAxis3 = Hypercube.axis(n=3, o=0.0, d=1.0) # Spatial coordinates
	recHyper = Hypercube.hypercube(axes=[recAxis1, recAxis2, recAxis3])
	recGeom = SepVector.getSepVector(recHyper)
	recGeomNp = recGeom.getNdArray()
	recGeomNp[2, :, :] = zRec

	# Overwrite bounds
	for iShot in range(nShot):
		yMinBound = sourceGeomNp[1, iShot]
		yMaxBound = sourceGeomNp[1, iShot] + 8
		xMaxBound = sourceGeomNp[0, iShot] + 1.5
		xMinBound = sourceGeomNp[0, iShot] - 1.5

		xPos = sourceGeomNp[0, iShot] + np.linspace(-1.5, 1.5, 5)
		yPos = sourceGeomNp[1, iShot] + np.linspace(0.1, 3, 6)
		xPos, yPos = np.meshgrid(xPos, yPos)
		recGeomNp[0, :, iShot] = xPos.flatten()
		recGeomNp[1, :, iShot] = yPos.flatten()


		# xPos = sourceGeomNp[0, iShot] + np.linspace(-0.1, 0.1, 5)
		# yPos = sourceGeomNp[1, iShot] + np.linspace(-0.1, 0.1, 6)
		# xPos, yPos = np.meshgrid(xPos, yPos)
		# recGeomNp[0, :, iShot] = xPos.flatten()
		# recGeomNp[1, :, iShot] = yPos.flatten()

		# for iRec in range(nRec):
		# 	# y-position
		# 	xPos = random.uniform(xMinBound, xMaxBound)
		# 	# yPos = sourceGeomNp[0, iShot] + np.linspace(-0.1, 0.1, 5)
		# 	# x-position
		# 	yPos = random.uniform(yMinBound, yMaxBound)
		# 	# xPos = 0.1 + (iRec-1) * 0.01
		# 	recGeomNp[0, iRec, iShot] = xPos
		# 	recGeomNp[1, iRec, iShot] = yPos

	# What happens if two shots/two receivers are at the same point? Is it a problem?

	# print("RecGeom = ", recGeomNp.shape)
	# print("recGeomNp axis 0 = ", recGeom.getHyper().axes[0].n)
	# print("recGeomNp axis 1 = ", recGeom.getHyper().axes[1].n)
	# print("recGeomNp axis 2 = ", recGeom.getHyper().axes[2].n)
	#
	# print("x max receiver shot 1 = ", np.amax(recGeomNp[0,:,0]))
	# print("x min receiver shot 1 = ", np.amin(recGeomNp[0,:,0]))
	# print("y max receiver shot 1 = ", np.amax(recGeomNp[1,:,0]))
	# print("y min receiver shot 1 = ", np.amin(recGeomNp[1,:,0]))
	# print("z max receiver shot 1 = ", np.amax(recGeomNp[2,:,0]))
	# print("z min receiver shot 1 = ", np.amin(recGeomNp[2,:,0]))

	# print("x max receiver shot 2 = ", np.amax(recGeomNp[0,:,1]))
	# print("x min receiver shot 2 = ", np.amin(recGeomNp[0,:,1]))
	# print("y max receiver shot 2 = ", np.amax(recGeomNp[1,:,1]))
	# print("y min receiver shot 2 = ", np.amin(recGeomNp[1,:,1]))
	# print("z max receiver shot 2 = ", np.amax(recGeomNp[2,:,1]))
	# print("z min receiver shot 2 = ", np.amin(recGeomNp[2,:,1]))

	# print("x max receiver shot 3 = ", np.amax(recGeomNp[0,:,2]))
	# print("x min receiver shot 3 = ", np.amin(recGeomNp[0,:,2]))
	# print("y max receiver shot 3 = ", np.amax(recGeomNp[1,:,2]))
	# print("y min receiver shot 3 = ", np.amin(recGeomNp[1,:,2]))

	# Geometry file
	recFile=parObject.getString("recFile")
	genericIO.defaultIO.writeVector(recFile, recGeom)
