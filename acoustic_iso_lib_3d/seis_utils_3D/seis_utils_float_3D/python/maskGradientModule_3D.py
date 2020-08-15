#Python module encapsulating PYBIND11 module
import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np

def maskGradientInit_3D(args):

	# IO object
	parObject=genericIO.io(params=args)

	# Check if user directly provides the mask for the gradient
	gradientMaskFile=parObject.getString("gradientMaskFile","noGradientMaskFile")
	# Read parameters
	bufferUp=parObject.getFloat("bufferUp",0) # Taper width above water bottom [km]
	bufferDown=parObject.getFloat("bufferDown",0) # Taper width below water bottom [km]
	taperExp=parObject.getFloat("taperExp",0) # Taper exponent
	fat=parObject.getInt("fat",4)
	velFile=parObject.getString("vel","noVelFile")
	vel=genericIO.defaultIO.getVector(velFile)
	wbShift=parObject.getFloat("wbShift",0) # Shift water bottom velocity [km] to start tapering at a different depth
	# waterVel=parObject.getFloat("waterVel",-1.0) # Shift water bottom velocity [km] to start tapering at a different depth

	# Case where the user wants to apply a mask but does not provide the file
	# The gradient is computed automatically by providing the following parameters
	if (gradientMaskFile=="noGradientMaskFile"):
		print("--- User has not provided a gradient mask file ---")
		print("--- Automatically generating the mask from the provided parameters ---")

	# The user provides the gradient mask file
	# If you provide both (parameters and mask file, the mask file has priority)
	else:
		print("--- User has provided a gradient mask file ---")

	return vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile

class maskGradient(Op.Operator):

	def __init__(self,domain,Range,vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile):

		# Set domain/range
		self.setDomainRange(domain,Range)

		# Case where the user wants to apply a mask but does not provide the file
		# The gradient is computed automatically
		if (gradientMaskFile=="noGradientMaskFile"):
			# Create mask and set it to zero
			self.mask=vel.clone()
			self.mask.set(0.0)
			maskNp=self.mask.getNdArray()

			# Compute water velocity (asssumed to be constant)
			velNp=vel.getNdArray()
			waterVel=1.5
			if (waterVel < 0.0):
				waterVel=velNp[fat][fat][fat]
				print("--- Water velocity value automatically identified as v = ", waterVel, " [km/s] ---")
			else:
				print("--- User input water velocity value of ", waterVel, " [km/s] ---")
			# Substract water velocity from velocity model
			velNp=velNp-waterVel

			# Compute index of water bottom
			nyVel=vel.getHyper().axes[2].n
			nxVel=vel.getHyper().axes[1].n
			ozVel=vel.getHyper().axes[0].o
			dzVel=vel.getHyper().axes[0].d
			nzVel=vel.getHyper().axes[0].n
			indexWaterBottom=np.zeros((nyVel,nxVel)) # Water bottom index
			depthWaterBottom=np.zeros((nyVel,nxVel)) # Water bottom depth [km]
			depthUpper=np.zeros((nyVel,nxVel)) # Upper bound depth [km]
			depthLower=np.zeros((nyVel,nxVel)) # Lower bound depth [km]
			indexUpper=np.zeros((nyVel,nxVel)) # Upper bound index
			indexLower=np.zeros((nyVel,nxVel)) # Lower bound index

			# Convert water bottom shift from km->samples
			iWbShift=int(wbShift/dzVel)

			# for iy in range(nyVel-2*fat):
			for iy in range(10):
				for ix in range(nxVel-2*fat):

					# Compute water bottom index
					indexWaterBottom[iy+fat][ix+fat]=np.argwhere(velNp[iy+fat][ix+fat][:]>0)[0][0]
					# print("iy = ", iy)
					# print("ix = ", ix)
					# print("indexWaterBottom = ", indexWaterBottom[iy+fat][ix+fat])
					indexWaterBottom[iy+fat][ix+fat]=indexWaterBottom[iy+fat][ix+fat]-1+iWbShift
					print("iy = ", iy)
					print("ix = ", ix)
					print("indexWaterBottom = ", indexWaterBottom[iy+fat][ix+fat])

					# if ( iy==1 and ix<10):
					# 	print("iy = ", iy)
					# 	print("ix = ", ix)
					# 	print("indexWaterBottom = ", indexWaterBottom[iy+fat][ix+fat])
					# 	print("velNp[iy+fat][ix+fat][:] = ", velNp[iy+fat][ix+fat][:])

					# Compute water bottom depth [km]
					depthWaterBottom[iy+fat][ix+fat]=ozVel+indexWaterBottom[iy+fat][ix+fat]*dzVel
					# print("depthWaterBottom = ", depthWaterBottom[iy+fat][ix+fat])

					# Compute water bottom upper bound
					depthUpper[iy+fat][ix+fat]=depthWaterBottom[iy+fat][ix+fat]-bufferUp # Upper bound [km]
					indexUpper[iy+fat][ix+fat]=(depthUpper[iy+fat][ix+fat]-ozVel)/dzVel # Upper bound [sample]
					# print("depthUpper = ", depthUpper[iy+fat][ix+fat])
					# print("indexUpper = ", indexUpper[iy+fat][ix+fat])

					# Compute water bottom upper bound [km]
					depthLower[iy+fat][ix+fat]=depthWaterBottom[iy+fat][ix+fat]+bufferDown # Lower bound [km]
					indexLower[iy+fat][ix+fat]=(depthLower[iy+fat][ix+fat]-ozVel)/dzVel # Lower bound [sample]
					# print("depthLower = ", depthLower[iy+fat][ix+fat])
					# print("indexLower = ", indexLower[iy+fat][ix+fat])

					iz1=int(indexUpper[iy+fat][ix+fat])
					iz2=int(indexLower[iy+fat][ix+fat])

					# Compute weight
					for iz in range(iz1,iz2):
						weight=(iz-iz1)/(iz2-iz1)
						weight=np.sin(np.pi*0.5*weight)
						maskNp[iy+fat][ix+fat][iz]=np.power(weight,taperExp)

					maskNp[iy+fat][ix+fat][iz2:]=1.0

		# Case where the user wants to apply a mask and provides the file for the mask
		else:
			self.mask=genericIO.defaultIO.getVector(gradientMaskFile,ndims=3)

		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		modelNp=model.getNdArray()
		maskNp=self.mask.getNdArray()
		if (not add):
			data.zero()
		dataNp=data.getNdArray()
		# dataNp+=modelNp*maskNp
		dataNp+=modelNp

		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		dataNp=data.getNdArray()
		maskNp=self.mask.getNdArray()
		if (not add):
			model.zero()
		modelNp=model.getNdArray()
		modelNp+=dataNp*maskNp

		return

	def getMask(self):
		return self.mask
