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
	bathymetryFile=parObject.getString("bathymetryFile","noBathymetryFile")
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

	if (gradientMaskFile!="noGradientMaskFile"):
		print("--- [maskGradientModule_3D]: User has provided a gradient mask file ---")

	elif (bathymetryFile!="noBathymetryFile"):
		print("--- [maskGradientModule_3D]: User has provided a bathymetry file ---")

	# If you provide both (parameters and mask file, the mask file has priority)
	else:
		raise ValueError("**** ERROR [maskGradientModule_3D]: Please provide either a gradient mask file or a bathymetry file ****\n")

	return vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile,bathymetryFile

class maskGradient_3D(Op.Operator):

	def __init__(self,domain,Range,vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile,bathymetryFile):

		# Set domain/range
		self.setDomainRange(domain,Range)

		# Case where the user wants to apply a mask but does not provide the file
		# The gradient is computed automatically
		if (gradientMaskFile!="noGradientMaskFile"):
			self.mask=genericIO.defaultIO.getVector(gradientMaskFile,ndims=3)

		# Case where the user wants to apply a mask and provides the file for the mask
		elif (bathymetryFile!="noBathymetryFile"):

			# Read the bathymetry file
			bathymetry=genericIO.defaultIO.getVector(bathymetryFile,ndims=2)

			# Create mask and set it to zero
			self.mask=vel.clone()
			self.mask.set(0.0)
			maskNp=self.mask.getNdArray()

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
			bathymetryNp=bathymetry.getNdArray()

			# Convert water bottom shift from km->samples
			iWbShift=int(wbShift/dzVel)

			for iy in range(nyVel-2*fat):
				for ix in range(nxVel-2*fat):

					# Apply shift to water bottom depth [km]
					depthWaterBottom[iy+fat][ix+fat]=bathymetryNp[iy+fat][ix+fat]+wbShift

					# Compute upper bound [km] and [sample]
					depthUpper[iy+fat][ix+fat]=depthWaterBottom[iy+fat][ix+fat]-bufferUp
					indexUpper[iy+fat][ix+fat]=(depthUpper[iy+fat][ix+fat]-ozVel)/dzVel # Upper bound [sample]

					# Compute lower bound [km] and [sample]
					depthLower[iy+fat][ix+fat]=depthWaterBottom[iy+fat][ix+fat]+bufferDown
					indexLower[iy+fat][ix+fat]=(depthLower[iy+fat][ix+fat]-ozVel)/dzVel # Lower bound [sample]

					# Compute indices for upper and lower bound
					iz1=int(indexUpper[iy+fat][ix+fat])
					iz2=int(indexLower[iy+fat][ix+fat])

					if (iz1 < 0):
						raise ValueError("**** ERROR [maskGradientModule_3D]: Upper index for mask tapering is out of bounds for point (iz,ix) = (",iz,",",ix,"), iz-upper = ", iz1, "****\n")
					if (iz2 < 0):
						raise ValueError("**** ERROR [maskGradientModule_3D]: Lower index for mask tapering is out of bounds for point (iz,ix) = (",iz,",",ix,"), iz-lower = ", iz2, "****\n")

					# Compute weight
					for iz in range(iz1,iz2):
						weight=(iz-iz1)/(iz2-iz1)
						weight=np.sin(np.pi*0.5*weight)
						maskNp[iy+fat][ix+fat][iz]=np.power(weight,taperExp)

					maskNp[iy+fat][ix+fat][iz2:]=1.0

		else:
			raise ValueError("**** ERROR [maskGradientModule_3D]: User did not provide a bathymetry nor a gradient mask file ****\n")

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

	def getMask_3D(self):
		return self.mask

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		dataNp=data.getNdArray()
		maskNp=self.mask.getNdArray()
		if (not add):
			model.zero()
		modelNp=model.getNdArray()
		modelNp+=dataNp*maskNp
		return
