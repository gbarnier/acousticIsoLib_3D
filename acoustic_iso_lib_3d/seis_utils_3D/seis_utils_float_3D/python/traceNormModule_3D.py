#Python module encapsulating PYBIND11 module
import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np
from numpy import linalg as LA

################################################################################
############################### Nonlinear forward ##############################
################################################################################
# Nonlinear operator
class traceNorm_3D(Op.Operator):

	def __init__(self,domain,range,epsilonTraceNorm):
		self.epsilonTraceNorm=epsilonTraceNorm
		if (self.epsilonTraceNorm <= 0.0):
		    raise ValueError("**** ERROR [phaseOnly_3D]: Epsilon-shift value must be > 0 ****\n")
		self.setDomainRange(domain,range)
		return

	def forward(self,add,model,data):
		# Check domain/range
		self.checkDomainRange(model,data)
		modelNp=model.getNdArray()
		dataNp=data.getNdArray()
		if (not add):
			data.zero()
		# Compute model norm for all traces
		modelNorm=LA.norm(modelNp,axis=2)
		dataNp[:]+=modelNp[:]/(np.expand_dims(modelNorm+self.epsilonTraceNorm,axis=2))
		return

################################################################################
############################### Jacobian operator ##############################
################################################################################
def traceNormDerivInit_3D(args):

	# IO object
	parObject=genericIO.io(params=args)

	# Allocate and read predicted data f(m0) (i.e, the "background" data)
	epsilonTraceNorm=parObject.getFloat("epsilonTraceNorm",1.0e-10)
	predDataFile=parObject.getString("predData")
	predDat=genericIO.defaultIO.getVector(predDataFile,ndims=3)
	predDatNp=predDat.getNdArray()

	return predDat,epsilonTraceNorm

class traceNormDeriv_3D(Op.Operator):

	def __init__(self,predDat,epsilonTraceNorm):

		# Set domain/range (same size as observed/predicted data)
		self.setDomainRange(predDat,predDat)
		self.epsilonTraceNorm=epsilonTraceNorm
		self.predDat=predDat
		return

	def forward(self,add,model,data):
		# Check domain/range
		self.checkDomainRange(model,data)
		modelNp=model.getNdArray()
		dataNp=data.getNdArray()
		predDatNp=self.predDat.getNdArray()
		if (not add):
			data.zero()

		# Compute inverse of predicted trace norm
		predDatNorm = np.expand_dims(LA.norm(predDatNp,axis=2),axis=2)
		predDatNormInv=1.0/predDatNorm
		predDatNormInvEps=1.0/(predDatNorm+self.epsilonTraceNorm)
		# Compute cube of inverse of predicted trace norm
		predDatNormCubeInv=predDatNormInvEps*predDatNormInvEps*predDatNormInv
		# Compute dot product between model and predicted trace
		dotProdDatMod=np.expand_dims(np.sum(predDatNp*modelNp,axis=2),axis=2)
		# Apply forward
		dataNp[:]+=modelNp*predDatNormInvEps-dotProdDatMod*predDatNormCubeInv*predDatNp

		return

	def adjoint(self,add,model,data):
		# Self-adjoint operator
		self.forward(add,data,model)
		return

	def setData(self,data):
		# Pointer assignement
		self.predDat=data
