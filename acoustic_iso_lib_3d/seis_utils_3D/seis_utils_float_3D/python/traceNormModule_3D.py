#Python module encapsulating PYBIND11 module
import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import pyTraceNorm_3D
import numpy as np
from numpy import linalg as LA

################################################################################
############################### Nonlinear forward ##############################
################################################################################
# Nonlinear operator
class traceNorm_3D(Op.Operator):

	def __init__(self,domain,range,epsilonTraceNorm):
		if (epsilonTraceNorm <= 0.0):
		    raise ValueError("**** ERROR [phaseOnly_3D]: Epsilon-shift value must be > 0 ****\n")
		self.setDomainRange(domain,range)
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		self.pyOp = pyTraceNorm_3D.traceNorm_3D(domain,epsilonTraceNorm)
		return

	def forward(self,add,model,data):
		# Check domain/range
		self.checkDomainRange(model,data)
		if (not add):
			data.zero()
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		# Apply trace norm operator
		with pyTraceNorm_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
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

	return predDat,epsilonTraceNorm

class traceNormDeriv_3D(Op.Operator):

	def __init__(self,predDat,epsilonTraceNorm):
		# Set domain/range (same size as observed/predicted data)
		self.setDomainRange(predDat,predDat)
		if (epsilonTraceNorm <= 0.0):
		    raise ValueError("**** ERROR [phaseOnly_3D]: Epsilon-shift value must be > 0 ****\n")
		if("getCpp" in dir(predDat)):
			predDat = predDat.getCpp()
		self.pyOp = pyTraceNorm_3D.traceNormJac_3D(predDat,epsilonTraceNorm)
		return

	def forward(self,add,model,data):
		# Check domain/range
		self.checkDomainRange(model,data)
		if (not add):
			data.zero()
		# Applying Jacobian trace norm operator
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		# Apply trace norm operator
		with pyTraceNorm_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		# Self-adjoint operator
		self.forward(add,data,model)
		return

	def setData(self,data):
		# Pointer assignement
		if("getCpp" in dir(data)):
			data = data.getCpp()
		self.pyOp.setDat(data)
