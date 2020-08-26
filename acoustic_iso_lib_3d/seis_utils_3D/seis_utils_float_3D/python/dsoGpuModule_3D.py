#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyDsoGpu_3D
import genericIO
import SepVector
import Hypercube
import numpy as np

def dsoGpuInit_3D(args):

	# IO object
	parObject=genericIO.io(params=args)

	nz=parObject.getInt("nz")
	nx=parObject.getInt("nx")
	nExt=parObject.getInt("nExt")
	fat=parObject.getInt("fat")
	dsoZeroShift=parObject.getFloat("dsoZeroShift")
	return nz,nx,nExt,fat,dsoZeroShift

class dsoGpu_3D(Op.Operator):

	def __init__(self,domain,range,nz,nx,nExt,fat,dsoZeroShift):

		self.setDomainRange(domain,range)
		self.pyOp = pyDsoGpu_3D.dsoGpu_3D(nz,nx,nExt,fat,dsoZeroShift)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyDsoGpu_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyDsoGpu_3D.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return
