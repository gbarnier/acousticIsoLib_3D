#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyDataTaper_3D
import genericIO
import SepVector
import Hypercube
import numpy as np

def dataTaperInit_3D(args):

	# IO object
	parObject=genericIO.io(params=args)

	offset=parObject.getInt("offset",0)
	time=parObject.getInt("time",0)
	t0=parObject.getFloat("t0",0.0)
	velMute=parObject.getFloat("velMute",0.0)
	expTime=parObject.getFloat("expTime",2.0)
	taperWidthTime=parObject.getFloat("taperWidthTime",0.0)
	moveout=parObject.getString("moveout","None")
	timeMuting=parObject.getString("timeMuting","None")
	maxOffset=parObject.getFloat("maxOffset",0.0)
	expOffset=parObject.getFloat("expOffset",2.0)
	taperWidthOffset=parObject.getFloat("taperWidthOffset",0.0)
	tPow=parObject.getFloat("tPow",0.0)
	offsetMuting=parObject.getString("offsetMuting","None")
	taperEndTraceWidth=parObject.getFloat("taperEndTraceWidth",0.0)
	sourceGeomFile=parObject.getString("sourceGeomFile","None")
	sourceGeometry=genericIO.defaultIO.getVector(sourceGeomFile,ndims=2)
	receiverGeomFile=parObject.getString("receiverGeomFile","None")
	receiverGeometry=genericIO.defaultIO.getVector(receiverGeomFile,ndims=3)
	dataMaskFile=parObject.getString("dataMaskFile","None")
	if dataMaskFile != "None":
		dataMask=genericIO.defaultIO.getVector(dataMaskFile,ndims=3)
	else:
		dataMask=None

	return t0,velMute,expTime,taperWidthTime,moveout,timeMuting,maxOffset,expOffset,taperWidthOffset,offsetMuting,taperEndTraceWidth,tPow,time,offset,sourceGeometry,receiverGeometry,dataMask

class dataTaper(Op.Operator):

	def __str__(self):
		"""Name of the operator"""
		return "dataTaper"

	def __init__(self,*args):
		domain = args[0]
		range = args[1]
		self.setDomainRange(domain,range)

		# Constructor for time and offset muting
		t0=args[2]
		velMute=args[3]
		expTime=args[4]
		taperWidthTime=args[5]
		moveout=args[6]
		reverseTime=args[7]
		maxOffset=args[8]
		expOffset=args[9]
		taperWidthOffset=args[10]
		reverseOffset=args[11]
		taperEndTraceWidth=args[12]
		tPow=args[13]
		time=args[14]
		offset=args[15]
		dataHyper=args[16]
		if("getCpp" in dir(dataHyper)):
			dataHyper = dataHyper.getCpp()
		sourceGeometry=args[17]
		if("getCpp" in dir(sourceGeometry)):
			sourceGeometry = sourceGeometry.getCpp()
		receiverGeometry=args[18]
		if("getCpp" in dir(receiverGeometry)):
			receiverGeometry = receiverGeometry.getCpp()
		dataMask = args[19]

		# Offset muting + end of trace tapering
		if (time==0 and offset==1):
			self.pyOp = pyDataTaper_3D.dataTaper_3D(maxOffset,expOffset,taperWidthOffset,reverseOffset,taperEndTraceWidth,tPow,dataHyper,sourceGeometry,receiverGeometry)

		# Time muting + end of trace tapering
		if (time==1 and offset==0):
			self.pyOp = pyDataTaper_3D.dataTaper_3D(t0,velMute,expTime,taperWidthTime,moveout,reverseTime,taperEndTraceWidth,tPow,dataHyper,sourceGeometry,receiverGeometry)

		# Time muting + end of trace tapering
		if (time==1 and offset==1):
			self.pyOp = pyDataTaper_3D.dataTaper_3D(t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,tPow,taperEndTraceWidth,dataHyper,sourceGeometry,receiverGeometry)

		# End of trace tapering
		if (time==0 and offset==0):
			self.pyOp=pyDataTaper_3D.dataTaper_3D(taperEndTraceWidth,tPow,dataHyper)

		# Adding user-defined mask if provided
		if dataMask is not None:
			if time == 0:
				raise ValueError("ERROR! Cannot use user provided dataMask muting if time muting is not requested!")
			taperMaskTime = self.getTaperMaskTime_3D()
			if not dataMask.checkSame(taperMaskTime):
				raise ValueError("ERROR! Data mask not consistent with data vector")
			taperMaskTime.getNdArray()[:] *= dataMask.getNdArray()
			print("---- [dataTaper_3D]: Adding Data taper provided from dataMaskFile ----")


		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
				model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		#Applying dataTaper operator
		with pyDataTaper_3D.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		"""Self-adjoint operator"""
		self.forward(add,data,model)
		return

	def getTaperMaskOffset_3D(self):
		with pyDataTaper_3D.ostream_redirect():
			# print("Here 1, taperMaskOffset = ", taperMaskOffset)
			taperMaskOffset = self.pyOp.getTaperMaskOffset_3D()
			taperMaskOffset = SepVector.floatVector(fromCpp=taperMaskOffset)
		return taperMaskOffset

	def getTaperMaskTime_3D(self):
		with pyDataTaper_3D.ostream_redirect():
			taperMaskTime = self.pyOp.getTaperMaskTime_3D()
			taperMaskTime = SepVector.floatVector(fromCpp=taperMaskTime)
		return taperMaskTime

	def getTaperMaskEndTrace_3D(self):
		with pyDataTaper_3D.ostream_redirect():
			taperMaskEndTrace = self.pyOp.getTaperMaskEndTrace_3D()
			taperMaskEndTrace = SepVector.floatVector(fromCpp=taperMaskEndTrace)
		return taperMaskEndTrace
