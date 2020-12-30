#Python module for subsurface offsets to angle conversion in 3D
import pyOperator as Op
import SepVector
import numpy as np

epsilon = 1e-16

class off2ang3D(Op.Operator):
	"""Operator class to transform ADCIGs to ODCIGs and vice versa"""

	def __init__(self,domain,range,oz,dz,ohx,dhx,ohy,dhy,og,dg,op,dp,p_inv=True,anti_alias=True):
		"""
		   Operator to convert imagese from/to angles to/from subsurface offsets
		   :param domain: vector class, Vector defining the size of the angle-domain image (z,x,y,gamma,phi)
		   :param range: vector class, Vector defining the size of the subsurface-offset-domain image (z,x,y,hx,hy)
		   :param oz: int, Origin of the z axis
		   :param dz: int, Sampling of the z axis
		   :param ohx: int, Origin of the subsurface-offset x axis
		   :param dhx: int, Sampling of the subsurface-offset x axis
		   :param ohy: int, Origin of the subsurface-offset y axis
		   :param dhy: int, Sampling of the subsurface-offset y axis
		   :param og: int, Origin of the angle axis in degree
		   :param dg: int, Sampling of the angle axis in degree
		   :param op: int, Origin of the azimuth axis in degree
		   :param dp: int, Sampling of the azimuth axis in degree
		   :param p_inv: boolean, whether to apply pseudo-inverse or simple adjoint operator [True]
		   :param anti_alias: boolean, applying a mask to avoid sampling of high k_hx and k_hy components (i.e., |kz*tg(gamma)*cos(phi)| < khx_max and |kz*tg(gamma)*sin(phi)| < khy_max) [True]
		"""
		self.setDomainRange(domain,range)
		# Number of elements
		self.nz = domain.getNdArray().shape[-1]
		if self.nz != range.getNdArray().shape[-1]:
			raise ValueError("Number of element of the image spaces must be the same (z samples: offset %s, angle %s )"%(self.nz, range.getNdArray().shape[-1]))
		self.np = domain.getNdArray().shape[0]
		self.ng = domain.getNdArray().shape[1]
		self.nhy = range.getNdArray().shape[0]
		self.nhx = range.getNdArray().shape[1]
		# Origins
		self.oz = oz
		self.ohx = ohx
		self.ohy = ohy
		self.og = og*np.pi/180.0
		self.op = op*np.pi/180.0
		# Sampling
		self.dz = dz
		self.dhx = dhx
		self.dhy = dhy
		self.dg = dg*np.pi/180.0
		self.dp = dp*np.pi/180.0
		self.p_inv = p_inv
		self.anti_alias = anti_alias


	def forward(self,add,model,data):
		"""Method to convert extended image from angles to offsets"""
		self.checkDomainRange(model,data)
		if not add:
			data.zero()

		# Getting Nd arrays
		m_arr = model.getNdArray()
		d_arr = data.getNdArray()

		# kz sampling information
		kz_axis = 2.0*np.pi*np.fft.rfftfreq(self.nz,self.dz)
		khx_max = 2.0*np.pi*np.max(np.fft.rfftfreq(self.nhx,self.dhx))
		khy_max = 2.0*np.pi*np.max(np.fft.rfftfreq(self.nhy,self.dhy))

		shape_data = list(d_arr.shape)
		shape_data[4] = kz_axis.shape[0]
		d_tmp = np.zeros(shape_data, dtype=complex)
		
		# Fourier transform of input ADCIGs
		m_kz = np.fft.rfft(m_arr, n=self.nz, axis=-1, norm="ortho")
		
		# Precomputing scaling factor for transformation
		hx_axis = np.linspace(self.ohx,self.ohx+(self.nhx-1)*self.dhx,self.nhx)
		hy_axis = np.linspace(self.ohy,self.ohy+(self.nhy-1)*self.dhy,self.nhy)
		g_vals = np.linspace(self.og,self.og+(self.ng-1)*self.dg,self.ng)
		p_vals = np.linspace(self.op,self.op+(self.np-1)*self.dp,self.np)
		exp_arg = np.expand_dims(np.expand_dims(np.outer(np.tan(g_vals),kz_axis),axis=1),axis=1)
		exp_arg_hx_phi = np.zeros((self.np,)+exp_arg.shape)
		exp_arg_hy_phi = np.zeros((self.np,)+exp_arg.shape)
		for p_idx,p_val in enumerate(p_vals):
			exp_arg_hx_phi[p_idx,:] = exp_arg_hx_phi*np.cos(p_val)
			exp_arg_hy_phi[p_idx,:] = exp_arg_hx_phi*np.sin(p_val)		
			# Applying anti-aliasing filter if requested
			if self.anti_alias:
				m_kz *= np.expand_dims(np.expand_dims((np.abs(np.outer(np.tan(g_vals),kz_axis)*np.cos(p_val)) <= khx_max).astype(np.int),axis=1),axis=1) * np.expand_dims(np.expand_dims((np.abs(np.outer(np.tan(g_vals),kz_axis)*np.sin(p_val)) <= khy_max).astype(np.int),axis=1),axis=1)
		
		for hy_idx,hy_val in enumerate(hy_axis):
			for hx_idx,hx_val in enumerate(hy_axis):
			d_tmp[hy_idx,hx_idx,:,:,:] = np.sum(m_kz[:,:,:,:,:]*np.exp(-1.0j*(exp_arg_hx_phi*hx_val-exp_arg_hy_phi*hy_val)),axis=(0,1))
		d_arr += np.real(np.fft.irfft(d_tmp, n=self.nz, axis=-1, norm="ortho"))
		return


	def adjoint(self,add,model,data):
		"""Method to convert extended image from offsets to angles"""
		self.checkDomainRange(model,data)
		if not add:
			model.zero()

		# Getting Nd arrays
		m_arr = model.getNdArray() # ADCIGs
		d_arr = data.getNdArray()  # ODCIGs

		# kz sampling information
		kz_axis = 2.0*np.pi*np.fft.rfftfreq(self.nz,self.dz)
		khx_max = 2.0*np.pi*np.max(np.fft.rfftfreq(self.nhx,self.dhx))
		khy_max = 2.0*np.pi*np.max(np.fft.rfftfreq(self.nhy,self.dhy))

		shape_model = list(m_arr.shape)
		shape_model[4] = kz_axis.shape[0]
		m_tmp = np.zeros(shape_model, dtype=complex)
		#Temporary I(z,x,y,hx)
		shape_data = list(d_arr.shape)
		shape_data[4] = kz_axis.shape[0]
		m_tmp_hx = np.zeros(shape_data[1:], dtype=complex)

		# Fourier transform of input ODCIGs
		d_kz = np.fft.rfft(d_arr, n=self.nz, axis=-1, norm="ortho")

		# Precomputing scaling factor for transformation
		hx_axis = np.linspace(self.ohx,self.ohx+(self.nhx-1)*self.dhx,self.nhx)
		hy_axis = np.linspace(self.ohy,self.ohy+(self.nhy-1)*self.dhy,self.nhy)
		g_vals = np.linspace(self.og,self.og+(self.ng-1)*self.dg,self.ng)
		p_vals = np.linspace(self.op,self.op+(self.np-1)*self.dp,self.np)
		exp_arg_hy = np.expand_dims(np.expand_dims(np.expand_dims(1.0j*np.outer(hy_axis,kz_axis),axis=1),axis=1),axis=1)
		exp_arg_hx = np.expand_dims(np.expand_dims(1.0j*np.outer(hx_axis,kz_axis),axis=1),axis=1)

		for p_idx,p_val in enumerate(p_vals):
			for g_idx,g_val in enumerate(g_vals):
				scale = kz_axis*kz_axis*np.tan(g_val)/(np.cos(g_val)*np.cos(g_val)+epsilon) if self.p_inv else 1.0
				if self.anti_alias:
					# Applying anti-aliasing filter if requested
					mask_hy = (np.abs(np.tan(g_val)*np.sin(p_val)*kz_axis) <= khy_max).astype(np.int)
					mask_hx = (np.abs(np.tan(g_val)*np.cos(p_val)*kz_axis) <= khx_max).astype(np.int)
					m_tmp_hx[:] = np.sum(d_kz[:,:,:,:,:]*np.exp(-np.tan(g_val)*np.sin(p_val)*exp_arg_hy)*mask_hy[:],axis=0)
					m_tmp[p_idx,g_idx,:,:] = scale*np.sum(m_tmp_hx[:,:,:,:]*np.exp(np.tan(g_val)*np.cos(p_val)*exp_arg_hx)*mask_hx[:],axis=0)
				else:
					m_tmp_hx[:] = np.sum(d_kz[:,:,:,:,:]*np.exp(-np.tan(g_val)*np.sin(p_val)*exp_arg_hy),axis=0)
					m_tmp[p_idx,g_idx,:,:] = scale*np.sum(m_tmp_hx[:,:,:,:]*np.exp(np.tan(g_val)*np.cos(p_val)*exp_arg_hx),axis=0)
		m_arr[:] += np.real(np.fft.irfft(m_tmp, n=self.nz, axis=-1, norm="ortho"))
		return
