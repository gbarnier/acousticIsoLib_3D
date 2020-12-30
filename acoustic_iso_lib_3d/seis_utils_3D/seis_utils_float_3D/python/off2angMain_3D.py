#!/usr/bin/env python3
"""

USAGE EXAMPLE:
	off2angMain_3D.py off_img= ang_img= nhx= ohx= dhx= nhy= ohy= dhy= ng= og= dg= np= op= dp= adj=1 p_inv=1 anti_alias=1

INPUT PARAMETERS:
"""

import genericIO
import SepVector
import Hypercube
from off2angModule_3D import off2ang3D
import numpy as np
import sys


if __name__ == '__main__':
	# Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

	# Getting parameter object
	parObject = genericIO.io(params=sys.argv)

	# Other parameters
	adj = parObject.getBool("adj",1)
	p_inv = parObject.getBool("p_inv",1)
	anti_alias = parObject.getBool("anti_alias",1)
	dp_test = parObject.getBool("dp_test",0)

	# Getting axes' info
	if adj:
		# angle axis (Forward)
		ng = parObject.getInt("ng")
		dg = parObject.getFloat("dg")
		og = parObject.getFloat("og")
		np = parObject.getInt("np")
		dp = parObject.getFloat("dp")
		op = parObject.getFloat("op")
	else:
		# subsurface-offset axis (Adjoint)
		nhx = parObject.getInt("nhx")
		dhx = parObject.getFloat("dhx")
		ohx = parObject.getFloat("ohx")
		nhy = parObject.getInt("nhy")
		dhy = parObject.getFloat("dhy")
		ohy = parObject.getFloat("ohy")


	# Check whether file names were provided or not
	off_img_file=parObject.getString("off_img","None")
	if off_img_file == "None":
		raise ValueError("**** ERROR: User did not provide subsurface-offset-domain image file ****")

	ang_img_file=parObject.getString("ang_img","None")
	if ang_img_file == "None":
		raise ValueError("**** ERROR: User did not provide angle-domain image file ****")

	# Applying forward
	if not adj:
		# Read offset-domain image
		ADCIGs = genericIO.defaultIO.getVector(ang_img_file,ndims=5)
		# Getting axis
		z_axis = ADCIGs.getHyper().getAxis(1)
		x_axis = ADCIGs.getHyper().getAxis(2)
		y_axis = ADCIGs.getHyper().getAxis(3)
		g_axis = ADCIGs.getHyper().getAxis(4)
		p_axis = ADCIGs.getHyper().getAxis(5)
		hx_axis = Hypercube.axis(n=nhx,o=ohx,d=dhx,label="hx")
		hy_axis = Hypercube.axis(n=nhy,o=ohy,d=dhy,label="hy")
		ODCIGs = SepVector.getSepVector(Hypercube.hypercube(axes=[z_axis,x_axis,y_axis,hx_axis,hy_axis]))

		# Constructing operator
		off2angOp = off2ang3D(ADCIGs,ODCIGs,z_axis.o,z_axis.d,hx_axis.o,hx_axis.d,hy_axis.o,hy_axis.d,g_axis.o,g_axis.d,p_axis.o,p_axis.d,p_inv,anti_alias)

		if dp_test:
			# Dot-product test if requested
			off2angOp.dotTest(True)
			quit()

		# Applying transformation
		off2angOp.forward(False,ADCIGs,ODCIGs)
		# Writing result
		ODCIGs.writeVec(off_img_file)

	# Applying adjoint
	else:
		# Read offset-domain image
		ODCIGs = genericIO.defaultIO.getVector(off_img_file,ndims=5)
		# Getting axis
		z_axis = ODCIGs.getHyper().getAxis(1)
		x_axis = ODCIGs.getHyper().getAxis(2)
		y_axis = ODCIGs.getHyper().getAxis(3)
		hx_axis = ODCIGs.getHyper().getAxis(4)
		hy_axis = ODCIGs.getHyper().getAxis(5)
		g_axis = Hypercube.axis(n=ng,o=og,d=dg,label="\F9 g \F-1 [deg]")
		p_axis = Hypercube.axis(n=np,o=op,d=dp,label="\F9 f \F-1 [deg]")
		ADCIGs = SepVector.getSepVector(Hypercube.hypercube(axes=[z_axis,x_axis,y_axis,g_axis,p_axis]))

		# Constructing operator
		off2angOp = off2ang3D(ADCIGs,ODCIGs,z_axis.o,z_axis.d,hx_axis.o,hx_axis.d,hy_axis.o,hy_axis.d,g_axis.o,g_axis.d,p_axis.o,p_axis.d,p_inv,anti_alias)
		if dp_test:
			# Dot-product test if requested
			off2angOp.dotTest(True)
			quit()

		# Applying transformation
		off2angOp.adjoint(False,ADCIGs,ODCIGs)
		# Writing result
		ADCIGs.writeVec(ang_img_file)
