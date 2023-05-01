from __future__ import division, print_function
import pandas as pd
import numpy as np

from .global_imports import *
from . import global_imports

from .cellmesh.mesh_tools import allocate_cells as _allocate_cells
from .cellmesh.mesh import construct_spot_cellMesh as _construct_spot_cellMesh
from .cellmesh.polar_mesh import construct_polar_cellMesh as _construct_polar_cellMesh
from .cellmesh.rays import compute_rays as _compute_rays

from .Parameter import Parameter, Derive
from .ParameterSubspace import ParameterSubspace
import xpsi
from xpsi.global_imports import _c, _G, _dpr, gravradius, _csq, _km, _2pi
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate 

class BHAC_Interpolator(ParameterSubspace):

	required_names = ['filename',
					'num_cells_theta',
					'num_cells_phi',
					'mytheta',
					'myphi',
					'first_spot',
					'second_spot',
					'elsewhere_xpsi',
					'everywhere_xpsi']

	def __init__(self,
				filename ='data',
				coderes = 256, 
				num_cells_theta = 256,
				num_cells_phi = 256,
				xpsi_theta = None,
				xpsi_phi = None,
				first_spot = False,
				second_spot = False,
				elsewhere_xpsi = False,
				everywhere_xpsi = False,
				**kwargs):

		self.filename=filename
		self.num_cells_theta = num_cells_theta
		self.coderes = int(coderes)
		self.num_cells_phi = num_cells_phi
		self.xpsi_theta = xpsi_theta
		self.xpsi_phi = xpsi_phi
		self.first_spot = first_spot
		self.second_spot = second_spot
		self.elsewhere_xpsi = elsewhere_xpsi
		self.everywhere_xpsi = everywhere_xpsi

	def read_regrid(self,filename,coderes):
		#Reading the csv file using pandas		
		data = pd.read_csv(filename)
		
		rg = 2.5e5
		c = 2.99e10
		Ledd = 1.3e38*(1.68)   #1.68 mass of the NS in solar mass
		rho_units = (0.1*Ledd/(4*np.pi*rg**2*c**3))
		sigma = 5.67e-5

		x = data['X']
		y = data['Y']
		z = data['Z']
		Temp = np.log10((-data['T_MArt']*rho_units*c**3/sigma)**(1./4.))
		k = int(coderes/2)

		num_cells_lt = 0
		
		if (self.first_spot or self.second_spot or self.elsewhere_xpsi or self.everywhere_xpsi) is False:
			raise ValueError('Neither of the 2 hotspot or elsewhere region specified')
		#Generating the required segments
		if self.first_spot:

			index_i = 0
			index_j = int(k - num_cells_lt)

			phicode = np.zeros((index_j-index_i,coderes))
			thetacode = np.zeros((index_j-index_i,coderes))
			Tempcode = np.zeros((index_j-index_i,coderes))

			for i in range(index_i,index_j):
				for j in range(0,k):

					phicode[i][j+k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i][j+k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Tempcode[i][j+k] = Temp[j+i*coderes]
				for j in range(k,coderes):

					phicode[i][j-k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i][j-k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Tempcode[i][j-k] = Temp[j+i*coderes]
			
		if self.second_spot:	

			index_i = int(k + num_cells_lt)
			index_j = coderes

			phicode = np.zeros((index_j-index_i,coderes))
			thetacode = np.zeros((index_j-index_i,coderes))
			Tempcode = np.zeros((index_j-index_i,coderes))

			for i in range(index_i,index_j):
				for j in range(0,k):

					phicode[i-index_i][j+k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i-index_i][j+k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Tempcode[i-index_i][j+k] = Temp[j+i*coderes]

				for j in range(k,coderes):

					phicode[i-index_i][j-k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i-index_i][j-k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Tempcode[i-index_i][j-k] = Temp[j+i*coderes]

		if self.elsewhere_xpsi:

			num_cells_lt = 64
			index_i = int(k - num_cells_lt)+1
			index_j = int(k + num_cells_lt)-1

			phicode = np.zeros((index_j-index_i,coderes))
			thetacode = np.zeros((index_j-index_i,coderes))
			Tempcode = np.zeros((index_j-index_i,coderes))

			for i in range(index_i,index_j):
				for j in range(0,k):

					phicode[i-index_i][j+k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i-index_i][j+k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Tempcode[i-index_i][j+k] = Temp[j+i*coderes]

				for j in range(k,coderes):

					phicode[i-index_i][j-k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i-index_i][j-k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Tempcode[i-index_i][j-k] = Temp[j+i*coderes]

		if self.everywhere_xpsi:

			index_i = 0
			index_j = coderes-1

			phicode = np.zeros((index_j-index_i,coderes))
			thetacode = np.zeros((index_j-index_i,coderes))
			Tempcode = np.zeros((index_j-index_i,coderes))

			for i in range(index_i,index_j):
				for j in range(0,k):

					phicode[i][j+k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i][j+k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Tempcode[i][j+k] = Temp[j+i*coderes]
				for j in range(k,coderes):

					phicode[i][j-k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i][j-k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Tempcode[i][j-k] = Temp[j+i*coderes]

		return phicode,thetacode,Tempcode

#First find the nearest index for XPSI grid
	def nearestpoint(self,thetacode,phicode,xpsitheta,xpsiphi):

		ix = np.searchsorted(thetacode[:,1], xpsitheta,side='right')
		if (ix < np.shape(phicode)[0]-1):
			iy = np.searchsorted(phicode[ix,:], xpsiphi,side='right')
		else:
			iy = np.searchsorted(phicode[np.shape(phicode)[0]-1,:], xpsiphi,side='right')
		return ix,iy

#Now interpolate
	def interpolation_func(self,
							coderes,
							thetacode,
							phicode,
							Tempcode):

		xpsitheta = self.xpsi_theta
		xpsiphi = self.xpsi_phi

		lenphi = (np.shape(self.xpsi_phi)[1])
		lentheta = (np.shape(self.xpsi_phi)[0])
		Tempxpsi = np.zeros((lentheta,lenphi))
		for i in range(0,lentheta):
			for j in range(0,lenphi):

				ix2,iy2 = self.nearestpoint(thetacode,phicode,xpsitheta[i,j],xpsiphi[i,j])
				ix1 = ix2 - 1
				iy1 = iy2 - 1

				if (ix1 < np.shape(thetacode)[0]-2 and iy1 < np.shape(phicode)[1]-2):
				
					x1 = phicode[ix1,iy1]
					x2 = phicode[ix2,iy2]
				
					y1 = thetacode[ix2,iy1]
					y2 = thetacode[ix1,iy2]

					#phi second index, theta first index
					c1 = Tempcode[ix1,iy1] * (x2 - xpsiphi[i,j])/(x2 - x1) + \
						Tempcode[ix2,iy1] * (xpsiphi[i,j] - x1)/(x2 - x1)

					c2 = Tempcode[ix1,iy2] * (x2 - xpsiphi[i,j])/(x2 - x1) + \
						Tempcode[ix2,iy2] * (xpsiphi[i,j] - x1)/(x2 - x1)

					Tempxpsi[i][j] = c1 * (y2 - xpsitheta[i,j])/(y2 - y1) + \
						c2 * (xpsitheta[i,j] - y1)/(y2 - y1)
				
				else:
					#At the theta or phi boundary so, doing a zero order interpolation
					if (iy1 >= np.shape(phicode)[0]-1 and ix1 >= np.shape(phicode)[0]-1):
						Tempxpsi[i,j] = Tempcode[ix1-1,iy1-1]		
					elif (ix1 >= np.shape(phicode)[0]-1):
						Tempxpsi[i,j] = Tempcode[ix1-1,iy1]
					elif (iy1 >= np.shape(phicode)[0]-1):
						Tempxpsi[i,j] = Tempcode[ix1,iy1-1]		
		return Tempxpsi

		def lorentz_interp(self,
							coderes,
							thetacode,
							phicode,
							Tempcode):


			return Tempxpsi

		def plot_sphere(self,theta,phi,temp):

			plt.rcParams["figure.figsize"] = [7.00, 3.50]
			plt.rcParams["figure.autolayout"] = True
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			r = 4.0
			x = r * np.sin(theta) * np.cos(phi)
			y = r * np.sin(theta) * np.sin(phi)
			z = r * np.cos(theta)
			ax.plot_surface(x, y, z, facecolors=cm.jet(temp))
			return 0 











		#super().__init__(*args, **kwargs)
