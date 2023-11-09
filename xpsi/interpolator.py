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
				coderes = 512, 
				num_cells_theta = 512,
				num_cells_phi = 512,
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
		
		#Mdot is 1% of Eddington Limit [Mdot = 0.01 * Ledd/(efficiency * c^2)]
		#Need local temperature.. TMA is the flux at infinity, so need to divide it by alpha to get the local flux
  
		x = data['X']
		y = data['Y']
		z = data['Z']
		thetabhac = data['theta']
		phibhac = data['phi']
		Flux = data['T_MArt']
		k = int(coderes/2)
		tracer = data['tr1']
		num_cells_lt = 0
		
		if (self.first_spot or self.second_spot or self.elsewhere_xpsi or self.everywhere_xpsi) is False:
			raise ValueError('Neither of the 2 hotspot or elsewhere region specified')
		#Generating the required segments
		if self.first_spot:

			index_i = 0
			index_j = int(k - num_cells_lt)

			phicode = np.zeros((index_j-index_i,coderes))
			thetacode = np.zeros((index_j-index_i,coderes))
			Fluxcode = np.zeros((index_j-index_i,coderes))
			tracercode = np.zeros((index_j-index_i,coderes))

			for i in range(index_i,index_j):
				for j in range(0,k):

					phicode[i,j+k] = phibhac[j+i*coderes]#np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i,j+k] = thetabhac[j+i*coderes]#np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						#+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Fluxcode[i,j+k] = Flux[j+i*coderes]
					tracercode[i,j+k] = tracer[j+i*coderes]
				for j in range(k,coderes):

					phicode[i,j-k] = phibhac[j+i*coderes]#np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i,j-k] = thetabhac[j+i*coderes]#np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						#+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Fluxcode[i,j-k] = Flux[j+i*coderes]
					tracercode[i,j-k] = tracer[j+i*coderes]
			
		if self.second_spot:	

			index_i = int(k + num_cells_lt)
			index_j = coderes

			phicode = np.zeros((index_j-index_i,coderes))
			thetacode = np.zeros((index_j-index_i,coderes))
			Fluxcode = np.zeros((index_j-index_i,coderes))
			tracercode = np.zeros((index_j-index_i,coderes))

			for i in range(index_i,index_j):
				for j in range(0,k):

					phicode[i-index_i,j+k] = phibhac[j+i*coderes]#np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i-index_i,j+k] = thetabhac[j+i*coderes]#np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						#+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Fluxcode[i-index_i,j+k] = Flux[j+i*coderes]
					tracercode[i-index_i,j+k] = tracer[j+i*coderes]

				for j in range(k,coderes):

					phicode[i-index_i,j-k] = phibhac[j+i*coderes]#np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i-index_i,j-k] = thetabhac[j+i*coderes]#np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						#+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Fluxcode[i-index_i,j-k] = Flux[j+i*coderes]
					tracercode[i-index_i,j-k] = tracer[j+i*coderes]

		if self.elsewhere_xpsi:

			num_cells_lt = 64
			index_i = int(k - num_cells_lt)+1
			index_j = int(k + num_cells_lt)-1

			phicode = np.zeros((index_j-index_i,coderes))
			thetacode = np.zeros((index_j-index_i,coderes))
			Fluxcode = np.zeros((index_j-index_i,coderes))
			tracercode = np.zeros((index_j-index_i,coderes))

			for i in range(index_i,index_j):
				for j in range(0,k):

					phicode[i-index_i,j+k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i-index_i,j+k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Fluxcode[i-index_i,j+k] = Flux[j+i*coderes]
					tracercode[i-index_i,j+k] = tracer[j+i*coderes]

				for j in range(k,coderes):

					phicode[i-index_i,j-k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i-index_i,j-k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Fluxcode[i-index_i,j-k] = Flux[j+i*coderes]
					tracercode[i-index_i,j-k] = tracer[j+i*coderes]

		if self.everywhere_xpsi:

			index_i = 0
			index_j = coderes-1

			phicode = np.zeros((index_j-index_i,coderes))
			thetacode = np.zeros((index_j-index_i,coderes))
			Fluxcode = np.zeros((index_j-index_i,coderes))
			tracercode = np.zeros((index_j-index_i,coderes))

			for i in range(index_i,index_j):
				for j in range(0,k):

					phicode[i,j+k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i,j+k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Fluxcode[i,j+k] = Flux[j+i*coderes]
					tracercode[i,j+k] = tracer[j+i*coderes]
				for j in range(k,coderes):

					phicode[i,j-k] = np.arctan2(y[j+i*coderes],x[j+i*coderes]) 
					thetacode[i,j-k] = np.arccos(z[j+i*coderes]/np.sqrt(x[j+i*coderes]**2 \
						+ y[j+i*coderes]**2 + z[j+i*coderes]**2))

					Fluxcode[i,j-k] = Flux[j+i*coderes]
					tracercode[i,j-k] = tracer[j+i*coderes]

		return phicode,thetacode,Fluxcode,tracercode

#First find the nearest index for XPSI grid
	def nearestpoint(self,thetacode,phicode,xpsitheta,xpsiphi):

		ix = np.searchsorted(thetacode[:,1], xpsitheta,side='right')
		if (ix < np.shape(phicode)[0]-1):
			iy = np.searchsorted(phicode[ix,:], xpsiphi,side='right')
		else:
			iy = np.searchsorted(phicode[np.shape(phicode)[0]-1,:], xpsiphi,side='right')
		return ix,iy

#Now interpolate
	def temp_interpolation_func(self,
							coderes,
							thetacode,
							phicode,
							Fluxcode,
							tracercode,
							tracer_threshold,
							T_everywhere):

		xpsitheta = self.xpsi_theta
		xpsiphi = self.xpsi_phi

		lenphi = (np.shape(self.xpsi_phi)[1])
		lentheta = (np.shape(self.xpsi_phi)[0])
		Fluxxpsi = np.zeros((lentheta,lenphi))
		Tempxpsi = np.zeros((lentheta,lenphi))
		Tracerxpsi = np.zeros((lentheta,lenphi))

		#Constants..................................................................
		solarmass = 1.989e33
		c = 2.99e10
		G = 6.67259e-8
		rg = G * 1.68 * solarmass/(c * c)
		Ledd = 1.26e38   #1.68 mass of the NS in solar mass
		Mdot_units = 0.01*Ledd/(0.1 * c * c)
		rho_units = Mdot_units/(rg * rg * c ) #for efficiency = 0.1
		sigma = 5.6704e-5
		#...........................................................................

		for i in range(0,lentheta):
			for j in range(0,lenphi):

				ix2,iy2 = self.nearestpoint(thetacode,phicode,xpsitheta[i,j],xpsiphi[i,j])
				ix1 = ix2 - 1
				iy1 = iy2 - 1

				if (ix1 < np.shape(thetacode)[0]-1 and iy1 < np.shape(thetacode)[1]-1):
				
					x1 = phicode[ix1,iy1]
					x2 = phicode[ix2,iy2]
				
					y1 = thetacode[ix2,iy1]
					y2 = thetacode[ix1,iy2]

					#phi second index, theta first index
					c1 = Fluxcode[ix1,iy1] * (x2 - xpsiphi[i,j])/(x2 - x1) + \
						Fluxcode[ix2,iy1] * (xpsiphi[i,j] - x1)/(x2 - x1)

					c2 = Fluxcode[ix1,iy2] * (x2 - xpsiphi[i,j])/(x2 - x1) + \
						Fluxcode[ix2,iy2] * (xpsiphi[i,j] - x1)/(x2 - x1)

					c3 = tracercode[ix1,iy1] * (x2 - xpsiphi[i,j])/(x2 - x1) + \
						tracercode[ix2,iy1] * (xpsiphi[i,j] - x1)/(x2 - x1)

					c4 = tracercode[ix1,iy2] * (x2 - xpsiphi[i,j])/(x2 - x1) + \
						tracercode[ix2,iy2] * (xpsiphi[i,j] - x1)/(x2 - x1)
            
					Tracerxpsi[i,j] = c3 * (y2 - xpsitheta[i,j])/(y2 - y1) + \
						c4 * (xpsitheta[i,j] - y1)/(y2 - y1)

					Fluxxpsi[i,j] = (c1 * (y2 - xpsitheta[i,j])/(y2 - y1) + \
						c2 * (xpsitheta[i,j] - y1)/(y2 - y1)) 
					#Fluxxpsi[i,j] = (c1 * (y2 - xpsitheta[i,j])/(y2 - y1) + \
					#	c2 * (xpsitheta[i,j] - y1)/(y2 - y1)) * Tracerxpsi[i,j]
     
					if (Tracerxpsi[i,j] < tracer_threshold):
						Fluxxpsi[i,j] = T_everywhere
						Tempxpsi[i,j] = Fluxxpsi[i,j]
					else:
						Fluxxpsi[i,j] = (c1 * (y2 - xpsitheta[i,j])/(y2 - y1) + \
							c2 * (xpsitheta[i,j] - y1)/(y2 - y1))
						Tempxpsi[i,j] = (_np.abs(Fluxxpsi[i,j])*rho_units*c**3/sigma)**(1./4.)

				else:
					#At the theta or phi boundary so, doing a zero order interpolation
					if (iy1 > np.shape(phicode)[0]-1 and ix1 > np.shape(phicode)[0]-1):
						Tracerxpsi[i,j] = tracercode[ix1-1,iy1-1]
						Fluxxpsi[i,j] = Fluxcode[ix1-1,iy1-1]
						if (Tracerxpsi[i,j] < tracer_threshold):
							Fluxxpsi[i,j] = T_everywhere
							Tempxpsi[i,j] = Fluxxpsi[i,j]
						else:
							Fluxxpsi[i,j] = Fluxcode[ix1-1,iy1-1]
							Tempxpsi[i,j] = (_np.abs(Fluxxpsi[i,j])*rho_units*c**3/sigma)**(1./4.)

					elif (ix1 > np.shape(phicode)[0]-1):
						Tracerxpsi[i,j] = tracercode[ix1-1,iy1]
						Fluxxpsi[i,j] = Fluxcode[ix1-1,iy1]
						if (Tracerxpsi[i,j] < tracer_threshold):
							Fluxxpsi[i,j] = T_everywhere
							Tempxpsi[i,j] = Fluxxpsi[i,j]
						else:
							Fluxxpsi[i,j] = Fluxcode[ix1-1,iy1]
							Tempxpsi[i,j] = (_np.abs(Fluxxpsi[i,j])*rho_units*c**3/sigma)**(1./4.)

					elif (iy1 > np.shape(phicode)[0]-1):
						Tracerxpsi[i,j] = tracercode[ix1,iy1-1]				
						Fluxxpsi[i,j] = Fluxcode[ix1,iy1-1]
						if (Tracerxpsi[i,j] < tracer_threshold):
							Fluxxpsi[i,j] = T_everywhere
							Tempxpsi[i,j] = Fluxxpsi[i,j]
						else:
							Fluxxpsi[i,j] = Fluxcode[ix1,iy1-1]
							Tempxpsi[i,j] = (_np.abs(Fluxxpsi[i,j])*rho_units*c**3/sigma)**(1./4.)

		return _np.log10(Tempxpsi)

		def lorentz_interp(self,
							coderes,
							thetacode,
							phicode,
							Fluxcode):


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
