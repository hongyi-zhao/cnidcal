#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cnidcal.cnidcal as cdc
import numpy as np
from numpy import dot, cross, pi, square
from numpy.linalg import det, norm, inv
from cnidcal.celldraw import cellsdrawer
import matplotlib.pyplot as plt

# # Inpute two primitive lattices, a rotation matrix

# In[6]:


lattice_1 = np.array([[-2.734364, -2.734364,  0],
                      [-2.734364,  0, -2.734364],
                      [ 0, -2.734364, -2.734364]])
lattice_2 = np.array([[-1.48839924e+00, -1.48839924e+00, -5.44941797e-16],
                      [-2.57798310e+00,  2.57798310e+00,  8.58061028e-17],
                      [-3.08915674e-16,  4.34918912e-16, -2.53091000e+01]])
R = np.array([[-0.40824829, -0.70710678, -0.57735027],
              [-0.40824829,  0.70710678, -0.57735027],
              [ 0.81649658,  0.        , -0.57735027]])


# # Get the miller indices of the interface plane expressed in the primitive lattice (you can obtain this by transfoming the conventional miller indices)

# In[7]:


conv_lattice_1 = 2.734364 * 2 * np.eye(3)
miller_ind_pri = cdc.get_primitive_hkl(hkl = [1,1,1], C_lattice = conv_lattice_1, P_lattice = lattice_1)
miller_ind_pri


# # Get the two plane bases

# In[8]:


PB_1, PB_2 = cdc.get_two_bases(lattice_1, lattice_2, R, miller_ind_pri)
PB_1, PB_2


# # Compute the two-D CSL & CNID, and you can express them in the primitive lattice or conventional lattice frame

# In[9]:


CSL = cdc.search_CSL(PB_1, PB_2, 20)
CNID = cdc.searchcnid(PB_1, PB_2, 20)


# In[10]:


#CSL expressed in the primitive and conventional cell
CSL_in_P = np.round(cdc.get_coef_exp_ltc(lattice_1, CSL),5)
CSL_in_C = np.round(cdc.get_coef_exp_ltc(conv_lattice_1, CSL),5)
CSL_in_P, CSL_in_C


# In[11]:


#CNID expressed in primitive lattice 1
CNID_in_P = cdc.get_coef_exp_ltc(lattice_1, CNID)
cdc.get_fraction_basis(CNID_in_P)


# In[12]:


#CNID expressed in conventional lattice 1
CNID_in_C = cdc.get_coef_exp_ltc(conv_lattice_1, CNID)
cdc.get_fraction_basis(CNID_in_C)


# In[13]:


#CNID expressed in lattice 2
CNID_in_P = cdc.get_coef_exp_ltc(dot(R,lattice_2), CNID)
cdc.get_fraction_basis(CNID_in_P)


# # Visualize the cells

# A drawer class containing information of all cells

# In[14]:


my_cells = cellsdrawer(PB_1, PB_2, CSL, CNID, 20)


# The CSL lattice of the two plane basis

# In[15]:


#Show the two lattices and their CSL
my_cells.draw_direct(xlow = -26, xhigh = 9, ylow = -18.4,
                     yhigh = 8, figsize_x = 10, figsize_y = 10, show_legend = True)

#plt.show()  # 确保在函数调用后添加此行
# In[16]:


#Show the zoomed figure and DSC
my_cells.draw_direct(xlow = -6, xhigh = 9, ylow = -3, yhigh = 6, \
                     figsize_x = 10, figsize_y = 10, size_LP_1 = 500, \
                     size_LP_2 = 300, show_CNID_points = True, show_CNID_cell = True, save=True)

#plt.show()  # 确保在函数调用后添加此行

# The CSL lattice of the two reciprocal lattices

# In[17]:


#Show the two reciprocal lattices and their CSL
my_cells.draw_reciprocal(xlow = -1.4, xhigh = 1.4, ylow = -3, 
                         yhigh = 0.8, figsize_x = 10, figsize_y = 10, show_legend = True)

#plt.show()  # 确保在函数调用后添加此行
# In[18]:


#check the cell
print(my_cells.CNID_screen)

if True:
    pass



