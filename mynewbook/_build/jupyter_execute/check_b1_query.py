#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.ipac.ned import Ned
from astropy.coordinates import SkyCoord, Galactic
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def which_source(source):
    result_table = Simbad.query_object(source)
    if(result_table == None):
        print("Unable to find", source.rstrip("\n"), ", please enter the source coordinates:")
        ra = float(input("ra (degrees):"))
        dec = float(input("dec (degrees):"))
    else:
        ra = result_table["RA"]
        dec = result_table["DEC"]
    return source, ra, dec

def read_B1():
    b1_law = np.genfromtxt("B1_law.txt", unpack = True)
    b1_ras = b1_law[1]
    b1_decs = b1_law[2]
    return b1_ras, b1_decs

def check_inB1_name(source):
    source, ra_source, dec_source = which_source(source)
    b1_ras, b1_decs = read_B1()
    c_source = SkyCoord(ra_source, dec_source, frame = "icrs", unit = (u.hourangle, u.deg))
    count = 0
    all_sep = []
    for b1_ra, b1_dec in zip(b1_ras, b1_decs):
        c_b1 = SkyCoord(b1_ra, b1_dec, frame = "icrs", unit = "deg")
        sep = c_source.separation(c_b1)
        all_sep.append(sep.value)
        
        if sep.value <= 10:
            count += 1
        else:
            continue

    if count == 0:
        print(source.rstrip('\n'), "is not in B1 law, minimal separation =", min(all_sep)*u.degree)
    else:
        print(source.rstrip('\n'), "is in B1 law")

def check_inB1_list(list_of_sources_file):
    list_of_source = open(list_of_sources_file, "r")
    
    for source in list_of_source:
        check_inB1_name(source)


# #### Using the source name

# In[3]:


check_inB1_name("Cygnus X-1")


# #### Using a list of sources

# In[4]:


check_inB1_list("sources.lst")


# In[ ]:


185.341428
30.176988


# In[ ]:





# In[ ]:




