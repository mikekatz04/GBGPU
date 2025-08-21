# This module mostly aliases lisatools.utils.constants
from lisatools.utils.constants import *
import numpy as np

######################
# Physical constants #
######################

# Mass of Jupiter
Mjup = 1.898e27

#################
# LISA constant #
#################

# Transfer frequency
fstar = C_SI / (lisaL * 2 * np.pi)
