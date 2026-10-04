#-------- corrlib ------------------------------------------------------------
#-----------------------------------------------------------------------------
# A toolbox for correlation, noise and shared signals. Five questions, five files:
#
#   correlation_measures     which kind of relationship am I measuring?
#   sampling_estimators      how do I get it out of messy, irregular data?
#   factor_removal           what shared influences should come out first?
#   random_matrix_cleaning   how much of what I computed is real?
#   array_stacking           many sensors, one shared signal - or the reverse
#
# plus the Correlator, which chains them for one application, and plotting.
#
# Everything takes X with shape (n_variables, n_observations).

from . import correlation_measures
from . import sampling_estimators
from . import factor_removal
from . import random_matrix_cleaning
from . import array_stacking
from . import plotting
from .correlator import Correlator
