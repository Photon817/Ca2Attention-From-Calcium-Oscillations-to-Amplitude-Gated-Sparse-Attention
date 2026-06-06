"""
CORC (Calcium-inspired Oscillatory Reservoir Computing)
AHER: Adaptive Hopf Event-coupled Reservoir
"""

__version__ = "0.1.0"

from .units import HopfSlowUnit
from .coupling import Coupling, LinearCoupling
from .reservoir import CORCReservoir, CORCState
from .observables import delay_embed, kuramoto_order, state_covariance_dimension
from . import tasks, baselines, analysis, plots
