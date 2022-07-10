"""Scludam.

Star Cluster Detection and Membership Probability Calculation

"""

from .detection import (
    CountPeakDetector,
    default_mask,
    extend_1dmask,
)
from .fetcher import Query, search_object, search_table
from .stat_tests import DipDistTest, HopkinsTest, RipleysKTest
from .shdbscan import SHDBSCAN
