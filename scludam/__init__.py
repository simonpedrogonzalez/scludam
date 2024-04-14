# scludam, Star CLUster Detection And Membership estimation package
# Copyright (C) 2022  Simón Pedro González

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

Star Cluster Detection and Membership Probability Calculation main module.

Simplified API:
    - :class:`~scludam.detection.CountPeakDetector`,
      :func:`~scludam.detection.default_mask`,
      :func:`~scludam.detection.extend_1dmask` from :py:mod:`~scludam.detection`
    - :class:`~scludam.fetcher.Query`, :func:`~scludam.fetcher.search_object`,
      :func:`~scludam.fetcher.search_table`,
      :func:`~scludam.fetcher.search_objects_near_data`,
      :func:`~scludam.fetcher.search_table` from :py:mod:`~scludam.fetcher`
    - :class:`~scludam.hkde.HKDE`,
      :class:`~scludam.hkde.PluginSelector`,
      :class:`~scludam.hkde.RuleOfThumbSelector` from :py:mod:`~scludam.hkde`
    - :class:`~scludam.membership.DBME` from :py:mod:`~scludam.membership`
    - :class:`~scludam.pipeline.DEP` from :py:mod:`~scludam.pipeline`
    - :class:`~scludam.shdbscan.SHDBSCAN` from :py:mod:`~scludam.shdbscan`
    - :class:`~scludam.stat_tests.DipDistTest`,
      :class:`~scludam.stat_tests.HopkinsTest`,
      :class:`~scludam.stat_tests.RipleysKTest` from :py:mod:`~scludam.stat_tests`
"""

from .detection import CountPeakDetector, default_mask, extend_1dmask
from .fetcher import Query, search_object, search_objects_near_data, search_table
from .hkde import HKDE, PluginSelector, RuleOfThumbSelector
from .membership import DBME
from .pipeline import DEP
from .shdbscan import SHDBSCAN
from .stat_tests import DipDistTest, HopkinsTest, RipleysKTest
from .cli import launch
