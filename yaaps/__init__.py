"""
YAAPS - Yet Another Athena Plot Script

A Python package for post-processing and plotting output from GRAthena++
(General Relativistic Athena++) simulations.

The main entry point is the Simulation class, which provides access to
simulation data and plotting functionality.

Example:
    >>> from yaaps import Simulation
    >>> sim = Simulation("/path/to/simulation/output")
    >>> sim.hst["time"]  # Access history data
    >>> sim.plot2d(time=100.0, var="rho")  # Create 2D density plot
"""

from .simulation import Simulation
