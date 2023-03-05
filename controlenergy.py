"""
A Python module to perform network control theory calculations.

Author: Alec Galin
"""

import numpy as np
import scipy as sp
from nctpy.energies import gramian
from scipy.linalg import expm


def min_eng_cont(A, T, B, x0, xf):
    """
    Returns a tuple containing the state trajectory, control input, and
    error of a state transition.

    Parameter A: System adjacency matrix.
    Precondition: A is of type numpy.ndarray with shape (n, n).

    Parameter T: Control horizon
    Precondition: T is a float or int.

    Parameter A: Control input matrix.
    Precondition: A is of type numpy.ndarray.

    Parameter x0: Initial state
    Precondition: x0 is of type numpy.ndarray with size n.

    Parameter xf: Final state
    Precondition: xf is of type numpy.ndarray with size n.
    """
    n = A.shape[0]

    # Compute matrix exponential
    AT = np.block([[A, -.5*(B @ B.T)],
                   [np.zeros_like(A), -A]])
    E = expm(AT * T)

    # Compute costate initial condition
    E12 = E[:n, n:(2*n)]
    E11 = E[:n, :n]
    p0 = np.linalg.pinv(E12) @ (xf - E11 @ x0)

    # Compute Costate Initial Condition Error Induced by Inverse
    n_err = np.linalg.norm(E12 @ p0 - (xf - E11 @ x0))

    # Prepare Simulation
    nStep = 1000
    t = np.linspace(0, T, nStep+1)

    v0 = np.concatenate((x0, p0), axis=0)  # Initial condition
    v = np.zeros((2*n, len(t)))  # Trajectory
    Et = expm(AT * T/(len(t)-1))
    v[:, 0] = v0

    # Simulate State and Costate Trajectories
    for i in range(1, len(t)):
        v[:, i] = Et @ v[:, i-1]
    x = v[:n, :]
    u = -0.5 * B.T @ v[n:, :]

    u = u.T
    x = x.T

    return x, u, n_err
