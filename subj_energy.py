"""
A script that calculates and plots the transition energies between brain states 
for the LSD and psilocybin data from 
https://www.nature.com/articles/s41467-022-33578-1.

Author: Alec Galin
"""
from nctpy.utils import matrix_normalization
from nctpy.energies import minimum_energy_fast
import numpy as np
import controlenergy as ct
from scipy.linalg import eig
from scipy.stats import ttest_rel
import scipy.io
import pingouin as pg
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib import ticker

# Set base directory
basedir = '/Users/alecsmac/coding/coco/Energy_Landscape'

# Set variables for energy calculations
c = 0
T = 0.001
numClusters = 4

# Load Data
main = scipy.io.loadmat(
    basedir + "/data/main.mat")
partition = scipy.io.loadmat(
    basedir + '/results/example/Partition_bpmain_k4.mat')
sch454 = scipy.io.loadmat(
    basedir + '/data/5HTvecs_sch454.mat')
sch232 = scipy.io.loadmat(
    basedir + '/data/5HTvecs_sch454.mat')
ls463 = scipy.io.loadmat(
    basedir + '/data/5HTvecs_ls463.mat')
subjCentroids = scipy.io.loadmat(
    basedir + '/results/example/subjcentroids_splitmain_k4.mat')

# Extract Relevant Variables
overallCentroids = partition['centroids']
clusterNames = np.squeeze(partition['clusterNames'])
centroids = subjCentroids['centroids']
tot = main['nscans']*2
sc = main['sc']
nparc = main['nparc'][0][0]
nsubjs = main['nsubjs'][0][0]


def transition_energies(A, B, nsubjs, numClusters, nparc, centroids):
    """
    Returns (E_full, E_weighted), where E_full is the transition energy
    matrix for a uniform control input and E_weighted is the transition energy
    matrix for a weighted control input.

    Parameter A: System adjacency matrix.
    Precondition: A is a numpy.ndarray with shape (nparc, nparc) and is already 
    normalized.

    Parameter B: Control Matrix.
    Precondition: B is a numpy.ndarray.

    Parameter nsubjs: Number of subjects.
    Precondition: nsubjs is an int.

    Parameter numClusters: Number of clusters.
    Precondition: nsubjs is an int.

    Parameter nparc: Number of brain regions.
    Precondition: nparc is an int.

    Parameter centroids: Centroids of each subjects brain activity.
    Precondition: centroids is a numpy.ndarray.
    """
    # Specify final and initial states
    clusterVec = np.arange(1, numClusters + 1)
    Xf_ind = np.tile(clusterVec,
                     numClusters)
    Xo_ind = np.repeat(clusterVec, numClusters)

    # Compute transition energies
    E_full = np.full((nsubjs*2, numClusters**2), np.nan)
    E_weighted = np.full((nsubjs*2, numClusters**2), np.nan)
    for subj in range(nsubjs*2):
        x0 = np.squeeze(centroids[subj, :, Xo_ind - 1])
        xf = np.squeeze(centroids[subj, :, Xf_ind - 1])
        # Compute minimum control energy for all transitions
        e_fast = minimum_energy_fast(
            A_norm=A, T=T, B=np.eye(nparc), x0=x0.T, xf=xf.T)
        e_fast = np.sum(e_fast, axis=0)
        for tran in range(numClusters**2):
            # Set corresponding entry of E_full
            E_full[subj, tran] = e_fast[tran]
            # Compute E_weighted and set corresponding entry
            x, u, n_err = ct.min_eng_cont(A, T, B, x0[tran], xf[tran])
            E_weighted[subj, tran] = np.sum(u**2) * T/1001

    return E_full, E_weighted


def set_HT(nparc, sch454, sch232, ls463):
    """
    Returns the correct HT based on nparc.

    Parameter nparc: Number of brain regions.
    Precondition: nparc is an int. 
    """
    if nparc == 454:
        HT = sch454['mean5HT2A_sch454']
    elif nparc == 232:
        HT = sch232['mean5HT2A_sch232']
    elif nparc == 463:
        HT = ls463['mean5HT2A_ls463']
    elif nparc == 461:
        HT = ls463['mean5HT2A_ls463']
        HT = np.delete(HT, [13, 462], axis=0)
    elif nparc == 462:
        HT = ls463['mean5HT2A_ls463']
        HT = np.delete(HT, 13, axis=0)

    return HT


def calculate_energies(c, T, numClusters, save=False):
    """
    Returns (E_full, E_weighted) by applying the helper functions above to
    the data from https://github.com/singlesp/energy_landscape. E_full is the 
    transition energy matrix for a uniform control input 
    and E_weighted is the transition energy matrix for a weighted control input.

    If save is set to true, saves E_full and E_weighted as NumPy ".npy" files 
    to basedir.

    Parameter c: Normalization constant.
    Precondition: c is a float.

    Parameter T: Time scale.
    Precondition: T is a float.

    Parameter numClusters: Number of clusters.
    Precondition: numClusters is an int.

    Parameter save: Whether or not to save the transition energy matrices.
    Precondition: save is a bool.
    """
    HT = set_HT(nparc, sch454, sch232, ls463)

    # Normalize structural connectivity matrix
    Anorm = matrix_normalization(sc, c=c, system='continuous')

    # Create control matrix B for weighted transition energies
    norm = HT/HT.max()
    B = norm * np.eye(nparc) + np.eye(nparc)

    # Compute transition energies
    E_full, E_weighted = transition_energies(
        Anorm, B, nsubjs, numClusters, nparc, centroids)

    if save:
        np.save(basedir + '/E_full.npy', E_full)
        np.save(basedir + '/E_weighted.npy', E_weighted)

    return E_full, E_weighted


E_full, E_weighted = calculate_energies(c, T, numClusters)

# Uncomment to load E_full and E_weighted if saved
# E_full = np.load(basedir + '/E_full.npy')
# E_weighted = np.load(basedir + '/E_weighted.npy')

# Define variables for first plot
Energy1 = E_weighted[nsubjs:, :]
Energy2 = E_full[nsubjs:, :]
t, pavg = ttest_rel(Energy1, Energy2)
fdravg = multipletests(pavg, method='fdr_bh')[1]
fdravg = np.reshape(fdravg, (numClusters, numClusters)).T
pavg = np.reshape(pavg, (numClusters, numClusters)).T
grpAvgLSD = np.reshape(np.mean(Energy1, axis=0),
                       (numClusters, numClusters)).T
grpAvgPl = np.reshape(np.mean(Energy2, axis=0),
                      (numClusters, numClusters)).T
grpDiff = np.reshape(np.squeeze(t),
                     (numClusters, numClusters)).T
maxVal = np.max([grpAvgLSD, grpAvgPl])
minVal = np.min([grpAvgLSD, grpAvgPl])

# Store the variables in dictionary firstplot
firstplot = {}
firstplot['t'] = t
firstplot['pavg'] = pavg
firstplot['fdravg'] = fdravg
firstplot['grpAvgLSD'] = grpAvgLSD
firstplot['grpAvgPl'] = grpAvgPl
firstplot['grpDiff'] = grpDiff
firstplot['maxVal'] = maxVal
firstplot['minVal'] = minVal

# Define variables for second plot
t, pavg = ttest_rel(E_full[:nsubjs, :],
                    E_full[nsubjs:nsubjs*2, :])
t[np.isnan(t)] = 0
fdravg = multipletests(pavg, method='fdr_bh')[1]
fdravg = fdravg.reshape(numClusters, numClusters).T
pavg = pavg.reshape(numClusters, numClusters).T

grpAvgLSD = np.reshape(np.mean(E_full[0:nsubjs, :], axis=0), [
                       numClusters, numClusters]).T
grpAvgPL = np.reshape(
    np.mean(E_full[nsubjs:nsubjs*2, :], axis=0), [numClusters, numClusters]).T

grpDiff = np.reshape(np.squeeze(t),
                     (numClusters, numClusters)).T
maxVal = np.max([grpAvgLSD, grpAvgPL])
minVal = np.min([grpAvgLSD, grpAvgPL])


# Store the variables in dictionary secondplot
secondplot = {}
secondplot['t'] = t
secondplot['pavg'] = pavg
secondplot['fdravg'] = fdravg
secondplot['grpAvgLSD'] = grpAvgLSD
secondplot['grpAvgPl'] = grpAvgPL
secondplot['grpDiff'] = grpDiff
secondplot['maxVal'] = maxVal
secondplot['minVal'] = minVal


def fmt(x, pos):
    """
    Returns a number x formatted in scientific notation (to be used as a
    formatter for plotting).

    Parameter x: the number to formatted.
    Precondition: x is an int or float.

    Parameter pos: the position of the number to be formatted (when it is a tick
    in matPlotLib).
    Precondition: pos is an int.
    """
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def plot_data(dict):
    """
    Plots the transition energies between brains states for LSD and Psilocybin
    as well as the difference between those transition energies.

    Parameter dict: a dictionary containing information to plot.
    Precondition: dict is either firstplot or secondplot.
    """
    plt.figure(constrained_layout=True)

    # LSD
    plt.subplot(1, 3, 1)
    plt.imshow(dict['grpAvgLSD'], cmap='plasma')
    plt.xticks(range(numClusters), clusterNames, rotation=90)
    plt.yticks(range(numClusters), clusterNames)
    plt.colorbar(format=ticker.FuncFormatter(fmt))
    plt.xlabel('Final State')
    plt.ylabel('Initial State')
    plt.title('LSD')
    plt.tick_params(axis='both', which='both', length=0, labelsize=10)

    # Psilocybin
    plt.subplot(1, 3, 2)
    plt.imshow(dict['grpAvgPl'], cmap='plasma')
    plt.xticks(range(numClusters), clusterNames, rotation=90)
    plt.yticks(range(numClusters), clusterNames)
    plt.colorbar(format=ticker.FuncFormatter(fmt))
    plt.xlabel('Final State')
    plt.ylabel('Initial State')
    plt.title('PL')
    plt.tick_params(axis='both', which='both', length=0, labelsize=10)

    # Difference
    plt.subplot(1, 3, 3)
    plt.imshow(dict['grpDiff'], cmap='plasma')
    plt.xticks(range(numClusters), clusterNames, rotation=90)
    plt.yticks(range(numClusters), clusterNames)
    u_caxis_bound = np.max(np.max(dict['grpDiff']))
    l_caxis_bound = np.min(np.min(dict['grpDiff']))
    h = plt.colorbar()
    h.set_label('t-stat')
    h.set_ticks([l_caxis_bound, (u_caxis_bound+l_caxis_bound)/2, u_caxis_bound])
    plt.xlabel('Final State')
    plt.ylabel('Initial State')
    plt.title('PL > LSD')
    plt.tick_params(axis='both', which='both', length=0, labelsize=10)

    plt.show()


plot_data(firstplot)
