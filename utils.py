import numpy as np
import celerite
from celerite import terms
from scipy.optimize import minimize
import scipy.special as sps 


def mag2fluxdensity(mag, magerr, mode = 'muJy'):
    """convert magnitude to flux density"""
    if mode == 'muJy':
        flux = 10**(-mag/2.5)*3631*1e6
        fluxerr = flux*np.log(10)*magerr/2.5
        return flux,fluxerr
    elif mode == 'cgs':
        #both are correct
        #return 10**(-mag/2.5)*3631*1e-23
        flux = 10**(-(mag+48.6)/2.5)
        fluxerr = flux*np.log(10)*magerr/2.5
        return flux, fluxerr

def fluxdensity2mag(flux, fluxerr, mode = 'muJy'):
    """convert flux density to magnitude"""
    if mode == 'muJy':
        mag = -2.5*np.log10(flux/3631/1e6)
        magerr = 2.5*fluxerr/np.log(10)/flux
        return mag, magerr
    elif mode == 'cgs':
        #both are correct
        #return -2.5*np.log10(flux/3631*1e23)
        mag = -2.5*np.log10(flux)-48.6
        magerr = 2.5*fluxerr/np.log(10)/flux
        return mag, magerr
    

def bin_data(phase,data,dataErr,interval,mode = 'mag',phase_start = None,phase_end = None):
    '''
    Bin the data into phase bins with a given interval.
    Parameters:
        phase: array-like
            The phase of the data.
        data: array-like
            The data to be binned.
        dataErr: array-like
            The error of the data.
        interval: float
            The interval of the phase bins.
        mode: string
            The mode of the data. 'mag' or 'flux'.
        phase_start: float
            The start of the phase to be binned.
        phase_end: float
            The end of the phase to be binned.
    Returns:
        phase_bin: array-like
            The phase of the binned data.
        data_bin: array-like
            The binned data.
        dataErr_bin: array-like
            The error of the binned data.
    '''
    if phase_start is None:
        phase_start = int(min(phase))
    if phase_end is None:
        phase_end = max(phase)

    mask = np.logical_and(phase>=phase_start,phase<=phase_end)
    phase = phase[mask]
    data = data[mask]
    dataErr = dataErr[mask]
    

    if mode == 'mag':
        flux, fluxErr = mag2fluxdensity(data, dataErr)
    elif mode == 'flux':
        flux = data
        fluxErr = dataErr
    weights = 1/fluxErr**2
    phase_bin = []
    flux_bin = []
    fluxErr_bin = []
    phase_bin_start = phase_start
    while phase_bin_start < phase_end:
        phase_bin_end = phase_bin_start + interval
        data_bin_mask = np.logical_and(phase > phase_bin_start, phase < phase_bin_end)
        phase_bin_start = phase_bin_end
        if sum(data_bin_mask) > 0:
            # flux_bin.append(np.mean(flux[data_bin_mask]))
            # fluxErr_bin.append(np.sqrt(np.sum(fluxErr[data_bin_mask]**2))/sum(data_bin_mask))
            # phase_bin.append(np.mean(phase[data_bin_mask]))
            
            # 以下是加权平均的计算
            flux_bin.append(np.average(flux[data_bin_mask],weights = weights[data_bin_mask]))
            fluxErr_bin.append(1/np.sqrt(np.sum(weights[data_bin_mask])))
            phase_bin.append(np.average(phase[data_bin_mask],weights = weights[data_bin_mask]))

    if mode == 'mag':
        data_bin, dataErr_bin = fluxdensity2mag(np.array(flux_bin),np.array(fluxErr_bin))
    elif mode == 'flux':
        data_bin = np.array(flux_bin)
        dataErr_bin = np.array(fluxErr_bin)

    return np.array(phase_bin),np.array(data_bin),np.array(dataErr_bin)

def remove_outliers(lc,level = 3):
    '''
    Remove the outliers in the light curve.
    Parameters:
        lc: array-like
            The light curve.
        level: float
            The sigma level of the outliers.
    Returns:
        outliers: array-like
            The index of the outliers.
    '''

    sigma = np.std(lc)
    mean = np.median(lc)
    # mean = np.mean(lc)
    outliers = []
    
    outliers_idx = np.where(lc<mean-sigma*level)[0]
    outliers_idx2 = np.where(lc<mean-sigma*(level-0.5))[0]
    for i in outliers_idx:
        if i-1 in outliers_idx or i+1 in outliers_idx:
            continue
        if i-1 in outliers_idx2 and i+1 in outliers_idx2:
            continue
        outliers.append(i)

    outliers_idx = np.where(lc>mean+sigma*level)[0]
    outliers_idx2 = np.where(lc>mean+sigma*(level-0.5))[0]
    for i in outliers_idx:
        if i-1 in outliers_idx or i+1 in outliers_idx:
            continue
        if i-1 in outliers_idx2 and i+1 in outliers_idx2:
            continue
        outliers.append(i)
    return outliers


def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def GP_fit(t, flux, flux_err):
    """
    Fit a Gaussian Process model to the light curve.
    Parameters:
        t: array-like
            The time of the data.
        flux: array-like
            The flux of the data.
        flux_err: array-like
            The error of the flux.
        kernel: celerite kernel
            The kernel of the GP model.
    Returns:
        gp: celerite GP
            The GP model.
    """
    # 计算初始参数
    # bounds = dict(log_sigma = (-10, 10), log_rho = (-10, 10))
    # kernel = terms.Matern32Term(log_sigma=-5, log_rho=5,eps=1e-12, bounds=bounds)
    kernel = terms.Matern32Term(log_sigma=5., log_rho=3.)   # using Matern32 kernel
    gp = celerite.GP(kernel, mean=np.mean(flux))
    gp.compute(t, flux_err)
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    r = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds, args=(flux, gp))
    gp.set_parameter_vector(r.x)
    params_vector = gp.get_parameter_vector()
    return gp, params_vector

def get_x_normed(data):
    """data generator with normalization for x """
    for k in range(data.shape[0]):
        x = data[k, :, 1:]
        x = x[~np.isnan(x).any(axis=1)]
        mean = np.nanmean(x, axis=0)[0]    
        std = np.nanstd(x, axis=0)[0]
        x[:,0] = (x[:,0] - mean) / std  # normalization for flux
        x[:,1] = x[:,1]/std  # normalization for flux_err
        x = np.expand_dims(x, axis=0)
        yield x

def get_x_y(data, labels, mode='train'):
    """data generator with normalization for x and y"""
    for k in range(int(1e9)):
        max_len = data.shape[0]
        x = data[k % max_len, :, 1:]
        x = x[~np.isnan(x).any(axis=1)]
        mean = np.nanmean(x, axis=0)[0]    
        std = np.nanstd(x, axis=0)[0]
        x[:,0] = (x[:,0] - mean) / std  # normalization for flux
        x[:,1] = x[:,1]/std  # normalization for flux_err
        x = np.expand_dims(x, axis=0)
        y = labels[k % max_len]
        if k % 10000 == 0 and mode == 'train':
            print('Proceed {:d}e4 samples'.format(k // 10000))
        yield x, np.expand_dims(y, axis=-1)


def generate_flare(x, x_start, peakmag, T_dur, type="Gamma", shape=2):
    """ Generate a flare with given parameters in magnitude scale.
    Parameters:
        x: array-like
            The time of the data.
        x_start: float
            The start time of the flare.
        peakmag: float
            The peak magnitude of the flare.
        T_dur: float
            The duration of the flare.
        type: string
            The type of the flare.
        shape: int
            The shape of the flare.
    Returns:
        y: array-like
            The magnitude of the flare.
    Note:
        original magitude - y is the mag with the flare.
    """

    if type=="Gamma":
        # sigma = math.sqrt(shape*scale**2)
        shape_FWHM = {2:2.44,3:3.4,4:4.13}
        scale = T_dur/shape_FWHM[shape]
        x = x-x_start
        y = np.heaviside(x,1)*x**(shape-1)*(np.exp(-x/scale)/(sps.gamma(shape)*scale**shape))
        y = y/np.max(y,axis=0)*peakmag
    elif type=="Gaussian":
        sigma = T_dur/2.355 #T_dur is the FWHM of the Gaussian distribution
        mu = x_start
        y = peakmag*np.exp(-0.5*((x-mu)/sigma)**2)
    return y
