'''
Removal of power line interferences

(Planned to be)
Submodule of the Modular EEg Toolkit - MEET for Python.

In order to achieve sufficient speed for larger data arrays,
the numba extension for python should be installed.
This enables run-time compilation of the core function into machine code
and leads to a speed up of about 500 times (and about 10 times as compared
to the matlab implementation). Hence, having installed numba on the system
is greatly encouraged.

The only 'non-private' function of this (sub-)module is removePLI,
see there for a detailed documentation. All other functions should not be
used externally, because no type-checking is done.

A simple test case is demonstrated if this file is run directly by a
python interpreter.

Modifications to the original matlab software (see below) by:
-------------------------------------------------------------
Gunnar Waterstraat
gunnar[dot]waterstraat[at]charite.de

This implements the method described in:

    M. R. Keshtkaran and Z. Yang, "A fast, robust algorithm for power
    line interference cancellation in neural recording,"
    J. Neural Eng., vol. 11, no. 2, p. 026017, Apr. 2014
    
If using this code, please always cite the reference above
    
The code is more or less a port from matlab to python of this package:
https://github.com/mrezak/removePLI
-------------------------------------------------------------
Copyright (c) 2013, Mohammad Reza Keshtkaran keshtkaran.github@gmail.com

All rights reserved.

"This program" refers to the m-files in the whole package. This program is
provided "AS IS" for non-commercial, educational and reseach purpose only.
Any commercial use, of any kind, of this program is prohibited.
The Copyright notice should remain intact.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see http://www.gnu.org/licenses/.
'''


import numpy as _np
import meet
import scipy as _sc
from numba import njit

def make_example():
    '''
    Create an example with 10 channels.

    Input:
    ------
    None

    Output:
    -------
    example - 2d numpy array
                - dim0 channels
                - dim1 datapoints
    fs - sampling rate in Hz
    '''
    fs = 500.
    n = 120*fs
    m = 10
    t = 2*_np.pi * _np.arange(n)/fs
    fline = 60 + _np.random.standard_normal(1) # random interference
    s = _sc.signal.lfilter((1,),(1,-0.99),
            _np.random.standard_normal((m,n)),
            axis=-1)
    p = (
        _np.sin(1*fline*t + _np.random.standard_normal(1)) * # base frequency
            (80 + 10*_np.random.standard_normal((m,1)))
        +
        _np.sin(2*fline*t + _np.random.standard_normal(1)) * # 1st harmonic
            (50 + 10*_np.random.standard_normal((m,1)))
        +
        _np.sin(3*fline*t + _np.random.standard_normal(1)) * # 2nd harmonic
            (20 + 10*_np.random.standard_normal((m,1)))
        )
    example = p+s
    return example, fs

def make_example2():
    '''
    Create an example with 1 channel.

    Input:
    ------
    None

    Output:
    -------
    example - 1d numpy array
    fs - sampling rate in Hz
    '''
    fs = 500
    n = 120*fs
    m = 1
    t = 2*_np.pi * _np.arange(n)/fs
    fline = 60 + _np.random.standard_normal(1) # random interference
    s = _sc.signal.lfilter((1,),(1,-0.99),
            100*_np.random.standard_normal((m,n)), axis=-1)
    p = (
        80*_np.sin(1*fline*t + _np.random.standard_normal(1)) # base frequency
        +
        50*_np.sin(2*fline*t + _np.random.standard_normal(1)) # 1st harmonic
        +
        20*_np.sin(3*fline*t + _np.random.standard_normal(1)) # 2nd harmonic
        )
    example = p+s
    return _np.ravel(example), fs

#initial band pass
def _AC_accent(data, ac, s_rate, axis=-1):
    '''
    Accentuate the DC frequency by filtering +- 5 Hz around it
    and calculating the first order difference

    Input:
    ------
    data - 1d or 2d numpy array
    ac - the ac-frequency in Hz
    s_rate - the sampling frequency in Hz
    axis - the axis along which the filter works

    Output:
    -------
    the filtered data array
    '''
    data_ac = meet.iir.butterworth(
            data = data,
            axis=axis,
            s_rate=s_rate,
            fp = [ac-5, ac+5],
            fs = [ac-10, ac+10])
    zeros_to_add_shape = list(data.shape)
    zeros_to_add_shape[axis]=1
    return _np.concatenate([
        _np.zeros(zeros_to_add_shape, data.dtype),
        _np.diff(data_ac,axis=axis)], axis=axis)

def remove_PLI(data, ac, s_rate, N_harmonics, B, P, W, axis=-1):
    '''
    Filter power line interferences from the data.

    Input:
    ------
    data - 1d or 2d numpy array - the data
    ac - number - the ac frequency in Hz (typically 50/60 Hz)
    s_rate - number - the sampling frequency in Hz
    N_harmonics - int - the number of harmonics to be removed
    B - an iterable with 3 elements - the parameters controlling the notch
                                      filter bandwidth; all entrys must
                                      be > 0
        B[0] - the initial notch filter bandwidth in Hz
               (recommended range: 10-50 Hz)
        B[1] - the asymptotic notch filter bandwidth in Hz
               (recommended range: 0.01 - 0.1 Hz)
        B[2] - the settling time for the adaption from B[0] to B[1] in s
               (recommeded range: 0.5 - 10)
    P - an iterable with 3 elements - the parameters controlling the frequency
                                      estimator; all entrys must be > 0
        P[0] - the initial settling time of the frequency estimator in s
               (recommended range:  0.01 - 0.5 s)
        P[1] - the asymptotic settling time of the frequency estimator in s
               (recommended range:  1 - 5 s)
        P[2] - the settling time for the adaption from P[0] to P[1] in s
               (recommeded range: 1 - 10 s)
    W - number>0 - the settling time of the amplitude/phase estimator in s
                   (recommended range: 0.5 - 5 s)
    axis - int - the axis along which the algorithm should be performed
                 defaults to last axis
    Output:
    -------
    output - a numpy floating point array of same shape as data
             containing the 'cleaned' dataset
    
    Reference:
    ----------
    M. R. Keshtkaran and Z. Yang, "A fast, robust algorithm for power
    line interference cancellation in neural recording,"
    J. Neural Eng., vol. 11, no. 2, p. 026017, Apr. 2014
    
    If using this code, please always cite the reference above
    
    The code is more or less a port from matlab to python from:
    https://github.com/mrezak/removePLI

    Obey the copyright notice a the top of this file.

    Notes:
    ------
    ###################################################
    Explain meaning of parameters
    ###################################################
    '''
    ################### Checks #######################
    # checking data
    if not isinstance(data, _np.ndarray):
        raise TypeError('data must be numpy array')
    if data.ndim > 3:
        raise ValueError('dimensionality of data must by 1 or 2')
    elif data.ndim == 0:
        raise ValueError('dimensionality of data must by 1 or 2')
    if not (_np.issubdtype(data.dtype, _np.integer) or 
            _np.issubdtype(data.dtype, _np.floating)):
        raise TypeError('data type must be integer or float')
    # force to be floating point array
    if _np.issubdtype(data.dtype, _np.integer):
        data = data.astype(_np.float64)
    # checking ac
    try:
        ac = float(ac)
    except:
        raise TypeError( \
                'ac must be a single number convertible into a float')
    if not ac>0:
        raise ValueError('ac frequency must be > 0')
    # checking s_rate
    try:
        s_rate = float(s_rate)
    except:
        raise TypeError( \
                's_rate must be a single number convertible into a float')
    if not s_rate > 0:
        raise ValueError('sampling rate must be > 0')
    # checking N_harmonics
    try:
        if not int(N_harmonics) == N_harmonics:
            raise ValueError
        N_harmonics = int(N_harmonics)
    except:
        raise TypeError('N_harmonics must be an integer valued number')
    if not N_harmonics > 0:
        raise ValueError('number of harmonics to be removed must be > 0')
    # checking B
    try:
        B = _np.array(B)
    except:
        raise TypeError( \
                'B must be an iterable and must be convertible' +
                ' into a numpy array')
    if not (_np.issubdtype(B.dtype, _np.integer) or 
            _np.issubdtype(B.dtype, _np.floating)):
        raise TypeError( \
                'All entries in B must be integers or floats')
    if not B.ndim == 1:
        raise ValueError('B must be 1-dimensional')
    if not len(B) == 3:
        raise ValueError('length of B must be exactly 3')
    if _np.any(B <= 0):
        raise ValueError('all entries in B must be > 0')
    if _np.issubdtype(B.dtype, _np.integer):
        B = B.astype(_np.float64)
    # checking P
    try:
        P = _np.array(P)
    except:
        raise TypeError( \
                'B must be an iterable and must be convertible' +
                ' into a numpy array')
    if not (_np.issubdtype(P.dtype, _np.integer) or 
            _np.issubdtype(P.dtype, _np.floating)):
        raise TypeError( \
                'All entries in P must be integers or floats')
    if not P.ndim == 1:
        raise ValueError('P must be 1-dimensional')
    if not len(P) == 3:
        raise ValueError('length of P must be exactly 3')
    if _np.any(P <= 0):
        raise ValueError('all entries in P must be > 0')
    if _np.issubdtype(P.dtype, _np.integer):
        P = P.astype(_np.float64)
    # checking W
    try:
        W = float(W)
    except:
        raise TypeError( \
                'W must be a single number convertible into a float')
    if not W > 0:
        raise ValueError('W must be > 0')
    # checking axis
    if not isinstance(axis, int):
        raise TypeError('axis must by of integer type')
    try:
        data.shape[axis]
    except:
        raise ValueError('axis argument is not a valid axis of data')
    #################################################
    data_shape = data.shape
    data = _np.atleast_2d(data)
    data = data.swapaxes(axis,-1)
    n_ch, n_dp = data.shape
    #remove mean
    data = (data.T - data.mean(1)).T
    # convert parameters
    lambda_f, lambda_inf, lambda_st = _np.exp(_np.log(0.05) / (P*s_rate + 1))
    lambda_a = _np.exp(_np.log(0.05) / (W*s_rate + 1))
    alpha_f = (
             (1 - _np.arctan(_np.pi*B[0]/s_rate))
            /(1 + _np.arctan(_np.pi*B[0]/s_rate))
        )
    alpha_inf = (
             (1 - _np.tan(_np.pi*B[1]/s_rate))
            /(1 + _np.tan(_np.pi*B[1]/s_rate))
        )
    alpha_st = _np.exp(_np.log(0.05) / (B[2]*s_rate + 1))
    # define initial conditions
    gamma = (( 1 - _np.tan(0.5*_np.pi*_np.min([90,s_rate/2.])/float(s_rate)))
            / (1 + _np.tan(0.5*_np.pi*_np.min([90,s_rate/2.])/float(s_rate))))
    C = 5.
    D = 10.
    f = _np.zeros(3, _np.float64)
    k_f = 0.
    ###
    u_h =  _np.ones(N_harmonics, _np.float64)
    u_hd = _np.ones(N_harmonics, _np.float64)
    r1_h = _np.ones(N_harmonics, _np.float64) * 100
    r4_h = _np.ones(N_harmonics, _np.float64) * 100
    a_h = _np.zeros(N_harmonics, _np.float64)
    b_h = _np.zeros(N_harmonics, _np.float64)
    k_h = _np.ones(N_harmonics+2, _np.float64)
    ###
    #initial band-pass
    data_ac = _AC_accent(data, ac, s_rate, axis=-1)
    output = _np.empty_like(data)
    # if numba is present, use the optimized machine-code version,
    # however, if it breaks use the original slow loop version
    for ch in range(n_ch):
        _remove_PLI(data[ch], data_ac[ch], output[ch], N_harmonics, gamma,
        alpha_f, alpha_inf, alpha_st,
        lambda_a, lambda_f, lambda_inf, lambda_st,
        a_h, b_h, k_h, r1_h, r4_h, u_h, u_hd,
        C, D, f, k_f)

    return output.swapaxes(-1,axis).reshape(data_shape)

'''
Try to compile _remove_PLI to machine code using the numba extension package.
If this fails, a slower python loop is used instead. The difference in execution
time is about factor 500, so numba is recommended if execution time matters.
'''
@njit
def _remove_PLI(data, data_ac, output, N_harmonics, gamma,
        alpha_f, alpha_inf, alpha_st,
        lambda_a, lambda_f, lambda_inf, lambda_st,
        a_h, b_h, k_h, r1_h, r4_h, u_h, u_hd,
        C, D, f, k_f):
    '''
    This is a helper function for remove_PLI. The magic
    is happening here. Partly by construction of the algorithm,
    partly intenionally no numpy or scipy commands are used within
    the nested loop structure.

    This enables translation of the code into fast machine code
    by the numba extension module (if present on the system).

    The code is more or less a direct port from matlab to python from:
    https://github.com/mrezak/removePLI

    If using this code, please always cite:
    M. R. Keshtkaran and Z. Yang, "A fast, robust algorithm for power
    line interference cancellation in neural recording,"
    J. Neural Eng., vol. 11, no. 2, p. 026017, Apr. 2014

    Obey the copyright notice at the top of the file.
    '''
    # initialize output
    N = len(data)
    for n in range(N):
        # Lattice Filter
        f[-1]= data_ac[n] + k_f*(1 + alpha_f)*f[-2] - alpha_f*f[-3]
        # Frequency Estimation
        C = lambda_f*C + (1-lambda_f)*f[-2]*(f[-1] + f[-3])
        D = lambda_f*D + (1-lambda_f)*2*f[-2]**2
        k_t = C/D
        # avoid numpy function here to be able to compile to fast machine code
        if k_t > 1: k_t = 1
        if k_t < -1: k_t = -1
        k_f = gamma*k_f + (1-gamma)*k_t
        # update lattice status
        f[-3] = f[-2]; f[-2] = f[-1]
        # Bandwidth and Forgetting Factor updates
        alpha_f = alpha_st*alpha_f + (1 - alpha_st)*alpha_inf
        lambda_f = lambda_st * lambda_f + (1-lambda_st)*lambda_inf
        ###
        #Remove Harmonics
        ###
        k_h[0] = k_f
        e = data[n]
        for h in range(N_harmonics):
            # k_h is starting with index -1
            # So, it must be indexed with (h+1) to get the current item,
            # while index h is the previous item
            # Harmonic frequency Calculation
            k_h[h+2] = 2*k_f*k_h[h+1] - k_h[h]
            # Discrete Oscillator
            s1 = k_h[h+2]*(u_h[h] + u_hd[h])
            s2 = u_h[h]
            u_h[h] = s1 - u_hd[h]
            u_hd[h] = s1 + s2
            # Gain Control
            G = 1.5 - (u_h[h]**2 - u_hd[h]**2*(k_h[h+2] - 1)/(k_h[h+2]+1))
            if G<0: G=1.
            u_h[h] = G*u_h[h]
            u_hd[h] = G*u_hd[h]
            # Amplitude/Phase Estimation
            sincmp = a_h[h]*u_hd[h] + b_h[h]*u_h[h]
            e = e - sincmp
            # Simplified RLS
            r1_h[h] = lambda_a*r1_h[h] + u_hd[h]**2
            r4_h[h] = lambda_a*r4_h[h] + u_h[h]**2
            a_h[h] = a_h[h] + e*u_hd[h]/r1_h[h]
            b_h[h] = b_h[h] + e*u_h[h]/r4_h[h]
        output[n] = e
    return output


if __name__ == '__main__':
    example, s_rate = make_example2()
    print(example.shape)
    s_rate = 500
    B = _np.array([100,0.01,4])
    P = _np.array([0.1,2.,5.])
    W = 3.
    result = remove_PLI(example, ac=60, s_rate=s_rate, N_harmonics=3, B=B, P=P, W=W)
    import matplotlib.pyplot as plt
    plt.psd(example[s_rate:], NFFT=1024, Fs=s_rate, label='original data')
    plt.psd(result[s_rate:], NFFT=1024, Fs=s_rate, label='cleaned data')
    plt.legend()
    plt.show()
