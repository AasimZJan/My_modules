import numpy as np
from scipy import interpolate
from binary_code import binary
import sys

#Note: when you are calculating arrays, I have selected each bin to be [f_min(b), f_max(b)), except the last bin which is  [f_min(b), f_max(b)],
def resample_psd(psd, df=None):  
    """Takes in a PSD file which should have two columns, frequnecy and data, and uses linear interpolation to return it with a new  deltaF"""
    frequency, data = np.loadtxt(psd, delimiter=" ", comments="#",unpack=True)
    f0, deltaF, f_final = frequency[0], frequency[1]-frequency[0], frequency[-1]
    interp = interpolate.interp1d(frequency, data, fill_value = 'extrapolate')
    new_frequency = np.arange(f0, f_final+5*df, df or deltaF)
    return new_frequency, interp(new_frequency)

def crop_waveform(hf, flow, fhigh):
    """Takes in a FD waveform and crops it such that only h(f=flow)->h(f=fhigh) is returned."""
    f_data = np.arange(hf.f0, hf.data.length* hf.deltaF, hf.deltaF)
    i_min=int((flow-f_data[0])/hf.deltaF)  
    i_max=int((fhigh-f_data[0])/hf.deltaF) 

    return hf.data.data[i_min:i_max]

def calculate_dpsi(f, gamma, f_star, tune):
    """Equation 9 of relative binning paper (http://arxiv.org/abs/1806.08792) to measure maximum differential phase change."""
    dpsi = 0
    for i in range(len(gamma)):
        dpsi += (f / f_star[i])**(gamma[i]) * gamma[i]/np.abs(gamma[i])
    return 2 * np.pi * tune * dpsi

#############################
def calculate_bins(freq_array, max_error, tune, gamma, f_star): #freq_array should go from flow -> fhigh with deltaF as increment.
    dpsi_all = calculate_dpsi(freq_array, gamma = gamma, f_star = f_star, tune = tune)

    dpsi_min = dpsi_all[0]
    bins = [freq_array[0]]
    index = [0]
    error = []
    for i in range(1, len(dpsi_all)):
        diff  = np.abs(dpsi_all[i] - dpsi_min)
        if diff >= max_error:
            bins.append(freq_array[i])
            error.append(diff)
            index.append(i)
            dpsi_min = dpsi_all[i]
    if bins[-1] != freq_array[-1]:
        bins.append(freq_array[-1])
        error.append(diff)
        index.append(i)
    assert len(bins)-1 == len(index)-1 == len(error), f"{len(bins),len(index),len(error)}"

    return np.array(bins), np.array(error), np.array(index)

def factors(hf, hf0, bins, index, flow, fhigh):
    fm =  bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
    hf_array = crop_waveform(hf, flow, fhigh)[index]
    hf0_array = crop_waveform(hf0, flow, fhigh)[index]

    ratio_array = hf_array / (hf0_array)

    ratio_max = ratio_array[1:]
    ratio_min = ratio_array[:-1]

    bins_max = bins[1:]
    bins_min = bins[:-1]

    r1 = (ratio_max - ratio_min) / (bins_max - bins_min)

    r0 = ratio_min - r1 * (bins_min - fm) 

    return r0, r1
    
def summary_data(hf0, signalf, bins, index, psd, flow = 20, fhigh = 2046):
    #creating psd array
    assert hf0.data.length ==  signalf.data.length and hf0.deltaF == signalf.deltaF
    #psd = "/Users/aasim/Desktop/Research/Codes/My_modules/PSD/LIGO_H1.txt"
    frequency, data = resample_psd(psd, df=hf0.deltaF)
    print(f"\t Resampling psd to {hf0.deltaF} Hz.")
    i_min=int((flow-frequency[0])/hf0.deltaF)  
    i_max=int((fhigh-frequency[0])/hf0.deltaF)
    #print(i_min, i_max)
    try:
        tmp=1/data[i_min:i_max]
    except Exception as e:
        print(f"Cannot proceed (potentially due to broadcasting error). It might be due to fhigh being greater than Nyquist frequency {1/hf0.deltaT/2} Hz.")
        print(e)
        sys.exit()
    psd_new = tmp   #this just goes from flow ->fhigh with deltaF as increment

    #populating a f_mean array with the same length as freq_array at maximal resolution. fm is the mean of bins
    freq_array = np.arange(flow, fhigh, hf0.deltaF)
    fm_array = np.zeros(len(freq_array))
    fm = bins[:-1] + 0.5* (bins[1:] - bins[:-1])
    for i in range(len(index)-1): 
        fm_array[index[i]:index[i+1]] = fm[i]  #index[i+1] doesn't get filled
    fm_array[-1] = fm_array[-2]
    assert len(freq_array) == len(fm_array)
    
    #getting signal and waveform values corresponding to flow ->fhigh with deltaF as increment
    hf0_array = crop_waveform(hf0, flow, fhigh)
    signalf_array = crop_waveform(signalf, flow, fhigh)

    #summary data
    assert len(hf0_array) == len(signalf_array) ==len(psd_new) == len(freq_array) == len(fm_array),f"{len(hf0_array), len(signalf_array), len(psd_new) , len(freq_array) , len(fm_array)}"


    A0, A1, B0, B1 = np.zeros(len(index)-1, dtype = complex),np.zeros(len(index)-1, dtype = complex),np.zeros(len(index)-1, dtype = complex),np.zeros(len(index)-1, dtype = complex)
    for i in range(len(index)-1):
        if i == len(index)-1: #otherwise the last entry doesn't get filled
            A0[i] = (np.sum(np.conj(hf0_array[index[i]:]) * signalf_array[index[i]:] * psd_new[index[i]:]))
            A1[i] = (np.sum(np.conj(hf0_array[index[i]:]) * signalf_array[index[i]:] * psd_new[index[i]:] * (freq_array[index[i]:]-fm_array[index[i]:])))

            B0[i] = (np.sum(np.conj(hf0_array[index[i]:]) * hf0_array[index[i]:] * psd_new[index[i]:] ))
            B1[i] = (np.sum(np.conj(hf0_array[index[i]:]) * hf0_array[index[i]:] * psd_new[index[i]:] *  (freq_array[index[i]:]-fm_array[index[i]:])))
        else:
            A0[i] = (np.sum(np.conj(hf0_array[index[i]:index[i+1]]) * signalf_array[index[i]:index[i+1]] * psd_new[index[i]:index[i+1]]))
            A1[i] = (np.sum(np.conj(hf0_array[index[i]:index[i+1]]) * signalf_array[index[i]:index[i+1]] * psd_new[index[i]:index[i+1]] * (freq_array[index[i]:index[i+1]]-fm_array[index[i]:index[i+1]])))

            B0[i] = (np.sum(np.conj(hf0_array[index[i]:index[i+1]]) * hf0_array[index[i]:index[i+1]] * psd_new[index[i]:index[i+1]] ))
            B1[i] = (np.sum(np.conj(hf0_array[index[i]:index[i+1]]) * hf0_array[index[i]:index[i+1]] * psd_new [index[i]:index[i+1]]*  (freq_array[index[i]:index[i+1]]-fm_array[index[i]:index[i+1]])))


    return 4 * hf0.deltaF* A0, 4 * hf0.deltaF* A1, 4 * hf0.deltaF * B0, 4 * hf0.deltaF*B1



def normal_lnL(hf, signalf, flow, fhigh, psd):
    frequency, data = psd[0], psd[1]
    assert hf.data.length ==  signalf.data.length and hf.deltaF == signalf.deltaF == np.diff(frequency)[0]
    i_min=int((flow-frequency[0])/hf.deltaF)  
    i_max=int((fhigh-frequency[0])/hf.deltaF)

    tmp = np.zeros(hf.data.length)
    tmp[i_min:i_max]=1/data[i_min:i_max]
    psd_new = tmp  
    
    hf0_array = hf.data.data
    signalf_array = signalf.data.data
    d_h = 4 * hf.deltaF* (np.sum(np.conj(hf0_array) * signalf_array * psd_new))
    h_h = 4 * hf.deltaF* (np.sum(np.conj(hf0_array) * hf0_array * psd_new))
    print(d_h, h_h)
    return   np.real(d_h - 0.5 * h_h)

def normal_lnL_double_sided_psd(hf, signalf, flow, fhigh, psd):
    frequency, data = psd[0], psd[1]
    assert hf.data.length ==  signalf.data.length and hf.deltaF == signalf.deltaF == np.diff(frequency)[0]
    i_min=int((flow-frequency[0])/hf.deltaF)  
    i_max=int((fhigh-frequency[0])/hf.deltaF)

    tmp = np.zeros(hf.data.length//2 +1)
    tmp[i_min:i_max]=1/data[i_min:i_max]

    psd_new=np.zeros(hf.data.length)
    psd_new[:len(tmp)]=tmp[::-1]    #[-N2--->0]
    psd_new[len(tmp):]=tmp[:-1] 

    hf0_array = hf.data.data
    signalf_array = signalf.data.data
    d_h = 4 * hf.deltaF* (np.sum(np.conj(hf0_array) * signalf_array * psd_new))
    h_h = 4 * hf.deltaF* (np.sum(np.conj(hf0_array) * hf0_array * psd_new))
    print(d_h, h_h)
    return   np.real(d_h - 0.5 * h_h)

def inner_product(hf, signalf, flow, fhigh, psd):
    frequency, data = psd[0], psd[1]
    assert hf.data.length ==  signalf.data.length and hf.deltaF == signalf.deltaF == np.diff(frequency)[0]
    i_min=int((flow-frequency[0])/hf.deltaF)  
    i_max=int((fhigh-frequency[0])/hf.deltaF)

    tmp = np.zeros(hf.data.length//2 +1)
    tmp[i_min:i_max]=1/data[i_min:i_max]

    psd_new=np.zeros(hf.data.length)
    psd_new[:len(tmp)]=tmp[::-1]    #[-N2--->0]
    psd_new[len(tmp):]=tmp[:-1] 

    hf0_array = hf.data.data
    signalf_array = signalf.data.data
    d_h = 4 * hf.deltaF* (np.sum(np.conj(hf0_array) * signalf_array * psd_new))
    return d_h


def get_mass_from_mc_eta(mc, eta):
    """Returns m1, m2 from mc and eta."""
    alpha = mc / eta**(3/5)
    beta = mc**2 / eta**(1/5)
    m1 = 0.5 * (alpha + np.sqrt(alpha**2 - 4*beta))
    m2 = 0.5 * (alpha - np.sqrt(alpha**2 - 4*beta)) 
    return m1, m2

def align_waveform(ht, buffer = 2):
    """This function ensures the injection and recovery waveforms are all aligned the same way. This needs to be checked, RIFT uses numpy.roll."""
    TDlen = int(ht.data.length)
    max_index = np.argmax(np.abs(ht.data.data))
    buffer_start = int(TDlen - buffer/ht.deltaT)
#   diff = TDlen//2 - max_index
    diff = buffer_start - max_index
    inv_diff = TDlen - diff
    tmp = np.zeros(TDlen)
    tmp[diff:] = ht.data.data[:inv_diff]
    ht.data.data = tmp
    return ht


def prior_transform(u, prior_array):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest. Note: Needed for dynesty sampler."""
    x = np.array(u)
    x[0] = (prior_array[0][1]-prior_array[0][0]) * u[0] + prior_array[0][0]
    x[1] = (prior_array[1][1]-prior_array[1][0]) * u[1] + prior_array[1][0]
    return x

def heterodyned_lnL(hf, hf0, bins, index, flow, fhigh, summary_data, phase_max = False):
    #hf is the recovery waveform, hf0 is the reference waveform
    r0, r1 = factors(hf = hf, hf0 = hf0, index = index, flow = flow, fhigh = fhigh, bins = bins)

    d_h = np.sum(summary_data[0] * np.conj(r0) + summary_data[1] * np.conj(r1))

    h_h = np.sum(summary_data[2] * np.conj(r0) * r0 + 2* summary_data[3] * np.real(r0*np.conj(r1)))

    if phase_max == True:
        return np.real(np.abs(d_h) - 0.5 * h_h)
    else:
        return np.real(d_h - 0.5*h_h)
    

def rejection_sampling(prior_range, log_likelihood_function, no_of_samples, max_log_likelihood, vectorized=True):
    #prior_range shape should be (nparams, 2)
    assert prior_range.shape[1] == 2, "prior_range shape should be (nparams, 2)"
    nparams = prior_range.shape[0]
    print(f"Initiating rejection sampling with number of parameters = {nparams}")
    print(f"Prior ranges are:")
    print(*prior_range, sep = "\n")

    samples = np.zeros((no_of_samples, nparams))
    lnL = np.zeros((no_of_samples,1))

    trial_params = np.random.uniform(prior_range[:,0], prior_range[:,1], (no_of_samples*10,nparams))
    lnL_array = np.random.uniform(0, max_log_likelihood, no_of_samples*10)

    samples_no = 0
    iteration = 0
    while samples_no<no_of_samples:
        print(f"Iteration = {iteration}, samples = {samples_no}")
        
        if vectorized:
            chi_tmp = log_likelihood_function(trial_params)
            if np.exp(chi_tmp)>= np.exp(lnL_array) and samples_no < no_of_samples:
                samples[samples_no] = [phi_array[i], omega_array[i]]
                lnL[samples_no] = chi_tmp
                samples_no += 1

        else:
            chi_tmp = chi_squared(signal, time_series_model, std)
            if np.exp(chi_tmp) >= L_array[i] and samples_no < no_of_samples:
                samples[samples_no] = [phi_array[i], omega_array[i]]
                lnL[samples_no] = chi_tmp
                samples_no += 1
        phi_array = np.random.uniform(prior_range[0][0], prior_range[0][1],no_of_samples*10)
        omega_array = np.random.uniform(prior_range[1][0], prior_range[1][1],no_of_samples*10)
        L_array = np.random.uniform(0,1,no_of_samples*10)
        iteration +=1
    return samples, lnL