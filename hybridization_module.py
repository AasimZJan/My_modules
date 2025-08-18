import numpy as np
import lal
from binary_code import binary
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

__author__ = "A. Jan"

def create_lal_COMPLEX16TimeSeries(deltaT, time_series,  epoch = 950000000, f0 = 0.0):
    """A helper function to create lal COMPLEX16TimeSeries.
        Args:
            deltaT (float): time step (1/sampling rate).
            time_series (numpy.array): strain in time domain.
            epoch (float): Needed to create COMPLEX16TimeSeries, by default it is 950000000.
            f0 (float): Needed to create COMPLEX16TimeSeries, by default it is 0.0 . 
        Returns:
            lal.COMPLEX16TimeSeries object"""
    ht_lal = lal.CreateCOMPLEX16TimeSeries("ht_lal", epoch, f0, deltaT, lal.DimensionlessUnit, len(time_series))
    ht_lal.data.data = time_series
    return ht_lal

def pow2(length):
    """Looks for the nearst power of 2 for a given number. Useful for FFT as the number of data points need to be a power of 2
        Args: 
            length
        Returns: 
            Nearest power of two (int)"""
    return int(2**(int(np.ceil(np.log2(length)))))

def get_closest_index(array, value):
    """A function that gives you the index at which the array has a value closest to the one you want.
        Args:
            array (np.array):
            value (float):
        Returns:
            index (float)"""
    return np.argmin(np.abs(array - value))

def get_amplitude_phase(ht):
    """This function splits h(t) into Amplitude A(t) and phase phase(t). We then express h_lm as A_lm(t) * exp(i*phase(t)). Note the sign in the exponent.
       Args:
           ht (CreateCOMPLEX16TimeSeries): Time domain waveform.
       Returns:
           Amplitude as a function of time (numpy.array), phase as a function of time (numpy.array) 
       """
    return np.abs(ht.data.data), np.unwrap(np.angle(ht.data.data))

def get_frequency_from_timeseries(time_series, method_type='numerical-order-1'):
    """This function computes the instantaneous frequency from a given time series using either a numerical or analytical method. 
       The phase of the signal is first extracted, and the frequency is determined by differentiating the phase.
       Args:
           time_series (COMPLEX16TimeSeries): Time-domain mode.
           method_type (str, optional): Method used for frequency computation. Options are:
               - 'numerical-order-1' (default): Computes the first-order numerical derivative of the unwrapped phase.
               - 'analytical': Uses a smoothing spline to compute the analytical derivative of the phase.
       Returns:
           tuple: 
               - tvals (numpy.array): Time values corresponding to the input time series.
               - frequency (numpy.array): Instantaneous frequency as a function of time.
    """
    amp, phase = get_amplitude_phase(time_series) 
    tvals = np.arange(time_series.data.length) * time_series.deltaT
    if method_type=='numerical-order-1':
        dphi = np.unwrap(np.diff(phase)) 
        frequency = np.divide(-dphi, (2.*np.pi*time_series.deltaT))
        tmp = np.zeros(len(frequency)+1)
        tmp[:-1] = frequency
        frequency = tmp
    elif method_type=='analytical':
        spline = UnivariateSpline(tvals, phase, s=0)
        dphase_dt = spline.derivative()(tvals)
        frequency = -dphase_dt/(2.*np.pi)
    else:
        raise ValueError(f"Unknown method_type '{method_type}'. Use 'numerical-order-1' or 'analytical'.")
    return tvals, frequency

def cost_function(NR_modes, model_modes, tvals, hybridization_time_range):
    """This function computes the cost function by evaluating the residual difference between 
       numerical relativity (NR) waveform modes and model waveform modes over a specified time window.
       Args:
           NR_modes (dict): Dictionary containing NR waveform modes.
           model_modes (dict): Dictionary containing model waveform modes, structured similarly to `NR_modes`.
           tvals (numpy.array): Time values corresponding to the input waveform data.
           hybridization_time_range (tuple): Time range (t_start, t_end) over which the cost function is computed.
       Returns:
           float: Sum of the residual difference between NR and model waveform modes over the hybridization time range.
    """
    modes = list(model_modes.keys())
    residual_mode_tmp = np.zeros(NR_modes[modes[0]].data.length)
    for i, mode in enumerate(modes):
        residual_mode_tmp += np.abs(NR_modes[mode].data.data - model_modes[mode].data.data)
    indices_vals_window = np.logical_and(tvals >= hybridization_time_range[0], tvals <= hybridization_time_range[1]).flatten()
    residual_mode = np.zeros(NR_modes[modes[0]].data.length)
    residual_mode[indices_vals_window] = residual_mode_tmp[indices_vals_window]
    return np.sum(residual_mode)

def hybridize(NR_modes, model_modes, reference_frequency_22, polarization=0.0, hybridization_time_range=None, derivative_method_type='analytical', NR_waveform_starting_frequency=None, verbose=True, debug=True, thorough_output=False):
    """This function performs waveform hybridization by aligning NR and model waveform modes 
       in time and phase at a given reference frequency. The function then smoothly transitions between NR and model 
       waveforms within a specified hybridization time range.
       Args:
           NR_modes (dict): Dictionary of NR waveform modes, where values are COMPLEX16TimeSeries objects.
           model_modes (dict): Dictionary of model waveform modes, structured similarly to `NR_modes`.
           reference_frequency_22 (float): The reference frequency used for time and phase alignment.
           polarization (float, optional): Additional phase shift applied to all model waveform modes. Default is 0.0.
           hybridization_time_range (tuple, optional): Time range `(t_start, t_end)` over which the hybridization transition occurs.
                                                       If not provided, it is automatically determined.
           derivative_method_type (str, optional): Method for computing frequency derivatives. Options are:
                                                   - 'numerical-order-1': Uses a first-order numerical derivative.
                                                   - 'analytical'(default): Uses a spline-based analytical derivative.
           NR_waveform_starting_frequency (float, optional): Starting frequency of the NR waveform (2,2 mode). If provided, ensures
                                                          hybridization does not begin before the NR waveform is valid. 
           verbose (bool, optional): If True, prints intermediate values for sanity. Default is True.
           debug (bool, optional): If True, performs an additional check. Default is True.
           thorough_output (bool, optional): If True, returns additional data for detailed analysis. Default is False.
       Returns:
           hybridize_modes (dict): Dictionary of hybridized waveform modes, where values are COMPLEX16TimeSeries objects.
           
           If `thorough_output` is True, the function returns:
           - hybrid_modes (dict): Hybridized waveform modes.
           - NR_modes_new (dict): Resized and time-shifted NR modes.
           - model_modes (dict): Phase-shifted model modes.
           - tau (numpy.array): Blending function used for the transition between NR and model waveforms.
           - frequency_vals_NR_22 (numpy.array): Frequency evolution of the (2,2) NR mode.
           - frequency_vals_model_22 (numpy.array): Frequency evolution of the (2,2) model mode.
           - time_vals_model (numpy.array): Time values corresponding to both model and NR waveform modes.
           - fref_index_NR (int): Index of the reference frequency in the NR (2,2) time series. Should be the same for model (2,2) time series.
           - hybridization_time_range (tuple): The final hybridization time range used.
           - cost (float): Cost function value quantifying the difference between NR and model waveforms.
    """
    "NOTE: For best use set derivative_method_type='analytical, use time series with high sampling rate (for LIGO deltaT = 1/16384 s), provide NR modes as the first argument to this function, model modes as the second, and let the code find out NR_waveform_starting_frequency (set it to None)." 
    "NOTE: Also, the code will only provide sensible answers if the modes are uniformly sampled in time and this code will only work if (2,2) mode is present. However, this code can be modified to set the reference frequency wrt to another mode."
    
    NR_modes_list, model_modes_list = list(NR_modes.keys()), list(model_modes.keys())
    common_modes_list = np.intersect1d(NR_modes_list, model_modes_list)
    assert (2,2) in NR_modes_list and (2,2) in model_modes_list, '(2,2) mode needs to be present for this hybridization procedure to work.'
    print(f"Initiaing hybridization of modes: {common_modes_list}")
    ###########################################################################################
    # RESIZE
    ###########################################################################################
    # Resize to nearest power of 2 based on whichever has the largest length. 
    inital_NR_len, inital_model_len = NR_modes[2,2].data.length, model_modes[2,2].data.length
    max_length = np.max([inital_NR_len, inital_model_len])
    resize_length = pow2(max_length)

    ## resize + save the NR modes that are available in the model waveform, discard the rest.
    NR_modes_new = {}
    for mode in common_modes_list:
        NR_modes_new[mode] = lal.ResizeCOMPLEX16TimeSeries(NR_modes[mode], 0, resize_length)
        model_modes[mode] = lal.ResizeCOMPLEX16TimeSeries(model_modes[mode], 0, resize_length)
    if verbose:
        print(f'Inital NR (2,2) mode length = {inital_NR_len}, model (2,2) mode length = {inital_model_len}. Resizing both their modes to {resize_length}.\nNew NR (2,2) mode length = {NR_modes_new[2,2].data.length}, model (2,2) mode length = {model_modes[2,2].data.length}.')

    ###########################################################################################
    # TIME SHIFTS
    ###########################################################################################
    #  Here, I find the time for NR and model at a reference frequency and I move NR modes by the time shift amount.
    ## for model
    time_vals_model, frequency_vals_model_22 = get_frequency_from_timeseries(model_modes[2,2], method_type=derivative_method_type)
    fref_index_model = get_closest_index(frequency_vals_model_22, reference_frequency_22)
    time_at_fref_model = time_vals_model[fref_index_model]

    ## for NR
    time_vals_NR, frequency_vals_NR_22 = get_frequency_from_timeseries(NR_modes_new[2,2], method_type=derivative_method_type)
    fref_index_NR = get_closest_index(frequency_vals_NR_22, reference_frequency_22)
    time_at_fref_NR  = time_vals_NR[fref_index_NR]

    ## find out NR starting frequency so hybridization window can be checked.
    if NR_waveform_starting_frequency == None:
        NR_waveform_starting_frequency = frequency_vals_NR_22[0]
    else:
        if NR_waveform_starting_frequency < frequency_vals_NR_22[0]:
            NR_waveform_starting_frequency = frequency_vals_NR_22[0] # sometimes the calculated frequency from the derivative is higher than what is actual f_min_NR and that throws off all calculations. This makes sure that doesn't happen.
    print(f'NR_waveform_starting_frequency is {NR_waveform_starting_frequency} Hz. This assumes the provided NR modes start at index 0.') # Comment this if block if that is not true

    ## sanity check:  
    assert NR_waveform_starting_frequency <= reference_frequency_22, f"The reference frequency for hybridization {reference_frequency_22} is lower than NR's starting frequency {NR_waveform_starting_frequency}. Set reference frequency >= {NR_waveform_starting_frequency} Hz."
    
    ## time shift
    time_difference = time_at_fref_model - time_at_fref_NR
    frequency_vals_NR_22 = np.roll(frequency_vals_NR_22, int(time_difference/NR_modes_new[mode].deltaT))

    ## roll NR by that amount so that reference frequency occurs at the same time.
    for mode in model_modes_list:
        NR_modes_new[mode].data.data = np.roll(NR_modes_new[mode].data.data, int(time_difference/NR_modes_new[mode].deltaT))

    ## new index at which fref occurs. For sanity, make sure it matches model's.
    fref_index_NR = get_closest_index(frequency_vals_NR_22, reference_frequency_22)
    
    ## sanity check: Both 2,2  modes now should have same frequency at same time.
    assert time_vals_NR[fref_index_NR] == time_vals_model[fref_index_model], f'Time values at reference_frequency_22 {reference_frequency_22} do not match, for NR {time_vals_NR[fref_index_NR]} and for model = {time_vals_model[fref_index_model]}'
    if verbose:
        print(f'Time at reference_frequency_22 of {reference_frequency_22} Hz is {time_vals_NR[fref_index_NR]} s for NR and {time_vals_model[fref_index_model]} s for model.')
    
    ## set hybridization time range (around 4 cycles) if not provided
    if hybridization_time_range == None:
        hybridization_time_range = np.array([time_vals_NR[fref_index_NR] - 2/reference_frequency_22, time_vals_NR[fref_index_NR] + 2/reference_frequency_22])
        print(f'Hybridization time range not provided, setting it to: {hybridization_time_range}')
    
    ## modify hybridization range if NR waveform starts after the hybridization window starting time
    if not(NR_waveform_starting_frequency is None):
        starting_frequency_index_NR = get_closest_index(frequency_vals_NR_22, NR_waveform_starting_frequency)
        starting_time_NR = time_vals_NR[starting_frequency_index_NR]
        if hybridization_time_range[0] < starting_time_NR:
            hybridization_time_range[0] = starting_time_NR
            print(f"Starting time of the hybridization window is earlier than the starting time of the time shifted NR waveform. Readjusting the window to: {hybridization_time_range}.")
    else:
        print("WARNING: NR_waveform_starting_frequency not provided. Please check the hybridation window doesn't start before the NR (2,2) mode.")
    
    ## Another sanity check cause we like to be careful
    assert np.all((hybridization_time_range >= time_vals_NR[0]) & (hybridization_time_range <= time_vals_NR[-1]) & (hybridization_time_range[0] < hybridization_time_range[1]))

    ###########################################################################################
    # PHASE SHIFTS
    ###########################################################################################     
    # Find out the phase at reference frequency/time for NR, and make sure that the model has the same phase at that time for (2,2)
    ## NR
    amp_22_NR, phase_22_NR = get_amplitude_phase(NR_modes_new[2,2])
    amp_22_NR_ref, phase_22_NR_ref = amp_22_NR[fref_index_NR], phase_22_NR[fref_index_NR]

    ## model
    amp_22_model, phase_22_model = get_amplitude_phase(model_modes[2,2])
    amp_22_model_ref, phase_22_model_ref = amp_22_model[fref_index_model], phase_22_model[fref_index_model]

    ## phase shift 
    phase_shift =  phase_22_NR_ref - phase_22_model_ref

    ## correct for this phase shift in all model modes
    phase_here_new_dict = {}
    for mode in common_modes_list:
        m = mode[1]
        amp_here, phase_here = get_amplitude_phase(model_modes[mode])
        phase_here_new = phase_here + phase_shift * m/2 + 2*polarization # any global phase shift
        phase_here_new_dict[mode] = phase_here_new
        if verbose:
            print(f"Mode {mode}, model's phase before shift = {phase_here[fref_index_model]},  model's phase after shift = {phase_here_new[fref_index_model]}, NR's 2,2 mode phase at fref = {phase_22_NR_ref}.")
        content_here = amp_here * np.exp(1j*phase_here_new)  * amp_22_NR_ref/amp_22_model_ref # fit amplitude too
        model_modes[mode] = create_lal_COMPLEX16TimeSeries(model_modes[mode].deltaT, content_here, model_modes[mode].epoch, model_modes[mode].f0) 

    if debug:
        b = binary()
        b.df = 1/resize_length/NR_modes_new[2,2].deltaT
        mismatch, ovlp, max_ovlp, norms = b.mismatch_complex(NR_modes_new[2,2],  model_modes[2,2], psd='Flat', time_series=True, phase_max=False)
        print(f'Minimum mismatch for 2,2 modes is at time shift = {np.argmax(ovlp)*NR_modes_new[2,2].deltaT}')
        #return NR_modes_new, model_modes, phase_here_new_dict, frequency_vals_NR_22, frequency_vals_model_22, time_vals_model, fref_index_NR
    
    ###########################################################################################
    # RECOMBINATION
    ###########################################################################################
    tvals = time_vals_NR

    ## construct blending function
    tau = np.zeros(len(tvals))
    indices_vals_just_model = np.argwhere(tvals < hybridization_time_range[0]).flatten()
    indices_vals_just_NR = np.argwhere(tvals > hybridization_time_range[1]).flatten()
    indices_vals_window = np.logical_and(tvals >= hybridization_time_range[0], tvals <= hybridization_time_range[1]).flatten()
    tau[indices_vals_just_NR] = 1
    tau[indices_vals_window] = 0.5 * (1 - np.cos(np.pi * (tvals[indices_vals_window]- hybridization_time_range[0])/( hybridization_time_range[1]- hybridization_time_range[0])))
    
    ## construct hybrid modes
    hybrid_modes ={}
    for mode in common_modes_list:
        tmp_content_here = tau * NR_modes_new[mode].data.data + (1-tau) * model_modes[mode].data.data
        hybrid_modes[mode] = create_lal_COMPLEX16TimeSeries(model_modes[mode].deltaT, tmp_content_here, model_modes[mode].epoch, model_modes[mode].f0)

    ## evaluate cost function
    cost =  cost_function(NR_modes_new, model_modes, tvals, hybridization_time_range)
    print(f'For reference frequency of {reference_frequency_22} Hz the cost is {cost}')

    if thorough_output:
        return hybrid_modes, NR_modes_new, model_modes, tau, frequency_vals_NR_22, frequency_vals_model_22, time_vals_model, fref_index_NR, hybridization_time_range, cost
    
    return hybrid_modes
   


def test_modes(hlm_NR, hlm_model, reference_frequency, NR_waveform_starting_frequency=None, derivative_method_type='numerical-order-1'):
    hybrid_modes, NR_modes, model_mode, tau, f_NR, f_model, tvals, index, hybridization_range, cost = hybridize(hlm_NR, hlm_model, reference_frequency_22=reference_frequency, NR_waveform_starting_frequency=NR_waveform_starting_frequency, derivative_method_type=derivative_method_type, polarization=0.0, debug=True, verbose=True, thorough_output=True)
    print("Single plots")
    plt.show()
    plt.title("Frequency vs time")
    plt.ylim([-10, 200])
    plt.plot(tvals, f_NR, label='NR f_22(t)')
    plt.plot(tvals, f_model, linestyle = '--', label='Model f_22(t)')
    plt.axvline(x = tvals[index], color='grey', linestyle='dotted')
    plt.legend(loc='upper left')
    plt.show()

    amp_NR, phase_NR = get_amplitude_phase(NR_modes[2,2])
    amp_model, phase_model = get_amplitude_phase(model_mode[2,2])

    plt.title("Amplitude")
    plt.plot(tvals, amp_NR, label='NR (2,2) amplitude')
    plt.plot(tvals, amp_model, linestyle = '--', label='Model (2,2) amplitude')
    plt.axvline(x = tvals[index], color='grey', linestyle='dotted')
    plt.legend(loc='upper left')
    plt.show()

    plt.title("Phase")
    plt.plot(tvals, phase_NR, label='NR (2,2) phase')
    plt.plot(tvals, phase_model, linestyle = '--', label='Model (2,2) phase')
    print((phase_NR[index] - phase_model[index])%(np.pi*2))
    plt.axvline(x = tvals[index], color='grey', linestyle='dotted')
    plt.legend(loc='upper left')
    plt.show()

    plt.title("2,2 mode comparison")
    plt.xlim([0,21])
    plt.plot(tvals, NR_modes[2,2].data.data, label='NR (2,2) mode')
    plt.plot(tvals, model_mode[2,2].data.data, linestyle = '--', label='Model (2,2) mode')
    plt.axvline(x = tvals[index], color='grey', linestyle='dotted')
    plt.legend(loc='upper left')
    plt.show()

    plt.title("Blending function")
    plt.plot(tvals, tau)
    plt.show()

    
    plt.xlim([0.95*hybridization_range[0], 1.08*hybridization_range[1]])
    plt.plot(tvals, NR_modes[2,2].data.data, label = 'NR (2,2) mode')
    plt.plot(tvals, model_mode[2,2].data.data, linestyle = '--', label = 'Model (2,2) mode')
    plt.plot(tvals, hybrid_modes[2,2].data.data, label = 'Hybrid (2,2) mode')
    plt.axvline(x = hybridization_range[0], color='grey', linestyle='dotted')
    plt.axvline(x = hybridization_range[1], color='grey', linestyle='dotted')
    plt.legend(loc='upper left')
    plt.show()

    print("Comparison")
    for mode in list(hybrid_modes.keys()):
        plt.title(f'comparison: {mode}')
        plt.xlim([0.95*hybridization_range[0], 1.05*hybridization_range[1]])
        plt.plot(tvals, hybrid_modes[mode].data.data, label = 'Hybrid mode (Re)')
        plt.plot(tvals, hybrid_modes[mode].data.data.imag, label = 'Hybrid mode (Im)')
        plt.axvline(x = hybridization_range[0], color='grey', linestyle='dotted')
        plt.axvline(x = hybridization_range[1], color='grey', linestyle='dotted')
        plt.legend(loc='upper left')
        plt.show()

    for mode in list(hybrid_modes.keys()):
        plt.title(f'Difference: {mode}')
        plt.xlim([0.95*hybridization_range[0], 1.05*hybridization_range[1]])
        plt.plot(tvals, hybrid_modes[mode].data.data -  NR_modes[mode].data.data, label = 'Hybrid NR residual')
        plt.plot(tvals, hybrid_modes[mode].data.data - model_mode[mode].data.data, label = 'Hybrid model residual')
        plt.axvline(x = hybridization_range[0], color='grey', linestyle='dotted')
        plt.axvline(x = hybridization_range[1], color='grey', linestyle='dotted')
        plt.legend(loc='upper left')
        plt.show()

    for mode in list(hybrid_modes.keys()):
        plt.title(f'NR modes: {mode}')
        plt.xlim([0.95*hybridization_range[0], 1.05*hybridization_range[1]])
        plt.plot(tvals, NR_modes[mode].data.data, label='NR modes (Re)')
        plt.plot(tvals, NR_modes[mode].data.data.imag, label='NR modes (Im)')
        plt.axvline(x = hybridization_range[0], color='grey', linestyle='dotted')
        plt.axvline(x = hybridization_range[1], color='grey', linestyle='dotted')
        plt.legend(loc='upper left')
        plt.show()

    for mode in list(hybrid_modes.keys()):
        plt.title(f'model modes: {mode}')
        plt.xlim([0.95*hybridization_range[0], 1.05*hybridization_range[1]])
        plt.plot(tvals, model_mode[mode].data.data, label='Model modes (Re)')
        plt.plot(tvals, model_mode[mode].data.data.imag, label='Model modes (Im)')
        plt.axvline(x = hybridization_range[0], color='grey', linestyle='dotted')
        plt.axvline(x = hybridization_range[1], color='grey', linestyle='dotted')
        plt.legend(loc='upper left')
        plt.show()