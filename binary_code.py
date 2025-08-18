
####################CONSISTENCY
###ensure ample documentation
###Docstrings: Definition, argument, use (if necessary) and Output (with units)
#AIM: Fix no_cycles,  
# write my own xml reader,  
# write my own frame/.gwf reader
#NOTE = When dealing with PSDs, the only thing you have to change is resample_psd
import sys
import inspect
import h5py
import numpy as np
import matplotlib.pyplot as plt
import romspline
import os
from scipy import interpolate, signal
try:
    import lalsimulation as lalsim
except:
    print("INSTALL lalsimulation. Exiting")
    sys.exit()
try:
    import lal
except:
    print("INSTALL lal. Exiting")
    sys.exit()
from lal.series import read_psd_xmldoc
from ligo.lw import lsctables, utils, ligolw 
from lalframe import frread
from glue.lal import Cache


class binary:
    """A class for a binary system.
    Use help(binary) to see all methods and their docstrings.
    Tip: b.__dict__  should give you the list of attributes/parameters (here b is the binary object).
    """
    def __init__(self,  m1= 50*lal.MSUN_SI, m2= 50*lal.MSUN_SI, s1x=0.0, s1y=0.0, s1z=0.0, s2x=0.0, s2y=0.0, s2z=0.0, dist=400*10**6*lal.PC_SI, phiref=0,\
        incl=0, psi=0, eccentricity=0, approx="SEOBNRv4", meanPerAno=0, fmin=20., fmax=1024, dt=1./4096, df= 1./8, 
        fref=20., params=lal.CreateDict(), dec=0, ra=0, NR_hdf5 = None):
        """m1, m2 in kg
        s1, s2 dimensionless spins
        fmin, fmax, fref, df  in hertz
        dist in meters
        incl in radians
        phiref in radians
        dt  in seconds
        approx: a string
        psi
        meanPerAno
        params:Non GR parmeters lal dictionary"""
    
        ##intrinsic variables
        self.m1=m1
        self.m2=m2
        self.s1x=s1x
        self.s1y=s1y
        self.s1z=s1z
        self.s2x=s2x
        self.s2y=s2y
        self.s2z=s2z
        self.eccentricity=eccentricity

        ##extrinsic variables
        self.dist=dist
        self.phiref=phiref
        self.incl=incl
        self.meanPerAno=meanPerAno
        self.psi=psi
        self.dec=dec   #DEC =0 on the equator; the south pole has DEC = - pi/2
        self.ra=ra



        ##detection variables
        self.fmin=fmin
        self.fmax=fmax
        self.fref=fref
        self.dt=dt
        self.df=df

        self.params=lal.CreateDict()
        self.approx=approx
        self.NR_hdf5=NR_hdf5

    @property
    def srate(self):
        return 1/self.dt
    
    @property
    def fnyq(self):
        return 1/2/self.dt



############################Utils#########################################################
    def taper_time_series(self, hp, hc, taper_percent = 20, fmin = 0.0001):
        ntaper = int(taper_percent/100*hp.data.length) 
        ntaper = np.max([ntaper, int(1./(fmin*hp.deltaT))])  # require at least one waveform cycle of tapering; should never happen
        vectaper= 0.5 - 0.5*np.cos(np.pi*np.arange(ntaper)/(1.*ntaper))
        # Apply a naive filter to the start. Ideally, use an earlier frequency to start with
        hp.data.data[:ntaper]*=vectaper
        hc.data.data[:ntaper]*=vectaper
        return hp, hc

    def spin_weighted_spherical_harmonics(self, incl, phiref):
        s = np.sin(incl)
        c = np.cos(phiref)
        Y2m2 = np.sqrt( 5.0 / ( 64.0 * np.pi )) * ( 1.0 - c)*( 1.0 - c)
        Y2m1 = np.sqrt( 5.0 / ( 16.0 * np.pi ) ) * s*( 1.0 - c)
        Y21 = np.sqrt( 5.0 / ( 16.0 * np.pi ) ) * s*( 1.0 + c)
        Y22 = np.sqrt( 5.0 / ( 64.0 * np.pi ) ) * ( 1.0 + c)*( 1.0 + c)
        factor = np.exp(1j*2*phiref)
        return Y2m2, Y2m1, Y21, Y22

    def get_mass_from_mc_eta(self, mc, eta):
        """Returns m1, m2 from mc and eta."""
        alpha = mc / eta**(3/5)
        beta = mc**2 / eta**(1/5)
        m1 = 0.5 * (alpha + np.sqrt(alpha**2 - 4*beta))
        m2 = 0.5 * (alpha - np.sqrt(alpha**2 - 4*beta)) 
        return m1, m2
    def get_mass_from_mtot_q(self, mtot, q):
        if q > 1:
            q = 1/q
            print(f'Inputed q > 1, function defined for q < 1. Changing q to {q}')
        m1 = mtot / (1+q)
        m2 = q * m1
        return m1, m2
    
    def get_mass_from_mc_q(self, mc, q):
        """Returns m1, m2 from mc and q. Note: q<1"""
        m1 = mc * (1+q)**(1/5) / q ** (3/5)
        m2 = q * m1
        return m1, m2
    
    def get_mc_q_from_mass(self, m1, m2):
        """Returns mc, q from m1 and m2. Note: q<1"""
        mc = (m1*m2)**(3/5) / (m1+m2)**(1/5)
        q = m2/m1 
        return mc, q
    
    def get_mc_eta_from_mass(self, m1, m2):
        """Returns mc, q from m1 and m2. Note: q<1"""
        mc = (m1*m2)**(3/5) / (m1+m2)**(1/5)
        eta = (m1*m2) / (m1+m2)**(2)
        return mc, eta
    def Mc(self):
        """Returns chirp mass for the given binary system in kg.
        Argument:  self
        Output: Chirp mass in kg."""
        return((self.m1*self.m2)**(3/5) / (self.m1+self.m2)**(1/5))

    def q(self):
        """Returns mass ratio for a given binary system. 
        Argument: self
        Output: mass ratio (Dimensionless) (q>=1)"""
        return self.m1/self.m2  if self.m1 >=self.m2 else self.m2/self.m1

        

    def eta(self):
        """Returns symmetric mass ratio.
        Arguement: self
        Output: symmetric mass ratio"""
        return (self.m1*self.m2 / (self.m1+self.m2)**2)

    def time_to_merger(self):
        """Returns an approximate time to merger in seconds. Uses chirp mass and fmin to calculate time to merger, doesn't include spin.
        Argument: self
        Output: Approximate time to merger"""
        return( 2.18 * (1.21 *lal.MSUN_SI/self.Mc())**(5/3) * (100/self.fmin)**(8/3))

    def estimate_cycles(self):
        """Returns an approximate no of GW cycles that will be observed.  Uses chirp mass and fmin to calculate number of cycles, doesn't include spin.
        Argument: self
        Output: GW Cycles"""
        return(1.6*10**4 * (10/self.fmin)**(5/3) * (1.2*lal.MSUN_SI/self.Mc())**(5/3))

    def max_strain(self, h_p, h_c):
        """Returns max amplitude and the index at which it occurs.
        Argument: self, h_p.data.data and h_c.data.data. Not confined to lal objects.
        Output: max strain, index at which it occurs."""
        amp=np.sqrt(h_p**2+h_c**2)
        max=np.max(amp)
        index=np.argwhere(amp==max)[0][0]
        return max, index

    def estimate_df(self):
        """Returns an estimate for maximum deltaF you can have for the given set of params. You can always decrese deltaF by zero padding the waveform.
        Argument: self.
        Output: max deltaF [Hz]"""
        return 2**np.floor(np.log2(1/self.time_to_merger()))

    def pow2(self, length):
        """Looks for the nearst power of 2 for a given number. Useful for FFT as the number of data points need to be a power of 2
        Argument: self, length
        Output: Nearest power of two (int)"""
        return int(2**(int(np.ceil(np.log2(length)))))

    def forward_FFT(self, wf):
        """Takes in a TD waveform (has to be a lal object) and returns it in the frequency domain.
        Argument: self, REAL8 time series
        Output: a COMPLEX 16 frequency series
        Note: Ensure the waveform is tapered before you perform a FFT to avoid contaminating your answer with noise."""
        TDlen = wf.data.length
        if TDlen != self.pow2(TDlen):
            TDlen = self.pow2(TDlen)
            wf = lal.ResizeREAL8TimeSeries(wf,0,TDlen)
        fwdplan = lal.CreateForwardREAL8FFTPlan(TDlen,0)
        df = 1/TDlen/wf.deltaT

        FDlen = int(TDlen/2+1)
        wf_f = lal.CreateCOMPLEX16FrequencySeries("fft", wf.epoch, wf.f0, df, lal.HertzUnit, FDlen)
        lal.REAL8TimeFreqFFT(wf_f, wf, fwdplan)
        return wf_f
    
    def forward_FFT_complex(self, wf):
        """Takes in a TD waveform (has to be a lal object) and returns it in the frequency domain.
        Argument: self, Complex16 time series
        returns: a COMPLEX 16 frequency series
        Note: Ensure the waveform is tapered before you perform a FFT to avoid contaminating your answer with noise."""
        TDlen = wf.data.length
        if TDlen != self.pow2(TDlen):
            TDlen = self.pow2(TDlen)
            wf = lal.ResizeCOMPLEX16TimeSeries(wf,0,TDlen)
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(TDlen,0)
        df = 1/TDlen/wf.deltaT
        
        wf_f = lal.CreateCOMPLEX16FrequencySeries("fft", wf.epoch, wf.f0, df, lal.HertzUnit, TDlen)
        lal.COMPLEX16TimeFreqFFT(wf_f, wf, fwdplan)
        return wf_f

    def reverse_FFT(self, wf):
        """Takes in a FD waveform and returns it in Time Domain.
        Argument: Waveform (lal object COMPLEX16FrequencySeries)
        returns: a time series (REAL8TimeSeries)"""
        FDlen = wf.data.length
        TDlen = 2*(FDlen-1)
        assert TDlen == self.pow2(TDlen)  # this needs more checking
        revplan = lal.CreateReverseREAL8FFTPlan(TDlen, 0)
        dt = 1/TDlen/wf.deltaF   #this needs more checking


        wf_t = lal.CreateREAL8TimeSeries("rev_fft", wf.epoch, wf.f0, dt, lal.DimensionlessUnit, TDlen)
        lal.REAL8FreqTimeFFT(wf_t, wf, revplan)
        return wf_t
    
    def reverse_FFT_complex(self, wf):
        """Takes in a FD waveform and returns it in Time Domain (complex). Suitable for modes.
        Argument: Waveform (lal object COMPLEX16FrequencySeries)
        returns: a time series (Complex16TimeSeries)"""
        TDlen = wf.data.length
        revplan = lal.CreateReverseCOMPLEX16FFTPlan(TDlen, 0)
        dt = 1/TDlen/wf.deltaF   #this needs more checking


        wf_t = lal.CreateCOMPLEX16TimeSeries("rev_fft", wf.epoch, wf.f0, dt, lal.DimensionlessUnit, TDlen)
        lal.COMPLEX16FreqTimeFFT(wf_t, wf, revplan)
        return wf_t
    
    def where_tol(self, array, number, tol):
        a = np.argwhere(array > (number -tol)).T[0]
        b = np.argwhere(array < (number -tol )).T[0]
        return(np.intersect1d(a,b))
  
    def tvals(self,h_p,h_c):
        """Takes in the two polarizations and returns time array. The peak is at 0.
        Argument: self, h_plus (REAL 8 Time series), h_cross (REAL 8 Time series)
        Returns: time array with peak at 0 (numpy array)"""
        time=np.arange(0,h_p.data.length * h_p.deltaT, h_p.deltaT)
        #time = time + h_p.epoch
        time=time- time[self.max_strain(h_p.data.data,h_c.data.data)[1]] 
        return time

    def fvals(self, h_p):
        """Takes in a frequency domain waveform (need not be a polarization) and returns the frequency array in Hz.
        Argument: self, frequency domain waveform
        Returns: Frequency array."""
        return(np.arange(h_p.f0, h_p.deltaF*h_p.data.length, h_p.deltaF))

    def tvals_det(self, ht):
        """Takes in a time series and returns time array. The peak is at 0.
        Argument: self, ht(REAL 8 Time series)
        Returns: time array with peak at 0 (numpy array)"""
        time = np.arange(0,ht.data.length * ht.deltaT, ht.deltaT)
        time = time - time[np.argmax(ht.data.data**2)]
        return(time)  

    def f_evol_from_TD(self, hp, hc):
        """Take in TD h_plus and h_cross and outputs the time evolution of frequency."""
        #need to fix this, somehow this doesn't match pycbc # freq_py = waveform.utils.frequency_from_polarizations(hp_1, hc_1) #freq_py.sample_times
        phase = np.unwrap(np.arctan2(hc.data.data,hp.data.data))
        freq = np.diff(phase) / ( 2 * lal.PI *hp.deltaT )  
        time = self.tvals_det(hp)
        return (time[1:],freq)

    def f_evol_from_FD(self, hf):
        """Under works"""
        phase = np.unwrap(np.angle(hf.data.data))
        phase += phase[0]
        dphi = np.diff(phase)
        sample_frequencies=np.arange(hf.f0,hf.data.length*hf.deltaF,hf.deltaF)
        time = -dphi / (2.*np.pi*np.diff(sample_frequencies))
        nzidx = np.nonzero(abs(hf.data.data))[0]
        kmin, kmax = nzidx[0], nzidx[-2]
        # exclude everything after a discontinuity
        discont_idx = np.where(abs(dphi[kmin:]) >= 0.99*np.pi)[0]
        if discont_idx.size != 0:
            kmax = min(kmax, kmin + discont_idx[0]-1)
        time[:kmin] = time[kmin]
        time[kmax:] = time[kmax]
        return(time, sample_frequencies[1:])
    

    def snr(self, wf, psd="H1", flow=20, fhigh=2046, complex_time=False, polarizations=False, FD=False):
        """Takes in a REAL 8 time series, psd (H1 by default), flow (20 Hz by default), fhigh(2046.5 by default) and calculates snr.
        Argument: self, ht (REAL 8 time series), psd, flow (20 Hz by default), fhigh(2046 by default)
        Returns: snr"""
        if FD:
            hf_1=wf
        else:
            if complex_time:
                hf_1=self.forward_FFT_complex(wf)
            elif polarizations:
                ht1_p, ht1_c = wf[0], wf[1]
                TDlen_1=int(self.pow2(ht1_p.data.length)) #if we want a particular deltaF, that's why everything needs to be a power of 2
                assert TDlen_1>=ht1_p.data.length
                lal.ResizeREAL8TimeSeries(ht1_p, 0, TDlen_1)
                lal.ResizeREAL8TimeSeries(ht1_c, 0, TDlen_1)
                ht1=lal.CreateCOMPLEX16TimeSeries("ht1",ht1_p.epoch,ht1_p.f0,ht1_p.deltaT,lal.DimensionlessUnit,TDlen_1)
                ht1.data.data=ht1_p.data.data-1j*ht1_c.data.data
                hf_1=self.forward_FFT_complex(ht1)
            else:
                hf_1=self.forward_FFT(wf)
        curr_path=inspect.getfile(inspect.currentframe())
        index_path=curr_path.find("binary")
        if psd == "H1":
            psd=curr_path[:index_path]+"/PSD/LIGO_H1.txt" #deltaF=1./8 Hz
            #psd="/Users/aasim/Desktop/Research/Codes/My_modules/PSD/LIGO_RIFT.txt"  #deltaF=1./8 Hz
        if psd == "L1":
            psd=curr_path[:index_path]+"/PSD/LIGO_L1.txt"
        if psd == "V1":
            psd=curr_path[:index_path]+"/PSD/LIGO_V1.txt"
        if psd == "ET":
            psd=curr_path[:index_path]+"/PSD/ET.txt"
        if psd == "CE":
            psd=curr_path[:index_path]+"/PSD/CE.txt"
        if psd == "LISA":
             psd=curr_path[:index_path]+"/PSD/LISA.txt"
        if psd == "LISA_sens":
             psd=curr_path[:index_path]+"/PSD/LISA_sens.txt"

        psd_file=psd.split("/")[-1]
        print(f"Calculating SNR from flow={flow} Hz, fhigh={fhigh} Hz for PSD = {psd_file}")
        frequency, data=self.resample_psd(psd, df=hf_1.deltaF)
        i_min=int((flow-frequency[0])/hf_1.deltaF)  
        i_max=int((fhigh-frequency[0])/hf_1.deltaF)
        
        psd_new=np.zeros(hf_1.data.length)
        psd_new[i_min:i_max]=1/data[i_min:i_max]
        snr=np.sum(np.conjugate(hf_1.data.data)*hf_1.data.data*psd_new)
        snr=np.sqrt(4*hf_1.deltaF*np.abs(snr))

        return snr

    def plot_waveforms(self, wfs, labels = None , domain = "TD", x_lim = None, save_path = None, dpi =200,color = ["blue"]):
        """Function to quickly plot both time domain and frequency domain waveforms. Takes in an array of waveforms, respective labels (optional), x_lim (range of x-axis (optional)), save_path and dpi"""
        if not(domain in ["TD", "FD"]):
            print("domain can only be 'TD' or 'FD'. Exiting")
            sys.exit()
        if domain == "TD":
            plt.xlabel("Time [s]")
            plt.ylabel("h(t)")
            if x_lim:
                plt.gca().set_xlim(x_lim)
            for i in np.arange(len(wfs)):
                print(i)
                if labels != None:
                    plt.plot(self.tvals_det(wfs[i]), wfs[i].data.data, label = labels[i], color = color[i])
                else:
                    plt.plot(self.tvals_det(wfs[i]), wfs[i].data.data, color = color[i])
        if domain == "FD":
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("|h(f)|")
            if x_lim:
                plt.gca().set_xlim(x_lim)
            for i in np.arange(len(wfs)):
                plt.loglog(self.fvals(wfs[i]), np.abs(wfs[i].data.data), label = labels[i])
        if labels != None:
            plt.legend()
        if save_path:
            plt.savefig(save_path, dpi = dpi, bbox_inches = 'tight')
        plt.show()


    def frame_data_to_hoft(self, fname, channel= 'H1:FAKE-STRAIN', start= None, stop= None):
        with open(fname) as cfile:
            cachef = Cache.fromfile(cfile)
        cachef = cachef.sieve(ifos=channel[:1])

        duration = stop - start if None not in (start, stop) else None

        tmp = frread.read_timeseries(cachef, channel, start=start,duration=duration,verbose=True,datatype='REAL8')
        return tmp

    def LISA_res(self, wf1, wf2, res1, res2, det, alpha=4, flow=20, fhigh=2046, verbose=False, mismatch_overide=False, snr_overide=False, extra=False):
        """Returns 1/minimum resolution (M^-1) required for indistinguishible PE."""
        assert len(wf1)==len(wf2)==len(det),"Inconsistent input."
        mismatch_array=[]
        snr_array=[]
        mismatch=0
        network_snr=0
        if self.df==None:
                self.df=1/wf1[0].data.length/wf1[0].deltaT
                print(f"deltaF was not provided and was calculated for you. It is {self.df}")
        if not(mismatch_overide) and not(snr_overide):
            for i in range(len(wf1)):
                mismatch_array.append(self.mismatch_real(wf1[i],wf2[i],flow=flow,fhigh=fhigh,psd=det[i]))
                snr_array.append((self.snr(wf1[i],flow=flow,fhigh=fhigh,psd=det[i])+self.snr(wf2[i],flow=flow,fhigh=fhigh,psd=det[i]))*0.5)
                if verbose:
                    print(f"mismatch = {mismatch_array[i]}, snr ={snr_array[i]}, det = {det[i]}")
                mismatch+=mismatch_array[i]**2
                network_snr+=snr_array[i]**2
            mismatch=np.sqrt(mismatch)
            network_snr=np.sqrt(network_snr)

        if mismatch_overide:
            mismatch=mismatch_overide       
        if snr_overide:
            network_snr=snr_overide
        print(f"Network SNR = {network_snr}, mismatch = {mismatch}")

        beta=np.sqrt((2*mismatch) / ((res1)**alpha - (res2)**alpha)**2)

        if verbose:
            print(f"Beta = {beta}, alpha= {alpha}")

        rec_delta=(network_snr*beta)**(1/alpha)

        if extra:
            return(rec_delta, beta, mismatch, network_snr)
        else:
            return rec_delta

    def no_cycles(self,hp, hc = None):
        if hc is not None:
            time=self.tvals(hp,hc)
            phase = np.unwrap(np.arctan2(hc.data.data, hp.data.data))
            phase_0 = phase[0]
            #num_cycles = (phase[np.argmax(time>0)]-phase_0)/(2*np.pi)
            num_cycles = np.abs((phase[np.argmax(time>0)]-phase_0)/(2*np.pi))
            return num_cycles
        else:
            peak=np.argmax(hp.data.data**2)
            sign=hp.data.data[:peak]/np.abs(hp.data.data[:peak])
            sign_initial=sign[0]
            tmp_sign=sign[0]
            index=[]
            for i in range(len(sign)):
                if sign[i]==-tmp_sign:
                    tmp_sign=sign[i]
                    if sign[i]==sign_initial:
                        index.append(i)
                continue
            index=np.delete(index,[0])
            cycles=len(index)-1 #subtract the initial half cycle for safety
        return cycles, index

    def condition_TD(self,hp, hc=None,  beta=8, taper_cycles=6):
        """Only tapers the left hand side of the waveform. Doesn't have the capacity to taper half cycles. Keep in mind it will change the passed object"""

        if hc is not None:
            #Unwrap phase and count number of cycles
            time=self.tvals(hp,hc)
            phase = np.unwrap(np.arctan2(hc.data.data, hp.data.data))
            phase_0 = phase[0]
            #num_cycles = (phase[np.argmax(time>0)]-phase_0)/(2*np.pi)
            num_cycles = np.abs((phase[np.argmax(time>0)]-phase_0)/(2*np.pi))
            print(f"No of cycles till peak {num_cycles}")
            #Define how many cycles to taper
            taper_cycles = taper_cycles if num_cycles>10 else 1.5
            print("taper_cycles = "+str(taper_cycles))
        
            #Find index and time to begin and end taper based on number of cycles to taper
            #index_at_end_phase = np.argmax(phase > phase_0+2*np.pi*taper_cycles)
            index_at_end_phase = np.argmax(np.abs(phase) > np.abs(phase_0)+2*np.pi*taper_cycles)
            time_end = time[index_at_end_phase]

            
            time_start = time[0]
            width = time_end - time_start
            winlen = 2 * int(width / hp.deltaT)
            window = np.array(signal.get_window(('kaiser', beta), winlen))
            xmin = int((time_start - time[0]) / hp.deltaT)
            xmax = xmin + winlen//2
            print(time_start,time_end,xmin,xmax)
            hp.data.data[xmin:xmax] *= window[:winlen//2]
            hc.data.data[xmin:xmax] *= window[:winlen//2]
            return hp,hc
        else:
            index_nz=np.argmax(np.abs(hp.data.data)>(0+0*1j))  #find the index at which the zero padding at the head end stops so we know where to taper from
            tmp=hp.data.data[index_nz:]
            ht=lal.CreateREAL8TimeSeries("zero_pad_removed",hp.epoch,hp.f0,hp.deltaT,lal.DimensionlessUnit,len(tmp))
            ht.data.data=tmp
            num_cycles,index=self.no_cycles(ht)
            print(f"No of cycles till peak {num_cycles}")
            cycles_taper = taper_cycles if num_cycles>10 else 2
            print("taper_cycles="+str(cycles_taper))
            #time
            time=self.tvals_det(ht)
            time_end=time[index[cycles_taper]]
            time_start=time[0]
            #tapering from pycbc
            width = time_end - time_start
            winlen = 2 * int(width / hp.deltaT)
            window = np.array(signal.get_window(('kaiser', beta), winlen))
            xmin = int((time_start - time[0]) / hp.deltaT)
            xmax = xmin + winlen//2
            print(time_start,time_end,xmin,xmax)
            ht.data.data[xmin:xmax] *= window[:winlen//2]
            hp.data.data[index_nz:]=ht.data.data
            return hp




    def detector_response(self,hp,hc,det="H1", use_lalsim = True):
        """Take in two polarizations, h_plus and h_cross, and convert them a detector response time series """
        if use_lalsim:
            hp.epoch = hp.epoch + 1000000000.0   #this helps align same as RIFT
            hc.epoch = hc.epoch + 1000000000.0
            hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc,  self.ra, self.dec,  self.psi, lalsim.DetectorPrefixToLALDetector(str(det)))
            return(hoft)
        else:
            Fp = 0.5*(1. + np.cos(self.dec)*np.cos(self.dec))*np.cos(2.*self.ra)*np.cos(2.*self.psi) - np.cos(self.dec)*np.sin(2.*self.ra)*np.sin(2.*self.psi)
            Fc = 0.5*(1. + np.cos(self.dec)*np.cos(self.dec))*np.cos(2.*self.ra)*np.sin(2.*self.psi) + np.cos(self.dec)*np.sin(2.*self.ra)*np.cos(2.*self.psi)

            print(f"Fp = {Fp}, Fc = {Fc}, ra = {self.ra}, dec = {self.dec}, psi = {self.psi}")
            hp.data.data = Fp * hp.data.data
            hc.data.data = Fc * hc.data.data
            
            tmp = hp.data.data

            hp.data.data = hc.data.data + tmp
            return hp
        

    def Ylm(self, inclination, phiref, l ,m, s = -2):
        """Returns spin weighted spherical harmonics, s is to -2 as default."""
        return lal.SpinWeightedSphericalHarmonic(inclination,phiref,s,l,m)
    ####################################################WAVEFORM GENERATION #######################################
    def NRRIFT_to_hoft(self, path_to_xml, NR_hdf5_filename,lmax, det, event = 0):
        import NRWaveformCatalogManager3 as nrwf
        import RIFT.lalsimutils as lalsimutils
        print(nrwf.__file__)
        wfP = nrwf.WaveformModeCatalog('Sequence-MAYA-Generic', NR_hdf5_filename, clean_initial_transient=True, clean_final_decay=True, shift_by_extraction_radius=True, extraction_radius=None, lmax=lmax, align_at_peak_l2_m2_emission=True, build_strain_and_conserve_memory=True, perturbative_extraction=False, perturbative_extraction_full=False, use_provided_strain=True)
        filename = path_to_xml
        event = event
        xmldoc = utils.load_filename(filename, verbose = True,contenthandler=lalsimutils.cthdler)
        sim_inspiral_table = lsctables.SimInspiralTable.get_table(xmldoc)
        wfP.P.copy_sim_inspiral(sim_inspiral_table[int(event)])
        wfP.P.print_params()
        wfP.P.detector = det
        # Rescale window if needed. Higher masses need longer, to get right start frequency
        print(" NR duration (in s) of simulation at this mass = ", wfP.estimateDurationSec())
        print(" NR starting 22 mode frequency at this mass = ", wfP.estimateFminHz())
        #T_window = max([16., 2**int(2+np.log2(np.power(mtotMsun/150, 1+3./8.)))])
        T_window = max([16, 2**int(np.log(wfP.estimateDurationSec())/np.log(2)+1)])
        wfP.P.deltaF = 1./T_window
        print(" Final T_window ", T_window)
        # Generate signal
        hoft = wfP.real_hoft(hybrid_use=False) 
        return(hoft)



    def lalsim_TD(self,taper=False, verbose=True, only_mode = None, lmax = None):
        """Returns h_plus(t), h_cross(t) and time array (0 at peak), with default approx being SEOBNRv4. Can take in FD approximants too.
        Argument: self
        Output: h_plus(t) (REAL8TimeSeries), h_cross(t) (REAL8TimeSeries) and time array (0 at peak) (numpy array)"""
        modes = []
        if only_mode==None and lmax is not None:
            for l in range(2,lmax+1):
                for m in range(-l,0):
                    if self.approx == "NRHybSur3dq8" and l==4 and (m==0 or m==-1): #Throws an error for these modes instead of pass nothing like a normal person
                        continue
                    modes.append((l,m))
                for m in range(1,l+1):
                    if self.approx == "NRHybSur3dq8" and l==4 and (m==0 or m==1): #Throws an error for these modes instead of pass nothing like a normal person
                        continue
                    modes.append((l,m))
            print(f"Using modes {modes}")
        if only_mode is not None and lmax is None:
            for j in only_mode:
                modes.append(j)
            print(f"Using modes {modes}")
        if only_mode is not None and lmax is not None:
            print("Inconsistent input, use either lmax or only_mode.")
            sys.exit()
        if only_mode or lmax:
            ma = lalsim.SimInspiralCreateModeArray()
            for l,m in modes:
                lalsim.SimInspiralModeArrayActivateMode(ma, l, m)
            lalsim.SimInspiralWaveformParamsInsertModeArray(self.params, ma)
        if lalsim.SimInspiralImplementedTDApproximants(getattr(lalsim, self.approx))==1 and taper==False:
            if verbose:
                print(f"Using SimInspiraChooseTDWaveform {self.approx}")

            hl_p, hl_c = lalsim.SimInspiralChooseTDWaveform(self.m1, self.m2, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z, self.dist, self.incl, \
            self.phiref, self.psi, self.eccentricity, self.meanPerAno, self.dt, self.fmin, self.fref,
            self.params, getattr(lalsim, self.approx))
            time = np.arange(0,hl_p.data.length * hl_p.deltaT, hl_p.deltaT)
            time = time - time[self.max_strain(hl_p.data.data,hl_c.data.data)[1]] 
            # TDlen=hl_p.data.length
            # if self.df != None:
            #     Tdlen=1/self.df/hl_p.deltaT
            #     lal.ResizeREAL8TimeSeries(hl_c, 0 ,TDlen)   #that's why it is preferred to have df and dt multiples of 2
            #     lal.ResizeREAL8TimeSeries(hl_p, 0 ,TDlen)
        else:
            if verbose:
                print(f"Using SimInspiralTD {self.approx}")
            hl_p, hl_c = lalsim.SimInspiralTD(self.m1, self.m2, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z, self.dist, self.incl, \
            self.phiref, self.psi, self.eccentricity, self.meanPerAno, self.dt, self.fmin, self.fref,
            self.params, getattr(lalsim, self.approx))
            time = np.arange(0,hl_p.data.length * self.dt, self.dt)
            time = time - time[self.max_strain(hl_p.data.data,hl_c.data.data)[1]] 
            # TDlen=hl_p.data.length
            # if self.df != None:
            #     Tdlen=1/self.df/hl_p.deltaT
            #     lal.ResizeREAL8TimeSeries(hl_c, 0 ,TDlen)   #that's why it is preferred to have df and dt multiples of 2
            #     lal.ResizeREAL8TimeSeries(hl_p, 0 ,TDlen)
        #ht=lal.AddREAL8TimeSeries(hl_p, hl_c)
        return hl_p, hl_c, time
    

    def lalsim_FD(self,taper=False, verbose=True):
        """Returns h_p, h_c and frequency array with the default being IMRPhenomD. Can take TD approximants and output h_p, h_c in FD"""
        if lalsim.SimInspiralImplementedFDApproximants(getattr(lalsim, self.approx))==1 and taper==False:
            if verbose:
                print(f"Using SimInspiraChooseFDWaveform {self.approx}")
            hf_p, hf_c=lalsim.SimInspiralChooseFDWaveform(self.m1, self.m2, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z, self.dist, self.incl, \
            self.phiref, self.psi, self.eccentricity, self.meanPerAno, self.df, self.fmin, self.fmax, self.fref, 
            self.params, getattr(lalsim, self.approx))
            frequency=np.arange(hf_p.f0, hf_p.deltaF*len(hf_p.data.data), hf_p.deltaF)
            #hf=lal.AddCOMPLEX16FrequencySeries(hf_p, hf_c)
        else:
            if verbose:
                print(f"Using SimInspiralFD {self.approx}")
            hf_p, hf_c=lalsim.SimInspiralFD(self.m1, self.m2, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z, self.dist, self.incl, \
            self.phiref, self.psi, self.eccentricity, self.meanPerAno, self.df, self.fmin, self.fmax, self.fref, 
            self.params, getattr(lalsim, self.approx))
            frequency=np.arange(hf_p.f0, hf_p.deltaF*len(hf_p.data.data), hf_p.deltaF)
            # ht_p, ht_c,time= self.lalsim_TD()
            # TDlen=ht_p.data.length
            # if TDlen != self.pow2(TDlen):
            #     TDlen = self.pow2(TDlen)
            #     lal.ResizeREAL8TimeSeries(ht_c, 0 ,TDlen)
            #     lal.ResizeREAL8TimeSeries(ht_p, 0 ,TDlen)
            # fwdplan=lal.CreateForwardREAL8FFTPlan(TDlen,0)
            # FDlen=int(TDlen/2 +1)
            # df=1/(TDlen*ht_c.deltaT)
            # hf_c=lal.CreateCOMPLEX16FrequencySeries("ht_fft",ht_c.epoch,ht_c.f0,df,lal.HertzUnit, FDlen)
            # hf_p=lal.CreateCOMPLEX16FrequencySeries("ht_fft",ht_p.epoch,ht_p.f0,df,lal.HertzUnit, FDlen)
            # lal.REAL8TimeFreqFFT(hf_c, ht_c, fwdplan)
            # lal.REAL8TimeFreqFFT(hf_p, ht_p, fwdplan)
            # frequency=np.arange(hf_c.f0, df*len(hf_c.data.data), df)
        return hf_p, hf_c, frequency

    def lalsim_FD_modes(self, lmax = None, only_mode = None):
        """Returns the modes of a frequency domain waveform."""
        hlm = lalsim.SimInspiralChooseFDModes(self.m1, self.m2, self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z, self.df, self.fmin, self.fmax, self.fref, self.phiref, self.dist, self.incl, self.params, getattr(lalsim, self.approx))
        hlm_dict = {}
        if only_mode == None and lmax == None:
            lmax = 2 # default
        if only_mode==None and lmax is not None:
            for l in range(2, lmax+1):
                for m in range(-l, l+1):
                    hxx = lalsim.SphHarmFrequencySeriesGetMode(hlm, l, m)
                    if hxx is not None:
                        hlm_dict[(l,m)] = hxx
        if only_mode is not None and lmax is None:
            for j in only_mode:
                l, m = j
                hxx = lalsim.SphHarmFrequencySeriesGetMode(hlm, l, m)
                if hxx is not None:
                    hlm_dict[(l,m)] = hxx
        if only_mode is not None and lmax is not None:
            print("Inconsistent input, use either lmax or only_mode.")
            sys.exit()

        return hlm_dict
    
    def lalsim_TD_modes(self, verbose=True, lmax=2):
        if self.approx == "SEOBNRv4":
            hp, hc, tvals = self.lalsim_TD(taper=False) #called ChooseTDWaveform

            #aligned systems h_(l,-m) = (-1)**(l) conj(h_(l,m))
            ht_2_2 = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, hp.deltaT, lal.DimensionlessUnit, hp.data.length)
            ht_2_m2 = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hp.epoch, hp.f0, hp.deltaT, lal.DimensionlessUnit, hp.data.length)
            ht_2_2.data.data = np.sqrt(5./np.pi)/2*(hp.data.data + 1j * -1 * hc.data.data)
            ht_2_m2.data.data = np.sqrt(5./np.pi)/2*(np.conj(hp.data.data + 1j * -1 * hc.data.data))

            hlm = {}
            hlm[(2,2)] = ht_2_2
            hlm[(2,-2)] = ht_2_m2
            return hlm
        elif self.approx == "IMRPhenomXHM":
            if verbose:
                print(f"Using ChooseTDWaveform to get modes {self.approx}")
            hlmf = self.lalsim_FD_modes(lmax = lmax)
            hlm = {}
            for mode in hlmf.keys():
                hlm[mode] = self.reverse_FFT_complex(hlmf[mode])
            return hlm
        elif self.approx == "IMRPhenomD":
            hlms = lalsim.SimInspiralTDModesFromPolarizations( \
                    self.m1, self.m2, \
                    self.s1x, self.s1y, self.s1z, \
                    self.s2x, self.s2y, self.s2z, \
                    self.dist, self.phiref,  \
                    self.psi, self.eccentricity, self.meanPerAno, \
                    self.dt, self.fmin, self.fref, \
                    self.params, getattr(lalsim, self.approx))
            hlm_dict = {}
            for l in range(2, lmax+1):
                for m in range(-l, l+1):
                    hxx = lalsim.SphHarmFrequencySeriesGetMode(hlms, l, m)
                    if hxx is not None:
                        hlm_dict[(l,m)] = hxx
            return hlm_dict
        else:
            hlms = lalsim.SimInspiralChooseTDModes(self.phiref, self.dt, self.m1, self.m2, \
            self.s1x, self.s1y, self.s1z, \
            self.s2x, self.s2y, self.s2z, \
            self.fmin, self.fref, self.dist, self.params, lmax, getattr(lalsim, self.approx))
            hlm_dict = {}
            for l in range(2, lmax+1):
                for m in range(0, l+1):
                    hxx = lalsim.SphHarmTimeSeriesGetMode(hlms, l, m)
                    if hxx is not None:
                        hlm_dict[(l,m)] = hxx
                        if self.s1x == self.s1y == self.s2x == self.s2y == 0.0:
                            hlm_negative_m = lal.CreateCOMPLEX16TimeSeries("Complex h(t)", hlm_dict[(l,m)].epoch, hlm_dict[(l,m)].f0,
                                                            hlm_dict[(l,m)].deltaT, lal.DimensionlessUnit, hlm_dict[(l,m)].data.length)
                            hlm_negative_m.data.data = (-1)**l * np.conj(hlm_dict[(l,m)].data.data)
                            hlm_dict[(l,-m)] = hlm_negative_m
            return hlm_dict

        
    def NR_to_lalsimTD(self, path_to_hdf5, mtotal= None, lmax= None, only_mode=None, taper = True, use_lalsim = True, taper_percent = 10):
        """Takes in a NR waveform in  LVK hdf5 format, binary object and total mass in kg (default is 100 MSUN) and generates a TD waveform but as a lal REAL8TIMESeries. \
            The binary object that you use to call this function will populate extrinsic and detection variables. \
            The h5 file only has mass (total mass = 1) and spin information.
        Argument = self, path/to/NR/hdf5 file (string), mtotal (in kg) (default is 100 MSUN)
        Output = h_plus(t) (REAL8TimeSeries), h_cross(t) (REAL8TimeSeries) and time array (0 at peak) (numpy array)"""
        if mtotal == None:
            mtotal= (self.m1 + self.m2) 

        if taper == True  and use_lalsim ==True:
            print(f"Using SimInspiralTD NR_hdf5")
        if taper == False and use_lalsim ==True:
            print(f"Using SimInspiralChooseTDWaveform NR_hdf5")

        data_1 = h5py.File(path_to_hdf5,"r")
        m1 = data_1.attrs["mass1"] * mtotal 
        m2 = data_1.attrs["mass2"] * mtotal
        fmin = data_1.attrs["f_lower_at_1MSUN"] * lal.MSUN_SI/mtotal
        fref = self.fref
        print(f"Smallest possible fmin for this waveform {fmin} Hz. fmin at 1 solar mass is {data_1.attrs['f_lower_at_1MSUN']}")

        #THIS DOESN'T MATTER IF WE DON'T USE LALSIM TO GENERATE WAVEFORMS. WE ARE USING FULL NR WAVEFORMS OTHERWISE
        if (self.fmin < fmin) and use_lalsim==True and self.fmin !=0.0:
            fmin = fmin + 0.5*10**(-2)*fmin
            print(f"Can't have fmin less than that of the NR waveform. Defaulting to fmin={fmin} Hz.")

        else:
            fmin = self.fmin

        s1x, s1y, s1z, s2x, s2y, s2z = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(self.fref, mtotal/lal.MSUN_SI, path_to_hdf5)
        params = lal.CreateDict()
        modes = []
        if only_mode == None and lmax == None:
            lmax = data_1.attrs["Lmax"]
        if only_mode==None and lmax is not None:
            for l in range(2,lmax+1):
                for m in range(-l,0):
                    modes.append((l,m))
                for m in range(1,l+1):
                    modes.append((l,m))
        if only_mode is not None and lmax is None:
            for j in only_mode:
                modes.append(j)
        if only_mode is not None and lmax is not None:
            print("Inconsistent input, use either lmax or only_mode.")
            sys.exit()
        print(f"modes used = {modes}")
        # if only_mode is and lmax == False:
        #     for j in only_mode:
        #         modes.append(j)
        # elif only_mode == False and lmax != False:
        #         for l in range(2,lmax+1):
        #             for m in range(-l,0):
        #                 modes.append((l,m))
        #             for m in range(1,l+1):
        #                 modes.append((l,m))
        # else:
        #     assert only_mode == None and lmax == None,"Inconsistent input, use either lmax or only_mode."
        #     lmax = data_1.attrs["Lmax"]
        #     for l in range(2,lmax+1):
        #         for m in range(-l,0):
        #             modes.append((l,m))
        #         for m in range(1,l+1):
        #             modes.append((l,m))
        # print(f"modes used = {modes}")
        if use_lalsim:
            ma = lalsim.SimInspiralCreateModeArray()
            for l,m in modes:
                lalsim.SimInspiralModeArrayActivateMode(ma, l, m)
            lalsim.SimInspiralWaveformParamsInsertModeArray(params, ma)
            lalsim.SimInspiralWaveformParamsInsertNumRelData(params, path_to_hdf5)
            print(f"Generating waveform with m1 = {m1/lal.MSUN_SI:0.4f} MSUN, m2 = {m2/lal.MSUN_SI:0.4f} MSUN \n s1 = {s1x, s1y, s1z}, s2 = {s2x, s2y, s2z}\n fmin = {fmin} Hz, fref= {self.fref}")
            if taper:
                h_p, h_c = lalsim.SimInspiralTD(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, self.dist, self.incl, \
                self.phiref, self.psi, self.eccentricity, self.meanPerAno, self.dt, fmin, fref, params, lalsim.NR_hdf5 )
                h_p, h_c = self.taper_time_series(h_p, h_c, taper_percent = taper_percent, fmin = fmin)
            else:
                h_p, h_c = lalsim.SimInspiralChooseTDWaveform(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, self.dist, self.incl, \
                self.phiref, self.psi, self.eccentricity, self.meanPerAno, self.dt, fmin, fref, params, lalsim.NR_hdf5 )
        else:
            taper_percent = taper_percent if taper == True else 0
            hlm = self.NR_to_lalsimTD_modes(path_to_hdf5=path_to_hdf5, lmax =lmax, only_mode=only_mode, mtotal= mtotal, taper_percent= taper_percent)
            keys = list(hlm.keys())
            for i in range(len(keys)):
                if i == 0 :
                    tmp = hlm[keys[i]].data.data * self.Ylm(self.incl,-self.phiref, keys[i][0], keys[i][1])
                else:
                    tmp +=hlm[keys[i]].data.data * self.Ylm(self.incl,-self.phiref, keys[i][0], keys[i][1])

            h_p = lal.CreateREAL8TimeSeries("hlm",0,0, self.dt,lal.DimensionlessUnit,len(tmp))
            h_p.data.data = np.real(tmp)
            h_c = lal.CreateREAL8TimeSeries("hlm",0,0, self.dt,lal.DimensionlessUnit,len(tmp))
            h_c.data.data = -np.imag(tmp)

        return h_p, h_c, self.tvals(h_p,h_c)
    
    def NR_to_lalsimTD_modes(self, path_to_hdf5, lmax= None, only_mode=None, mtotal =None, taper_percent = 10, beta = 8, verbose = False, include_m_0_modes = False):
        """Takes in an NR h5 file and uses romspline interpolation to generate hlm. Outputs a hlm dict. The binary class will only populate distance and deltaT, intrinsic params are set by the simulation/file and other extrinsic params either go into detector response or Ylm.
        Note: Would need to see how precessing waveforms work with this, considering I would need to change frames depending on fref."""

        
        assert 0<=taper_percent <=100, "taper_percent should be between 0 and 100."
        #For unit conversion
        MSUN_sec = lal.G_SI/lal.C_SI**3
        if mtotal == None:
            mtot_in_sec= (self.m1 + self.m2) * MSUN_sec
        else:
            mtot_in_sec= mtotal *  MSUN_sec
        dist_in_sec = self.dist * 1/lal.C_SI

        #just to know what time array we are dealing with
        data_1 = h5py.File(path_to_hdf5)
        
        m1 = data_1.attrs["mass1"] * mtotal 
        m2 = data_1.attrs["mass2"] * mtotal
        fmin = data_1.attrs["f_lower_at_1MSUN"] * lal.MSUN_SI/mtotal
        if verbose:
            print(f"Smallest possible fmin for this waveform {fmin} Hz. fmin at 1 solar mass is {data_1.attrs['f_lower_at_1MSUN']}")
        s1x, s1y, s1z, s2x, s2y, s2z = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(self.fref, mtotal/lal.MSUN_SI, path_to_hdf5)
        print(f"Generating waveform with m1 = {m1/lal.MSUN_SI:0.4f} MSUN, m2 = {m2/lal.MSUN_SI:0.4f} MSUN \n s1 = {s1x, s1y, s1z}, s2 = {s2x, s2y, s2z}\n fmin = {fmin} Hz")

        #Which modes to get
        modes = []
        if only_mode == None and lmax == None:
            lmax = data_1.attrs["Lmax"]
        if only_mode==None and lmax is not None:
            for l in range(2,lmax+1):
                if include_m_0_modes:
                    for m in range(-l,l+1):
                        modes.append((l,m))
                else:
                    for m in range(-l,0):
                        modes.append((l,m))
                    for m in range(1,l+1):
                        modes.append((l,m))
        if only_mode is not None and lmax is None:
            for j in only_mode:
                modes.append(j)
        if only_mode is not None and lmax is not None:
            print("Inconsistent input, use either lmax or only_mode.")
            sys.exit()
        print(f"modes used = {modes}")

        #interpolating using romspline
        hlm = {}

        for i in range(len(modes)):
            amp22_time_0=np.array(data_1[f"phase_l{modes[i][0]}_m{modes[i][1]}"]["X"])

            amp = romspline.readSpline(path_to_hdf5, f"amp_l{modes[i][0]}_m{modes[i][1]}")
            phase = romspline.readSpline(path_to_hdf5, f"phase_l{modes[i][0]}_m{modes[i][1]}")
            
            amp22_time_0 = np.arange(np.min(amp22_time_0), np.max(amp22_time_0), self.dt/mtot_in_sec)
            generated_amp = amp(amp22_time_0 )
            generated_phase = phase(amp22_time_0 )
            generated_phase = self.unwind_phase(generated_phase)

            #tapering
            tvals = np.arange(0, self.dt * len(generated_amp), self.dt)
            if 100 >= taper_percent > 0: #percent defined with respect to peak time, 100 percent mean taper all the way to peak
                peak_index = generated_amp.argmax()
                time_peak = tvals[peak_index]
                taper_time = time_peak * taper_percent/100
                index_taper = np.abs(tvals-taper_time).argmin()

                time_start = tvals[np.argwhere(generated_amp > 0)[0][0]]
                width = tvals[index_taper] - time_start
                winlen = 2 * int(width / self.dt)
                window = np.array(signal.get_window(('kaiser', beta), winlen))
                xmin = int((time_start - tvals[0]) / self.dt)
                xmax = xmin + winlen//2
                if verbose and i == 0:
                    print(f"total time = {tvals[-1]}s, taper till {tvals[index_taper]} which is {tvals[index_taper]/time_peak * 100} percent.")
                    print(time_start, tvals[index_taper], xmin, xmax)
                generated_amp[xmin:xmax] *= window[:winlen//2]
                
                

            wf_data = mtot_in_sec/dist_in_sec * generated_amp * np.exp(1j*generated_phase)

            max_Re, max_Im = np.max(np.real(wf_data)), -np.max(np.imag(wf_data))
            print(f"Reading mode {modes[i]}, max for this mode: {max_Re, max_Im}")
            wf = lal.CreateCOMPLEX16TimeSeries("hlm",0,0,self.dt,lal.DimensionlessUnit,len(wf_data))
            wf.data.data = wf_data
            hlm[modes[i][0],modes[i][1]] = wf
        return hlm
    
    def unwind_phase(self, phase,thresh=5.):
        """
        Unwind an array of values of a periodic variable so that it does not jump
        discontinuously when it hits the periodic boundary, but changes smoothly
        outside the periodic range.

        Note: 'thresh', which determines if a discontinuous jump occurs, should be
        somewhat less than the periodic interval. Empirically, 5 is usually a safe
        value of thresh for a variable with period 2 pi.

        Fast method: take element-by-element differences, use mod 2 pi, and then add
        """
        cnt = 0 # count number of times phase wraps around branch cut
        length = len(phase)
        unwound = np.zeros(length)
        delta = np.zeros(length)

        unwound[0] =phase[0]
        delta = np.mod(phase[1:] - phase[:-1]+np.pi,2*np.pi)-np.pi                 # d(n)= p(n+1)-p(n) : the step forward item. The modulus is positive, so use an offset. The phase delta should be ~ 0 for each step
        unwound[1:] =unwound[0]+np.cumsum(delta)            # d(n)+d(n-1)=p(n)
    #    print delta, unwound

        # unwound[0] = phase[0]
        # for i in range(1,length):
        #     if phase[i-1] - phase[i] > thresh: # phase wrapped forward
        #         cnt += 1
        #     elif phase[i] - phase[i-1] > thresh: # phase wrapped backward
        #         cnt -= 1
        #     unwound[i] = phase[i] + cnt * 2. * np.pi
        return unwound


#######################################MISMATCH###########################################
    def resample_psd(self, psd, df=None):   #this acts weird due to non integer steps size, need to test it
        """Takes in a PSD which should have two columns, frequnecy and data, and returns it with a new  deltaF"""
        frequency, data = np.loadtxt(psd, delimiter=" ", comments="#",unpack=True)
        f0, deltaF, f_final = frequency[0], frequency[1]-frequency[0], frequency[-1]
        interp = interpolate.interp1d(frequency, data, fill_value = 'extrapolate')
        new_frequency = np.arange(f0, f_final+5*df, df or deltaF)
        return new_frequency, interp(new_frequency)

    def mismatch_complex(self, ht1, ht2, flow=20, fhigh=2046, psd="H1", time_series=False, plots=False, verbose=False, phase_max = True):
        """Calculates mismatch (maximised over time and phase) between two waveforms at a given set of parameters and for a given psd. 
        Starts with h(t) (complex meaning h_p - ih_c) of the approximants. 
        Potential tests: do we always need to have a double sided FD wf or single sided will work. .
        #DON"T give same polarizations, or you will get weird answers.
        Arguments: ht_1, ht_2, flow (default 20 Hz), fhigh (default 2046 Hz) (flow and fhigh are integration limits)
        optional: time_series = outputs mismatch time series and the maximum of that series gives you the time maximimsed mismatch, verbose = give more information
         
        """
        #print("WARNING: THIS FUNCTION WILL PAD THE WAVEFORM. ")
        ####Fourier Transform of approx1.######################
        if verbose:
            print(f"Phase maximization == {phase_max}")

        TDlen_1=int(self.pow2(1/ht1.deltaT/self.df)) #if we want a particular deltaF, that's why everything needs to be a power of 2
        assert TDlen_1>=ht1.data.length,f"The deltaF you requested cannot be used without losing information. Based on your params df <= {1/self.dt/ht1.data.length}. Consider decreasing deltaF or increasing fmin if you\
            really want to use this deltaF"
        lal.ResizeCOMPLEX16TimeSeries(ht1, 0, TDlen_1)

        deltaF_1=1/TDlen_1/ht1.deltaT
        hf_1=lal.CreateCOMPLEX16FrequencySeries("ht1_fft",ht1.epoch,ht1.f0,deltaF_1,lal.HertzUnit,TDlen_1)
        fwd=lal.CreateForwardCOMPLEX16FFTPlan(TDlen_1, 0)
        lal.COMPLEX16TimeFreqFFT(hf_1,ht1,fwd)

        ####Fourier transform of approx2.########################
        TDlen_2=int(self.pow2(1/ht2.deltaT/self.df)) #if we want a particular deltaF, that's why everything should be a power of 2
        assert TDlen_2>=ht2.data.length, f"The deltaF you requested cannot be used without losing information. Based on your params deltaF <= {1/self.dt/ht2.data.length}. Consider decreasing deltaF or increasing fmin if you\
            really want to use this deltaF"

        lal.ResizeCOMPLEX16TimeSeries(ht2, 0, TDlen_2)


        deltaF_2=1/TDlen_2/ht2.deltaT
        hf_2=lal.CreateCOMPLEX16FrequencySeries("ht_fft",ht2.epoch,ht2.f0,deltaF_2,lal.HertzUnit,TDlen_2)
        fwd=lal.CreateForwardCOMPLEX16FFTPlan(TDlen_2, 0)
        lal.COMPLEX16TimeFreqFFT(hf_2,ht2,fwd)

        assert deltaF_1==deltaF_2 and TDlen_1==TDlen_2,"deltaF and FDlen should be the same. Probably your waveforms don't have the same deltaT or same number of points in time series" #ht1_p.length and ht2_plength can be different, depends on approximant, need not a power of 2
        if deltaF_1 != self.df:
            print(f"Requested deltaF could not be used (probably not a power of 2). Instead the deltaF is {deltaF_1}. The psd has been resampled to reflect that change but that might introduce some errors.")


        #####double sided psd
        #####this psd takes into account the intergation range, it has non zero value in the range and zero outside the range.
        curr_path=inspect.getfile(inspect.currentframe())
        index_path=curr_path.find("binary")
        if psd == "H1":
            psd=curr_path[:index_path]+"/PSD/LIGO_H1.txt" #deltaF=1./8 Hz
            #psd="/Users/aasim/Desktop/Research/Codes/My_modules/PSD/LIGO_RIFT.txt"  #deltaF=1./8 Hz
        if psd == "L1":
            psd=curr_path[:index_path]+"/PSD/LIGO_L1.txt"
        if psd == "V1":
            psd=curr_path[:index_path]+"/PSD/LIGO_V1.txt"
        if psd == "ET":
            psd=curr_path[:index_path]+"/PSD/ET.txt"
        if psd == "CE":
            psd=curr_path[:index_path]+"/PSD/CE.txt"
        if psd == "LISA":
             psd=curr_path[:index_path]+"/PSD/LISA.txt"
        if psd == "LISA_sens":
             psd=curr_path[:index_path]+"/PSD/LISA_sens.txt"

        print(f"Integrating from flow={flow} Hz, fhigh={fhigh} Hz")
        if psd == "Flat":
            frequency = np.arange(flow-10*hf_1.deltaF, fhigh+10*hf_1.deltaF,hf_1.deltaF)
            data = np.ones(len(frequency))
        else:
            frequency, data=self.resample_psd(psd, df=hf_1.deltaF)
        i_min=int((flow-frequency[0])/hf_1.deltaF)  
        i_max=int((fhigh-frequency[0])/hf_1.deltaF)
        
        tmp=np.zeros(int(hf_1.data.length/2+1))
        try:
            tmp[i_min:i_max]=1/data[i_min:i_max]
        except Exception as e:
            print(f"Cannot proceed (potentially due to broadcasting error). It might be due to fhigh being greater than Nyquist frequency {1/ht2.deltaT/2} Hz.")
            print(e)
            sys.exit()
        psd_new=np.zeros(hf_1.data.length)
        psd_new[:len(tmp)]=tmp[::-1]    #[-N2--->0]
        psd_new[len(tmp)-1:]=tmp[:-1]   #[0--->N/2)   #zero index filled twice, +N/2 not there

        ####Calculating Norms#####################
        
        val_1=np.sum(np.conjugate(hf_1.data.data)*hf_1.data.data*psd_new)
        val_1=np.sqrt(4*deltaF_1*np.abs(val_1))


        val_2=np.sum(np.conjugate(hf_2.data.data)*hf_2.data.data*psd_new)
        val_2=np.sqrt(4*deltaF_2*np.abs(val_2))
        if verbose:
            print(f"norm_1 = {val_1}, norm_2 = {val_2}")

        # # ##########################################
        revplan=lal.CreateReverseCOMPLEX16FFTPlan(self.pow2(hf_1.data.length), 0)
        intgd = lal.CreateCOMPLEX16FrequencySeries("SNR integrand",lal.LIGOTimeGPS(0.), 0., hf_1.deltaF,
                   lal.HertzUnit, self.pow2(hf_1.data.length))
        deltaT=1/hf_1.deltaF/self.pow2(hf_1.data.length)
        ovlp = lal.CreateCOMPLEX16TimeSeries("Complex overlap",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit,
                   self.pow2(hf_1.data.length))
        ######only sum over a specific frequency range
        intgd.data.data = 4*np.conj(hf_1.data.data)*hf_2.data.data*psd_new
        #####maximising over time and phase (lalsimutils.py)
        lal.COMPLEX16FreqTimeFFT(ovlp, intgd,revplan)
        rhoSeries = np.abs(ovlp.data.data)
        rho = rhoSeries.max()
        if verbose:
            print(f"max innerproduct = {rho}")
        if plots:
            try:
                os.path.exists(plots)
                print("Not yet implemented")
            except:
                print("----Provided path doesn't exist. Skipping plots----")

        if time_series == True and phase_max == True:
            match = rho/val_1/val_2
            return (1-match, ovlp, rho, [val_1,val_2])
        
        if time_series == False and phase_max ==True:
            match = rho/val_1/val_2
            return 1-match
        
        if time_series == True and phase_max == False:
            time_shift_series = np.real(ovlp.data.data)
            match = time_shift_series.max()/val_1/val_2
            return(1-match, ovlp, time_shift_series.max(), [val_1,val_2])
        
        if time_series == False  and phase_max == False:
            time_shift_series = np.real(ovlp.data.data)
            match = time_shift_series.max()/val_1/val_2
            return 1-match

    def mismatch(self, wf1, wf2, flow=20, fhigh=2046, psd="H1", time_series=False, plots=False, verbose=False, phase_max = True):
        """Calculates mismatch (maximised over time and phase) between two waveforms at a given set of parameters and for a given psd. 
        Starts with h(t) (complex meaning h_p - ih_c) of the approximants. 
        Potential tests: do we always need to have a double sided FD wf or single sided will work. .
        #DON"T give same polarizations, or you will get weird answers.
        Arguments: ht_1, ht_2, flow (default 20 Hz), fhigh (default 2046 Hz) (flow and fhigh are integration limits)
        optional: time_series = outputs mismatch time series and the maximum of that series gives you the time maximimsed mismatch, verbose = give more information
         
        """
        #print("WARNING: THIS FUNCTION WILL PAD THE WAVEFORM. ")
        ####Fourier Transform of approx1.######################
        if verbose:
            print(f"Phase maximization == {phase_max}")
        ht1_p, ht1_c=wf1[0],wf1[1]

        TDlen_1=int(self.pow2(1/ht1_p.deltaT/self.df)) #if we want a particular deltaF, that's why everything needs to be a power of 2
        assert TDlen_1>=ht1_p.data.length,f"The deltaF you requested cannot be used without losing information. Based on your params df <= {1/self.dt/ht1_p.data.length}. Consider decreasing deltaF or increasing fmin if you\
            really want to use this deltaF"
        lal.ResizeREAL8TimeSeries(ht1_p, 0, TDlen_1)
        lal.ResizeREAL8TimeSeries(ht1_c, 0, TDlen_1)
        ht1=lal.CreateCOMPLEX16TimeSeries("ht1",ht1_p.epoch,ht1_p.f0,ht1_p.deltaT,lal.DimensionlessUnit,TDlen_1)
        ht1.data.data=ht1_p.data.data-1j*ht1_c.data.data

        deltaF_1=1/TDlen_1/ht1.deltaT
        hf_1=lal.CreateCOMPLEX16FrequencySeries("ht1_fft",ht1_p.epoch,ht1_p.f0,deltaF_1,lal.HertzUnit,TDlen_1)
        fwd=lal.CreateForwardCOMPLEX16FFTPlan(TDlen_1, 0)
        lal.COMPLEX16TimeFreqFFT(hf_1,ht1,fwd)

        ####Fourier transform of approx2.########################
        ht2_p, ht2_c = wf2[0], wf2[1]
        TDlen_2=int(self.pow2(1/ht2_p.deltaT/self.df)) #if we want a particular deltaF, that's why everything should be a power of 2
        assert TDlen_2>=ht2_p.data.length, f"The deltaF you requested cannot be used without losing information. Based on your params deltaF <= {1/self.dt/ht2_p.data.length}. Consider decreasing deltaF or increasing fmin if you\
            really want to use this deltaF"

        lal.ResizeREAL8TimeSeries(ht2_p, 0, TDlen_2)
        lal.ResizeREAL8TimeSeries(ht2_c, 0, TDlen_2)
        ht2=lal.CreateCOMPLEX16TimeSeries("ht_fft",ht2_p.epoch,ht2_p.f0,ht2_p.deltaT,lal.DimensionlessUnit,TDlen_2)
        ht2.data.data=ht2_p.data.data-1j*ht2_c.data.data

        deltaF_2=1/TDlen_2/ht2.deltaT
        hf_2=lal.CreateCOMPLEX16FrequencySeries("ht_fft",ht2_p.epoch,ht2_p.f0,deltaF_2,lal.HertzUnit,TDlen_2)
        fwd=lal.CreateForwardCOMPLEX16FFTPlan(TDlen_2, 0)
        lal.COMPLEX16TimeFreqFFT(hf_2,ht2,fwd)
        assert deltaF_1==deltaF_2 and TDlen_1==TDlen_2,"deltaF and FDlen should be the same. Probably your waveforms don't have the same deltaT or same number of points in time series" #ht1_p.length and ht2_plength can be different, depends on approximant, need not a power of 2
        if deltaF_1 != self.df:
            print(f"Requested deltaF could not be used (probably not a power of 2). Instead the deltaF is {deltaF_1}. The psd has been resampled to reflect that change but that might introduce some errors.")


        #####double sided psd
        #####this psd takes into account the intergation range, it has non zero value in the range and zero outside the range.
        curr_path=inspect.getfile(inspect.currentframe())
        index_path=curr_path.find("binary")
        if psd == "H1":
            psd=curr_path[:index_path]+"/PSD/LIGO_H1.txt" #deltaF=1./8 Hz
            #psd="/Users/aasim/Desktop/Research/Codes/My_modules/PSD/LIGO_RIFT.txt"  #deltaF=1./8 Hz
        if psd == "L1":
            psd=curr_path[:index_path]+"/PSD/LIGO_L1.txt"
        if psd == "V1":
            psd=curr_path[:index_path]+"/PSD/LIGO_V1.txt"
        if psd == "ET":
            psd=curr_path[:index_path]+"/PSD/ET.txt"
        if psd == "CE":
            psd=curr_path[:index_path]+"/PSD/CE.txt"
        if psd == "LISA":
             psd=curr_path[:index_path]+"/PSD/LISA.txt"
        if psd == "LISA_sens":
             psd=curr_path[:index_path]+"/PSD/LISA_sens.txt"

        print(f"Integrating from flow={flow} Hz, fhigh={fhigh} Hz")
        if psd == "Flat":
            frequency = np.arange(flow-10*hf_1.deltaF, fhigh+10*hf_1.deltaF,hf_1.deltaF)
            data = np.ones(len(frequency))
        else:
            frequency, data=self.resample_psd(psd, df=hf_1.deltaF)
        i_min=int((flow-frequency[0])/hf_1.deltaF)  
        i_max=int((fhigh-frequency[0])/hf_1.deltaF)
        
        tmp=np.zeros(int(hf_1.data.length/2+1))
        try:
            tmp[i_min:i_max]=1/data[i_min:i_max]
        except Exception as e:
            print(f"Cannot proceed (potentially due to broadcasting error). It might be due to fhigh being greater than Nyquist frequency {1/ht2.deltaT/2} Hz.")
            print(e)
            sys.exit()
        psd_new=np.zeros(hf_1.data.length)
        psd_new[:len(tmp)]=tmp[::-1]    #[-N2--->0]
        psd_new[len(tmp)-1:]=tmp[:-1]   #[0--->N/2)   #zero index filled twice, +N/2 not there

        ####Calculating Norms#####################
        
        val_1=np.sum(np.conjugate(hf_1.data.data)*hf_1.data.data*psd_new)
        val_1=np.sqrt(4*deltaF_1*np.abs(val_1))


        val_2=np.sum(np.conjugate(hf_2.data.data)*hf_2.data.data*psd_new)
        val_2=np.sqrt(4*deltaF_2*np.abs(val_2))
        if verbose:
            print(f"norm_1 = {val_1}, norm_2 = {val_2}")

        # # ##########################################
        revplan=lal.CreateReverseCOMPLEX16FFTPlan(self.pow2(hf_1.data.length), 0)
        intgd = lal.CreateCOMPLEX16FrequencySeries("SNR integrand",lal.LIGOTimeGPS(0.), 0., hf_1.deltaF,
                   lal.HertzUnit, self.pow2(hf_1.data.length))
        deltaT=1/hf_1.deltaF/self.pow2(hf_1.data.length)
        ovlp = lal.CreateCOMPLEX16TimeSeries("Complex overlap",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit,
                   self.pow2(hf_1.data.length))
        ######only sum over a specific frequency range
        intgd.data.data = 4*np.conj(hf_1.data.data)*hf_2.data.data*psd_new
        #####maximising over time and phase (lalsimutils.py)
        lal.COMPLEX16FreqTimeFFT(ovlp, intgd,revplan)
        rhoSeries = np.abs(ovlp.data.data)
        rho = rhoSeries.max()
        if verbose:
            print(f"max innerproduct = {rho}")
        if plots:
            try:
                os.path.exists(plots)
                print("Not yet implemented")
            except:
                print("----Provided path doesn't exist. Skipping plots----")

        if time_series == True and phase_max == True:
            match = rho/val_1/val_2
            return (1-match, ovlp, rho, [val_1,val_2])
        
        if time_series == False and phase_max ==True:
            match = rho/val_1/val_2
            return 1-match
        
        if time_series == True and phase_max == False:
            time_shift_series = np.real(ovlp.data.data)
            match = time_shift_series.max()/val_1/val_2
            return(1-match, ovlp, time_shift_series.max(), [val_1,val_2])
        
        if time_series == False  and phase_max == False:
            time_shift_series = np.real(ovlp.data.data)
            match = time_shift_series.max()/val_1/val_2
            return 1-match

    
    def mismatch_real(self, wf1, wf2, flow=20, fhigh=2046, psd="H1", time_series=False, plots=False, verbose=False, phase_max = True):
        """Calculates mismatch (maximised over time and phase) between two waveforms at a given set of parameters and for a given psd. 
        Starts with h(t) (real) of the approximants. 
        Potential tests: do we always need to have a double sided FD wf or single sided will work. .
        #DON"T give same polarizations, or you will get weird answers.
        Arguments: ht_1, ht_2, flow (default 20 Hz), fhigh (default 2046 Hz) (flow and fhigh are integration limits)
        optional: time_series = outputs mismatch time series and the maximum of that series gives you the time maximimsed mismatch, verbose = give more information
         
        """
        #print("WARNING: THIS FUNCTION WILL PAD THE WAVEFORM. ")
        ####Fourier Transform of approx1.######################
        if verbose:
            print(f"Phase maximization == {phase_max}")
        ht1= wf1

        TDlen_1=int(self.pow2(1/ht1.deltaT/self.df)) #if we want a particular deltaF, that's why everything needs to be a power of 2
        assert TDlen_1>=ht1.data.length,f"The deltaF you requested cannot be used without losing information. Based on your params df <= {1/self.dt/ht1.data.length}. Consider decreasing deltaF or increasing fmin if you\
            really want to use this deltaF"
        lal.ResizeREAL8TimeSeries(ht1, 0, TDlen_1)

        FDlen_1=int(TDlen_1/2+1)
        deltaF_1=1/TDlen_1/ht1.deltaT
        hf_1=lal.CreateCOMPLEX16FrequencySeries("ht1_fft",ht1.epoch,ht1.f0,deltaF_1,lal.HertzUnit,FDlen_1)
        fwd=lal.CreateForwardREAL8FFTPlan(TDlen_1, 0)
        lal.REAL8TimeFreqFFT(hf_1,ht1,fwd)
        if verbose:
            print(f"TDlen_1 = {TDlen_1}, FDlen_1= {FDlen_1}")

        ####Fourier transform of approx2.########################
        ht2= wf2
        TDlen_2=int(self.pow2(1/ht2.deltaT/self.df)) #if we want a particular deltaF, that's why everything should be a power of 2
        assert TDlen_2>=ht2.data.length, f"The deltaF you requested cannot be used without losing information. Based on your params df <= {1/self.dt/ht2.data.length}. Consider decreasing deltaF or increasing fmin if you\
            really want to use this deltaF"
        lal.ResizeREAL8TimeSeries(ht2, 0, TDlen_2)

        FDlen_2=int(TDlen_2/2+1)
        deltaF_2=1/TDlen_2/ht2.deltaT
        hf_2=lal.CreateCOMPLEX16FrequencySeries("ht_fft",ht2.epoch,ht2.f0,deltaF_2,lal.HertzUnit,FDlen_2)
        fwd=lal.CreateForwardREAL8FFTPlan(TDlen_2, 0)
        lal.REAL8TimeFreqFFT(hf_2,ht2,fwd)
        if plots:
            fig, ax = plt.subplots()
            ax.set_yscale("log")
            ax.plot(np.abs(hf_1.data.data))
            ax.plot(np.abs(hf_2.data.data))
        if verbose:
            print(f"TDlen_2 = {TDlen_2}, FDlen_2= {FDlen_2}")
        assert deltaF_1==deltaF_2 and FDlen_1==FDlen_2, "deltaF and FDlen should be the same. Probably your waveforms don't have the same deltaT or same number of points in time series"  #ht1_p.length and ht2_plength can be different, depends on approximant, need not a power of 2
        if deltaF_1 != self.df:
            print(f"Requested deltaF could not be used (probably not a power of 2). Instead the deltaF is {deltaF_1}. The psd has been resampled to reflect that change but that might introduce some errors.")

        curr_path=inspect.getfile(inspect.currentframe())
        index_path=curr_path.find("binary")
        if psd == "H1":
            psd=curr_path[:index_path]+"/PSD/LIGO_H1.txt" #deltaF=1./8 Hz
            #psd="/Users/aasim/Desktop/Research/Codes/My_modules/PSD/LIGO_RIFT.txt"  #deltaF=1./8 Hz
        if psd == "L1":
            psd=curr_path[:index_path]+"/PSD/LIGO_L1.txt"
        if psd == "V1":
            psd=curr_path[:index_path]+"/PSD/LIGO_V1.txt"
        if psd == "ET":
            psd=curr_path[:index_path]+"/PSD/ET.txt"
        if psd == "CE":
            psd=curr_path[:index_path]+"/PSD/CE.txt"
        if psd == "LISA":
             psd=curr_path[:index_path]+"/PSD/LISA.txt"
        if psd == "LISA_sens":
             psd=curr_path[:index_path]+"/PSD/LISA_sens.txt"
        print(f"Integrating from flow={flow} Hz, fhigh={fhigh} Hz")
        if psd == "Flat":
            frequency = np.arange(flow-10*hf_1.deltaF, fhigh+10*hf_1.deltaF,hf_1.deltaF)
            data = np.ones(len(frequency))
        else:
            frequency, data=self.resample_psd(psd, df=hf_1.deltaF)

        i_min=int((flow-frequency[0])/hf_1.deltaF)  
        i_max=int((fhigh-frequency[0])/hf_1.deltaF)
        #print(len(data),hf_1.data.length,i_min,i_max,i_max-i_min) #check for broadcasting error
        tmp=np.zeros(hf_1.data.length)
        print(i_min, i_max)
        try:
            tmp[i_min:i_max]=1/data[i_min:i_max]
        except Exception as e:
            print(f"Cannot proceed (potentially due to broadcasting error). It might be due to fhigh {fhigh} being greater than Nyquist frequency {1/ht2.deltaT/2} Hz.")
            print(e)
            sys.exit(0)
        psd_new=np.zeros(TDlen_1)
        psd_new[:len(tmp)]=tmp[::-1]    #[-N2--->0]
        psd_new[len(tmp)-1:]=tmp[:-1]   #[0--->N/2)   #zero index filled twice, +N/2 not there

        ####Calculating Norms#####################
        
        hf1=lal.CreateCOMPLEX16FrequencySeries("ht_fft",ht1.epoch,ht1.f0,deltaF_1,lal.HertzUnit,TDlen_1)
        hf1.data.data=np.zeros(TDlen_1)
        hf1.data.data[:len(tmp)]=hf_1.data.data[::-1]
        val_1=np.sum(np.conjugate(hf1.data.data)*hf1.data.data*psd_new)
        val_1=np.sqrt(4*deltaF_1*np.abs(val_1))

        hf2=lal.CreateCOMPLEX16FrequencySeries("ht_fft",ht2.epoch,ht2.f0,deltaF_2,lal.HertzUnit,TDlen_2)
        hf2.data.data=np.zeros(TDlen_2)
        hf2.data.data[:len(tmp)]=hf_2.data.data[::-1]
        val_2=np.sum(np.conjugate(hf2.data.data)*hf2.data.data*psd_new)
        val_2=np.sqrt(4*deltaF_2*np.abs(val_2))
        if verbose:
            print(f"norm_1 = {val_1},norm_2 = {val_2}")

        # # ##########################################
        revplan=lal.CreateReverseCOMPLEX16FFTPlan(self.pow2(hf1.data.length), 0)
        intgd = lal.CreateCOMPLEX16FrequencySeries("SNR integrand",lal.LIGOTimeGPS(0.), 0., hf1.deltaF,
                   lal.HertzUnit, self.pow2(hf1.data.length))
        deltaT=1/hf1.deltaF/self.pow2(hf1.data.length)
        ovlp = lal.CreateCOMPLEX16TimeSeries("Complex overlap",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit,
                   self.pow2(hf1.data.length))
        ######only sum over a specific frequency range
        intgd.data.data = 4*np.conj(hf1.data.data)*hf2.data.data*psd_new
        #####maximising over time and phase (lalsimutils.py)
        lal.COMPLEX16FreqTimeFFT(ovlp, intgd,revplan)
        rhoSeries = np.abs(ovlp.data.data)
        rho = rhoSeries.max()
        if verbose:
            print(f"max innerproduct ={rho}")
        if plots:
            try:
                os.path.exists(plots)
                print("Not yet implemented")
            except:
                print("----Provided path doesn't exist. Skipping plots----")
                pass
        if time_series == True and phase_max == True:
            match = rho/val_1/val_2
            return (1-match, ovlp, rho, [val_1,val_2])
        
        if time_series == False and phase_max ==True:
            match = rho/val_1/val_2
            return 1-match
        
        if time_series == True and phase_max == False:
            time_shift_series = np.real(ovlp.data.data)
            match = time_shift_series.max()/val_1/val_2
            return(1-match, ovlp, time_shift_series.max(), [val_1,val_2])
        
        if time_series == False  and phase_max == False:
            time_shift_series = np.real(ovlp.data.data)
            match = time_shift_series.max()/val_1/val_2
            return 1-match
###Inner products

    def IP_complex(self, wf1, wf2, flow=20, fhigh=2046, psd="H1", time_series=False, verbose=False, time_max = True):
        """Calculates mismatch (maximised over time and phase) between two waveforms at a given set of parameters and for a given psd. 
        Starts with h(t) (complex meaning h_p - ih_c) of the approximants. 
        optional: time_series = outputs mismatch time series and the maximum of that series gives you the time maximimsed mismatch, verbose = give more information
         
        """
        ####Fourier Transform of wf1######################

        assert wf1.deltaT == wf2.deltaT, "Both timeseries should have the same deltaT."
        TDlen_1=int(self.pow2(1/wf1.deltaT/self.df)) #if we want a particular deltaF, that's why everything needs to be a power of 2
        assert TDlen_1>=wf1.data.length,f"The deltaF you requested cannot be used without losing information. Based on your params df <= {1/self.dt/wf1.data.length}. Consider decreasing deltaF or increasing fmin if you\
            really want to use this deltaF"
        lal.ResizeCOMPLEX16TimeSeries(wf1, 0, TDlen_1)

        deltaF_1=1/TDlen_1/wf1.deltaT
        hf_1=lal.CreateCOMPLEX16FrequencySeries("ht1_fft",wf1.epoch,wf1.f0,deltaF_1,lal.HertzUnit,TDlen_1)
        fwd=lal.CreateForwardCOMPLEX16FFTPlan(TDlen_1, 0)
        lal.COMPLEX16TimeFreqFFT(hf_1,wf1,fwd)

        ####Fourier transform of wf2########################
        TDlen_2=int(self.pow2(1/wf2.deltaT/self.df)) #if we want a particular deltaF, that's why everything should be a power of 2
        assert TDlen_2>=wf2.data.length, f"The deltaF you requested cannot be used without losing information. Based on your params deltaF <= {1/self.dt/wf2.data.length}. Consider decreasing deltaF or increasing fmin if you\
            really want to use this deltaF"
        lal.ResizeCOMPLEX16TimeSeries(wf2, 0, TDlen_2)


        deltaF_2=1/TDlen_2/wf2.deltaT
        hf_2=lal.CreateCOMPLEX16FrequencySeries("ht_fft",wf2.epoch,wf2.f0,deltaF_2,lal.HertzUnit,TDlen_2)
        fwd=lal.CreateForwardCOMPLEX16FFTPlan(TDlen_2, 0)
        lal.COMPLEX16TimeFreqFFT(hf_2,wf2,fwd)

        if verbose:
            print(f"TDlen_1 = {TDlen_1}, TDlen_2 {TDlen_2}")
        assert deltaF_1==deltaF_2 and TDlen_1==TDlen_2,"deltaF and FDlen should be the same. Probably your waveforms don't have the same deltaT or same number of points in time series" #ht1_p.length and ht2_plength can be different, depends on approximant, need not a power of 2
        if deltaF_1 != self.df:
            print(f"Requested deltaF could not be used (probably not a power of 2). Instead the deltaF is {deltaF_1}. The psd has been resampled to reflect that change but that might introduce some errors.")


        #####double sided psd
        #####this psd takes into account the intergation range, it has non zero value in the range and zero outside the range.
        curr_path=inspect.getfile(inspect.currentframe())
        index_path=curr_path.find("binary")
        if psd == "H1":
            psd=curr_path[:index_path]+"/PSD/LIGO_H1.txt" #deltaF=1./8 Hz
            #psd="/Users/aasim/Desktop/Research/Codes/My_modules/PSD/LIGO_RIFT.txt"  #deltaF=1./8 Hz
        if psd == "L1":
            psd=curr_path[:index_path]+"/PSD/LIGO_L1.txt"
        if psd == "V1":
            psd=curr_path[:index_path]+"/PSD/LIGO_V1.txt"
        if psd == "ET":
            psd=curr_path[:index_path]+"/PSD/ET.txt"
        if psd == "CE":
            psd=curr_path[:index_path]+"/PSD/CE.txt"
        if psd == "LISA":
             psd=curr_path[:index_path]+"/PSD/LISA.txt"

        print(f"Integrating from flow={flow} Hz, fhigh={fhigh} Hz")
        if psd == "Flat":
            frequency = np.arange(flow-10*hf_1.deltaF, fhigh+10*hf_1.deltaF,hf_1.deltaF)
            data = np.ones(len(frequency))
        else:
            frequency, data=self.resample_psd(psd, df=hf_1.deltaF)
        i_min=int((flow-frequency[0])/hf_1.deltaF)  
        i_max=int((fhigh-frequency[0])/hf_1.deltaF)
        
        tmp=np.zeros(int(hf_1.data.length/2+1))
        try:
            tmp[i_min:i_max]=1/data[i_min:i_max]
        except Exception as e:
            print(f"Cannot proceed (potentially due to broadcasting error). It might be due to fhigh being greater than Nyquist frequency {1/wf2.deltaT/2} Hz.")
            print(e)
            sys.exit()
        psd_new=np.zeros(hf_1.data.length)
        psd_new[:len(tmp)]=tmp[::-1]    #[-N2--->0]
        psd_new[len(tmp)-1:]=tmp[:-1]   #[0--->N/2)   #zero index filled twice, +N/2 not there

        # # ##########################################
        if time_max==True:
            revplan=lal.CreateReverseCOMPLEX16FFTPlan(self.pow2(hf_1.data.length), 0)
            intgd = lal.CreateCOMPLEX16FrequencySeries("Inner product",lal.LIGOTimeGPS(0.), 0., hf_1.deltaF,
                    lal.HertzUnit, self.pow2(hf_1.data.length))
            deltaT=1/hf_1.deltaF/self.pow2(hf_1.data.length)
            ovlp = lal.CreateCOMPLEX16TimeSeries("Complex inner product",lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit,
                    self.pow2(hf_1.data.length))
            ######only sum over a specific frequency range
            intgd.data.data = 4 * np.conj(hf_1.data.data) * hf_2.data.data * psd_new
            #####maximising over time and phase (lalsimutils.py)
            lal.COMPLEX16FreqTimeFFT(ovlp, intgd, revplan)
            #overlap_series = np.real(ovlp.data.data)
            overlap_series = ovlp.data.data
            #overlap_max = overlap_series.max()
            tvals = self.tvals_det(ovlp)
            #ime_shift = tvals[overlap_series.argmax()]

            return overlap_series ,tvals
        
        else:
            IP = np.sum(4 * np.conj(hf_1.data.data) * hf_2.data.data * psd_new) * deltaF_1
            return IP
        
    def choosewaveformparams_to_binary(self, p):
        """Takes in RIFT's ChooseWaveformParams object and populates binary"""
        ##intrinic variables
        self.m1=p.m1
        self.m2=p.m2
        self.s1x,self.s1y,self.s1z=p.s1x,p.s1y,p.s1z
        self.s2x,self.s2y,self.s2z=p.s2x,p.s2y,p.s2z

        ##extrinsic variables
        self.dist=p.dist
        self.phiref=p.phiref
        self.incl=p.incl
        self.psi=p.psi
        self.eccentricity=p.eccentricity
        self.meanPerAno=p.meanPerAno
        if not(p.nonGRparams):
            self.params = lal.CreateDict()
        else:
            self.params=p.nonGRparams

        self.ra=p.phi
        self.dec=p.theta

        ##detection variables
        self.fmin=p.fmin
        self.fmax=p.fmax
        self.dt=p.deltaT
        if p.deltaF:  #sometimes it is type NONE and that can be annoying and in the case it is NONE, the binary object will assume default value (check __init__ for default value)
            self.df=p.deltaF
        
        self.fref=p.fref




    def binary_from_coalesence(self, coalescence_object, total_mass):
        """Takes in a coalescence object of mayawaves and populates binary object. It populates the mass, spin and eccentricity.
        Still need to define the extrinsic variables. Note: Can we get initial and final frequency from a NR simulation. Also, does the waveform exptrapolated to infinity
        take into account redshift. Horizon mass same as the mass we use for models?. Total mass needed. Need to clarify all this."""
        primary=coalescence_object.primary_compact_object 
        secondary=coalescence_object.secondary_compact_ibject 
        self.m1=primary.initial_horizon_mass * total_mass * lal.MSUN_SI
        self.m2=secondary.initial_horizon_mass * total_mass * lal.MSUN_SI
        s1, s2=primary.initial_dimensionless_spin, secondary.initial_dimensionless_spin
        self.s1x, self.s1y, self.s1z = s1[0], s1[1], s1[2]
        self.s2x, self.s2y, self.s2z = s2[0], s2[0], s2[0]

 

##########Test

if __name__ =='__main__':
    ###input params here
    p= binary()
    p.m1=120*lal.MSUN_SI
    p.m2=30*lal.MSUN_SI
    p.dist=400*10**6 *lal.PC_SI
    p.psi=np.pi/4
    p.phiref=0

    p.fmin=20
    p.dt=0.001
    p.fmax=0
    p.df=0.01
    p.approx="IMRPhenomD"
    hf_p, hf_c, frequency=p.lalsim_FD()
    print(len(hf_p.data.data)*p.df+hf_p.f0)
  
    
