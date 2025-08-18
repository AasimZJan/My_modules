import numpy as np
import lal


e = 0.004824185218078991
omega0 = 1.9909865927683788 * 10**(-7) #1/seconds
a = 149597870700. #meters
C_SI = 299792458.
L = 2*np.sqrt(3)*a*e
YRSID_SI = 31558149.763545603

def create_lal_frequency_series(frequency_values, hf, epoch = 950000000):
    assert len(frequency_values) == len(hf)
    hf_lal = lal.CreateCOMPLEX16FrequencySeries("hf", epoch, 0,  np.abs(np.diff(frequency_values)[0]), lal.HertzUnit, len(frequency_values))
    hf_lal.data.data = hf
    return hf_lal


def get_Ylm(inclination, phiref, l ,m, s = -2):
    """Returns spin weighted spherical harmonics, s is to -2 as default."""
    return lal.SpinWeightedSphericalHarmonic(inclination, phiref, s, l, m)



def transformed_Hplus_Hcross(beta, lamda, psi, theta, phiref, l, m):
    """This function transforms the plus and cross polarization from wave frame to SSB FRAME.
        Args: 
            beta  (float) = ecliptic latitude,  
            lamda (float) = ecliptic longitude, 
            psi   (float) = polarization
        Returns:
            ?
        """    

    #defining cos and sin for ease (checked)
    cl, sl = np.cos(lamda), np.sin(lamda)
    cb, sb = np.cos(beta),  np.sin(beta)
    cp, sp = np.cos(psi),   np.sin(psi)

    #polarization in waveframe (checked)
    Hp = [[1,0,0], [0,-1,0], [0,0,0]]
    Hc = [[0,1,0], [1,0,0] , [0,0,0]]

    #Rotation matrix (checked)
    O1 = np.zeros((3,3))
    
    O1[0,0] = cp*sl - cl*sb*sp
    O1[0,1] = -cl*cp*sb - sl*sp
    O1[0,2] = -cb*cl

    O1[1,0] = -cl*cp - sb*sl*sp
    O1[1,1] = -cp*sb*sl + cl*sp
    O1[1,2] = -cb*sl

    O1[2,0] = cb*sp
    O1[2,1] = cb*cp
    O1[2,2] = -sb
    
    #Transverse of Rotation matrix (checked)
    TO1 = np.zeros((3,3))

    TO1[0,0] = cp*sl - cl*sb*sp
    TO1[0,1] = -cp*cl - sl*sb*sp
    TO1[0,2] = cb*sp

    TO1[1,0] = -cl*cp*sb - sl*sp
    TO1[1,1] = -sl*cp*sb + cl*sp
    TO1[1,2] = cb*cp

    TO1[2,0] = -cb*cl
    TO1[2,1] = -cb*sl
    TO1[2,2] = -sb

    #Ylm factors (checked) (HMM)
    Ylm = get_Ylm(theta, -phiref, l,m,-2)
    Y_lm = (-1)**(l) * np.conj(get_Ylm(theta, -phiref, l,-m,-2))

    Yfactorplus = 0.5*(Ylm + Y_lm)
    Yfactorcross = 0.5*1j*(Ylm - Y_lm)

    Plm = Yfactorplus*np.matmul(O1, np.matmul(Hp,TO1)) + Yfactorcross*np.matmul(O1, np.matmul(Hc,TO1))
    return Plm


def get_tf_from_phase(hlm, fmax, debug = False):#tested
    """This func differentiates phase to get tf.
        Args: 
            hlm (COMPLEX16FrequencySeries) = The mode for which you are calculating tf , 
            fmax  (numpy.array)= frequency array, 
        Returns:
            tf (np.array), frequency (np.array)
        """    
    #((hlm[2,2].data.length-1)/2)*hlm[2,2].deltaF
    #send in hlm not shifted in time


    #get frequency and mode data
    freq = np.arange(-fmax, fmax+hlm.deltaF, hlm.deltaF)
    if debug:
        print(f"len(freq) = {len(freq)}, freq[0] = {freq[0]} Hz, freq[-1] = {freq[-1]} Hz, len(hlm) = {hlm.data.length}")

    #get amplitude and phase
    phase = np.unwrap(np.angle(hlm.data.data))
    #compute tf = -1/(2pi) * d(phase)/df
    phase  = phase - phase[0]
    dphi = np.unwrap(np.diff(phase))
    time = -dphi / (2.*np.pi*np.diff(freq))
    #diff reduces len by 1 so artifically increasing it by adding an extra zero at the end
    tmp = np.zeros(len(time)+1)
    tmp[:-1] = time
    time = tmp

    #only focusing on f bins where data exists
    nzidx = np.nonzero(abs(hlm.data.data))[0]
    if debug:
        print(nzidx)
        print(f"tf[0](after stripping zeros) = {time[kmin]}, tf[-1](after stripping zeros) {time[kmax]}")
    kmin, kmax = nzidx[0], nzidx[-2]
    time[:kmin] = time[kmin]
    time[kmax:] = time[kmax]
    if debug:
        print(f"len(time) = {len(time)}, len(freq) = {len(freq)}")

    #saving data
    return time, freq[::-1]

def Evaluate_Gslr(tf, f, H, beta, lamda, return_raw = False):
    """This is the main function, takes in tf, f, H, beta and lamda to generate transfer function for a given mode. yslr = Sum_l^r Gslr * hlm.
        Args: 
            tf (numpy.array)= -1/2pi d(phase)/df, 
            f  (numpy.array)= frequency array, 
            H  (numpy.array)= The plus and cross polarization matrices transformed from wave to SSB frame ,
            beta (float)    = ecliptic latitude, 
            lamda (float)   = ecliptic longitude
        Returns:
            Transfer function L1 (numpy.array), Transfer function L2 (numpy.array), Transfer function L3 (numpy.array)
        """
    alpha = omega0*tf
    c, s = np.cos(alpha), np.sin(alpha)
    k = np.array([-np.cos(beta)*np.cos(lamda), -np.cos(beta)*np.sin(lamda), -np.sin(beta)])
    p0 = np.array([a*c, a*s, np.zeros(len(tf))]) # (3, N)
    kR = np.dot(k, p0) # (N,)
    phaseRdelay = 2.*np.pi/C_SI *f*kR #(N,)

    p1L =np.array([-a*e*(1 + s*s), a*e*c*s, -a*e*np.sqrt(3)*c]) # (3, N)
    p2L =np.array([a*e/2*(np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s - np.sqrt(3)*(1 + c*c)),  -a*e*np.sqrt(3)/2*(np.sqrt(3)*s - c)]) # (3, N)
    p3L =np.array([a*e/2*(-np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s + np.sqrt(3)*(1 + c*c)), -a*e*np.sqrt(3)/2*(-np.sqrt(3)*s - c)]) # (3, N)


    n1 = np.array([-1./2*c*s, 1./2*(1 + c*c), np.sqrt(3)/2*s]) # (3, N)
    kn1= np.dot(k, n1) #(N,)
    n1Hn1 = np.einsum("ij,ji->i",n1.T, np.einsum("ij,jk", H, n1))

    n2 = 1./4. * np.array([c*s - np.sqrt(3)*(1 + s*s), np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s - 3*c])
    kn2= np.dot(k, n2)
    n2Hn2 = np.einsum("ij,ji->i",n2.T, np.einsum("ij,jk", H, n2))

    n3 = 1./4*np.array([c*s + np.sqrt(3)*(1 + s*s), -np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s + 3*c])
    kn3= np.dot(k, n3)
    n3Hn3 = np.einsum("ij,ji->i",n3.T, np.einsum("ij,jk", H, n3))

    

    kp1Lp2L = np.dot(k, (p1L+p2L))
    kp2Lp3L = np.dot(k, (p2L+p3L))
    kp3Lp1L = np.dot(k, (p3L+p1L))
    kp0 = np.dot(k, p0)

    factorcexp0 = np.exp(1j*2.*np.pi*f/C_SI * kp0)
    prefactor = np.pi*f*L/C_SI

    factorcexp12 = np.exp(1j*prefactor * (1.+kp1Lp2L/L))
    factorcexp23 = np.exp(1j*prefactor * (1.+kp2Lp3L/L))
    factorcexp31 = np.exp(1j*prefactor * (1.+kp3Lp1L/L))

    factorsinc12 = np.sinc( (prefactor * (1.-kn3))/np.pi)
    factorsinc21 = np.sinc( (prefactor * (1.+kn3))/np.pi)
    factorsinc23 = np.sinc( (prefactor * (1.-kn1))/np.pi)
    factorsinc32 = np.sinc( (prefactor * (1.+kn1))/np.pi)
    factorsinc31 = np.sinc( (prefactor * (1.-kn2))/np.pi)
    factorsinc13 = np.sinc( (prefactor * (1.+kn2))/np.pi)

    commonfac = 1j*prefactor*factorcexp0
    G12 = commonfac * n3Hn3 * factorsinc12 * factorcexp12 * np.exp(-1j*phaseRdelay)
    G21 = commonfac * n3Hn3 * factorsinc21 * factorcexp12 * np.exp(-1j*phaseRdelay)
    G23 = commonfac * n1Hn1 * factorsinc23 * factorcexp23 * np.exp(-1j*phaseRdelay)
    G32 = commonfac * n1Hn1 * factorsinc32 * factorcexp23 * np.exp(-1j*phaseRdelay)
    G31 = commonfac * n2Hn2 * factorsinc31 * factorcexp31 * np.exp(-1j*phaseRdelay)
    G13 = commonfac * n2Hn2 * factorsinc13 * factorcexp31 * np.exp(-1j*phaseRdelay)

    x = np.pi*f*L/C_SI
    z = np.exp(1j*2.*x)

    factor_convention = 2
    factorAE = 1j*np.sqrt(2)*np.sin(2.*x)*z
    factorT = 2.*np.sqrt(2)*np.sin(2.*x)*np.sin(x)*np.exp(1j*3.*x)

    Araw = 0.5 * ( (1.+z)*(G31 + G13) - G23 - z*G32 - G21 - z*G12 )
    Eraw = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13 - G31) + (2.+z)*(G12 - G32) + (1.+2.*z)*(G21 - G23) )
    Traw = 1/np.sqrt(6) * (G21 - G12 + G32 - G23 + G13 - G31)
    transferL1 = factor_convention * factorAE * Araw
    transferL2 = factor_convention * factorAE * Eraw
    transferL3 = factor_convention * factorT * Traw
    if return_raw:
        return factor_convention * Araw, factor_convention * Eraw, factor_convention * Traw
    else:
        return transferL1, transferL2, transferL3




### testing part


def transformed_Hplus_Hcross_test(beta, lamda, psi, return_Hp_Hc = False):
    """This function transforms the plus and cross polarization from wave frame to SSB FRAME.
        Args: 
            beta  (float) = ecliptic latitude,  
            lamda (float) = ecliptic longitude, 
            psi   (float) = polarization
        Returns:
            ?
        """    

    #defining cos and sin for ease (checked)
    cl, sl = np.cos(lamda), np.sin(lamda)
    cb, sb = np.cos(beta),  np.sin(beta)
    cp, sp = np.cos(psi),   np.sin(psi)

    #polarization in waveframe (checked)
    Hp = [[1,0,0], [0,-1,0], [0,0,0]]
    Hc = [[0,1,0], [1,0,0] , [0,0,0]]

    #Rotation matrix (checked)
    O1 = np.zeros((3,3))
    
    O1[0,0] = cp*sl - cl*sb*sp
    O1[0,1] = -cl*cp*sb - sl*sp
    O1[0,2] = -cb*cl

    O1[1,0] = -cl*cp - sb*sl*sp
    O1[1,1] = -cp*sb*sl + cl*sp
    O1[1,2] = -cb*sl

    O1[2,0] = cb*sp
    O1[2,1] = cb*cp
    O1[2,2] = -sb
    
    # Transverse of Rotation matrix (checked)
    TO1 = np.zeros((3,3))

    TO1[0,0] = cp*sl - cl*sb*sp
    TO1[0,1] = -cp*cl - sl*sb*sp
    TO1[0,2] = cb*sp

    TO1[1,0] = -cl*cp*sb - sl*sp
    TO1[1,1] = -sl*cp*sb + cl*sp
    TO1[1,2] = cb*cp

    TO1[2,0] = -cb*cl
    TO1[2,1] = -cb*sl
    TO1[2,2] = -sb


    # Transformed Hp and Hc
    hp_T = np.matmul(O1, np.matmul(Hp,TO1))
    hc_T = np.matmul(O1, np.matmul(Hc,TO1))
    if return_Hp_Hc:
        return hp_T, hc_T
    else:
        return 0.5 * (hp_T + 1j*hc_T), 0.5 * (hp_T - 1j*hc_T)




def Evaluate_Gslr_test(tf, f, Hp, Hm, beta, lamda):
    """This is the main function, takes in tf, f, H, beta and lamda to generate transfer function for a given mode. yslr = Sum_l^r Gslr * hlm.
        Args: 
            tf (numpy.array)= -1/2pi d(phase)/df, 
            f  (numpy.array)= frequency array, 
            H  (numpy.array)= The plus and cross polarization matrices transformed from wave to SSB frame ,
            beta (float)    = ecliptic latitude, 
            lamda (float)   = ecliptic longitude
        Returns:
            Transfer function L1 (numpy.array), Transfer function L2 (numpy.array), Transfer function L3 (numpy.array)
        """
    alpha = omega0*tf
    c, s = np.cos(alpha), np.sin(alpha)
    k = np.array([-np.cos(beta)*np.cos(lamda), -np.cos(beta)*np.sin(lamda), -np.sin(beta)])
    p0 = np.array([a*c, a*s, np.zeros(len(tf))]) # (3, N)
    kR = np.dot(k, p0) # (N,)
    phaseRdelay = 2.*np.pi/C_SI *f*kR #(N,)

    p1L =np.array([-a*e*(1 + s*s), a*e*c*s, -a*e*np.sqrt(3)*c]) # (3, N)
    p2L =np.array([a*e/2*(np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s - np.sqrt(3)*(1 + c*c)),  -a*e*np.sqrt(3)/2*(np.sqrt(3)*s - c)]) # (3, N)
    p3L =np.array([a*e/2*(-np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s + np.sqrt(3)*(1 + c*c)), -a*e*np.sqrt(3)/2*(-np.sqrt(3)*s - c)]) # (3, N)


    n1 = np.array([-1./2*c*s, 1./2*(1 + c*c), np.sqrt(3)/2*s]) # (3, N)
    kn1= np.dot(k, n1) #(N,)
    n1Hpn1 = np.einsum("ij,ji->i",n1.T, np.einsum("ij,jk", Hp, n1))
    n1Hmn1 = np.einsum("ij,ji->i",n1.T, np.einsum("ij,jk", Hm, n1))

    n2 = 1./4. * np.array([c*s - np.sqrt(3)*(1 + s*s), np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s - 3*c])
    kn2= np.dot(k, n2)
    n2Hpn2 = np.einsum("ij,ji->i",n2.T, np.einsum("ij,jk", Hp, n2))
    n2Hmn2 = np.einsum("ij,ji->i",n2.T, np.einsum("ij,jk", Hm, n2))

    n3 = 1./4*np.array([c*s + np.sqrt(3)*(1 + s*s), -np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s + 3*c])
    kn3= np.dot(k, n3)
    n3Hpn3 = np.einsum("ij,ji->i",n3.T, np.einsum("ij,jk", Hp, n3))
    n3Hmn3 = np.einsum("ij,ji->i",n3.T, np.einsum("ij,jk", Hm, n3))

    

    kp1Lp2L = np.dot(k, (p1L+p2L))
    kp2Lp3L = np.dot(k, (p2L+p3L))
    kp3Lp1L = np.dot(k, (p3L+p1L))
    kp0 = np.dot(k, p0)

    factorcexp0 = np.exp(1j*2.*np.pi*f/C_SI * kp0)
    prefactor = np.pi*f*L/C_SI

    factorcexp12 = np.exp(1j*prefactor * (1.+kp1Lp2L/L))
    factorcexp23 = np.exp(1j*prefactor * (1.+kp2Lp3L/L))
    factorcexp31 = np.exp(1j*prefactor * (1.+kp3Lp1L/L))

    factorsinc12 = np.sinc( (prefactor * (1.-kn3))/np.pi)
    factorsinc21 = np.sinc( (prefactor * (1.+kn3))/np.pi)
    factorsinc23 = np.sinc( (prefactor * (1.-kn1))/np.pi)
    factorsinc32 = np.sinc( (prefactor * (1.+kn1))/np.pi)
    factorsinc31 = np.sinc( (prefactor * (1.-kn2))/np.pi)
    factorsinc13 = np.sinc( (prefactor * (1.+kn2))/np.pi)

    commonfac = 1j*prefactor*factorcexp0
    G12p = commonfac * n3Hpn3 * factorsinc12 * factorcexp12 * np.exp(-1j*phaseRdelay)
    G21p = commonfac * n3Hpn3 * factorsinc21 * factorcexp12 * np.exp(-1j*phaseRdelay)
    G23p = commonfac * n1Hpn1 * factorsinc23 * factorcexp23 * np.exp(-1j*phaseRdelay)
    G32p = commonfac * n1Hpn1 * factorsinc32 * factorcexp23 * np.exp(-1j*phaseRdelay)
    G31p = commonfac * n2Hpn2 * factorsinc31 * factorcexp31 * np.exp(-1j*phaseRdelay)
    G13p = commonfac * n2Hpn2 * factorsinc13 * factorcexp31 * np.exp(-1j*phaseRdelay)

    G12m = commonfac * n3Hmn3 * factorsinc12 * factorcexp12 * np.exp(-1j*phaseRdelay)
    G21m = commonfac * n3Hmn3 * factorsinc21 * factorcexp12 * np.exp(-1j*phaseRdelay)
    G23m = commonfac * n1Hmn1 * factorsinc23 * factorcexp23 * np.exp(-1j*phaseRdelay)
    G32m = commonfac * n1Hmn1 * factorsinc32 * factorcexp23 * np.exp(-1j*phaseRdelay)
    G31m = commonfac * n2Hmn2 * factorsinc31 * factorcexp31 * np.exp(-1j*phaseRdelay)
    G13m = commonfac * n2Hmn2 * factorsinc13 * factorcexp31 * np.exp(-1j*phaseRdelay)

    x = np.pi*f*L/C_SI
    z = np.exp(1j*2.*x)

    factor_convention = 2
    factorAE = 1j*np.sqrt(2)*np.sin(2.*x)*z
    factorT = 2.*np.sqrt(2)*np.sin(2.*x)*np.sin(x)*np.exp(1j*3.*x)

    Araw_p = 0.5 * ( (1.+z)*(G31p + G13p) - G23p - z*G32p - G21p - z*G12p )
    Araw_m = 0.5 * ( (1.+z)*(G31m + G13m) - G23m - z*G32m - G21m - z*G12m )
    Eraw_p = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13p - G31p) + (2.+z)*(G12p - G32p) + (1.+2.*z)*(G21p - G23p) )
    Eraw_m = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13m - G31m) + (2.+z)*(G12m - G32m) + (1.+2.*z)*(G21m - G23m) )
    Traw_p = 1/np.sqrt(6) * (G21p - G12p + G32p - G23p + G13p - G31p)
    Traw_m = 1/np.sqrt(6) * (G21m - G12m + G32m - G23m + G13m - G31m)

    transferL1_p = factor_convention * factorAE * Araw_p
    transferL1_m = factor_convention * factorAE * Araw_m
    transferL2_p = factor_convention * factorAE * Eraw_p
    transferL2_m = factor_convention * factorAE * Eraw_m
    transferL3_p = factor_convention * factorT * Traw_p
    transferL3_m = factor_convention * factorT * Traw_m

    return transferL1_p, transferL1_m, transferL2_p, transferL2_m, transferL3_p, transferL3_m







######################################################################################################################
def get_amplitude_phase(hf): #tested
    #send in hlm not shifted in time
    """Express h_lm as A_lm(f) * exp(i*phase(f)). Note the sign in the exponent."""
    return np.abs(hf.data.data), np.unwrap(np.angle(hf.data.data))



def get_tf_from_phase_dict(hlm, fmax, debug = False):#tested
    #((hlm[2,2].data.length-1)/2)*hlm[2,2].deltaF
    #send in hlm not shifted in time
    tf_dict = {}
    freq_dict = {}
    amp_dict = {}
    phase_dict = {}
    for mode in list(hlm.keys()):
        print(f"\n\tMode = {mode}")

        #get frequency and mode data
        freq, hlm_tmp = np.arange(-fmax, fmax+hlm[mode].deltaF, hlm[mode].deltaF), hlm[mode]
        # freq, hlm_tmp = np.arange(-fmax, fmax+hlm[mode].deltaF, hlm[mode].deltaF), hlm[mode]
        if debug:
            print(f"len(freq) = {len(freq)}, freq[0] = {freq[0]} Hz, freq[-1] = {freq[-1]} Hz, len(hlm) = {hlm_tmp.data.length}")

        #get amplitude and phase
        # phase = np.unwrap(np.angle(hlm[mode].data.data))
        amp, phase = get_amplitude_phase(hlm[mode])
        #compute tf = -1/(2pi) * d(phase)/df
        phase  = phase - phase[0]
        dphi = np.unwrap(np.diff(phase))
        time = -dphi / (2.*np.pi*np.diff(freq))
        #diff reduces len by 1 so artifically increasing it by adding an extra zero at the end
        tmp = np.zeros(len(time)+1)
        tmp[:-1] = time
        time = tmp

        #only focusing on f bins where data exists
        nzidx = np.nonzero(abs(hlm_tmp.data.data))[0]
        if debug:
            print(nzidx)
            print(f"tf[0](after stripping zeros) = {time[kmin]}, tf[-1](after stripping zeros) {time[kmax]}")
        kmin, kmax = nzidx[0], nzidx[-2]
        time[:kmin] = time[kmin]
        time[kmax:] = time[kmax]
        if debug:
            print(f"len(time) = {len(time)}, len(freq) = {len(freq)}")

        #saving data
        tf_dict[mode] = time 
        freq_dict[mode] = freq[::-1]
        amp_dict[mode] = amp
        phase_dict[mode] = phase
    return tf_dict, freq_dict, amp_dict, phase_dict


def get_beta_lamda_psi_terms_Hp(beta, lamda, psi):
    xx_term = (np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi))**2 + \
              (-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))*(np.cos(lamda)*np.cos(psi)*np.sin(beta) + np.sin(lamda)*np.sin(psi))

    xy_term = ((np.cos(psi)*np.sin(beta)*np.sin(lamda)-np.cos(lamda)*np.sin(psi)) * (-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi)) + 
               (np.cos(psi)*np.sin(lamda)-np.cos(lamda)*np.sin(beta)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)* np.sin(psi))) + \
              ((-np.cos(psi)*np.sin(beta)*np.sin(lamda)+ np.cos(lamda)*np.sin(psi))*(np.cos(lamda)*np.cos(psi)*np.sin(beta) + np.sin(lamda)*np.sin(psi)) + 
               (np.cos(psi)*np.sin(lamda)-np.cos(lamda)*np.sin(beta)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi)))
    
    xz_term = (np.cos(beta)*np.sin(psi)*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               np.cos(beta)*np.cos(psi)*(np.cos(lamda)*np.cos(psi)*np.sin(beta) + np.sin(lamda)*np.sin(psi))) + \
              (np.cos(beta)*np.sin(psi)*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) - 
               np.cos(beta)*np.cos(psi)*(-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi)))
                        
    yy_term = (np.cos(psi)*np.sin(beta)*np.sin(lamda) - np.cos(lamda)*np.sin(psi))*(-np.cos(psi)*np.sin(beta)*np.sin(lamda)+ np.cos(lamda)*np.sin(psi)) + \
               (-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))**2
                                    
    yz_term = (np.cos(beta)*np.cos(psi)*(np.cos(psi)*np.sin(beta)*np.sin(lamda) - np.cos(lamda)*np.sin(psi)) + 
               np.cos(beta)*np.sin(psi)*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))) + \
              (-np.cos(beta)*np.cos(psi)*(-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi)) + 
                np.cos(beta)*np.sin(psi)*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi)))
    
    zz_term = (-np.cos(beta)**2 * np.cos(psi)**2 + np.cos(beta)**2 * np.sin(psi)**2)

    return xx_term, xy_term, xz_term, yy_term, yz_term, zz_term

def get_beta_lamda_psi_terms_Hc(beta, lamda, psi):
    xx_term = 2*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))
    
    xy_term = ((-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi))*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               (-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))) + \
              ((-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi))*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               (-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi)))
                                     
    xz_term = (np.cos(beta)*np.cos(psi)*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               np.cos(beta)*np.sin(psi)*(-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi))) + \
              (np.cos(beta)*np.cos(psi)*(np.cos(psi)*np.sin(lamda) - np.cos(lamda)*np.sin(beta)*np.sin(psi)) + 
               np.cos(beta)*np.sin(psi)*(-np.cos(lamda)*np.cos(psi)*np.sin(beta) - np.sin(lamda)*np.sin(psi)))
                         
    yy_term = 2*(-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi))*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))
                                   
    yz_term = (np.cos(beta)*np.sin(psi)*(-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi)) + 
               np.cos(beta)*np.cos(psi)*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi))) + \
              (np.cos(beta)*np.sin(psi)*(-np.cos(psi)*np.sin(beta)*np.sin(lamda) + np.cos(lamda)*np.sin(psi)) + 
               np.cos(beta)*np.cos(psi)*(-np.cos(lamda)*np.cos(psi) - np.sin(beta)*np.sin(lamda)*np.sin(psi)))
    
    zz_term = 2 * np.cos(beta)**2 * np.cos(psi) * np.sin(psi)

    return xx_term, xy_term, xz_term, yy_term, yz_term, zz_term







def Evaluate_Gslr_test_2(tf, f, beta, lamda):
    """This is the main function, takes in tf, f, H, beta and lamda to generate transfer function for a given mode. yslr = Sum_l^r Gslr * hlm.
        Args: 
            tf (numpy.array)= -1/2pi d(phase)/df, 
            f  (numpy.array)= frequency array, 
            H  (numpy.array)= The plus and cross polarization matrices transformed from wave to SSB frame ,
            beta (float)    = ecliptic latitude, 
            lamda (float)   = ecliptic longitude
        Returns:
            Transfer function L1 (numpy.array), Transfer function L2 (numpy.array), Transfer function L3 (numpy.array)
        """
    alpha = omega0*tf
    c, s = np.cos(alpha), np.sin(alpha)
    k = np.array([-np.cos(beta)*np.cos(lamda), -np.cos(beta)*np.sin(lamda), -np.sin(beta)])
    p0 = np.array([a*c, a*s, np.zeros(len(tf))]) # (3, N)
    kR = np.dot(k, p0) # (N,)
    phaseRdelay = 2.*np.pi/C_SI *f*kR #(N,)

    p1L =np.array([-a*e*(1 + s*s), a*e*c*s, -a*e*np.sqrt(3)*c]) # (3, N)
    p2L =np.array([a*e/2*(np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s - np.sqrt(3)*(1 + c*c)),  -a*e*np.sqrt(3)/2*(np.sqrt(3)*s - c)]) # (3, N)
    p3L =np.array([a*e/2*(-np.sqrt(3)*c*s + (1 + s*s)), a*e/2*(-c*s + np.sqrt(3)*(1 + c*c)), -a*e*np.sqrt(3)/2*(-np.sqrt(3)*s - c)]) # (3, N)


    n1 = np.array([-1./2*c*s, 1./2*(1 + c*c), np.sqrt(3)/2*s]) # (3, N)
    kn1= np.dot(k, n1) #(N,)


    n2 = 1./4. * np.array([c*s - np.sqrt(3)*(1 + s*s), np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s - 3*c])
    kn2= np.dot(k, n2)


    n3 = 1./4*np.array([c*s + np.sqrt(3)*(1 + s*s), -np.sqrt(3)*c*s - (1 + c*c), -np.sqrt(3)*s + 3*c])
    kn3= np.dot(k, n3)
    

    kp1Lp2L = np.dot(k, (p1L+p2L))
    kp2Lp3L = np.dot(k, (p2L+p3L))
    kp3Lp1L = np.dot(k, (p3L+p1L))
    kp0 = np.dot(k, p0)

    factorcexp0 = np.exp(1j*2.*np.pi*f/C_SI * kp0)
    prefactor = np.pi*f*L/C_SI

    factorcexp12 = np.exp(1j*prefactor * (1.+kp1Lp2L/L))
    factorcexp23 = np.exp(1j*prefactor * (1.+kp2Lp3L/L))
    factorcexp31 = np.exp(1j*prefactor * (1.+kp3Lp1L/L))

    factorsinc12 = np.sinc( (prefactor * (1.-kn3))/np.pi)
    factorsinc21 = np.sinc( (prefactor * (1.+kn3))/np.pi)
    factorsinc23 = np.sinc( (prefactor * (1.-kn1))/np.pi)
    factorsinc32 = np.sinc( (prefactor * (1.+kn1))/np.pi)
    factorsinc31 = np.sinc( (prefactor * (1.-kn2))/np.pi)
    factorsinc13 = np.sinc( (prefactor * (1.+kn2))/np.pi)

    commonfac = 1j*prefactor*factorcexp0
    G12_term = commonfac * factorcexp12 * np.exp(-1j*phaseRdelay)
    G23_term = commonfac * factorcexp23 * np.exp(-1j*phaseRdelay)
    G31_term = commonfac * factorcexp31 * np.exp(-1j*phaseRdelay)
    
    G12xx = G12_term * n3[0,:]*n3[0,:] * factorsinc12 
    G21xx = G12_term * n3[0,:]*n3[0,:] * factorsinc21 
    G23xx = G23_term * n1[0,:]*n1[0,:] * factorsinc23 
    G32xx = G23_term * n1[0,:]*n1[0,:] * factorsinc32 
    G31xx = G31_term * n2[0,:]*n2[0,:] * factorsinc31
    G13xx = G31_term * n2[0,:]*n2[0,:] * factorsinc13 

    G12xy = G12_term * n3[0,:]*n3[1,:] * factorsinc12 
    G21xy = G12_term * n3[0,:]*n3[1,:] * factorsinc21 
    G23xy = G23_term * n1[0,:]*n1[1,:] * factorsinc23 
    G32xy = G23_term * n1[0,:]*n1[1,:] * factorsinc32 
    G31xy = G31_term * n2[0,:]*n2[1,:] * factorsinc31
    G13xy = G31_term * n2[0,:]*n2[1,:] * factorsinc13 

    G12xz = G12_term * n3[0,:]*n3[2,:] * factorsinc12 
    G21xz = G12_term * n3[0,:]*n3[2,:] * factorsinc21 
    G23xz = G23_term * n1[0,:]*n1[2,:] * factorsinc23 
    G32xz = G23_term * n1[0,:]*n1[2,:] * factorsinc32 
    G31xz = G31_term * n2[0,:]*n2[2,:] * factorsinc31
    G13xz = G31_term * n2[0,:]*n2[2,:] * factorsinc13 

    G12yy = G12_term * n3[1,:]*n3[1,:] * factorsinc12 
    G21yy = G12_term * n3[1,:]*n3[1,:] * factorsinc21 
    G23yy = G23_term * n1[1,:]*n1[1,:] * factorsinc23 
    G32yy = G23_term * n1[1,:]*n1[1,:] * factorsinc32 
    G31yy = G31_term * n2[1,:]*n2[1,:] * factorsinc31
    G13yy = G31_term * n2[1,:]*n2[1,:] * factorsinc13 

    G12yz = G12_term * n3[1,:]*n3[2,:] * factorsinc12 
    G21yz = G12_term * n3[1,:]*n3[2,:] * factorsinc21 
    G23yz = G23_term * n1[1,:]*n1[2,:] * factorsinc23 
    G32yz = G23_term * n1[1,:]*n1[2,:] * factorsinc32 
    G31yz = G31_term * n2[1,:]*n2[2,:] * factorsinc31
    G13yz = G31_term * n2[1,:]*n2[2,:] * factorsinc13 

    G12zz = G12_term * n3[2,:]*n3[2,:] * factorsinc12 
    G21zz = G12_term * n3[2,:]*n3[2,:] * factorsinc21 
    G23zz = G23_term * n1[2,:]*n1[2,:] * factorsinc23 
    G32zz = G23_term * n1[2,:]*n1[2,:] * factorsinc32 
    G31zz = G31_term * n2[2,:]*n2[2,:] * factorsinc31
    G13zz = G31_term * n2[2,:]*n2[2,:] * factorsinc13 

    x = np.pi*f*L/C_SI
    z = np.exp(1j*2.*x)

    factor_convention = 2
    factorAE = 1j*np.sqrt(2)*np.sin(2.*x)*z
    factorT = 2.*np.sqrt(2)*np.sin(2.*x)*np.sin(x)*np.exp(1j*3.*x)

    Araw_xx = 0.5 * ( (1.+z)*(G31xx + G13xx) - G23xx - z*G32xx - G21xx - z*G12xx )
    Araw_xy = 0.5 * ( (1.+z)*(G31xy + G13xy) - G23xy - z*G32xy - G21xy - z*G12xy )
    Araw_xz = 0.5 * ( (1.+z)*(G31xz + G13xz) - G23xz - z*G32xz - G21xz - z*G12xz )
    Araw_yy = 0.5 * ( (1.+z)*(G31yy + G13yy) - G23yy - z*G32yy - G21yy - z*G12yy )
    Araw_yz = 0.5 * ( (1.+z)*(G31yz + G13yz) - G23yz - z*G32yz - G21yz - z*G12yz )
    Araw_zz = 0.5 * ( (1.+z)*(G31zz + G13zz) - G23zz - z*G32zz - G21zz - z*G12zz )

    Eraw_xx = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13xx - G31xx) + (2.+z)*(G12xx - G32xx) + (1.+2.*z)*(G21xx - G23xx) )
    Eraw_xy = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13xy - G31xy) + (2.+z)*(G12xy - G32xy) + (1.+2.*z)*(G21xy - G23xy) )
    Eraw_xz = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13xz - G31xz) + (2.+z)*(G12xz - G32xz) + (1.+2.*z)*(G21xz - G23xz) )
    Eraw_yy = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13yy - G31yy) + (2.+z)*(G12yy - G32yy) + (1.+2.*z)*(G21yy - G23yy) )
    Eraw_yz = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13yz - G31yz) + (2.+z)*(G12yz - G32yz) + (1.+2.*z)*(G21yz - G23yz) )
    Eraw_zz = 0.5*1/np.sqrt(3) * ( (1.-z)*(G13zz - G31zz) + (2.+z)*(G12zz - G32zz) + (1.+2.*z)*(G21zz - G23zz) )

    Traw_xx = 1/np.sqrt(6) * (G21xx - G12xx + G32xx - G23xx + G13xx - G31xx)
    Traw_xy = 1/np.sqrt(6) * (G21xy - G12xy + G32xy - G23xy + G13xy - G31xy)
    Traw_xz = 1/np.sqrt(6) * (G21xz - G12xz + G32xz - G23xz + G13xz - G31xz)
    Traw_yy = 1/np.sqrt(6) * (G21yy - G12yy + G32yy - G23yy + G13yy - G31yy)
    Traw_yz = 1/np.sqrt(6) * (G21yz - G12yz + G32yz - G23yz + G13yz - G31yz)
    Traw_zz = 1/np.sqrt(6) * (G21zz - G12zz + G32zz - G23zz + G13zz - G31zz)

    AE_term = factor_convention * factorAE
    transferL1_xx = AE_term * Araw_xx
    transferL1_xy = AE_term * Araw_xy
    transferL1_xz = AE_term * Araw_xz
    transferL1_yy = AE_term * Araw_yy
    transferL1_yz = AE_term * Araw_yz
    transferL1_zz = AE_term * Araw_zz

    transferL2_xx = AE_term * Eraw_xx
    transferL2_xy = AE_term * Eraw_xy
    transferL2_xz = AE_term * Eraw_xz
    transferL2_yy = AE_term * Eraw_yy
    transferL2_yz = AE_term * Eraw_yz
    transferL2_zz = AE_term * Eraw_zz

    T_term = factor_convention * factorT
    transferL3_xx = T_term *  Traw_xx
    transferL3_xy = T_term * Traw_xy
    transferL3_xz = T_term * Traw_xz
    transferL3_yy = T_term * Traw_yy
    transferL3_yz = T_term * Traw_yz
    transferL3_zz = T_term * Traw_zz

    return np.array([transferL1_xx, transferL1_xy, transferL1_xz, transferL1_yy, transferL1_yz, transferL1_zz]), np.array([transferL2_xx, transferL2_xy, transferL2_xz, transferL2_yy, transferL2_yz, transferL2_zz]), np.array([transferL3_xx, transferL3_xy, transferL3_xz, transferL3_yy, transferL3_yz, transferL3_zz])