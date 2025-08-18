#Defined Classes
# filename  NrWaveform_Catalog_Manager
# classes Simulation
#
# INSTALLATION AND USAGE
#    - interface.py  : for each simulation provides the specific API (see 'execfile).
#    - util_NRWriteMetadataLookupTables.py   : provides table of peak 22 mode values, useful for lookup
#    - util_NRPlotSimulationDimensionless.py : validate simulation raw i/o and cleaning, & raw FFT
#    - util_NRPlotSimulation.py                     : validate scale factors, standard wrapper, and comparison to lalsimulation waveforms
#    
#
# HOW TO INTEGRATE A NEW SIMULATION
#    o copy interface.py and edit
#
#    o test cleaning
#        python util_NRPlotSimulationDimensionless.py --group Sequence-GT-IdenticalTiltingCircular-UnequalMass --param RO3_D10_q2.00_a0.60_oth.045_M120 --clean --psi4-only
#
#
#    o write metadata
#        util_NRWriteSimulationMetadata --group
#
#    o test end to end conversion
#        python util_NRPlotSimulationStrain.py --group Sequence-GT-Aligned-UnequalMass --param RO3_D10_q2.00_a0.60_oth.45_M120
#
#    o test perturbative extraction


# TOOLS
#    - util_NRPlotSimulationDimensionless
#
# BUGS
#    - poorly adapted to irregular sampling rates
#    - LAL mem bugs: MUST CAREFULLY ZERO OUT ALL MEMORY. LAL is supposed to zero the buffer on creation, but it does not do so.

debug_output = False
rosDebug = True


import numpy as np
import os
import RIFT.lalsimutils as lalsimutils
import lalsimulation as lalsim
import lal
import re  # for matching

#import LALHybrid

from scipy.interpolate import interp1d

import pickle

try:
    import romspline
except ImportError:
    print(" - no romspline - ")

DataFourier = '';
if hasattr(lalsimutils,'DataFourier'):
    DataFourier = 'DataFourier'
else:
    DataFourier = 'data_fourier'

dirBaseFiles =os.environ["NR_BASE"] # HOME"] + "/unixhome/PersonalNRArchive/Archives"
dirBaseMetadata = dirBaseFiles
nameMetadataFile = dirBaseMetadata+"/metadata_MasterInternalCache3.pkl"   # python3 is incompatible, build another file
try:
    nameMetadataFile = os.environ["NR_BASE_METADATA"] # Full path. If empty, assume subdirectory of main
    print(" Using user-specified metadata file ", nameMetadataFile)
except:
    print(" Using default metadata directory ", nameMetadataFile)

#default_interpolation_kind = 'quadratic'  # spline interpolation   # very slow! 
default_interpolation_kind = 'linear'  # spline interpolation   # very slow! 

internal_ParametersAvailable ={}
internal_ParametersAreExpressions = {}  # dictionary, use true if param values need to be 'eval'd before using, on i/o
internal_ModesAvailable = {}
internal_DefaultExtractionRadius ={}
internal_SpecialDefaultExtractionRadius ={}
internal_SpecialUseReportedStrainOnly = {}
internal_ExtractionRadiiAvailable={}
internal_GenerateDirectoryName = {}
internal_FilenamesForParameters = {}
internal_JunkRadiationUntilTime = {}
internal_SpecialJunkRadiationUntilTime ={}
internal_JunkRadiationLateTimeCutoffLogPsir= {}
internal_EstimatePeakL2M2Emission = {}
internal_EstimateStartingMOmega0 = {}
internal_SpecialNonuniformInTime = {}
internal_UseNRAR_Filename_Pattern = {}  # change fnameMode pattern used (Cactus vs NRAR)
# For each interesting simulation, store the definitions in a file
# Use 'execfile' to load those defintions now

internal_SignError ={}
internal_BaseWeylName={}
internal_BaseStrainName={}
internal_BaseStrainPrefixHDF={}
internal_WaveformPacking={}  # gz (ascii) or hdf5 (SXS)
internal_WaveformMetadata = {}   # each quantity for 'param' holds 'q', 'Chi1, 'Chi2', 'MF', 'ChiF'. Initial state more critical.
internal_SpinsAlignedQ = {}    # quick and dirty.  Should really use metadata-based test

# A list of the above names (should be auto-generated!), so we can identify what to cache
internal_MasterListOfFieldsPopulatedByAPI = ["ParametersAvailable", "ParametersAreExpressions", "ModesAvailable", "DefaultExtractionRadius", "SpecialDefaultExtractionRadius", "SpecialUseReportedStrainOnly", "ExtractionRadiiAvailable", "GenerateDirectoryName", "FilenamesForParameters", "JunkRadiationUntilTime", "SpecialJunkRadiationUntilTime", "JunkRadiationLateTimeCutoffLogPsir", "EstimatePeakL2M2Emission", "EstimateStartingMOmega0", "SignError", "BaseWeylName", "BaseStrainName", "BaseStrainPrefixHDF", "WaveformPacking", "WaveformMetadata", "SpinsAlignedQ","UseNRAR_Filename_Pattern", "SpecialNonuniformInTime"];

import lal
import h5py
MsunInSec = lal.MSUN_SI*lal.G_SI/lal.C_SI**3


import functools
def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def myzero(arg):
    return 0
def RangeWrap1d(bound, val,fn):
    return fn   # IDEALLY not necessary with modern interp1d
def RangeWrap1dAlt(bound,val, fn):
    """
    RangeWrap1d: Uses np.piecewise to construct a piecewise function which is =fn inside the boundary, and 0 outside.
    SHOULD be syntactic sugar, but depending on the python version the language needed to implement this changes.
    """
#    return (lambda x: fn(x) if  (x>bound[0] and x<bound[1]) else val)
#  WARNING: piecewise is much faster, but will fail for numpy versions less than 1.8-ish :http://stackoverflow.com/questions/20800324/scipy-pchipinterpolator-error-array-cannot-be-safely-cast-to-required-type
#     Unfortunately that is the version LIGO uses on their clusters.
    return (lambda x: np.piecewise( x,        [
                np.logical_and(x> bound[0], x<bound[1]), 
                np.logical_not(np.logical_and(x> bound[0], x<bound[1])) 
                ], [fn, myzero]))
                                    #    return (lambda x: np.where( np.logical_and(x> bound[0], x<bound[1]), fn(x),0))  # vectorized , but does not protect the call

def ModeToString(pair):
    return "l"+str(pair[0])+"_m"+str(pair[1]) 
def ModeToStringNRAR(pair):
    return "l"+str(pair[0])+".m"+str(pair[1]) 

def CreateCompatibleComplexOverlap(hlmf,**kwargs):
    modes = list(hlmf.keys())
    hbase = hlmf[modes[0]]
    deltaF = hbase.deltaF
    fNyq = np.max(lalsimutils.evaluate_fvals(hbase))
    if debug_output:
#        for key, value in kwargs.iteritems():
#            print (key, value)
        print(kwargs)
        print("dF, fNyq, npts = ",deltaF, fNyq, len(hbase.data.data))
    IP = lalsimutils.ComplexOverlap(fNyq=fNyq, deltaF=deltaF, **kwargs)
    return IP

def CreateCompatibleComplexIP(hlmf,**kwargs):
    """
    Creates complex IP (no maximization)
    """
    modes = list(hlmf.keys())
    hbase = hlmf[modes[0]]
    deltaF = hbase.deltaF
    fNyq = np.max(lalsimutils.evaluate_fvals(hbase))
    if debug_output:
#        for key, value in kwargs.iteritems():
#            print (key, value)
        print(kwargs)
        print("dF, fNyq, npts = ",deltaF, fNyq, len(hbase.data.data))
    IP = lalsimutils.ComplexIP(fNyq=fNyq, deltaF=deltaF, **kwargs)
    return IP

class NRError(Exception):
    """Base class for this module"""
    pass
class NRNoSimulation(NRError):
    """Not used"""
    def __init__(self,expr,msg):
        print("No known simulation ", expr, msg)
    pass

def HackRoundTransverseSpin(P):
            P.s1x = int(P.s1x*1e4)/1.e4
            P.s2x = int(P.s2x*1e4)/1.e4
            P.s1y = int(P.s1y*1e4)/1.e4
            P.s2y = int(P.s2y*1e4)/1.e4
#    print " ----- REPRINTING PARAMS ---- "
#    P.print_params()


class WaveformModeCatalog:
    """
    Class containing NR harmonics
    """


    def __init__(self, group ,param, extraction_radius=False, clean_initial_transient=False, clean_final_decay=False, lmax=2, 
                 align_at_peak_l2_m2_emission=False, shift_by_extraction_radius=False,  fix_sign_error=True,mode_list_to_load=[],skip_fourier=False, build_fourier_time_window=1000,clean_with_taper=True,use_internal_interpolation_deltaT=None,build_strain_and_conserve_memory=False,reference_phase_at_peak=None,fix_phase_zero_at_coordinate=False,metadata_only=False,perturbative_extraction=False,perturbative_extraction_full=False,use_provided_strain=False,manual_MOmega0=None,quiet=False):
        self.group  = group
        self.param = param
        self.quantity = "psi4"
        self.deltaToverM =0
        self.ToverM_max =0
        self.fOrbitLower =0.    #  Used to clean results.  Based on the phase of the 22 mode.  This is Momega0 /(2 *pi)
        self.fMinMode ={}
        self.waveform_modes = {}
        self.waveform_modes_uniform_in_time={}
        self.waveform_modes_nonuniform_smallest_timestep = {}
        self.waveform_modes_nonuniform_largest_timestep = {}
        self.waveform_modes[(2,2)] = []
        self.waveform_modes_complex = {}
        self.waveform_modes_complex_interpolated = {}
        self.waveform_modes_complex_interpolated_amplitude = {}
        self.waveform_modes_complex_interpolated_phase = {}
        self.waveform_modes_complex_padded = {}
        self.waveform_modes_fourier = {}
        self.waveform_modes_strain_fourier = {}   # like modes_fourier, but with low-pass filtering to fix the f=0 result
        self.waveform_modes_strain_fourier_AtRadius = {}   #  NO attempt to perform perturbative extrapolation. Enables superior perturbative extrapolation at final loop
        self.waveform_modes_strain = {}             # ifft of the previous result
        self.waveform_modes_strain_interpolated_amplitude = {}
        self.waveform_modes_strain_interpolated_phase = {}

        flag_rom_spline = False  # by default, do not directly attempt to store interpolated_amplitude and interpolated_phase

        # Create structure holding metadata about the spins, etc.
        # By default DIMENSIONLESS
        if (group in list(internal_WaveformMetadata.keys()))  and (param in list(internal_WaveformMetadata[group].keys())):
            self.P = lalsimutils.ChooseWaveformParams()
#            self.P.taper = lalsim.SIM_INSPIRAL_TAPER_NONE
            self.P.taper = lalsim.SIM_INSPIRAL_TAPER_START

            q =  internal_WaveformMetadata[group][param]['q']   # Here, as m1/m2: BE CAREFUL
            chi1 =  internal_WaveformMetadata[group][param]['Chi1']
            chi2 =  internal_WaveformMetadata[group][param]['Chi2']
            self.P.m1  = 100*lal.MSUN_SI*q/(1+q)
            self.P.m2  = 100*lal.MSUN_SI*1./(1.+q)
            self.P.s1x = chi1[0]
            self.P.s1y = chi1[1]
            self.P.s1z = chi1[2]
            self.P.s2x = chi2[0]
            self.P.s2y = chi2[1]
            self.P.s2z = chi2[2]
            print(internal_WaveformMetadata[group][param])
            self.P.eccentricity = internal_WaveformMetadata[group][param]['ecc']
        else:
            print("   INVALID NR COMBO ", group, param)
            if not (group in list(internal_WaveformMetadata.keys())):
                print(" No such group")
            if not (param in list(internal_WaveformMetadata[group].keys())):
                print(" No such param ", param, " in ")
                for p in  sorted(internal_WaveformMetadata[group].keys()):
                    print(p)
            import sys
            sys.exit(0)

        try:
            if internal_WaveformPacking[group][param] == 'romspline' and 'Momega0' in internal_WaveformMetadata[group][param]:
                self.fOrbitLower = internal_WaveformMetadata[group][param]['Momega0'] /(2*np.pi)

            if internal_WaveformPacking[group][param] == 'hdf' and use_provided_strain and 'Momega0' in internal_WaveformMetadata[group][param]:
                self.fOrbitLower = internal_WaveformMetadata[group][param]['Momega0'] /(2*np.pi) 
        except:
            print(" -- Warning: something wrong with internal catalog metadata  for ", group, param, " -- ")

        if metadata_only:
            return None

        # Test if mode available in sequence
        if not (group in list(internal_ParametersAvailable.keys())):  # modify to simply throw error if not true
            raise Exception('Unknown NR simulation', group)
              # **Find the filename associated with that parameter**
        if not ((use_provided_strain and internal_WaveformPacking[group][param]=='hdf') or (internal_WaveformPacking[group][param]=='hdf') and internal_SpecialUseReportedStrainOnly[group][param]   ) and not internal_WaveformPacking[group][param]=='romspline' :
         # First loop: Create all the basic mode data
         for mode in internal_ModesAvailable[group]:
              if mode[0]<= lmax:   
                if rosDebug:
                      print("  --------------------- ")
                if extraction_radius:
                    r_str = "{0:.2f}".format(extraction_radius)
                else:
                    if param in internal_SpecialDefaultExtractionRadius[group]:  # Modify logic to allow zero
                        r_str = "{0:.2f}".format(internal_SpecialDefaultExtractionRadius[group][param])
                    else:
                        r_str = "{0:.2f}".format(internal_DefaultExtractionRadius[group])
                r_val = eval(str(r_str))
                if internal_WaveformPacking[group][param] == 'hdf':  # cornell-caltech, ALWAYS at infinity
                    # Cornell extraction procedure: http://arxiv.org/pdf/0905.3177v2.pdf
                    # Cornell caltech code unit data needs M
                    mtot = internal_WaveformMetadata[group][param]["M"];  # could not run unless this is available
                    dir_name = "R0" + str(int(r_val))+".dir"
                    # Cornell caltech data is NOT necessarily uniform in time
                    # Cornell caltech also uses code units
                    file =  h5py.File(dirBaseFiles+"/"+group+internal_FilenamesForParameters[group][param]+'/'+    internal_BaseWeylName[group][param]+'.h5', 'r')        
                    rvals_ok =  [int(k.replace('R','').replace(".dir",'')) for k,q in list(file.items())]
                    if not r_val in rvals_ok:
                        print("   FAILURE : No such extraction radius ", r_val, " in ", rvals_ok)
                        import sys
                        sys.exit(0)
                    self.waveform_modes[mode] =  np.array(file[dir_name]['Y_'+ModeToString(mode)+'.dat'])  # Force type convert
                    self.waveform_modes[mode]*= 1./mtot   # scale t/m, r/m h into t/M, r/M h
                    rval_arr = r_val*np.ones(len(self.waveform_modes[mode]))
                    try:
                        # Wasteful: load areal radius each time
                        fR = h5py.File(dirBaseFiles+"/"+group+internal_FilenamesForParameters[group][param]+'/'+ "rPsi4_FiniteRadii_CodeUnits.h5")
                        rval_arr = np.array(fR[dir_name]["ArealRadius.dat"])[:,1]
                        if rosDebug:
                            print("  Import: SXS HDF5: Initial radius ", rval_arr[0])
                    except:
                        rval_arr = r_val*np.ones(len(self.waveform_modes[mode]))
                    if rosDebug:
                        print("   Import: SXS HDF5: Radius comparison ", rval_arr[0]/r_val)
                    self.waveform_modes[mode][:,1] *= rval_arr/r_val
                    self.waveform_modes[mode][:,2] *= rval_arr/r_val
                    r_val = rval_arr[0]

                    # Shift by extraction radius 
                    if shift_by_extraction_radius:
                        self.waveform_modes[mode][:,0] -= r_val

                    self.waveform_modes_uniform_in_time[mode] = False
                    npts = len(self.waveform_modes[mode]); tvals = self.waveform_modes[mode][:,0]
                    self.waveform_modes_nonuniform_smallest_timestep[mode] = np.min( tvals[1:npts] - tvals[:npts-1]) # generally not stable
                    self.waveform_modes_nonuniform_largest_timestep[mode] = np.max( tvals[1:npts] - tvals[:npts-1]) # generally not stable
#                    if rosDebug:
#                        # Confirm sampling rate is uniform in time; if not, interpolate and resample!
#                        print " Timesteps uniform? Probably not", np.max( tvals[1:npts] - tvals[:npts-1] ), np.min( tvals[1:npts] - tvals[:npts-1])
                else:
                    fnameMode = None
                    if not internal_UseNRAR_Filename_Pattern[group]:
                        fnameMode= dirBaseFiles+"/"+group+internal_FilenamesForParameters[group][param]+"/"+internal_BaseWeylName[group][param]+"_"+ModeToString(mode)+"_r"+r_str+".asc"
                    else:
                        fnameMode= dirBaseFiles+"/"+group+internal_FilenamesForParameters[group][param]+"/"+internal_BaseWeylName[group][param]+"r5.l5."+ModeToStringNRAR(mode)
                    if not os.path.isfile(fnameMode):
                        fnameMode = fnameMode + ".gz"
                    if not os.path.isfile(fnameMode):
                        print(" NO MODE FILE ", fnameMode)
                        import sys
                        sys.exit(0)
                    if rosDebug:
                        print(" mode file ", mode, fnameMode)
                    self.waveform_modes[mode] = np.loadtxt(fnameMode)
                    
                    self.waveform_modes_nonuniform_smallest_timestep[mode] = self.waveform_modes[mode][1,0]-self.waveform_modes[mode][0,0]  # uniform in time
                    self.waveform_modes_nonuniform_largest_timestep[mode] = self.waveform_modes[mode][1,0]-self.waveform_modes[mode][0,0]  # uniform in time
                    self.waveform_modes_uniform_in_time[mode] =True
                    if group in internal_SpecialNonuniformInTime:
                        if param in internal_SpecialNonuniformInTime[group]:
                            if not quiet:
                                print(" Using nonuniform spacing for ", group, param, mode)
                            self.waveform_modes_uniform_in_time[mode] =False
                    # scale by extraction radius. ONLY DO THIS IF WE AE NOT NRAR
                    r_val = float(r_str)
                    if internal_UseNRAR_Filename_Pattern[group]:
                        r_val = 1
                    for indx in np.arange(len(self.waveform_modes[mode])):
                       if r_val > 0:   # Enable extrapolation to infinity for r_val=0.
                        self.waveform_modes[mode][indx,1]*= r_val
                        self.waveform_modes[mode][indx,2]*= r_val
                        # shift times by extraction radius
                        if shift_by_extraction_radius:
                            self.waveform_modes[mode][indx,0] -= r_val

                        # Fix sign error (-1)^m for m odd, if this flag is set.
                        if internal_SignError[group][param] and (-1)**int(mode[1])==-1:
                            self.waveform_modes[mode][:,1]*= -1.
                            self.waveform_modes[mode][:,2]*= -1.
                # Quick and dirty way to identify index of max time. Uses regular spacing of raw data
                # Deletion-based.  Only works because time array values are kept.
                if clean_initial_transient:
                    tmin = self.waveform_modes[mode][0,0]
                    dt = self.waveform_modes[mode][1,0] -tmin
                    # *delete* data before cleaning time, entirely
                    tToClean = 0
                    if internal_SpecialJunkRadiationUntilTime[group] and (param in list(internal_SpecialJunkRadiationUntilTime[group].keys())):
                        tToClean = internal_SpecialJunkRadiationUntilTime[group][param] 
                    else:
                        tToClean =internal_JunkRadiationUntilTime[group] 
                    indx_ok = np.real(self.waveform_modes[mode][:,0]) > tToClean
#                    nbad = int( (tToClean - tmin)/dt)         # Assumes regular sampling up to this point; should use bool indexing
#                    self.waveform_modes[mode] = self.waveform_modes[mode][nbad:]      # Note: array sizes are DIFFERENT for different modes in general!
                    self.waveform_modes[mode] = self.waveform_modes[mode][indx_ok]      # Note: array sizes are DIFFERENT for different modes in general!
#                    print "   Post junk removal of psi4, length  ", mode, len(self.waveform_modes[mode])

                    # taper the start 
                    if clean_with_taper:
                        tmaxHere = 0
                        if param in internal_EstimatePeakL2M2Emission[group]:
                            tmaxHere = internal_EstimatePeakL2M2Emission[group][param]
                        elif (2,2) in self.waveform_modes:  # backup plan
                            print(" FAILING TO POPULATE REFERENCE TIME CORRECTLY ")
                            tmaxHere = self.waveform_modes[(2,2)][np.argmax(self.waveform_modes[(2,2)])]
                        else:  # emergency case - SHOULD NEVER HAPPEN
                            print(" FAILING TO POPULATE REFERENCE TIME CORRECTLY (v2) ")
                            tmaxHere = self.waveform_modes[mode][np.argmax(np.abs(self.waveform_modes[mode]))]
                        tTaper = np.max([5, 0.05* (tmaxHere-tmin)])  # 10% of length! very aggressive!
                        nTaper = int(tTaper/dt)
#                        factorTaper = (np.exp( -4.*(nTaper-1.0*np.arange(nTaper+1))/nTaper))
                        hoft_window = lal.CreateTukeyREAL8Window(nTaper*2, 0.8)
                        factorTaper = hoft_window.data.data[0:nTaper]
#                        print factorTaper
                        self.waveform_modes[mode][0:nTaper,1]*=factorTaper
                        self.waveform_modes[mode][0:nTaper,2]*=factorTaper

                if clean_final_decay:
                    tmin = self.waveform_modes[mode][0,0]
                    dt = self.waveform_modes[mode][1,0] -tmin
                    # boolTooEarly =  self.waveform_modes[mode][:,0]> internal_EstimatePeakL2M2Emission[group][param]
                    # boolTooSmall =  np.log10(np.sqrt(self.waveform_modes[mode][:,1]**2+self.waveform_modes[mode][:,2]**2)) < internal_JunkRadiationLateTimeCutoffLogPsir[wfHere]
                    # indx = np.arange(len(boolTooEarly))[np.nonzero(  np.logical_and(np.logical_not(boolTooEarly),   boolTooSmall) )]
                    # if len(indx)>1:
                    #     indx = indx[0]
                    # else:
                    #     indx = len(boolTooEarly)
                    # print mode, indx
                    # This loop is incredibly inefficient and can be vectorized.
                    # Drop everything below our cutoff, AFTER this time. But be CAREFUL with m=0 mode
                    indx = 0
                    startflag = False
                    stopflag = False                    
                    while (indx < len(self.waveform_modes[mode]) and not stopflag) :
                        if not startflag and self.waveform_modes[mode][indx,0]> internal_EstimatePeakL2M2Emission[group][param]:
                            startflag=True
                        if startflag  and np.log10(np.sqrt(self.waveform_modes[mode][indx,1]**2+self.waveform_modes[mode][indx,2]**2)) < internal_JunkRadiationLateTimeCutoffLogPsir[group]:
                            if mode[1]:
                                stopflag = True
                            else: # m=0 case is special, can dump to zero many times...so keep a constant 100 M after
                               if self.waveform_modes[mode][indx,0]> internal_EstimatePeakL2M2Emission[group][param]+100:
                                   stopflag=True
                        indx+=1
#                    next( indx for indx in np.arange(len(self.waveform_modes[mode])) if self.waveform_modes[mode][indx,0] > internal_EstimatePeakL2M2Emission[group][param] and np.log10(np.sqrt(self.waveform_modes[mode][indx,1]**2+self.waveform_modes[mode][indx,2]**2)) < internal_JunkRadiationLateTimeCutoffLogPsir[wfHere])
                    indx = min(int(indx),len(self.waveform_modes[mode])-1)
                    self.waveform_modes[mode] =self.waveform_modes[mode][:indx]

                    # Taper the final decay: 10 M (but should be tied to dur
                    # Time *and frequency* which is tapered depends on the mode!
                    tTaper = min(15, 0.1*(-internal_EstimatePeakL2M2Emission[group][param]+self.waveform_modes[mode][-1,0]))  # taper by 15 M or 5% of ringdown, whichever is less
                    ntaper = int(tTaper/dt)
                    if rosDebug:
                        print("   Import: tapering by ", nTaper, " on top of ", len(self.waveform_modes[mode]))
                    
#                    tmp = self.waveform_modes[mode][-ntaper:-1,1]
#                    print tmp.shape
                    if ntaper > 0 and ntaper < len(self.waveform_modes[mode]):
                        vectaper= 0.5 - 0.5*np.cos(np.pi* (1-np.arange(ntaper)/(1.*ntaper)))
                        self.waveform_modes[mode][-ntaper:,1]*=vectaper
                        self.waveform_modes[mode][-ntaper:,2]*=vectaper

                # timeshift to origin
                if align_at_peak_l2_m2_emission:
                    self.waveform_modes[mode][:,0] -= internal_EstimatePeakL2M2Emission[group][param]   # This should undo any extraction radius alignment applied earlier, if used CONSISTENTLY
                if rosDebug:
                    print("  Import: Loaded ", group, param, mode, "; length = ", len(self.waveform_modes[mode]), " processed interval = ", self.waveform_modes[mode][0,0], self.waveform_modes[mode][-1,0], " sampling interval at start ",  self.waveform_modes[mode][1,0]-self.waveform_modes[mode][0,0], " and smallest interval = ", self.waveform_modes_nonuniform_smallest_timestep[mode])

                # Either store in a different field OR reformat
                self.waveform_modes_complex[mode] = np.array([self.waveform_modes[mode][:,0], self.waveform_modes[mode][:,1]+1.j*self.waveform_modes[mode][:,2]]).T

                # Create an interpolating function for the complex amplitude and phase
                datAmp = np.abs(self.waveform_modes_complex[mode][:,1])
                datPhase = lalsimutils.unwind_phase(np.angle(self.waveform_modes_complex[mode][:,1]))    #needs to be unwound to be continuous
                datT = np.real(self.waveform_modes_complex[mode][:,0])
                # Compute starting frequency for this mode (used for FFT)
                # Index offset corresponding to 10 M
                dt = np.real(self.waveform_modes_complex[mode][1,0]-self.waveform_modes_complex[mode][0,0])
                nOffsetForPhase = int(20./dt)  # ad-hoc offset based on uniform sampling, and avoid taper
                if not clean_initial_transient:
                    nOffsetForPhase += int(internal_JunkRadiationUntilTime[group]/dt)
                dtStride = 10.0 # use time \Delta t = 10 M to measure the initial frequency. Note we fit a straight line.
                nStride = np.max([5, int(dtStride/dt)])
                # EXPERIMENTAL OPTION (not to be enabled on main branch) FOR TESTING: fit a straight line? 
#                print np.array([np.real(datT[nOffsetForPhase:nOffsetForPhase+nStride]),np.abs(datPhase[nOffsetForPhase:nOffsetForPhase+nStride])]).T
                z = np.polyfit(np.real(datT[nOffsetForPhase:nOffsetForPhase+nStride]),np.abs(datPhase[nOffsetForPhase:nOffsetForPhase+nStride]),1)
                f0 = (z[-2])/(2*np.pi)  # phase derivative. 
		#print " Proposed stride size ", nStride, " Note \Delta t, dt, proposal (two-sided)", dtStride, dt, dtStride/dt
		#nStride =5
#                self.fMinMode[mode] = np.abs((datPhase[nOffsetForPhase+nStride]-datPhase[nOffsetForPhase])/(2*np.pi*(datT[nOffsetForPhase+nStride]-datT[nOffsetForPhase]))) # for FFT. Modified to work with nonuniform sampling
                self.fMinMode[mode] = f0
                if not quiet:
                    print("  Starting frequency : mode, estimate_new, estimate_standard", mode,f0, self.fMinMode[mode])
                # WARNING: this is really only reliable for non-junky resolved modes.  We will clean this lower frequency later.
                if mode[0]==2 and mode[1]==2 :
                    self.fOrbitLower  = 0.5*np.abs(self.fMinMode[mode])  # define the orbital frequency from the 22 mode. This is used later.  Make sure it is positive
                if not (quiet) and not (manual_MOmega0 is None):
                    print("   : Enforcing MOmega0 cutoff for ", mode , " changing fminM =", self.fMinMode[mode], " to ", (manual_MOmega0/(2*np.pi))*np.max([np.abs(mode[1]),1]), " = |m| * MOmega0/2pi ")
                    self.fMinMode[mode] = (manual_MOmega0/(2*np.pi))*np.max([np.abs(mode[1]),1])  # force low frequency cutoff tied to MOmega0


                # Interpolate. (Works even for nonuniform time sampling)
                if True: #not build_strain_and_conserve_memory:
                    # this is gravy - not critical. BUT it is critical for nonuniform
                    self.waveform_modes_complex_interpolated_amplitude[mode] = RangeWrap1d([np.min(self.waveform_modes_complex[mode][:,0]), np.max(self.waveform_modes_complex[mode][:,0])], 0, interp1d( np.real(self.waveform_modes_complex[mode][:,0]).astype(float), datAmp,kind=default_interpolation_kind, fill_value=0.,bounds_error=False))
                    self.waveform_modes_complex_interpolated_phase[mode] = RangeWrap1d([np.min(self.waveform_modes_complex[mode][:,0]), np.max(self.waveform_modes_complex[mode][:,0])], 0,  interp1d( np.real(self.waveform_modes_complex[mode][:,0]).astype(float), datPhase,kind=default_interpolation_kind, fill_value=0.,bounds_error=False))

         # Intermediate loop: phase align all the modes
         #     - phase align to the peak of the 22 mode.  This CHANGES the definition of the \phi angle, but is useful for testing waveforms 
         if reference_phase_at_peak:
            indxmax = np.argmax(np.abs(self.waveform_modes_complex[(2,2)][:,1]))
            phasemax = np.real(np.angle(np.abs(self.waveform_modes_complex[(2,2)][indxmax,1])))
            for mode in list(self.waveform_modes_complex.keys()):
                        self.waveform_modes_complex[mode][:,1]*= np.exp(-1j*phasemax)
                        self.waveform_modes[mode][:,1] = np.real(self.waveform_modes_complex[mode][:,1])
                        self.waveform_modes[mode][:,2] = np.imag(self.waveform_modes_complex[mode][:,1])
                        datAmp = np.abs(self.waveform_modes_complex[mode][:,1])
                        datPhase = lalsimutils.unwind_phase(np.angle(self.waveform_modes_complex[mode][:,1]))    #needs to be unwound to be continuo
                        self.waveform_modes_complex_interpolated_amplitude[mode] = RangeWrap1d([np.min(self.waveform_modes_complex[mode][:,0]), np.max(self.waveform_modes_complex[mode][:,0])], 0,  interp1d( np.real(self.waveform_modes_complex[mode][:,0]).astype(float), datAmp,fill_value=0.,bounds_error=False))
                        self.waveform_modes_complex_interpolated_phase[mode] = RangeWrap1d([np.min(self.waveform_modes_complex[mode][:,0]), np.max(self.waveform_modes_complex[mode][:,0])], 0,  interp1d( np.real(self.waveform_modes_complex[mode][:,0]).astype(float), datPhase,fill_value=0.,bounds_error=False))
                    

         if skip_fourier:
            self.deltaToverM = self.waveform_modes_complex[(2,2)][1,0] - self.waveform_modes_complex[(2,2)][0,0] # end of loop
            if not self.ToverM_max >0:
                self.ToverM_max = np.max(np.real(self.waveform_modes_complex[(2,2)][:,0]))
            if align_at_peak_l2_m2_emission:
                self.ToverM_max += np.real(internal_EstimatePeakL2M2Emission[group][param])  # correct for undoing this
            return None

        # SANITY: Repair fMinMode statements
        #  - Needed: Some modes start barely-resolved, in  junk time.  Our estimation technique will fai.
        #               If fminMode is too far from the physical range, we will amplify junk or suppress important content
        #  - NOTE: A similar constraint is applied below
         if rosDebug:
             print("  --------------------- ")
         for mode in self.fMinMode:
            if mode[1]==0:
                if not quiet:
                    print(" m=0 mode test, comparing " , self.fMinMode[mode], " to ", self.fOrbitLower)
                freq_ratio_m0 = np.abs(self.fMinMode[mode])/np.abs(self.fOrbitLower)
                if freq_ratio_m0 <0.8 or freq_ratio_m0 > 2.2 :  # Force override on m=0 frequency. Don't let it be too boffset
                    if not quiet:
                        print("  Starting frequency: override for m=0 mode ", mode)
                    self.fMinMode[mode] = self.fOrbitLower
                continue   # prevent fMinMode from being set to zero
            elif self.P.SoftAlignedQ() and np.abs(mode[1]) < mode[0]:   # If we have a subdominant mode for a nonprecessing binary!  Note this introduces an asymmetry between how we treat nonprecessing and precessing binaries
                print("   ... nonprecessing binary, symmetry overrride on fmin  for ", mode)
                self.fMinMode[mode] = np.abs(mode[1])* self.fOrbitLower
            elif  (manual_MOmega0 is None) and  not (0.3/np.abs(mode[1])< np.abs(self.fMinMode[mode]/self.fOrbitLower/mode[1]) <1.5*mode[0]/np.abs(mode[1])):
                if not quiet:
                    print(" Initial mode frequency not resolved, overriding  for ", mode, " as ", self.fOrbitLower*np.abs(mode[1]))
                self.fMinMode[mode] = self.fOrbitLower*np.abs(mode[1])   # Strictly only appropriate for a nonprecessing source, but not a terrible guess, and won't cause catastrophes
         if not quiet:
             print(" Mode minimum frequencies", self.fMinMode, " versus ")
             print(" Expected mode minmum frequencies  |m|* forb ", [(mode, self.fOrbitLower*np.abs(mode[1])) for mode in self.fMinMode])
         if rosDebug:
             print("  --------------------- ")


         # Second loop: Create all the fourier modes
         #   - define time range over which all modes are available: time  [-T,T] where T is large enough to encompass the signal+pad
         #     the time zero of the waveform maps to the t=0 point in the waveform, which MAY be the peak of the signal (if aligned)
         #   - define array of points over which all modes will be padded to cover.
         #   - pad all the modes to that length (storing them)
         tStart = float(np.real(self.waveform_modes_complex[(2,2)][0,0]))
         tEnd  = float(np.real(self.waveform_modes_complex[(2,2)][-1,0]))
         dt      = self.waveform_modes_nonuniform_largest_timestep[(2,2)] #float(self.waveform_modes_complex[(2,2)][1,0]-tStart)  # this may not be the smallest time sample.  Use a large stride to make it more efficient for SXS memory-intensive work
#         if align_at_peak_l2_m2_emission:
#            nMax = np.argmin(np.abs(self.waveform_modes_complex[(2,2)][:,0]))  # argument of best time
#            tMax = self.waveform_modes_complex[(2,2)][nMax,0]
#         else:
#            nMax = int(internal_EstimatePeakL2M2Emission[group][param]/dt)
         T =2*np.max([np.abs(tStart), np.abs(tEnd),build_fourier_time_window])+100 # add a pad of 100M at minimum
         nOut = int(2*T/dt)
         if nOut %2:
             nOut +=1   # FORCE EVEN
         tvalsOut = (np.arange(nOut)-nOut/2)*dt   # puts t=0 at the center of the interval
#        print self.waveform_modes_complex.keys()
#        print self.waveform_modes_complex_interpolated_amplitude.keys()
        mykeys = []
        if self.waveform_modes_complex:
            mykeys=list(self.waveform_modes_complex.keys())
        if ((use_provided_strain and internal_WaveformPacking[group][param]=='hdf') or (internal_SpecialUseReportedStrainOnly[group][param] and internal_WaveformPacking[group][param]=='hdf')) or (internal_WaveformPacking[group][param]=='romspline'):
            mykeys = internal_ModesAvailable[group]
            if rosDebug:
                print(" MANUAL STRAIN: ", group, param, mykeys)
            for mode in mykeys:
                self.fMinMode[mode] =0  # Need to declare for sanity, PROVIDED STRAIN ONLY. SHOULD set to metadata
        for mode in mykeys:
           if mode[0]<=lmax:
                if not quiet:
                    print(" ---------------  ")
                if mode[1] is 0:
                    self.fMinMode[mode] = self.fOrbitLower   # The m=0 mode doesn't really have *zero* frequency content.
#                    print " Lower frequency fM ", self.fOrbitLower, " based on n=", nOffsetForPhase, " or t = ", self.waveform_modes_complex[mode][nOffsetForPhase,0]
                else:
                 if not use_provided_strain:
                    # Make some ssanitizing of the mode frequencies, to account for junk radiation, so I don't accidentally set the fMinMode to some insanely high value
                    if not quiet:
                        print(" m * fOrb, fMinMode ", mode, self.fOrbitLower*mode[1], np.abs(self.fMinMode[mode]))
                    self.fMinMode[mode] = np.max([self.fOrbitLower*np.abs(mode[1]), np.abs(self.fMinMode[mode])])
                    if not quiet:
                        print("  FT: Fourier transforming ", mode,  "fmin = ", self.fMinMode[mode], " so omegaM/m min = ", self.fMinMode[mode]*2*np.pi/np.abs(int(mode[1])))
                if not ((use_provided_strain and internal_WaveformPacking[group][param]=='hdf') or (internal_SpecialUseReportedStrainOnly[group][param] and internal_WaveformPacking[group][param]=='hdf')) and not (internal_WaveformPacking[group][param]=='romspline'):
                  if self.waveform_modes_uniform_in_time[mode]:
                    if rosDebug:
                        print(" Import: Mode uniformly sampled ", mode)
                    # Create a fourier transformed mode [compare to : RawLoadWaveformDataModesFT
                    #  - zero pad so t=0 is at the CENTER (=simple convention), but need to be CONSISTENT ACROSS MODES
                    #    so the padding needs to be based on TIMES rather than INDICES
                    tvals =self.waveform_modes_complex[mode][:,0]
                    dat = self.waveform_modes_complex[mode][:,1]
                    nOrig = len(dat)
                    nStart =int( (T + np.real(tvals[0]))/dt)  
                    nEnd = int( (T - np.real(tvals[-1]))/dt)
                    nEnd = nOut - nOrig-nStart   #  force size agreement as desired.
                    if not quiet:
                        print("Tapering length check :", mode, nOrig, nOut, nEnd)
                    dat=np.concatenate((np.zeros(nStart),dat,np.zeros(nEnd)))
                    if rosDebug:
                        print(" Padded length", len(dat), " over time range ", tvalsOut[0], tvalsOut[-1], " with time interval ", tvalsOut[1]-tvalsOut[0],dt)
                    # Store the padded quantities for the complex h(t) for comparison and validation as needed.
                    self.waveform_modes_complex_padded[mode] = np.array([tvalsOut,dat]).T
                  else:
                    # sampling is NOT uniform in time. Pad via the interpolating function RangeWrap1d call (slow!)
                    # tvalsOut already uses the smaller sampling rate to fully populate the desired interval uniformly
                    # Allow end user to specify a smaller deltaT. JUST for hdf5
                    if True: #use_internal_interpolation_deltaT:
                        dtHere = np.sqrt(self.waveform_modes_nonuniform_smallest_timestep[mode]*self.waveform_modes_nonuniform_largest_timestep[mode])
                        nOut = int(2*T/dtHere)
                        if nOut %2:
                            nOut +=1   # FORCE EVEN
                        tvalsOut = (np.arange(nOut)-nOut/2)*dtHere   # puts t=0 at the center of the interval
                        if rosDebug:
                            print(" FT: Import resampling: using different timesampling rate ", dtHere, " = geometric mean of smallest and largest ")

                    self.waveform_modes_complex_padded[mode] =np.array([tvalsOut,self.waveform_modes_complex_interpolated_amplitude[mode](tvalsOut)*np.exp(1.j*self.waveform_modes_complex_interpolated_phase[mode](tvalsOut))]).T
                  # self-consistency check: peak occurs at the same time
                  if rosDebug:
                    print("   FT: Padded, interpolated mode size: ", self.waveform_modes_complex_padded[mode].shape)
                    nMax = np.argmax(np.abs(self.waveform_modes_complex[mode][:,1]))
                    nMaxPad = np.argmax(np.abs(self.waveform_modes_complex_padded[mode][:,1]))
                    tMaxPad = self.waveform_modes_complex_padded[mode][nMaxPad,0]
                    nMax = np.argmax(np.abs(self.waveform_modes_complex[mode][:,1]))
                    tMax = self.waveform_modes_complex[mode][nMax,0]
                    print("   FT: Pad, interpolation sanity check : tmax raw=", tMax, " padded = ", tMaxPad, " value = ", np.abs(self.waveform_modes_complex_padded[mode][nMaxPad,1]), " ", np.abs(self.waveform_modes_complex[mode][nMax,1]))


                  self.waveform_modes_fourier[mode] = DataFourierNumpy(self.waveform_modes_complex_padded[mode])

                  # Create a time domain strain
                  #   - WARNING: This will need regularization to be well-behaved.
                  #  - WARNING: too many transposes. Use one working buffer to save time?
                  fvals = self.waveform_modes_fourier[mode][:,0]
#                df = np.abs(fvals[1]-fvals[0])
                  facToReduce = 1.  # After discussions with Jim, this tends to make a time-domain h(t) that better reflects the real waveform.  This should be tuned by Jake. Jim default is half of the orbital frequency.
                  if rosDebug:
                      print("  FT: Psi4->strain: fM cutoff used  ", mode, self.fMinMode[mode])   # Provenance checking
                  datStrainFourier= -1.0*self.waveform_modes_fourier[mode][:,1]/(2*np.pi*np.maximum(np.abs(fvals), np.abs(self.fMinMode[mode])/facToReduce))**2

                  self.waveform_modes_strain_fourier_AtRadius[mode]=np.array([fvals, datStrainFourier]).T
                  if perturbative_extraction and r_val <=70:
                    print(" +++++ Warning: Perturbative extraction not enabled for r<70 +++++ ")
                  if perturbative_extraction and  r_val>70 and not perturbative_extraction_full:
                    print("  Extraction:  Using perturbative extraction  (l,m,r)= ",mode,  r_val)
                    # http://adsabs.harvard.edu/abs/2015PhRvD..91j4022N, Eq. 29 (but without the 'a' term)
                    # ASSUME M=1. If not, some overall error
                    iomega_regular = 1j*2*np.pi*np.sign(fvals+1e-9)*np.maximum(np.abs(fvals), np.abs(self.fMinMode[mode]))
                    lval = mode[0]
                    mval = mode[1]
                    r_val_pt = r_val
                    if not (internal_WaveformPacking[group][param]=='hdf'):
                        r_val_pt = r_val*(1+1/(2.*r_val) )*(1+1./(2.*r_val)) # conformal: r+M (Jim suggestion); see after Eq. 30 in http://arxiv.org/pdf/1503.00718v2.pdf 
                    else:
                        r_val_pt =r_val   # hdf is already areal, ideally
                    a_pt = 0.
                    if "ChiFMagnitude" in internal_WaveformMetadata[group][param] :
                        print("  Extraction: Performing perturbative extrapolation using a, version 0 (no terms included in a) ")
                        a_pt = internal_WaveformMetadata[group][param]["ChiFMagnitude"]
                        if not  (internal_WaveformPacking[group][param]=='hdf'):
                            r_val_pt = r_val*(1+(1+a_pt)/(2.*r_val) )*(1+(1.-a_pt)/(2.*r_val)) # conformal: r+M (Jim suggestion); see after Eq. 30 in http://arxiv.org/pdf/1503.00718v2.pdf 
#                        print a_pt, r_val_pt
                    datStrainFourier *= (r_val_pt)/(r_val) * (1-2./r_val_pt)*(np.ones(len(datStrainFourier)) - (lval-1)*(lval+2)/(2.*r_val_pt)/iomega_regular + (lval-1)*(lval+2)*(lval*lval + lval-4)/(8*r_val_pt*r_val_pt)/iomega_regular/iomega_regular)  #

                  self.waveform_modes_strain_fourier[mode]=np.array([fvals, datStrainFourier]).T
                  self.waveform_modes_strain[mode] = DataInverseFourierNumpy(self.waveform_modes_strain_fourier[mode])

                # USE PROVIDED STRAIN:  override previous work (!) entirely
                # WARNING: Timing issues!
                if (use_provided_strain and internal_WaveformPacking[group][param]=='hdf') or (internal_SpecialUseReportedStrainOnly[group][param] and internal_WaveformPacking[group][param]=='hdf'):
                    if rosDebug:
                        print(" : STRAIN OVERRIDE ENABLED : mode  ", mode)
                        print("   Use internal HDF5 strain: reading strain from ", internal_BaseStrainName[group][param])
#                    print "   Warning: no cleaning services are provided! "
                    if param not in internal_EstimatePeakL2M2Emission[group]:
                        print(" No peak time information available, setting default ")
                        internal_EstimatePeakL2M2Emission[group][param] = 0
                    f =  h5py.File(dirBaseFiles+"/"+group+internal_FilenamesForParameters[group][param]+"/"+internal_BaseStrainName[group][param]+'.h5', 'r')
                    tmp =  np.array(f[internal_BaseStrainPrefixHDF[group][param]]['Y_'+ModeToString(mode)+'.dat'])  # Force type convert
                    if 'encoding' in list(internal_WaveformMetadata[group][param].keys()):
                        amp_phase_orig = tmp
                        my_re = np.real(tmp[:,1]*np.exp(tmp[:,2]))
                        my_imag = np.imag(tmp[:,1]*np.exp(tmp[:,2]))
                        tmp[:,1] = my_re
                        tmp[:,2] = my_imag
                    self.waveform_modes_strain[mode]  = np.array([tmp[:,0], tmp[:,1]+1.j*tmp[:,2]]).T
                    # timeshift to origin. Note timeshift MAY BE INCONSISTENT with the reported value

                    # Delete junk radiation. SXS doesn't always do this.
                    if clean_initial_transient:
                        tmin = self.waveform_modes_strain[mode][0,0]
                        dt = self.waveform_modes_strain[mode][1,0] -tmin
                        tToClean = 0
                        if internal_SpecialJunkRadiationUntilTime[group] and (param in list(internal_SpecialJunkRadiationUntilTime[group].keys())):
                            tToClean = internal_SpecialJunkRadiationUntilTime[group][param]
                        else:
                            tToClean =internal_JunkRadiationUntilTime[group] 
                        indx_ok = np.real(self.waveform_modes_strain[mode][:,0]) > tToClean
                        self.waveform_modes_strain[mode] = self.waveform_modes_strain[mode][indx_ok]      # Note: array sizes are DIFFERENT for different modes in general!
                        print(" Cleaned length ", len(self.waveform_modes_strain[mode]))

                    if align_at_peak_l2_m2_emission:
                        self.waveform_modes_strain[mode][:,0] -= internal_EstimatePeakL2M2Emission[group][param]
                    tmin = np.min(self.waveform_modes_strain[mode][:,0])


                    # taper the start  FOR NONUNIFORM TIME SAMPLING
                    if clean_with_taper:
                        tTaper = np.max([5, 0.05* (internal_EstimatePeakL2M2Emission[group][param]-tmin)])  # 10% of length! very aggressive!
                        indx_end = self.waveform_modes_strain[mode][:,0] < tmin+tTaper
                        vectaper = np.where(indx_end, 0.5 - 0.5*np.cos(np.pi*( (self.waveform_modes_strain[mode][:,0] - tmin)/tTaper)),1)
                        self.waveform_modes_strain[mode][:,1]*=vectaper
                    # taper the end (IMPORTANT: discontinuous)
#                    indxMax = np.argmax(np.abs(self.waveform_modes_strain[mode][:,1]))
                    t_max = np.max(self.waveform_modes_strain[mode][:,0]) #self.waveform_modes_strain[mode][indxMax,0]                    
                    # Taper the final decay: 10 M (but should be tied to dur
                    # Time *and frequency* which is tapered depends on the mode!
                    tTaper = min(15, 0.1*(-t_max+self.waveform_modes_strain[mode][-1,0]))  # taper by 15 M or 10% of post-peak time, whichever is less
                    indx_end = self.waveform_modes_strain[mode][:,0] > t_max  - tTaper
                    vectaper = np.where(indx_end, 0.5 - 0.5*np.cos(np.pi*(1 - self.waveform_modes_strain[mode][:,0]/t_max)),1)
                    self.waveform_modes_strain[mode][:,1]*=vectaper


                if (use_provided_strain and internal_WaveformPacking[group][param]=='romspline') or (internal_SpecialUseReportedStrainOnly[group][param] and internal_WaveformPacking[group][param]=='romspline'):
                    flag_rom_spline = True
                    if rosDebug:
                        print(" : STRAIN OVERRIDE ENABLED : mode  ", mode)
                        print("   Use internal romspline waveform ", internal_BaseStrainName[group][param])
                        print(" DIRECTLY saving access to interpolated_amplitude and interpolated_strain ")
                    fname = dirBaseFiles+"/"+group+internal_FilenamesForParameters[group][param]+'/'+    internal_BaseStrainName[group][param] # ROM
                    if not quiet:
                        print(" Opening LVC git annex file ", fname)
                    f = h5py.File(fname,'r');
                    tmin = np.min(f["amp_l2_m2"]['X'])
                    tmax = np.max(f["amp_l2_m2"]['X'])
                    f.close()

                    t_ref = 0
                    if align_at_peak_l2_m2_emission:
                        t_ref= internal_EstimatePeakL2M2Emission[group][param]

                    s_amp = romspline.ReducedOrderSpline()
                    s_phase = romspline.ReducedOrderSpline()
                    try:
                        s_amp.read(fname, 'amp_l'+str(mode[0])+"_m"+str(mode[1]))
                        s_phase.read(fname, 'phase_l'+str(mode[0])+"_m"+str(mode[1]))
                    except:
                        print('  ==> Mode not present in romspline ', mode, " <== ")
                        continue  # mode not present
                    # Tapering makes no real difference.  I will provide it to make sure this is not an issue.
                    def fnTaperHere(x,tmax=tmax,tmin=tmin):
                        tTaperStart= np.max([5, 0.05* (tmax-tmin)])
                        return np.piecewise(x , [x<tmin+tTaperStart, x>tmax-2], 
                                     [(lambda z, tm=tmin,dt=tTaperStart: 0.5-0.5*np.cos(np.pi* (z-tm)/dt)),
                                      (lambda z, tm=tmax: 0.5-0.5*np.cos(np.pi* (tm-z)/2)),
                                       lambda z: 1])
                    self.waveform_modes_strain_interpolated_amplitude[mode] = compose(RangeWrap1dAlt([tmin,tmax], 0,lambda x,s=s_amp,t=fnTaperHere: t(x)*s.eval(x) ), lambda x,ts=t_ref: x+ts)
                    self.waveform_modes_strain_interpolated_phase[mode] = compose(RangeWrap1dAlt([tmin,tmax], 0,lambda x,s=s_phase,t=fnTaperHere: s.eval(x) ), lambda x,ts=t_ref: x+ts)  # do not need to taper phase!

                    # FOR FILLING STANDARD CODE: use a dense sampling to populate the usual array, for plotting purposes
                    tvals = np.linspace(tmin,tmax, int((tmax-tmin)/0.02))   # for T =10k, this is a pretty large array
                    self.waveform_modes_strain[mode] = np.zeros( (len(tvals),2),dtype=np.complex64)
                    self.waveform_modes_strain[mode][:,0] = tvals - t_ref
                    self.waveform_modes_strain[mode][:,1]  = s_amp.eval(tvals)*np.exp(1j*s_phase.eval(tvals)) * fnTaperHere(tvals,tmax=tmax,tmin=tmin)
                    # timeshift to origin. Note timeshift MAY BE INCONSISTENT with the reported value

                    # Create fMinMode (used in hybridization applications)
                    datT=tvals; datPhase = s_phase.eval(tvals)
                    dt = tvals[1]-tvals[0]
                    nOffsetForPhase = int(20./dt)  # ad-hoc offset based on uniform sampling, and avoid taper
                    nOffsetForPhase += int(internal_JunkRadiationUntilTime[group]/dt) # skip initial transients
                    dtStride = 10.0 # use time \Delta t = 10 M to measure the initial frequency. Note we fit a straight line.
                    nStride = np.max([5, int(dtStride/dt)])
                    z = np.polyfit(np.real(datT[nOffsetForPhase:nOffsetForPhase+nStride]),np.abs(datPhase[nOffsetForPhase:nOffsetForPhase+nStride]),1)
                    f0 = (z[-2])/(2*np.pi)  # phase derivative. 
                    self.fMinMode[mode] = np.abs(f0)
                    if mode[0]==2 and mode[1]==2 :
                        self.ToverM_max = np.abs(tmax-tmin)
                        self.fOrbitLower  = 0.5*np.abs(self.fMinMode[mode])  # define the orbital frequency from the 22 mode. 
                        if rosDebug:
                            print(" Setting orbital frequency for 'romspline' ", mode, self.fOrbitLower)
                            print(" Setting duration ", tmin, tmax, self.ToverM_max)


                if not (flag_rom_spline):  
                    # Create interpolating functions for the time domain strain
                    #   - used to generate h(t) for physical sources
                    #   - benefits from all the cleaning and data conditioning done above
                    # For models which *provide* amplitude and phase, should directly use these structures
                    datAmp = np.abs(self.waveform_modes_strain[mode][:,1]).astype(float)
                    datPhase = lalsimutils.unwind_phase(np.angle(self.waveform_modes_strain[mode][:,1]))    #needs to be unwound to be continuous
                    # Tricky to use original structures, because we have cleaned the waveform of transients. Retain original code for now
                    self.waveform_modes_strain_interpolated_amplitude[mode] = RangeWrap1d([np.min(self.waveform_modes_strain[mode][:,0]), np.max(self.waveform_modes_strain[mode][:,0])], 0,  interp1d(np.real(self.waveform_modes_strain[mode][:,0]).astype(float), datAmp,fill_value=0,bounds_error=False))
                    self.waveform_modes_strain_interpolated_phase[mode] = RangeWrap1d([np.min(self.waveform_modes_strain[mode][:,0]), np.max(self.waveform_modes_strain[mode][:,0])], 0,  interp1d(np.real(self.waveform_modes_strain[mode][:,0]).astype(float), datPhase,fill_value=0,bounds_error=False))

                    # report on quantities entering into interpolation
                    if rosDebug:
                        print("   Sanity check: min, max strain amplitude (dimensionless) = ", np.min(datAmp), np.max(datAmp))

                # Delete intermediate quantities. 
                # Minimal needed:
                #    - waveform_modes_complex
                #    - waveform_modes_strain_interpolated_amplitude
                #    - waveform_modes_strain_interpolated_phase
                if build_strain_and_conserve_memory:
                    self.waveform_modes[mode] = None
                    self.waveform_modes_complex_interpolated[mode] = None
                    self.waveform_modes_complex_padded[mode] = None
                    self.waveform_modes_fourier[mode] = None

        self.deltaToverM = self.waveform_modes_strain[(2,2)][1,0] - self.waveform_modes_strain[(2,2)][0,0] # end of loop
        if not self.ToverM_max >0:
          # This odd piece of code tries to work around padding, and work reliably if we have strain or psi4
          if self.waveform_modes_complex :
            self.ToverM_max = self.waveform_modes_complex[(2,2)][-1,0]   # usually filled, unless we use reported strain
          else:
            self.ToverM_max = self.waveform_modes_strain[(2,2)][-1,0]
          if align_at_peak_l2_m2_emission:
                self.ToverM_max += internal_EstimatePeakL2M2Emission[group][param]  # correct for undoing this


        # Special: Add terms for perturbative strain due to mode mixing
        # MUST be done by special postprocessing step
        # WARNING: Far too easy to add complete junk here. Must take great care.
        # NOT YET READY TO USE ON PRODUCTION SCALE: DISABLED
        if perturbative_extraction_full and not (use_provided_strain): # perturbative_extraction and not use_provided_strain:
            if not quiet:
                print(" Performing perturbative extrapolation with full mode mixing")
            a_pt = 0.
            r_val_pt = r_val
            if "ChiFMagnitude" in internal_WaveformMetadata[group][param] :
                a_pt = internal_WaveformMetadata[group][param]["ChiFMagnitude"]
                if   not(internal_WaveformPacking[group][param]=='hdf'):
                    r_val_pt = r_val*(1+(1+a_pt)/(2.*r_val) )*(1+(1.-a_pt)/(2.*r_val)) # conformal: r+M (Jim suggestion); see 
            for mode in self.waveform_modes_strain_fourier_AtRadius:
                lval = mode[0]
                mval = mode[1]
                modePlus = (mode[0]+1,mode[1])
                modeMinus = (mode[0]-1,mode[1])
                iomega_regular = 1j*2*np.pi*np.sign(fvals+1e-9)*np.maximum(np.abs(fvals), self.fMinMode[mode])
                datStrainFourier = np.zeros(len( self.waveform_modes_strain_fourier_AtRadius[mode]),dtype=np.complex128)
                datStrainFourier +=  self.waveform_modes_strain_fourier_AtRadius[mode][:,1] # make sure NOT to pass by reference
                datStrainFourierPlus =np.zeros(len(self.waveform_modes_strain_fourier_AtRadius[mode]),dtype=np.complex128) 
                datStrainFourierMinus =np.zeros(len(self.waveform_modes_strain_fourier_AtRadius[mode]),dtype=np.complex128) 
                # These expressions already include the 1/(i omega)^2 prefactor terms, note
                if modePlus in list(self.waveform_modes_strain_fourier_AtRadius.keys()):
                    datStrainFourierPlus += self.waveform_modes_strain_fourier_AtRadius[modePlus][:,1]
                if modeMinus in list(self.waveform_modes_strain_fourier_AtRadius.keys()):
                    datStrainFourierMinus += self.waveform_modes_strain_fourier_AtRadius[modeMinus][:,1]

                datStrainFourier *= (r_val_pt)/(r_val) * (1-2./r_val_pt)*(np.ones(len(datStrainFourier)) - (lval-1)*(lval+2)/(2.*r_val_pt)/iomega_regular + (lval-1)*(lval+2)*(lval*lval + lval-4)/(8*r_val_pt*r_val_pt)/iomega_regular/iomega_regular)
                datStrainFourier += datStrainFourierPlus* 2*1j*a_pt/((lval+1)*(lval+1))*np.sqrt( (lval+3)*(lval-1)*(lval+mval+1)*(lval-mval+1)/( (2*lval+1)*(2*lval+3)))*(iomega_regular - lval*(lval+3)/r_val_pt) 
                datStrainFourier += -1*datStrainFourierMinus* 2*1j*a_pt/((lval*lval))*np.sqrt( (lval+2)*(lval-2)*(lval+mval)*(lval-mval)/( (2*lval+1)*(2*lval-1)))*(iomega_regular - (lval-2)*(lval+1)/r_val_pt)
                self.waveform_modes_strain_fourier[mode]=np.array([fvals, datStrainFourier]).T
                self.waveform_modes_strain[mode] = DataInverseFourierNumpy(self.waveform_modes_strain_fourier[mode])


    def complex_hoft(self,  force_T=False, deltaT=1./16384, time_over_M_zero=0.,sgn=-1,no_memory=False,**kwargs):
        hlmT = self.hlmoft( force_T, deltaT,time_over_M_zero,**kwargs)
        npts = hlmT[(2,2)].data.length
        wfmTS = lal.CreateCOMPLEX16TimeSeries("Psi4", lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
        wfmTS.data.data[:] = 0   # SHOULD NOT BE NECESARY, but the creation operator doesn't robustly clean memory
        wfmTS.epoch = hlmT[(2,2)].epoch
        for mode in list(hlmT.keys()):
            # Skip memory mode. ONLY USE THIS IF YOU KNOW WHAT YOU ARE DOING.
            if no_memory and mode[1]==0:
                continue
            if rosDebug:
                print(mode, np.max(hlmT[mode].data.data), " running max ",  np.max(np.abs(wfmTS.data.data)))
            # PROBLEM: Be careful with interpretation. The incl and phiref terms are NOT tied to L.
            # BEWARE: phiref is treated as an orbital reference phase, not GW ref phase: exp[-i m (phi -phiref)]
            wfmTS.data.data +=  hlmT[mode].data.data*lal.SpinWeightedSphericalHarmonic(self.P.incl,-1.0*self.P.phiref,-2, int(mode[0]),int(mode[1])) #*np.exp(2*sgn*1j*self.P.psi)
        return wfmTS
    def complex_hoff(self, force_T=False):
        htC  = self.complex_hoft( force_T=force_T,deltaT= self.P.deltaT)
        TDlen = int(1./self.P.deltaF * 1./self.P.deltaT)
        assert TDlen == htC.data.length
        hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)",
                                                htC.epoch, htC.f0, 1./htC.deltaT/htC.data.length, lal.HertzUnit, 
                                                htC.data.length)
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(htC.data.length,0)
        lal.COMPLEX16TimeFreqFFT(hf, htC, fwdplan)
        return hf
    def real_hoft(self,Fp=None, Fc=None,no_memory=False,**kwargs):
        """
        Returns the real-valued h(t) that would be produced in a single instrument.
        Translates epoch as needed.
        Based on 'hoft' in lalsimutils.py
        WARNING: complex_hoft already applies psi!
        """
        # Create complex timessereis
        htC = self.complex_hoft(force_T=1./self.P.deltaF, deltaT= self.P.deltaT,no_memory=no_memory,**kwargs)  # note P.tref is NOT used in the low-level code
        TDlen  = htC.data.length
        if rosDebug:
            print("Size sanity check ", TDlen, 1/(self.P.deltaF*self.P.deltaT))
            print(" Raw complex magnitude , ", np.max(htC.data.data))
            
        # Create working buffers to extract data from it -- wasteful.
        hp = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            self.P.deltaT, lal.DimensionlessUnit, TDlen)
        hc = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            self.P.deltaT, lal.DimensionlessUnit, TDlen)
        hT = lal.CreateREAL8TimeSeries("h(t)", htC.epoch, 0.,
            self.P.deltaT, lal.DimensionlessUnit, TDlen)
        # Copy data components over
        hp.data.data = np.real(htC.data.data)
        hc.data.data = -1*np.imag(htC.data.data)  # hc  = hp - i hx  with standard convention (sgn=-1).
        # transform as in lalsimutils.hoft
        if Fp!=None and Fc!=None:
            hp.data.data *= Fp
            hc.data.data *= Fc
            hp = lal.AddREAL8TimeSeries(hp, hc)
            hoft = hp
        elif self.P.radec==False:
            fp = lalsimutils.Fplus(self.P.theta, self.P.phi, self.P.psi)   # complex_hoft already applies psi
            fc = lalsimutils.Fcross(self.P.theta, self.P.phi, self.P.psi) # complex_hoft already applies psi
            hp.data.data *= fp
            hc.data.data *= fc
            hp.data.data  = lal.AddREAL8TimeSeries(hp, hc)
            hoft = hp
        else:
            # Note epoch must be applied FIRST, to make sure the correct event time is being used to construct the modulation functions
            hp.epoch = hp.epoch + self.P.tref
            hc.epoch = hc.epoch + self.P.tref
            if rosDebug:
                print(" Real h(t) before detector weighting, ", np.max(hp.data.data), np.max(hc.data.data))
            hoft = lalsim.SimDetectorStrainREAL8TimeSeries(hp, hc,    # beware, this MAY alter the series length??
                self.P.phi, self.P.theta,  self.P.psi,     # Important!  complex_hoft already applies psi
                lalsim.DetectorPrefixToLALDetector(str(self.P.detector)))
            hoft = lal.CutREAL8TimeSeries(hoft, 0, hp.data.length)       # force same length as before??
            if rosDebug:
                print("Size before and after detector weighting " , hp.data.length, hoft.data.length)
        if rosDebug:
            print(" Real h_{IFO}(t) generated, pre-taper : max strain =", np.max(hoft.data.data))
        if self.P.taper != lalsimutils.lsu_TAPER_NONE: # Taper if requested
            lalsim.SimInspiralREAL8WaveTaper(hoft.data, self.P.taper)
        if self.P.deltaF is not None:
            TDlen = int(1./self.P.deltaF * 1./self.P.deltaT)
            if rosDebug:
                print("Size sanity check 2 ", int(1./self.P.deltaF * 1./self.P.deltaT), hoft.data.length)
            assert TDlen >= hoft.data.length
            npts = hoft.data.length
            hoft = lal.ResizeREAL8TimeSeries(hoft, 0, TDlen)
            # Zero out the last few data elements -- NOT always reliable for all architectures; SHOULD NOT BE NECESSARY
            hoft.data.data[npts:TDlen] = 0

        if rosDebug:
            print(" Real h_{IFO}(t) generated : max strain =", np.max(hoft.data.data))
        return hoft

    def non_herm_hoff(self):
        """
        Returns the 2-sided h(f) associated with the real-valued h(t) seen in a real instrument.
        Translates epoch as needed.
        Based on 'non_herm_hoff' in lalsimutils.py
        """
        htR = self.real_hoft() # Generate real-valued TD waveform, including detector response
        if self.P.deltaF == None: # h(t) was not zero-padded, so do it now
            TDlen = nextPow2(htR.data.length)
            htR = lal.ResizeREAL8TimeSeries(htR, 0, TDlen)
        else: # Check zero-padding was done to expected length
            TDlen = int(1./self.P.deltaF * 1./self.P.deltaT)
            assert TDlen == htR.data.length
        fwdplan=lal.CreateForwardCOMPLEX16FFTPlan(htR.data.length,0)
        htC = lal.CreateCOMPLEX16TimeSeries("hoft", htR.epoch, htR.f0,
            htR.deltaT, htR.sampleUnits, htR.data.length)
        # copy h(t) into a COMPLEX16 array which happens to be purely real
        htC.data.data[:htR.data.length] = htR.data.data
#        for i in range(htR.data.length):
#            htC.data.data[i] = htR.data.data[i]
        hf = lal.CreateCOMPLEX16FrequencySeries("Template h(f)",
            htR.epoch, htR.f0, 1./htR.deltaT/htR.data.length, lal.HertzUnit, 
            htR.data.length)
        lal.COMPLEX16TimeFreqFFT(hf, htC, fwdplan)
        return hf


    def estimateFminHz(self,fmin=10.):
        return 2*self.fOrbitLower/(MsunInSec*(self.P.m1+self.P.m2)/lal.MSUN_SI)

    def defineHybridFrequencyHz(self,fmin=10.):
        """
        defineHybridFrequencyHz : Used to define the hybridization frequency used henceforth
        Note the true hybridization condition is DIFFERENT and is based on a *time*.  [I throw out times inside the tapering region, etc]
        """
        return 1.3*self.estimateFminHz(fmin=fmin)  

    def estimateDurationSec(self,fmin=10.):
        """
        estimateDuration uses fmin*M from the (2,2) mode to estimate the waveform duration from the *well-posed*
        part.  By default it uses the *entire* waveform duration.
        CURRENTLY DOES NOT IMPLEMENT frequency-dependent duration
        CURRENTLY DOES NOT correct for tapering impact on duration, which reduces the de facto duration a bit
        """
        return float(MsunInSec*((self.P.m1+self.P.m2)/(lal.MSUN_SI))*(self.ToverM_max)) # self.deltaToverM*(self.len(self.waveform_modes[(2,2)])

    def hlmoft(self,  force_T=False, deltaT=1./16384, time_over_M_zero=0.,hybrid_time=None,hybrid_use=False,hybrid_method='taper_add',hybrid_frequency=None,verbose=False):
        """
        hlmoft uses stored interpolated values for hlm(t) generated via the standard cleaning process, scaling them 
        to physical units for use in injection code.

        If the time window is sufficiently short, the result is NOT tapered (!!) -- no additional tapering is applied
        """

        # Pick the hybridization frequency. If I am specifying it by hand (!), overriding default, do so...
        internal_hybrid_frequency = hybrid_frequency
        if not internal_hybrid_frequency:
            internal_hybrid_frequency = self.defineHybridFrequencyHz(self)
            

        if rosDebug:
            print("URGENT add taper code to hlmoft -- not necessarily tapered at start ! ")
        hybrid_time_viaf = hybrid_time
        hlmT ={}
        # Define units, note d and m are measured in seconds in this routine
        m_total_s = MsunInSec*(self.P.m1+self.P.m2)/lal.MSUN_SI
        distance_s = self.P.dist/lal.C_SI  

        # Create a suitable set of time samples.  Zero pad to 2^n samples.
        # FIXME: this assumes the internal data structure uses uniform time sampling. Caltech codes break this assumptin
        ToverM_max =self.ToverM_max  #len(self.waveform_modes_complex[(2,2)])*self.deltaToverM   # an estimate of the duration of the signal. Note different modes may have differnt duration
        T_estimated = ToverM_max * m_total_s
#        print " estimated time window ", T_estimated
        npts=0
        if not force_T:
            npts_estimated = int(T_estimated/deltaT)
#            print " Estimated length: ",npts_estimated, T_estimated
            npts = lalsimutils.nextPow2(npts_estimated)
        else:
            npts = int(force_T/deltaT)
            if rosDebug:
                print(" Forcing length T=", force_T, " length ", npts,  " corresponding to dimensionless range ", ToverM_max)
        tvals = (np.arange(npts)-npts/2)*deltaT   # Use CENTERED time to make sure I handle CENTERED NR signal (usual)
        if rosDebug:
            print(" time range being sampled ", [min(tvals),max(tvals)], " corresponding to dimensionless range", [min(tvals)/m_total_s,max(tvals)/m_total_s])

        # Loop over all modes in the system
        for mode in list(self.waveform_modes_strain_interpolated_amplitude.keys()):

#            print " hlmoft: interpolating mode ", mode, " onto time sample of size ", len(tvals)
            # For each mode, interpolate, then rescale
#            amp_vals = 1./distance_s * np.array(map(self.waveform_modes_strain_interpolated_amplitude[mode], tvals/(m_total_s)))
#            phase_vals = np.array(map(self.waveform_modes_strain_interpolated_phase[mode], tvals/(m_total_s)))
            amp_vals = m_total_s/distance_s * self.waveform_modes_strain_interpolated_amplitude[mode](tvals/(m_total_s))  # vectorized interpolation with piecewise
            phase_vals = self.waveform_modes_strain_interpolated_phase[mode]( tvals/(m_total_s))
            phase_vals = lalsimutils.unwind_phase(phase_vals)

            if rosDebug:
                print("  Mode ", mode, " physical strain max, indx,", np.max(amp_vals), np.argmax(amp_vals))
            
            # Copy into a new LIGO time series object
            wfmTS = lal.CreateCOMPLEX16TimeSeries("Psi4", lal.LIGOTimeGPS(0.), 0., deltaT, lal.DimensionlessUnit, npts)
            wfmTS.data.data =amp_vals*np.exp(1j*phase_vals)

            # Set the epoch for the time series correctly: should have peak near center of series by construction
            # note all have the same length
            wfmTS.epoch = -deltaT*wfmTS.data.length/2

            # Store the resulting mode
            hlmT[mode] = wfmTS

            # if the 22 mode, use to identify the natural frequency.  Can afford to be a bit sloppy, since only used for hybridization
            if hybrid_use and hybrid_time == None and mode[0]==2 and mode[1]==2 :
                if internal_hybrid_frequency: # should ALWAYS be true
                    datFreqReduced = (-1)* (np.roll(phase_vals,-1) - phase_vals)/deltaT/(2*np.pi)   # Hopefully this is smooth and monotonic
                    indx_ok =  np.logical_and(np.abs(datFreqReduced) < internal_hybrid_frequency*1.001, tvals < 0.1)  # peak occurs at center. ASSUMES ALIGNMENT
                    # Reject times that are inside the taper region -- for very short waveforms, this could be a problem
                    t_after_taper  = -0.9 * self.estimateDurationSec()
                    if rosDebug:
                        print(" ------ Time where taper is applied (impacts hybridization) " ,t_after_taper)
                    indx_ok = np.logical_and(indx_ok, tvals > t_after_taper)
                    tvals_ok = tvals[indx_ok]
                    f_ok = datFreqReduced[indx_ok]
                    hybrid_time_viaf = tvals_ok[np.argmax(f_ok)]   # rely on high sampling rate. No interpolation!
                    if verbose:
                        print(" NR catalog: revising hybridization time from 22 mode  to ", hybrid_time_viaf, " given frequency ", internal_hybrid_frequency)


        # hybridize
        if hybrid_use:
            my_hybrid_time = hybrid_time_viaf # should ALWAYS be set by above logic
            HackRoundTransverseSpin(self.P) # Hack, sub-optimal
            if my_hybrid_time == None:  # should NEVER happen
                my_hybrid_time = -0.5*self.estimateDurationSec()  # note fmin is not used. Note this is VERY conservative
            if verbose:
                print("  hybridization performed for ", self.group, self.param, " at time ", my_hybrid_time)
            self.P.deltaT = deltaT # sanity
            # HACK: round digits, so I can get a spin-aligned approximant if I need it
            hlmT_hybrid = LALHybrid.hybridize_modes(hlmT,self.P,hybrid_time_start=my_hybrid_time,hybrid_method=hybrid_method)
            return hlmT_hybrid
        else:
            if verbose:
                print(" ------ NO HYBRIDIZATION PERFORMED AT LOW LEVEL (=automatic) for ", self.group, self.param, "----- ")
            return hlmT

    def hlmoff(self, force_T=False, deltaT=1./16384, time_over_M_zero=0.,hybrid_time=None,hybrid_frequency=None,hybrid_use=False,hybrid_method='taper_add',verbose=False):
        """
        hlmoff takes fourier transforms of LAL timeseries generated from hlmoft.
        All modes have physical units, appropriate to a physical signal.
        """
        hlmF ={}
        hlmT = self.hlmoft(force_T=force_T,deltaT=deltaT,time_over_M_zero=time_over_M_zero,hybrid_time=hybrid_time,hybrid_frequency=hybrid_frequency,hybrid_use=hybrid_use,hybrid_method=hybrid_method,verbose=verbose)
        for mode in list(hlmT.keys()):
            wfmTS=hlmT[mode]
            # Take the fourier transform
            wfmFD = getattr(lalsimutils,DataFourier)(wfmTS)  # this creates a new f series for *each* call.
            # Store the resulting mode
            hlmF[mode] = wfmFD
        return hlmF

    def conj_hlmoff(self, force_T=False, deltaT=1./16384, time_over_M_zero=0.,hybrid_time=None,hybrid_use=False,hybrid_method='taper_add'):
        """
        conj_hlmoff takes fourier transforms of LAL timeseries generated from hlmoft, but after complex conjugation.
        All modes have physical units, appropriate to a physical signal.
        """
        hlmF ={}
        hlmT = self.hlmoft(force_T=force_T,deltaT=deltaT,time_over_M_zero=time_over_M_zero,hybrid_time=hybrid_time,hybrid_use=hybrid_use,hybrid_method=hybrid_method)
        for mode in list(hlmT.keys()):
            wfmTS=hlmT[mode]
            wfmTS.data.data = np.conj(wfmTS.data.data)  # complex conjugate
            # Take the fourier transform
            wfmFD = getattr(lalsimutils,DataFourier)(wfmTS)  # this creates a new f series for *each* call.
            # Store the resulting mode
            hlmF[mode] = wfmFD
        return hlmF


class WaveformMode:
    """
    Class representing a dimensionless timeseries X(\tau=t/M)
    """

###
### FFT syntatic sugar
###
def DataFourierNumpy(wfComplex):    # assume (n,2) size array of [tvals, g(t)]; return fvals, tilde(g). 
    # FFT
    # Unroll -- it will save me time later if we are continuous
    T = wfComplex[-1,0] - wfComplex[0,0]
    n = len(wfComplex)
    dt = T/n
    gtilde = np.fft.fft(wfComplex[:,1])*dt   
    # Option 1: Raw pairing, not caring about continuity in the frequency array.
#    fvals = 1./T* np.array([ k if  k<=n/2 else k-n for k in np.arange(n)]) 
    # Option 2: Unroll
    gtilde = np.roll(gtilde, n/2)
    fvals = np.zeros(n)
    if n%2 ==0: 
#        print "  FT: Even  (should always get this)"
        fvals = 1./T*(np.arange(n) - n/2.+1)     # consistency with DataFourier: \int dt e(i2\pi f t), note REVERSING
    else:
 #       print " FT : Odd (SHOULD NEVER HAPPEN)"
        fvals = 1./T*(np.arange(n) - n/2.+1)     # consistency with DataFourier: \int dt e(i2\pi f t), note REVERSING
    fvals = 1./T*(np.arange(n) - n/2.+1)     # consistency with DataFourier: \int dt e(i2\pi f t), note REVERSING
    wfComplexF = np.array([fvals,gtilde[::-1]]).T
    return wfComplexF

def DataInverseFourierNumpy(wfComplex):    # assume (n,2) size array of [fvals, gtilde(f)]; return tvals, g
#    print "NAN check ", wfComplex[np.isnan(wfComplex[:,1]),1]
    df = wfComplex[1,0] - wfComplex[0,0]
    n = len(wfComplex)
    T = 1./df
    dt = T/n
    # Undo the roll, then perform the FFT
    datReversed = wfComplex[:,1][::-1]
    g = np.fft.ifft(np.roll(datReversed, -n/2))*n*df # undo the reverse and roll
    # Assign correct time values.  
    #    - Note the zero of time is now centered in the array -- we don't carry a time reference with us.
    tvals = dt*(np.arange(n) -n/2+1)
    wfComplexT = np.array([tvals,g]).T
    return wfComplexT

###
### Mode syntactic sugar
###
def RawGetModePeakTime(wfMode):   # assumed applied to complex data sequence
    nmax = np.argmax(np.abs(wfMode[:,1]))
    return np.real(wfMode[nmax][0])


###
### Lookup tools (full catalog infrastructure). User must specify parameters via 'P' structure
###
def NRSimulationLookup(params_to_test,valid_groups=None):

        # params_to_test = {}
        # params_to_test['q'] = q
        # params_to_test['s1z'] = chi1z
        # params_to_test['s2z'] = chi2z

        global internal_ParametersAvailable

        good_sims = []
        group_list = []
        if valid_groups:
            group_list =list(set(valid_groups).intersection(set(internal_ParametersAvailable.keys())))
        else:
            group_list = list(internal_ParametersAvailable.keys())
        for group in  group_list:
                # print group, nrwf.internal_ParametersAvailable[group]                                                                           \
                                                                                                                                                   
                for param in internal_ParametersAvailable[group]:
      #  print group, param, param in nrwf.internal_WaveformMetadata[group]                                                                       \
                      # print nrwf.internal_ParametersAvailable[group], param                                                                     
                        wfP = WaveformModeCatalog(group,param,metadata_only=True)
                        P_NR = wfP.P
                        add_simulation = True
                        for key in list(params_to_test.keys()):
                                if hasattr(P_NR,'extract_param'):
                                    val = P_NR.extract_param(key)   # standard format
                                else:
                                    val = getattr(P_NR, key)  # NR variant
                                # SPECIAL TEST IF PARAMETER IS MASS RATIO: very near mass ratio possible, can screw up
                                if key =='q' and val >1:
                                    val = 1./val
                                if key =='q' and params_to_test[key]>1:
                                    params_to_test[key] = 1./params_to_test[key]
                                if np.abs(val - params_to_test[key]) > 1e-3:
#                                    print "Bad Sim:  ", group, param, key, val, params_to_test[key], np.abs(val - params_to_test[key])              
                                    add_simulation=False
                                if not add_simulation:
                                    break
#				print add_simulation
                        if add_simulation:
#                                print "Good Sim:  ", group, param
                                good_sims.append((group,param))

        return good_sims
def NRSimulationLookupRestrict(params_to_test,valid_groups=None):
    # Similar to before except (a) acts on raw NR simulation metadata and (b)  the parameter provided is an upper bound. Used specifically for Momega0
    # Also, tests on NR metadata

        global internal_ParametersAvailable

        good_sims = []
        group_list = []
        if valid_groups:
            group_list =list(set(valid_groups).intersection(set(internal_ParametersAvailable.keys())))
        else:
            group_list = list(internal_ParametersAvailable.keys())
        for group in  group_list:
                for param in internal_ParametersAvailable[group]:
                        metadata_now = internal_WaveformMetadata[group][param]
                        metadata_now
                        add_simulation = True
                        for key in list(params_to_test.keys()):
                                if (key not in metadata_now) or metadata_now[key]  > params_to_test[key] :
#                                    print "Bad Sim:  ", group, param, metadata_now
                                    add_simulation=False
                                if not add_simulation:
                                    break
                        if add_simulation:
                                good_sims.append((group,param))

        return good_sims

###
### Reading a '.par' file and turning it into key, value pairs 
###
def util_InterpretCactusPar(parname):
    myvars = {}
    with open(parname) as myfile:
#        print " Opened ", parname
        for line in myfile:
            # Drop comments
            line= re.sub(r'#.+', '', line)
            # if line contains an equal sign
            # split
            elems = line.partition("=")
            if len(elems) > 1:
                name, var = elems[::2]
                name = name.strip()
                var=var.strip()
#                print "X", name, var
                # drop whitespace. Do NOT cast
                name  = re.sub(r'\s+','', name)
#                print "Y", name
                var  = re.sub(r'\s+','', var)
                myvars[name.strip()] = var
    return myvars

###
### Plotting syntactic sugar
###

# http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot    
mode_line_style= {}
mode_line_style[(2,2)] = 'r'
mode_line_style[(2,1)] = 'b'
mode_line_style[(2,-1)] = 'b'
mode_line_style[(2,-2)] = 'r'
mode_line_style[(3,3)] = 'r-'
mode_line_style[(3,-3)] = 'r-'
mode_line_style[(3,1)] = 'b-'
mode_line_style[(3,-1)] = 'b-'

mode_line_style[(2,0)] = 'g'


for l in np.arange(2,5):
    for m in np.arange(-l,l+1):
        if not ((l,m) in list(mode_line_style.keys())):
            mode_line_style[(l,m)] = 'k'



###
### Load things
###
def loadseq(seq):
    if True: #try:
        exec(compile(open(dirBaseFiles + "/"+group+"/interface3.py", "rb").read(), dirBaseFiles + "/"+group+"/interface3.py", 'exec'))
    else: #except:
        print(" Fail ", seq)


try:
#if True:
    print("  ---> Attempting to load all metadata from a cache file <---")
    with open(nameMetadataFile,'rb') as f:   # dirBaseFiles+"/metadata_MasterInternalCache.pkl"
        master_internal_dict = pickle.load(f)
        for key in internal_MasterListOfFieldsPopulatedByAPI:
            if rosDebug:
                print(key)
            #print master_internal_dict[key]
            # http://stackoverflow.com/questions/2933470/how-do-i-call-setattr-on-the-current-module
            vars()["internal_"+key] = master_internal_dict[key]  # m+" = master_internal_dict[\'" + key + "\']")  # horrible hack
except:
#if False:
 print("  ---> Failed to load all stored test metadata...regenerating <---")
# try:
# if True:
#     exec(open(dirBaseFiles+"/"+"Sequence-MAYA-Generic/interface3.py").read())
# except:
# else:
#     print(" Fail ", "Sequence-MAYA-Generic")


###### ONLY MAYA #####
# try:
#     exec(open(dirBaseFiles+"/"+"Sequence-GT-Aligned-UnequalMass/interface3.py").read())
     #exec(compile(open(dirBaseFiles + "/"+"Sequence-GT-Aligned-UnequalMass/interface3.py", "rb").read(), dirBaseFiles + "/"+"Sequence-GT-Aligned-UnequalMass/interface3.py", 'exec'))
# except:
#     print(" Fail: Aligned ")
# try:
#     exec(open(dirBaseFiles+"/"+"Sequence-RIT-Generic/interface3.py").read())
# except:
#     print(" Fail ", "Sequence-RIT-Generic")
# try:
#     exec(open(dirBaseFiles+"/"+"/Sequence-RIT-Kicks/interface3.py").read())
     #exec(compile(open(dirBaseFiles+"/Sequence-RIT-Kicks/interface3.py", "rb").read(), dirBaseFiles+"/Sequence-RIT-Kicks/interface3.py", 'exec'))
# except:
#     print("Fail", "Sequence-RIT-Kicks")
# try:
#     exec(open(dirBaseFiles+"/"+"Sequence-SXS-All/interface3.py").read())
     #exec(compile(open(dirBaseFiles + "/"+"Sequence-SXS-All/interface3.py", "rb").read(), dirBaseFiles + "/"+"Sequence-SXS-All/interface3.py", 'exec'))
# except:
#     print(" Fail SXS")
#######
# try:
 if True:
     exec(open(dirBaseFiles+"/"+"Sequence-SXS-All-nr-followup/interface3.py").read())
    #exec(compile(open(dirBaseFiles + "/"+"Sequence-SXS-All/interface3.py", "rb").read(), dirBaseFiles + "/"+"Sequence-SXS-All/interface3.py", 'exec'))
# except:
 else:
     print(" Fail SXS-nr-followup")
#######


# try:
#     exec(compile(open(dirBaseFiles+"/Sequence-RIT-OlderWork/interface3.py", "rb").read(), dirBaseFiles+"/Sequence-RIT-OlderWork/interface3.py", 'exec'))
# except:
#     print("Fail ", "Sequence-RIT-OlderWork")
# exec(compile(open(dirBaseFiles + "/"+"Sequence-GT-IdenticalMisalignedCircular-UnequalMass/interface3.py", "rb").read(), dirBaseFiles + "/"+"Sequence-GT-IdenticalMisalignedCircular-UnequalMass/interface3.py", 'exec'))
# exec(compile(open(dirBaseFiles + "/"+"Sequence-GT-IdenticalTiltingCircular-UnequalMass/interface3.py", "rb").read(), dirBaseFiles + "/"+"Sequence-GT-IdenticalTiltingCircular-UnequalMass/interface3.py", 'exec'))
# exec(compile(open(dirBaseFiles + "/"+"Sequence-GT-IdenticalVaryPhi/interface3.py", "rb").read(), dirBaseFiles + "/"+"Sequence-GT-IdenticalVaryPhi/interface3.py", 'exec'))
# exec(compile(open(dirBaseFiles + "/"+"Sequence-PSU-IdenticalMisalignedCircular-HighResolution/interface3.py", "rb").read(), dirBaseFiles + "/"+"Sequence-PSU-IdenticalMisalignedCircular-HighResolution/interface3.py", 'exec'))  # metadata completely unav
# exec(compile(open(dirBaseFiles + "/"+"Sequence-PSU-Assorted/interface3.py", "rb").read(), dirBaseFiles + "/"+"Sequence-PSU-Assorted/interface3.py", 'exec'))  # missing Momega0
# exec(compile(open(dirBaseFiles + "/"+"Sequence-Hinder-Eccentric/interface3.py", "rb").read(), dirBaseFiles + "/"+"Sequence-Hinder-Eccentric/interface3.py", 'exec'))
# exec(compile(open(dirBaseFiles+"/"+"Sequence-GT-BurstSet/interface3.py", "rb").read(), dirBaseFiles+"/"+"Sequence-GT-BurstSet/interface3.py", 'exec'))
# exec(compile(open(dirBaseFiles+"/"+"Sequence-BAM-GitAnnex/interface3.py", "rb").read(), dirBaseFiles+"/"+"Sequence-BAM-GitAnnex/interface3.py", 'exec'))

# exec(compile(open(dirBaseFiles+"/"+"Sequence-LVC-GitAnnex/interface3.py", "rb").read(), dirBaseFiles+"/"+"Sequence-LVC-GitAnnex/interface3.py", 'exec'))

###
### LVC catalog: Always do this manually!
###


###
### Testing
###
#execfile(dirBaseFiles+"/"+"Sequence-LVC-GitAnnex/interface.py")
#exec(open(dirBaseFiles+"/"+"Sequence-RIT-Generic/interface3.py").read())
#execfile(dirBaseFiles + "/"+"Sequence-SXS-All/interface.py")     
#execfile(dirBaseFiles+"/"+"Sequence-GT-Aligned-UnequalMass/interface.py")
#execfile(dirBaseFiles + "/"+"Sequence-GT-IdenticalMisalignedCircular-UnequalMass/interface.py")
#execfile(dirBaseFiles+"/"+"Sequence-GT-BurstSet/interface.py")
#execfile(dirBaseFiles+"/Sequence-RIT-Kicks/interface.py")
#execfile(dirBaseFiles+"/"+"Sequence-LVC-NRMatter/interface.py")

# internal_SpecialDefaultExtractionRadius["Sequence-RIT-Generic"]["Z4_D10.00_q0.6628_a0.95_-0.95_n144"] = 102.60  # standard location does not exist 
# internal_EstimateStartingMOmega0["Sequence-RIT-Generic"]["Z4_D10.00_q0.6628_a0.95_-0.95_n144"]=0
# internal_EstimatePeakL2M2Emission["Sequence-RIT-Generic"]["Z4_D10.00_q0.6628_a0.95_-0.95_n144"]=800
# internal_SpecialJunkRadiationUntilTime["Sequence-RIT-Generic"]["Z4_D10.00_q0.6628_a0.95_-0.95_n144"]=180
