# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


import numpy as np
import logging
import time

from astropy import constants
from cython.view cimport array as cvarray

cimport numpy as np
np.import_array()

from cython.parallel import *


ctypedef np.float64_t float_type_t
ctypedef np.int64_t int_type_t


cdef extern from "math.h":
    float_type_t log(float_type_t)
    float_type_t sqrt(float_type_t)
    float_type_t exp(float_type_t)
    int_type_t floor(float_type_t)
    bint isnan(double x)



cdef extern from "randomkit/randomkit.h":
    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

    ctypedef enum rk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    void rk_seed(unsigned long seed, rk_state *state)
    float_type_t rk_double(rk_state *state)
cdef rk_state mt_state

cdef np.ndarray x

cdef struct Packet_struct:
    int_type_t current_packet_id
    float_type_t current_nu
    float_type_t current_energy
    float_type_t current_mu
    int_type_t current_shell_id
    float_type_t current_r
    float_type_t tau_event
    int_type_t current_line_id
    int_type_t last_line
    int_type_t close_line
    int_type_t recently_crossed_boundary
    float_type_t distance_to_move
    int_type_t virtual_packet #0 real, 1 virtual
    int_type_t type #0 Disabled,  1 R-Packet, 2 K-Packet, 3 I-Packet, 11 VR-Packet from the inner boundary, 12 VR-Packet from anywhere in the ejecta,98 reabsorbed ,99 escaped
    float_type_t comov_nu #handle with care
    float_type_t comov_energy #handle with care

cdef struct Options_struct:
    int_type_t enable_bf





cdef class StorageModel:
    """
    Class for storing the arrays in a cythonized way (as pointers). This ensures fast access during the calculations.
    """

    cdef np.ndarray packet_nus_a
    cdef float_type_t*packet_nus
    cdef np.ndarray packet_mus_a
    cdef float_type_t*packet_mus
    cdef np.ndarray packet_energies_a
    cdef float_type_t*packet_energies
    ######## Setting up the output ########
    cdef np.ndarray output_nus_a
    cdef float_type_t*output_nus
    cdef np.ndarray output_energies_a
    cdef float_type_t*output_energies

    cdef np.ndarray last_line_interaction_in_id_a
    cdef int_type_t*last_line_interaction_in_id

    cdef np.ndarray last_line_interaction_out_id_a
    cdef int_type_t*last_line_interaction_out_id

    cdef np.ndarray last_line_interaction_shell_id_a
    cdef int_type_t*last_line_interaction_shell_id

    cdef np.ndarray last_interaction_type_a
    cdef int_type_t*last_interaction_type

    cdef int_type_t no_of_packets
    cdef int_type_t no_of_shells
    cdef np.ndarray r_inner_a
    cdef float_type_t*r_inner
    cdef np.ndarray r_outer_a
    cdef float_type_t*r_outer
    cdef np.ndarray v_inner_a
    cdef float_type_t*v_inner
    cdef float_type_t time_explosion
    cdef float_type_t inverse_time_explosion
    cdef np.ndarray electron_densities_a
    cdef float_type_t*electron_densities
    cdef np.ndarray inverse_electron_densities_a
    cdef float_type_t*inverse_electron_densities
    cdef np.ndarray line_list_nu_a
    cdef float_type_t*line_list_nu
    cdef float_type_t[:] line_list_nu_view
    cdef np.ndarray line_lists_tau_sobolevs_a
    cdef float_type_t*line_lists_tau_sobolevs
    cdef int_type_t line_lists_tau_sobolevs_nd

    #J_BLUES initialize
    cdef np.ndarray line_lists_j_blues_a
    cdef float_type_t*line_lists_j_blues
    cdef int_type_t line_lists_j_blues_nd

    cdef int_type_t no_of_lines
    cdef int_type_t line_interaction_id
    cdef np.ndarray transition_probabilities_a
    cdef float_type_t*transition_probabilities
    cdef int_type_t transition_probabilities_nd
    cdef np.ndarray line2macro_level_upper_a
    cdef int_type_t*line2macro_level_upper
    cdef np.ndarray macro_block_references_a
    cdef int_type_t*macro_block_references
    cdef np.ndarray transition_type_a
    cdef int_type_t*transition_type
    cdef np.ndarray destination_level_id_a
    cdef int_type_t*destination_level_id
    cdef np.ndarray transition_line_id_a
    cdef int_type_t*transition_line_id
    cdef np.ndarray js_a
    cdef float_type_t*js
    cdef np.ndarray nubars_a
    cdef float_type_t*nubars
    cdef float_type_t spectrum_start_nu
    cdef float_type_t spectrum_delta_nu
    cdef float_type_t spectrum_end_nu
    cdef float_type_t*spectrum_virt_nu

    cdef float_type_t sigma_thomson
    cdef float_type_t inverse_sigma_thomson
    cdef int_type_t current_packet_id

    cdef np.ndarray kappa_bf_gray
    cdef float_type_t[:] kappa_bf_gray_view

    cdef np.ndarray kappa_bf_nu
    cdef float_type_t[:,:] kappa_bf_nu_view


    cdef np.ndarray bf_nu_bin
    cdef float_type_t[:] bf_nu_bin_view
    cdef int_type_t bf_nu_bin_size

    cdef float_type_t nu_min
    cdef float_type_t nu_max


    cdef float_type_t* t_electron
    cdef np.ndarray t_electron_a

    cdef int_type_t disableBoundFree

    cdef float_type_t black_body_peak
    cdef float_type_t nu_sampling_lower_lim
    cdef float_type_t nu_sampling_upper_lim




    def __init__(self, model):

        rk_seed(250819801106, &mt_state)

        cdef np.ndarray[float_type_t, ndim=1] packet_nus = model.packet_src.packet_nus
        self.packet_nus_a = packet_nus
        self.packet_nus = <float_type_t*> self.packet_nus_a.data
        #
        cdef np.ndarray[float_type_t, ndim=1] packet_mus = model.packet_src.packet_mus
        self.packet_mus_a = packet_mus
        self.packet_mus = <float_type_t*> self.packet_mus_a.data
        #
        cdef np.ndarray[float_type_t, ndim=1] packet_energies = model.packet_src.packet_energies
        self.packet_energies_a = packet_energies
        self.packet_energies = <float_type_t*> self.packet_energies_a.data
        #
        self.no_of_packets = packet_nus.size
        #@@@ Setup of Geometry @@@
        self.no_of_shells = model.no_of_shells
        cdef np.ndarray[float_type_t, ndim=1] r_inner = model.r_inner
        self.r_inner_a = r_inner
        self.r_inner = <float_type_t*> self.r_inner_a.data
        #
        cdef np.ndarray[float_type_t, ndim=1] r_outer = model.r_outer
        self.r_outer_a = r_outer
        self.r_outer = <float_type_t*> self.r_outer_a.data
        #
        cdef np.ndarray[float_type_t, ndim=1] v_inner = model.v_inner
        self.v_inner_a = v_inner
        self.v_inner = <float_type_t*> self.v_inner_a.data
        #@@@ Setup the rest @@@
        #times
        self.time_explosion = model.time_explosion
        self.inverse_time_explosion = 1 / model.time_explosion
        #electron density
        cdef np.ndarray[float_type_t, ndim=1] electron_densities = model.electron_densities
        self.electron_densities_a = electron_densities
        self.electron_densities = <float_type_t*> self.electron_densities_a.data
        #
        cdef np.ndarray[float_type_t, ndim=1] inverse_electron_densities = 1 / electron_densities
        self.inverse_electron_densities_a = inverse_electron_densities
        self.inverse_electron_densities = <float_type_t*> self.inverse_electron_densities_a.data
        #Line lists
        cdef np.ndarray[float_type_t, ndim=1] line_list_nu = model.line_list_nu.values
        self.line_list_nu_a = line_list_nu
        self.line_list_nu = <float_type_t*> self.line_list_nu_a.data
        cdef float_type_t[:] line_list_nu_view = line_list_nu
        self.line_list_nu_view = line_list_nu_view
        #
        self.no_of_lines = line_list_nu.size

        cdef np.ndarray[float_type_t, ndim=2] line_lists_tau_sobolevs = model.tau_sobolevs
        self.line_lists_tau_sobolevs_a = line_lists_tau_sobolevs
        self.line_lists_tau_sobolevs = <float_type_t*> self.line_lists_tau_sobolevs_a.data
        self.line_lists_tau_sobolevs_nd = self.line_lists_tau_sobolevs_a.shape[1]

        cdef np.ndarray[float_type_t, ndim=2] line_lists_j_blues = model.j_blues
        model.j_blues[:] = 0.0
        self.line_lists_j_blues_a = line_lists_j_blues
        self.line_lists_j_blues = <float_type_t*> self.line_lists_j_blues_a.data
        self.line_lists_j_blues_nd = self.line_lists_j_blues_a.shape[1]


        #The bound-free kappas.
        #At the moment only the gray operatics are passed to the cython module
        cdef np.ndarray[float_type_t, ndim = 1] kappa_bf_gray = model.kappa_bf_gray
        cdef float_type_t[:] kappa_bf_gray_view = kappa_bf_gray

        self.kappa_bf_gray = kappa_bf_gray
        self.kappa_bf_gray_view = kappa_bf_gray_view

        cdef np.ndarray[float_type_t,ndim = 2] kappa_bf_nu = model.kappa_bf_nu
        cdef float_type_t[:,:] kappa_bf_nu_view = kappa_bf_nu

        self.kappa_bf_nu = kappa_bf_nu
        self.kappa_bf_nu_view = kappa_bf_nu_view


        #TODO chagne this to a pointer
        cdef np.ndarray[float_type_t, ndim = 1] bf_nu_bin  = model.kappa_bf_nu_bins.astype('f8')
        cdef float_type_t[:] bf_nu_bin_view = bf_nu_bin
        self.bf_nu_bin = bf_nu_bin
        self.bf_nu_bin_view = bf_nu_bin_view

        # cdef float_type_t [:,:] bf_nu_bin_view = self.bf_nu_bin_a
        # self.bf_nu_bin_view = bf_nu_bin_view

        self.bf_nu_bin_size = len(self.bf_nu_bin)
        size_of_bf_nu_bin = self.no_of_shells * self.bf_nu_bin_size

        self.disableBoundFree = model.disableBoundFree


        #electron temperature
        cdef np.ndarray[float_type_t, ndim = 1] t_electron = model.t_electron
        self.t_electron_a = t_electron
        self.t_electron = <float_type_t*> self.t_electron_a.data

        #
        self.line_interaction_id = model.line_interaction_id
        #macro atom & downbranch
        cdef np.ndarray[float_type_t, ndim=2] transition_probabilities
        cdef np.ndarray[int_type_t, ndim=1] line2macro_level_upper
        cdef np.ndarray[int_type_t, ndim=1] macro_block_references
        cdef np.ndarray[int_type_t, ndim=1] transition_type
        cdef np.ndarray[int_type_t, ndim=1] destination_level_id
        cdef np.ndarray[int_type_t, ndim=1] transition_line_id
        if model.line_interaction_id >= 1:
            transition_probabilities = model.transition_probabilities
            self.transition_probabilities_a = transition_probabilities
            self.transition_probabilities = <float_type_t*> self.transition_probabilities_a.data
            self.transition_probabilities_nd = self.transition_probabilities_a.shape[1]
            #
            line2macro_level_upper = model.atom_data.lines_upper2macro_reference_idx
            self.line2macro_level_upper_a = line2macro_level_upper
            self.line2macro_level_upper = <int_type_t*> self.line2macro_level_upper_a.data
            macro_block_references = model.atom_data.macro_atom_references['block_references'].values
            self.macro_block_references_a = macro_block_references
            self.macro_block_references = <int_type_t*> self.macro_block_references_a.data
            transition_type = model.atom_data.macro_atom_data['transition_type'].values
            self.transition_type_a = transition_type
            self.transition_type = <int_type_t*> self.transition_type_a.data

            #Destination level is not needed and/or generated for downbranch
            destination_level_id = model.atom_data.macro_atom_data['destination_level_idx'].values
            #print "preparing stuff " , destination_level_id[0], destination_level_id[5], destination_level_id[10]
            self.destination_level_id_a = destination_level_id
            #print "preparing stuff(2) " , self.destination_level_id_a[0], self.destination_level_id_a[5], self.destination_level_id_a[10]
            self.destination_level_id = <int_type_t*> self.destination_level_id_a.data

            #print "preparing stuff(3) " , self.destination_level_id[0], self.destination_level_id[5], self.destination_level_id[10]
            transition_line_id = model.atom_data.macro_atom_data['lines_idx'].values
            self.transition_line_id_a = transition_line_id
            self.transition_line_id = <int_type_t*> self.transition_line_id_a.data

        cdef np.ndarray[float_type_t, ndim=1] output_nus = np.zeros(self.no_of_packets, dtype=np.float64)
        cdef np.ndarray[float_type_t, ndim=1] output_energies = np.zeros(self.no_of_packets, dtype=np.float64)

        self.output_nus_a = output_nus
        self.output_nus = <float_type_t*> self.output_nus_a.data

        self.output_energies_a = output_energies
        self.output_energies = <float_type_t*> self.output_energies_a.data

        cdef np.ndarray[int_type_t, ndim=1] last_line_interaction_in_id = -1 * np.ones(self.no_of_packets,
                                                                                       dtype=np.int64)
        cdef np.ndarray[int_type_t, ndim=1] last_line_interaction_out_id = -1 * np.ones(self.no_of_packets,
                                                                                        dtype=np.int64)

        cdef np.ndarray[int_type_t, ndim=1] last_line_interaction_shell_id = -1 * np.ones(self.no_of_packets,
                                                                                          dtype=np.int64)

        cdef np.ndarray[int_type_t, ndim=1] last_interaction_type = -1 * np.ones(self.no_of_packets,
                                                                                 dtype=np.int64)

        self.last_line_interaction_in_id_a = last_line_interaction_in_id
        self.last_line_interaction_in_id = <int_type_t*> self.last_line_interaction_in_id_a.data

        self.last_line_interaction_out_id_a = last_line_interaction_out_id
        self.last_line_interaction_out_id = <int_type_t*> self.last_line_interaction_out_id_a.data

        self.last_line_interaction_shell_id_a = last_line_interaction_shell_id
        self.last_line_interaction_shell_id = <int_type_t*> last_line_interaction_shell_id.data

        self.last_interaction_type_a = last_interaction_type
        self.last_interaction_type = <int_type_t*> last_interaction_type.data

        cdef np.ndarray[float_type_t, ndim=1] js = np.zeros(model.no_of_shells, dtype=np.float64)
        cdef np.ndarray[float_type_t, ndim=1] nubars = np.zeros(model.no_of_shells, dtype=np.float64)

        self.js_a = js
        self.js = <float_type_t*> self.js_a.data
        self.nubars_a = nubars
        self.nubars = <float_type_t*> self.nubars_a.data
        self.spectrum_start_nu = model.spec_nu_bins.min()
        self.spectrum_end_nu = model.spec_nu_bins.max()
        self.spectrum_delta_nu = model.spec_nu_bins[1] - model.spec_nu_bins[0]

        cdef np.ndarray[float_type_t, ndim=1] spectrum_virt_nu = model.spec_virtual_flux_nu
        self.spectrum_virt_nu = <float_type_t*> spectrum_virt_nu.data

        if model.sigma_thomson is None:
            self.sigma_thomson = 6.652486e-25 #cm^(-2)
        else:
            self.sigma_thomson = model.sigma_thomson

        self.inverse_sigma_thomson = 1 / self.sigma_thomson


        #constants for the nu sampling
        cdef float_type_t nuPeak = 5.8789254e10 * self.t_electron[0]
        cdef float_type_t nu_sampling_c1 = 2. * h_planck / (c**2)
        cdef float_type_t nu_sampling_c2 = h_planck / boltzmann
        cdef float_type_t bbPeak
        cdef int_type_t i = 0
        cdef float_type_t nu_sampling_lower_lim = self.spectrum_start_nu
        cdef float_type_t nu_sampling_upper_lim = self.spectrum_end_nu
        cdef float_type_t t_e = self.t_electron[0]

        print("self.spectrum_start_nu %g self.spectrum_end_nu  %g nuPeak %g"%(self.spectrum_start_nu,self.spectrum_end_nu ,nuPeak))
        if self.spectrum_start_nu > nuPeak and self.spectrum_end_nu > nuPeak:
            bbPeak = ((nu_sampling_c1 * self.spectrum_start_nu **3)/(np.exp(nu_sampling_c2*(self.spectrum_start_nu /t_e)) - 1 ))
            test_bins = 100
            while i < test_bins:
                print("left; startnu %g endnu %g"%(self.spectrum_start_nu ,self.spectrum_end_nu ))
                i += 1
                interval = (self.spectrum_end_nu - self.spectrum_start_nu)/ test_bins
                nu_tmp = self.spectrum_start_nu + interval*i
                bb_tmp = ((nu_sampling_c1 * nu_tmp **3)/(np.exp(nu_sampling_c2*(nu_tmp /t_e)) - 1 ))
                print("bb_tmp/bbPeak %g"%(bb_tmp /bbPeak))
                print("nu_tmp %g"%(nu_tmp))
                if  bb_tmp / bbPeak <0.1:
                    nu_sampling_upper_lim = nu_tmp
                    break
        elif self.spectrum_start_nu < nuPeak and self.spectrum_end_nu < nuPeak:
            bbPeak = ((nu_sampling_c1 * self.spectrum_end_nu**3)/(np.exp(nu_sampling_c2*(self.spectrum_end_nu /t_e)) - 1 ))
            test_bins = 10
            while i < test_bins:
                print("right")
                i += 1
                interval = (self.spectrum_end_nu - self.spectrum_start_nu)/ test_bins
                nu_tmp = self.spectrum_end_nu - interval*i
                bb_tmp = ((nu_sampling_c1 * nu_tmp **3)/(np.exp(nu_sampling_c2*(nu_tmp /t_e)) - 1 ))
                print("bbPeak %g bb_tmp %g"%(bbPeak,bb_tmp))
                if  bb_tmp /bbPeak < 0.1:
                    nu_sampling_lower_lim = nu_tmp
                    break
        else:
            # print("mid")
            bbPeak = ((nu_sampling_c1 * nuPeak**3)/(np.exp(nu_sampling_c2*(nuPeak/t_e)) - 1 ))

        #find cutoff

        self.black_body_peak = bbPeak
        self.nu_sampling_upper_lim = nu_sampling_upper_lim
        self.nu_sampling_lower_lim = nu_sampling_lower_lim

        #
        #
        #

DEF packet_logging = False
IF packet_logging == True:
    packet_logger = logging.getLogger('tardis_packet_logger')
#DEF packet_logging = False

logger = logging.getLogger(__name__)
#Log Level
#cdef int_type_t loglevel = logging.DEBUG








#constants
cdef float_type_t miss_distance = 1e99
cdef float_type_t c = constants.c.cgs.value # cm/s
cdef float_type_t inverse_c = 1 / c
cdef float_type_t h_planck = constants.h.cgs.value
cdef float_type_t boltzmann = constants.k_B.cgs.value




#DEBUG STATEMENT TAKE OUT
#cdef float_type_t sigma_thomson = 6.652486e-25
cdef float_type_t sigma_thomson = 6.652486e-125
cdef float_type_t inverse_sigma_thomson = 1 / sigma_thomson


cdef rand_nu_planck(float_type_t t_electron,float_type_t nuStart, float_type_t nuEnd, float_type_t bPeak):
    """
    This function draws a random nu corresponding to the temperature. Using the Neumann rejection sampling.
    :param t_electron: the electron temperature of the plasma
    :return: Random nu Boltzmann distributed
    """
    cdef float_type_t c1 = 2. * h_planck / (c**2)
    cdef float_type_t c2 = h_planck / boltzmann
    cdef float_type_t nuPeak = 5.8789254e10 * t_electron
    # cdef float_type_t  bPeak #= ((c1 * nuPeak**3)/(np.exp(c2*(nuPeak/t_electron)) - 1 ))
    cdef float_type_t nuRand, bRand
    #Find the max
    if nuStart > nuPeak and nuEnd > nuPeak:
        bPeak = ((c1 * nuStart**3)/(np.exp(c2*(nuStart/t_electron)) - 1 ))
         # print("left")
    elif nuStart < nuPeak and nuEnd < nuPeak:
        bPeak = ((c1 * nuEnd**3)/(np.exp(c2*(nuEnd/t_electron)) - 1 ))
    #     # print("right")
    else:
    #     # print("mid")
        bPeak = ((c1 * nuPeak**3)/(np.exp(c2*(nuPeak/t_electron)) - 1 ))
    #cdef float_type_t nuStart, nuEnd, nuInterval
    #tmp definition only for debug. Take this values from the config
    #nuStart = 1.4989623e+14
    #nuEnd = 5.9958492e+15
    # print(bPeak)
    # print("start %g end %g peak %g"%(nuStart,nuEnd,nuPeak ))
    ###
    nuInterval = nuEnd - nuStart
    # print(c1)
    # print(c2)
    # print(nuPeak)
    # print("-")
    # print(nuEnd)
    # print("----")
    # print(nuStart)
    # print("####")
    while True:
        nuRand = rk_double(&mt_state) * nuInterval + nuStart
        bRand = bPeak * rk_double(&mt_state)
        bFromNu = ((c1 * nuRand**3)/(np.exp(c2*(nuRand/t_electron)) - 1 ))
        if bRand < bFromNu:
            break
    return nuRand


cdef float_type_t getGrayKappaBFbyNu(float_type_t nu, float_type_t[:] nu_bins, float_type_t[:,:] kappa_bf_nu_view,int_type_t* current_shell_id, float_type_t disableBoundFree):
    cdef float_type_t kappa_bf_nu = 0
    cdef int_type_t i
    if not disableBoundFree:
        return 0
    #for i in range(len(nu_bins)-1):
        #print(nu_bins[i],'<',nu,'<',nu_bins[i+1])
        # if ((nu > nu_bins[i]) and (nu <= nu_bins[i+1])):
        #     kappa_bf_nu =  kappa_bf_nu_view[current_shell_id[0],i]
        #     bla = i
        #     break
    ii =  binary_search(nu_bins,nu,0,len(nu_bins))
    return kappa_bf_nu_view[current_shell_id[0],ii]

cdef int_type_t getNextLineId(float_type_t[:] nu, float_type_t nu_insert, int_type_t imin,
                              int_type_t imax):
    # print('startNextLineId')
    if nu_insert > nu[imin]:
        # print("nu[imin]= %g" %nu[imin])
        # print("nu_insert %g" %nu_insert)
        return imin
    elif nu_insert < nu[imax]:
        # print("nu[imax]= %g" %nu[imax])
        # print("nu_insert %g" %nu_insert)
        return imax +1
    else:
        # print('start bin search')
        # print("nu[imax]= %g" %nu[imax])
        # print("nu[imin]= %g" %nu[imin])
        # print("nu_insert %g" %nu_insert)
        return binary_search(nu,nu_insert,imin,imax)

cdef int_type_t binary_search(float_type_t[:] nu, float_type_t nu_insert, int_type_t imin,
                              int_type_t imax):
    #continually narrow search until just one element remains
    cdef int_type_t imid
    while imax - imin > 2:
        imid = (imin + imax) / 2

        #code must guarantee the interval is reduced at each iteration
        #assert(imid < imax);
        # note: 0 <= imin < imax implies imid will always be less than imax

        # reduce the search
        if (nu[imid] < nu_insert):
            imax = imid + 1
        else:
            imin = imid
            #print imin, imax, imid, imax - imin
    return imin + 1

#variables are restframe if not specified by prefix comov_
#cdef inline int_type_t macro_atom(int_type_t activate_level,
#                                  float_type_t*p_transition,
#                                  int_type_t p_transition_nd,
#                                  int_type_t*type_transition,
#                                  int_type_t*target_level_id,
#                                  int_type_t*target_line_id,
#                                  int_type_t*unroll_reference,
#                                  int_type_t cur_zone_id):
cdef int_type_t macro_atom(StorageModel storage, Packet_struct* packet,int_type_t activate_level):


    cdef int_type_t emit, i = 0
    cdef float_type_t p, event_random = 0.0
    cdef float_type_t*p_transition = storage.transition_probabilities
    cdef int_type_t p_transition_nd = storage.transition_probabilities_nd
    cdef int_type_t*type_transition = storage.transition_type
    cdef int_type_t*target_level_id = storage.destination_level_id
    cdef int_type_t*target_line_id = storage.transition_line_id
    cdef int_type_t*unroll_reference = storage.macro_block_references
    cdef int_type_t cur_zone_id = packet.current_shell_id
    #print "Activating Level %d" % activate_level
    while True:
        event_random = rk_double(&mt_state)
        #activate_level = 7
        i = unroll_reference[activate_level]
        #i_end = unroll_reference[activate_level + 1]
        #print "checking tprops %.5g" % (p_transition[cur_zone_id * p_transition_nd +i: cur_zone_id * p_transition_nd +i_end)
        #        print "jumping to block_references %d" % i
        p = 0.0
        #cur_zone_id=10

        #print "bunch of numbers %g %g %g %g %g %g %g %g %g" % ( p_transition[i], p_transition[i+1], p_transition[i+2], p_transition[i+3], p_transition[i+4], p_transition[i+5], p_transition[i+6], p_transition[i+7], p_transition[i+8])
        #print "new bunch of numbers %g %g %g %g %g %g %g %g %g" % ( p_transition[cur_zone_id * p_transition_nd + i], p_transition[cur_zone_id * p_transition_nd + i+1], p_transition[cur_zone_id * p_transition_nd + i+2], p_transition[cur_zone_id * p_transition_nd + i+3], p_transition[cur_zone_id * p_transition_nd + i+4], p_transition[cur_zone_id * p_transition_nd + i+5], p_transition[cur_zone_id * p_transition_nd + i+6], p_transition[cur_zone_id * p_transition_nd + i+7], p_transition[cur_zone_id * p_transition_nd + i+8])
        #print "p_transition_nd %g" % p_transition_nd


        while True:
            p = p + p_transition[cur_zone_id * p_transition_nd + i]
            #print " p %g %g %g" % (p, cur_zone_id, i)
            if p > event_random:
                emit = type_transition[i]
                #                print "assuming transition_id %d" % emit
                activate_level = target_level_id[i]
                #print "reActivating Level %g %g" % (activate_level, i)
                break
            i += 1
            if i == unroll_reference[activate_level + 1]:
                logger.warn("overrolling!")
                logger.warn(
                    "activate level %g unroll_reference[activate_level] %g unroll_reference[activate_level+1] %g i=%g event_random=%g current_p=%g" % (
                        activate_level, unroll_reference[activate_level], unroll_reference[activate_level + 1], i,
                        event_random,
                        p))
                time.sleep(10000000)

                #print "I just broke " , emit
        if emit == -1:
            IF packet_logging == True:
                packet_logger.debug('Emitting in level %d', activate_level + 1)

            return target_line_id[i]

cdef float_type_t move_packet(StorageModel storage,Packet_struct*packet,float_type_t distance):


    cdef float_type_t new_r, doppler_factor, comov_energy, comov_nu
    doppler_factor = (1 - (packet.current_mu * packet.current_r * storage.inverse_time_explosion * inverse_c))
    IF packet_logging == True:
        if distance < 0:
            packet_logger.warn('Distance in move packets less than 0.')

    if distance <= 0:
        return doppler_factor

    new_r = sqrt( packet.current_r** 2 + distance ** 2 + 2 *packet.current_r * distance * packet.current_mu)
    packet.current_mu = (packet.current_mu* packet.current_r  + distance) / new_r

    if packet.current_mu == 0.0:
        print "-------- move packet: mu turned 0.0"
        print distance, new_r,packet.current_r, packet.current_shell_id
        print distance / new_r
        #print "move packet after mu", mu[0]
    packet.current_r = new_r

    if (packet.type > 10):
        return doppler_factor

    comov_energy = packet.current_energy * doppler_factor
    comov_nu = packet.current_nu * doppler_factor
    storage.js[packet.current_shell_id] += comov_energy * distance

    storage.nubars[packet.current_shell_id] += comov_energy * distance * comov_nu

    #print "move packet before mu", mu[0], distance, new_r, r[0]
    #    if distance/new_r > 1e-6:
    #        mu[0] = (distance**2 + new_r**2 - r[0]**2) / (2*distance*new_r)
    #if ((((mu[0] * r[0] + distance) / new_r) < 0.0) and (mu[0] > 0.0)):

    return doppler_factor
cdef void increment_j_blue_estimator(StorageModel storage, Packet_struct* packet,float_type_t d_line, int_type_t j_blue_idx):
#cdef void increment_j_blue_estimator(int_type_t*current_line_id, float_type_t*current_nu, float_type_t*current_energy,
#                                     float_type_t*mu, float_type_t*r, float_type_t d_line, int_type_t j_blue_idx,
#                                     StorageModel storage):
    cdef float_type_t comov_energy, comov_nu, r_interaction, mu_interaction, distance, doppler_factor

    distance = d_line

    r_interaction = sqrt(packet.current_r ** 2 + distance ** 2 + 2 * packet.current_r  * distance * packet.current_mu)
    mu_interaction = (packet.current_mu * packet.current_r + distance) / r_interaction

    doppler_factor = (1 - (mu_interaction * r_interaction * storage.inverse_time_explosion * inverse_c))

    comov_energy = packet.current_energy * doppler_factor
    comov_nu = packet.current_nu * doppler_factor

    storage.line_lists_j_blues[j_blue_idx] += (comov_energy / packet.current_nu)

    #print "incrementing j_blues = %g" % storage.line_lists_j_blues[j_blue_idx]

cdef float_type_t compute_distance2outer(float_type_t r, float_type_t  mu, float_type_t r_outer):
    cdef float_type_t d_outer
    d_outer = sqrt(r_outer ** 2 + ((mu ** 2 - 1.) * r ** 2)) - (r * mu)
    return d_outer

cdef float_type_t compute_distance2inner(float_type_t r, float_type_t mu, float_type_t r_inner):
    #compute distance to the inner layer
    #check if intersection is possible?
    cdef float_type_t check, d_inner
    check = r_inner ** 2 + (r ** 2 * (mu ** 2 - 1.))
    if check < 0:
        return miss_distance
    else:
        if mu < 0:
            d_inner = -r * mu - sqrt(check)
            return d_inner
        else:
            return miss_distance

cdef float_type_t compute_distance2line(float_type_t r, float_type_t mu,
                                        float_type_t nu, float_type_t nu_line,
                                        float_type_t t_exp, float_type_t inverse_t_exp,
                                        float_type_t last_line, float_type_t next_line, int_type_t cur_zone_id):
    #computing distance to line
    cdef float_type_t comov_nu, doppler_factor
    doppler_factor = (1. - (mu * r * inverse_t_exp * inverse_c))
    comov_nu = nu * doppler_factor

    #check if the comov_nu is smaller then nu_line
    if (comov_nu < nu_line):
            print("-->WARNING comoving nu less than nu_line shouldn't happen:")
            print "WARNING comoving nu less than nu_line shouldn't happen:"
            print "comov_nu = ", comov_nu
            print "nu_line", nu_line
            print "(comov_nu - nu_line) nu_lines", (comov_nu - nu_line) / nu_line
            print "last_line", last_line
            print "next_line", next_line
            print "r", r
            print "mu", mu
            print "nu", nu
            print "doppler_factor", doppler_factor
            print "cur_zone_id", cur_zone_id
          #  a = input()
            return miss_distance

    return ((comov_nu - nu_line) / nu) * c * t_exp

cdef float_type_t compute_distance2electron(float_type_t r, float_type_t mu, float_type_t tau_event,
                                            float_type_t inverse_ne):
    return tau_event * inverse_ne # * inverse_sigma_thomson folded into inverse_ne

#This is the new distance2continum function. The aim of it is to compute the distance to the next continum event which
# is the sum of electron, bound-free and, free-free opacities.

# cdef float_type_t compute_distance2continum(float_type_t tau_event,
#                                             float_type_t nu,
#                                             float_type_t[:,:] kappa_bf_nu_view,
#                                             float_type_t[:] nu_bins_view,
#                                             float_type_t ne,
#                                             float_type_t sigma_thomson,
#                                             int_type_t* current_shell_id,
#                                             int_type_t disableBoundFree):

cdef float_type_t compute_distance2continum(StorageModel storage,Packet_struct* packet,int_type_t enable_bf):


    cdef float_type_t kappa_cont = 0
    cdef float_type_t kappa_bf_nu = 0
    cdef float_type_t kappa_ff = 0
    cdef float_type_t kappa_th = storage.electron_densities[packet.current_shell_id] * storage.sigma_thomson
    kappa_bf_nu = getGrayKappaBFbyNu(packet.current_nu,storage.bf_nu_bin_view,storage.kappa_bf_nu_view,&packet.current_shell_id,enable_bf)
    kappa_cont = kappa_bf_nu + kappa_ff + kappa_th
    # print(kappa_cont)
    # print(kappa_bf_nu)
    # print(kappa_th)
    return packet.tau_event/kappa_cont

####################New montecarlo
def montecarlo_radial1d(model, int_type_t virtual_packet_flag=0,int_type_t enable_bf = 0):

    """
    The montecarlo_radial1d function takes the packets including there properties from the model and
    starts for every one a montecarlo loop. For the montecarlo loop the properties of the packets are stored in the
    packet and the model properties in the storage. If virtual_packet_flag > 0 this function starts virtual packets
    to track the input packet.

    Parameters
    ---------


    model : `tardis.model_radial_oned.ModelRadial1D`
        complete model

    param photon_packets : PacketSource object
        photon packets

    Returns
    -------

    output_nus : `numpy.ndarray`

    output_energies : `numpy.ndarray`



    TODO
                    np.ndarray[float_type_t, ndim=1] line_list_nu,
                    np.ndarray[float_type_t, ndim=2] tau_lines,
                    np.ndarray[float_type_t, ndim=1] ne,
                    float_type_t packet_energy,
                    np.ndarray[float_type_t, ndim=2] p_transition,
                    np.ndarray[int_type_t, ndim=1] type_transition,
                    np.ndarray[int_type_t, ndim=1] target_level_id,
                    np.ndarray[int_type_t, ndim=1] target_line_id,
                    np.ndarray[int_type_t, ndim=1] unroll_reference,
                    np.ndarray[int_type_t, ndim=1] line2level,
                    int_type_t log_packets,
                    int_type_t do_scatter

    """

    storage = StorageModel(model)


    ######## Setting up the output ########
    cdef np.ndarray[float_type_t, ndim=1] output_nus = np.zeros(storage.no_of_packets, dtype=np.float64)
    cdef np.ndarray[float_type_t, ndim=1] output_energies = np.zeros(storage.no_of_packets, dtype=np.float64)
    ######## Setting up the packet
    cdef Packet_struct packet


    for i in range(storage.no_of_packets):
        if i % (storage.no_of_packets / 5) == 0:
            logger.info("At packet %d of %d", i, storage.no_of_packets)

        #set the v-packet properties
        packet.virtual_packet =  virtual_packet_flag
        packet.current_packet_id = i

        #setting up the properties of the packet
        packet.type = 1 # at the beginning the packet is always a r packet
        packet.current_nu = storage.packet_nus[i]
        packet.current_energy = storage.packet_energies[i]
        packet.current_mu = storage.packet_mus[i]
        #Location of the packet and location specific parameters
        comov_current_nu = packet.current_nu

        packet.recently_crossed_boundary =1
        packet.current_shell_id = 0
        packet.current_r = storage.r_inner[0]
        packet.current_nu /= (1 - (packet.current_mu * packet.current_r * storage.inverse_time_explosion
        * inverse_c))
        packet.current_energy /= (1 - (packet.current_mu * packet.current_r * storage.inverse_time_explosion
        * inverse_c))

        #Set the position of the packet in the linelist
        packet.current_line_id = getNextLineId(storage.line_list_nu_view,comov_current_nu,0, storage.no_of_lines -1)

        if packet.current_line_id == storage.no_of_lines:
            #setting flag that the packet is off the red end of the line list
            packet.last_line = 1
        else:
            packet.last_line = 0
        packet.close_line = 0

        if packet.virtual_packet > 0:
            #this is a run for which we want the virtual packet spectrum. So first thing we need to do is spawn virtual packets to track the input packet
            #print "I'M STARTING A VIRTUAL FOR A NEW PACKET"
            packet.type = 11#Set type to VR-Packet
            reabsorbed = montecarlo_one_packet(storage, &packet ,enable_bf)
            packet.type = 1 # at the beginning and after the vr-packet is the packet is a r packet

        #Do it real
        reabsorbed = montecarlo_one_packet(storage, &packet,enable_bf)

        if packet.type == 98: #reabsorbed
            storage.output_nus[i] = -packet.current_nu
            storage.output_energies[i] = -packet.current_energy
        elif packet.type == 99: #emitted
            storage.output_nus[i] = packet.current_nu
            storage.output_energies[i] = packet.current_energy
        else:
            print("we lost the trak of one packet ")

    return storage.output_nus_a, storage.output_energies_a, storage.js_a, storage.nubars_a, \
        storage.last_line_interaction_in_id_a, storage.last_line_interaction_out_id_a,\
        storage.last_interaction_type_a, storage.last_line_interaction_shell_id_a

################ montecarlo_one_virtuel_packet####
cdef montecarlo_one_virtuel_packet(StorageModel storage,Packet_struct*packet,int_type_t enable_bf):
    """
    Do virtual packets. This is extract of the real packet
    :param storage:
    :param packet:
    :param enable_bf:
    :return:
    """
    cdef Packet_struct virtual_sub_packet
    cdef float_type_t mu_bin
    cdef float_type_t mu_min
    cdef float_type_t weight
    cdef float_type_t doppler_factor_ratio
    cdef int_type_t virt_id_nu

    for i in range(packet.virtual_packet):
    #copy the properties of the mother packet
        virtual_sub_packet = packet[0]
        #Set the virtual sub packet specific values
        #choose a direction for the extract packet. We don't want any directions that will hit the inner boundary. So this sets a minimum value for the packet mu
        mu_min = -1. * sqrt(1.0 - ( storage.r_inner[0] / virtual_sub_packet.current_r) ** 2)
        mu_bin = (1 - mu_min) / packet.virtual_packet
        virtual_sub_packet.current_mu = mu_min + ((i + rk_double(&mt_state)) * mu_bin)
        if packet.type == 11: # Packet type 11 is for virtual_mode. if v packet is created at the inner boundary
            #this is a virtual packet calculation based on a newly born packet - so the weights are more subtle than for a isotropic emission process
            weight = 2. * virtual_sub_packet.current_mu / virtual_sub_packet.virtual_packet
        elif packet.type ==12:#if v packet is created at event
            #isotropic emission case ("normal case") for a source in the ejecta
            weight = (1 - mu_min) / 2. / virtual_sub_packet.virtual_packet

        #the virtual packets are spawned with known comoving frame energy and frequency
        doppler_factor_ratio = (1 - (virtual_sub_packet.current_mu *virtual_sub_packet.current_r
                                     * storage.inverse_time_explosion * inverse_c)) / (
            1 - (virtual_sub_packet.current_mu * virtual_sub_packet.current_r
                 * storage.inverse_time_explosion * inverse_c))

        virtual_sub_packet.current_energy *=  doppler_factor_ratio
        virtual_sub_packet.current_nu *= doppler_factor_ratio

        reabsorbed = montecarlo_one_packet_loop(storage,&virtual_sub_packet,enable_bf)

        if virtual_sub_packet.current_nu < storage.spectrum_end_nu and \
                        storage.spectrum_start_nu < virtual_sub_packet.current_nu:
            virt_id_nu = floor(( virtual_sub_packet.current_nu - storage.spectrum_start_nu) / storage.spectrum_delta_nu)
            storage.spectrum_virt_nu[virt_id_nu] += virtual_sub_packet.current_energy * weight

    return reabsorbed





##################################################



#################The new montecarlo one packet#####
cdef int_type_t montecarlo_one_packet(StorageModel storage, Packet_struct*packet,
                                      int_type_t enable_bf):
    """
    montecarlo_one_packet decides if we have a real or an virtual packet.
    should we do a check if the packet is consistent
    """

    if (packet.type < 10):
        #Do a real packet
        reabsorbed = montecarlo_one_packet_loop(storage,packet,enable_bf)
    elif (packet.type > 10) and (packet.type < 20):
        #Do a virtual packet
        reabsorbed = montecarlo_one_virtuel_packet(storage,packet,enable_bf)

    return reabsorbed





#########################



################## MonteCarlo One Packet Loop###
cdef int_type_t montecarlo_one_packet_loop(StorageModel storage, Packet_struct* packet,int_type_t enable_bf):


    #@@@ MAIN LOOP START @@@
    #Type = 0 Disabled,  1 R-Packet, 2 K-Packet, 3 I-Packet, 11 VR-Packet from the inner boundary, 12 VR-Packet from anywhere in the ejecta, 99 escaped

    #Set the inital tau_event for this packet
    packet.tau_event = -log(rk_double(&mt_state))

    while True:
    #Do R Packet
        if packet.type == 1:
            reabsorbed = do_r_packet(storage,packet,enable_bf)
    #Do K Packet
        elif packet.type ==2:
            pass
#            do_i_packet(storage,packet,enable_bf)
    #Do I Packet
        elif packet.type == 3:
            do_i_packet(storage,packet,enable_bf) # The macro atom

    #Do V Packet
        elif (packet.type > 10) and (packet.type <20):
            pass

    #Escaped Packet. This is the last if
        elif packet.type > 90:
            break
    return reabsorbed


################################################
#
#
cdef int_type_t do_r_packet(StorageModel storage, Packet_struct*packet, int_type_t enable_bf):


    #print("Do r packet")

    cdef float_type_t nu_electron = 0.0
    cdef float_type_t comov_nu = 0.0
    cdef float_type_t comov_energy = 0.0
    cdef float_type_t energy_electron = 0.0
    cdef int_type_t emission_line_id = 0
    cdef int_type_t activate_level_id = 0

    #doppler factor definition
    cdef float_type_t doppler_factor = 0.0
    cdef float_type_t old_doppler_factor = 0.0
    cdef float_type_t inverse_doppler_factor = 0.0

    cdef float_type_t tau_line = 0.0
    cdef float_type_t tau_electron = 0.0
    cdef float_type_t tau_combined = 0.0


    #defining distances
    cdef float_type_t d_inner = 0.0
    cdef float_type_t d_outer = 0.0
    cdef float_type_t d_line = 0.0
    cdef float_type_t d_continuum = 0.0

    cdef int_type_t reabsorbed = 0
    cdef float_type_t nu_line = 0.0

    cdef int_type_t virtual_close_line = 0
    cdef int_type_t j_blue_idx = -1


    cdef Packet_struct virtual_sub_packet

    #Initializing tau_event if it's a real packet
    #if (packet.type == 1):
    #    packet.tau_event = -log(rk_double(&mt_state))

    #For a virtual packet tau_event is the sum of all the tau's that the packet passes.

    #Check if current is smaller than the nu of the line with the highest frequency
    if packet.current_nu < storage.line_list_nu[storage.no_of_lines - 1] and packet.last_line == 0:
        print("WARNING comoving nu less than nu_line shouldn't happen; MAIN LOOP")
        print("current_nu %g" %packet.current_nu)
        packet.last_line = 1



    #check if we are at the end of linelist
    if packet.last_line == 0:
        nu_line = storage.line_list_nu[packet.current_line_id]

    #check if the last line was the same nu as the current line
    if packet.close_line == 1:
        #if yes set the distance to the line to 0.0
        d_line = 0.0
        #reset close_line
        packet.close_line = 0

        #CHECK if 3 lines in a row work
        #print "CLOSE LINE WAS 1"
    else:# -- if close line didn't happen start calculating the the distances
    #print "CLOSE LINE WASN'T 1"
        # ------------------ INNER DISTANCE CALCULATION ---------------------
        if packet.recently_crossed_boundary == 1:
            #if the packet just crossed the inner boundary it will not intersect again unless it interacts. So skip
            #calculation of d_inner
            d_inner = miss_distance
        else:
            #compute distance to the inner shell
            d_inner = compute_distance2inner(packet.current_r, packet.current_mu, storage.r_inner[packet.current_shell_id])
            # ^^^^^^^^^^^^^^^^^^ INNER DISTANCE CALCULATION ^^^^^^^^^^^^^^^^^^^^^

        # ------------------ OUTER DISTANCE CALCULATION ---------------------
        #computer distance to the outer shell basically always possible
        d_outer = compute_distance2outer(packet.current_r, packet.current_mu, storage.r_outer[packet.current_shell_id])
        # ^^^^^^^^^^^^^^^^^^ OUTER DISTANCE CALCULATION ^^^^^^^^^^^^^^^^^^^^^

        # ------------------ LINE DISTANCE CALCULATION ---------------------
        if packet.last_line == 1:
            d_line = miss_distance
        else:
            d_line = compute_distance2line(packet.current_r, packet.current_mu, packet.current_nu, nu_line,
                                           storage.time_explosion,
                                           storage.inverse_time_explosion,
                                           storage.line_list_nu[packet.current_line_id - 1],
                                           storage.line_list_nu[packet.current_line_id + 1],
                                           packet.current_shell_id)
            # ^^^^^^^^^^^^^^^^^^ LINE DISTANCE CALCULATION ^^^^^^^^^^^^^^^^^^^^^

        # ------------------ ELECTRON DISTANCE CALCULATION ---------------------
        # a virtual packet should never be stopped by continuum processes
        if (packet.type > 10):
            d_continuum = miss_distance
        else:
            #d_continuum is the distance to the nex continuums event. The continuums distance contains the bound-free, the free-free and, the thomson distance.
            # d_continuum = compute_distance2continum(packet.tau_event,
            #                                         packet.current_nu,
            #                                         storage.kappa_bf_nu_view,
            #                                         storage.bf_nu_bin_view,
            #                                         storage.electron_density[packet.current_shell_id],
            #                                         storage.sigma_thomson,
            #                                         &packet.current_shell_id,
            #                                         storage.disableBoundFree)
            d_continuum = compute_distance2continum(storage,packet,enable_bf)

            #---Disable cont opa.
            #d_continuum = miss_distance



            #d_continuum = compute_distance2electron(current_r[0], current_mu[0], tau_event,
            #                                      storage.inverse_electron_density[current_shell_id[0]] * \
            #                                      storage.inverse_sigma_thomson)
            # ^^^^^^^^^^^^^^^^^^ ELECTRON DISTANCE CALCULATION ^^^^^^^^^^^^^^^^^^^^^


    # ------------------------------ LOGGING ---------------------- (with precompiler IF)
    IF packet_logging == True:
        packet_logger.debug('%s\nCurrent packet state:\n'
                            'current_mu=%s\n'
                            'current_nu=%s\n'
                            'current_energy=%s\n'
                            'd_inner=%s\n'
                            'd_outer=%s\n'
                            'd_continuum=%s\n'
                            'd_line=%s\n%s',
                            '-' * 80,
                            packet.current_mu,
                            packet.current_nu,
                            packet.current_energy,
                            d_inner,
                            d_outer,
                            d_continuum,
                            d_line,
                            '-' * 80)
        if packet.tau_event < 0:
            packet_logger.warning('tau_event is less than 0, %f '% packet.tau_event)

        if isnan(d_inner) or d_inner < 0:
            packet_logger.warning('d_inner is nan or less than 0')
        if isnan(d_outer) or d_outer < 0:
            packet_logger.warning('d_outer is nan or less than 0')
        if isnan(d_continuum) or d_continuum < 0:
            packet_logger.warning('d_continuum is nan or less than 0, %f'% d_continuum)
        if isnan(d_line) or d_line < 0:
            packet_logger.warning('d_line is nan or less than 0')

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^ LOGGING # ^^^^^^^^^^^^^^^^^^^^^^^^^^^


            #if ((abs(storage.r_inner[current_shell_id[0]] - current_r[0]) < 10.0) and (current_mu[0] < 0.0)):
            #print "d_outer %g, d_inner %g, d_line %g, d_continuum %g" % (d_outer, d_inner, d_line, d_continuum)
            #print "nu_line %g current_nu[0] %g current_line_id[0] %g" % (nu_line, current_nu[0], current_line_id[0])
            #print "%g " % current_shell_id[0]

    # ------------------------ PROPAGATING OUTWARDS ---------------------------
    if (d_outer <= d_inner) and (d_outer <= d_continuum) and (d_outer < d_line):
        #moving one zone outwards. If it's already in the outermost one this is escaped. Otherwise just move, change the zone index
        #and flag as an outwards propagating packet
        move_packet(storage,packet,d_outer)
        # move_packet(current_r, current_mu, current_nu[0], current_energy[0], d_outer, storage.js, storage.nubars,
        #             storage.inverse_time_explosion,
        #             current_shell_id[0], virtual_packet)
        #for a virtual packet, add on the opacity contribution from the continuum
        if (packet.type > 10) and (packet.type <20):
            packet.tau_event += (d_outer * storage.electron_densities[packet.current_shell_id] * storage.sigma_thomson)
        else:
            packet.tau_event = -log(rk_double(&mt_state))

        if (packet.current_shell_id < storage.no_of_shells - 1): # jump to next shell
            packet.current_shell_id += 1
            packet.recently_crossed_boundary = 1


        else:
            # ------------------------------ LOGGING ---------------------- (with precompiler IF)
            IF packet_logging == True:
                packet_logger.debug(
                    'Packet has left the simulation through the outer boundary nu=%s mu=%s energy=%s',
                    packet.current_nu, packet.current_mu, packet.current_energy)

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^ LOGGING # ^^^^^^^^^^^^^^^^^^^^^^^^^^^

            reabsorbed = 0
            #print "That one got away"
            packet.type = 99 #escaped
            return reabsorbed
            #
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^ PROPAGATING OUTWARDS ^^^^^^^^^^^^^^^^^^^^^^^^^^


    # ------------------------ PROPAGATING inwards ---------------------------
    elif (d_inner <= d_outer) and (d_inner <= d_continuum) and (d_inner < d_line):
        #moving one zone inwards. If it's already in the innermost zone this is a reabsorption
        move_packet(storage ,packet, d_inner)
        #move_packet(current_r, current_mu, current_nu[0], current_energy[0], d_inner, storage.js, storage.nubars,
        #            storage.inverse_time_explosion,
        #            current_shell_id[0], virtual_packet)

        #for a virtual packet, add on the opacity contribution from the continuum
        if (packet.type > 10) and (packet.type <20):
            packet.tau_event += (d_inner * storage.electron_densities[packet.current_shell_id] * storage.sigma_thomson)
        else:
            packet.tau_event = -log(rk_double(&mt_state))

        if packet.current_shell_id > 0:
            packet.current_shell_id -= 1
            packet.recently_crossed_boundary = -1



        else:
            # ------------------------------ LOGGING ---------------------- (with precompiler IF)
            IF packet_logging == True:
                packet_logger.debug(
                    'Packet has left the simulation through the inner boundary nu=%s mu=%s energy=%s',
                    packet.current_nu, packet.current_mu, packet.current_energy)
            reabsorbed = 1
            packet.type = 98 #reabsorbed
            return reabsorbed
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^ LOGGING # ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^ PROPAGATING INWARDS ^^^^^^^^^^^^^^^^^^^^^^^^^^


    # ------------------------ ELECTRON SCATTER EVENT ELECTRON ---------------------------
    elif (d_continuum <= d_outer) and (d_continuum <= d_inner) and (d_continuum < d_line):
        # we should never enter this branch for a virtual packet
        # ------------------------------ LOGGING ----------------------
        IF packet_logging == True:
            packet_logger.debug('%s\nElectron scattering occuring\n'
                                'current_nu=%s\n'
                                'current_mu=%s\n'
                                'current_energy=%s\n',
                                '-' * 80,
                                packet.current_nu,
                                packet.current_mu,
                                packet.current_energy)

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^ LOGGING # ^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@Continum Event@@@@@@@@@@@
        #Sample new z normalized to the sum of the cont kappas.
        #for i in range(len(storage.bf_nu_bins)-1):
        #    if ((current_nu[0] <= storage.bf_nu_bins[i]) and (current_nu[0] > storage.bf_nu_bin[i+1])):
        #        kappa_bf_nu = [i]
        #       break
        #TODO:chagne to Standard API
        kappa_bf = getGrayKappaBFbyNu(packet.current_nu, storage.bf_nu_bin_view, storage.kappa_bf_nu_view,&packet.current_shell_id,storage.disableBoundFree)
        kappa_ff = 0#TODO: set kappa ff
        kappa_th = storage.electron_densities[packet.current_shell_id] * storage.sigma_thomson
        kappa_cont = kappa_bf + kappa_th + kappa_ff

        #Select the cont. event which occurs by using the new z
        z = kappa_cont * rk_double(&mt_state)
        if (z > (kappa_ff + kappa_bf)):
            #electron scattering



            #compute the doppler_factor
            doppler_factor = move_packet(StorageModel,packet, d_continuum)

            comov_nu = packet.current_nu * doppler_factor
            comov_energy = packet.current_energy * doppler_factor

            #new mu chosen
            packet.current_mu = 2 * rk_double(&mt_state) - 1
            inverse_doppler_factor = 1 / (
                1 - (packet.current_mu * packet.current_r * storage.inverse_time_explosion * inverse_c))
            packet.current_nu = comov_nu * inverse_doppler_factor
            packet.current_energy = comov_energy * inverse_doppler_factor

        elif (z > kappa_ff):
            #print('!Bound-Free at %g, kappa_bf = %g, kappa_th %g '%(current_nu[0],kappa_bf,kappa_th ))
            #bound-free event
            #Get new mu

            doppler_factor = move_packet(storage,packet,d_continuum)
            comov_energy = packet.current_energy * doppler_factor

            packet.current_mu = 2 * rk_double(&mt_state) - 1
            comov_nu = rand_nu_planck(storage.t_electron[packet.current_shell_id],storage.nu_sampling_lower_lim,storage.nu_sampling_upper_lim,storage.black_body_peak)


            inverse_doppler_factor = 1 / (
                1 - (packet.current_mu * packet.current_r * storage.inverse_time_explosion * inverse_c))

            #comov_energy = current_energy[0] * doppler_factor

            packet.current_nu = comov_nu *inverse_doppler_factor
            packet.current_energy = comov_energy *inverse_doppler_factor

            #since we have drawn a new nu we have to erase the line memories of our photon and create a new memories
            #current_line_id[0] = getNextLineId(storage.line_list_nu, comov_nu, 0, storage.no_of_lines -1)
            #TODO remove this quick fix
            i = getNextLineId(storage.line_list_nu_view, comov_nu, 0, storage.no_of_lines -1)
            #                while storage.line_list_nu[i] > comov_nu:
            #                    print(i)
            #                    i-=1
            packet.current_line_id = i


            if packet.current_line_id == storage.no_of_lines:
                #setting flag that the packet is off the red end of the line list
                packet.last_line = 1
            else:
                packet.last_line = 0

                #print('bound-free DEBUG:')
                #print('current_mu=%g'%current_mu[0])
                #print('current_nu=%g'%current_nu[0])
                #print('comov_nu=%g'%comov_nu)
                #print('current_line_id=%d'%current_line_id[0])
                #print('nu line - 1=%g'%storage.line_list_nu[current_line_id[0]-1])
                #print('nu line=%g'%storage.line_list_nu[current_line_id[0]])
                #print('nu line=%g +1'%storage.line_list_nu[current_line_id[0]+1])


                # ------------------------------ LOGGING ----------------------
            IF packet_logging == True:
                packet_logger.debug("\n Bound free event occuring\n comoving_nu=%s\n comoving_energy=%s\n current_nu=%s\n current_mu=%s\n current_energy=%s\n line_id=%d\n line_nu=%s\n",
                                    comov_nu,
                                    comov_energy,
                                    packet.current_nu,
                                    packet.current_mu,
                                    packet.current_energy,
                                    packet.current_line_id,
                                    storage.line_list_nu[packet.current_line_id])


        else:
            print('!-!-!-!-!-!2')
            #free-free event
            # ------------------------------ LOGGING ----------------------
            IF packet_logging == True: #THIS SHOULD NEVER HAPPEN SINCE KAPPA_ff is 0
                packet_logger.debug("\n Free free event occuring\n comoving_nu=%s\n comoving_energy=%s\n current_nu=%s\n current_mu=%s\n current_energy=%s\n line_id=%d\n line_nu=%s\n",
                                    comov_nu,
                                    comov_energy,
                                    packet.current_nu,
                                    packet.current_mu,
                                    packet.current_energy,
                                    packet.current_line_id,
                                    storage.line_list_nu[packet.current_line_id])

                #If electron scattering occurs then correct the photon with the doppler_factor

        #If a bound-free event occurs sample a new nu for the photon

        #If a free-free event occurs, do ??

        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #
        #     doppler_factor = move_packet(current_r, current_mu, current_nu[0], current_energy[0], d_continuum,
        #                              storage.js, storage.nubars
        #     , storage.inverse_time_explosion, current_shell_id[0], virtual_packet)
        #
        #     comov_nu = current_nu[0] * doppler_factor
        #     comov_energy = current_energy[0] * doppler_factor
        #
        # #new mu chosen
        #     current_mu[0] = 2 * rk_double(&mt_state) - 1
        #     inverse_doppler_factor = 1 / (
        #     1 - (current_mu[0] * current_r[0] * storage.inverse_time_explosion * inverse_c))
        #     current_nu[0] = comov_nu * inverse_doppler_factor
        #     current_energy[0] = comov_energy * inverse_doppler_factor
        # ------------------------------ LOGGING ----------------------
        #     print("\n Free-free event occuring\n comoving_nu=%s\n comoving_energy=%s\n current_nu=%s\n current_mu=%s\n current_energy=%s\n line_id=%d\n line_nu=%s\n"
        #           %
        #           (comov_nu,
        #            comov_energy,
        #            current_nu[0],
        #            current_mu[0],
        #            current_energy[0],
        #            current_line_id[0],
        #            storage.line_list_nu[current_line_id[0]]))

        IF packet_logging == True:
            packet_logger.debug('Continuums event occured\n'
                                'current_nu=%s\n'
                                'current_mu=%s\n'
                                'current_energy=%s\n%s',
                                packet.current_nu,
                                packet.current_mu,
                                packet.current_energy,
                                '-' * 80)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^ LOGGING # ^^^^^^^^^^^^^^^^^^^^^^^^^^^

        packet.tau_event = -log(rk_double(&mt_state))

        #scattered so can re-cross a boundary now
        packet.recently_crossed_boundary = 0
        #We've had an electron scattering event in the SN. This corresponds to a source term - we need to spawn virtual packets now
        #TODO: Check if virtual_packet is ok here
        if (packet.virtual_packet > 0):
            #print "AN ELECTRON SCATTERING HAPPENED: CALLING VIRTUAL PARTICLES!!!!!!"
            virtual_sub_packet = packet[0] #create a virtual sub packet
            virtual_sub_packet.type = 12 #type for virtual packet
            montecarlo_one_packet(storage,&virtual_sub_packet,enable_bf)

    # ^^^^^^^^^^^^^^^^^^^^^^^^^ SCATTER EVENT LINE ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # ------------------------ LINE SCATTER EVENT  ---------------------------
    elif (d_line <= d_outer) and (d_line <= d_inner) and (d_line <= d_continuum):
    #Line scattering
        #It has a chance to hit the line
        if (packet.type > 10) and (packet.type < 20): #TODO check if this is ok
            j_blue_idx = packet.current_shell_id * storage.line_lists_j_blues_nd + packet.current_line_id
            increment_j_blue_estimator(storage,packet,d_line,j_blue_idx)

        tau_line = storage.line_lists_tau_sobolevs[
            packet.current_shell_id * storage.line_lists_tau_sobolevs_nd + packet.current_line_id]


        kappa_bf = getGrayKappaBFbyNu(packet.current_nu, storage.bf_nu_bin_view, storage.kappa_bf_nu_view,&packet.current_shell_id,storage.disableBoundFree)
        kappa_ff = 0#TODO: set kappa ff
        kappa_th = storage.electron_densities[packet.current_shell_id] * storage.sigma_thomson
        kappa_cont = kappa_bf + kappa_th + kappa_ff
        #print("line scatter: kappa_cont = %g"%kappa_cont)
        #the kappa_th contains the tau_electron
        tau_continuum = kappa_cont * d_line
        ##The tau electron is disabled for debugging
        #tau_electron = storage.sigma_thomson * storage.electron_density[packet.current_shell_id] * d_line
        #tau_electron = 0
        tau_combined = tau_line + tau_continuum
        #            tau_combined = tau_line + tau_electron
        #print("tau_line = %g"%tau_line)
        #print("tau_electron = %g"%tau_electron)
        #print("d_line = %g"%d_line)
        #print("d_cont = %g"%d_continuum)

        # ------------------------------ LOGGING ----------------------

        # print("\n Line event occuring\n comoving_nu=%s\n comoving_energy=%s\n current_nu=%s\n current_mu=%s\n current_energy=%s\n line_id=%d\n line_nu=%s\n"
        #       %
        #       (comov_nu,
        #        comov_energy,
        #        current_nu[0],
        #        current_mu[0],
        #        current_energy[0],
        #        current_line_id[0],
        #        storage.line_list_nu[current_line_id[0]]))

        IF packet_logging == True:
            packet_logger.debug('%s\nEntering line scattering routine\n'
                                'Scattering at line %d (nu=%s)\n'
                                'tau_line=%s\n'
                                'tau_electron=%s\n'
                                'tau_combined=%s\n'
                                'tau_event=%s\n',
                                '-' * 80,
                                packet.current_line_id + 1,
                                storage.line_list_nu[packet.current_line_id],
                                tau_line,
                                tau_electron,
                                tau_combined,
                                packet.tau_event)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^ LOGGING # ^^^^^^^^^^^^^^^^^^^^^^^^^^^

        #Advancing to next line
        packet.current_line_id += 1

        #check for last line
        if packet.current_line_id >= storage.no_of_lines:
            packet.current_line_id = storage.no_of_lines
            packet.last_line = 1

        if (packet.type >10):
            #here we should never stop the packet - only account for its tau increment from the line
            packet.tau_event += tau_line

        else:
            #Check for line interaction
            if packet.tau_event < tau_combined:

            #A line interaction trigers a i packet


                old_doppler_factor = move_packet(storage,packet,d_line)
                packet.current_mu =  2 * rk_double(&mt_state) - 1
                inverse_doppler_factor = 1/(
                        1- (packet.current_mu * packet.current_r * storage.inverse_time_explosion * inverse_c)
                        )

                packet.comov_nu = packet.current_nu * old_doppler_factor
                packet.comov_energy = packet.current_energy * old_doppler_factor

                packet.current_energy = packet.comov_energy * inverse_doppler_factor

                if storage.line_interaction_id ==0: #scatter
                    emission_line_id = packet.current_line_id -1
                    packet.current_nu = storage.line_list_nu[emission_line_id] * inverse_doppler_factor
                    nu_line = storage.line_list_nu[emission_line_id]
                    packet.current_line_id = emission_line_id + 1
                elif storage.line_interaction_id >= 1:# downbranch & macro
                    # print("set packet type to 3")
                    packet.distance_to_move =d_line
                    # print("nu in do_r %g"%packet.current_nu)
                    # print("old_doppler in do_r %g"%old_doppler_factor)
                    packet.type = 3 # 3 I-Packet
                    reabsorbed = 0
                    return  reabsorbed

                IF packet_logging == True:
                    packet_logger.debug('Line interaction over. New Line %d (nu=%s; rest)', emission_line_id + 1,
                                        storage.line_list_nu[emission_line_id])

                packet.tau_event = -log(rk_double(&mt_state))
                packet.recently_crossed_boundary = 0

            else: #tau_event > tau_line no interaction so far
                #reducing event tau to take this probability into account
                packet.tau_event -= tau_line
                if packet.tau_event < 0:
                    print('tau_event < 0, %f' % packet.tau_event)
                    #print('tau_line: %f '% packet.tau_line[0])
                    print('tau_combined %f '%tau_combined)
                    a = input()

                IF packet_logging == True:
                    packet_logger.debug('No line interaction happened. Tau_event decreasing %s\n%s', packet.tau_event,'-' * 80)

        IF packet_logging == True: #THIS SHOULD NEVER HAPPEN
            if packet.tau_event < 0:
                logging.warn('tau_event less than 0: %s', packet.tau_event)

        if packet.last_line == 0: #Next line is basically the same just making sure we take this into account
            if abs(storage.line_list_nu[packet.current_line_id] - nu_line) / nu_line < 1e-7:
                packet.close_line = 1

    if (packet.virtual_packet > 0):
        if (packet.tau_event > 10.0):
            packet.tau_event = 100.0
            reabsorbed = 0
            packet.type = 91 #


#decide if we want to use the macroatom

                #here comes the macro atom

                #print "A LINE EVENT HAPPENED AT R = %g in shell %g which has range %g to %g" % (current_r[0], current_shell_id[0], storage.r_inner[current_shell_id[0]], storage.r_outer[current_shell_id[0]])
                #print "d_outer %g, d_inner %g, d_line %g, d_continuum %g" % (d_outer, d_inner, d_line, d_continuum)
                #print "nu_line_1 %g nu_line_2 %g" % (storage.line_list_nu[current_line_id[0]-1], storage.line_list_nu[current_line_id[0]])
                #print "nu_line_1 %g nu_line_2 %g" % (current_line_id[0]-1, current_line_id[0])
                #print "last_line %g" % (last_line[0])



cdef void do_i_packet(StorageModel storage, Packet_struct* packet,enable_bf):
    """
    The i packet dose the line interaction. At the moment it can do scatter, downbranch and macro.
    :param storage:
    :param packet:
    :param enable_bf:
    :return:
    """
    cdef float_type_t old_doppler_factor, inverse_doppler_factor, comov_nu, comov_energy, nu_line
    cdef int_type_t  activate_level_id, emission_line_id
    ##FOR_DEBUG:
    # print("Do i-packet")
    ##



    # print("nu in do_i %g"%packet.current_nu)
    # print("i do line nr. %d"%packet.current_line_id)
    # print("old_doppler in do_i %g"%old_doppler_factor)
    packet.current_mu =  2 * rk_double(&mt_state) - 1
    inverse_doppler_factor = 1 / (
         1 - (packet.current_mu * packet.current_r * storage.inverse_time_explosion * inverse_c))

    packet.current_energy = packet.comov_energy * inverse_doppler_factor
    # print("all necessary data are stored in local variabels")
    # if packet.comov_nu < storage.line_list_nu[packet.current_line_id-1]:
    #     print "/BEGIN//////WARNING comoving nu less than nu_line shouldn't happen:"
    #     print "comov_nu = ", packet.comov_nu
    #     print "nu_line", storage.line_list_nu[packet.current_line_id-1]
    #     print "current line list id", packet.current_line_id
    #     print "last line list id", storage.no_of_lines


    # print("start with macro atom")
        #here comes the macro atom
    activate_level_id = storage.line2macro_level_upper[packet.current_line_id - 1]
    emission_line_id = macro_atom(storage, packet, activate_level_id)
    # print('macroatom is finished')


    packet.current_nu = storage.line_list_nu[emission_line_id] * inverse_doppler_factor
    nu_line = storage.line_list_nu[emission_line_id]
    packet.current_line_id = emission_line_id + 1
    IF packet_logging == True:
        packet_logger.debug('Line interaction over. New Line %d (nu=%s; rest)', emission_line_id + 1,
                            storage.line_list_nu[emission_line_id])
    packet.tau_event = -log(rk_double(&mt_state))
    packet.recently_crossed_boundary = 0

    if packet.virtual_packet > 0:
        virtual_sub_packet = packet#create a virtual sub packet
        virtual_sub_packet.type = 12 #type for virtual packet
        virtual_sub_packet.close_line = 0
        if packet.last_line == 0:
            if abs(storage.line_list_nu[packet.current_line_id] - nu_line) / nu_line < 1e-7:
                virtual_sub_packet.close_line = 1
            montecarlo_one_packet(storage,virtual_sub_packet,enable_bf)

    # if packet.comov_nu < storage.line_list_nu[packet.current_line_id-1]:
    #     print "/END//////WARNING comoving nu less than nu_line shouldn't happen:"
    #     print "comov_nu = ", packet.comov_nu
    #     print "nu_line", storage.line_list_nu[packet.current_line_id-1]

    #There are no virtual packets started for i packet TODO: do this
    # print("set type to 1")

    packet.type = 1 #set type to radiation



