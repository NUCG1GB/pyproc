import argparse
import numpy as np
import json, os, sys
import pickle

from pyproc.cherab_bridge.cherab_plasma import CherabPlasma
from pyproc.atomic import get_ADAS_dict
from pyproc.analyse import AnalyseSynthDiag


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def read_input_dict(input_dict_str):

    with open(input_dict_str, mode='r', encoding='utf-8') as f:
        # Remove comments
        with open("temp.json", 'w') as wf:
            for line in f.readlines():
                if line[0:2] == '//' or line[0:1] == '#':
                    continue
                wf.write(line)

    with open("temp.json", 'r') as f:
        input_dict = json.load(f)

    os.remove('temp.json')

    return input_dict

def recover_line_int_particle_bal(ADAS_dict, res_dict, sion_H_transition=[3,2], ne_scal=1.0):
    """
        ESTIMATE RECOMBINATION/IONISATION RATES USING ADF11 ACD, SCD COEFF
    """

    for diag_key in res_dict.keys():
        for chord_key in res_dict[diag_key].keys():

            if (res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne'] and
                res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']):

                fit_ne = ne_scal*res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']
                fit_Te = res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']
                # Use highest Te estimate from continuum (usually not available from experiment)
                # fit_Te = res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_400_500']

                print('Ionization/recombination, LOS id= :', diag_key, ' ', chord_key)

                # area_cm2 = 2*pi*R*dW
                w2unmod = res_dict[diag_key][chord_key]['chord']['w2']
                # area_cm2 = 1.0e04 * 2.*np.pi*res_dict[diag_key][chord_key]['chord']['d2unmod']*res_dict[diag_key][chord_key]['coord']['v2'][0]
                area_cm2 = 1.0e04 * 2. * np.pi * w2unmod * \
                           res_dict[diag_key][chord_key]['chord']['p2'][0]
                idxTe, Te_val = find_nearest(ADAS_dict['adf11']['1'].Te_arr, fit_Te)
                idxne, ne_val = find_nearest(ADAS_dict['adf11']['1'].ne_arr, fit_ne * 1.0E-06)

                # Recombination:
                # NOTE: D7-2 line must be read from standard ADAS adf15 data as it is above the
                # max transition available in the Ly-trapped adf15 files
                for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():
                    if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == '7' and \
                                    res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == '2':
                        h72 = res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit'] + \
                              res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom']
                        srec = 1.0E-04 * area_cm2 * h72 * 4. * np.pi * \
                               ADAS_dict['adf11']['1'].acd[idxTe, idxne] / \
                               ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].pec[idxTe, idxne]

                        # Add to results dict
                        res_dict[diag_key][chord_key]['los_int']['adf11_fit'] = {'Srec': srec, 'units': 's^-1'}

                # Ionization:
                # Use Ly-trapping adf15,11 data if available
                for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():
                    if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == str(sion_H_transition[0]) and \
                                    res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == str(sion_H_transition[1]):
                        h_excit = res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit']
                        sion = 1.0E-04 * area_cm2 * h_excit * 4. * np.pi * \
                               ADAS_dict['adf11']['1'].scd[idxTe, idxne] / \
                        ADAS_dict['adf15']['1']['1'][H_line_key + 'excit'].pec[idxTe, idxne]

                        # Add to results dict
                        res_dict[diag_key][chord_key]['los_int']['adf11_fit']['Sion'] = sion

def recover_delL_atomden_product(ADAS_dict, res_dict, sion_H_transition=[3,2]):
    """
        ESTIMATE DEL_L * ATOMIC DENSITY PRODUCT FROM H LINE ASSUMING EXCITATION DOMINATED
    """
    for diag_key in res_dict.keys():
        for chord_key in res_dict[diag_key].keys():

            if (res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne'] and
                    res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']):

                fit_ne = res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']
                fit_Te = res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']

                print('delL * n0 from transition', str(sion_H_transition), ' LOS id= :', diag_key, ' ', chord_key)

                for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():
                    if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == str(sion_H_transition[0]) and \
                                    res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == str(sion_H_transition[1]):
                        h_ij = res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit'] + \
                              res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom']
                        idxTe, Te_val = find_nearest(
                            ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].Te_arr, fit_Te)
                        idxne, ne_val = find_nearest(
                            ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].ne_arr, fit_ne * 1.0E-06)
                        n0delL_Hij_tmp = 4. * np.pi * 1.0e-04 * h_ij / (
                            ADAS_dict['adf15']['1']['1'][H_line_key + 'excit'].pec[idxTe, idxne] * ne_val)
                        n0delL_Hij_tmp = n0delL_Hij_tmp * 1.0e06 * 1.0e-02  # convert to m^-2
                        ##### Add fit n0*delL result to dictionary
                        res_dict[diag_key][chord_key]['los_int']['n0delL_fit'] = {'n0delL': n0delL_Hij_tmp,
                                                                         'units': 'm^-2'}

if __name__=='__main__':

    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Run cherab_bridge')
    parser.add_argument('cherab_bridge_input_dict')
    args = parser.parse_args()

    # Handle the input arguments
    input_dict_file = args.cherab_bridge_input_dict

    if os.path.isfile(input_dict_file):
        print('Found input dictionary: ', input_dict_file)
        input_dict = read_input_dict(input_dict_file)
    else:
        sys.exit(input_dict_file + ' not found')

    # Handle the input arguments
    pyproc_case = input_dict['save_dir']+input_dict['pyproc_case']

    if os.path.isdir(pyproc_case):
        if os.path.isfile(pyproc_case + '/pyproc.2ddata.pkl'):
            print('Found EDGE2D pickled data: ', pyproc_case + '/pyproc.2ddata.pkl')
            infile = open(pyproc_case + '/pyproc.2ddata.pkl', 'rb')
            edge2d_pkl = pickle.load(infile)
        else:
            sys.exit('Pyproc EDGE2D pickle file not found.')
        if os.path.isfile(pyproc_case + '/pyproc.proc_synth_diag.json'):
            print('Found synthetic diagnostic data file: ', pyproc_case + '/pyproc.proc_synth_diag.json')
            # Read synth diag saved data
            try:
                with open(pyproc_case +  '/pyproc.proc_synth_diag.json', 'r') as f:
                    synth_diag_dict = json.load(f)
            except IOError as e:
                raise
        else:
            sys.exit('Pyproc synthetic diagnostic data file not found.')
        if os.path.isfile(pyproc_case + '/pyproc.proc_synth_diag.json'):
            print('Found synthetic diagnostic processed data file: ', pyproc_case + '/pyproc.proc_synth_diag.json')
        else:
            sys.exit('Pyproc synthetic diagnostic processed data file not found.')

    else:
        sys.exit('Pyproc case ' + pyproc_case + ' not found.')

    # Inputs from cherab_bridge_input_dict
    import_jet_surfaces = input_dict['cherab_options']['import_jet_surfaces']
    include_reflections = input_dict['cherab_options']['include_reflections']
    spectral_bins = input_dict['cherab_options']['spectral_bins']
    pixel_samples = input_dict['cherab_options']['pixel_samples']
    spec_line_dict = input_dict['spec_line_dict']
    diag_list = input_dict['diag_list']
    read_ADAS = input_dict['read_ADAS']
    stark_transition = input_dict['cherab_options']['stark_transition']
    ff_fb = input_dict['cherab_options']['ff_fb_emission']
    sion_H_transition = input_dict['cherab_options']['Sion_H_transition']

    # Read ADAS data
    ADAS_dict = get_ADAS_dict(input_dict['save_dir'],
                              spec_line_dict, adf11_year=12, restore=not input_dict['read_ADAS'])

    # Generate cherab plasma
    plasma = CherabPlasma(edge2d_pkl, ADAS_dict, include_reflections = include_reflections,
                          import_jet_surfaces = import_jet_surfaces)

    # Create output dict
    outdict = {}

    # Loop through diagnostics, their LOS, integrate over Lyman/Balmer
    for diag_key in diag_list:
        for diag_key_pyproc, val in synth_diag_dict.items():
            if diag_key == diag_key_pyproc:
                outdict[diag_key] = {}
                for diag_chord, val in synth_diag_dict[diag_key].items():
                    los_p1 = val['chord']['p1']
                    los_p2 = val['chord']['p2']
                    los_w1 = 0.0
                    los_w2 = val['chord']['w2']
                    H_lines = spec_line_dict['1']['1']

                    outdict[diag_key][diag_chord] = {
                        'chord':{'p1':los_p1, 'p2':los_p2, 'w1':los_w1, 'w2':los_w2}
                    }
                    outdict[diag_key][diag_chord]['spec_line_dict'] = spec_line_dict

                    outdict[diag_key][diag_chord]['los_int'] = {'H_emiss': {}}

                    print(diag_key, los_p2)
                    for H_line_key, val in H_lines.items():
                        transition = (int(val[0]), int(val[1]))
                        wavelength = float(H_line_key)/10. #nm
                        min_wavelength = (wavelength)-1.0
                        max_wavelength = (wavelength)+1.0

                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                   include_excitation=True, include_recombination=False)
                        exc_radiance, exc_spectrum, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)

                        plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                   include_excitation=False, include_recombination=True)

                        rec_radiance, rec_spectrum, wave = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2,
                                                                                min_wavelength, max_wavelength,
                                                                                spectral_bins=spectral_bins, pixel_samples=pixel_samples)

                        outdict[diag_key][diag_chord]['los_int']['H_emiss'][H_line_key] = {
                            'excit':exc_radiance,
                            'recom':rec_radiance,
                            'units':'ph.s^-1.m^-2.sr^-1'
                        }

                        # Stark broadening
                        if stark_transition:
                            if transition == tuple(stark_transition):
                                plasma.define_plasma_model(atnum=1, ion_stage=0, transition=transition,
                                                           include_excitation=True, include_recombination=True,
                                                           include_stark=True)
                                radiance, spectrum, wave_arr = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2,
                                                                                    min_wavelength, max_wavelength,
                                                                                    spectral_bins=spectral_bins,
                                                                                    pixel_samples=pixel_samples,
                                                                                    display_progress=False)

                                outdict[diag_key][diag_chord]['los_int']['stark']={'cwl': wavelength, 'wave': wave_arr.tolist(),
                                                                                  'intensity': spectrum.tolist(),
                                                                                  'units': 'nm, ph s^-1 m^-2 sr^-1 nm^-1'}

                    # Free-free + free-bound using adaslib/continuo
                    if ff_fb:
                            plasma.define_plasma_model(atnum=1, ion_stage=0,
                                                       include_excitation=False, include_recombination=False,
                                                       include_stark=False, include_ff_fb=True)
                            min_wave = 300
                            max_wave = 500
                            spec_bins = 50
                            radiance, spectrum, wave_arr = plasma.integrate_los(los_p1, los_p2, los_w1, los_w2,
                                                                                min_wave, max_wave,
                                                                                spectral_bins=spec_bins,
                                                                                pixel_samples=pixel_samples,
                                                                                display_progress=False)

                            outdict[diag_key][diag_chord]['los_int']['ff_fb_continuum'] = {
                                'wave': wave_arr.tolist(),
                                'intensity': spectrum.tolist(),
                                'units': 'nm, ph s^-1 m^-2 sr^-1 nm^-1'}


    # Analyse synthetic spectra
    if input_dict['cherab_options']['analyse_synth_spec_features']:
        AnalyseSynthDiag.recover_line_int_Stark_ne(outdict)
        AnalyseSynthDiag.recover_line_int_ff_fb_Te(outdict)
        recover_line_int_particle_bal(ADAS_dict, outdict, sion_H_transition=sion_H_transition,
                                      ne_scal=1.0)
        recover_delL_atomden_product(ADAS_dict, outdict, sion_H_transition=sion_H_transition)

    # SAVE IN JSON FORMAT TO ENSURE PYTHON 2/3 COMPATIBILITY
    if include_reflections:
        savefile = pyproc_case + '/cherab_refl.synth_diag.json'
    else:
        savefile = pyproc_case + '/cherab.synth_diag.json'
    with open(savefile, mode='w', encoding='utf-8') as f:
        json.dump(outdict, f, indent=2)

    print('Saving cherab_bridge synthetic diagnostic data to:', savefile)
