
# Import external modules

import numpy as np
# import scipy.io as io
import os, errno
import json
from pyproc.pyprocprocess import PyprocProcess
from pyADASread import adas_adf11_read, adas_adf15_read, continuo_read
## http://lmfit.github.io/lmfit-py/parameters.html
from lmfit import minimize, Parameters, report_fit

class PyprocAnalyse(PyprocProcess):
    """
        Inherits from Pyproc and adds methods for analysis of synthetic spectra
    """
    def __init__(self, input_dict):
        self.input_dict = input_dict

        tmpstr = input_dict['tranfile'].replace('/','_')
        tmpstr = tmpstr[3:-5]
        savedir = input_dict['save_dir'] + tmpstr + '/'

        # Create dir from tran file, if it does not exist
        try:
            os.makedirs(savedir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.data2d_save_file = savedir +'pyproc.2ddata.pkl'
        self.synth_diag_save_file = savedir + 'pyproc.synth_diag.json'
        self.proc_synth_diag_save_file = savedir+ 'pyproc.proc_synth_diag.json'

        # Read all necessary ADAS data here and store in dict
        self.spec_line_dict = input_dict['spec_line_dict']
        self.ADAS_dict = {}
        Te_rnge = [0.2, 5000]
        ne_rnge = [1.0e11, 1.0e15]
        self.ADAS_samples = 100
        self.ADAS_dict['adf15'] = adas_adf15_read.get_adas_imp_PECs_interp(self.spec_line_dict,  Te_rnge, ne_rnge, npts=self.ADAS_samples, npts_interp=1000)
        # Also get adf11 for the ionisation balance fractional abundance. No Te_arr, ne_arr interpolation
        # available in the adf11 reader at the moment, so generate more coarse array (sloppy!)
        # TODO: add interpolation capability to the adf11 reader so that adf15 and adf11 are on the same Te, ne grid
        Te_arr_adf11 = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), 500)
        ne_arr_adf11 = np.logspace(np.log10(ne_rnge[0]), np.log10(ne_rnge[1]), 30)
        self.ADAS_dict['adf11'] = {}
        for atnum in self.spec_line_dict:
            if int(atnum) > 1:
                self.ADAS_dict['adf11'][atnum] = adas_adf11_read.get_adas_imp_adf11(int(atnum), Te_arr_adf11, ne_arr_adf11)
            elif int(atnum) == 1:
                self.ADAS_dict['adf11'][atnum] = adas_adf11_read.get_adas_H_adf11_interp(Te_rnge, ne_rnge,
                                                                                          npts=self.ADAS_samples, npts_interp=1000, pwr=True)
        super().__init__(self.ADAS_dict, tranfile=input_dict['tranfile'],
                         machine=input_dict['machine'],
                         interactive_plots = input_dict['interactive_plots'],
                         spec_line_dict=self.spec_line_dict,
                         diag_list=input_dict['diag_list'],
                         calc_synth_spec_features=input_dict['run_options']['calc_synth_spec_features'],
                         calc_NII_afg_feature=False,
                         save_synth_diag=True,
                         synth_diag_save_file=self.synth_diag_save_file,
                         data2d_save_file=self.data2d_save_file)

        if self.input_dict['run_options']['analyse_synth_spec_features']:
            # Read synth diag saved data
            try:
                with open(self.synth_diag_save_file, 'r') as f:
                    res_dict = json.load(f)
                self.analyse_synth_spectra(res_dict)
            except IOError as e:
                raise


    # Analyse synthetic spectra
    def analyse_synth_spectra(self, res_dict):

        # Estimate parameters and update res_dict. Call order matters since ne and Te
        # are needed as constraints
        #
        # Electron density estimate from Stark broadening of H6-2 line
        self.recover_line_int_Stark_ne(res_dict)

        # Electron temperature estimate from ff+fb continuum
        self.recover_line_int_ff_fb_Te(res_dict)

        # Recombination and Ionization
        self.recover_line_int_particle_bal(res_dict)

        # delL * neutral density from Ly-alpha assuming excitation dominated
        self.recover_delL_atomden_product(res_dict)

        if self.proc_synth_diag_save_file:
            with open(self.proc_synth_diag_save_file, mode='w', encoding='utf-8') as f:
                json.dump(res_dict, f, indent=2)

    @staticmethod
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    @staticmethod
    def residual_lorentz_52(params, x, data=None, eps_data=None):

        cwl = params['cwl'].value
        area = params['area'].value
        stark_fwhm = params['stark_fwhm'].value

        model = 1. / (np.power(np.abs(x - cwl), 5. / 2.) + np.power(stark_fwhm / 2.0, 5. / 2.))

        model_area = np.trapz(model, x=x)
        amp_scal = area / model_area
        model *= amp_scal

        if data is None:
            return model
        if eps_data is None:
            return (model - data)
        return (model - data) / eps_data

    @staticmethod
    def residual_continuo(params, wave, data=None, eps_data=None):
        delL = params['delL'].value
        te = params['te_360_400'].value
        ne = params['ne'].value

        model_ff, model_ff_fb = continuo_read.adas_continuo_py(wave, te, 1, 1)
        model_ff = model_ff * ne * ne * delL
        model_ff_fb = model_ff_fb * ne * ne * delL

        if data is None:
            return model_ff_fb
        if eps_data is None:
            return (model_ff_fb - data)
        return (model_ff_fb - data) / eps_data

    def recover_line_int_ff_fb_Te(self, res_dict):
        """
            RECOVER LINE-AVERAGED ELECTRON TEMPERATURE FROM FF-FB CONTINUUM SPECTRA
            Balmer edge ratio estimate: 360 nm and 400 nm
            Balmer ff-fb below edge ratio: 300 nm and 400 nm
            Balmer ff-fb above edge ratio: 400 nm and 500 nm
        """
        cont_ratio_360_400 = continuo_read.get_fffb_intensity_ratio_fn_T(360.0, 400.0, 1.0, save_output=False, restore=True)
        cont_ratio_300_360 = continuo_read.get_fffb_intensity_ratio_fn_T(300.0, 360.0, 1.0, save_output=False, restore=True)
        cont_ratio_400_500 = continuo_read.get_fffb_intensity_ratio_fn_T(400.0, 500.0, 1.0, save_output=False, restore=True)

        for diag_key in res_dict.keys():
            for chord_key in res_dict[diag_key].keys():

                print('Fitting ff+fb continuum spectra, LOS id= :', diag_key, ' ', chord_key)

                wave_fffb = np.asarray(res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['wave'])
                synth_data_fffb = np.asarray(
                    res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['intensity'])
                idx_300, val = self.find_nearest(wave_fffb, 300.0)
                idx_360, val = self.find_nearest(wave_fffb, 360.0)
                idx_400, val = self.find_nearest(wave_fffb, 400.0)
                idx_500, val = self.find_nearest(wave_fffb, 500.0)
                ratio_360_400 = synth_data_fffb[idx_360] / synth_data_fffb[idx_400]
                ratio_300_360 = synth_data_fffb[idx_300] / synth_data_fffb[idx_360]
                ratio_400_500 = synth_data_fffb[idx_400] / synth_data_fffb[idx_500]
                icont_ratio, vcont_ratio = self.find_nearest(cont_ratio_360_400[:, 1], ratio_360_400)
                fit_te_360_400 = cont_ratio_360_400[icont_ratio, 0]
                icont_ratio, vcont_ratio = self.find_nearest(cont_ratio_300_360[:, 1], ratio_300_360)
                fit_te_300_360 = cont_ratio_300_360[icont_ratio, 0]
                icont_ratio, vcont_ratio = self.find_nearest(cont_ratio_400_500[:, 1], ratio_400_500)
                fit_te_400_500 = cont_ratio_400_500[icont_ratio, 0]

                ##### Add fit Te result to dictionary
                res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum'] = {
                    'fit': {'fit_te_360_400': fit_te_360_400, 'fit_te_300_360': fit_te_300_360,
                            'fit_te_400_500': fit_te_400_500, 'units': 'eV'}}

                ###############################################################
                # CALCULATE EFFECTIVE DEL_L USING LINE-INT ne AND Te VALUES AND CONTINUUM
                ###############################################################
                if res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']:
                    params = Parameters()
                    params.add('delL', value=0.5, min=0.0001, max=10.0)
                    params.add('te_360_400', value=fit_te_360_400)
                    fit_ne = res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']
                    params.add('ne', value=fit_ne)
                    params['te_360_400'].vary = False
                    params['ne'].vary = False

                    fit_result = minimize(self.residual_continuo, params, args=(wave_fffb, synth_data_fffb),
                                          method='leastsq')
                    data_fit_ff_fb = self.residual_continuo(params, wave_fffb, None, None)
                    print("Chi squred: ", fit_result.chisqr)
                    report_fit(params)

                    ##### Add fit delL result to dictionary
                    res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['delL_360_400'] = \
                        params['delL'].value

    def recover_line_int_Stark_ne(self, res_dict):
        """
            RECOVER LINE-AVERAGED ELECTRON DENSITY FROM H6-2 STARK BROADENED SPECTRA
        """
        mmm_coeff = {'6t2': {'C': 3.954E-16, 'a': 0.7149, 'b': 0.028}}

        for diag_key in res_dict.keys():
            for chord_key in res_dict[diag_key].keys():

                print('Fitting Stark broadened H6-2 spectra, LOS id= :', diag_key, ' ', chord_key)

                for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():

                    if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == '6' and \
                                    res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == '2':

                        wave_stark = np.asarray(res_dict[diag_key][chord_key]['los_int']['stark']['wave'])
                        synth_data_stark = np.asarray(res_dict[diag_key][chord_key]['los_int']['stark']['intensity'])

                        params = Parameters()
                        params.add('cwl', value=float(H_line_key) / 10.0)
                        params.add('area', value=float(
                            res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit'] +
                            res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom']))
                        params.add('stark_fwhm', value=0.5, min=0.001, max=10.0)

                        params['cwl'].vary = False
                        params['area'].vary = False

                        fit_result = minimize(self.residual_lorentz_52, params,
                                              args=(wave_stark, synth_data_stark), method='leastsq')
                        data_fit_ne = self.residual_lorentz_52(params, wave_stark, None, None)
                        print("Chi squred: ", fit_result.chisqr)
                        report_fit(params)

                        # Assume Te = 1 eV
                        fit_ne = np.power((params['stark_fwhm'].value / mmm_coeff['6t2']['C']),
                                          1. / mmm_coeff['6t2']['a'])

                        ##### Add fit ne result to dictionary
                        res_dict[diag_key][chord_key]['los_int']['stark'] = {'fit': {'ne': fit_ne, 'units': 'm^-3'}}

    def recover_line_int_particle_bal(self, res_dict):
        """
            ESTIMATE RECOMBINATION/IONISATION RATES USING ADF11 ACD, SCD COEFF
        """
        for diag_key in res_dict.keys():
            for chord_key in res_dict[diag_key].keys():

                if (res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne'] and
                    res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']):

                    fit_ne = res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']
                    fit_Te = res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']

                    print('Ionization/recombination, LOS id= :', diag_key, ' ', chord_key)

                    # area_cm2 = 2*pi*R*dW
                    w2unmod = res_dict[diag_key][chord_key]['chord']['w2']
                    # area_cm2 = 1.0e04 * 2.*np.pi*res_dict[diag_key][chord_key]['chord']['d2unmod']*res_dict[diag_key][chord_key]['coord']['v2'][0]
                    area_cm2 = 1.0e04 * 2. * np.pi * w2unmod * \
                               res_dict[diag_key][chord_key]['chord']['p2'][0]
                    idxTe, Te_val = self.find_nearest(self.ADAS_dict['adf11']['1'].Te_arr, fit_Te)
                    idxne, ne_val = self.find_nearest(self.ADAS_dict['adf11']['1'].ne_arr, fit_ne * 1.0E-06)
                    for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():
                        if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == '7' and \
                                        res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == '2':
                            h72 = res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit'] + \
                                  res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom']
                            srec = 1.0E-04 * area_cm2 * h72 * 4. * np.pi * \
                                   self.ADAS_dict['adf11']['1'].acd[idxTe, idxne] / \
                                   self.ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].pec[idxTe, idxne]
                        if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == '4' and \
                                        res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == '2':
                            h42_excit = res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit']
                            sion = 1.0E-04 * area_cm2 * h42_excit * 4. * np.pi * \
                                   self.ADAS_dict['adf11']['1'].scd[idxTe, idxne] / \
                                   self.ADAS_dict['adf15']['1']['1'][H_line_key + 'excit'].pec[idxTe, idxne]

                    ##### Add adf11 srec sion estimates to dictionary
                    res_dict[diag_key][chord_key]['los_int']['adf11_fit'] = {'Srec': srec, 'Sion': sion,
                                                                  'units': 's^-1'}

    def recover_delL_atomden_product(self, res_dict):
        """
            ESTIMATE DEL_L * ATOMIC DENSITY PRODUCT FROM LY-ALPHA ASSUMING EXCITATION DOMINATED
        """
        for diag_key in res_dict.keys():
            for chord_key in res_dict[diag_key].keys():

                if (res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne'] and
                        res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']):

                    fit_ne = res_dict[diag_key][chord_key]['los_int']['stark']['fit']['ne']
                    fit_Te = res_dict[diag_key][chord_key]['los_int']['ff_fb_continuum']['fit']['fit_te_360_400']

                    print('delL * n0 from Ly-alpha, LOS id= :', diag_key, ' ', chord_key)

                    for H_line_key in res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'].keys():
                        if res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][0] == '2' and \
                                        res_dict[diag_key][chord_key]['spec_line_dict']['1']['1'][H_line_key][1] == '1':
                            h21 = res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['excit'] + \
                                  res_dict[diag_key][chord_key]['los_int']['H_emiss'][H_line_key]['recom']
                            idxTe, Te_val = self.find_nearest(
                                self.ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].Te_arr, fit_Te)
                            idxne, ne_val = self.find_nearest(
                                self.ADAS_dict['adf15']['1']['1'][H_line_key + 'recom'].ne_arr, fit_ne * 1.0E-06)
                            n0delL_Lya_tmp = 4. * np.pi * 1.0e-04 * h21 / (
                                self.ADAS_dict['adf15']['1']['1'][H_line_key + 'excit'].pec[idxTe, idxne] * ne_val)
                            n0delL_Lya_tmp = n0delL_Lya_tmp * 1.0e06 * 1.0e-02  # convert to m^-2
                            ##### Add fit n0*delL result to dictionary
                            res_dict[diag_key][chord_key]['los_int']['Ly_alpha_fit'] = {'n0delL': n0delL_Lya_tmp,
                                                                             'units': 'm^-2'}


if __name__=='__main__':

    spec_line_dict = {
        # Must include at least one spectral line from each impurity species in EDGE2D

        '1': # HYDROGEN
            {'1': {'1215.2': ['2', '1'],
                   '1025.3': ['3', '1'],
                   '6561.9': ['3', '2'],
                   '4860.6': ['4', '2'],
                   '4339.9': ['5', '2'],
                   '4101.2': ['6', '2'],
                   '3969.5': ['7', '2'],},
             },

        '4':  # BERYLLIUM
            {'2':{'5272.32':['4s', '3p']}},

        '7': # NITROGEN
            {'2':{'4042.07':['4f', '3d'],
                  '5002.18':['3d', '3p'],
                  '5005.86':['3d', '3p']},
             '3':{'4100.51':['3p', '3s']},
             '4':{'3481.83':['3p', '3s']}
             }
    }

    input_dict = {
        'machine':'JET',
        # 'tranfile': '/u/cstavrou/cmg/catalog/edge2d/jet/81472/may1117/seq#1/tran',
        'tranfile':'/u/bloman/cmg/catalog/edge2d/jet/81472/sep1517/seq#1/tran',
        # 'tranfile': '/u/cstavrou/cmg/catalog/edge2d/jet/81472/jun0617/seq#2/tran',
        # 'tranfile': '/u/cstavrou/cmg/catalog/edge2d/jet/81472/may2717/seq#2/tran',
        'diag_list': ['KT3A'],
        # 'diag_list':['KT3A', 'KT1V'],
        'spec_line_dict':spec_line_dict,
        'interactive_plots':False,
        'save_dir':'/work/bloman/pyproc/',
        'run_options':{
            'calc_synth_spec_features':True,
            'analyse_synth_spec_features':True}
    }

    PyprocAnalyse(input_dict)