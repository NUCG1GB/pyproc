
import numpy as np
import json, pprint, pickle
import operator
from functools import reduce
import matplotlib
matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import patches
from collections import OrderedDict
from pyproc import process
from pyproc.analyse import AnalyseSynthDiag
from pyproc.process import ProcessEdgeSim

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

class Plot():
    """
        Class for retrieving, reducing and plotting pyproc saved data
    """
    def __init__(self, work_dir, case, plot_dict=None, icase=1):
        self.work_dir = work_dir
        self.case = case
        self.plot_dict = plot_dict
        self.icase = icase

        # Read pickled pyproc object
        try:
            with open(self.work_dir + self.case + '/pyproc.2ddata.pkl', 'rb') as f:
                self.__data2d = pickle.load(f)
        except IOError as e:
            raise

        # Read processed synth diag saved data
        try:
            with open(self.work_dir + self.case +  '/pyproc.proc_synth_diag.json', 'r') as f:
                self.__res_dict = json.load(f)
        except IOError as e:
            raise

        if plot_dict:
            # First restore (or re-read) the ADAS_dict
            self.ADAS_dict = AnalyseSynthDiag.get_ADAS_dict(self.work_dir, plot_dict['spec_line_dict'], restore=True)
            for key, val in plot_dict.items():
                if key == 'spec_line_dict_lytrap':
                    self.ADAS_dict_lytrap = AnalyseSynthDiag.get_ADAS_dict(self.work_dir + self.case + '/',
                                                                           plot_dict['spec_line_dict_lytrap'],
                                                                           restore=True, lytrap=True)
                if key == 'prof_param_defs':
                    self.plot_param_profiles()
                if key == 'prof_Hemiss_defs':
                    self.plot_Hemiss_profiles()
                if key == 'prof_impemiss_defs':
                    self.plot_impemiss_profiles()
                if key == '2d_defs':
                    diagLOS = val['diagLOS']
                    savefig = val['save']
                    Rrng = val['Rrng']
                    Zrng = val['Zrng']
                    self.plot_2d_ff_fb(diagLOS, Rrng=Rrng, Zrng=Zrng)
                    for at_num in val['lines']:
                        for stage in val['lines'][at_num]:
                            for line in val['lines'][at_num][stage]:
                                self.plot_2d_spec_line(at_num, stage, line, diagLOS, Rrng=Rrng, Zrng=Zrng, savefig=savefig)
                if key == 'imp_rad_coeff':
                    self.plot_imp_rad_coeff(val['region'], val['atnum'], val['ion_stages'])
                if key == 'imp_rad_dist':
                    self.plot_imp_rad_dist(val['region'], val['atnum'], val['te_nbins'])
                if key == 'nii_adas_afg':
                    self.plot_nii_adas_afg()

    @property
    def data2d(self):
        return self.__data2d

    @property
    def res_dict(self):
        return self.__res_dict

    @staticmethod
    def pprint_json(resdict, indent=0):
        for key, value in resdict.items():
            print('\t' * indent + str(key))
            if isinstance(value, dict):
                Plot.pprint_json(value, indent + 1)
            else:
                if isinstance(value, list):
                    print('\t' * (indent+1) + '[list]')
                else:
                    if isinstance(value, str):
                        print('\t' * (indent + 1) + value)
                    else:
                        print('\t' * (indent + 1) + '[float]')

    # get item from nested dict
    @staticmethod
    def get_from_dict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)

    def plot_param_profiles(self):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        axs = self.plot_dict['prof_param_defs']['axs']
        diag = self.plot_dict['prof_param_defs']['diag']
        color = self.plot_dict['prof_param_defs']['color']
        zorder = self.plot_dict['prof_param_defs']['zorder']
        coord = self.plot_dict['prof_param_defs']['coord']

        if coord == 'R':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:,0]
        elif coord == 'Z':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:,1]
        elif coord == 'angle':
            x = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'los_angle'])
        else:
            # default R coord
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:,0]

        ne = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'stark', 'fit', 'ne'])
        # Te_hi = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_360_400'])
        Te_hi = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_400_500'])
        Te_lo = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_300_360'])

        Sion_adf11 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'adf11_fit', 'Sion'])
        Srec_adf11 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'adf11_fit', 'Srec'])
        n0delL = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Ly_alpha_fit', 'n0delL'])

        # Ne
        axs[0].plot(x, ne, c=color, lw=2, zorder=zorder)

        # Te
        axs[1].plot(x, Te_hi, c=color, lw=2, zorder=zorder)
        axs[1].plot(x, Te_lo, c=color, lw=2, zorder=zorder)
        axs[1].fill_between(x, Te_hi, Te_lo, facecolor=color,
                            edgecolor=color, alpha=0.25, linewidth=0, zorder=zorder)

        # Total recombination/ionisation (derived from emission with adf11)
        axs[2].semilogy(x, Srec_adf11, '--', c=color, lw=2, zorder=zorder)
        axs[2].semilogy(x, Sion_adf11, c=color, lw=2, zorder=zorder)
        axs[2].plot(0, 0, c='k', linewidth=2, label='Ionization')
        axs[2].plot(0, 0, '--', c='k', linewidth=2, label='Recombination')

        # N0
        axs[3].semilogy(x, n0delL, c=color, lw=2, zorder=zorder)

        # plot ne, Te profiles at max ne along LOS
        if self.plot_dict['prof_param_defs']['include_pars_at_max_ne_along_LOS']:
            ne_max, te_max = self.get_param_at_max_ne_along_los(diag, 'te')
            ne_max, n0_max = self.get_param_at_max_ne_along_los(diag, 'n0')
            axs[0].plot(x, ne_max, '-', c='darkgray', lw=2, zorder=1)
            axs[1].plot(x, te_max, '-', c='darkgray', lw=2, zorder=1)
            axs[3].plot(x, n0_max, '-', c='darkgray', lw=2, zorder=1)
        
        if self.plot_dict['prof_param_defs']['include_target_vals']:
            axs[0].plot(self.data2d.denel_OT['xdata'][:self.data2d.denel_OT['npts']]+self.data2d.osp[0],
                        self.data2d.denel_OT['ydata'][:self.data2d.denel_OT['npts']], 'o', mfc='None',
                        mec=color, mew=2.0, ms=8)
            axs[1].plot(self.data2d.teve_OT['xdata'][:self.data2d.teve_OT['npts']]+self.data2d.osp[0],
                        self.data2d.teve_OT['ydata'][:self.data2d.teve_OT['npts']], 'o', mfc='None',
                        mec=color, mew=2.0, ms=8)
            # axs[1].plot(self.data2d.teve_IT['xdata'][:self.data2d.teve_IT['npts']]+self.data2d.isp[0],
            #             self.data2d.teve_IT['ydata'][:self.data2d.teve_IT['npts']], 'o', mfc='None',
            #             mec='darkgray', mew=2.0, ms=8)
            # Ion flux to outer target
            axs[2].plot(self.data2d.pflxd_OT['xdata'][:self.data2d.pflxd_OT['npts']]+self.data2d.osp[0],
                        -1.0*self.data2d.pflxd_OT['ydata'][:self.data2d.pflxd_OT['npts']], 'o', mfc='None',
                        mec=color, mew=2.0, ms=8)
            # neutral density
            axs[3].plot(self.data2d.da_OT['xdata'][:self.data2d.da_OT['npts']] + self.data2d.osp[0],
                        self.data2d.da_OT['ydata'][:self.data2d.da_OT['npts']], 'o', mfc='None',
                        mec=color, mew=2.0, ms=8)

        if self.plot_dict['prof_param_defs']['include_sum_Sion_Srec']:
            Sion = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Sion', 'val'])
            Srec = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Srec', 'val'])
            axs[2].plot(x, Sion, '-', c='darkgray', lw=2, zorder=1)
            axs[2].plot(x, -1.0*Srec, '--', c='darkgray', lw=2, zorder=1)

            # Output Sion/Srec direct and spectroscopically inferred summations
            R = p2[:, 0]
            inner_idx, = np.where(R < self.__data2d.geom['rpx'])
            outer_idx, = np.where(R >= self.__data2d.geom['rpx'])
            print('')
            print(diag, ' particle balance')
            print('Direct sum of Sion, Srec along LOS [s^-1]')
            print('Total (R < R_xpt) :', np.sum(Sion[inner_idx]), np.sum(Srec[inner_idx]))
            print('Total (R >= R_xpt) :', np.sum(Sion[outer_idx]), np.sum(Srec[outer_idx]))
            print('adf11 Sion, Srec estimates[s^-1]')
            print('Total (R < R_xpt) :', np.sum(Sion_adf11[inner_idx]), np.sum(Srec_adf11[inner_idx]))
            print('Total (R >= R_xpt) :', np.sum(Sion_adf11[outer_idx]), np.sum(Srec_adf11[outer_idx]))
            print('')

        # legend
        # axes_dict['main'][0].plot([0, 0], [0, 0], c=sim_c, lw=2, label='simulation')

        # xpt, isp, osp locations
        for i in range(len(axs)):
            axs[i].axvline(self.__data2d.geom['rpx'], ls=':', c='darkgrey', linewidth=1.)
            axs[i].axvline(self.__data2d.osp[0], ls=':', c='darkgrey', linewidth=1.)
            axs[i].axvline(self.__data2d.isp[0], ls=':', c='darkgrey', linewidth=1.)

        axs[3].set_xlabel('Major radius on tile 5 (m)')
        axs[0].set_ylabel(r'$\mathrm{n_{e}\/(m^{-3})}$')
        axs[3].set_ylabel(r'$\mathrm{n_{H}\/(m^{-3})}$')
        axs[1].set_ylabel(r'$\mathrm{T_{e}\/(eV)}}$')
        axs[2].set_ylabel(r'$\mathrm{(s^{-1})}$')
        axs[2].set_ylabel(r'$\mathrm{(s^{-1})}$')

        axs[3].set_xlim(x[0], x[-1])

        # axes_dict['main'][3].set_ylabel(r'$\mathrm{n_{H}\/(m^{-3})}$')

    def plot_Hemiss_profiles(self):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        lines = self.plot_dict['prof_Hemiss_defs']['lines']
        lines_lytrap = None
        if 'lines_lytrap' in self.plot_dict['prof_Hemiss_defs']:
            lines_lytrap = self.plot_dict['prof_Hemiss_defs']['lines_lytrap']
        axs = self.plot_dict['prof_Hemiss_defs']['axs']
        diag = self.plot_dict['prof_Hemiss_defs']['diag']
        color = self.plot_dict['prof_Hemiss_defs']['color']
        zorder = self.plot_dict['prof_Hemiss_defs']['zorder']
        excrec = self.plot_dict['prof_Hemiss_defs']['excrec']
        coord = self.plot_dict['prof_Hemiss_defs']['coord']

        if coord == 'R':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 0]
        elif coord == 'Z':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 1]
        elif coord == 'angle':
            x = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'los_angle'])
        else:
            # default R coord
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 0]

        for i, line in enumerate(lines.keys()):
            excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'excit'])
            recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'recom'])
            label = '{:5.1f}'.format(float(line)/10.) + ' nm'
            axs[i].plot(x, excit+recom, '-', lw=2, c=color, zorder=zorder, label=label)
            if excrec:
                axs[i].plot(x, excit, '--', lw=1, c=color, zorder=zorder, label=label+' excit')
                axs[i].plot(x, recom, ':', lw=1, c=color, zorder=zorder, label=label+' recom')

            leg = axs[i].legend(loc='upper left')
            leg.get_frame().set_alpha(0.2)
            if i == len(lines.keys())-1:
                axs[i].set_xlabel(coord)

        # also plot Ly-series with photon trapping
        if lines_lytrap:
            if '1215.67'in lines_lytrap:
                excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1215.67', 'excit'])
                recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '1215.67', 'recom'])
                label = '{:5.1f}'.format(float('1215.67') / 10.) + ' nm; ' + 'ad hoc opacity'
                axs[0].plot(x, excit + recom, '--', lw=2, c=color, zorder=zorder, label=label)
                if excrec:
                    axs[0].plot(x, excit, '--', lw=1, c=color, zorder=zorder, label=label + ' excit')
                    axs[0].plot(x, recom, ':', lw=1, c=color, zorder=zorder, label=label + ' recom')
                leg = axs[0].legend(loc='upper left')
                leg.get_frame().set_alpha(0.2)
            if '6564.57'in lines_lytrap:
                excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6564.57', 'excit'])
                recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', '6564.57', 'recom'])
                label = '{:5.1f}'.format(float('6564.57') / 10.) + ' nm; ' + 'ad hoc opacity'
                axs[1].plot(x, excit + recom, '--', lw=2, c=color, zorder=zorder, label=label)
                if excrec:
                    axs[1].plot(x, excit, '--', lw=1, c=color, zorder=zorder, label=label + ' excit')
                    axs[1].plot(x, recom, ':', lw=1, c=color, zorder=zorder, label=label + ' recom')
                leg = axs[1].legend(loc='upper left')
                leg.get_frame().set_alpha(0.2)

        # xpt, isp, osp locations
        for i in range(len(axs)):
            axs[i].axvline(self.__data2d.geom['rpx'], ls=':', c='darkgrey', linewidth=1.)
            axs[i].axvline(self.__data2d.osp[0], ls=':', c='darkgrey', linewidth=1.)
            axs[i].axvline(self.__data2d.isp[0], ls=':', c='darkgrey', linewidth=1.)

    def plot_nii_adas_afg(self):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        axs = self.plot_dict['nii_adas_afg']['axs']
        color = self.plot_dict['nii_adas_afg']['color']
        zorder = self.plot_dict['nii_adas_afg']['zorder']

        for diagname, diag in self.__res_dict.items():
            if diagname == 'KT3':
                p2 = self.get_line_int_sorted_data_by_chord_id(diagname, ['chord', 'p2'])
                R = p2[:, 0]

                wav = np.asarray(diag['1']['los_int']['afg_adasn1_kt3b1200']['wave'])
                nii_adas_afg_intensity = self.get_line_int_sorted_data_by_chord_id(
                    diagname, ['los_int', 'afg_adasn1_kt3b1200', 'intensity'])
                axs.semilogy(wav, nii_adas_afg_intensity[0], '-', lw=2.)

                # Also write the results to file for Stuart to process
                filename = self.work_dir + self.case + '/kt3_nii_adas_afg' '.wav'
                np.savetxt(filename, wav.T, newline='\n')
                filename = self.work_dir + self.case + '/kt3_nii_adas_afg' + '.coord'
                np.savetxt(filename, R, newline='\n')
                filename = self.work_dir + self.case + '/kt3_nii_adas_afg' + '.data'
                header = 'units: ph s^-1 m^-2 sr^-1 nm^-1'
                np.savetxt(filename, nii_adas_afg_intensity.T, header=header, delimiter=',', newline='\n')

            leg = axs.legend(loc='upper right')
            leg.get_frame().set_alpha(0.2)

    def plot_impemiss_profiles(self):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        lines = self.plot_dict['prof_impemiss_defs']['lines']
        axs = self.plot_dict['prof_impemiss_defs']['axs']
        diag = self.plot_dict['prof_impemiss_defs']['diag']
        color = self.plot_dict['prof_impemiss_defs']['color']
        zorder = self.plot_dict['prof_impemiss_defs']['zorder']
        excrec = self.plot_dict['prof_impemiss_defs']['excrec']
        coord = self.plot_dict['prof_impemiss_defs']['coord']

        if coord == 'R':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 0]
        elif coord == 'Z':
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 1]
        elif coord == 'angle':
            x = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'los_angle'])
        else:
            # default R coord
            p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])
            x = p2[:, 0]

        icol = 0
        for at_num in lines.keys():
            if int(at_num) > 1 : # skip hydrogen
                for i, ion_stage in enumerate(lines[at_num].keys()):
                    for line in lines[at_num][ion_stage]:
                        line_wv = float(line) / 10.

                        label = process.at_sym[int(at_num) - 1] + ' ' + \
                                process.roman[int(ion_stage) - 1] + ' ' + '{:5.1f}'.format(
                            line_wv) + ' nm'

                        excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'imp_emiss', at_num, ion_stage, line, 'excit'])
                        recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'imp_emiss', at_num, ion_stage, line, 'recom'])

                        axs[i].plot(x, excit+recom, '-', lw=2, c=color[icol], zorder=zorder, label=label)
                        if excrec:
                            axs[i].plot(x, excit, '--', lw=1, c=color[icol], zorder=zorder, label=label +' excit')
                            axs[i].plot(x, recom, ':', lw=1, c=color[icol], zorder=zorder, label=label +' recom')

                        leg = axs[i].legend(loc='upper right')
                        leg.get_frame().set_alpha(0.2)
                icol += 1

        # xpt, isp, osp locations
        for i in range(len(axs)):
            axs[i].axvline(self.__data2d.geom['rpx'], ls=':', c='darkgrey', linewidth=1.)
            axs[i].axvline(self.__data2d.osp[0], ls=':', c='darkgrey', linewidth=1.)
            axs[i].axvline(self.__data2d.isp[0], ls=':', c='darkgrey', linewidth=1.)

    def plot_imp_rad_coeff(self, region, atnum, ion_stages):

        axs = self.plot_dict['imp_rad_coeff']['axs']
        color = self.plot_dict['imp_rad_coeff']['color']
        zorder = self.plot_dict['imp_rad_coeff']['zorder']

        if self.data2d.imp1_atom_num or self.data2d.imp2_atom_num:
            if atnum == self.data2d.imp1_atom_num or atnum == self.data2d.imp2_atom_num:
                atnumstr = str(atnum)
                # rad loss coeff not very sensitive to elec. density so choose a sensible value
                ine, vne = find_nearest(self.ADAS_dict['adf11'][atnumstr].ne_arr, 1.0e14)

                # plot ionisation balance radiative loss coeff
                axs[0].loglog(self.ADAS_dict['adf11'][atnumstr].Te_arr,
                              1.0e-06 * self.ADAS_dict['adf11'][atnumstr].ion_bal_pwr['total'][ine, :], '-k',
                              lw=3.0)
                for i, stage in enumerate(ion_stages):
                    axs[0].loglog(self.ADAS_dict['adf11'][atnumstr].Te_arr,
                                  1.0e-06 * self.ADAS_dict['adf11'][atnumstr].ion_bal_pwr['ion'][ine, :, stage-1],
                                  ':', c='k', lw=1.0)
                    axs[i + 1].loglog(self.ADAS_dict['adf11'][atnumstr].Te_arr,
                                      1.0e-06 * self.ADAS_dict['adf11'][atnumstr].ion_bal_pwr['ion'][ine, :, stage-1],
                                      '-', c='k', lw=2.0)

                # plot sim rad loss coeff/pwr for each stage
                imp_radpwr_coeff_collate = []
                imp_radpwr_collate = []
                te_collate = []
                for cell in self.data2d.regions[region].cells:
                    if atnum == self.data2d.imp1_atom_num:
                        imp_radpwr_coeff_collate.append(cell.imp1_radpwr_coeff)
                        imp_radpwr_collate.append(cell.imp1_radpwr)
                    elif atnum == self.data2d.imp2_atom_num:
                        imp_radpwr_coeff_collate.append(cell.imp2_radpwr_coeff)
                        imp_radpwr_collate.append(cell.imp2_radpwr)

                    te_collate.append(cell.te)

                imp_radpwr_coeff_collate_arr = np.asarray(imp_radpwr_coeff_collate)
                imp_radpwr_collate_arr = np.sum(np.asarray(imp_radpwr_collate), axis=1)
                imp_radpwr_collate_arr_max = np.max(imp_radpwr_collate_arr)
                imp_radpwr_collate_arr/= imp_radpwr_collate_arr_max
                te_collate_arr = np.asarray(te_collate)

                axs[0].scatter(te_collate_arr, np.sum(imp_radpwr_coeff_collate_arr, axis=1),
                               s=500*imp_radpwr_collate_arr, c=color, edgecolors='none')
                for i, stage in enumerate(ion_stages):
                    scale = np.asarray(imp_radpwr_collate)[:, i]
                    scale/=imp_radpwr_collate_arr_max
                    axs[i + 1].scatter(te_collate_arr, imp_radpwr_coeff_collate_arr[:, i],
                                       s=500*scale, c=color,  edgecolors='none')
                    axs[i + 1].set_ylabel(r'$\mathrm{P_{rad}\/+}$' + str(stage-1) + r'$\mathrm{\/(W m^{3})}$')

                    if i == len(ion_stages):
                        axs[i + 1].set_xlabel('Te (eV)')

                axs[0].set_title(self.case + ' ' + process.at_sym[atnum - 1] + ' in region: ' + region)

    def plot_imp_rad_dist(self, region, atnum, te_nbins):

        axs = self.plot_dict['imp_rad_dist']['axs']
        color = self.plot_dict['imp_rad_dist']['color']
        zorder = self.plot_dict['imp_rad_dist']['zorder']

        if self.data2d.imp1_atom_num or self.data2d.imp2_atom_num:
            if atnum == self.data2d.imp1_atom_num or atnum == self.data2d.imp2_atom_num:
                atnumstr = str(atnum)
                # rad loss coeff not very sensitive to elec. density so choose a sensible value
                ine, vne = find_nearest(self.ADAS_dict['adf11'][atnumstr].ne_arr, 1.0e14)

                # Get max and min Te in region for Te bin range
                min_Te = 100000.
                max_Te = 0.0
                for cell in self.data2d.regions[region].cells:
                    if cell.te > max_Te: max_Te = cell.te
                    if cell.te < min_Te: min_Te = cell.te

                # Set up elec. temp bins and labels
                te_bins = np.logspace(np.log10(min_Te), np.log10(max_Te), te_nbins)
                te_bin_labels = []

                for ite, vte in enumerate(te_bins):
                    if (ite + 1) != len(te_bins):
                        label = '{:6.1f}'.format(te_bins[ite]) + '-' + '{:6.1f}'.format(te_bins[ite + 1])
                        te_bin_labels.append(label)

                        # BIN RADIATED POWER BY TE
                        te_bin_imp_radpwr = np.zeros((te_nbins-1, atnum))
                        te_bin_H_radpwr = np.zeros((te_nbins-1))
                        for cell in self.data2d.regions[region].cells:
                            for ite, vte in enumerate(te_bins):
                                if (ite + 1) != len(te_bins):
                                    if cell.te > te_bins[ite] and cell.te <= te_bins[ite + 1]:
                                        te_bin_H_radpwr[ite] += cell.H_radpwr
                                        if atnum == self.data2d.imp1_atom_num:
                                            te_bin_imp_radpwr[ite] += cell.imp1_radpwr
                                        elif atnum == self.data2d.imp2_atom_num:
                                            te_bin_imp_radpwr[ite] += cell.imp2_radpwr
                        # convert to MW
                        te_bin_imp_radpwr *= 1.0e-06
                        te_bin_H_radpwr *= 1.0e-06

                # IMP CHARGE STATE DIST
                axs[0].plot(np.sum(te_bin_imp_radpwr, axis=0), '-o', c=color, mfc=color, mec=color, ms=4, mew=2.0)
                axs[0].set_ylabel(r'$\mathrm{P_{RAD}\/(MW)}$')
                axs[0].set_xlabel('Ionisation stage')

                # BAR PLOT BY TE BINS
                x_pos = np.arange(len(te_bin_labels))
                width = 0.2
                barspace = width * self.icase
                axs[1].bar(x_pos+barspace, te_bin_H_radpwr, width, align='center', color='darkgrey', edgecolor=color, alpha=0.3)
                axs[1].bar(x_pos+barspace, np.sum(te_bin_imp_radpwr, axis=1) , width, align='center', color=color, alpha=0.3)
                axs[1].set_xticks(x_pos+width*(self.icase-1))
                axs[1].set_xticklabels(te_bin_labels, rotation=90)
                axs[1].set_xlabel(r'$\mathrm{T_{e}\/(eV)}$')
                axs[1].set_ylabel(r'$\mathrm{P_{RAD}\/(MW)}$')

                axs[0].set_title(self.case + ' ' + process.at_sym[atnum - 1] + ' in region: ' + region)

    def get_line_int_sorted_data_by_chord_id(self, diag, mapList):
        """
            input:
                mapList: list of dict keys below the 'chord' level (e.g., ['los_int', 'stark', 'fit', 'ne']
                diag: synth diag name string
        """
        tmp = []
        chordidx = []
        for chord in self.__res_dict[diag]:
            parval = Plot.get_from_dict(self.__res_dict[diag][chord], mapList)
            # if isinstance(parval, float):
            tmp.append(parval)
            chordidx.append(int(chord)-1)

        chords = np.asarray(chordidx)
        sort_idx = np.argsort(chords, axis=0)
        sorted_parvals = np.asarray(tmp)[sort_idx]

        return sorted_parvals

    def plot_2d_spec_line(self, at_num, ion_stage, line_key, diagLOS, Rrng=None, Zrng=None,
                          min_clip=0.0, max_clip = 1.0, savefig=False):

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
        fig.patch.set_facecolor('white')

        if Rrng and Zrng:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches = []
        spec_line = []
        for cell in self.__data2d.cells:
            cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
            if int(at_num) > 1:
                spec_line.append(cell.imp_emiss[at_num][ion_stage][line_key]['excit'] +
                                cell.imp_emiss[at_num][ion_stage][line_key]['recom'])
            else:
                spec_line.append(cell.H_emiss[line_key]['excit'] +
                                cell.H_emiss[line_key]['recom'])

            # imp_line.append((cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_excit']+cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_recom'])*cell.ne)

        # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.hot, norm=matplotlib.colors.LogNorm(), zorder=1, lw=0)
        # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.hot, zorder=1, lw=0)
        coll1 = PatchCollection(cell_patches, zorder=1)
        # coll1.set_array(np.asarray(imp_line))

        # Clip color scale with min and max threshold
        cscale_min = min_clip * np.max(spec_line)
        cscale_max = max_clip * np.max(spec_line)

        spec_line_clipped = []
        for i, cell_emiss in enumerate(spec_line):
            if cell_emiss >= cscale_min and cell_emiss <= cscale_max:
                spec_line_clipped.append(cell_emiss)
            elif cell_emiss >= cscale_max:
                spec_line_clipped.append(cscale_max)
            else:
                spec_line_clipped.append(0.0)
        colors = plt.cm.hot(spec_line_clipped / np.max(spec_line_clipped))

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        ax.set_yscale
        line_wv = float(line_key) / 10.
        title = self.case + ' ' + process.at_sym[int(at_num) - 1] + ' ' + process.roman[int(ion_stage) - 1] + ' ' + '{:5.1f}'.format(
            line_wv) + ' nm'
        ax.set_title(title)
        plt.gca().set_aspect('equal', adjustable='box')

        # ADD COLORBAR
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='7%', pad=0.1)

        # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
        # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot,
                                   norm=plt.Normalize(vmin=cscale_min,
                                                      vmax=cscale_max))
        sm._A = []
        cbar = fig.colorbar(sm, cax=cbar_ax)
        label = '$\mathrm{ph\/s^{-1}\/m^{-3}\/sr^{-1}}$'
        cbar.set_label(label)

        # ADD DIAG LOS
        if diagLOS:
            for diag in diagLOS:
                self.__data2d.synth_diag[diag].plot_LOS(ax, color='w', lw=1.0)

        # PLOT SEPARATRIX AND WALL
        ax.add_patch(self.__data2d.wall_poly)
        ax.add_patch(self.__data2d.sep_poly)


        if savefig:
            plt.savefig(self.work_dir + self.case + '/' + title + '.png', dpi=plt.gcf().dpi)


    def plot_2d_ff_fb(self, diagLOS, Rrng=None, Zrng=None, savefig=False):

        fig, ax = plt.subplots(ncols=1, figsize=(10, 8))
        fig.patch.set_facecolor('white')
        if Rrng and Zrng:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches = []
        ff_fb_emiss = []
        for cell in self.__data2d.cells:
            cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
            ff_fb_emiss.append(cell.ff_fb_emiss['ff_fb'])

        coll1 = PatchCollection(cell_patches, zorder=1)
        colors = plt.cm.hot(ff_fb_emiss / np.max(ff_fb_emiss))

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        ax.set_yscale
        title = self.case + ' Bremss. (ff+fb) 400.96 nm'
        ax.set_title(title)
        plt.gca().set_aspect('equal', adjustable='box')

        # ADD COLORBAR
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='7%', pad=0.1)

        # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
        # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin=0, vmax=np.max(ff_fb_emiss)))
        sm._A = []

        cbar = fig.colorbar(sm, cax=cbar_ax)
        label = '$\mathrm{ph\/s^{-1}\/m^{-3}\/sr^{-1}}$'
        cbar.set_label(label)

        # ADD DIAG LOS
        if diagLOS:
            for diag in diagLOS:
                self.__data2d.synth_diag[diag].plot_LOS(ax, color='w', lw=1.0)

        # PLOT SEPARATRIX AND WALL
        ax.add_patch(self.__data2d.sep_poly)
        ax.add_patch(self.__data2d.wall_poly)

        if savefig:
            plt.savefig(title + '.png', dpi=plt.gcf().dpi)


    def get_param_at_max_ne_along_los(self, diag, paramstr, nAvgNeighbs=2):

        ne = self.get_line_int_sorted_data_by_chord_id(diag, ['los_1d', 'ne'])
        par = self.get_line_int_sorted_data_by_chord_id(diag, ['los_1d', paramstr])

        ne_max = []
        par_at_ne_max = []

        for i in range(len(ne)):
            if ne[i]:
                ne_los = np.asarray(ne[i])
                par_los = np.asarray(par[i])

                ne_max_idx, val = find_nearest(ne_los, np.max(ne_los))

                # ne_max.append(ne_los[ne_max_idx])
                # par_at_ne_max.append(par_los[ne_max_idx])

                # Find parameter value at position corresponding to max ne along LOS (include nearest neighbours and average)
                if (ne_max_idx + 1) == len(ne_los):
                    ne_max.append(np.average(np.array((ne_los[ne_max_idx - 1], ne_los[ne_max_idx]))))
                    par_at_ne_max.append(np.average(np.array((par_los[ne_max_idx - 1], par_los[ne_max_idx]))))
                elif (ne_max_idx + 2) == len(ne_los):
                    ne_max.append(np.average(np.array((ne_los[ne_max_idx - 1], ne_los[ne_max_idx],
                                                    ne_los[ne_max_idx + 1]))))
                    par_at_ne_max.append(np.average(np.array((par_los[ne_max_idx - 1], par_los[ne_max_idx],
                                                    par_los[ne_max_idx + 1]))))
                else:
                    ne_max.append(np.average(np.array((ne_los[ne_max_idx + 2], ne_los[ne_max_idx],
                                                    ne_los[ne_max_idx + 1]))))
                    par_at_ne_max.append(np.average(np.array((par_los[ne_max_idx + 2], par_los[ne_max_idx],
                                                    par_los[ne_max_idx + 1]))))
            else:
                ne_max.append(0)
                par_at_ne_max.append(0)

        return np.asarray(ne_max), np.asarray(par_at_ne_max)


if __name__=='__main__':

    """
    JET region names:
        hfs_sol
        lfs_sol
        hfs_div
        lfs_div
        xpt_conreg
        hfs_lower
        lfs_lower
        rhon_09_10
    """

    #Example

    left  = 0.2  # the left side of the subplots of the figure
    right = 0.95    # the right side of the subplots of the figure
    bottom = 0.15   # the bottom of the subplots of the figure
    top = 0.93      # the top of the subplots of the figure
    wspace = 0.18   # the amount of width reserved for blank space between subplots
    hspace = 0.1  # the amount of height reserved for white space between subplots

    fig1, ax1 = plt.subplots(nrows=4, ncols=1, figsize=(6,12), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    fig2, ax2 = plt.subplots(nrows=3, ncols=1, figsize=(6,12), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    fig3, ax3 = plt.subplots(nrows=3, ncols=1, figsize=(6,12), sharex=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    fig4, ax4 = plt.subplots(nrows=5, ncols=1, figsize=(6,12), sharex=True, sharey=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    fig5, ax5 = plt.subplots(nrows=1, ncols=1, figsize=(12,12), sharex=True, sharey=True)
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    workdir = '/work/bloman/pyproc/'
    # case = 'bloman_cmg_catalog_edge2d_jet_81472_oct2117_seq#3'
    case = 'jsimpson_cmg_catalog_edge2d_jet_85274_aug0717_seq#27'

    Hlines_dict = OrderedDict([
        ('1215.2', ['2', '1']),
        ('6561.9', ['3', '2']),
        ('4339.9', ['5', '2']),
    ])

    nitrogen_lines_dict = OrderedDict([
        ('2', {
            # '4042.07': ['4f', '3d'],
            '3996.13': ['4f', '3d']
            # '5002.18': ['3d', '3p'],
            # '5005.86': ['3d', '3p']
        }),
        ('3', {'4100.51':['3p', '3s']}),
        ('4', {'4058.90':['3d', '3p']})
    ])

    beryllium_lines_dict = OrderedDict([
        ('2', {'5272.32':['4s', '3p']})
    ])

    spec_line_dict = OrderedDict([
        ('1', {'1': Hlines_dict}),
        # ('4', beryllium_lines_dict),
        # ('7', nitrogen_lines_dict)
    ])

    plot_dict = {
        'spec_line_dict':spec_line_dict,
        'prof_param_defs':{'diag': 'KT3', 'axs': ax1,
                           'include_pars_at_max_ne_along_LOS': True,
                           'include_sum_Sion_Srec': True,
                           'include_target_vals': False,
                           'coord': 'R', # 'angle' 'R' 'Z'
                           'color': 'blue', 'zorder': 10},
        'prof_Hemiss_defs':{'diag': 'KT3',
                            'lines': spec_line_dict['1']['1'],
                            'excrec': True,
                            'axs': ax2,
                            'coord': 'R', # 'angle' 'R' 'Z'
                            'color': 'b',
                            'zorder': 10},
        # 'prof_impemiss_defs':{'diag': 'KT3',
        #                       'lines': spec_line_dict,
        #                       'excrec': False,
        #                       'coord': 'R', # 'angle' 'R' 'Z'
        #                       'axs': ax3,
        #                       'color': ['r', 'g'],
        #                       'zorder': 10},
        # 'imp_rad_coeff': {'region': 'vessel',
        #                   'atnum': 7,
        #                   'ion_stages': [1, 2, 3, 4],
        #                   'axs': ax4,
        #                   'color': 'r',
        #                   'zorder': 10},
        # 'nii_adas_afg': { 'axs': ax5,
        #                   'color': 'r',
        #                   'zorder': 10},
        # 'los_param_defs':{'diag':'KT3', 'axs':ax1, 'color':'blue', 'zorder':10},
        # 'los_Hemiss_defs':{'diag':'KT3', 'axs':ax1, 'color':'blue', 'zorder':10},
        # 'los_impemiss_defs':{'diag':'KT3', 'axs':ax1, 'color':'blue', 'zorder':10},
        '2d_defs': {'lines': spec_line_dict, 'diagLOS': ['KT3'], 'Rrng': [2.36, 2.96], 'Zrng': [-1.73, -1.29], 'save': False}
    }

    o = Plot(workdir, case, plot_dict=plot_dict)

    # Print out results dictionary tree
    Plot.pprint_json(o.res_dict['KT3']['1']['los_int'])
    # Plot.pprint_json(o.res_dict['KT3']['1'])

    plt.show()