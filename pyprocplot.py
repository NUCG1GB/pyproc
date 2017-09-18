
# Import external modules

import numpy as np
import json, pprint, pickle
import operator
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import patches
from collections import OrderedDict
from pyproc import pyprocprocess
from pyproc.pyprocanalyse import PyprocAnalyse
from pyproc.pyprocprocess import PyprocProcess

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

class PyprocPlot():
    """
        Class for retrieving, reducing and plotting pyproc saved data
    """
    def __init__(self, savedir, plot_dict=None):
        self.savedir = savedir
        self.plot_dict = plot_dict

        # Read pickled pyproc object
        try:
            with open(self.savedir + '/pyproc.2ddata.pkl', 'rb') as f:
                self.__data2d = pickle.load(f)
        except IOError as e:
            raise

        # Read processed synth diag saved data
        try:
            with open(self.savedir + '/pyproc.proc_synth_diag.json', 'r') as f:
                self.__res_dict = json.load(f)
        except IOError as e:
            raise

        if plot_dict:
            for key in plot_dict.keys():
                if key == 'prof_param_defs':
                    self.plot_profiles()
                if key == 'prof_Hemiss_defs':
                    self.plot_Hemiss_prof()
                if key == '2d_defs':
                    diagLOS = self.plot_dict['2d_defs']['diagLOS']
                    Rrng = self.plot_dict['2d_defs']['Rrng']
                    Zrng = self.plot_dict['2d_defs']['Zrng']
                    for at_num in self.plot_dict['2d_defs']['lines']:
                        for stage in self.plot_dict['2d_defs']['lines'][at_num]:
                            for line in self.plot_dict['2d_defs']['lines'][at_num][stage]:
                                self.plot_2d_spec_line(at_num, stage, line, diagLOS, Rrng=Rrng, Zrng=Zrng)


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
                PyprocPlot.pprint_json(value, indent + 1)
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

    def plot_profiles(self):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        axs = self.plot_dict['prof_param_defs']['axs']
        diag = self.plot_dict['prof_param_defs']['diag']
        color = self.plot_dict['prof_param_defs']['color']
        zorder = self.plot_dict['prof_param_defs']['zorder']

        ne = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'stark', 'fit', 'ne'])
        Te_hi = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_360_400'])
        Te_lo = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_300_360'])

        Sion_adf11 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'adf11_fit', 'Sion'])
        Srec_adf11 = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'adf11_fit', 'Srec'])
        n0delL = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'Ly_alpha_fit', 'n0delL'])
        p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])

        # Ne
        axs[0].plot(p2[:,0], ne, c=color, lw=2, zorder=zorder)

        # Te
        axs[1].plot(p2[:,0], Te_hi, c=color, lw=2, zorder=zorder)
        axs[1].plot(p2[:,0], Te_lo, c=color, lw=2, zorder=zorder)
        axs[1].fill_between(p2[:,0], Te_hi, Te_lo, facecolor=color,
                            edgecolor=color, alpha=0.25, linewidth=0, zorder=zorder)

        # Total recombination/ionisation (derived from emission with adf11)
        axs[2].semilogy(p2[:,0], Srec_adf11, '--', c=color, lw=2, zorder=zorder)
        axs[2].semilogy(p2[:,0], Sion_adf11, c=color, lw=2, zorder=zorder)
        axs[2].plot(0, 0, c='k', linewidth=2, label='Ionization')
        axs[2].plot(0, 0, '--', c='k', linewidth=2, label='Recombination')

        # N0
        axs[3].semilogy(p2[:,0], n0delL, c=color, lw=2, zorder=zorder)

        # plot ne, Te profiles at max ne along LOS
        if self.plot_dict['prof_param_defs']['include_pars_at_max_ne_along_LOS']:
            ne_max = self.get_param_at_max_ne_along_los(diag, 'ne')
            Te_max = self.get_param_at_max_ne_along_los(diag, 'te')
            axs[0].plot(p2[:, 0], ne_max, '-', c='darkgray', lw=2, zorder=1)
            axs[1].plot(p2[:, 0], Te_max, '-', c='darkgray', lw=2, zorder=1)

        if self.plot_dict['prof_param_defs']['include_sum_Sion_Srec']:
            Sion = self.get_line_int_sorted_data_by_chord_id(diag, ['los_1d', 'Sion', 'val'])
            Srec = self.get_line_int_sorted_data_by_chord_id(diag, ['los_1d', 'Srec ', 'val'])
            axs[2].plot(p2[:, 0], Sion, '-', c='darkgray', lw=2, zorder=1)
            axs[2].plot(p2[:, 0], Srec, '--', c='darkgray', lw=2, zorder=1)

        # legend
        # axes_dict['main'][0].plot([0, 0], [0, 0], c=sim_c, lw=2, label='simulation')

        # xpt, osp locations
        axs[0].plot([self.__data2d.geom['rpx'], self.__data2d.geom['rpx']], [0, 1e21], ':', c='darkgrey', linewidth=1.)
        axs[1].plot([self.__data2d.geom['rpx'], self.__data2d.geom['rpx']], [0, 20], ':', c='darkgrey', linewidth=1.)
        axs[2].plot([self.__data2d.geom['rpx'], self.__data2d.geom['rpx']], [1e20, 1e24], ':', c='darkgrey', linewidth=1.)
        axs[3].plot([self.__data2d.geom['rpx'], self.__data2d.geom['rpx']], [1e17, 1e21], ':', c='darkgrey', linewidth=1.)
        axs[0].plot([self.__data2d.osp[0], self.__data2d.osp[0]], [0, 1e21], ':', c='darkgrey', linewidth=1.)
        axs[1].plot([self.__data2d.osp[0], self.__data2d.osp[0]], [0, 20], ':', c='darkgrey', linewidth=1.)
        axs[2].plot([self.__data2d.osp[0], self.__data2d.osp[0]], [1e20, 1e24], ':', c='darkgrey', linewidth=1.)
        axs[3].plot([self.__data2d.osp[0], self.__data2d.osp[0]], [1e17, 1e21], ':', c='darkgrey', linewidth=1.)

        axs[3].set_xlabel('Major radius on tile 5 (m)')
        axs[0].set_ylabel(r'$\mathrm{n_{e}\/(m^{-3})}$')
        axs[1].set_ylabel(r'$\mathrm{T_{e}\/(eV)}}$')
        axs[2].set_ylabel(r'$\mathrm{(s^{-1})}$')
        axs[2].set_ylabel(r'$\mathrm{(s^{-1})}$')

        # axes_dict['main'][3].set_ylabel(r'$\mathrm{n_{H}\/(m^{-3})}$')

    def plot_Hemiss_prof(self):

        # PLOT RADIAL PROFILES OF SYNTHETIC LINE-INTEGRATED RECOVERED PARAMS
        lines = self.plot_dict['prof_Hemiss_defs']['lines']
        axs = self.plot_dict['prof_Hemiss_defs']['axs']
        diag = self.plot_dict['prof_Hemiss_defs']['diag']
        color = self.plot_dict['prof_Hemiss_defs']['color']
        zorder = self.plot_dict['prof_Hemiss_defs']['zorder']
        p2 = self.get_line_int_sorted_data_by_chord_id(diag, ['chord', 'p2'])

        for i, line in enumerate(lines.keys()):
            excit = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'excit'])
            recom = self.get_line_int_sorted_data_by_chord_id(diag, ['los_int', 'H_emiss', line, 'recom'])
            axs[i].plot(p2[:,0], excit+recom, '-', lw=2, c=color[i], zorder=zorder, label=line)
            axs[i].plot(p2[:,0], excit, '--', lw=1, c=color[i], zorder=zorder, label=line+' excit')
            axs[i].plot(p2[:,0], recom, ':', lw=1, c=color[i], zorder=zorder, label=line+' recom')

            axs[i].legend(loc='upper right')

    def get_line_int_sorted_data_by_chord_id(self, diag, mapList):
        """
            input:
                mapList: list of dict keys below the 'chord' level (e.g., ['los_int', 'stark', 'fit', 'ne']
                diag: synth diag name string
        """
        tmp = []
        chordidx = []
        for chord in self.__res_dict[diag]:
            parval = PyprocPlot.get_from_dict(self.__res_dict[diag][chord], mapList)
            # if isinstance(parval, float):
            tmp.append(parval)
            chordidx.append(int(chord)-1)

        chords = np.asarray(chordidx)
        sort_idx = np.argsort(chords, axis=0)
        sorted_parvals = np.asarray(tmp)[sort_idx]

        return sorted_parvals

    def plot_2d_spec_line(self, at_num, ion_stage, line_key, diagLOS, Rrng=None, Zrng=None,
                       savefig=False):

        fig, ax = plt.subplots(ncols=1, figsize=(10, 8))
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
        colors = plt.cm.hot(spec_line / np.max(spec_line))

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        # collplt.set_array(np.array(colors[:,0]))
        ax.set_yscale
        line_wv = float(line_key) / 10.
        pyprocprocess.at_sym[int(at_num) - 1]
        title = pyprocprocess.at_sym[int(at_num) - 1] + ' ' + pyprocprocess.roman[int(ion_stage) - 1] + ' ' + '{:5.1f}'.format(
            line_wv) + ' nm'
        ax.set_title(title)
        plt.gca().set_aspect('equal', adjustable='box')

        # ADD COLORBAR
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='7%', pad=0.1)

        # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
        # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin=0, vmax=np.max(spec_line)))
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

        ne = self.get_line_int_sorted_data_by_chord_id(self, diag, ['los_1d', 'ne'])

        for i, chord in range(len(ne)):
        ne_los_1d_max_idx, val = find_nearest(self.__res_dict[diag][chord_key]['los_1d']['ne'],
                                              np.max(self.__res_dict[diag][chord_key]['los_1d']['ne']))
        # Find parameter value at position corresponding to max ne along LOS (include nearest neighbours and average)
        if (ne_los_1d_max_idx + 1) == len(res_dict[diag_key][chord_key]['los_1d']['n0']):
            ne_los_1d_max.append(
                np.average(np.array((res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx - 1],
                                     res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx]))))

        elif (ne_los_1d_max_idx + 2) == len(res_dict[diag_key][chord_key]['los_1d']['n0']):
            ne_los_1d_max.append(
                np.average(np.array((res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 1],
                                     res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx]))))

        elif (ne_los_1d_max_idx + 3) == len(res_dict[diag_key][chord_key]['los_1d']['n0']):
            ne_los_1d_max.append(
                np.average(np.array((res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 2],
                                     res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 1],
                                     res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx]))))

        else:
            ne_los_1d_max.append(
                np.average(np.array((res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 3],
                                     res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 2],
                                     res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 1],
                                     res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx]))))


        return sorted_parvals


if __name__=='__main__':

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

    workdir = '/work/bloman/pyproc/'
    save_dir = workdir + 'cstavrou_cmg_catalog_edge2d_jet_81472_may2717_seq#2'
    Hlines_dict = OrderedDict([
        ('1215.2', ['2', '1']),
        ('6561.9', ['3', '2']),
        ('4339.9', ['5', '2']),
    ])

    nitrogen_lines_dict = OrderedDict([
        ('2', {'5002.18':['3d', '3p']}),
        ('3', {'4100.51':['3p', '3s']}),
        ('4', {'3481.83':['3p', '3s']}),
    ])

    spec_line_dict = {
        '1': # HYDROGEN
            {'1': Hlines_dict},
        '7': nitrogen_lines_dict
    }

    plot_dict = {
        'prof_param_defs':{'diag': 'KT3A', 'axs': ax1,
                           'include_pars_at_max_ne_along_LOS': False
                           'include_sum_Sion_Srec': False,
                           'include_target_vals': False,
                           'include_target_vals': False,
                           'color': 'blue', 'zorder': 10},
        'prof_Hemiss_defs':{'diag': 'KT3A',
                            'lines': Hlines_dict,
                            'axs': ax2,
                            'color': ['b', 'r', 'g', 'y', 'm', 'pink', 'orange'],
                            'zorder': 10},
        'prof_impemiss_defs':{'diag': 'KT3A',
                              'lines': spec_line_dict,
                              'axs': ax2,
                              'color': [],
                              'zorder': 10},
        # 'los_param_defs':{'diag':'KT3A', 'axs':ax1, 'color':'blue', 'zorder':10},
        # 'los_Hemiss_defs':{'diag':'KT3A', 'axs':ax1, 'color':'blue', 'zorder':10},
        # 'los_impemiss_defs':{'diag':'KT3A', 'axs':ax1, 'color':'blue', 'zorder':10},
        '2d_defs': {'lines': spec_line_dict, 'diagLOS': ['KT3A'], 'Rrng': [2.36, 2.96], 'Zrng': [-1.73, -1.29]}
    }

    o = PyprocPlot(save_dir, plot_dict=plot_dict)

    # Print out results dictionary tree
    PyprocPlot.pprint_json(o.res_dict['KT3A']['1']['los_int'])

    plt.show()