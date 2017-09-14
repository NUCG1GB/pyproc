
# Import external modules

import numpy as np
import json, pprint, pickle
import operator
from functools import reduce
import matplotlib.pyplot as plt
from collections import namedtuple
from pyproc.pyprocanalyse import PyprocAnalyse
from pyproc.pyprocprocess import PyprocProcess

class PyprocPlot():
    """
        Class for retrieving, reducing and plotting pyproc saved data
    """
    def __init__(self, savedir, stdJETplots = False,
                 plot_LOS_defs  = None,
                 plot_profile_defs = None):
        self.savedir = savedir
        self.stdJETplots = stdJETplots
        self.plot_LOS_defs = plot_LOS_defs
        self.plot_profile_defs = plot_profile_defs

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

        if stdJETplots:
            self.plotJET()

    @property
    def data2d(self):
        return self.__data2d

    @property
    def res_dict(self):
        return self.__res_dict

    @staticmethod
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

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
    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)

    def getLineIntSortedDataByChordidx(self, diag, mapList):
        """
            input:
                mapList: list of dict keys below the 'chord' level (e.g., ['los_int', 'stark', 'fit', 'ne']
                diag: synth diag name string
        """
        tmp = []
        chordidx = []
        for chord in self.res_dict[diag]:
            parval = PyprocPlot.getFromDict(self.res_dict[diag][chord], mapList)
            # if isinstance(parval, float):
            tmp.append(parval)
            chordidx.append(int(chord)-1)

        chords = np.asarray(chordidx)
        sort_idx = np.argsort(chords, axis=0)
        sorted_parvals = np.asarray(tmp)[sort_idx]

        return sorted_parvals

    # def get1DdataAtMaxLOSElecDen(self, diag, paramstr, nAvgNeighbs=2):
    #
    #     ne_los_1d_max_idx, val = find_nearest(res_dict[diag_key][chord_key]['los_1d']['ne'],
    #                                           np.max(res_dict[diag_key][chord_key]['los_1d']['ne']))
    #     # Find n0, Te corresponding to max ne along LOS (include nearest neighbours and average)
    #     if (ne_los_1d_max_idx + 1) == len(res_dict[diag_key][chord_key]['los_1d']['n0']):
    #         ne_los_1d_max.append(
    #             np.average(np.array((res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx - 1],
    #                                  res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx]))))
    #
    #     elif (ne_los_1d_max_idx + 2) == len(res_dict[diag_key][chord_key]['los_1d']['n0']):
    #         ne_los_1d_max.append(
    #             np.average(np.array((res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 1],
    #                                  res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx]))))
    #
    #     elif (ne_los_1d_max_idx + 3) == len(res_dict[diag_key][chord_key]['los_1d']['n0']):
    #         ne_los_1d_max.append(
    #             np.average(np.array((res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 2],
    #                                  res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 1],
    #                                  res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx]))))
    #
    #     else:
    #         ne_los_1d_max.append(
    #             np.average(np.array((res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 3],
    #                                  res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 2],
    #                                  res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx + 1],
    #                                  res_dict[diag_key][chord_key]['los_1d']['ne'][ne_los_1d_max_idx]))))
    #
    #
    #     return sorted_parvals

    # def plotLOS(self, include):

    # SOME USEFUL PLOTTING CONFIGURATIONS

    @staticmethod
    def plotProf_ne_te_parbal_atden(x, ne, Te, Sion, Srec, n0, axobj=None, plot_at_max_ne_along_LOS=False,
                                plot_target=False):
        if axobj == None:
            fig, axobj = plt.subplots(nrows=4, ncols=1, figsize=(6, 12), sharex=True)
            left = 0.2  # the left side of the subplots of the figure
            right = 0.95  # the right side of the subplots of the figure
            bottom = 0.15  # the bottom of the subplots of the figure
            top = 0.93  # the top of the subplots of the figure
            wspace = 0.18  # the amount of width reserved for blank space between subplots
            hspace = 0.1  # the amount of height reserved for white space between subplots
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        axobj[0].plot(x, ne)
        axobj[1].plot(x, Te)
        axobj[2].semilogy(x, Sion)
        axobj[3].semilogy(x, n0)

if __name__=='__main__':

    plot_LOS_defs = namedtuple('plot_LOS_defs', 'diag', 'chord', 'llim')
    plot_profile_defs = namedtuple('plot_profile_defs', 'diag', 'chord', 'llim')

    plot_LOS_defs.diag = 'KT3A'
    plot_LOS_defs.chord = '12'
    plot_LOS_defs.llim = [5.3, 5.6]

    # 'tranfile': '/u/cstavrou/cmg/catalog/edge2d/jet/81472/jun0617/seq#2/tran',
    # 'tranfile': '/u/cstavrou/cmg/catalog/edge2d/jet/81472/may2717/seq#2/tran',

    workdir = '/work/bloman/pyproc/'
    # Example
    savedir = workdir + 'cstavrou_cmg_catalog_edge2d_jet_81472_may2717_seq#2'
    o = PyprocPlot(savedir)

    # Print out results dictionary tree
    PyprocPlot.pprint_json(o.res_dict['KT3A']['1']['los_int'])

    ne_kt3a = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'stark', 'fit', 'ne'])
    Te_kt3a = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_360_400'])
    Sion = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'adf11_fit', 'Sion'])
    Srec = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'adf11_fit', 'Srec'])
    n0delL = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'Ly_alpha_fit', 'n0delL'])
    p2_kt3a = o.getLineIntSortedDataByChordidx('KT3A', ['chord', 'p2'])


    fig1, ax1 = plt.subplots(nrows=4, ncols=1, figsize=(6,12), sharex=True)
    left  = 0.2  # the left side of the subplots of the figure
    right = 0.95    # the right side of the subplots of the figure
    bottom = 0.15   # the bottom of the subplots of the figure
    top = 0.93      # the top of the subplots of the figure
    wspace = 0.18   # the amount of width reserved for blank space between subplots
    hspace = 0.1  # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    o.plotProf_ne_te_parbal_atden(p2_kt3a[:,0], ne_kt3a, Te_kt3a, Sion, Srec, n0delL, axobj=ax1)

    savedir = workdir + 'bloman_cmg_catalog_edge2d_jet_81472_sep1217_seq#3'
    o = PyprocPlot(savedir)

    ne_kt3a = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'stark', 'fit', 'ne'])
    Te_kt3a = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'ff_fb_continuum', 'fit', 'fit_te_360_400'])
    Sion = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'adf11_fit', 'Sion'])
    Srec = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'adf11_fit', 'Srec'])
    n0delL = o.getLineIntSortedDataByChordidx('KT3A', ['los_int', 'Ly_alpha_fit', 'n0delL'])
    p2_kt3a = o.getLineIntSortedDataByChordidx('KT3A', ['chord', 'p2'])

    o.plotProf_ne_te_parbal_atden(p2_kt3a[:,0], ne_kt3a, Te_kt3a, Sion, Srec, n0delL, axobj=ax1)

    plt.show()