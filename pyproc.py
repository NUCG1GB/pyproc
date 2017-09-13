
# Import external modules

import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import idlbridge as idl
from matplotlib.collections import PatchCollection
from matplotlib import patches
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, LineString
from pyADASutils import adas_adf15_utils, adas_adf11_utils, continuum_utils
from pyproc import machine_defs

# font = {'family': 'normal',
#         'weight': 'normal',
#         'size': 16}
# import matplotlib
# matplotlib.rc('font', **font)
# import matplotlib.font_manager as font_manager
# path = '/usr/share/fonts/msttcore/arial.ttf'
# prop = font_manager.FontProperties(fname=path)
# matplotlib.rcParams['font.family'] = prop.get_name()
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = prop.get_name()

at_sym = ['H','HE','LI','BE','B','C','N','O','F','NE','NA','MG',
          'AL','SI','P','S','CL','AR','K','CA','SC','TI','V','CR',
          'MN','FE','CO','NI','CU','ZN','GA','GE','AS','SE','BR',
          'KR','RB','SR','Y','ZR','NB','MO','TC','RU','RH','PD',
          'AG','CD','IN','SN','SB','TE','I','XE','CS','BA','LA',
          'CE','PR','ND','PM','SM','EU','GD','TB','DY','HO','ER',
          'TM','YB','LU','HF','TA','W','RE','OS','IR','PT','AU',
          'HG','TL','PB','BI','PO','AT','RN','FR','RA','AC','TH','P']

roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def gaussian(cwl, wv, area, fwhm):
    sigma = fwhm / (2. * np.sqrt(2 * np.log(2.)) )
    g = area * (1./(sigma*np.sqrt(2.*np.pi))) * np.exp(-1.*((wv - cwl)**2) / (2*(sigma**2)) )
    return g

class Pyproc:
    """
        Class to read and store EDGE2D-EIRENE results
    """
    def __init__(self, ADAS_dict, tranfile='', machine='JET', interactive_plots = False,
                 spec_line_dict=None, diag_list=None, calc_synth_spec_features=None,
                 calc_NII_afg_feature=False, save_synth_diag=False,
                 synth_diag_save_file=None, data2d_save_file=None):

        self.ADAS_dict = ADAS_dict
        self.tranfile = tranfile
        self.machine = machine

        #Flags
        self.calc_NII_afg_feature = calc_NII_afg_feature

        self.geom = None
        self.teve = None
        self.den = None
        self.denel = None
        self.da = None
        self.korpg = None
        self.rmesh = None
        self.rvertp = None
        self.zmesh = None
        self.zvertp = None
        self.NE2Ddata = None
        self.patches = None
        self.cells = None
        self.H_adf11 = None
        self.spec_line_dict = spec_line_dict

        self.sep_poly = None
        self.shply_sep_poly = None
        self.sep_poly_below_xpt = None
        self.shply_sep_poly_below_xpt = None

        self.wall_poly = None
        self.shply_wall_poly = None

        self.rv = None
        self.zv = None

        # variables mapped onto edge2d grid
        self.te = None
        self.ne = None
        self.ni = None
        self.n0 = None

        # impurities
        self.zch = None
        self.imp1_atom_num = None
        self.imp2_atom_num = None
        self.imp1_chrg_idx = []
        self.imp2_chrg_idx = []
        self.imp1_denz = []
        self.imp2_denz = []
        self.imp1_adf11 = None
        self.imp2_adf11 = None

        # Dictionary for storing synthetic diagnostic objects
        self.synth_diag = {}

        self.readtran_eproc_idl_temp()

        # self.plot_regions()
        # plt.show()

        self.calc_H_emiss()
        self.calc_H_rad_power()

        # Calc ff+fb emissivity.
        # Use KL11 cam e WI filter.
        # TODO: generalise to accept other filters
        filter_file = '/home/bloman/python_bal/TomI/kl11_filters/WI40096_kl11_filter_peak_norm.txt'
        filter_curve = np.genfromtxt(filter_file, delimiter=',', skiprows=4)
        print('FF+FB emission filter file: ', filter_file)
        self.calc_ff_fb_emiss(filter_curve[:,0], filter_curve[:,1])

        if self.imp1_atom_num or self.imp2_atom_num:
            self.calc_imp_rad_power()

        if self.spec_line_dict and (self.imp1_atom_num or self.imp2_atom_num):
            self.calc_imp_emiss()
            # Rrng = [2.36, 2.96]
            # Zrng = [-1.73, -1.29]
            # self.plot_imp_lines('7', '2', '5005.86', diag_list=['KT3A'], plot_LOS=True, Rrng=Rrng, Zrng=Zrng, savefig=True)
            # plt.show()


        # CALULCATE PRAD CORRESPONDING TO BORP MACRO REGIONS
        # NOTE: Divertor Prad below Z=-1.2, not Z=Zxpt (consistent with ppfuid='bloman' BORP processing)
        self.calc_BORP()
        # CALCULATE POWER FLOW INTO INNER AND OUTER DIVERTOR AT Z=-1.2
        self.calc_qpol_div()

        # Get machine definitions
        if self.machine == 'JET':
            self.defs = machine_defs.get_JETdefs()
        elif self.machine == 'DIID':
            self.defs = machine_defs.get_DIIIDdefs()

        if diag_list:
            for key in diag_list:
                if key in self.defs.diag_dict.keys():
                    self.synth_diag[key] = SynthDiag(self.defs, diag=key,
                                                     spec_line_dict = self.spec_line_dict,
                                                     imp1_atom_num=self.imp1_atom_num, imp2_atom_num=self.imp2_atom_num,
                                                     calc_NII_afg_feature=self.calc_NII_afg_feature)
                    for chord in self.synth_diag[key].chords:
                        self.los_intersect(chord)
                        chord.orthogonal_polys()
                    # Rrng = [2.36, 2.96]
                    # Zrng = [-1.73, -1.29]
                    # self.plot_imp_lines('7', '2', '5005.86', diag_list=['KT3A'], plot_LOS=True, Rrng=Rrng, Zrng=Zrng, savefig=True)
                    # plt.show()
                        chord.calc_int_and_1d_los_quantities()
                        if calc_synth_spec_features:
                            print('Calculating synthetic spectra for diag: ', key)
                            chord.calc_int_and_1d_los_synth_spectra()

        if save_synth_diag:
            if self.synth_diag:
                self.save_synth_diag_data(savefile=synth_diag_save_file)

        if data2d_save_file:
            # pickle serialization of e2deirpostproc object
            # delete ADAS_dict object to save space
            output = open(data2d_save_file, 'wb')
            pickle.dump(self, output)
            output.close()

        # Plotting routines
        if interactive_plots:
            self.plot_te_ne_n0()
            if diag_list:
                for key in diag_list:
                    self.synth_diag[key].plot_synth_spec_edge2d_data()

            self.plot_H_lines()
            
            #
            Rrng = [2.36, 2.96]
            Zrng = [-1.73, -1.29]
            self.plot_ff_fb_filtered_emiss(diag_list=['KT3A'], plot_LOS=True, Rrng=Rrng, Zrng=Zrng, savefig=True)

            plt.show()

            if self.spec_line_dict and (self.imp1_atom_num or self.imp2_atom_num):
                fig1, ax1 = plt.subplots(nrows=6, ncols=1, figsize=(12,8), sharex=True)
                fig2, ax2 = plt.subplots(nrows=4, ncols=1, figsize=(6,12))
                self.plot_imp_rad_loss(ax1, ax2, color='b')

                Rrng = [2.36, 2.96]
                Zrng = [-1.73, -1.29]
                self.plot_imp_lines('7', '4', '3481.83', diag_list=['KT3A'], plot_LOS=True, Rrng=Rrng, Zrng=Zrng, savefig=True)
                self.plot_imp_lines('7', '3', '4100.51', diag_list=['KT3A'], plot_LOS=True, Rrng=Rrng, Zrng=Zrng, savefig=True)
                self.plot_imp_lines('7', '2', '5005.86', diag_list=['KT3A'], plot_LOS=True, Rrng=Rrng, Zrng=Zrng, savefig=True)

            plt.show()

    def __getstate__(self):
        """
            For removing the large ADAS_dict from the object for pickling
            See: https://docs/python/org/2/library/pickle.html#example
        """
        odict = self.__dict__.copy() # copy the dict since we change it
        del odict['ADAS_dict']
        return odict

    def __setstate__(self, dict):
        # TODO: Read external ADAS_dict object and add to dict for unpickling
        self.__dict__.update(dict)

    def save_synth_diag_data(self, savefile=None):
        # output = open(savefile, 'wb')
        outdict = {}
        for diag_key in self.synth_diag:
            outdict[diag_key] = {}
            for chord in self.synth_diag[diag_key].chords:
                outdict[diag_key].update({chord.chord_num:{}})
                # outdict[diag_key][chord.chord_num].update({'H_emiss':chord.los_int['H_emiss']})
                outdict[diag_key][chord.chord_num]['spec_line_dict'] = self.spec_line_dict
                outdict[diag_key][chord.chord_num]['los_1d'] = chord.los_1d
                outdict[diag_key][chord.chord_num]['los_int'] = chord.los_int
                for spectrum in chord.los_int_spectra:
                    outdict[diag_key][chord.chord_num]['los_int'].update({spectrum:chord.los_int_spectra[spectrum]})
                    # outdict[diag_key][chord.chord_num]['los_1d'].update({spectrum: chord.los_1d_spectra[spectrum]})
                # outdict[diag_key][chord.chord_num]['Srec'] = chord.los_int['Srec']
                # outdict[diag_key][chord.chord_num]['Sion'] = chord.los_int['Sion']
                if chord.shply_intersects_w_sep:
                    outdict[diag_key][chord.chord_num].update({'chord':{'p1':chord.p1, 'p2':chord.p2unmod, 'w2':chord.w2unmod, 'sep_intersect_below_xpt':[chord.shply_intersects_w_sep.coords.xy[0][0],chord.shply_intersects_w_sep.coords.xy[1][0]]}})
                else:
                    outdict[diag_key][chord.chord_num].update({'chord':{'p1':chord.p1, 'p2':chord.p2unmod, 'w2':chord.w2unmod, 'sep_intersect_below_xpt':None}})

        # SAVE IN JSON FORMAT TO ENSURE PYTHON 2/3 COMPATIBILITY
        with open (savefile, mode='w', encoding='utf-8') as f:
            json.dump(outdict, f, indent=2)

        print('Saving synthetic diagnostic data to:', savefile)

        # pickle.dump(outdict, output)
        #
        # output.close()

    def get_eproc_param_temp(self, funstr, parstr, par2str=None, args=None):

        idl.execute("""tranfile=' """ + self.tranfile + """ ' """)
        if par2str and args:
            if type(par2str) == int:
                cmd = 'ret=' + funstr +'(tranfile,'+ """'""" + parstr + """'""" + ',' + str(par2str) + ',' + args + ')'
            elif type(par2str) == str:
                cmd = 'ret=' + funstr +'(tranfile,'+ """'""" + parstr + """'""" + ','  + """'""" + par2str + """'""" + ',' + args + ')'

            idl.execute(cmd)
            idl.execute('xdata=ret.xdata')
            idl.execute('ydata=ret.ydata')
            idl.execute('npts=ret.npts')
            return {'xdata':idl.get('xdata'), 'ydata':idl.get('ydata'), 'npts':idl.get('npts')}
        else:
            cmd = 'ret=' + funstr +'(tranfile,'+ """'""" + parstr + """'""" + ')'
            idl.execute(cmd)
            idl.execute('data=ret.data')
            idl.execute('npts=ret.npts')
            return {'data':idl.get('data'), 'npts':idl.get('npts')}

    def readtran_eproc_idl_temp(self):

        print('Getting data from ' + self.tranfile )

        IDL_EprocGeom= idl.export_function("EprocGeom")
        self.geom = IDL_EprocGeom(self.tranfile)
        IDL_EprocDataRead= idl.export_function("EprocDataRead")

        self.rmesh = self.get_eproc_param_temp("EprocDataRead", 'RMESH')

        self.rmesh = self.get_eproc_param_temp("EprocDataRead", 'RMESH')
        self.zmesh = self.get_eproc_param_temp("EprocDataRead", 'ZMESH')
        self.rvertp = self.get_eproc_param_temp("EprocDataRead", 'RVERTP')
        self.zvertp = self.get_eproc_param_temp("EprocDataRead", 'ZVERTP')
        self.teve = self.get_eproc_param_temp("EprocDataRead", 'TEVE')
        self.den = self.get_eproc_param_temp("EprocDataRead", 'DEN')
        self.denel = self.get_eproc_param_temp("EprocDataRead", 'DENEL')
        self.da = self.get_eproc_param_temp("EprocDataRead", 'DA')
        self.korpg = self.get_eproc_param_temp("EprocDataRead", 'KORPG')
        self.rmesh = self.get_eproc_param_temp("EprocDataRead", 'RMESH')
        self.sirec = self.get_eproc_param_temp("EprocDataRead", 'SIREC')
        self.soun = self.get_eproc_param_temp("EprocDataRead", 'SOUN')

        # GET INNER AND OUTER TARGET DATA
        self.pflxd_OT = self.get_eproc_param_temp("EprocRow", 'PFLXD', 'OT', args = 'ALL_POINTS=0')
        self.pflxd_IT = self.get_eproc_param_temp("EprocRow", 'PFLXD', 'IT', args = 'ALL_POINTS=0')
        self.teve_OT = self.get_eproc_param_temp("EprocRow", 'TEVE', 'OT', args = 'ALL_POINTS=0')
        self.teve_IT = self.get_eproc_param_temp("EprocRow", 'TEVE', 'IT', args = 'ALL_POINTS=0')
        self.denel_OT = self.get_eproc_param_temp("EprocRow", 'DENEL', 'OT', args = 'ALL_POINTS=0')
        self.denel_IT = self.get_eproc_param_temp("EprocRow", 'DENEL', 'IT', args = 'ALL_POINTS=0')
        self.da_OT = self.get_eproc_param_temp("EprocRow", 'DA', 'OT', args = 'ALL_POINTS=0')
        self.da_IT = self.get_eproc_param_temp("EprocRow", 'DA', 'IT', args = 'ALL_POINTS=0')


        # GET GEOM INFO
        IDL_EprocGeom= idl.export_function("EprocGeom") # This function is now deprecated!
        self.geom = IDL_EprocGeom(self.tranfile)
        self.geom['zpx']*=-1.0

        # GET MID-PLANE NE PROFILE
        self.ne_OMP = self.get_eproc_param_temp("EprocRow", 'DENEL', 'OMP', args = 'ALL_POINTS=0')

        # GET INNER AND OUTER QPARTOT AT Z=-1.2
        # FOR 81472 GEOM THIS CORRESPONDS TO ROWS 66 (IN) AND 26 (OUT)
        # TODO: GENERALIZE TO OTHER GEOMETRIES
        self.qpartot_HFS = self.get_eproc_param_temp("EprocRow", 'QPARTOT', 66, args = 'ALL_POINTS=0')
        self.qpartot_LFS = self.get_eproc_param_temp("EprocRow", 'QPARTOT', 26, args = 'ALL_POINTS=0')
        self.bpol_btot_HFS = self.get_eproc_param_temp("EprocRow", 'SH', 66, args = 'ALL_POINTS=0')
        self.bpol_btot_LFS = self.get_eproc_param_temp("EprocRow", 'SH', 26, args = 'ALL_POINTS=0')
        self.qpartot_HFS_rmesh = self.get_eproc_param_temp("EprocRow", 'RMESH', 66, args = 'ALL_POINTS=0')
        self.qpartot_LFS_rmesh = self.get_eproc_param_temp("EprocRow", 'RMESH', 26, args = 'ALL_POINTS=0')
        self.qpartot_HFS_zmesh = self.get_eproc_param_temp("EprocRow", 'ZMESH', 66, args = 'ALL_POINTS=0')
        self.qpartot_LFS_zmesh = self.get_eproc_param_temp("EprocRow", 'ZMESH', 26, args = 'ALL_POINTS=0')
        # GET POWER CROSSING THE SEPARATRIX
        self.powsol = self.get_eproc_param_temp("EprocDataRead", 'POWSOL')

        # GET IMPURITY DATA
        self.zch = self.get_eproc_param_temp("EprocDataRead", 'ZCH') # impurity atomic number (max 2 impurities)
        if self.zch['data'][0] > 0.0:
            self.imp1_atom_num = self.zch['data'][0]
            # First append neutral density, then each charge state density
            self.imp1_denz.append(self.get_eproc_param_temp("EprocDataRead", 'DZ_1'))
            for i in range(int(self.zch['data'][0])):
                if i < 9 :
                    stridx = '0' + str(i+1)
                    self.imp1_chrg_idx.append(stridx)
                else:
                    stridx = str(i+1)
                    self.imp1_chrg_idx.append(stridx)
                self.imp1_denz.append(self.get_eproc_param_temp("EprocDataRead", 'DENZ'+stridx))
            if self.zch['data'][1] > 0.0:
                self.imp2_atom_num = self.zch['data'][1]
                # First append neutral density, then each charge state density
                self.imp2_denz.append(self.get_eproc_param_temp("EprocDataRead", 'DZ_2'))
                for i in range(int(self.zch['data'][1])):
                    if i + int(self.zch['data'][0]) < 9:
                        stridx = '0' + str(i + 1 + int(self.zch['data'][0]))
                        self.imp2_chrg_idx.append(stridx)
                    else:
                        stridx = str(i + 1 + int(self.zch['data'][0]))
                        self.imp2_chrg_idx.append(stridx)
                    self.imp2_denz.append(self.get_eproc_param_temp("EprocDataRead", 'DENZ'+stridx))
        else:
            # If no impurities found reset
            self.zch = None

        # Z data is upside down
        self.zvertp['data'] *=-1.0
        self.zmesh['data'] *=-1.0

        self.NE2Ddata = 0
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i]
            if j != 0: self.NE2Ddata = self.NE2Ddata+1

        self.cells = []
        self.patches = []

        self.rv = np.zeros((self.NE2Ddata, 5))
        self.zv = np.zeros((self.NE2Ddata, 5))
        self.te = np.zeros((self.NE2Ddata))
        self.ne = np.zeros((self.NE2Ddata))
        self.ni = np.zeros((self.NE2Ddata))
        self.n0 = np.zeros((self.NE2Ddata))
        self.srec = np.zeros((self.NE2Ddata))
        self.sion = np.zeros((self.NE2Ddata))
        self.imp1_den = np.zeros((len(self.imp1_chrg_idx)+1,self.NE2Ddata))
        self.imp2_den = np.zeros((len(self.imp2_chrg_idx)+1,self.NE2Ddata))
        k = 0
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i] - 1 # gotcha: convert from fortran indexing to idl/python
            if j >= 0:
                j*=5
                self.rv[k] = [self.rvertp['data'][j],  self.rvertp['data'][j+1], self.rvertp['data'][j+2], self.rvertp['data'][j+3], self.rvertp['data'][j]]
                self.zv[k] = [self.zvertp['data'][j],  self.zvertp['data'][j+1], self.zvertp['data'][j+2], self.zvertp['data'][j+3], self.zvertp['data'][j]]
                self.te[k] = self.teve['data'][i]
                self.ni[k] = self.den['data'][i]
                self.ne[k] = self.denel['data'][i]
                self.n0[k] = self.da['data'][i]
                self.srec[k] = self.sirec['data'][i]
                self.sion[k] = self.soun['data'][i]

                if self.imp1_atom_num:
                    for ichrg in range(len(self.imp1_chrg_idx)+1):# +1 to include neutral density
                        self.imp1_den[ichrg,k] = self.imp1_denz[ichrg]['data'][i]
                if self.imp2_atom_num:
                    for ichrg in range(len(self.imp2_chrg_idx)+1):# +1 to include neutral density
                        self.imp2_den[ichrg,k] = self.imp2_denz[ichrg]['data'][i]

                poly = patches.Polygon([(self.rv[k,0], self.zv[k,0]), (self.rv[k,1], self.zv[k,1]), (self.rv[k,2], self.zv[k,2]), (self.rv[k,3], self.zv[k,3]), (self.rv[k,4], self.zv[k,4])], closed=True)
                self.patches.append(poly)
                # create Cell object for each polygon containing the relevant field data
                shply_poly = Polygon(poly.get_xy())

                # SCALE TE NE N0: FOR TESTING ONLY. MUST BE COMMENTED OUT!
                # te_scal=1.0
                # ne_scal=1.0
                # if self.zv[k][0] <= -1.6:
                #     n0_scal=1000.0
                # else:
                #     n0_scal = 1.0
                te_scal = 1.0
                ne_scal = 1.0
                n0_scal = 1.0

                self.cells.append(Cell(self.rmesh['data'][i], self.zmesh['data'][i],
                                       poly=shply_poly, te=te_scal*self.te[k],
                                       ne=ne_scal*self.ne[k], ni=ne_scal*self.ni[k],
                                       n0=n0_scal*self.n0[k], Srec=self.srec[k], Sion=self.sion[k],
                                       imp1_den = self.imp1_den[:,k], imp2_den=self.imp2_den[:,k]))
                k+=1

        ##############################################
        # GET STRIKE POINT COORDS AND SEPARATRIX POLY
        rtmp = self.get_eproc_param_temp("EprocRing", 'RMESH', 'S01', args = 'parallel=1')
        ztmp = self.get_eproc_param_temp("EprocRing", 'ZMESH', 'S01', args = 'parallel=1')
        ztmp['ydata'] = -1.0*ztmp['ydata']
        self.osp = [rtmp['ydata'][0], ztmp['ydata'][0]]
        self.isp = [rtmp['ydata'][rtmp['npts']-1], ztmp['ydata'][ztmp['npts']-1]]

        # find start and end index
        istart = -1
        iend = -1
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i] - 1 # gotcha: convert from fortran indexing to idl/python
            j*=5
            if j >= 0:
                if (self.rvertp['data'][j+4] == rtmp['ydata'][0]) and (self.zvertp['data'][j+4] == ztmp['ydata'][0]):
                    istart = i
                if (self.rvertp['data'][j+4] == rtmp['ydata'][rtmp['npts']-1]) and (self.zvertp['data'][j+4] == ztmp['ydata'][ztmp['npts']-1]):
                    iend = i

        sep_points = []
        sep_points_below_xpt = []
        k = 0
        for i in range(istart, iend+1):
            j = self.korpg['data'][i] - 1
            j*=5
            sep_points.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            if self.zvertp['data'][j+0] <= self.geom['zpx']:
                sep_points_below_xpt.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            # sep_points.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            k+=1
        sep_points_below_xpt.append((self.geom['rpx'], self.geom['zpx']))

        self.sep_poly = patches.Polygon(sep_points, closed=False, ec='w', linestyle='dashed', lw=2.0, fc='None', zorder=10)
        self.shply_sep_poly = Polygon(self.sep_poly.get_xy())
        self.sep_poly_below_xpt = patches.Polygon(sep_points_below_xpt, closed=False, ec='k', fc='None', zorder=10)
        self.shply_sep_poly_below_xpt = Polygon(self.sep_poly_below_xpt.get_xy())

        ##############################################
        # GET WALL POLYGON
        rtmp1 = self.get_eproc_param_temp("EprocDataRead", 'RVESM1')
        ztmp1 = self.get_eproc_param_temp("EprocDataRead", 'ZVESM1')
        rtmp2 = self.get_eproc_param_temp("EprocDataRead", 'RVESM2')
        ztmp2 = self.get_eproc_param_temp("EprocDataRead", 'ZVESM2')
        # z coord is upside down
        ztmp1['data'] = -1.0*ztmp1['data']
        ztmp2['data'] = -1.0*ztmp2['data']
        # remove non connected segments
        useSegment = np.zeros((rtmp1['npts']))
        nsegs = 0
        for i in range(rtmp1['npts']):
            check = 0
            if i != 0:
                if ((rtmp1['data'][i] == rtmp2['data'][i-1]) and (ztmp1['data'][i] == ztmp2['data'][i-1])) or \
                    ((rtmp2['data'][i] == rtmp1['data'][i-1]) and (ztmp2['data'][i] == ztmp1['data'][i-1])):
                    check = 1
            if i != (rtmp1['npts']):
                if ((rtmp1['data'][i] == rtmp2['data'][i+1]) and (ztmp1['data'][i] == ztmp2['data'][i+1])) or \
                    ((rtmp2['data'][i] == rtmp1['data'][i+1]) and (ztmp2['data'][i] == ztmp1['data'][i+1])):
                    check = 1
            if check:
                useSegment[i] = 1
                nsegs += 1

        wall_poly_pts = []
        for i in range(rtmp1['npts']):
            if useSegment[i]:
                wall_poly_pts.append((rtmp1['data'][i],ztmp1['data'][i]))
                wall_poly_pts.append((rtmp2['data'][i],ztmp2['data'][i]))
        wall_poly_pts.append(wall_poly_pts[0]) # connect last point to first to complete wall polygon

        self.wall_poly = patches.Polygon(wall_poly_pts, closed=False, ec='k', lw=2.0, fc='None', zorder=10)
        self.shply_wall_poly = Polygon(self.wall_poly.get_xy())

    def readtran_eproc_idl(self):

        print('Getting data from ' + self.tranfile )

        IDL_EprocGeom= idl.export_function("EprocGeom")
        self.geom = IDL_EprocGeom(self.tranfile)
        IDL_EprocDataRead= idl.export_function("EprocDataRead")

        self.rmesh = IDL_EprocDataRead(self.tranfile, 'RMESH')
        self.zmesh = IDL_EprocDataRead(self.tranfile, 'ZMESH')
        self.rvertp = IDL_EprocDataRead(self.tranfile, 'RVERTP')
        self.zvertp = IDL_EprocDataRead(self.tranfile, 'ZVERTP')
        self.teve = IDL_EprocDataRead(self.tranfile, 'TEVE')
        self.den = IDL_EprocDataRead(self.tranfile, 'DEN')
        self.denel = IDL_EprocDataRead(self.tranfile, 'DENEL')
        self.da = IDL_EprocDataRead(self.tranfile, 'DA')
        self.korpg = IDL_EprocDataRead(self.tranfile, 'KORPG')
        self.rmesh = IDL_EprocDataRead(self.tranfile, 'RMESH')
        self.sirec = IDL_EprocDataRead(self.tranfile, 'SIREC')
        self.soun = IDL_EprocDataRead(self.tranfile, 'SOUN')

        # GET INNER AND OUTER TARGET DATA
        IDL_EprocRow= idl.export_function("EprocRow") # This function is now depracated!
        self.pflxd_OT = IDL_EprocRow(self.tranfile, 'PFLXD', 'OT', ALL_POINTS=0)
        self.pflxd_IT = IDL_EprocRow(self.tranfile, 'PFLXD', 'IT', ALL_POINTS=0)
        self.teve_OT = IDL_EprocRow(self.tranfile, 'TEVE', 'OT', ALL_POINTS=0)
        self.teve_IT = IDL_EprocRow(self.tranfile, 'TEVE', 'IT', ALL_POINTS=0)
        self.denel_OT = IDL_EprocRow(self.tranfile, 'DENEL', 'OT', ALL_POINTS=0)
        self.denel_IT = IDL_EprocRow(self.tranfile, 'DENEL', 'IT', ALL_POINTS=0)
        self.da_OT = IDL_EprocRow(self.tranfile, 'DA', 'OT', ALL_POINTS=0)
        self.da_IT = IDL_EprocRow(self.tranfile, 'DA', 'IT', ALL_POINTS=0)
        # plt.plot(self.teve_OT['xdata'][:self.teve_OT['npts']], self.teve_OT['ydata'][:self.teve_OT['npts']], 'ok')
        # plt.show()

        # GET GEOM INFO
        IDL_EprocGeom= idl.export_function("EprocGeom") # This function is now deprecated!
        self.geom = IDL_EprocGeom(self.tranfile)
        self.geom['zpx']*=-1.0

        # GET MID-PLANE NE PROFILE
        IDL_EprocRing= idl.export_function("EprocRing")
        self.ne_OMP = IDL_EprocRow(self.tranfile, 'DENEL', 'OMP', ALL_POINTS=0)

        # GET INNER AND OUTER QPARTOT AT Z=-1.2
        # FOR 81472 GEOM THIS CORRESPONDS TO ROWS 66 (IN) AND 26 (OUT)
        # TODO: GENERALIZE TO OTHER GEOMETRIES
        self.qpartot_HFS = IDL_EprocRow(self.tranfile, 'QPARTOT', 66, ALL_POINTS=0)
        self.qpartot_LFS = IDL_EprocRow(self.tranfile, 'QPARTOT', 26, ALL_POINTS=0)
        self.bpol_btot_HFS = IDL_EprocRow(self.tranfile, 'SH', 66, ALL_POINTS=0)
        self.bpol_btot_LFS = IDL_EprocRow(self.tranfile, 'SH', 26, ALL_POINTS=0)
        self.qpartot_HFS_rmesh = IDL_EprocRow(self.tranfile, 'RMESH', 66, ALL_POINTS=0)
        self.qpartot_LFS_rmesh = IDL_EprocRow(self.tranfile, 'RMESH', 26, ALL_POINTS=0)
        self.qpartot_HFS_zmesh = IDL_EprocRow(self.tranfile, 'ZMESH', 66, ALL_POINTS=0)
        self.qpartot_LFS_zmesh = IDL_EprocRow(self.tranfile, 'ZMESH', 26, ALL_POINTS=0)
        # GET POWER CROSSING THE SEPARATRIX
        self.powsol  = IDL_EprocDataRead(self.tranfile, 'POWSOL')

        # GET IMPURITY DATA
        self.zch = IDL_EprocDataRead(self.tranfile, 'ZCH') # impurity atomic number (max 2 impurities)
        if self.zch['data'][0] > 0.0:
            self.imp1_atom_num = self.zch['data'][0]
            # First append neutral density, then each charge state density
            self.imp1_denz.append(IDL_EprocDataRead(self.tranfile, 'DZ_1'))
            for i in range(int(self.zch['data'][0])):
                if i < 9 :
                    stridx = '0' + str(i+1)
                    self.imp1_chrg_idx.append(stridx)
                else:
                    stridx = str(i+1)
                    self.imp1_chrg_idx.append(stridx)
                self.imp1_denz.append(IDL_EprocDataRead(self.tranfile, 'DENZ'+stridx))
            if self.zch['data'][1] > 0.0:
                self.imp2_atom_num = self.zch['data'][1]
                # First append neutral density, then each charge state density
                self.imp2_denz.append(IDL_EprocDataRead(self.tranfile, 'DZ_2'))
                for i in range(int(self.zch['data'][1])):
                    if i + int(self.zch['data'][0]) < 9:
                        stridx = '0' + str(i + 1 + int(self.zch['data'][0]))
                        self.imp2_chrg_idx.append(stridx)
                    else:
                        stridx = str(i + 1 + int(self.zch['data'][0]))
                        self.imp2_chrg_idx.append(stridx)
                    self.imp2_denz.append(IDL_EprocDataRead(self.tranfile, 'DENZ'+stridx))
        else:
            # If no impurities found reset
            self.zch = None

        # Z data is upside down
        self.zvertp['data'] *=-1.0
        self.zmesh['data'] *=-1.0

        self.NE2Ddata = 0
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i]
            if j != 0: self.NE2Ddata = self.NE2Ddata+1

        self.cells = []
        self.patches = []

        self.rv = np.zeros((self.NE2Ddata, 5))
        self.zv = np.zeros((self.NE2Ddata, 5))
        self.te = np.zeros((self.NE2Ddata))
        self.ne = np.zeros((self.NE2Ddata))
        self.ni = np.zeros((self.NE2Ddata))
        self.n0 = np.zeros((self.NE2Ddata))
        self.srec = np.zeros((self.NE2Ddata))
        self.sion = np.zeros((self.NE2Ddata))
        self.imp1_den = np.zeros((len(self.imp1_chrg_idx)+1,self.NE2Ddata))
        self.imp2_den = np.zeros((len(self.imp2_chrg_idx)+1,self.NE2Ddata))
        k = 0
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i] - 1 # gotcha: convert from fortran indexing to idl/python
            if j >= 0:
                j*=5
                self.rv[k] = [self.rvertp['data'][j],  self.rvertp['data'][j+1], self.rvertp['data'][j+2], self.rvertp['data'][j+3], self.rvertp['data'][j]]
                self.zv[k] = [self.zvertp['data'][j],  self.zvertp['data'][j+1], self.zvertp['data'][j+2], self.zvertp['data'][j+3], self.zvertp['data'][j]]
                self.te[k] = self.teve['data'][i]
                self.ni[k] = self.den['data'][i]
                self.ne[k] = self.denel['data'][i]
                self.n0[k] = self.da['data'][i]
                self.srec[k] = self.sirec['data'][i]
                self.sion[k] = self.soun['data'][i]

                if self.imp1_atom_num:
                    for ichrg in range(len(self.imp1_chrg_idx)+1):# +1 to include neutral density
                        self.imp1_den[ichrg,k] = self.imp1_denz[ichrg]['data'][i]
                if self.imp2_atom_num:
                    for ichrg in range(len(self.imp2_chrg_idx)+1):# +1 to include neutral density
                        self.imp2_den[ichrg,k] = self.imp2_denz[ichrg]['data'][i]

                poly = patches.Polygon([(self.rv[k,0], self.zv[k,0]), (self.rv[k,1], self.zv[k,1]), (self.rv[k,2], self.zv[k,2]), (self.rv[k,3], self.zv[k,3]), (self.rv[k,4], self.zv[k,4])], closed=True)
                self.patches.append(poly)
                # create Cell object for each polygon containing the relevant field data
                shply_poly = Polygon(poly.get_xy())

                # SCALE TE NE N0: FOR TESTING ONLY. MUST BE COMMENTED OUT!
                # te_scal=1.0
                # ne_scal=1.0
                # if self.zv[k][0] <= -1.6:
                #     n0_scal=1000.0
                # else:
                #     n0_scal = 1.0
                te_scal = 1.0
                ne_scal = 1.0
                n0_scal = 1.0

                self.cells.append(Cell(self.rmesh['data'][i], self.zmesh['data'][i],
                                       poly=shply_poly, te=te_scal*self.te[k],
                                       ne=ne_scal*self.ne[k], ni=ne_scal*self.ni[k],
                                       n0=n0_scal*self.n0[k], Srec=self.srec[k], Sion=self.sion[k],
                                       imp1_den = self.imp1_den[:,k], imp2_den=self.imp2_den[:,k]))
                k+=1

        ##############################################
        # GET STRIKE POINT COORDS AND SEPARATRIX POLY
        IDL_EprocRing= idl.export_function("EprocRing") # This function is now deprecated!
        rtmp = IDL_EprocRing(self.tranfile,"RMESH","S01",parallel=1)
        ztmp = IDL_EprocRing(self.tranfile,"ZMESH","S01",parallel=1)
        ztmp['ydata'] = -1.0*ztmp['ydata']
        self.osp = [rtmp['ydata'][0], ztmp['ydata'][0]]
        self.isp = [rtmp['ydata'][rtmp['npts']-1], ztmp['ydata'][ztmp['npts']-1]]

        # find start and end index
        istart = -1
        iend = -1
        for i in range(self.korpg['npts']):
            j = self.korpg['data'][i] - 1 # gotcha: convert from fortran indexing to idl/python
            j*=5
            if j >= 0:
                if (self.rvertp['data'][j+4] == rtmp['ydata'][0]) and (self.zvertp['data'][j+4] == ztmp['ydata'][0]):
                    istart = i
                if (self.rvertp['data'][j+4] == rtmp['ydata'][rtmp['npts']-1]) and (self.zvertp['data'][j+4] == ztmp['ydata'][ztmp['npts']-1]):
                    iend = i

        sep_points = []
        sep_points_below_xpt = []
        k = 0
        for i in range(istart, iend+1):
            j = self.korpg['data'][i] - 1
            j*=5
            sep_points.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            if self.zvertp['data'][j+0] <= self.geom['zpx']:
                sep_points_below_xpt.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            # sep_points.append((self.rvertp['data'][j+0],self.zvertp['data'][j+0]))
            k+=1
        sep_points_below_xpt.append((self.geom['rpx'], self.geom['zpx']))

        self.sep_poly = patches.Polygon(sep_points, closed=False, ec='w', linestyle='dashed', lw=2.0, fc='None', zorder=10)
        self.shply_sep_poly = Polygon(self.sep_poly.get_xy())
        self.sep_poly_below_xpt = patches.Polygon(sep_points_below_xpt, closed=False, ec='k', fc='None', zorder=10)
        self.shply_sep_poly_below_xpt = Polygon(self.sep_poly_below_xpt.get_xy())

        ##############################################
        # GET WALL POLYGON
        rtmp1 = IDL_EprocDataRead(self.tranfile, 'RVESM1')
        ztmp1 = IDL_EprocDataRead(self.tranfile, 'ZVESM1')
        rtmp2 = IDL_EprocDataRead(self.tranfile, 'RVESM2')
        ztmp2 = IDL_EprocDataRead(self.tranfile, 'ZVESM2')
        # z coord is upside down
        ztmp1['data'] = -1.0*ztmp1['data']
        ztmp2['data'] = -1.0*ztmp2['data']
        # remove non connected segments
        useSegment = np.zeros((rtmp1['npts']))
        nsegs = 0
        for i in range(rtmp1['npts']):
            check = 0
            if i != 0:
                if ((rtmp1['data'][i] == rtmp2['data'][i-1]) and (ztmp1['data'][i] == ztmp2['data'][i-1])) or \
                    ((rtmp2['data'][i] == rtmp1['data'][i-1]) and (ztmp2['data'][i] == ztmp1['data'][i-1])):
                    check = 1
            if i != (rtmp1['npts']):
                if ((rtmp1['data'][i] == rtmp2['data'][i+1]) and (ztmp1['data'][i] == ztmp2['data'][i+1])) or \
                    ((rtmp2['data'][i] == rtmp1['data'][i+1]) and (ztmp2['data'][i] == ztmp1['data'][i+1])):
                    check = 1
            if check:
                useSegment[i] = 1
                nsegs += 1

        wall_poly_pts = []
        for i in range(rtmp1['npts']):
            if useSegment[i]:
                wall_poly_pts.append((rtmp1['data'][i],ztmp1['data'][i]))
                wall_poly_pts.append((rtmp2['data'][i],ztmp2['data'][i]))
        wall_poly_pts.append(wall_poly_pts[0]) # connect last point to first to complete wall polygon

        self.wall_poly = patches.Polygon(wall_poly_pts, closed=False, ec='k', lw=2.0, fc='None', zorder=10)
        self.shply_wall_poly = Polygon(self.wall_poly.get_xy())


    def calc_H_emiss(self):
        # Te_rnge = [0.2, 5000]
        # ne_rnge = [1.0e11, 1.0e15]
        # PEC_dict = adas_adf15_utils.get_adas_H_PECs_interp(self.spec_line_dict, Te_rnge, ne_rnge, npts=self.ADAS_npts, npts_interp=1000)
        print('Calculating H emission...')
        for cell in self.cells:
            for line_key in self.spec_line_dict['1']['1']:
                E_excit, E_recom= adas_adf15_utils.get_H_line_emiss(line_key, self.ADAS_dict['adf15']['1']['1'], cell.te, cell.ne*1.0E-06, cell.ni*1.0E-06, cell.n0*1.0E-06)
                cell.H_emiss[line_key] = {'excit':E_excit, 'recom':E_recom, 'units':'ph.s^-1.m^-3.sr^-1'}

    def calc_ff_fb_emiss(self, filter_wv_nm, filter_tran):
        # TODO: ADD ZEFF CAPABILITY
        Te_rnge = [0.2, 5000]
        ne_rnge = [1.0e11, 1.0e15]

        wave_nm = np.linspace(filter_wv_nm[0], filter_wv_nm[-1], 10)

        print('Calculating FF+FB emission...')
        for cell in self.cells:
            ff_only, ff_fb_tot = continuum_utils.adas_continuo_py(wave_nm, cell.te, 1, 1)
            f = interp1d(wave_nm, ff_fb_tot)
            ff_fb_tot_interp = f(filter_wv_nm)
            # convert to spectral emissivity: ph s-1 m-3 sr-1 nm-1
            ff_fb_tot_interp = ff_fb_tot_interp * cell.ne * cell.ne
            # multiply by filter transmission
            ff_fb_tot_interp *= filter_tran
            # Integrate over wavelength
            ff_fb_tot_emiss = np.trapz(ff_fb_tot_interp, filter_wv_nm)
            cell.ff_fb_emiss = {'ff_fb':ff_fb_tot_emiss, 'units':'ph.s^-1.m^-3.sr^-1',
                                'filter_wv_nm':filter_wv_nm, 'filter_tran':filter_tran}


    def calc_H_rad_power(self):
        # Te_rnge = [0.2, 5000]
        # ne_rnge = [1.0e11, 1.0e15]
        # self.H_adf11 = adas_adf11_utils.get_adas_H_adf11_interp(Te_rnge, ne_rnge, npts=self.ADAS_npts, npts_interp=1000, pwr=True)
        print('Calculating H radiated power...')
        sum_pwr = 0
        for cell in self.cells:
            iTe, vTe = find_nearest(self.ADAS_dict['adf11']['1'].Te_arr, cell.te)
            ine, vne = find_nearest(self.ADAS_dict['adf11']['1'].ne_arr, cell.ne*1.0e-06)
            # plt/prb absolute rad pow contr in units W
            plt_contr = self.ADAS_dict['adf11']['1'].plt[iTe,ine]*(1.0e-06*cell.n0)*(1.0e-06*cell.ne)
            prb_contr = self.ADAS_dict['adf11']['1'].prb[iTe,ine]*(1.0e-06*cell.ni)*(1.0e-06*cell.ne)
            cell_vol = cell.poly.area * 2.0 * np.pi * cell.R # m^-3
            plt_contr = plt_contr * 1.e06 * cell_vol # Watts
            prb_contr = prb_contr * 1.e06 * cell_vol # Watts
            cell.H_radpwr = plt_contr+prb_contr

            sum_pwr += np.sum(np.asarray(cell.H_radpwr)) # sanity check. compare to eproc
        self.Prad_H = sum_pwr
        print('Total H radiated power:', sum_pwr, ' [W]')

    def calc_imp_emiss(self):
        # Te_rnge = [0.2, 5000]
        # ne_rnge = [1.0e11, 1.0e15]
        # PEC_dict = adas_adf15_utils.get_adas_imp_PECs_interp(self.spec_line_dict, Te_rnge, ne_rnge, npts=self.ADAS_npts, npts_interp=1000)
        # Also get adf11 for the ionisation balance fractional abundance. No Te_arr, ne_arr interpolation
        # available in the adf11 reader at the moment, so generate more coarse array (sloppy!)
        # TODO: add interpolation capability to the adf11 reader so that adf15 and adf11 are on the same Te, ne grid
        # Te_arr_adf11 = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), 500)
        # ne_arr_adf11 = np.logspace(np.log10(ne_rnge[0]), np.log10(ne_rnge[1]), 30)

        e2d_imps = self.zch['data'][0:2]
        for e2d_imp_idx, e2d_at_num in enumerate(e2d_imps):
            for req_at_num in self.spec_line_dict:
                if int(req_at_num) == int(e2d_at_num):
                    print('Calculating impurity (atomic num. =' , req_at_num, ') emission...')
                    # Get the adf11 data for the ionisation balance frac. abundances (adas_405 code)
                    # self.imp_adf11 = adas_adf11_utils.get_adas_imp_adf11(e2d_at_num, Te_arr_adf11, ne_arr_adf11)
                    for cell in self.cells:
                        cell.imp_emiss[req_at_num] = {}
                        for ion_stage in self.spec_line_dict[req_at_num]:
                            cell.imp_emiss[req_at_num][ion_stage] = {}
                            if e2d_imp_idx == 0: # impurity 1
                                imp_den_parent_stage = cell.imp1_den[int(ion_stage)]
                                imp_den_ion_stage = cell.imp1_den[int(ion_stage)-1]
                                ion_frac = cell.imp1_den[int(ion_stage)-1] / np.sum(cell.imp1_den)
                                ion_frac_parent = cell.imp1_den[int(ion_stage)] / np.sum(cell.imp1_den)
                            else: # impurity 2
                                imp_den_parent_stage = cell.imp2_den[int(ion_stage)]
                                imp_den_ion_stage = cell.imp2_den[int(ion_stage)-1]
                                ion_frac = cell.imp2_den[int(ion_stage)-1] / np.sum(cell.imp2_den)
                                ion_frac_parent = cell.imp2_den[int(ion_stage)] / np.sum(cell.imp2_den)
                            for line_key in self.spec_line_dict[req_at_num][ion_stage]:
                                idxTe_adf11, valTe_adf11 = find_nearest(self.ADAS_dict['adf11'][req_at_num].Te_arr, cell.te)
                                idxne_adf11, valne_adf11 = find_nearest(self.ADAS_dict['adf11'][req_at_num].ne_arr, cell.ne*1.0E-06)
                                ion_frac_ionbal = self.ADAS_dict['adf11'][req_at_num].ion_bal_frac['ion'][idxne_adf11,idxTe_adf11,int(ion_stage)-1]
                                ion_frac_parent_ionbal = self.ADAS_dict['adf11'][req_at_num].ion_bal_frac['ion'][idxne_adf11,idxTe_adf11,int(ion_stage)]

                                E_excit, E_recom= adas_adf15_utils.get_imp_line_emiss(line_key, self.ADAS_dict['adf15'][req_at_num][ion_stage], cell.te, cell.ne*1.0E-06, imp_den_parent_stage*1.0E-06, imp_den_ion_stage*1.0E-06)
                                idxTe, valTe = find_nearest(self.ADAS_dict['adf15'][req_at_num][ion_stage][line_key+'recom'].Te_arr, cell.te)
                                idxne, valne = find_nearest(self.ADAS_dict['adf15'][req_at_num][ion_stage][line_key+'recom'].ne_arr, cell.ne*1.0E-06)
                                PEC_recom =  self.ADAS_dict['adf15'][req_at_num][ion_stage][line_key+'recom'].pec[idxTe, idxne]
                                PEC_excit =  self.ADAS_dict['adf15'][req_at_num][ion_stage][line_key+'excit'].pec[idxTe, idxne]
                                cell.imp_emiss[req_at_num][ion_stage][line_key] = {'excit':E_excit, 'recom':E_recom, 'units':'ph.s^-1.m^-3.sr^-1',
                                                                                   'PEC_excit':PEC_excit, 'PEC_recom':PEC_recom,
                                                                                   'fPEC_excit':ion_frac*PEC_excit, 'fPEC_recom':ion_frac_parent*PEC_recom,
                                                                                   'fPEC_excit_ionbal':ion_frac_ionbal*PEC_excit, 'fPEC_recom_ionbal':ion_frac_parent_ionbal*PEC_recom}

    def calc_imp_rad_power(self):
        Te_rnge = [0.2, 5000]
        ne_rnge = [1.0e11, 1.0e15]
        Te_arr = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), 500)
        ne_arr = np.logspace(np.log10(ne_rnge[0]), np.log10(ne_rnge[1]), 30)

        e2d_imps = self.zch['data'][0:2]
        for e2d_imp_idx, e2d_at_num in enumerate(e2d_imps):
            if e2d_imp_idx == 0: # impurity 1
                # self.imp1_adf11 = adas_adf11_utils.get_adas_imp_adf11(e2d_at_num, Te_arr, ne_arr)
                print('Calculating impurity (atomic num. =' , e2d_at_num, ') power...')
                sum_pwr = 0
                for cell in self.cells:
                    iTe, vTe = find_nearest(Te_arr, cell.te)
                    ine, vne = find_nearest(ne_arr, cell.ne*1.0e-06)
                    for ion_stage in range(e2d_at_num):
                        # plt/prb absolute rad pow contr in units W
                        plt_contr = self.ADAS_dict['adf11'][str(e2d_at_num)].plt[iTe,ine,ion_stage]*(1.0e-06*cell.imp1_den[ion_stage])*(1.0e-06*cell.ne)
                        prb_contr = self.ADAS_dict['adf11'][str(e2d_at_num)].prb[iTe,ine,ion_stage]*(1.0e-06*cell.imp1_den[ion_stage+1])*(1.0e-06*cell.ne)
                        prc_contr = self.ADAS_dict['adf11'][str(e2d_at_num)].prc[iTe,ine,ion_stage]*(1.0e-06*cell.imp1_den[ion_stage+1])*(1.0e-06*cell.n0)
                        cell_vol = cell.poly.area * 2.0 * np.pi * cell.R # m^-3
                        plt_contr = plt_contr * 1.e06 * cell_vol # Watts
                        prb_contr = prb_contr * 1.e06 * cell_vol # Watts
                        prc_contr = prc_contr * 1.e06 * cell_vol # Watts
                        cell.imp1_radpwr.append(plt_contr+prb_contr+prc_contr)
                        # plt/prb contr to rad loss coefficient in units W.m^3
                        ion_frac = cell.imp1_den[ion_stage] / np.sum(cell.imp1_den)
                        ion_frac_parent = cell.imp1_den[ion_stage+1] / np.sum(cell.imp1_den)
                        cell.imp1_radpwr_coeff.append(1.0e-06*(self.ADAS_dict['adf11'][str(e2d_at_num)].plt[iTe,ine,ion_stage]*ion_frac+
                                                               self.ADAS_dict['adf11'][str(e2d_at_num)].prb[iTe,ine,ion_stage]*ion_frac_parent))
                    sum_pwr += np.sum(np.asarray(cell.imp1_radpwr)) # sanity check. compare to eproc
                print('Total power (atomic num. =' , e2d_at_num, '):', sum_pwr, ' [W]')
                self.Prad_imp1 = sum_pwr
            else: # impurity 2
                # self.imp2_adf11 = adas_adf11_utils.get_adas_imp_adf11(e2d_at_num, Te_arr, ne_arr)
                print('Calculating impurity (atomic num. =' , e2d_at_num, ') power...')
                sum_pwr = 0
                for cell in self.cells:
                    iTe, vTe = find_nearest(Te_arr, cell.te)
                    ine, vne = find_nearest(ne_arr, cell.ne*1.0e-06)
                    for ion_stage in range(e2d_at_num):
                        # plt/prb_contr in units W.cm^-3
                        plt_contr = self.ADAS_dict['adf11'][str(e2d_at_num)].plt[iTe,ine,ion_stage]*(1.0e-06*cell.imp2_den[ion_stage])*(1.0e-06*cell.ne)
                        prb_contr = self.ADAS_dict['adf11'][str(e2d_at_num)].prb[iTe,ine,ion_stage]*(1.0e-06*cell.imp2_den[ion_stage+1])*(1.0e-06*cell.ne)
                        prc_contr = self.ADAS_dict['adf11'][str(e2d_at_num)].prc[iTe,ine,ion_stage]*(1.0e-06*cell.imp2_den[ion_stage+1])*(1.0e-06*cell.n0)
                        cell_vol = cell.poly.area * 2.0 * np.pi * cell.R # m^3
                        plt_contr = plt_contr * 1.e06 * cell_vol # Watts
                        prb_contr = prb_contr * 1.e06 * cell_vol # Watts
                        prc_contr = prc_contr * 1.e06 * cell_vol # Watts
                        cell.imp2_radpwr.append(plt_contr+prb_contr+prc_contr)
                        # plt/prb contr to rad loss coefficient in units W.m^3
                        ion_frac = cell.imp2_den[ion_stage] / np.sum(cell.imp2_den)
                        ion_frac_parent = cell.imp2_den[ion_stage+1] / np.sum(cell.imp2_den)
                        cell.imp2_radpwr_coeff.append(1.0e-06*(self.ADAS_dict['adf11'][str(e2d_at_num)].plt[iTe,ine,ion_stage]*ion_frac+
                                                               self.ADAS_dict['adf11'][str(e2d_at_num)].prb[iTe,ine,ion_stage]*ion_frac_parent))
                    sum_pwr += np.sum(np.asarray(cell.imp2_radpwr)) # sanity check. compare to eproc
                print('Total power (atomic num. =' , e2d_at_num, '):', sum_pwr, ' [W]')
                self.Prad_imp2 = sum_pwr

    def los_intersect(self, los):
        for cell in self.cells:
            # check if cell lies within los.poly
            if los.los_poly.contains(cell.poly):
                los.cells.append(cell)
            # check if cell interstects with los.poly
            if los.los_poly.intersects(cell.poly):
                clipped_poly = los.los_poly.intersection(cell.poly)
                if clipped_poly.geom_type == 'Polygon':
                    centroid_p = clipped_poly.centroid
                    clipped_cell = Cell(centroid_p.x, centroid_p.y, poly=clipped_poly, te=cell.te, ne=cell.ne, ni=cell.ni, n0=cell.n0, Srec=cell.Srec, Sion=cell.Sion, imp1_den = cell.imp1_den, imp2_den=cell.imp2_den)
                    clipped_cell.H_emiss = cell.H_emiss
                    clipped_cell.imp_emiss = cell.imp_emiss
                    area_ratio = clipped_cell.poly.area /  cell.poly.area
                    clipped_cell.H_radpwr = cell.H_radpwr * area_ratio
                    clipped_cell.imp1_radpwr = np.asarray(cell.imp1_radpwr) * area_ratio
                    clipped_cell.imp2_radpwr = np.asarray(cell.imp2_radpwr) * area_ratio
                    los.cells.append(clipped_cell)

        # Intersection of los centerline with separatrix: returns points where los crosses sep
        # generate los centerline shapely object
        los.shply_cenline = LineString([(los.p1),(los.p2)])
        if los.los_poly.intersects(self.shply_sep_poly_below_xpt):
            los.shply_intersects_w_sep = los.shply_cenline.intersection(self.shply_sep_poly_below_xpt)

    def ion_balance_OT(self):
        # RING-WISE IONISATION VS TARGET FLUX COMPARISON

        IDL_EprocRing= idl.export_function("EprocRing") # This function is now depracated!
        IDL_EprocRow= idl.export_function("EprocRow") # This function is now depracated!

        nrings = 20
        for iring in range(0,nrings):
            labelarg = 'S' + str(iring)
            pflxd = IDL_EprocRing(self.tranfile, 'PFLXD', labelarg, PARALLEL=0)
            pflxd['ydata']*=-1.0
            dv = IDL_EprocRing(self.tranfile, 'DV', labelarg, PARALLEL=0)
            soun = IDL_EprocRing(self.tranfile, 'SOUN', labelarg, PARALLEL=0)
            rmesh = IDL_EprocRing(self.tranfile, 'RMESH', labelarg, PARALLEL=0)
            soun_dv = soun['ydata']*dv['ydata']
            sum_soun_dv = np.sum(np.abs(soun_dv[0:40]))
            plt.plot(rmesh['ydata'][0], sum_soun_dv, 'ob')
            plt.plot(rmesh['ydata'][0],pflxd['ydata'][0], 'or')

        pflxd_OT = IDL_EprocRow(self.tranfile, 'PFLXD', 'OT', ALL_POINTS=0)
        plt.plot(pflxd_OT['xdata']+self.osp[0], -1.0*pflxd_OT['ydata'], '-ok')
        plt.show()

    def calc_qpol_div(self):
        # LFS
        endidx = self.qpartot_LFS['npts']
        xidx, = np.where(self.qpartot_LFS['xdata'][0:endidx] > 0.0)
        # CONVERT QPAR TO QPOL (QPOL = QPAR * Btheta/Btot)
        qpol_LFS = self.qpartot_LFS['ydata'][xidx] * self.bpol_btot_LFS['ydata'][xidx]
        # calculate DR from neighbours
        dR_LFS = np.zeros((len(xidx)))
        for idx, val in enumerate(xidx):
            left_neighb = np.sqrt(((self.qpartot_LFS_rmesh['ydata'][val]-self.qpartot_LFS_rmesh['ydata'][val-1])**2)+
                                  ((self.qpartot_LFS_zmesh['ydata'][val]-self.qpartot_LFS_zmesh['ydata'][val-1])**2))
            if val != xidx[-1]:
                right_neighb = np.sqrt(((self.qpartot_LFS_rmesh['ydata'][val+1]-self.qpartot_LFS_rmesh['ydata'][val])**2)+
                                       ((self.qpartot_LFS_zmesh['ydata'][val+1]-self.qpartot_LFS_zmesh['ydata'][val])**2))
            else:
                right_neighb = left_neighb
            dR_LFS[idx] = (left_neighb+right_neighb)/2.0
        area = 2. * np.pi * self.qpartot_LFS_rmesh['ydata'][xidx] * dR_LFS
        self.qpol_div_LFS = np.sum(qpol_LFS*area)

        # HFS
        endidx = self.qpartot_HFS['npts']
        xidx, = np.where(self.qpartot_HFS['xdata'][0:endidx] > 0.0)
        # CONVERT QPAR TO QPOL (QPOL = QPAR * Btheta/Btot)
        qpol_HFS = self.qpartot_HFS['ydata'][xidx] * self.bpol_btot_HFS['ydata'][xidx]
        # calculate DR from neighbours
        dR_HFS = np.zeros((len(xidx)))
        for idx, val in enumerate(xidx):
            left_neighb = np.sqrt(((self.qpartot_HFS_rmesh['ydata'][val]-self.qpartot_HFS_rmesh['ydata'][val-1])**2)+
                                  ((self.qpartot_HFS_zmesh['ydata'][val]-self.qpartot_HFS_zmesh['ydata'][val-1])**2))
            if val != xidx[-1]:
                right_neighb = np.sqrt(((self.qpartot_HFS_rmesh['ydata'][val+1]-self.qpartot_HFS_rmesh['ydata'][val])**2)+
                                       ((self.qpartot_HFS_zmesh['ydata'][val+1]-self.qpartot_HFS_zmesh['ydata'][val])**2))
            else:
                right_neighb = left_neighb
            dR_HFS[idx] = (left_neighb+right_neighb)/2.0
        area = 2. * np.pi * self.qpartot_HFS_rmesh['ydata'][xidx] * dR_HFS
        self.qpol_div_HFS = np.sum(qpol_HFS*area)

        print('Pdiv_LFS (MW): ', self.qpol_div_LFS*1.e-06, 'Pdiv_HFS (MW): ', self.qpol_div_HFS*1.e-06, 'POWSOL (MW): ', self.powsol['data'][0]*1.e-06)




    def calc_BORP(self):
        self.BORP = {}

        # H/D power
        PR4_H = [] # HFS-SOL
        PR5_H = [] # LFS-SOL
        PR6_H = [] # HFS-DIV
        PR7_H = [] # LFS-DIV
        PR8_H = [] # Xpt-ConReg
        PR11_H = [] # HFS-Lower
        PR12_H = [] # LFS-Lower
        PR14_H = [] # rhoN=0.9-1.0

        # imp1 power
        PR4_imp1 = [] # HFS-SOL
        PR5_imp1 = [] # LFS-SOL
        PR6_imp1 = [] # HFS-DIV
        PR7_imp1 = [] # LFS-DIV
        PR8_imp1 = [] # Xpt-ConReg
        PR11_imp1 = [] # HFS-Lower
        PR12_imp1 = [] # LFS-Lower
        PR14_imp1 = [] # rhoN=0.9-1.0
        
        # imp2 power
        PR4_imp2 = [] # HFS-SOL
        PR5_imp2 = [] # LFS-SOL
        PR6_imp2 = [] # HFS-DIV
        PR7_imp2 = [] # LFS-DIV
        PR8_imp2 = [] # Xpt-ConReg
        PR11_imp2 = [] # HFS-Lower
        PR12_imp2 = [] # LFS-Lower
        PR14_imp2 = [] # rhoN=0.9-1.0        

        Zdiv = -1.2

        for cell in self.cells:
            # HFS-SOL
            if not self.shply_sep_poly.contains(cell.poly) and cell.Z >= Zdiv and cell.R <= self.geom['rpx']:
                PR4_H.append(cell.H_radpwr)
                if self.imp1_atom_num:
                    PR4_imp1.append(np.sum(cell.imp1_radpwr))
                if self.imp2_atom_num:
                    PR4_imp2.append(np.sum(cell.imp2_radpwr))                    
            # LFS-SOL
            if not self.shply_sep_poly.contains(cell.poly) and cell.Z >= Zdiv and cell.R >= self.geom['rpx']:
                PR5_H.append(cell.H_radpwr)
                if self.imp1_atom_num:
                    PR5_imp1.append(np.sum(cell.imp1_radpwr))
                if self.imp2_atom_num:
                    PR5_imp2.append(np.sum(cell.imp2_radpwr)) 
            # HFS-DIV
            if not self.shply_sep_poly.contains(cell.poly) and cell.Z <= Zdiv and cell.R <= self.geom['rpx']:
                PR6_H.append(cell.H_radpwr)
                if self.imp1_atom_num:
                    PR6_imp1.append(np.sum(cell.imp1_radpwr))
                if self.imp2_atom_num:
                    PR6_imp2.append(np.sum(cell.imp2_radpwr)) 
            # LFS-DIV
            if not self.shply_sep_poly.contains(cell.poly) and cell.Z <= Zdiv and cell.R >= self.geom['rpx']:
                PR7_H.append(cell.H_radpwr)
                if self.imp1_atom_num:
                    PR7_imp1.append(np.sum(cell.imp1_radpwr))
                if self.imp2_atom_num:
                    PR7_imp2.append(np.sum(cell.imp2_radpwr))
            # Xpt-ConReg
            # TODO: Xpt-ConReg define region using edge2d rows/rings
            if self.shply_sep_poly.contains(cell.poly) and cell.Z >= self.geom['zpx'] and cell.Z <= -0.80 and cell.R >= 2.3 and cell.R <= 2.95:
                PR8_H.append(cell.H_radpwr)
                if self.imp1_atom_num:
                    PR8_imp1.append(np.sum(cell.imp1_radpwr))
                if self.imp2_atom_num:
                    PR8_imp2.append(np.sum(cell.imp2_radpwr))
            # HFS-Lower
            if cell.Z <= Zdiv and cell.R <= self.geom['rpx']:
                PR11_H.append(cell.H_radpwr)
                if self.imp1_atom_num:
                    PR11_imp1.append(np.sum(cell.imp1_radpwr))
                if self.imp2_atom_num:
                    PR11_imp2.append(np.sum(cell.imp2_radpwr)) 
            # LFS-Lower
            if cell.Z <= Zdiv and cell.R >= self.geom['rpx']:
                PR12_H.append(cell.H_radpwr)
                if self.imp1_atom_num:
                    PR12_imp1.append(np.sum(cell.imp1_radpwr))
                if self.imp2_atom_num:
                    PR12_imp2.append(np.sum(cell.imp2_radpwr)) 
            # rhoN=0.9-1.0
            if self.shply_sep_poly.contains(cell.poly) and cell.Z >= self.geom['zpx']:
                PR14_H.append(cell.H_radpwr)
                if self.imp1_atom_num:
                    PR14_imp1.append(np.sum(cell.imp1_radpwr))
                if self.imp2_atom_num:
                    PR14_imp2.append(np.sum(cell.imp2_radpwr)) 

        # Sum up and store
        self.BORP['PR4'] = {'region':'HFS-SOL', 'H':np.sum(PR4_H), 'imp1':np.sum(PR4_imp1), 'imp2':np.sum(PR4_imp2), 'units':'W'}
        self.BORP['PR5'] = {'region':'LFS-SOL', 'H':np.sum(PR5_H), 'imp1':np.sum(PR5_imp1), 'imp2':np.sum(PR5_imp2), 'units':'W'}
        self.BORP['PR6'] = {'region':'HFS-DIV', 'H':np.sum(PR6_H), 'imp1':np.sum(PR6_imp1), 'imp2':np.sum(PR6_imp2), 'units':'W'}
        self.BORP['PR7'] = {'region':'LFS-DIV', 'H':np.sum(PR7_H), 'imp1':np.sum(PR7_imp1), 'imp2':np.sum(PR7_imp2), 'units':'W'}
        self.BORP['PR8'] = {'region':'Xpt-ConReg', 'H':np.sum(PR8_H), 'imp1':np.sum(PR8_imp1), 'imp2':np.sum(PR8_imp2), 'units':'W'}
        self.BORP['PR11'] = {'region':'HFS-Lower', 'H':np.sum(PR11_H), 'imp1':np.sum(PR11_imp1), 'imp2':np.sum(PR11_imp2), 'units':'W'}
        self.BORP['PR12'] = {'region':'LFS-Lower', 'H':np.sum(PR12_H), 'imp1':np.sum(PR12_imp1), 'imp2':np.sum(PR12_imp2), 'units':'W'}
        self.BORP['PR14'] = {'region':'rhoN=0.9-1.0', 'H':np.sum(PR14_H), 'imp1':np.sum(PR14_imp1), 'imp2':np.sum(PR14_imp2), 'units':'W'}


    def plot_regions(self):
        fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True)
        region_patches = []
        Zdiv=-1.2
        for cell in self.cells:
            # check if cell lies within los.poly
            if self.shply_sep_poly.contains(cell.poly) and cell.Z >= self.geom['zpx'] and cell.Z <= -0.80 and cell.R >= 2.3 and cell.R <= 2.95:
            # if not self.shply_sep_poly.contains(cell.poly) and cell.Z >= -1.2 and cell.R >= self.geom['rpx']:
                region_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=False))
        # region_patches.append(patches.Polygon(self.shply_sep_poly.exterior.coords, closed=False))
        coll = PatchCollection(region_patches)
        ax.add_collection(coll)
        ax.set_xlim(1.8, 4.0)
        ax.set_ylim(-2.0, 2.0)


    def plot_te_ne_n0(self):
        fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True)
        cmap = plt.get_cmap('jet')
        coll1 = PatchCollection(self.patches)
        colors = plt.cm.jet(self.te/(np.max(self.te)/10.))
        coll1.set_color(colors)
        coll2 = PatchCollection(self.patches)
        colors = plt.cm.jet(self.ne/np.max(self.ne))
        coll2.set_color(colors)
        coll3 = PatchCollection(self.patches)
        colors = plt.cm.jet(self.n0/np.max(self.n0))
        coll3.set_color(colors)
        ax[0].add_collection(coll1)
        ax[1].add_collection(coll2)
        ax[2].add_collection(coll3)

        ax[0].set_xlim(1.8, 4.0)
        ax[0].set_ylim(-2.0, 2.0)

        # sanity check - plot grid from list of cell objects
        # fig2, ax2 = plt.subplots(ncols=1)
        # for cell in self.cells:
        #     r,z = cell.poly.exterior.xy
        #     ax2.plot(r,z, '-k')
        #     ax2.plot(cell.R, cell.Z, 'ob')
        # plt.show()

    def plot_H_lines(self):

        # PLOT H-ALPHA, BETA, GAMMA, EPSILON
        fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True)
        ax[0].set_xlim(2.36, 2.96)
        ax[0].set_ylim(-1.73, -1.29)
        for H_line_key in self.spec_line_dict['1']['1'].keys():
            if self.spec_line_dict['1']['1'][H_line_key][0] == '3' and self.spec_line_dict['1']['1'][H_line_key][1] == '2':
                cell_patches=[]
                H_line=[]
                for cell in self.cells:
                    cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
                    H_line.append(cell.H_emiss[H_line_key]['excit'] + cell.H_emiss[H_line_key]['recom'])

                coll1 = PatchCollection(cell_patches, zorder=1)
                colors = plt.cm.jet(H_line/np.max(H_line))
                coll1.set_color(colors)
                ax[0].add_collection(coll1)
                ax[0].set_yscale
                ax[0].set_title(H_line_key)
            if self.spec_line_dict['1']['1'][H_line_key][0] == '4' and self.spec_line_dict['1']['1'][H_line_key][1] == '2':
                cell_patches=[]
                H_line=[]
                for cell in self.cells:
                    cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
                    H_line.append(cell.H_emiss[H_line_key]['excit'] + cell.H_emiss[H_line_key]['recom'])

                coll1 = PatchCollection(cell_patches, zorder=1)
                colors = plt.cm.jet(H_line/np.max(H_line))
                coll1.set_color(colors)
                ax[1].add_collection(coll1)
                ax[1].set_yscale
                ax[1].set_title(H_line_key)
            if self.spec_line_dict['1']['1'][H_line_key][0] == '5' and self.spec_line_dict['1']['1'][H_line_key][1] == '2':
                cell_patches=[]
                H_line=[]
                for cell in self.cells:
                    cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
                    H_line.append(cell.H_emiss[H_line_key]['excit'] + cell.H_emiss[H_line_key]['recom'])

                coll1 = PatchCollection(cell_patches, zorder=1)
                colors = plt.cm.jet(H_line/np.max(H_line))
                coll1.set_color(colors)
                ax[2].add_collection(coll1)
                ax[2].set_yscale
                ax[2].set_title(H_line_key)
            if self.spec_line_dict['1']['1'][H_line_key][0] == '7' and self.spec_line_dict['1']['1'][H_line_key][1] == '2':
                cell_patches=[]
                H_line=[]
                for cell in self.cells:
                    cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
                    H_line.append(cell.H_emiss[H_line_key]['excit'] + cell.H_emiss[H_line_key]['recom'])

                coll1 = PatchCollection(cell_patches, zorder=1)
                colors = plt.cm.jet(H_line/(0.5*np.max(H_line)))
                coll1.set_color(colors)
                ax[3].add_collection(coll1)
                ax[3].set_yscale
                ax[3].set_title(H_line_key)

        # PLOT SEPARATRIX
        ax[0].add_patch(self.sep_poly)
        ax[1].add_patch(self.sep_poly)
        ax[2].add_patch(self.sep_poly)
        ax[3].add_patch(self.sep_poly)

        # PLOT WALL
        ax[0].add_patch(self.wall_poly)
        ax[1].add_patch(self.wall_poly)
        ax[2].add_patch(self.wall_poly)
        ax[3].add_patch(self.wall_poly)

        # # PLOT ENTIRE VESSEL
        # RStart = 1.6
        # REnd = 4.0
        # ZStart = 2.05
        # ZEnd = -1.85
        # PlotWall(RStart, REnd, ZStart, ZEnd, ax[2])

        ax[0].set_xlim(2.36, 2.96)
        ax[0].set_ylim(-1.73, -1.29)

    def plot_ff_fb_filtered_emiss(self, diag_list=[], plot_LOS=False, Rrng=None, Zrng=None, savefig=False):

        fig, ax = plt.subplots(ncols=1, figsize=(10,8))
        fig.patch.set_facecolor('white')
        if Rrng != None and Zrng != None:
            ax.set_xlim(Rrng[0], Rrng[1])
            ax.set_ylim(Zrng[0], Zrng[1])
        else:
            ax.set_xlim(1.8, 4.0)
            ax.set_ylim(-2.0, 2.0)

        cell_patches=[]
        ff_fb_emiss=[]
        for cell in self.cells:
            cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
            ff_fb_emiss.append(cell.ff_fb_emiss['ff_fb'])

        coll1 = PatchCollection(cell_patches, zorder=1)
        colors = plt.cm.hot(ff_fb_emiss/(1.5*np.max(ff_fb_emiss)))

        coll1.set_color(colors)
        collplt = ax.add_collection(coll1)
        # collplt.set_array(np.array(colors[:,0]))
        ax.set_yscale

        # Get cwl of the bandpass filter
        title = 'Bremss. 400.96 nm'
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
        label =  '$\mathrm{ph\/s^{-1}\/m^{-3}\/sr^{-1}}$'
        cbar.set_label(label)

        # ADD DIAG LOS
        if self.synth_diag and plot_LOS:
            for diag_key in self.synth_diag:
                for req_diag in diag_list:
                    if req_diag == diag_key:
                        self.synth_diag[diag_key].plot_LOS(ax, color='w', lw=1.0, Rrng=[2.55, 2.82])

        # PLOT SEPARATRIX AND WALL
        ax.add_patch(self.sep_poly)
        ax.add_patch(self.wall_poly)

        if savefig:
            plt.savefig(title + '.png', dpi=plt.gcf().dpi)

    def plot_imp_lines(self, at_num, ion_stage, line_key, diag_list=[], plot_LOS=False, Rrng=None, Zrng=None, savefig=False):

        # PLOT nitrogen lines
        # TODO: need to make this more general

        if self.spec_line_dict[at_num][ion_stage][line_key]:
            fig, ax = plt.subplots(ncols=1, figsize=(10,8))
            fig.patch.set_facecolor('white')
            if Rrng != None and Zrng != None:
                ax.set_xlim(Rrng[0], Rrng[1])
                ax.set_ylim(Zrng[0], Zrng[1])
            else:
                ax.set_xlim(1.8, 4.0)
                ax.set_ylim(-2.0, 2.0)

            cell_patches=[]
            imp_line=[]
            for cell in self.cells:
                cell_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True, zorder=1))
                imp_line.append(cell.imp_emiss[at_num][ion_stage][line_key]['excit'] + cell.imp_emiss[at_num][ion_stage][line_key]['recom'])
                # imp_line.append((cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_excit']+cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_recom'])*cell.ne)

            # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.hot, norm=matplotlib.colors.LogNorm(), zorder=1, lw=0)
            # coll1 = PatchCollection(cell_patches, cmap=matplotlib.cm.hot, zorder=1, lw=0)
            coll1 = PatchCollection(cell_patches, zorder=1)
            # coll1.set_array(np.asarray(imp_line))
            colors = plt.cm.hot(imp_line/np.max(imp_line))

            coll1.set_color(colors)
            collplt = ax.add_collection(coll1)
            # collplt.set_array(np.array(colors[:,0]))
            ax.set_yscale
            line_wv = float(line_key) / 10.
            at_sym[int(at_num)-1]
            title = at_sym[int(at_num)-1] + ' ' + roman[int(ion_stage)-1] + ' ' + '{:5.1f}'.format(line_wv) + ' nm'
            ax.set_title(title)
            plt.gca().set_aspect('equal', adjustable='box')

            # ADD COLORBAR
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cbar_ax = divider.append_axes('right', size='7%', pad=0.1)

            # Very ugly workaround to scale the colorbar without clobbering the patch collection plot
            # (https://medium.com/data-science-canvas/way-to-show-colorbar-without-calling-imshow-or-scatter)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(vmin=0, vmax=np.max(imp_line)))
            sm._A = []

            cbar = fig.colorbar(sm, cax=cbar_ax)
            label =  '$\mathrm{ph\/s^{-1}\/m^{-3}\/sr^{-1}}$'
            cbar.set_label(label)

            # ADD DIAG LOS
            if self.synth_diag and plot_LOS:
                for diag_key in self.synth_diag:
                    for req_diag in diag_list:
                        if req_diag == diag_key:
                            self.synth_diag[diag_key].plot_LOS(ax, color='w', lw=1.0, Rrng=[2.55, 2.82])

            # PLOT SEPARATRIX AND WALL
            ax.add_patch(self.sep_poly)
            ax.add_patch(self.wall_poly)

            if savefig:
                plt.savefig(title + '.png', dpi=plt.gcf().dpi)
        else:
            print(line_key, ' not found in post-processed impurity emisison.')

    def plot_imp_rad_loss(self, ax1_5by2, ax2_4by1, barspace=0.2, color='b'):
        # rad loss coeff not very sensitive to elec. density so choose a sensible value
        ine, vne = find_nearest(self.imp2_adf11.ne_arr, 1.0e14)

        # Set up elec. temp bins and labels
        te_bins = [0.5, 2.0, 5.0, 10., 20., 50., 100., 500., 1000.]
        te_bin_labels = []
        for ite, vte in enumerate(te_bins):
            if (ite+1) != len(te_bins):
                label = '{:6.1f}'.format(te_bins[ite]) + '-' + '{:6.1f}'.format(te_bins[ite+1])
                te_bin_labels.append(label)

        # IMPURITY RADIATED POWER
        e2d_imps = self.zch['data'][0:2]
        for e2d_imp_idx, e2d_at_num in enumerate(e2d_imps):

            if e2d_imp_idx == 0: # impurity 1

                # BIN RADIATE POWER BY MACRO REGION, TE, CHARGE STATE
                div_bin_imp1 = np.zeros((e2d_at_num))
                div_bin_lfs_imp1 = np.zeros((e2d_at_num))
                main_bin_imp1 = np.zeros((e2d_at_num))
                for cell in self.cells:
                    if cell.Z <= -1.2:
                        div_bin_imp1+=cell.imp1_radpwr
                    if cell.Z >= -1.2:
                        main_bin_imp1+=cell.imp1_radpwr
                    # LFS-DIV
                    if not self.shply_sep_poly.contains(cell.poly) and cell.Z <= -1.2 and cell.R >= self.geom['rpx']:
                        div_bin_lfs_imp1+=cell.imp1_radpwr
                # conerty to MW
                div_bin_imp1*=1.0e-06
                main_bin_imp1*=1.0e-06
                div_bin_lfs_imp1*=1.0e-06

            elif e2d_imp_idx == 1: # impurity 2

                # plot ionisation balance radiative loss coeff
                ax1_5by2[0].loglog(self.imp2_adf11.Te_arr, 1.0e-06*self.imp2_adf11.ion_bal_pwr['total'][ine,:], '-k', lw=3.0)
                c = ['b', 'g', 'y', 'm', 'r']
                for i in range(e2d_at_num-4):
                    ax1_5by2[0].loglog(self.imp2_adf11.Te_arr, 1.0e-06*self.imp2_adf11.ion_bal_pwr['ion'][ine,:,i], ':', c='k', lw=1.0)
                    ax1_5by2[i+1].loglog(self.imp2_adf11.Te_arr, 1.0e-06*self.imp2_adf11.ion_bal_pwr['ion'][ine,:,i], '-', c='k', lw=2.0)

                # plot sim rad loss coeff/pwr for each stage
                imp2_radpwr_coeff_collate = []
                imp2_radpwr_collate = []
                te_collate = []
                for cell in self.cells:
                    # LFS DIV below Z=-1.2
                    if not self.shply_sep_poly.contains(cell.poly) and cell.Z <= -1.2 and cell.R >= self.geom['rpx']:
                        imp2_radpwr_coeff_collate.append(cell.imp2_radpwr_coeff)
                        imp2_radpwr_collate.append(cell.imp2_radpwr)
                        te_collate.append(cell.te)
                imp2_radpwr_coeff_collate_arr = np.asarray(imp2_radpwr_coeff_collate)
                imp2_radpwr_collate_arr = np.asarray(imp2_radpwr_collate)
                te_collate_arr = np.asarray(te_collate)
                ax1_5by2[0].loglog(te_collate_arr, np.sum(imp2_radpwr_coeff_collate_arr, axis=1), '.', c=color, ms=4, mew=1.0)
                # ax1_5by2[0].loglog(te_collate_arr, np.sum(imp2_radpwr_collate_arr, axis=1), '.', c='k', ms=4, mew=2.0)
                for i in range(e2d_at_num-4):
                    ax1_5by2[i+1].loglog(te_collate_arr, imp2_radpwr_coeff_collate_arr[:,i], '.', c=color, ms=4, mew=1.0)
                    # ax1_5by2[i+1].loglog(te_collate_arr, imp2_radpwr_collate_arr[:,i], '.', c=c[i], ms=4, mew=2.0)
                r'$\mathrm{P_{rad,N}\/(W m^{-3}})$'
                ax1_5by2[0].set_ylabel(r'$\mathrm{P_{rad,N}\/(W m^{3}})$')
                ax1_5by2[1].set_ylabel(r'$\mathrm{P_{rad,N0+}\/(W m^{3}})$')
                ax1_5by2[2].set_ylabel(r'$\mathrm{P_{rad,N1+}\/(W m^{3}})$')
                ax1_5by2[3].set_ylabel(r'$\mathrm{P_{rad,N2+}\/(W m^{3}})$')
                # ax1_5by2[4].set_ylabel(r'$\mathrm{P_{rad,N3+}\/(W m^{3}})$')
                # ax1_5by2[5].set_ylabel(r'$\mathrm{P_{rad,N4+}\/(W m^{3}})$')
                ax1_5by2[3].set_xlabel('Te (eV)')
                
                ax1_5by2[0].set_ylim(1.1e-36, 0.9e-31)
                ax1_5by2[1].set_ylim(1.1e-36, 0.9e-31)
                ax1_5by2[2].set_ylim(1.1e-36, 0.9e-31)
                ax1_5by2[3].set_ylim(1.1e-36, 0.9e-31)
                # ax1_5by2[4].set_ylim(1e-36, 1e-31)
                # ax1_5by2[5].set_ylim(1e-36, 1e-31)

                ax1_5by2[0].set_xlim(1, 100)
                ax1_5by2[1].set_xlim(1, 100)
                ax1_5by2[2].set_xlim(1, 100)
                ax1_5by2[3].set_xlim(1, 100)
                # ax1_5by2[4].set_xlim(1, 1000)
                # ax1_5by2[5].set_xlim(1, 1000)
                
                # BIN RADIATE POWER BY MACRO REGION AND TE
                div_bin_imp2 = np.zeros((e2d_at_num))
                main_bin_imp2 = np.zeros((e2d_at_num))
                div_bin_lfs_imp2 = np.zeros((e2d_at_num))
                te_bin_imp2_radpwr = np.zeros((len(te_bins)-1))
                for cell in self.cells:
                    for ite, vte in enumerate(te_bins):
                        if (ite+1) != len(te_bins):
                            if cell.te > te_bins[ite] and cell.te <= te_bins[ite+1]:
                                te_bin_imp2_radpwr[ite]+=np.sum(cell.imp2_radpwr)
                    if cell.Z <= -1.2:
                        div_bin_imp2+=cell.imp2_radpwr
                    if cell.Z >= -1.2:
                        main_bin_imp2+=cell.imp2_radpwr
                    # LFS-DIV
                    if not self.shply_sep_poly.contains(cell.poly) and cell.Z <= -1.2 and cell.R >= self.geom['rpx']:
                        div_bin_lfs_imp2+=cell.imp2_radpwr
                # conerty to MW
                div_bin_imp2*=1.0e-06
                main_bin_imp2*=1.0e-06
                div_bin_lfs_imp2*=1.0e-06
                te_bin_imp2_radpwr*=1.0e-06

        # ALSO BIN H RAD POWER TO CONTRAST AGAINST IMPURITY RAD POWER
        div_bin_H = 0.0
        main_bin_H = 0.0
        te_bin_H_radpwr = np.zeros((len(te_bins)-1))
        for cell in self.cells:
            for ite, vte in enumerate(te_bins):
                if (ite+1) != len(te_bins):
                    if cell.te > te_bins[ite] and cell.te <= te_bins[ite+1]:
                        te_bin_H_radpwr[ite]+=cell.H_radpwr
            if cell.Z <= -1.2:
                div_bin_H+=cell.H_radpwr
            else:
                main_bin_H+=cell.H_radpwr
        # conerty to MW
        div_bin_H*=1.0e-06
        main_bin_H*=1.0e-06
        te_bin_H_radpwr*=1.0e-06

        # IMP2 CHARGE STATE DIST
        # ax2_4by1[0].plot(div_bin_imp2, '-o', c=color, mfc=color, mec=color, ms=4, mew=2.0)
        ax2_4by1[0].plot(div_bin_lfs_imp2, '-o', c=color, mfc=color, mec=color, ms=4, mew=2.0)
        ax2_4by1[1].plot(main_bin_imp2, '-o', c=color, mfc=color, mec=color, ms=4, mew=2.0)
        ax2_4by1[0].set_ylim(0,0.1)
        ax2_4by1[1].set_ylim(0,0.2)
        ax2_4by1[0].set_ylabel(r'$\mathrm{P_{RAD,N}\/(MW)}$')
        ax2_4by1[0].set_xlabel('Ionisation stage')
        ax2_4by1[0].set_ylabel(r'$\mathrm{P_{RAD,N}\/(MW)}$')
        ax2_4by1[1].set_xlabel('Ionisation stage')
        # BAR PLOT
        xlabel = ('Divertor', 'Main')
        x_pos = np.arange(len(xlabel))
        width=0.2
        # BAR PLOT BY MACRO REGION
        H_bar = np.array((div_bin_H, main_bin_H))
        imp1_bar = np.array((np.sum(div_bin_imp1), np.sum(main_bin_imp1)))
        imp2_bar = np.array((np.sum(div_bin_imp2), np.sum(main_bin_imp2)))
        ax2_4by1[2].bar(x_pos+barspace, H_bar, width, align='center', color='darkgrey')
        # ax2_4by1[2].bar(x_pos, imp1_bar, width, bottom=H_bar, align='center', color='blue')
        ax2_4by1[2].bar(x_pos+barspace, imp2_bar, width, bottom=H_bar, align='center', color=color)
        ax2_4by1[2].set_xticks(x_pos+2*width)
        ax2_4by1[2].set_xticklabels(xlabel)
        ax2_4by1[2].set_ylim(0,1.3)
        ax2_4by1[2].set_ylabel('(MW)')

        # BAR PLOT BY TE BINS
        x_pos = np.arange(len(te_bin_labels))
        # ax2_4by1[3].bar(x_pos, te_bin_H_radpwr/ (np.sum(te_bin_H_radpwr)+np.sum(te_bin_imp2_radpwr)), width, align='center', color='darkgrey')
        ax2_4by1[3].bar(x_pos+barspace, te_bin_imp2_radpwr/np.sum(te_bin_imp2_radpwr), width, align='center', color=color)
        ax2_4by1[3].set_xticks(x_pos+2*width)
        ax2_4by1[3].set_xticklabels(te_bin_labels, rotation=90)
        ax2_4by1[3].set_ylim(0,0.45)
        ax2_4by1[3].set_xlabel('T_e (eV)')
        ax2_4by1[3].set_ylabel('N power frac.')


class LOS:

    def __init__(self, los_poly = None, chord_num=None, p1 = None, w1 = None, p2orig = None, p2 = None,
                 w2orig = None, w2 = None, l12 = None, theta=None, spec_line_dict=None,
                 imp1_atom_num=None, imp2_atom_num=None, calc_NII_afg_feature=False):

        self.calc_NII_afg_feature = calc_NII_afg_feature

        self.los_poly = los_poly
        self.chord_num = chord_num
        self.p1 = p1
        self.w1 = w1
        self.p2unmod = p2orig
        self.p2 = p2
        self.w2unmod = w2orig
        self.w2 = w2
        self.l12 = l12
        self.theta = theta
        self.cells = []
        self.spec_line_dict = spec_line_dict
        self.imp1_atom_num = imp1_atom_num
        self.imp2_atom_num = imp2_atom_num

        # center line shapely object
        self.shply_cenline = None
        # intersection with separatrix below x-point (for identifying PFR-SOL regions)
        self.shply_intersects_w_sep = None

        # LINE INTEGRATED QUANTITIES - DICT
        # {'param':val, 'units':''}
        self.los_int = {}

        # 1D QUANTITIES ALONG LOS - DICT
        # {'param':val, 'units':''}
        self.los_1d = {}

        # SYNTHETIC SPECTRA - DICT
        # e.g. features: 'stark', 'ff_fb_continuum'
        # {'feature':{'wave':1d_arr, 'intensity':1d_arr, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}}
        self.los_int_spectra = {}
        # {'feature':{'wave':2d_arr[dl_idx,arr], 'intensity':2d_arr[dl_idx,arr], 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}}
        self.los_1d_spectra = {}
        # Stark broadening coefficients (Lomanowski et al 2015, NF)
        self.lorentz_5_2_coeffs = {}
        if self.spec_line_dict['1']['1']:
            for key in self.spec_line_dict['1']['1']:
                # H 6-2
                if self.spec_line_dict['1']['1'][key][1] == '2' and self.spec_line_dict['1']['1'][key][0] == '6':
                    self.lorentz_5_2_coeffs[key] = {'C':3.954E-16, 'a':0.7149, 'b':0.028}


    def orthogonal_polys(self):
        for cell in self.cells:
            # FOR EACH CELL IN LOS, COMPUTE EQUIVALENT ORTHOGONAL POLY

            cell_area = cell.poly.area
            cell.dist_to_los_v1 = np.sqrt(((self.p1[0] - cell.R) ** 2) + ((self.p1[1] - cell.Z) ** 2))


            # Handle case of uniform (or close to uniform) LOS beam width (e.g., KG1V)
            if np.abs(self.w2-self.w1) <= 1.0e-03:
                cell.los_ortho_width = self.w2
            # Similar triangles method
            else:
                dw = (cell.dist_to_los_v1 / self.l12 ) * ((self.w2 - self.w1) / 2.0)
                cell.los_ortho_width = self.w1 + 2.0 * dw

            cell.los_ortho_delL = cell_area / cell.los_ortho_width

    def calc_int_and_1d_los_quantities(self):

        # SUM H EMISSION METHOD 1: TRAVERSE CELLS AND SUM EMISSIVITIES*ORTHO_DELL
        self.los_int['H_emiss'] = {}
        for key in self.spec_line_dict['1']['1']:
            sum_excit = 0
            sum_recom = 0
            for cell in self.cells:
                sum_excit += (cell.H_emiss[key]['excit'] * cell.los_ortho_delL)
                sum_recom += (cell.H_emiss[key]['recom'] * cell.los_ortho_delL)
            self.los_int['H_emiss'].update({key:{'excit':sum_excit, 'recom':sum_recom, 'units':'ph.s^-1.m^-2.sr^-1'}})

        # SUM IMPURITY EMISSION
        if self.spec_line_dict:
            self.los_int['imp_emiss'] = {}
            for at_num in self.spec_line_dict:
                if int(at_num)>1: # skip hydrogen
                    self.los_int['imp_emiss'][at_num] = {}
                    for ion_stage in self.spec_line_dict[at_num]:
                        self.los_int['imp_emiss'][at_num][ion_stage] = {}
                        for line_key in self.spec_line_dict[at_num][ion_stage]:
                            sum_excit = 0
                            sum_recom = 0
                            for cell in self.cells:
                                sum_excit += (cell.imp_emiss[at_num][ion_stage][line_key]['excit'] * cell.los_ortho_delL)
                                sum_recom += (cell.imp_emiss[at_num][ion_stage][line_key]['recom'] * cell.los_ortho_delL)
                            self.los_int['imp_emiss'][at_num][ion_stage].update({line_key:{'excit':sum_excit, 'recom':sum_recom, 'units':'ph.s^-1.m^-2.sr^-1'}})

        # SUM TOTAL RECOMBINATION/IONISATION
        self.los_int['Srec'] = {}
        self.los_int['Sion'] = {}
        sum_Srec = 0
        sum_Sion = 0
        for cell in self.cells:
            sum_Srec += (cell.Srec * cell.poly.area * 2.*np.pi*cell.R)
            sum_Sion += (cell.Sion * cell.poly.area * 2.*np.pi*cell.R)
        self.los_int['Srec'].update({'val':sum_Srec, 'units':'s^-1'})
        self.los_int['Sion'].update({'val':sum_Sion, 'units':'s^-1'})

        # SUM RADIATED POWER
        self.los_int['Prad'] = {}
        sum_Prad_H = 0
        for cell in self.cells:
            sum_Prad_H += (cell.H_radpwr)
        self.los_int['Prad'].update({'H':sum_Prad_H,  'units':'W'})

        if self.imp1_atom_num:
            sum_Prad_imp1 = np.zeros((self.imp1_atom_num))
            for cell in self.cells:
                sum_Prad_imp1 += (cell.imp1_radpwr)
            self.los_int['Prad'].update({'imp1':sum_Prad_imp1})
            self.los_int['Prad']['imp1'] = self.los_int['Prad']['imp1'].tolist()

        if self.imp2_atom_num:
            sum_Prad_imp2 = np.zeros((self.imp2_atom_num))
            for cell in self.cells:
                sum_Prad_imp2 += (cell.imp2_radpwr)
            self.los_int['Prad'].update({'imp2':sum_Prad_imp2})
            self.los_int['Prad']['imp2'] = self.los_int['Prad']['imp2'].tolist()


        ####################################################
        # COMPUTE AVERAGED QUANTITIES ALONG LOS
        ####################################################
        self.los_1d['dl'] = 0.01 # m
        self.los_1d['l'] = np.arange(0, self.l12, self.los_1d['dl'])
        for item in ['ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = np.zeros((len(self.los_1d['l'])))
        if self.imp1_atom_num:
            self.los_1d['imp1_den'] = np.zeros((len(self.los_1d['l']), self.imp1_atom_num+1))
        if self.imp2_atom_num:
            self.los_1d['imp2_den'] = np.zeros((len(self.los_1d['l']), self.imp2_atom_num+1))

        # H EMISS LOS 1D METHOD 2: SIMILAR TO METHOD 1 ABOVE, EXCEPT 1D PROFILE INFO IS ALSO STORED ALONG LOS
        recom = {}
        excit = {}
        recom_pervol = {} # emissivity per m^-3
        excit_pervol = {} # emissivity per m^-3

        for key in self.spec_line_dict['1']['1']:
            recom[key] = np.zeros((len(self.los_1d['l'])))
            excit[key] = np.zeros((len(self.los_1d['l'])))
            recom_pervol[key] = np.zeros((len(self.los_1d['l'])))
            excit_pervol[key] = np.zeros((len(self.los_1d['l'])))

        # Impurity emission dict. along LOS
        if self.spec_line_dict:
            imp_recom = {}
            imp_excit = {}
            imp_PEC_recom = {}
            imp_PEC_excit = {}
            imp_fPEC_recom = {}
            imp_fPEC_excit = {}
            imp_fPEC_recom_ionbal = {}
            imp_fPEC_excit_ionbal = {}
            for at_num in self.spec_line_dict:
                if int(at_num) > 1:  # skip hydrogen
                    imp_recom[at_num] = {}
                    imp_excit[at_num] = {}
                    imp_PEC_recom[at_num]  = {}
                    imp_PEC_excit[at_num]  = {}
                    imp_fPEC_recom[at_num] = {}
                    imp_fPEC_excit[at_num] = {}
                    imp_fPEC_recom_ionbal[at_num] = {}
                    imp_fPEC_excit_ionbal[at_num] = {}
                    for ion_stage in self.spec_line_dict[at_num]:
                        imp_recom[at_num][ion_stage] = {}
                        imp_excit[at_num][ion_stage] = {}
                        imp_PEC_recom[at_num][ion_stage]  = {}
                        imp_PEC_excit[at_num][ion_stage]  = {}
                        imp_fPEC_recom[at_num][ion_stage] = {}
                        imp_fPEC_excit[at_num][ion_stage] = {}
                        imp_fPEC_recom_ionbal[at_num][ion_stage] = {}
                        imp_fPEC_excit_ionbal[at_num][ion_stage] = {}
                        for line_key in self.spec_line_dict[at_num][ion_stage]:
                            imp_recom[at_num][ion_stage][line_key] = np.zeros((len(self.los_1d['l'])))
                            imp_excit[at_num][ion_stage][line_key] = np.zeros((len(self.los_1d['l'])))
                            imp_PEC_recom[at_num][ion_stage][line_key] = np.zeros((len(self.los_1d['l'])))
                            imp_PEC_excit[at_num][ion_stage][line_key] = np.zeros((len(self.los_1d['l'])))
                            imp_fPEC_recom[at_num][ion_stage][line_key] = np.zeros((len(self.los_1d['l'])))
                            imp_fPEC_excit[at_num][ion_stage][line_key] = np.zeros((len(self.los_1d['l'])))
                            imp_fPEC_recom_ionbal[at_num][ion_stage][line_key] = np.zeros((len(self.los_1d['l'])))
                            imp_fPEC_excit_ionbal[at_num][ion_stage][line_key] = np.zeros((len(self.los_1d['l'])))

        for dl_idx, dl_val in enumerate(self.los_1d['l']):
            # temp storage
            ne_tmp = []
            n0_tmp = []
            te_tmp = []
            imp1_den_tmp = []
            imp2_den_tmp = []
            delL_tmp = []
            H_emiss_excit_tmp = {}
            H_emiss_recom_tmp = {}
            H_emiss_pervol_excit_tmp = {}
            H_emiss_pervol_recom_tmp = {}
            for key in self.spec_line_dict['1']['1']:
                H_emiss_excit_tmp[key] = []
                H_emiss_recom_tmp[key] = []
                H_emiss_pervol_excit_tmp[key] = []
                H_emiss_pervol_recom_tmp[key] = []
                
            # Impurity emission dict. along LOS
            if self.spec_line_dict:
                imp_recom_tmp = {}
                imp_excit_tmp = {}
                imp_PEC_recom_tmp = {}
                imp_PEC_excit_tmp = {}
                imp_fPEC_recom_tmp = {}
                imp_fPEC_excit_tmp = {}
                imp_fPEC_recom_ionbal_tmp = {}
                imp_fPEC_excit_ionbal_tmp = {}
                for at_num in self.spec_line_dict:
                    if int(at_num) > 1:  # skip hydrogen
                        imp_recom_tmp[at_num] = {}
                        imp_excit_tmp[at_num] = {}
                        imp_PEC_recom_tmp[at_num]  = {}
                        imp_PEC_excit_tmp[at_num]  = {}
                        imp_fPEC_recom_tmp[at_num]  = {}
                        imp_fPEC_excit_tmp[at_num]  = {}
                        imp_fPEC_recom_ionbal_tmp[at_num]  = {}
                        imp_fPEC_excit_ionbal_tmp[at_num]  = {}
                        for ion_stage in self.spec_line_dict[at_num]:
                            imp_recom_tmp[at_num][ion_stage] = {}
                            imp_excit_tmp[at_num][ion_stage] = {}
                            imp_PEC_recom_tmp[at_num][ion_stage]  = {}
                            imp_PEC_excit_tmp[at_num][ion_stage]  = {}
                            imp_fPEC_recom_tmp[at_num][ion_stage]  = {}
                            imp_fPEC_excit_tmp[at_num][ion_stage]  = {}
                            imp_fPEC_recom_ionbal_tmp[at_num][ion_stage]  = {}
                            imp_fPEC_excit_ionbal_tmp[at_num][ion_stage]  = {}
                            for line_key in self.spec_line_dict[at_num][ion_stage]:
                                imp_recom_tmp[at_num][ion_stage][line_key] = []
                                imp_excit_tmp[at_num][ion_stage][line_key] = []
                                imp_PEC_recom_tmp[at_num][ion_stage][line_key] = []
                                imp_PEC_excit_tmp[at_num][ion_stage][line_key] = []
                                imp_fPEC_recom_tmp[at_num][ion_stage][line_key]  = []
                                imp_fPEC_excit_tmp[at_num][ion_stage][line_key]  = []
                                imp_fPEC_recom_ionbal_tmp[at_num][ion_stage][line_key]  = []
                                imp_fPEC_excit_ionbal_tmp[at_num][ion_stage][line_key]  = []
                            
            for cell in self.cells:
                if cell.dist_to_los_v1 >= dl_val and cell.dist_to_los_v1 < dl_val + self.los_1d['dl']:
                    ne_tmp.append(cell.ne)
                    n0_tmp.append(cell.n0)
                    te_tmp.append(cell.te)
                    if self.imp1_atom_num:
                        imp1_den_tmp.append(cell.imp1_den)
                    if self.imp2_atom_num:
                        imp2_den_tmp.append(cell.imp2_den)
                    delL_tmp.append(cell.los_ortho_delL)
                    for key in self.spec_line_dict['1']['1']:
                        H_emiss_excit_tmp[key].append(cell.H_emiss[key]['excit']*cell.los_ortho_delL)
                        H_emiss_recom_tmp[key].append(cell.H_emiss[key]['recom']*cell.los_ortho_delL)
                        H_emiss_pervol_excit_tmp[key].append(cell.H_emiss[key]['excit'])
                        H_emiss_pervol_recom_tmp[key].append(cell.H_emiss[key]['recom'])
                    if self.spec_line_dict:
                        for at_num in self.spec_line_dict:
                            if int(at_num)>1: # skip hydrogen
                                for ion_stage in self.spec_line_dict[at_num]:
                                    for line_key in self.spec_line_dict[at_num][ion_stage]:
                                        imp_recom_tmp[at_num][ion_stage][line_key].append(cell.imp_emiss[at_num][ion_stage][line_key]['recom']*cell.los_ortho_delL)
                                        imp_excit_tmp[at_num][ion_stage][line_key].append(cell.imp_emiss[at_num][ion_stage][line_key]['excit']*cell.los_ortho_delL)
                                        imp_PEC_recom_tmp[at_num][ion_stage][line_key].append(cell.imp_emiss[at_num][ion_stage][line_key]['PEC_recom'])#*cell.los_ortho_delL)
                                        imp_PEC_excit_tmp[at_num][ion_stage][line_key].append(cell.imp_emiss[at_num][ion_stage][line_key]['PEC_excit'])#*cell.los_ortho_delL)
                                        imp_fPEC_recom_tmp[at_num][ion_stage][line_key].append(cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_recom'])#*cell.los_ortho_delL)
                                        imp_fPEC_excit_tmp[at_num][ion_stage][line_key].append(cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_excit'])#*cell.los_ortho_delL)
                                        imp_fPEC_recom_ionbal_tmp[at_num][ion_stage][line_key].append(cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_recom_ionbal'])#*cell.los_ortho_delL)
                                        imp_fPEC_excit_ionbal_tmp[at_num][ion_stage][line_key].append(cell.imp_emiss[at_num][ion_stage][line_key]['fPEC_excit_ionbal'])#*cell.los_ortho_delL)

            if ne_tmp:
                self.los_1d['ortho_delL'][dl_idx] = np.sum(np.asarray(delL_tmp))
                delL_norm = np.asarray(delL_tmp) / self.los_1d['ortho_delL'][dl_idx]
                self.los_1d['ne'][dl_idx] = np.average(np.asarray(ne_tmp),weights = delL_norm)
                self.los_1d['n0'][dl_idx] = np.average(np.asarray(n0_tmp),weights = delL_norm)
                self.los_1d['te'][dl_idx] = np.average(np.asarray(te_tmp),weights = delL_norm)
                if imp1_den_tmp:
                    self.los_1d['imp1_den'][dl_idx] = np.average(np.asarray(imp1_den_tmp),axis=0, weights = delL_norm)
                if imp2_den_tmp:
                    self.los_1d['imp2_den'][dl_idx] = np.average(np.asarray(imp2_den_tmp),axis=0, weights = delL_norm)
                for key in self.spec_line_dict['1']['1']:
                    excit[key][dl_idx] = np.sum(np.asarray(H_emiss_excit_tmp[key]))
                    recom[key][dl_idx] = np.sum(np.asarray(H_emiss_recom_tmp[key]))
                    excit_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_excit_tmp[key]), weights = delL_norm)
                    recom_pervol[key][dl_idx] = np.average(np.asarray(H_emiss_pervol_recom_tmp[key]), weights = delL_norm )
                if self.spec_line_dict:
                    for at_num in self.spec_line_dict:
                        if int(at_num) > 1:  # skip hydrogen
                            for ion_stage in self.spec_line_dict[at_num]:
                                for line_key in self.spec_line_dict[at_num][ion_stage]:
                                    imp_recom[at_num][ion_stage][line_key][dl_idx] = np.sum(np.asarray(imp_recom_tmp[at_num][ion_stage][line_key]))
                                    imp_excit[at_num][ion_stage][line_key][dl_idx] = np.sum(np.asarray(imp_excit_tmp[at_num][ion_stage][line_key]))
                                    imp_PEC_recom[at_num][ion_stage][line_key][dl_idx] = np.average(np.asarray(imp_PEC_recom_tmp[at_num][ion_stage][line_key]),weights = delL_norm)
                                    imp_PEC_excit[at_num][ion_stage][line_key][dl_idx] = np.average(np.asarray(imp_PEC_excit_tmp[at_num][ion_stage][line_key]),weights = delL_norm)
                                    imp_fPEC_recom[at_num][ion_stage][line_key][dl_idx] = np.average(np.asarray(imp_fPEC_recom_tmp[at_num][ion_stage][line_key]),weights = delL_norm)
                                    imp_fPEC_excit[at_num][ion_stage][line_key][dl_idx] = np.average(np.asarray(imp_fPEC_excit_tmp[at_num][ion_stage][line_key]),weights = delL_norm)
                                    imp_fPEC_recom_ionbal[at_num][ion_stage][line_key][dl_idx] = np.average(np.asarray(imp_fPEC_recom_ionbal_tmp[at_num][ion_stage][line_key]),weights = delL_norm)
                                    imp_fPEC_excit_ionbal[at_num][ion_stage][line_key][dl_idx] = np.average(np.asarray(imp_fPEC_excit_ionbal_tmp[at_num][ion_stage][line_key]),weights = delL_norm)

        # REMOVE ZEROS FROM COMPUTED LOS ARRAYS
        nonzero_idx = np.nonzero(self.los_1d['ne'])
        for item in ['l', 'ne', 'n0', 'te', 'ortho_delL']:
            self.los_1d[item] = self.los_1d[item][nonzero_idx]
            # convert numpy arrays to lists for JSON serialization
            self.los_1d[item] = list(self.los_1d[item])

        if self.imp1_atom_num:
            self.los_1d['imp1_den'] = self.los_1d['imp1_den'][nonzero_idx]
            self.los_1d['imp1_den'] = self.los_1d['imp1_den'].tolist()
        if self.imp2_atom_num:
            self.los_1d['imp2_den'] = self.los_1d['imp2_den'][nonzero_idx]
            self.los_1d['imp2_den'] = self.los_1d['imp2_den'].tolist()

        self.los_1d['H_emiss'] = {}
        self.los_1d['H_emiss_per_vol'] = {}
        for key in self.spec_line_dict['1']['1']:
            excit[key] = excit[key][nonzero_idx]
            recom[key] = recom[key][nonzero_idx]
            excit_pervol[key] = excit_pervol[key][nonzero_idx]
            recom_pervol[key] = recom_pervol[key][nonzero_idx]
            self.los_1d['H_emiss'].update({key:{'excit':excit[key], 'recom':recom[key], 'units':'ph s^-1 m^-2 sr^-1'}})
            self.los_1d['H_emiss_per_vol'].update({key:{'excit':excit_pervol[key], 'recom':recom_pervol[key], 'units':'ph s^-1 m^-3 sr^-1'}})
            # convert numpy arrays to lists for JSON serialization
            self.los_1d['H_emiss'][key]['excit'] = list(self.los_1d['H_emiss'][key]['excit'])
            self.los_1d['H_emiss'][key]['recom'] = list(self.los_1d['H_emiss'][key]['recom'])
            self.los_1d['H_emiss_per_vol'][key]['excit'] = list(self.los_1d['H_emiss_per_vol'][key]['excit'])
            self.los_1d['H_emiss_per_vol'][key]['recom'] = list(self.los_1d['H_emiss_per_vol'][key]['recom'])

        if self.spec_line_dict:
            self.los_1d['imp_emiss'] = {}
            for at_num in self.spec_line_dict:
                if int(at_num) > 1:  # skip hydrogen
                    self.los_1d['imp_emiss'][at_num] = {}
                    for ion_stage in self.spec_line_dict[at_num]:
                        self.los_1d['imp_emiss'][at_num][ion_stage]= {}
                        for line_key in self.spec_line_dict[at_num][ion_stage]:
                            imp_recom[at_num][ion_stage][line_key] = imp_recom[at_num][ion_stage][line_key][nonzero_idx]
                            imp_excit[at_num][ion_stage][line_key] = imp_excit[at_num][ion_stage][line_key][nonzero_idx]
                            imp_PEC_recom[at_num][ion_stage][line_key] = imp_PEC_recom[at_num][ion_stage][line_key][nonzero_idx]
                            imp_PEC_excit[at_num][ion_stage][line_key] = imp_PEC_excit[at_num][ion_stage][line_key][nonzero_idx]
                            imp_fPEC_recom[at_num][ion_stage][line_key] = imp_fPEC_recom[at_num][ion_stage][line_key][nonzero_idx]
                            imp_fPEC_excit[at_num][ion_stage][line_key] = imp_fPEC_excit[at_num][ion_stage][line_key][nonzero_idx]
                            imp_fPEC_recom_ionbal[at_num][ion_stage][line_key] = imp_fPEC_recom_ionbal[at_num][ion_stage][line_key][nonzero_idx]
                            imp_fPEC_excit_ionbal[at_num][ion_stage][line_key] = imp_fPEC_excit_ionbal[at_num][ion_stage][line_key][nonzero_idx]
                            self.los_1d['imp_emiss'][at_num][ion_stage].update({line_key:{'excit':imp_excit[at_num][ion_stage][line_key],
                                                                                          'recom':imp_recom[at_num][ion_stage][line_key],
                                                                                          'PEC_excit':imp_PEC_excit[at_num][ion_stage][line_key],
                                                                                          'PEC_recom':imp_PEC_recom[at_num][ion_stage][line_key],
                                                                                          'fPEC_excit':imp_fPEC_excit[at_num][ion_stage][line_key],
                                                                                          'fPEC_recom':imp_fPEC_recom[at_num][ion_stage][line_key],
                                                                                          'fPEC_excit_ionbal':imp_fPEC_excit_ionbal[at_num][ion_stage][line_key],
                                                                                          'fPEC_recom_ionbal':imp_fPEC_recom_ionbal[at_num][ion_stage][line_key],
                                                                                          'units':'ph s^-1 m^-2 sr^-1'}})
                            # convert numpy arrays to lists for JSON serialization
                            self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['excit'] = self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['excit'].tolist()
                            self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['recom'] = self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['recom'].tolist()
                            self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['PEC_excit'] = self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['PEC_excit'].tolist()
                            self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['PEC_recom'] = self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['PEC_recom'].tolist()
                            self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['fPEC_excit'] = self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['fPEC_excit'].tolist()
                            self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['fPEC_recom'] = self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['fPEC_recom'].tolist()
                            self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['fPEC_excit_ionbal'] = self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['fPEC_excit_ionbal'].tolist()
                            self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['fPEC_recom_ionbal'] = self.los_1d['imp_emiss'][at_num][ion_stage][line_key]['fPEC_recom_ionbal'].tolist()

    def calc_int_and_1d_los_synth_spectra(self):

        ###############################################################
        # FREE-FREE AND FREE-BOUND CONTINUUM
        ###############################################################
        print('Calculating ff fb continuum spectra for chord', self.chord_num)
        wave_nm = np.linspace(300, 500, 50)

        # METHOD 1: SUM CELL-WISE (OK, good agreement with METHOD 2 below, so comment out)
        # sum_ff_fb = np.zeros((len(wave_nm)))
        # for cell in self.cells:
        #     # call adas continuo function (return units: ph s-1 m3 sr-1 nm-1)
        #     ff_only, ff_fb_tot = continuum_utils.adas_continuo_py(wave_nm, cell.te, 1, 1)
        #     # convert to spectral radiance: ph s-1 m-2 sr-1 nm-1
        #     sum_ff_fb += ff_fb_tot * cell.ne * cell.ne * cell.los_ortho_delL

        # METHOD 2: SUM BASED ON AVERAGED 1D LOS PARAMS
        # TODO: ADD ZEFF CAPABILITY
        dl_ff_fb_abs = np.zeros((len(self.los_1d['l']), len(wave_nm)))
        for dl_idx, dl_val in enumerate(self.los_1d['l']):
            # call adas continuo function (return units: ph s-1 m3 sr-1 nm-1)
            ff_only, ff_fb_tot = continuum_utils.adas_continuo_py(wave_nm, self.los_1d['te'][dl_idx], 1, 1)
            # convert to spectral radiance: ph s-1 m-2 sr-1 nm-1
            dl_ff_fb_abs[dl_idx] = ff_fb_tot * self.los_1d['ne'][dl_idx] * self.los_1d['ne'][dl_idx] * self.los_1d['ortho_delL'][dl_idx]

        # store spectra
        self.los_1d_spectra['ff_fb_continuum'] = {'wave':wave_nm, 'intensity':dl_ff_fb_abs, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
        self.los_int_spectra['ff_fb_continuum'] = {'wave':wave_nm, 'intensity':np.sum(dl_ff_fb_abs, axis=0), 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
        # convert numpy array to list for JSON serialization
        self.los_1d_spectra['ff_fb_continuum']['wave'] = list(self.los_1d_spectra['ff_fb_continuum']['wave'])
        self.los_1d_spectra['ff_fb_continuum']['intensity'] = list(self.los_1d_spectra['ff_fb_continuum']['intensity'])
        self.los_int_spectra['ff_fb_continuum']['wave'] = list(self.los_int_spectra['ff_fb_continuum']['wave'])
        self.los_int_spectra['ff_fb_continuum']['intensity'] = list(self.los_int_spectra['ff_fb_continuum']['intensity'])

        # fig, ax = plt.subplots(ncols=1)
        # ax.semilogy(self.los_int_spectra['ff_fb_continuum']['wave'], self.los_int_spectra['ff_fb_continuum']['intensity'], '-k', lw=3.0)
        # # ax.semilogy(self.los_int_spectra['ff_fb_continuum']['wave'], sum_ff_fb, '-r')
        # for dl_idx, dl_val in enumerate(self.los_1d['l']):
        #     ax.semilogy(self.los_int_spectra['ff_fb_continuum']['wave'], self.los_1d_spectra['ff_fb_continuum']['intensity'][dl_idx], '--k')
        # ax.set_xlabel('Wavelength (nm)')
        # ax.set_ylabel(r'$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}\/nm^{-1}}$')
        # ax.set_title(str(self.v2unmod[0]))
        # plt.show()

        ###############################################################
        # STARK BROADENED H6-2 LINE
        ###############################################################
        print('Calculating Stark H6-2 spectra for chord ', self.chord_num)
        # Generate modified lorentzian profile for each dl position along los (cf Lomanowski et al 2015, NF)
        for key in self.spec_line_dict['1']['1']:
            # H 6-2
            if self.spec_line_dict['1']['1'][key][1] == '2' and self.spec_line_dict['1']['1'][key][0] == '6':
                cwl = float(key) / 10.0 # nm
                # wave_nm = np.linspace(cwl-5, cwl+5, 10000)
                wave_nm = np.linspace(cwl-15, cwl+15, 1000)
                dl_stark = np.zeros((len(self.los_1d['l']), len(wave_nm)))
                for dl_idx, dl_val in enumerate(self.los_1d['l']):
                    stark_fwhm = ( self.lorentz_5_2_coeffs[key]['C']*np.power(self.los_1d['ne'][dl_idx], self.lorentz_5_2_coeffs[key]['a']) /
                                   np.power(self.los_1d['te'][dl_idx], self.lorentz_5_2_coeffs[key]['b']) )
                    dl_stark[dl_idx] = 1. / ( np.power(np.abs(wave_nm-cwl), 5./2.) + np.power(stark_fwhm / 2., 5./2.) )
                    # normalise by emissivity
                    dl_emiss = self.los_1d['H_emiss'][key]['excit'][dl_idx] + self.los_1d['H_emiss'][key]['recom'][dl_idx]
                    wv_area = np.trapz(dl_stark[dl_idx], x = wave_nm)
                    amp_scal = dl_emiss / wv_area
                    dl_stark[dl_idx] *= amp_scal

                # store spectra
                self.los_1d_spectra['stark'] = {'cwl':cwl, 'wave':wave_nm, 'intensity':dl_stark, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                self.los_int_spectra['stark'] = {'cwl':cwl, 'wave':wave_nm, 'intensity':np.sum(dl_stark, axis=0), 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                # convert numpy array to list for JSON serialization
                self.los_1d_spectra['stark']['wave'] = list(self.los_1d_spectra['stark']['wave'])
                self.los_1d_spectra['stark']['intensity'] = list(self.los_1d_spectra['stark']['intensity'])
                self.los_int_spectra['stark']['wave'] = list(self.los_int_spectra['stark']['wave'])
                self.los_int_spectra['stark']['intensity'] = list(self.los_int_spectra['stark']['intensity'])


                # fig, ax = plt.subplots(ncols=1)
                # ax.semilogy(self.los_int_spectra['stark']['wave'], self.los_int_spectra['stark']['intensity'], '-k', lw=3.0)
                # for dl_idx, dl_val in enumerate(self.los_1d['l']):
                #     ax.semilogy(self.los_int_spectra['stark']['wave'], self.los_1d_spectra['stark']['intensity'][dl_idx], '--k')
                # ax.set_xlabel('Wavelength (nm)')
                # ax.set_ylabel(r'$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}\/nm^{-1}}$')
                # ax.set_title(str(self.v2unmod[0]))
                # plt.show()

        ###############################################################
        # N II AFG feature (399 < wv < 410 nm)
        ###############################################################
        if self.calc_NII_afg_feature:
            if self.imp1_atom_num or self.imp2_atom_num:

                # ionization balance flag
                ionbal=0
                afg_adasn1_tmp = []
                IDL_afg_adasn1= idl.export_function("calc_nii_emiss__afg_adasn1")
                for dl_idx, dl_val in enumerate(self.los_1d['l']):
                    if self.imp1_atom_num == 7:
                        nii_cm3 = 1.e-06 * self.los_1d['imp1_den'][dl_idx][1]
                        niii_cm3 = 1.e-06 * self.los_1d['imp1_den'][dl_idx][2]
                    elif self.imp2_atom_num == 7:
                        nii_cm3 = 1.e-06 * self.los_1d['imp2_den'][dl_idx][1]
                        niii_cm3 = 1.e-06 * self.los_1d['imp2_den'][dl_idx][2]
                    # call IDL calc_nii_emiss__afg_adas1 function (returns: wv[nm] intensity [ph s-1 cm-3])
                    # if ionbal=1, nii_cm3 and niii_cm3 are ignored
                    res_dict = IDL_afg_adasn1(self.los_1d['te'][dl_idx], 1.e-06 * self.los_1d['ne'][dl_idx], nii_cm3, niii_cm3, ionbal)
                    if dl_idx == 0: wv_nm = res_dict['wv'] / 10.
                    if ionbal == 0:
                        # convert to radiance: ph s-1 m-2 sr-1
                        afg_adasn1_tmp.append( (1./ (4.*np.pi)) * 1.e06*res_dict['intensity'] * self.los_1d['ortho_delL'][dl_idx] )
                    else:
                        afg_adasn1_tmp.append( res_dict['intensity'] * self.los_1d['ortho_delL'][dl_idx] )

                afg_adasn1_arr = np.asarray(afg_adasn1_tmp)

                # store spectra
                if ionbal == 0:
                    units = 'nm, ph s^-1 m^-2 sr^-1'
                else:
                    units = 'arb. units'
                self.los_1d_spectra['afg_adasn1'] = {'wave':wv_nm, 'intensity':afg_adasn1_arr, 'units':units}
                self.los_int_spectra['afg_adasn1'] = {'wave':wv_nm, 'intensity':np.sum(afg_adasn1_arr, axis=0), 'units':units}
                # # convert numpy array to list for JSON serialization
                self.los_1d_spectra['afg_adasn1']['wave'] = self.los_1d_spectra['afg_adasn1']['wave'].tolist()
                self.los_1d_spectra['afg_adasn1']['intensity'] = self.los_1d_spectra['afg_adasn1']['intensity'].tolist()
                self.los_int_spectra['afg_adasn1']['wave'] = self.los_int_spectra['afg_adasn1']['wave'].tolist()
                self.los_int_spectra['afg_adasn1']['intensity'] = self.los_int_spectra['afg_adasn1']['intensity'].tolist()

                # DEBUGGING
                # fig, ax = plt.subplots(ncols=1)
                # ax.semilogy(self.los_int_spectra['afg_adasn1']['wave'], self.los_int_spectra['afg_adasn1']['intensity'], 'ok', lw=3.0)
                # for dl_idx, dl_val in enumerate(self.los_1d['l']):
                #     ax.semilogy(self.los_int_spectra['afg_adasn1']['wave'], self.los_1d_spectra['afg_adasn1']['intensity'][dl_idx], 'xk')

                # GENERATE SYNTHETIC NII AFG SPECTRA CONVOLVED WITH INST. FN
                KT3B_1200_instfwhm = 0.085 #nm
                KT3B_1200_dwv = 0.0135 #nm
                wv_nm_fine = np.arange(398, 410, KT3B_1200_dwv)
                afg_adasn1_spectrum = np.zeros((len(wv_nm_fine)))
                for iline_int, vline_int in enumerate(self.los_int_spectra['afg_adasn1']['intensity']):
                    # Convert line intensity to Gaussian with FWHM = KT3B_1200_instfwhm [units: phs/s/m2/sr/nm]
                    afg_adasn1_spectrum += gaussian(self.los_int_spectra['afg_adasn1']['wave'][iline_int], wv_nm_fine,
                                                    self.los_int_spectra['afg_adasn1']['intensity'][iline_int], KT3B_1200_instfwhm)
                # NOW ADD 6-2 LINE AND FF-FB CONTINUUM
                # f = interp1d(self.los_int_spectra['stark']['wave'], self.los_int_spectra['stark']['intensity'])
                # stark_interp = f(wv_nm_fine)
                f = interp1d(self.los_int_spectra['ff_fb_continuum']['wave'], self.los_int_spectra['ff_fb_continuum']['intensity'])
                ff_fb_interp = f(wv_nm_fine)

                # DEBUGGING
                # fig, ax = plt.subplots(ncols=1)
                # ax.plot(wv_nm_fine, ff_fb_interp+afg_adasn1_spectrum, '-k', lw=2.0)
                # plt.show()

                # store spectra
                # self.los_int_spectra['afg_adasn1_kt3b1200'] = {'wave':wv_nm_fine, 'intensity':ff_fb_interp+afg_adasn1_spectrum, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                self.los_int_spectra['afg_adasn1_kt3b1200'] = {'wave':wv_nm_fine, 'intensity':afg_adasn1_spectrum, 'units':'nm, ph s^-1 m^-2 sr^-1 nm^-1'}
                # convert numpy array to list for JSON serialization
                self.los_int_spectra['afg_adasn1_kt3b1200']['wave'] = list(self.los_int_spectra['afg_adasn1_kt3b1200']['wave'])
                self.los_int_spectra['afg_adasn1_kt3b1200']['intensity'] = list(self.los_int_spectra['afg_adasn1_kt3b1200']['intensity'])

class SynthDiag:

    def __init__(self, defs, diag, pulse=None, spec_line_dict=None, imp1_atom_num=None,
                 imp2_atom_num=None, calc_NII_afg_feature=False):

        self.calc_NII_afg_feature = calc_NII_afg_feature
        self.diag = diag
        self.pulse = pulse
        self.chords = []
        self.spec_line_dict = spec_line_dict
        self.imp1_atom_num = imp1_atom_num
        self.imp2_atom_num = imp2_atom_num

        self.get_spec_geom(defs)

    def get_spec_geom(self, defs):

        if self.diag in defs.diag_dict.keys():
            p2new = np.zeros((len(defs.diag_dict[self.diag]['p2']), 2))
            for i in range(len(defs.diag_dict[self.diag]['p1'])):
                r1 = defs.diag_dict[self.diag]['p1'][i, 0]
                z1 = defs.diag_dict[self.diag]['p1'][i, 1]
                r2 = defs.diag_dict[self.diag]['p2'][i, 0]
                z2 = defs.diag_dict[self.diag]['p2'][i, 1]
                w1 = defs.diag_dict[self.diag]['w'][i, 0]
                w2 = defs.diag_dict[self.diag]['w'][i, 1]
                theta = np.arctan((r2 - r1) / (z2 - z1))
                # elongate los to ensure defined LOS intersects machine wall (otherwise grid cells may be excluded)
                chord_L = np.sqrt((r1 - r2) ** 2 + (z1 - z2) ** 2)
                p2new[i, 0] = r2 - 1.0 * np.sin(theta)
                p2new[i, 1] = z2 - 1.0 * np.cos(theta)
                chord_L_elong = np.sqrt((r1 - p2new[i, 0]) ** 2 + (z1 - p2new[i, 1]) ** 2)
                w2_elong = w2 * chord_L_elong / chord_L
                self.chords.append(LOS(los_poly=Polygon([(r1, z1),
                                                         (p2new[i, 0] - 0.5 * w2_elong * np.cos(theta),
                                                          p2new[i, 1] + 0.5 * w2_elong * np.sin(theta)),
                                                         (p2new[i, 0] + 0.5 * w2_elong * np.cos(theta),
                                                          p2new[i, 1] - 0.5 * w2_elong * np.sin(theta)),
                                                         (r1, z1)]),
                                       chord_num=defs.diag_dict[self.diag]['id'][i], p1=[r1, z1], w1=w1, p2orig=[r2, z2],
                                       p2=[p2new[i, 0], p2new[i, 1]], w2orig=w2, w2=w2_elong, l12=chord_L_elong, theta=theta,
                                       spec_line_dict=self.spec_line_dict,
                                       imp1_atom_num=self.imp1_atom_num, imp2_atom_num=self.imp2_atom_num,
                                       calc_NII_afg_feature=self.calc_NII_afg_feature))

    def plot_LOS(self, ax, color='w', lw='2.0', Rrng=None):
        for chord in self.chords:
            if Rrng:
                if chord.v2unmod[0] >= Rrng[0] and chord.v2unmod[0] <= Rrng[1]:
                    los_patch = patches.Polygon(chord.los_poly.exterior.coords, closed=False, ec=color, lw=lw, fc='None', zorder=10)
                    ax.add_patch(los_patch)
            else:
                los_patch = patches.Polygon(chord.los_poly.exterior.coords, closed=False, ec=color, lw=lw, fc='None', zorder=10)
                ax.add_patch(los_patch)

    def plot_synth_spec_edge2d_data(self):
        fig, ax1 = plt.subplots(ncols=3, sharex=True, sharey=True)
        ax1[0].set_xlim(1.8, 4.0)
        ax1[0].set_ylim(-2.0, 2.0)
        fig.suptitle(self.spec)
        ax1[0].set_title(r'$\mathrm{T_{e}}$')
        ax1[1].set_title(r'$\mathrm{n_{e}}$')
        ax1[2].set_title(r'$\mathrm{n_{0}}$')

        recon_patches=[]
        recon_grid = []
        los_patches = []
        te=[]
        ne=[]
        n0=[]
        for chord in self.chords:
            los_patches.append(patches.Polygon(chord.los_poly.exterior.coords, closed=False, color='r'))
            for cell in chord.cells:
                recon_patches.append(patches.Polygon(cell.poly.exterior.coords, closed=True))
                recon_grid.append(patches.Polygon(cell.poly.exterior.coords, closed=False, color='k', alpha=1.0))
                te.append(cell.te)
                ne.append(cell.ne)
                n0.append(cell.n0)


        coll1 = PatchCollection(recon_patches)
        colors = plt.cm.jet(te/(np.max(te)/10.))
        coll1.set_color(colors)
        ax1[0].add_collection(coll1)

        coll2 = PatchCollection(recon_patches)
        colors = plt.cm.jet(ne/np.max(ne))
        coll2.set_color(colors)
        ax1[1].add_collection(coll2)

        coll3 = PatchCollection(recon_patches)
        colors = plt.cm.jet(n0/np.max(n0))
        coll3.set_color(colors)
        ax1[2].add_collection(coll3)

        split_grid = PatchCollection(recon_grid)
        split_grid.set_facecolor('None')
        split_grid.set_edgecolor('k')
        split_grid.set_linewidth(0.25)
        ax1[0].add_collection(split_grid)

        split_grid2 = PatchCollection(recon_grid)
        split_grid2.set_facecolor('None')
        split_grid2.set_edgecolor('k')
        split_grid2.set_linewidth(0.25)
        ax1[1].add_collection(split_grid2)

        split_grid3 = PatchCollection(recon_grid)
        split_grid3.set_facecolor('None')
        split_grid3.set_edgecolor('k')
        split_grid3.set_linewidth(0.25)
        ax1[2].add_collection(split_grid3)

        # los_coll = PatchCollection(los_patches)
        # los_coll.set_facecolor('None')
        # los_coll.set_edgecolor('r')
        # los_coll.set_linewidth(1.0)
        # ax1[0].add_collection(los_coll)
        #
        # los_coll1 = PatchCollection(los_patches)
        # los_coll1.set_facecolor('None')
        # los_coll1.set_edgecolor('r')
        # los_coll1.set_linewidth(1.0)
        # ax1[1].add_collection(los_coll1)
        #
        # los_coll2 = PatchCollection(los_patches)
        # los_coll2.set_facecolor('None')
        # los_coll2.set_edgecolor('r')
        # los_coll2.set_linewidth(1.0)
        # ax1[2].add_collection(los_coll2)

        # PLOT SEP INTERSECTIONS POINTS ON OUTER DIVERTOR LEG
        for chord in self.chords:
            if chord.shply_intersects_w_sep:
                ax1[0].plot(chord.shply_intersects_w_sep.coords.xy[0][0],chord.shply_intersects_w_sep.coords.xy[1][0], 'rx', ms=8, mew=3.0, zorder=10)

        if self.diag == 'KT3A':
            # PLOT PLASMA PROPERTIES ALONG LOS
            fig2, ax2 = plt.subplots(nrows=4, sharex=True)
            ax2[0].set_xlim(0,6)
            for chord in self.chords:
                if chord.v2unmod[0] >=2.74 and chord.v2unmod[0]<=2.76:
                    col = np.random.rand(3,1)
                    ax2[0].plot(chord.los_1d['l'], chord.los_1d['te'], '-', color=col, lw=2.0)
                    ax2[1].plot(chord.los_1d['l'], chord.los_1d['ne'], '-', color=col, lw=2.0)
                    ax2[2].plot(chord.los_1d['l'], chord.los_1d['n0'], '-', color=col, lw=2.0)
                    for key in self.spec_line_dict['1']['1']:
                        ax2[3].plot(chord.los_1d['l'], np.asarray(chord.los_1d['H_emiss_per_vol'][key]['excit'])+np.asarray(chord.los_1d['H_emiss_per_vol'][key]['recom']), '-', color=col)

            ax2[0].set_ylabel(r'$\mathrm{T_{e}}$')
            ax2[1].set_ylabel(r'$\mathrm{n_{e}}$')
            ax2[2].set_ylabel(r'$\mathrm{n_{0}}$')
            ax2[3].set_ylabel(r'$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}\/nm^{-1}}$')
            ax2[3].set_xlabel('distance along LOS (m)')

            # PLOT H INTEGRATED EMISSION (compare both methods for summing emission - expect identical results)
            fig3, ax3 = plt.subplots(nrows=1)
            # col_dict = {'6561.9':'r','4860.6':'m', '4339.9':'orange', '4101.2':'darkgreen', '3969.5':'b'}
            for key in self.spec_line_dict['1']['1']:
                emiss = []
                coord = []
                for chord in self.chords:
                    emiss.append(chord.los_int['H_emiss'][key]['excit'] + chord.los_int['H_emiss'][key]['recom'])
                    coord.append(chord.v2unmod[0])
                col = np.random.rand(3,1)
                ax3.semilogy(coord, emiss, '-', color=col, lw=3.0)
            ax3.set_xlabel('R tile 5 (m)')
            ax3.set_ylabel(r'$\mathrm{ph\/s^{-1}\/m^{-2}\/sr^{-1}\/nm^{-1}}$')
                # ax3.plot(chord.v2unmod[0], np.sum(chord.los_1d['H_emiss']['6561.9']['recom']), 'rx', markersize=10),

class Cell:

    def __init__(self, R = None, Z = None, poly=None, te=None, ne=None, ni=None, n0=None, Srec=None, Sion=None, imp1_den=None, imp2_den=None):

        self.R = R # m
        self.Z = Z # m
        self.poly = poly
        self.te = te # eV
        self.ni = ni # m^-3
        self.ne = ne # m^-3
        self.n0 = n0 # m^-3
        self.imp1_den = imp1_den # m^-3
        self.imp2_den = imp2_den # m^-3
        self.imp1_radpwr = []
        self.imp1_radpwr_coeff = [] # W m^3
        self.imp2_radpwr = []
        self.imp2_radpwr_coeff = []  # W m^3
        self.H_emiss = {}
        self.ff_fb_emiss = None
        self.H_radpwr = None
        self.imp_emiss = {}
        self.Srec = Srec # m^-3 s^-1
        self.Sion = Sion # m^-3 s^-1

        # LOS ORTHOGONAL POLYGON PROPERTIES
        self.dist_to_los_v1 = None
        self.los_ortho_width = None
        self.los_ortho_delL = None

if __name__=='__main__':

    # EXAMPLE

    # compare to 90423
    tranfile = '/u/cstavrou/cmg/catalog/edge2d/jet/81472/may1117/seq#2/tran'

    # H_line_dict = {'1215.2':[2,1], '1025.3':[3,1], '6561.9':[3,2],'4860.6':[4,2], '4339.9':[5,2], '4101.2':[6,2], '3969.5':[7,2]}

    spec_line_dict = {
        '1': # HYDROGEN
            {'1': {'1215.2': ['2', '1'],
                   '1025.3': ['3', '1'],
                   '6561.9': ['3', '2'],
                   '4860.6': ['4', '2'],
                   '4339.9': ['5', '2'],
                   '4101.2': ['6', '2'],
                   '3969.5': ['7', '2'],},
             },
        '7': # NITROGEN
            {'2':{'4042.07':['4f', '3d'],
                  '5002.18':['3d', '3p'],
                  '5005.86':['3d', '3p']},
             '3':{'4100.51':['3p', '3s']},
             '4':{'3481.83':['3p', '3s']}
             }
    }
    
    # Read all necessary ADAS data here and store in dict
    ADAS_dict = {}
    Te_rnge = [0.2, 5000]
    ne_rnge = [1.0e11, 1.0e15]
    ADAS_samples = 10
    ADAS_dict['adf15'] = adas_adf15_utils.get_adas_imp_PECs_interp(spec_line_dict, Te_rnge, ne_rnge,
                                                                        npts=ADAS_samples, npts_interp=1000)
    # Also get adf11 for the ionisation balance fractional abundance. No Te_arr, ne_arr interpolation
    # available in the adf11 reader at the moment, so generate more coarse array (sloppy!)
    # TODO: add interpolation capability to the adf11 reader so that adf15 and adf11 are on the same Te, ne grid
    Te_arr_adf11 = np.logspace(np.log10(Te_rnge[0]), np.log10(Te_rnge[1]), 500)
    ne_arr_adf11 = np.logspace(np.log10(ne_rnge[0]), np.log10(ne_rnge[1]), 30)
    ADAS_dict['adf11'] = {}
    for atnum in spec_line_dict:
        if int(atnum) > 1:
            ADAS_dict['adf11'][atnum] = adas_adf11_utils.get_adas_imp_adf11(int(atnum), Te_arr_adf11, ne_arr_adf11)
        elif int(atnum) == 1:
            ADAS_dict['adf11'][atnum] = adas_adf11_utils.get_adas_H_adf11_interp(Te_rnge, ne_rnge,
                                                                                      npts=ADAS_samples,
                                                                                      npts_interp=1000, pwr=True)

    res = Pyproc(ADAS_dict, tranfile=tranfile, machine='JET', interactive_plots= True, spec_line_dict=spec_line_dict, calc_synth_spec_features=True, diag_list= ['KT3A'])

    plt.show()

