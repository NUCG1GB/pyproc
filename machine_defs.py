
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt

class MachineDefs:

    def __init__(self, name, wall_coords, pulse_ref=90531):
        self.name = name
        self.pulse_ref = pulse_ref
        self.__wall_coords = wall_coords
        self.__diag_dict = {}

    def set_diag_los(self, diag, los_geom):
        if diag not in self.__diag_dict.keys():
            self.__diag_dict[diag] = los_geom

    @property
    def diag_dict(self):
        return self.__diag_dict

    @property
    def wall_coords(self):
        return self.__wall_coords

def los_width_from_neigbh(los, los_neigh):
    los_m = (los[0,0]-los[1,0]) / (los[0,1]-los[1,1])
    los_neigh_m = (los_neigh[0,0]-los_neigh[1,0]) / (los_neigh[0,1]-los_neigh[1,1])

    los_mag = np.sqrt( (los[0,0]-los[1,0])**2 +  (los[0,1]-los[1,1])**2 )

    theta = np.arctan((los_m-los_neigh_m)/(1 + los_m*los_neigh_m))

    width = 2.* los_mag * np.sin(0.5*theta)

    return width, theta / 2.0

def rotate_los(origin, p2, angle):
    p2_rot = p2 - origin
    # p2 = p2[:,None]
    rot_mat = np.array(([[np.cos(angle), -1.0*np.sin(angle)],[np.sin(angle), np.cos(angle)]]))

    p2_rot = rot_mat.dot(p2_rot)
    p2_rot+=origin

    return p2_rot

def get_JETdefs(plot_defs = False, pulse_ref = 90531):

    wall_coords = np.array((
        [0,0],[0,0]
    ))

    JET = MachineDefs('JET', wall_coords, pulse_ref = pulse_ref)


    # DIAGNOSTIC LOS DEFINITIONS: [r1, z1], [r2, z2], [w1, w2] for each LOS

    ###############
    # KT3A
    ###############
    origin = [3.28422, 3.56166]
    width = 0.017226946
    p2 = np.array(([2.55361, -1.59557],
                   [2.57123, -1.60076],
                   [2.58885, -1.60595],
                   [2.60647, -1.61114],
                   [2.62408, -1.61633],
                   [2.64170, -1.62152],
                   [2.65932, -1.62671],
                   [2.67694, -1.63190],
                   [2.69456, -1.63709],
                   [2.71218, -1.64228],
                   [2.72979, -1.64747],
                   [2.74741, -1.65266],
                   [2.76503, -1.65785],
                   [2.78265, -1.66304],
                   [2.80027, -1.66823],
                   [2.81789, -1.67343],
                   [2.83551, -1.67862],
                   [2.85312, -1.68381],
                   [2.87074, -1.68900],
                   [2.88836, -1.69419],
                   [2.90598, -1.69938],
                   [2.92360, -1.70457]))
    kt3a_los = np.zeros((len(p2), 3, 2))
    for i in range(len(p2)):
        kt3a_los[i, 0] = origin
        kt3a_los[i, 1] = p2[i]
        kt3a_los[i, 2] = [0, width]
    los_dict = {}
    los_dict['p1'] = kt3a_los[:,0]
    los_dict['p2'] = kt3a_los[:,1]
    los_dict['w'] = kt3a_los[:,2]
    los_dict['id'] = []
    for id in range(len(kt3a_los)):
        los_dict['id'].append(str(id+1))
    JET.set_diag_los('KT3A', los_dict)

    ###############
    # KT1V
    ###############
    origin = [3.326, 3.807]
    p2 = np.array(([2.2787447, -1.31082201],
                   [2.30082607, -1.33440006],
                   [2.32782602, -1.33440006],
                   [2.35472608, -1.33440006],
                   [2.378052, -1.35385561],
                   [2.40063381, -1.37906277],
                   [2.41289091, -1.46512616],
                   [2.41917992, -1.59241283],
                   [2.42798877, -1.71315801],
                   [2.45742607, -1.70969999],
                   [2.48642612, -1.70969999],
                   [2.51542616, -1.70969999],
                   [2.55589104, -1.62998962],
                   [2.58804083, -1.60308468],
                   [2.61560178, -1.61040914],
                   [2.64301133, -1.6201551],
                   [2.67078233, -1.62739968],
                   [2.69836617, -1.63720798],
                   [2.72634029, -1.64494753],
                   [2.75428176, -1.65316343],
                   [2.78240013, -1.66152966],
                   [2.81055236, -1.66882849],
                   [2.83572602, -1.71150005],
                   [2.88960505, -1.41644132],
                   [2.92077351, -1.36994159],
                   [2.94975448, -1.34599328],
                   [2.97752619, -1.3348],
                   [3.00452614, -1.3348],
                   [3.02952623, -1.37039995],
                   [3.06067014, -1.29725635],
                   [3.08823347, -1.28003955]))
    kt1v_los = np.zeros((len(p2), 3, 2))
    kt1v_los_half_angle = np.zeros((len(p2)))
    # determine end point width using half angle between adjacent sight lines
    for i, los in enumerate(p2):
        if i > 0 and i < len(p2)-1:
            width, half_angle = los_width_from_neigbh(np.array((origin, p2[i])), np.array((origin, p2[i+1])))
            kt1v_los[i, 0] = origin
            kt1v_los[i, 1] = p2[i]
            kt1v_los[i, 2] = [0, width]
            kt1v_los_half_angle[i] = half_angle
    # end points
    kt1v_los[0, 0] = origin
    kt1v_los[0, 1] = p2[0]
    kt1v_los[0, 2] = kt1v_los[1, 2]
    kt1v_los_half_angle[0] = kt1v_los_half_angle[1]
    kt1v_los[len(p2)-1, 0] = origin
    kt1v_los[len(p2)-1, 1] = p2[len(p2)-1]
    kt1v_los[len(p2)-1, 2] = kt1v_los[len(p2)-2, 2]
    kt1v_los_half_angle[len(p2)-1] = kt1v_los_half_angle[len(p2)-2]
    los_dict = {}
    los_dict['p1'] = kt1v_los[:,0]
    los_dict['p2'] = kt1v_los[:,1]
    los_dict['w'] = kt1v_los[:,2]
    los_dict['id'] = []
    for id in range(len(kt1v_los)):
        los_dict['id'].append(str(id+1))
    los_dict['half_angle'] = kt1v_los_half_angle
    JET.set_diag_los('KT1V', los_dict)

    ###############
    # KB5 (including pulse dependent configurations)
    ###############
    #Default vs. re-configured sight line config
    # if JET.pulse_ref >=73758 and JET.pulse_ref <=82263:
    #     file = 'KB5_Bolometer_LOS_73758_82263.txt'
    # else:
    #     file = 'KB5_Bolometer_LOS_default.txt'
    # lines = loadtxt(file, comments="#", delimiter=" ", unpack=False)
    #
    # print('')

    if plot_defs:
        for i, los in enumerate(JET.diag_dict['KT3A']['id']):
            plt.plot([JET.diag_dict['KT3A']['p1'][i,0], JET.diag_dict['KT3A']['p2'][i, 0]],
                     [JET.diag_dict['KT3A']['p1'][i,1], JET.diag_dict['KT3A']['p2'][i, 1]],
                     '-k')
            plt.text(2.5,2, 'KT3A', color='black')
        for i, los in enumerate(JET.diag_dict['KT1V']['id']):
            plt.plot([JET.diag_dict['KT1V']['p1'][i,0], JET.diag_dict['KT1V']['p2'][i, 0]],
                     [JET.diag_dict['KT1V']['p1'][i,1], JET.diag_dict['KT1V']['p2'][i, 1]],
                     '-r')
            plt.text(2.5,2+0.2, 'KT1V', color='red')
            p2_rot = rotate_los(JET.diag_dict['KT1V']['p1'][i],
                                JET.diag_dict['KT1V']['p2'][i], kt1v_los_half_angle[i])
            plt.plot([JET.diag_dict['KT1V']['p1'][i,0], p2_rot[0]],
                     [JET.diag_dict['KT1V']['p1'][i,1], p2_rot[1]], ':r')
            p2_rot2 = rotate_los(JET.diag_dict['KT1V']['p1'][i],
                                 JET.diag_dict['KT1V']['p2'][i], -1.0*kt1v_los_half_angle[i])
            plt.plot([JET.diag_dict['KT1V']['p1'][i,0], p2_rot2[0]],
                     [JET.diag_dict['KT1V']['p1'][i,1], p2_rot2[1]], ':r')

            # p2_rot = rotate_los(los[0], los[1], kt1v_los_half_angle[i])
            # plt.plot([los[0, 0], p2_rot[0]], [los[0, 1], p2_rot[1]], ':r')
            # p2_rot2 = rotate_los(los[0], los[1], -1.0*kt1v_los_half_angle[i])
            # plt.plot([los[0, 0], p2_rot2[0]], [los[0, 1], p2_rot2[1]], ':r')
        plt.show()

    return JET


def get_DIIIDdefs():
    wall_coords = np.array((
        [], []
    ))

    DIIID = MachineDefs('DIIID', wall_coords)

    # define diagnostics
    # DIIID.set_diag_los('')

    return DIIID

if __name__=='__main__':

    JET = get_JETdefs(plot_defs = True, pulse_ref = 85000)

    print('')