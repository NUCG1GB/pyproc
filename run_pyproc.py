from pyproc.analyse import PyprocAnalyse
import json, os, sys
import argparse
from collections import OrderedDict

def run_pyproc(input_dict_str):

    # spec_line_dict = {
    #     # Must include at least one spectral line from each impurity species in EDGE2D
    #
    #     '1': # HYDROGEN
    #         {'1': {'1215.2': ['2', '1'],
    #                '1025.3': ['3', '1'],
    #                '6561.9': ['3', '2'],
    #                '4860.6': ['4', '2'],
    #                '4339.9': ['5', '2'],
    #                '4101.2': ['6', '2'],
    #                '3969.5': ['7', '2'],},
    #          },
    #
    #     '4':  # BERYLLIUM
    #         {'2':{'5272.32': ['4s', '3p']}},
    #
    #     '7': # NITROGEN
    #         {'2':{'3996.13': ['4f', '3d'],
    #               '4042.07': ['4f', '3d'],
    #               '5002.18': ['3d', '3p'],
    #               '5005.86': ['3d', '3p']},
    #          '3':{'4100.51': ['3p', '3s']},
    #          '4':{'3481.83': ['3p', '3s']}
    #          }
    # }

    # spec_line_dict_lytrap = {
    #     # Must include at least one spectral line from each impurity species in EDGE2D
    #
    #     '1':  # HYDROGEN
    #         {
    #          '1': {'1215.67': ['2', '1'],  # opacity
    #                 '1025.72': ['3', '1'],
    #                 '6564.57': ['3', '2']},
    #          },
    #
    # }

    # input_dict = OrderedDict([
    #     ('machine','JET'),
    #     ('pulse', 90000),
    #     ('tranfile', '/work/bloman/cmg/catalog/edge2d/jet/81472/oct0817/seq#1/tran'),
    #     ('read_ADAS', False),
    #     ('read_ADAS_lytrap', {'read':True,
    #                           'pec_file': '/home/bloman/python_tools/pyADASread/adas_data/edge2d_opacity/pec16_h0.dat',
    #                          'adf11_dir': '/home/bloman/python_tools/pyADASread/adas_data/edge2d_opacity',
    #                          'adf11_year': 96,
    #                           'spec_line_dict': spec_line_dict_lytrap}),
    #     ('spec_line_dict', spec_line_dict),
    #     ('diag_list', ['KT3', 'KT1V', 'KB5V', 'KB5H']),
    #     ('interactive_plots', False),
    #     ('save_dir','/work/bloman/pyproc/'),
    #     ('run_options', {
    #         'calc_synth_spec_features': True,
    #         'analyse_synth_spec_features': True,
    #         'calc_NII_afg_feature': False}),
    # ])

    # with open('input_dict.json', mode='w', encoding='utf-8') as f:
    #     json.dump(input_dict, f, indent=2)

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

    PyprocAnalyse(input_dict)

if __name__=='__main__':

    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Run pyproc')
    parser.add_argument('input_dict')
    args = parser.parse_args()

    # Handle the input arguments
    input_dict_file = args.input_dict

    if os.path.isfile(input_dict_file):
        print('Found input dictionary: ', input_dict_file)
        run_pyproc(input_dict_file)
    else:
        sys.exit(input_dict_file + ' not found')