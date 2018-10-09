
from cherab.core.atomic import AtomicData
from cherab.openadas.rates import ImpactExcitationRate, RecombinationRate
from cherab.core.utility import PerCm3ToPerM3, PhotonToJ, Cm3ToM3

_excit_lookup_table = {
    (2, 1) : '1215.2excit',
    (3, 1) : '1025.3excit',
    (4, 1) : '972.1excit',
    (3, 2) : '6561.9excit',
    (4, 2) : '4860.6excit',
    (5, 2) : '4339.9excit',
    (6, 2) : '4101.2excit',
    (7, 2) : '3969.5excit'
}

_recom_lookup_table = {
    (2, 1) : '1215.2recom',
    (3, 1) : '1025.3recom',
    (4, 1) : '972.1recom',
    (3, 2) : '6561.9recom',
    (4, 2) : '4860.6recom',
    (5, 2) : '4339.9recom',
    (6, 2) : '4101.2recom',
    (7, 2) : '3969.5recom'
}


class PyprocAtomicData(AtomicData):

    def __init__(self, atomic_data_dict):

        self.atomic_data_dict = atomic_data_dict

    def wavelength(self, ion, ion_stage, transition):

        atomic_number_str = str(ion.atomic_number)
        ion_stage_str = str(ion_stage+1)
        trans_key = _excit_lookup_table[transition]

        # convert from A to nm!
        return self.atomic_data_dict['adf15'][atomic_number_str][ion_stage_str][trans_key].info['wavelength']/10.

    def impact_excitation_rate(self, ion, ion_stage, transition):

        atomic_number_str = str(ion.atomic_number)
        ion_stage_str = str(ion_stage+1)
        trans_key = _excit_lookup_table[transition]

        wavelength = self.wavelength(ion, ion_stage, transition)

        Te_arr = self.atomic_data_dict['adf15'][atomic_number_str][ion_stage_str][trans_key].Te_arr
        ne_arr = self.atomic_data_dict['adf15'][atomic_number_str][ion_stage_str][trans_key].ne_arr
        # Need to transport pec data to form pec(dens, temp)
        PEC = self.atomic_data_dict['adf15'][atomic_number_str][ion_stage_str][trans_key].pec.T

        rate_data = {
            "TE": Te_arr,  # eV
            'DENS': ne_arr,  # cm^3
            'PEC': PEC  # W.cm^3
        }

        pec_interpolator = ImpactExcitationRate(wavelength, rate_data)

        return pec_interpolator

    def recombination_rate(self, ion, ion_stage, transition):

        atomic_number_str = str(ion.atomic_number)
        ion_stage_str = str(ion_stage+1)
        trans_key = _recom_lookup_table[transition]

        wavelength = self.wavelength(ion, ion_stage, transition)

        Te_arr = self.atomic_data_dict['adf15'][atomic_number_str][ion_stage_str][trans_key].Te_arr
        ne_arr = self.atomic_data_dict['adf15'][atomic_number_str][ion_stage_str][trans_key].ne_arr
        # Need to transport pec data to form pec(dens, temp)
        PEC = self.atomic_data_dict['adf15'][atomic_number_str][ion_stage_str][trans_key].pec.T

        rate_data = {
            "TE": Te_arr,  # eV
            'DENS': ne_arr,  # cm^3
            'PEC': PEC  # cm^3 s^-1
        }

        pec_interpolator = RecombinationRate(wavelength, rate_data)

        return pec_interpolator
