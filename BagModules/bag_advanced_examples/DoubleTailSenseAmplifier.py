# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'DoubleTailSenseAmplifier.yaml'))


# noinspection PyPep8Naming
class bag_advanced_examples__DoubleTailSenseAmplifier(Module):
    """Module for library bag_advanced_examples cell DoubleTailSenseAmplifier.

    Fill in high level description here.
    """

    def __init__(self, bag_config, parent=None, prj=None, **kwargs):
        Module.__init__(self, bag_config, yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            lch='Length of transistors',
            tx_info='List of dictionaries of transistor information',
            dum_info='Dummy info',
        )

    def design(self, lch, tx_info, dum_info):
        # Define mapping from schematic transistors to layout transistor data structure
        tran_info_list = [
            ('X_IN_P', 'in_p'), ('X_IN_N', 'in_n'),
            ('X_TAIL_1', 'tail_1'), ('X_TAIL_2', 'tail_2'),
            ('X_LOAD_P', 'pre_load_p'), ('X_LOAD_N', 'pre_load_n'),
            ('X_INV_N_P', 'regen_n_p'), ('X_INV_N_N', 'regen_n_n'),
            ('X_INV_P_P', 'regen_p_p'), ('X_INV_P_N', 'regen_p_n'),
            ('X_RESET_P', 'reset_p'), ('X_RESET_N', 'reset_n'),
            ('X_TAIL_P_1', 'tail_p_1'), ('X_TAIL_P_2', 'tail_p_2'),
        ]

        for sch_name, layout_name in tran_info_list:
            w = tx_info[layout_name]['w']
            th = tx_info[layout_name]['th']
            nf = tx_info[layout_name]['fg']
            self.instances[sch_name].design(w=w, l=lch, nf=nf, intent=th, )

        self.design_dummy_transistors(dum_info, 'X_DUMMY', 'VDD', 'VSS')
