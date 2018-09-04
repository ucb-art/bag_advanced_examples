# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'DynamicToStaticLatch_Inverter.yaml'))


# noinspection PyPep8Naming
class bag_advanced_examples__DynamicToStaticLatch_Inverter(Module):
    """Module for library bag_advanced_examples cell DynamicToStaticLatch_Inverter.

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
            ('XINV1N', 'inv1_n'), ('XINV2N', 'inv2_n'),
            ('XINV1P', 'inv1_p'), ('XINV2P', 'inv2_p'),
        ]

        for sch_name, layout_name in tran_info_list:
            w = tx_info[layout_name]['w']
            th = tx_info[layout_name]['th']
            nf = tx_info[layout_name]['fg']
            self.instances[sch_name].design(w=w, l=lch, nf=nf, intent=th, )

        self.design_dummy_transistors(dum_info, 'XDUMMY', 'VDD', 'VSS')
