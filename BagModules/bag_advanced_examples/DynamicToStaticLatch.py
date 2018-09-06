# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'DynamicToStaticLatch.yaml'))


# noinspection PyPep8Naming
class bag_advanced_examples__DynamicToStaticLatch(Module):
    """Module for library bag_advanced_examples cell DynamicToStaticLatch.

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
            input_reset_high='True if inputs reset high (active low set/reset signals), '
                             'False if inputs reset low (active high set/reset signals)',
            lch='Length of transistors',
            tx_info='List of dictionaries of transistor information',
            dum_info='Dummy info',
        )

    def design(self, input_reset_high, lch, tx_info, dum_info):
        # Define mapping from schematic transistors to layout transistor data structure
        tran_info_list = [
            ('XL1N', 'latch1_n'), ('XL2N', 'latch2_n'),
            ('XL1P', 'latch1_p'), ('XL2P', 'latch2_p'),
            ('XL1NK1', 'latch1_nk1'), ('XL2NK1', 'latch2_nk1'),
            ('XL1NK2', 'latch1_nk2'), ('XL2NK2', 'latch2_nk2'),
            ('XL1PK1', 'latch1_pk1'), ('XL2PK1', 'latch2_pk1'),
            ('XL1PK2', 'latch1_pk2'), ('XL2PK2', 'latch2_pk2'),
            ('XINV1N', 'inv1_n'), ('XINV2N', 'inv2_n'),
            ('XINV1P', 'inv1_p'), ('XINV2P', 'inv2_p'),
            ('XOUTINV1_N', 'outinv1_n'), ('XOUTINV2_N', 'outinv2_n'),
            ('XOUTINV1_P', 'outinv1_p'), ('XOUTINV2_P', 'outinv2_p'),
        ]

        for sch_name, layout_name in tran_info_list:
            w = tx_info[layout_name]['w']
            th = tx_info[layout_name]['th']
            nf = tx_info[layout_name]['fg']
            self.instances[sch_name].design(w=w, l=lch, nf=nf, intent=th, )

        # If input_reset_high, then inputs are S_B and R_B
        if input_reset_high:
            self.rename_pin('S_IN', 'S_B')
            self.rename_pin('R_IN', 'R_B')

            self.reconnect_instance_terminal('XINV1N', 'G', 'R_B')
            self.reconnect_instance_terminal('XINV1P', 'G', 'R_B')
            self.reconnect_instance_terminal('XINV2N', 'G', 'S_B')
            self.reconnect_instance_terminal('XINV2P', 'G', 'S_B')
            self.reconnect_instance_terminal('XINV1N', 'D', 'R')
            self.reconnect_instance_terminal('XINV1P', 'D', 'R')
            self.reconnect_instance_terminal('XINV2N', 'D', 'S')
            self.reconnect_instance_terminal('XINV2P', 'D', 'S')
        # If input_reset_high == False, then inputs are S and R
        else:
            self.rename_pin('S_IN', 'S')
            self.rename_pin('R_IN', 'R')

            self.reconnect_instance_terminal('XINV1N', 'G', 'R')
            self.reconnect_instance_terminal('XINV1P', 'G', 'R')
            self.reconnect_instance_terminal('XINV2N', 'G', 'S')
            self.reconnect_instance_terminal('XINV2P', 'G', 'S')
            self.reconnect_instance_terminal('XINV1N', 'D', 'R_B')
            self.reconnect_instance_terminal('XINV1P', 'D', 'R_B')
            self.reconnect_instance_terminal('XINV2N', 'D', 'S_B')
            self.reconnect_instance_terminal('XINV2P', 'D', 'S_B')

        self.design_dummy_transistors(dum_info, 'XDUMMY', 'VDD', 'VSS')
