# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design.module import Module


# noinspection PyPep8Naming
class bag_advanced_examples__wrapper_DTSA_overdrive_tb(Module):
    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'wrapper_DTSA_overdrive_tb.yaml'))

    def __init__(self, database, parent=None, prj=None, **kwargs):
        Module.__init__(self, database, self.yaml_file, parent=parent, prj=prj, **kwargs)

    @classmethod
    def get_params_info(cls):
        # type: () -> Dict[str, str]
        return dict(
            dut_lib='DUT library name.',
            dut_cell='DUT cell name.',
            dut_conns='DUT connection dictionary.',
        )

    def design(self, dut_lib: str, dut_cell: str, dut_conns: Dict[str, str]) -> None:
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)

        for dut_pin, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', dut_pin, net_name)

