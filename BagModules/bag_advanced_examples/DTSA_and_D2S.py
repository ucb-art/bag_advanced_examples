# -*- coding: utf-8 -*-

from typing import Dict

import os
import pkg_resources

from bag.design import Module


yaml_file = pkg_resources.resource_filename(__name__, os.path.join('netlist_info', 'DTSA_and_D2S.yaml'))


# noinspection PyPep8Naming
class bag_advanced_examples__DTSA_and_D2S(Module):
    """Module for library bag_advanced_examples cell DTSA_and_D2S.

    Schematic generator for the double tail sense amplifier + dynamic to static latch that forms the analog to
    digital sampler.
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
            DTSA_params='Schematic parameters for the double tail sense amplifier.',
            D2S_params='Schematic parameters for the dynamic to static latch.',

        )

    def design(self, DTSA_params, D2S_params):
        # Call the design functions for the amplifier and latch
        self.instances['X_DTSA'].design(**DTSA_params)
        self.instances['X_D2S_LATCH'].design(**D2S_params)

        # The latch input pins will be renamed, and need to be reconnected to VON and VOP of the DTSA
        self.reconnect_instance_terminal('X_D2S_LATCH', 'R', 'VON')
        self.reconnect_instance_terminal('X_D2S_LATCH', 'S', 'VOP')
