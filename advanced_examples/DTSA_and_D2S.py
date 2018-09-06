# -*- coding: utf-8 -*-
########################################################################################################################
#
# Copyright (c) 2018, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#   disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
#    following disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################################################################


from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Any, Set, Dict, Optional, Union, List, Tuple

from bag.layout.routing import TrackManager, TrackID, WireArray
from bag.layout.template import TemplateDB
from bag.layout.util import BBox

from abs_templates_ec.analog_core import AnalogBase
from bag.layout.template import TemplateBase
import math

# import the classes of the generators that will be used
from advanced_examples.DoubleTailSenseAmplifier import DoubleTailSenseAmplifier
from advanced_examples.DynamicToStaticLatch import DynamicToStaticLatch


class DTSA_and_D2S(TemplateBase):
    """The assembled double tail sense amplifier (DTSA) and dynamic to static latch (D2S)

        Parameters
        ----------
        temp_db : TemplateDB
                the template database.
        lib_name : str
            the layout library name.
        params : Dict[str, Any]
            the parameter values.
        used_names : Set[str]
            a set of already used cell names.
        **kwargs
            dictionary of optional parameters.  See documentation of
            :class:`bag.layout.template.TemplateBase` for details.
        """
    def __init__(self, temp_db, lib_name, params, used_names, **kwargs):
        # type: (TemplateDB, str, Dict[str, Any], Set[str], **Any) -> None
        TemplateBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        return dict(
            top_layer='the top routing layer.',
            tr_widths='Track width dictionary.',
            tr_spaces='Track spacing dictionary.',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            DTSA_params='Layout parameters for the double tail sense amplifier',
            D2S_params='Layout parameters for the dynamic to static latch',
            show_pins='True to create pin labels.',
        )

    def place_instance_on_track(self,
                                instance,  # type: Union[TemplateBase, AnalogBase]
                                x_offset=0,  # type: Optional[Union[int, float]]
                                y_offset=0,  # type: Optional[Union[int, float]]
                                unit_mode=True,  # type: Optional[bool]
                                mode=0,  # type: Optional[Union[int, Tuple[int, int]]]
                                half_track=False,  # type: Optional[Union[bool, Tuple[bool, bool]]]
                                ):
        # type: (...) -> (Union[int, float], Union[int, float])
        """
        Find block location offsets on nearest track for x and y

        Parameters
        ----------
        instance : Union[TemplateBase, AnalogBase]
            the instance to add
        x_offset : Union[int, float]
            the x offset at which the instance should be added
        y_offset : Union[int, float]
            the y offset at which the instance should be added
        unit_mode : bool
            True if x and y offsets are given in resolution units
        mode : Union[int, Tuple[int, int]]
            The rounding mode for finding the nearest track to the provided offset.
            A single value specifies for both x and y rounding.
            A tuple can be passed to specify x and y rounding modes separately.

            If mode == 0, return the nearest track (default).
            If mode == -1, return the nearest track with coordinate less than or equal to coord.
            If mode == -2, return the nearest track with coordinate less than coord.
            If mode == 1, return the nearest track with coordinate greater than or equal to coord.
            If mode == 2, return the nearest track with coordinate greater than coord.
        half_track : Union[bool, Tuple[bool, bool]]
            True to allow instance to be placed on half-track grid.
            False to not allow half track positions (ie, that non-half-track wires in the instance will remain on
            non-half-tracks in the current level of hierarchy)
            A single value specifies for both x and y half-track mode.
            A tuple can be passed to specify x and y half-track modes separately.

            As the x and y axes are actually track -0.5 for any TemplateBase instance, True technically forces the
            origin onto half-track only coordinates.

        Returns
        -------
        x_ontrack, y_ontrack : Tuple[Union[int, float], Union[int, float]]
            the offset rounded to the appropriate on (half) track position at which the instance should be placed

        """

        # Do all calculations in unit mode
        res = self.grid.resolution
        if not unit_mode:
            x_offset = int(round(x_offset / res))
            y_offset = int(round(y_offset / res))

        block_top_layer = instance.top_layer
        top_layer_dir = self.grid.get_direction(block_top_layer)

        # Looks swapped because if top is 'x', then must align VERTICALLY based on top pitch
        if top_layer_dir == 'x':
            x_rounding_layer = block_top_layer - 1
            y_rounding_layer = block_top_layer
        else:
            x_rounding_layer = block_top_layer
            y_rounding_layer = block_top_layer - 1

        # If half_track is False, it means we must restrict actual block tracks to be on track,
        # which means offsets must be restricted to only half tracks (-0.5, 0.5, 1.5, etc), not whole tracks
        # as in-block tracks are already offset by half a track
        # Easiest way is to add half a pitch, call functions with half_track=False, then subtract
        if isinstance(half_track, tuple):
            if len(half_track) != 2:
                raise ValueError("half_track must either be a signle value or a length 2 tuple")
            x_half_track = half_track[0]
            y_half_track = half_track[1]
        elif isinstance(half_track, bool):
            x_half_track = half_track
            y_half_track = half_track
        else:
            raise ValueError("half_track must either be a single value or a length 2 tuple")

        if not x_half_track:
            x_offset += self.grid.get_track_pitch(x_rounding_layer, unit_mode=True)//2
            x_track_offset = -0.5
        else:
            x_track_offset = 0
        if not y_half_track:
            y_offset += self.grid.get_track_pitch(y_rounding_layer, unit_mode=True)//2
            y_track_offset = -0.5
        else:
            y_track_offset = 0

        # Handle seperate x and y modes
        if isinstance(mode, tuple):
            if len(mode) != 2:
                raise ValueError("mode must either be a single value or a length 2 tuple")
            x_mode = mode[0]
            y_mode = mode[1]
        elif isinstance(mode, int):
            x_mode = mode
            y_mode = mode
        else:
            raise ValueError("mode must either be a single value or a length 2 tuple")

        x_ontrack = self.grid.track_to_coord(
            x_rounding_layer,
            self.grid.coord_to_nearest_track(
                x_rounding_layer,
                x_offset,
                half_track=x_half_track,
                unit_mode=True,
                mode=x_mode
            ) + x_track_offset,
            unit_mode=True
        )
        y_ontrack = self.grid.track_to_coord(
            y_rounding_layer,
            self.grid.coord_to_nearest_track(
                y_rounding_layer,
                y_offset,
                half_track=y_half_track,
                unit_mode=True,
                mode=y_mode
            ) + y_track_offset,
            unit_mode=True
        )

        if not unit_mode:
            x_ontrack, y_ontrack = x_ontrack * res, y_ontrack * res

        return x_ontrack, y_ontrack

    def get_mos_conn_layer(self):
        # type: (...) -> int
        """
        Returns the layer on which mos vertical connections are made.

        Returns
        -------
        mos_conn_layer : int
            the mos_conn_layer for the current technology class
        """
        return self.grid.tech_info.tech_params['layout']['mos_tech_class'].get_mos_conn_layer()

    def draw_layout(self):
        # Get parameters from the specifications
        top_layer = self.params['top_layer']
        show_pins = self.params['show_pins']
        tr_widths = self.params['tr_widths']
        tr_spaces = self.params['tr_spaces']
        guard_ring_nf = self.params['guard_ring_nf']

        # Set up the sub-instance parameters. Some parametes are shared from the top-hierarchy to the sub-instances
        params_d2s = self.params['D2S_params']
        params_d2s['tr_widths'] = tr_widths
        params_d2s['tr_spaces'] = tr_spaces

        params_dtsa = self.params['DTSA_params']
        params_dtsa['tr_widths'] = tr_widths
        params_dtsa['tr_spaces'] = tr_spaces

        # Define the instance masters
        master_dtsa = self.new_template(params=params_dtsa, temp_cls=DoubleTailSenseAmplifier)
        master_d2s = self.new_template(params=params_d2s, temp_cls=DynamicToStaticLatch)

        # Define horizontal and vertical connection layers. Set up TrackManager
        horz_conn_layer = self.get_mos_conn_layer() + 1
        vert_conn_layer = self.get_mos_conn_layer() + 2
        power_grid_layer = self.get_mos_conn_layer() + 3
        tr_manager = TrackManager(grid=self.grid, tr_widths=tr_widths, tr_spaces=tr_spaces)

        # Calculate x placement of the DTSA and D2S based on which is wider. Center the smaller in the x direction.
        if master_dtsa.bound_box.width_unit > master_d2s.bound_box.width_unit:
            x_offset_unit_dtsa = 0
            x_offset_unit_d2s = (master_dtsa.bound_box.width_unit - master_d2s.bound_box.width_unit) // 2
            x_total = master_dtsa.bound_box.width_unit
        else:
            x_offset_unit_dtsa = (master_d2s.bound_box.width_unit - master_dtsa.bound_box.width_unit) // 2
            x_offset_unit_d2s = 0
            x_total = master_d2s.bound_box.width_unit

        # Place the DTSA:
        # Find the closest x and y offsets that will result in the instance being placed on-track
        x_dtsa, y_dtsa = self.place_instance_on_track(
            master_dtsa,
            x_offset=x_offset_unit_dtsa,
            y_offset=0,
            unit_mode=True, mode=(0, 0), half_track=False
        )
        # Instantiate the instance
        inst_dtsa = self.add_instance(
            master_dtsa, 'X_DTSA',
            loc=(x_dtsa, y_dtsa),
            orient='R0',
            unit_mode=True
        )

        # Place the D2S:  Note it will be flipped MX so we to account for that extra y offset
        x_d2s, y_d2s = self.place_instance_on_track(
            master_d2s,
            x_offset=x_offset_unit_d2s,
            y_offset=inst_dtsa.bound_box.top_unit + master_d2s.bound_box.height_unit,
            unit_mode=True, mode=(0, 0), half_track=False
        )
        inst_d2s = self.add_instance(
            master_d2s, 'X_D2S',
            loc=(x_d2s, y_d2s),
            orient='MX',
            unit_mode=True
        )

        # Set block width of the top level hierarchy
        [blk_w, blk_h] = self.grid.get_block_size(top_layer, unit_mode=True)
        right = math.ceil(x_total / blk_w) * blk_w
        top = math.ceil(inst_d2s.bound_box.top_unit / blk_h) * blk_h
        bound_box = BBox(
            left=0, bottom=0, right=right, top=top,
            resolution=self.grid.resolution, unit_mode=True
        )
        self.set_size_from_bound_box(top_layer, bound_box)

        # Get the output wires of the DTSA and the input wires of the D2S, and connect them
        warr_dtsa_vop_h = inst_dtsa.get_all_port_pins(name='VOP', layer=horz_conn_layer)[0]
        warr_dtsa_von_h = inst_dtsa.get_all_port_pins(name='VON', layer=horz_conn_layer)[0]

        warr_d2s_r_in_v = inst_d2s.get_all_port_pins(name='R', layer=vert_conn_layer)
        warr_d2s_s_in_v = inst_d2s.get_all_port_pins(name='S', layer=vert_conn_layer)

        self.connect_differential_wires(
            pin_warrs=warr_d2s_r_in_v,
            nin_warrs=warr_d2s_s_in_v,
            pout_warr=warr_dtsa_von_h,
            nout_warr=warr_dtsa_vop_h,
        )

        # Reexport the VIP, VIN, DIP DIN, Q, and Q_B pins
        self.reexport(port=inst_dtsa.get_port('VIP'), show=show_pins)
        self.reexport(port=inst_dtsa.get_port('VIN'), show=show_pins)
        self.reexport(port=inst_dtsa.get_port('DIP'), show=show_pins)
        self.reexport(port=inst_dtsa.get_port('DIN'), show=show_pins)
        self.reexport(port=inst_d2s.get_port('Q'), show=show_pins)
        self.reexport(port=inst_d2s.get_port('Q_B'), show=show_pins)
        self.reexport(port=inst_dtsa.get_port('CLK'), show=show_pins)
        self.reexport(port=inst_dtsa.get_port('CLK_B'), show=show_pins)

        # Create list of VSS and VDD wire arrays, from bottom to top of layout
        warr_vss = []
        warr_vdd = []

        warr_vss_list = []
        warr_vdd_list = []
        warr_vss_list.append(inst_dtsa.get_all_port_pins(name='VSS'))
        warr_vss_list.append(inst_d2s.get_all_port_pins(name='VSS'))
        warr_vdd_list.append(inst_dtsa.get_all_port_pins(name='VDD'))
        warr_vdd_list.append(inst_d2s.get_all_port_pins(name='VDD'))

        for warr_list in warr_vss_list:
            warr_vss.extend(warr_list)
        for warr_list in warr_vdd_list:
            warr_vdd.extend(warr_list)

        # Extend each horizontal power strap to the width of the block, minus some margin
        for i, warr in enumerate(warr_vss):
            warr_vss[i] = self.extend_wires(
                warr,
                lower=self.bound_box.left_unit + self.grid.get_track_pitch(vert_conn_layer, unit_mode=True),
                upper=self.bound_box.right_unit - self.grid.get_track_pitch(vert_conn_layer, unit_mode=True),
                unit_mode=True
            )[0]
        for i, warr in enumerate(warr_vdd):
            warr_vdd[i] = self.extend_wires(
                warr,
                lower=self.bound_box.left_unit + self.grid.get_track_pitch(vert_conn_layer, unit_mode=True),
                upper=self.bound_box.right_unit - self.grid.get_track_pitch(vert_conn_layer, unit_mode=True),
                unit_mode=True
            )[0]

        # Power grid space / width parameters
        vert_power_grid_width_tracks = 2
        vert_power_grid_space_tracks = 2
        horz_power_grid_width_tracks = 2
        horz_power_grid_space_tracks = 2

        warr_vdd, warr_vss = self.do_power_fill(
            layer_id=vert_conn_layer,
            space=2 * self.grid.get_track_pitch(vert_conn_layer, unit_mode=True),
            space_le=3 * self.grid.get_line_end_space(vert_conn_layer, vert_power_grid_width_tracks, unit_mode=True),
            vdd_warrs=warr_vdd,
            vss_warrs=warr_vss,
            fill_width=vert_power_grid_width_tracks,
            fill_space=vert_power_grid_space_tracks,
            x_margin=self.grid.get_track_pitch(vert_conn_layer, unit_mode=True),
            y_margin=self.grid.get_track_pitch(vert_conn_layer, unit_mode=True),
            unit_mode=True,
            min_len=self.grid.get_track_width(power_grid_layer, horz_power_grid_width_tracks, unit_mode=True)
        )

        warr_vdd, warr_vss = self.do_power_fill(
            layer_id=power_grid_layer,
            space=self.grid.get_track_pitch(power_grid_layer, unit_mode=True),
            space_le=3 * self.grid.get_line_end_space(power_grid_layer, horz_power_grid_width_tracks, unit_mode=True),
            vdd_warrs=warr_vdd,
            vss_warrs=warr_vss,
            fill_width=horz_power_grid_width_tracks,
            fill_space=horz_power_grid_space_tracks,
            x_margin=self.grid.get_track_pitch(power_grid_layer, unit_mode=True),
            y_margin=self.grid.get_track_pitch(power_grid_layer, unit_mode=True),
            unit_mode=True,
        )

        self.add_pin('VDD', warr_vdd, show=show_pins)
        self.add_pin('VSS', warr_vss, show=show_pins)

        # Assign schematic parameters
        self._sch_params = dict(
            DTSA_params=master_dtsa.sch_params,
            D2S_params=master_d2s.sch_params,
        )
