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


"""
########################################################################################################################
    README

    This is the layout generator for Dynamic to Static Latch (D2S)

    This generator uses a particular coding style in an attempt to organize all the parameters and simplify the
    complexity of the schematic and layout generator codes.
    The style/use of row/transistor data structures are by no means a necessary style when using BAG
    In particular:
        - Transistor information (row, column location, size, orientation, etc) are stored in per-transistor
        dictionaries
        - Row information (name, orientation, width, threshold, etc) are stored in per-row dictionaries
        - Lists of the rows in this analogBase class are used to store information about rows
        - Many helper functions are defined to simplify the placement/alignment of transistors
        - Transistor ports ('s', 'd', 'g') are stored in the transistor dictionaries to make access more intuitive
########################################################################################################################
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# noinspection PyUnresolvedReferences,PyCompatibility
from builtins import *

from typing import Any, Set, Dict, Optional, Union, List

from bag.layout.routing import TrackManager, TrackID, WireArray
from bag.layout.template import TemplateDB

from abs_templates_ec.analog_core import AnalogBase
from bag.layout.template import TemplateBase
import math


class DynamicToStaticLatch(AnalogBase):
    """Latch for the dynamic to static converter

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
        AnalogBase.__init__(self, temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

        ################################################################################
        # Define global variables for holding transistor/row information
        ################################################################################
        self._global_rows = []
        self._global_nrows = []
        self._global_prows = []
        self._w_dict = None
        self._th_dict = None
        self._seg_dict = None
        self._ptap_w = None
        self._ntap_w = None
        self.wire_names = None

    @property
    def sch_params(self):
        # type: () -> Dict[str, Any]
        return self._sch_params

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            flip_out_sd=False,
            guard_ring_nf=0,
            top_layer=None,
            show_pins=True,
            make_tracks_extended=False,
        )

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            top_layer='the top routing layer.',
            tr_widths='Track width dictionary.',
            tr_spaces='Track spacing dictionary.',
            show_pins='True to create pin labels.',
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            seg_dict='NMOS/PMOS number of segments dictionary.',
            fg_dum='Number of single-sided edge dummy fingers.',
            input_reset_high='True if inputs reset high (active low set/reset signals), '
                             'False if inputs reset low (active high set/reset signals)',
            make_tracks_extended='Extend sig and sig_b tracks to span their total lengths. Cap vs match tradeoff.',
        )

    def get_row_index(self,
                      row,  # type: Dict
                      ):
        """
        Returns the index of the row within the nch or pch rows

        Parameters
        ----------
        row : Dict
            the row whose index should be returned

        Returns
        -------
        index : int
            the index of the row

        """
        # type: (...) -> int
        if row['type'] == 'nch':
            return self._global_nrows.index(row)
        else:
            return self._global_prows.index(row)

    def get_row_from_rowname(self,
                             rowname,  # type: str
                             ):
        """
        Returns the row dictionary cooresponding to the provided row name

        Parameters
        ----------
        rowname : str
            the name of the row whose information should be returned

        Returns
        -------
        output : Dict
            the row information, or None if the row cannot be found

        """
        # type: (...) -> Union[Dict, None]
        output = None
        for ind, row in enumerate(self._global_rows):
            if row['name'] == rowname:
                output = row
        return output

    def set_global_rows(self,
                        row_list,  # type: List[Dict]
                        ):
        # type: (...) -> None
        """
        Given an ordered list of rows (from bottom of design to top), sets up the global variables used for accessing
        row properties.

        Parameters
        ----------
        row_list : List
            The ordered list of row dictionaries. To abide by analogBase restrictions, all nch rows must come before
            pch rows

        Returns
        -------

        """
        self._global_rows = row_list
        n_ind = 0
        p_ind = 0
        for ind, row in enumerate(row_list):
            row['global_index'] = ind
            if row['type'] == 'nch':
                row['index'] = n_ind
                n_ind += 1
                self._global_nrows.append(row)
            elif row['type'] == 'pch':
                row['index'] = p_ind
                p_ind += 1
                self._global_prows.append(row)
            else:
                raise ValueError('Row type must be "nch" or "pch" to indicate MOS type')

    def initialize_rows(self,
                        row_name,  # type: str
                        orient,  # type: str
                        nch_or_pch,  # type: str
                        ):
        # type : (...) -> Dict[str, Union[float, str, int]]
        """
        Initializes a data structure to hold useful information for each row

        Parameters
        ----------
        row_name : str
            the name to give the row. Should match the python variable name storing the row dictionary
        orient : str
            either 'R0' or 'MX' to specify whether the row is oriented gate down or gate up
        nch_or_pch : str
            either 'nch' to specify an nch row or 'pch' to specify a pch row

        Returns
        -------

        """
        if row_name not in self._w_dict.keys():
            raise ValueError("row_name '{}' must appear in w_dict".format(row_name))
        if row_name not in self._th_dict.keys():
            raise ValueError("row_name '{}' must appear in th_dict".format(row_name))

        if not ((orient == 'R0') or (orient == 'MX')):
            raise ValueError("orient parameter must be 'R0' or 'MX' for transistor row {}".format(row_name))

        row = {
            'name': row_name,
            'width': self._w_dict[row_name],
            'th': self._th_dict[row_name],
            'orient': orient,
            'type': nch_or_pch
        }

        return row

    def _row_prop(self,
                  property_name,  # type: str
                  nch_or_pch  # type: str
                  ):
        # type: (...) -> List
        """
        Returns a list of the given property type for all rows of the nch or pch type specified.
        Useful helper function for draw_base

        Parameters
        ----------
        property_name : str
            The row property to be returned
        nch_or_pch : str
            either 'nch' or 'pch', to return the property for all nch rows or pch rows respectively

        Returns
        -------
        prop_list : List
            A list of the given property for all nch rows or pch rows

        """
        prop_list = []
        for row in self._global_rows:
            if row['type'] == nch_or_pch:
                prop_list += [row[property_name]]
        return prop_list

    def initialize_tx(self,
                      name,  # type: str
                      row,  # type: Dict
                      fg_spec,  # type: Union[str, int]
                      seff_net=None,  # type: Optional[str]
                      deff_net=None,  # type: Optional[str]
                      ):
        # type: (...) -> Dict[str, Union[str, int, float, Dict, WireArray]]
        """
        Initialize the transistor data structure.
        s_net and d_net are the source and drain nets of the transistor in the schematic. i.e. it is the effective
        source and drain connections, regardless of whether the source is drawn on the even or odd diffusion regions

        Parameters
        ----------
        name : str
            name of the transistor. Should match the python variable used to store the transistor dictionary
        row : Dict
            the row dictionary for the row this transistor will be placed on
        fg_spec : Union[str, int]
            Either)
                - the name of the dictionary key in the YAML spec file for number of fingers in this transistor
                - the (integer) number of fingers in this transistor
        seff_net : Optional[str]
            the name of the effective source net for dummy calculation.
        deff_net : Optional[str]
            the name of the effective drain net for dummy calculation.

        Returns
        -------

        """
        return {'name': name,
                'fg': fg_spec if isinstance(fg_spec, int) else self._seg_dict[fg_spec],
                'type': row['type'],
                'w': row['width'],
                'th': row['th'],
                'row_ind': self.get_row_index(row),
                'row': row,
                'seff_net': '' if seff_net is None else seff_net,
                'deff_net': '' if deff_net is None else deff_net,
                }

    @staticmethod
    def align_in_column(fg_col,  # type: int
                        fg_tx,  # type: int
                        align=0,  # type: int
                        ):
        # type: (...) -> int
        """
        Returns the column index offset to justify a transistor within a column/stack

        Parameters
        ----------
        fg_col : int
            width (in fingers) of the column
        fg_tx : int
            width (in fingers) of the transistor
        align : int
            how to align the transistor in the column

        Returns
        -------
        col : int
            the column index of the start of the transistor
        """
        if align == 0:
            col = (fg_col - fg_tx) // 2
        elif align == -1:
            col = 0
        elif align == 1:
            col = fg_col - fg_tx
        else:
            raise ValueError("align must be either -1, 0, or 1")

        return col

    def assign_tx_column(self,
                         tx,  # type: Dict
                         offset,  # type: int
                         fg_col=None,  # type: Optional[int]
                         align=0,  # type: int
                         ):
        # type: (...) -> None
        """
        Calculates and assigns the transistor's column index (the position of the leftmost finger of the transistor).
        If fg_col is passed, the transistor is assumed to be in some stack of transistors that should all be
        horizontally aligned within a given "column/stack". The offset in this case refers to the start of the column of
        transistors rather than the offset of the passed transistor

        Parameters
        ----------
        tx : Dict
            the transistor whose position is being assigned
        offset : int
            the offset (in column index) of the transistor, or the column (stack) the transistor is in if fg_col is
            not None
        fg_col : int
            the width in fingers of the column/stack in which the passed transistor should be aligned
        align : int
            How to align the transistor within the column/stack. 0 to center, 1 to right justify, -1 to left justify.

        Returns
        -------

        """

        if fg_col is None:
            fg_col = tx['fg']
        tx['col'] = offset + self.align_in_column(fg_col, tx['fg'], align)

    @staticmethod
    def set_tx_directions(tx,  # type: Dict
                          seff,  # type: str
                          seff_dir,  # type: int
                          deff_dir=None,  # type: Optional[int]
                          ):
        # type: (...) -> None
        """
        Sets the source/drain direction of the transistor. Sets the effective source/drain location
        BAG defaults source to be the left most diffusion, and then alternates source-drain-source-...
        seff specifies whether the transistors effective source should be in the 's' or 'd' position.
        seff_dir specifies whether the effective source's vertical connections will go up or down

        If not specified, drain location and direction are assigned automatically to be oppositve to the source

        Note: For mirrored rows, source and drain direction are flipped by BAG

        Parameters
        ----------
        tx : Dict
            the transistor to set
        seff : str
            's' or 'd' to indicate whether the transistor's effective source is the odd or even diffusion regions
        seff_dir : int
            0 to indicate connection goes down. 2 to indicate connection goes up. 1 to indicate middle connection
        deff_dir : Optional[int]
            0 to indicate connection goes down. 2 to indicate connection goes up. 1 to indicate middle connection


        Returns
        -------

        """
        if seff != 's' and seff != 'd':
            raise ValueError("seff must be either 's' or 'd' \n"
                             "Transistor    %s    violates this" % tx['name'])
        if not (seff_dir == 0 or seff_dir == 2 or seff_dir == 1):
            raise ValueError("seff_dir must be either 0, 1, or 2 \n"
                             "Transistor    %s    violates this" % tx['name'])
        if deff_dir is not None and not (deff_dir == 0 or deff_dir == 1 or deff_dir == 2):
            raise ValueError("deff_dir must be either 0, 1, or 2 \n"
                             "Transistor    %s    violates this" % tx['name'])
        tx['seff'] = seff
        tx['deff'] = 's' if seff == 'd' else 'd'
        tx['seff_dir'] = seff_dir
        if deff_dir is None:
            tx['deff_dir'] = 2 - seff_dir
        else:
            tx['deff_dir'] = deff_dir

        # Based on whether the source or drain is on the even diffusion regions, assign the sdir (corresponding
        #  to the even diffusion regions) and ddir (corresponding to odd diffusion regions)
        tx['sdir'] = tx['seff_dir'] if seff == 's' else tx['deff_dir']
        tx['ddir'] = tx['deff_dir'] if seff == 's' else tx['seff_dir']
        # Based on whether the source or drain is on the even diffusion regions, assign the s_net name (corresponding
        #  to the even diffusion regions) and d_net name (corresponding to odd diffusion regions)
        tx['s_net'] = tx['seff_net'] if seff == 's' else tx['deff_net']
        tx['d_net'] = tx['deff_net'] if seff == 's' else tx['seff_net']

    def assign_tx_matched_direction(self,
                                    target_tx,  # type: Dict
                                    source_tx,  # type: Dict
                                    seff_dir,  # type: int
                                    aligned=True,  # type: bool
                                    deff_dir=None,  # type: Optional[int]
                                    ):
        # type: (...) -> None
        """
        Align the source/drain position (is source or drain leftmost diffusion) to a reference transistor
        Also assign the source and drain connection directions (up/down)

        Parameters
        ----------
        target_tx : Dict
            the transistor to assign
        source_tx : Dict
            the reference transistor whose source/drain alignment should be matched
        seff_dir : int
            the connection direction of the effective source for the target transistor (0 for down, 2 for up,
            1 for middle)
        aligned : bool
            True to have the target transistor's s/d connection be aligned with the source transistor.
            False to have s_target align with d_source and d_target align with s_source
        deff_dir : Optional[int]
            if specified, sets the connection direction of the effective drain for the target transistor.
            If not specified, drain direction is assumed to be opposite to the source direction

        Returns
        -------

        """
        if (target_tx['fg'] - source_tx['fg']) % 4 == 0 and aligned is True:
            self.set_tx_directions(target_tx, source_tx['seff'], seff_dir, deff_dir)
        else:
            self.set_tx_directions(target_tx, source_tx['deff'], seff_dir, deff_dir)

    def get_tracks(self):
        # type: (...) -> Dict[str, Dict[str, List]]
        """
        Returns the number of tracks present in each row of the design. Useful in TemplateBase designs when two
        horizontally adjacent analogBase blocks must have rows/tracks align.

        Returns
        -------

        """
        nds = []
        ng = []
        pds = []
        pg = []
        norient = []
        porient = []
        for row in self._global_rows:
            if row['type'] == 'nch':
                nds.append(self.get_num_tracks('nch', row['index'], 'ds'))
                ng.append(self.get_num_tracks('nch', row['index'], 'g'))
                norient.append(row['orient'])

            elif row['type'] == 'pch':
                pds.append(self.get_num_tracks('pch', row['index'], 'ds'))
                pg.append(self.get_num_tracks('pch', row['index'], 'g'))
                porient.append(row['orient'])
            else:
                raise ValueError('row does not have type nch or pch')

        return dict(
            nch=dict(
                ds=nds,
                g=ng,
                orient=norient,
            ),
            pch=dict(
                ds=pds,
                g=pg,
                orient=porient,
            )
        )

    def get_wire_names(self):
        return self.wire_names

    def draw_layout(self):
        # Get parameters from the parameter dictionary
        top_layer = self.params['top_layer']
        tr_widths = self.params['tr_widths']
        tr_spaces = self.params['tr_spaces']
        show_pins = self.params['show_pins']
        lch = self.params['lch']

        self._ptap_w = self.params['ptap_w']
        self._ntap_w = self.params['ntap_w']
        self._w_dict = self.params['w_dict']
        self._th_dict = self.params['th_dict']
        self._seg_dict = self.params['seg_dict']
        fg_dum = self.params['fg_dum']

        input_reset_high = self.params['input_reset_high']
        make_tracks_extended = self.params['make_tracks_extended']

        # Define layers on which horizontal and vertical connections will be made, relative to mos_conn_layer
        horz_conn_layer = self.mos_conn_layer + 1
        vert_conn_layer = self.mos_conn_layer + 2

        # Initialize the TrackManager, and define what horizontal wire types will be needed in each row
        tr_manager = TrackManager(grid=self.grid, tr_widths=tr_widths, tr_spaces=tr_spaces)
        wire_names = dict(
            nch=[
                # nch row
                dict(
                    g=['sig', 'sig', 'sig'],
                    ds=['sig', 'sig', 'sig']
                ),
            ],
            pch=[
                # pch row
                dict(
                    g=['sig', 'sig', 'sig'],
                    ds=['sig']
                ),
            ]
        )

        # Set up row information
        row_nch = self.initialize_rows(row_name='nch_row', orient='R0', nch_or_pch='nch')
        row_pch = self.initialize_rows(row_name='pch_row', orient='R0', nch_or_pch='pch')

        # Define the order of the rows (bottom to top) for this analogBase cell
        self.set_global_rows(
            [row_nch, row_pch]
        )

        # Initialize the transistors
        latch1_n = self.initialize_tx(name='latch1_n', row=row_nch, fg_spec='latch_n', deff_net='QM')
        latch1_p = self.initialize_tx(name='latch1_p', row=row_pch, fg_spec='latch_p', deff_net='QM')
        latch1_nk1 = self.initialize_tx(name='latch1_nk1', row=row_nch, fg_spec='latch_nk1', deff_net='QM_CASCODE_N')
        latch1_nk2 = self.initialize_tx(name='latch1_nk2', row=row_nch, fg_spec='latch_nk2',
                                        seff_net='QM_CASCODE_N', deff_net='QM')
        latch1_pk1 = self.initialize_tx(name='latch1_pk1', row=row_pch, fg_spec='latch_pk1', deff_net='QM_CASCODE_P')
        latch1_pk2 = self.initialize_tx(name='latch1_pk2', row=row_pch, fg_spec='latch_pk2',
                                        seff_net='QM_CASCODE_P', deff_net='QM')
        latch2_n = self.initialize_tx(name='latch2_n', row=row_nch, fg_spec='latch_n', deff_net='QM_B')
        latch2_p = self.initialize_tx(name='latch2_p', row=row_pch, fg_spec='latch_p', deff_net='QM_B')
        latch2_nk1 = self.initialize_tx(name='latch2_nk1', row=row_nch, fg_spec='latch_nk1', deff_net='QM_B_CASCODE_N')
        latch2_nk2 = self.initialize_tx(name='latch2_nk2', row=row_nch, fg_spec='latch_nk2',
                                        seff_net='QM_B_CASCODE_N', deff_net='QM_B')
        latch2_pk1 = self.initialize_tx(name='latch2_pk1', row=row_pch, fg_spec='latch_pk1', deff_net='QM_B_CASCODE_P')
        latch2_pk2 = self.initialize_tx(name='latch2_pk2', row=row_pch, fg_spec='latch_pk2',
                                        seff_net='QM_B_CASCODE_P', deff_net='QM_B')

        inv1n = self.initialize_tx(name='inv1_n', row=row_nch, fg_spec='inv_n', deff_net='R_IN_B')
        inv1p = self.initialize_tx(name='inv1_p', row=row_pch, fg_spec='inv_p', deff_net='R_IN_B')
        inv2n = self.initialize_tx(name='inv2_n', row=row_nch, fg_spec='inv_n', deff_net='S_IN_B')
        inv2p = self.initialize_tx(name='inv2_p', row=row_pch, fg_spec='inv_p', deff_net='S_IN_B')

        transistors = [
            latch1_n, latch1_p, latch1_nk1, latch1_nk2, latch1_pk1, latch1_pk2,
            latch2_n, latch2_p, latch2_nk1, latch2_nk2, latch2_pk1, latch2_pk2,
            inv1n, inv1p, inv2n, inv2p
        ]

        # Check that all transistors are even fingered, for symmetry of layout
        for tx in transistors:
            if tx['fg'] % 2 == 1:
                raise ValueError(
                    "Transistors must have even number of fingers. Transistor '{}' has {}".format(tx['name'], tx['fg]'])
                )

        # Floorplan goes:
        # dum, l1k1, l1k2, l1, inv, inv, l2, l2k2, l2k1, dum
        # Calculate width of each column/stack of transistors
        fg_k1 = max(latch1_nk1['fg'], latch1_pk1['fg'])
        fg_k2 = max(latch1_nk2['fg'], latch1_pk2['fg'])
        fg_main = max(latch1_n['fg'], latch1_p['fg'])
        fg_inv = max(inv1n['fg'], inv1p['fg'])

        # Get tech - specific required space between transistors with different diffusion regions
        fg_space = self._tech_cls.get_min_fg_sep(lch)

        col_l1k1 = fg_dum
        col_l1k2 = col_l1k1 + fg_k1 + fg_space
        col_l1 = col_l1k2 + fg_k2 + fg_space
        col_inv1 = col_l1 + fg_main + fg_space
        col_inv2 = col_inv1 + fg_inv + fg_space
        col_l2 = col_inv2 + fg_inv + fg_space
        col_l2k2 = col_l2 + fg_main + fg_space
        col_l2k1 = col_l2k2 + fg_k2 + fg_space

        fg_total = col_l2k1 + fg_k1 + fg_dum

        # Assign the transistor positions
        self.assign_tx_column(tx=latch1_nk1, offset=col_l1k1, fg_col=fg_k1, align=0)
        self.assign_tx_column(tx=latch1_pk1, offset=col_l1k1, fg_col=fg_k1, align=0)
        self.assign_tx_column(tx=latch1_nk2, offset=col_l1k2, fg_col=fg_k2, align=0)
        self.assign_tx_column(tx=latch1_pk2, offset=col_l1k2, fg_col=fg_k2, align=0)
        self.assign_tx_column(tx=latch1_n, offset=col_l1, fg_col=fg_main, align=0)
        self.assign_tx_column(tx=latch1_p, offset=col_l1, fg_col=fg_main, align=0)
        self.assign_tx_column(tx=inv1n, offset=col_inv1, fg_col=fg_inv, align=0)
        self.assign_tx_column(tx=inv1p, offset=col_inv1, fg_col=fg_inv, align=0)

        self.assign_tx_column(tx=inv2p, offset=col_inv2, fg_col=fg_inv, align=0)
        self.assign_tx_column(tx=inv2n, offset=col_inv2, fg_col=fg_inv, align=0)
        self.assign_tx_column(tx=latch2_p, offset=col_l2, fg_col=fg_main, align=0)
        self.assign_tx_column(tx=latch2_n, offset=col_l2, fg_col=fg_main, align=0)
        self.assign_tx_column(tx=latch2_pk2, offset=col_l2k2, fg_col=fg_k2, align=0)
        self.assign_tx_column(tx=latch2_nk2, offset=col_l2k2, fg_col=fg_k2, align=0)
        self.assign_tx_column(tx=latch2_pk1, offset=col_l2k1, fg_col=fg_k1, align=0)
        self.assign_tx_column(tx=latch2_nk1, offset=col_l2k1, fg_col=fg_k1, align=0)

        # Assign the transistor s/d directions
        self.set_tx_directions(tx=latch1_nk1, seff='s', seff_dir=0)
        self.set_tx_directions(tx=latch1_nk2, seff='s', seff_dir=0)
        self.set_tx_directions(tx=latch1_n, seff='s', seff_dir=0)
        self.set_tx_directions(tx=inv1n, seff='d', seff_dir=0)
        self.set_tx_directions(tx=latch1_pk1, seff='s', seff_dir=2)
        self.set_tx_directions(tx=latch1_pk2, seff='s', seff_dir=2)
        self.set_tx_directions(tx=latch1_p, seff='s', seff_dir=2)
        self.set_tx_directions(tx=inv1p, seff='d', seff_dir=2)

        self.set_tx_directions(tx=latch2_nk1, seff='s', seff_dir=0)
        self.set_tx_directions(tx=latch2_nk2, seff='s', seff_dir=0)
        self.set_tx_directions(tx=latch2_n, seff='s', seff_dir=0)
        self.set_tx_directions(tx=inv2n, seff='d', seff_dir=0)
        self.set_tx_directions(tx=latch2_pk1, seff='s', seff_dir=2)
        self.set_tx_directions(tx=latch2_pk2, seff='s', seff_dir=2)
        self.set_tx_directions(tx=latch2_p, seff='s', seff_dir=2)
        self.set_tx_directions(tx=inv2p, seff='d', seff_dir=2)

        # Draw the transistor row bases
        self.draw_base(lch, fg_total, self._ptap_w, self._ntap_w,
                       self._row_prop('width', 'nch'), self._row_prop('th', 'nch'),
                       self._row_prop('width', 'pch'), self._row_prop('th', 'pch'),
                       tr_manager=tr_manager, wire_names=wire_names,
                       n_orientations=self._row_prop('orient', 'nch'),
                       p_orientations=self._row_prop('orient', 'pch'),
                       top_layer=top_layer,
                       half_blk_x=False, half_blk_y=False,
                       )

        # Draw the transistors
        for tx in transistors:
            tx['ports'] = self.draw_mos_conn(mos_type=tx['type'],
                                             row_idx=tx['row_ind'],
                                             col_idx=tx['col'],
                                             fg=tx['fg'],
                                             sdir=tx['sdir'],
                                             ddir=tx['ddir'],
                                             s_net=tx['s_net'],
                                             d_net=tx['d_net'],
                                             )
            tx['s'] = tx['ports'][tx['seff']]
            tx['d'] = tx['ports'][tx['deff']]
            tx['g'] = tx['ports']['g']

        # Create TrackIDs from the wire_names and TrackManager
        tid_s_in = self.get_wire_id('nch', row_nch['index'], 'g', wire_name='sig', wire_idx=0)
        tid_r_in = self.get_wire_id('nch', row_nch['index'], 'g', wire_name='sig', wire_idx=1)

        tid_qm = self.get_wire_id('nch', row_nch['index'], 'ds', wire_name='sig', wire_idx=1)
        tid_qm_b = self.get_wire_id('nch', row_nch['index'], 'ds', wire_name='sig', wire_idx=2)

        tid_cascode_n = self.get_wire_id('nch', row_nch['index'], 'ds', wire_name='sig', wire_idx=0)
        tid_cascode_p = self.get_wire_id('pch', row_pch['index'], 'ds', wire_name='sig', wire_idx=0)

        tid_s_in_b = self.get_wire_id('pch', row_pch['index'], 'g', wire_name='sig', wire_idx=0)
        tid_r_in_b = self.get_wire_id('pch', row_pch['index'], 'g', wire_name='sig', wire_idx=1)

        tid_nch_gate = self.get_wire_id('nch', row_nch['index'], 'g', wire_name='sig', wire_idx=2)
        tid_pch_gate = self.get_wire_id('pch', row_nch['index'], 'g', wire_name='sig', wire_idx=2)

        # Connect the cascode nodes:
        # warr_l1_n_cascode =
        self.connect_to_tracks(
            [latch1_nk1['d'], latch1_nk2['s']],
            tid_cascode_n,
            min_len_mode=0
        )
        # warr_l2_n_cascode =
        self.connect_to_tracks(
            [latch2_nk1['d'], latch2_nk2['s']],
            tid_cascode_n,
            min_len_mode=0
        )
        # warr_l1_p_cascode =
        self.connect_to_tracks(
            [latch1_pk1['d'], latch1_pk2['s']],
            tid_cascode_p,
            min_len_mode=0
        )
        # warr_l2_p_cascode =
        self.connect_to_tracks(
            [latch2_pk1['d'], latch2_pk2['s']],
            tid_cascode_p,
            min_len_mode=0
        )

        # Define connections for R_B and S_B wires.
        # These exclude the r_b: l2p/l1pk2 and  s_b: l1p/l2pk2 transistors, which change based on input_reset_high
        if input_reset_high:
            ports_r_in = [inv1n['g'], latch2_nk2['g']]
            ports_s_in = [inv2n['g'], latch1_nk2['g']]
        else:
            ports_r_in = [inv1n['g'], latch1_nk2['g']]
            ports_s_in = [inv2n['g'], latch2_nk2['g']]

        (warr_r_in, warr_s_in) = self.connect_differential_tracks(
            pwarr_list=ports_r_in, nwarr_list=ports_s_in,
            tr_layer_id=tid_r_in.layer_id, ptr_idx=tid_r_in.base_index, ntr_idx=tid_s_in.base_index,
            width=tr_manager.get_width(layer_id=tid_r_in.layer_id, track_type='sig'),
        )

        # Define the ports that QM and QM_B connect to
        # These exclude the nk1 transistors, which need a special connection, below
        ports_qm = [latch1_n['d'], latch1_nk2['d'], latch2_pk1['g'], latch1_p['d'], latch1_pk2['d']]
        ports_qm_b = [latch2_n['d'], latch2_nk2['d'], latch1_pk1['g'], latch2_p['d'], latch2_pk2['d']]

        (warr_qm, warr_qm_b) = self.connect_differential_tracks(
            pwarr_list=ports_qm, nwarr_list=ports_qm_b,
            tr_layer_id=tid_qm.layer_id, ptr_idx=tid_qm.base_index, ntr_idx=tid_qm_b.base_index,
            width=tr_manager.get_width(layer_id=tid_qm.layer_id, track_type='sig')
        )

        # Connect the latch*_nk1 gates to qm/qm_b, which require connections on vert_conn_layer
        # 1) Connect the gates to horizontal wires
        warr_l1_nk1_gate_h = self.connect_to_tracks(
            [latch1_nk1['g']],
            tid_nch_gate,
            min_len_mode=0
        )
        warr_l2_nk1_gate_h = self.connect_to_tracks(
            [latch2_nk1['g']],
            tid_nch_gate,
            min_len_mode=0
        )
        # 2) Define the vertical tracks on which vertical connection will be between the horizontal nk1 and qm wires
        tid_l1_nk1_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=self.grid.coord_to_nearest_track(
                layer_id=vert_conn_layer,
                coord=warr_l1_nk1_gate_h.middle_unit,
                mode=-1,
                unit_mode=True
            ),
            width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
        )
        tid_l2_nk1_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=self.grid.coord_to_nearest_track(
                layer_id=vert_conn_layer,
                coord=warr_l2_nk1_gate_h.middle_unit,
                mode=1,
                unit_mode=True
            ),
            width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
        )
        # 3) Perform the connection. For symmetry, extend the wires to the top of the higher qm / qm_b track
        self.connect_to_tracks(
            [warr_l1_nk1_gate_h, warr_qm_b],
            tid_l1_nk1_vert,
            min_len_mode=0,
            track_upper=max([tid.get_bounds(self.grid, unit_mode=True)[1] for tid in [tid_qm_b, tid_qm]]),
            unit_mode=True
        )
        self.connect_to_tracks(
            [warr_l2_nk1_gate_h, warr_qm],
            tid_l2_nk1_vert,
            min_len_mode=0,
            track_upper=max([tid.get_bounds(self.grid, unit_mode=True)[1] for tid in [tid_qm_b, tid_qm]]),
            unit_mode=True
        )

        # Define connections for R_B and S_B wires.
        # These exclude the R_B: l1n/l2nk2   and  S_B: l2n/l1nk2  gates, which need special connections, and change
        # based on input_reset_high
        if input_reset_high:
            ports_r_in_b = [inv1n['d'], inv1p['d'], latch1_pk2['g']]
            ports_s_in_b = [inv2n['d'], inv2p['d'], latch2_pk2['g']]
        else:
            ports_r_in_b = [inv1n['d'], inv1p['d'], latch2_p['g']]
            ports_s_in_b = [inv2n['d'], inv2p['d'], latch1_p['g']]

        warr_r_in_b, warr_s_in_b = self.connect_differential_tracks(
            pwarr_list=ports_r_in_b, nwarr_list=ports_s_in_b,
            tr_layer_id=tid_r_in_b.layer_id, ptr_idx=tid_r_in_b.base_index, ntr_idx=tid_s_in_b.base_index,
            width=tid_r_in_b.width
        )

        # Connect pmos gates of inverters to input signals
        # 3 steps: 1) Connect gates to horizontal track. 2) Define vertical connection track. 3) Connect vertically,
        # extending lower coordinate for symmetry
        warr_inv1p_g_h = self.connect_to_tracks(
            [inv1p['g']],
            tid_pch_gate,
            min_len_mode=0
        )
        warr_inv2p_g_h = self.connect_to_tracks(
            [inv2p['g']],
            tid_pch_gate,
            min_len_mode=0
        )
        tid_inv1p_g_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=self.grid.coord_to_nearest_track(
                layer_id=vert_conn_layer,
                coord=warr_inv1p_g_h.middle_unit,
                mode=-1,
                unit_mode=True
            ),
            width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
        )
        tid_inv2p_g_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=self.grid.coord_to_nearest_track(
                layer_id=vert_conn_layer,
                coord=warr_inv2p_g_h.middle_unit,
                mode=1,
                unit_mode=True
            ),
            width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
        )
        self.connect_to_tracks(
            [warr_inv1p_g_h, warr_r_in],
            tid_inv1p_g_vert,
            min_len_mode=0,
            track_lower=min([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in, tid_s_in]]),
            unit_mode=True
        )
        self.connect_to_tracks(
            [warr_inv2p_g_h, warr_s_in],
            tid_inv2p_g_vert,
            min_len_mode=0,
            track_lower=min([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in, tid_s_in]]),
            unit_mode=True
        )

        if input_reset_high:
            # L1P_g to s_in;  L2P_g to r_in
            # 1) Connect gates to horizontal track
            warr_l1p_g_h = self.connect_to_tracks(
                [latch1_p['g']], tid_pch_gate, min_len_mode=0
            )
            warr_l2p_g_h = self.connect_to_tracks(
                [latch2_p['g']], tid_pch_gate, min_len_mode=0
            )
            # 2) Define vertical track
            tid_l1p_vert = TrackID(
                layer_id=vert_conn_layer,
                track_idx=self.grid.coord_to_nearest_track(
                    layer_id=vert_conn_layer,
                    coord=warr_l1p_g_h.middle_unit,
                    mode=-1,
                    unit_mode=True
                ),
                width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
            )
            tid_l2p_vert = TrackID(
                layer_id=vert_conn_layer,
                track_idx=self.grid.coord_to_nearest_track(
                    layer_id=vert_conn_layer,
                    coord=warr_l2p_g_h.middle_unit,
                    mode=1,
                    unit_mode=True
                ),
                width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
            )
            # 3) Connect vertically
            self.connect_to_tracks(
                [warr_s_in, warr_l1p_g_h],
                tid_l1p_vert,
                min_len_mode=0,
                track_lower=min([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in, tid_s_in]]),
                unit_mode=True
            )
            self.connect_to_tracks(
                [warr_r_in, warr_l2p_g_h],
                tid_l2p_vert,
                min_len_mode=0,
                track_lower=min([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in, tid_s_in]]),
                unit_mode=True
            )

            # l1n to r_in_b, l2n to s_in_b
            # 1) Connect gates to horizontal track
            warr_l1n_g_h = self.connect_to_tracks(
                [latch1_n['g']], tid_nch_gate, min_len_mode=0
            )
            warr_l2n_g_h = self.connect_to_tracks(
                [latch2_n['g']], tid_nch_gate, min_len_mode=0
            )
            # 2) Define vertical track
            tid_l1n_vert = TrackID(
                layer_id=vert_conn_layer,
                track_idx=self.grid.coord_to_nearest_track(
                    layer_id=vert_conn_layer,
                    coord=warr_l1n_g_h.middle_unit,
                    mode=-1,
                    unit_mode=True
                ),
                width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
            )
            tid_l2n_vert = TrackID(
                layer_id=vert_conn_layer,
                track_idx=self.grid.coord_to_nearest_track(
                    layer_id=vert_conn_layer,
                    coord=warr_l2n_g_h.middle_unit,
                    mode=1,
                    unit_mode=True
                ),
                width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
            )
            # 3) Connect vertically
            self.connect_to_tracks(
                [warr_r_in_b, warr_l1n_g_h],
                tid_l1n_vert,
                min_len_mode=0,
                track_upper=max([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in_b, tid_s_in_b]]),
                unit_mode=True
            )
            self.connect_to_tracks(
                [warr_s_in_b, warr_l2n_g_h],
                tid_l2n_vert,
                min_len_mode=0,
                track_upper=max([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in_b, tid_s_in_b]]),
                unit_mode=True
            )
        else:
            # l1pk2 to rin, l2pk2 to sin
            # 1) Connect gates to horizontal track
            warr_l1pk2_g_h = self.connect_to_tracks(
                [latch1_pk2['g']], tid_pch_gate, min_len_mode=0
            )
            warr_l2pk2_g_h = self.connect_to_tracks(
                [latch2_pk2['g']], tid_pch_gate, min_len_mode=0
            )
            # 2) Define vertical track
            tid_l1p_vert = TrackID(
                layer_id=vert_conn_layer,
                track_idx=self.grid.coord_to_nearest_track(
                    layer_id=vert_conn_layer,
                    coord=warr_l1pk2_g_h.middle_unit,
                    mode=-1,
                    unit_mode=True
                ),
                width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
            )
            tid_l2p_vert = TrackID(
                layer_id=vert_conn_layer,
                track_idx=self.grid.coord_to_nearest_track(
                    layer_id=vert_conn_layer,
                    coord=warr_l2pk2_g_h.middle_unit,
                    mode=1,
                    unit_mode=True
                ),
                width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
            )
            # 3) Connect vertically
            self.connect_to_tracks(
                [warr_r_in, warr_l1pk2_g_h],
                tid_l1p_vert,
                min_len_mode=0,
                track_lower=min([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in, tid_s_in]]),
                unit_mode=True
            )
            self.connect_to_tracks(
                [warr_s_in, warr_l2pk2_g_h],
                tid_l2p_vert,
                min_len_mode=0,
                track_lower=min([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in, tid_s_in]]),
                unit_mode=True
            )

            # l1nk2 to s_in_b, l2nk2 to r_in_b
            # 1) Connect gates to horizontal track
            warr_l1nk2_g_h = self.connect_to_tracks(
                [latch1_nk2['g']], tid_nch_gate, min_len_mode=0
            )
            warr_l2nk2_g_h = self.connect_to_tracks(
                [latch2_nk2['g']], tid_nch_gate, min_len_mode=0
            )
            # 2) Define vertical track
            tid_l1n_vert = TrackID(
                layer_id=vert_conn_layer,
                track_idx=self.grid.coord_to_nearest_track(
                    layer_id=vert_conn_layer,
                    coord=warr_l1nk2_g_h.middle_unit,
                    mode=-1,
                    unit_mode=True
                ),
                width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
            )
            tid_l2n_vert = TrackID(
                layer_id=vert_conn_layer,
                track_idx=self.grid.coord_to_nearest_track(
                    layer_id=vert_conn_layer,
                    coord=warr_l2nk2_g_h.middle_unit,
                    mode=1,
                    unit_mode=True
                ),
                width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='sig')
            )
            # 3) Connect vertically
            self.connect_to_tracks(
                [warr_s_in_b, warr_l1nk2_g_h],
                tid_l1n_vert,
                min_len_mode=0,
                track_upper=max([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in_b, tid_s_in_b]]),
                unit_mode=True
            )
            self.connect_to_tracks(
                [warr_r_in_b, warr_l2nk2_g_h],
                tid_l2n_vert,
                min_len_mode=0,
                track_upper=max([tid.get_bounds(self.grid, unit_mode=True)[0] for tid in [tid_r_in_b, tid_s_in_b]]),
                unit_mode=True
            )

        self.add_pin('R_IN', warr_r_in)
        self.add_pin('S_IN', warr_s_in)
        self.add_pin('R_IN_B', warr_r_in_b)
        self.add_pin('S_IN_B', warr_s_in_b)
        self.add_pin('QM', warr_qm),
        self.add_pin('QM_B', warr_qm_b)

        self.connect_to_substrate(
            'ptap',
            [latch1_n['s'], latch1_nk1['s'], inv1n['s'], latch2_n['s'], latch2_nk1['s'], inv2n['s']]
        )

        self.connect_to_substrate(
            'ntap',
            [latch1_p['s'], latch1_pk1['s'], inv1p['s'], latch2_p['s'], latch2_pk1['s'], inv2p['s']]
        )

        warr_vss, warr_vdd = self.fill_dummy()

        if input_reset_high:
            # input_reset_high = True --> inputs are R_B and S_B (active low).

            self.add_pin('R_B', warr_r_in, show=show_pins)
            self.add_pin('S_B', warr_s_in, show=show_pins)
        else:
            # input_reset_high = False --> inputs are R and S (active high)
            self.add_pin('R', warr_r_in, show=show_pins)
            self.add_pin('S', warr_s_in, show=show_pins)
        self.add_pin('QM', warr_qm, show=show_pins)
        self.add_pin('QM_B', warr_qm_b, show=show_pins)
        self.add_pin('VSS', warr_vss, show=show_pins)
        self.add_pin('VDD', warr_vdd, show=show_pins)

        # Define transistor properties for schematic
        tx_info = {}
        for tx in transistors:
            tx_info[tx['name']] = {}
            tx_info[tx['name']]['w'] = tx['w']
            tx_info[tx['name']]['th'] = tx['th']
            tx_info[tx['name']]['fg'] = tx['fg']

        self._sch_params = dict(
            input_reset_high=input_reset_high,
            lch=lch,
            dum_info=self.get_sch_dummy_info(),
            tx_info=tx_info,
        )

        '''
        

        # If inputs are reset high (active low)
        # Connect the LxPK2 gates on the inner side track
        # Connect the LxNk2 gates on the outer side track
        if input_reset_high is False:
        

            # Extend r_b and s_b to symmetric widths
            if make_tracks_extended:
                r_in_warr = self.add_wires(
                    r_in_warr.layer_id, r_in_warr.track_id.base_index,
                    lower=self.grid.track_to_coord(l1pk2_verttrack.layer_id, 
                                                   l1pk2_verttrack.base_index, unit_mode=True),
                    upper=self.grid.track_to_coord(l2pk2_verttrack.layer_id, 
                                                   l2pk2_verttrack.base_index, unit_mode=True),
                    unit_mode=True)
                s_in_warr = self.add_wires(
                    s_in_warr.layer_id, s_in_warr.track_id.base_index,
                    lower=self.grid.track_to_coord(l1pk2_verttrack.layer_id, 
                                                   l1pk2_verttrack.base_index, unit_mode=True),
                    upper=self.grid.track_to_coord(l2pk2_verttrack.layer_id, 
                                                   l2pk2_verttrack.base_index, unit_mode=True),
                    unit_mode=True)

        '''


class D2S_output_inv(AnalogBase):
    """Output inverters for the D2S latch

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
        super(D2S_output_inv, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        # type: () -> Dict[str, Any]
        return self._sch_params

    @classmethod
    def get_default_param_values(cls):
        # type: () -> Dict[str, Any]
        """Returns a dictionary containing default parameter values.

        Override this method to define default parameter values.  As good practice,
        you should avoid defining default values for technology-dependent parameters
        (such as channel length, transistor width, etc.), but only define default
        values for technology-independent parameters (such as number of tracks).

        Returns
        -------
        default_params : dict[str, any]
            dictionary of default parameter values.
        """
        return dict(
            flip_out_sd=False,
            guard_ring_nf=0,
            top_layer=None,
            show_pins=True,
        )

    @classmethod
    def get_params_info(cls):
        """Returns a dictionary containing parameter descriptions.

        Override this method to return a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : dict[str, str]
            dictionary from parameter name to description.
        """
        return dict(
            lch='channel length, in meters.',
            ptap_w='NMOS substrate width, in meters/number of fins.',
            ntap_w='PMOS substrate width, in meters/number of fins.',
            w_dict='NMOS/PMOS width dictionary.',
            th_dict='NMOS/PMOS threshold flavor dictionary.',
            seg_dict='NMOS/PMOS number of segments dictionary.',
            fg_dum='Number of single-sided edge dummy fingers.',
            show_pins='True to create pin labels.',
        )

    def draw_layout(self):
        lch = self.params['lch']
        ptap_w = self.params['ptap_w']
        ntap_w = self.params['ntap_w']
        w_dict = self.params['w_dict']
        th_dict = self.params['th_dict']
        seg_dict = self.params['seg_dict']
        fg_dum = self.params['fg_dum']
        show_pins = self.params['show_pins']

        fg_outputinv_n = seg_dict['outputinv_n']
        fg_outputinv_p = seg_dict['outputinv_p']


        if (fg_outputinv_n%2 or fg_outputinv_p%2):
            print("WARNING:\n    For best matching, all number of fingers should be even")


        # Compute fingers per row

        # Corresponding n and p transistors should be in the same column,
        # so need to take the max of each of them
        fg_outputinv = max(fg_outputinv_n, fg_outputinv_p)

        fg_half = fg_outputinv
        # + 2 for gap in the middle
        fg_tot = (fg_half + fg_dum) * 2 + 2

        # specify widths and thresholds of each row
        nw_list = [w_dict['output_nmos']]
        pw_list = [w_dict['output_pmos']]
        nth_list = [th_dict['output_nmos']]
        pth_list  = [th_dict['output_pmos']]

        # specify the horizontal tracks for each row
        # Here, ng_tracks=[1,3] means 1st nmos row has 1 track 2nd pmos row has 3 tracks
        ng_tracks = [1]
        nds_tracks = [1]
        pds_tracks = [1]
        pg_tracks = [1]
        n_orient = ['MX']
        p_orient = ['R0']

        self.draw_base(lch, fg_tot, ptap_w, ntap_w, nw_list,
                       nth_list, pw_list, pth_list,
                       ng_tracks=ng_tracks, nds_tracks=nds_tracks,
                       pg_tracks=pg_tracks, pds_tracks=pds_tracks,
                       n_orientations=n_orient, p_orientations=p_orient,
                       )
        # figure out drain or source connection order (ie even or odd)

        #if (fg_invnmos - fg_invpmos) % 4 == 0:
        #    aout, aoutb, nsdir, nddir = 'd', 's', 0, 2
        #else:
        #    aout, aoutb, nsdir, nddir = 's', 'd', 2, 0
        fg_outputinv_offset = fg_outputinv_n - fg_outputinv_p

        # Determine FET column locations
        # Base columns for n-p pair
        outputinv_col = fg_dum
        outputinv_n_col = outputinv_col + (0 if (fg_outputinv_offset>0) else abs(fg_outputinv_offset)//2)
        outputinv_p_col = outputinv_col + (0 if (fg_outputinv_offset<0) else abs(fg_outputinv_offset)//2)


        # If fingers are odd vs even, need to flip the right half of the cell
        # lnk1
        (inv1n_seff, inv1n_deff, inv1n_sdir, inv1n_ddir) = ('s', 'd', 0, 2)
        if (fg_outputinv_n % 2 == 0):
            (inv2n_seff, inv2n_deff, inv2n_sdir, inv2n_ddir) = (inv1n_seff, inv1n_deff, inv1n_sdir, inv1n_ddir)
        else:
            (inv2n_seff, inv2n_deff, inv2n_sdir, inv2n_ddir) = ('d' if inv1n_seff == 's' else 's', 's' if inv1n_deff == 'd' else 'd', 2 - inv1n_sdir, 2 - inv1n_ddir)

        # invp - Direction NOT flipped to make gate connection easier
        if fg_outputinv_offset % 4 != 2:
            (inv1p_seff, inv1p_deff, inv1p_sdir, inv1p_ddir) = ('s', 'd', 2, 0)
        else:
            (inv1p_seff, inv1p_deff, inv1p_sdir, inv1p_ddir) = ('d', 's', 0, 2)
        if fg_outputinv_p % 2 == 0:
            (inv2p_seff, inv2p_deff, inv2p_sdir, inv2p_ddir) = (inv1p_seff, inv1p_deff, inv1p_sdir, inv1p_ddir)
        else:
            (inv2p_seff, inv2p_deff, inv2p_sdir, inv2p_ddir) = ('d' if inv1p_seff == 's' else 's', 's' if inv1p_deff == 'd' else 'd', 2 - inv1p_sdir, 2 - inv1p_ddir)

        # Create Transistors
        outputinv1_n_ports = self.draw_mos_conn('nch', 0, outputinv_n_col, fg_outputinv_n, inv1n_sdir, inv1n_ddir)

        outputinv2_n_ports = self.draw_mos_conn('nch', 0, fg_tot - outputinv_n_col - fg_outputinv_n, fg_outputinv_n, inv2n_sdir, inv2n_ddir)

        outputinv1_p_ports = self.draw_mos_conn('pch', 0, outputinv_p_col, fg_outputinv_p, inv1p_sdir, inv1p_ddir)

        outputinv2_p_ports = self.draw_mos_conn('pch', 0, fg_tot - outputinv_p_col - fg_outputinv_p, fg_outputinv_p, inv2p_sdir, inv2p_ddir)

        in_track = self.make_track_id('pch', 0, 'g', 0)
        out_track = self.make_track_id('nch', 0, 'ds', 0)

        in1_warr = self.connect_to_tracks(
            [outputinv1_n_ports['g'], outputinv1_p_ports['g']],
            in_track, min_len_mode=0)
        in2_warr = self.connect_to_tracks(
            [outputinv2_n_ports['g'], outputinv2_p_ports['g']],
            in_track, min_len_mode=0)

        out1_warr = self.connect_to_tracks(
            [outputinv1_n_ports[inv1n_deff], outputinv1_p_ports[inv1p_deff]],
            out_track,
            min_len_mode=0
        )
        out2_warr = self.connect_to_tracks(
            [outputinv2_n_ports[inv2n_deff], outputinv2_p_ports[inv2p_deff]],
            out_track,
            min_len_mode=0
        )

        self.connect_to_substrate('ptap',
                                  [outputinv1_n_ports[inv1n_seff], outputinv2_n_ports[inv2n_seff]])
        self.connect_to_substrate('ntap',
                                  [outputinv1_p_ports[inv1p_seff], outputinv2_p_ports[inv2p_seff]])

        vss_warrs, vdd_warrs = self.fill_dummy()

        self.add_pin('VSS', vss_warrs, show=show_pins)
        self.add_pin('VDD', vdd_warrs, show=show_pins)
        self.add_pin('QM_B', in1_warr, show=show_pins)
        self.add_pin('QM', in2_warr, show=show_pins)
        self.add_pin('Q', out1_warr, show=show_pins)
        self.add_pin('Q_B', out2_warr, show=show_pins)

        # compute schematic parameters
        self._sch_params = dict(
            lch=lch,
            w_dict=w_dict.copy(),
            th_dict=th_dict.copy(),
            seg_dict=seg_dict.copy(),
            dum_info=self.get_sch_dummy_info(),
        )


class D2S(TemplateBase):
    """The assembled latch and output inverter for the D2S

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
        super(D2S, self).__init__(temp_db, lib_name, params, used_names, **kwargs)
        self._sch_params = None

    @property
    def sch_params(self):
        return self._sch_params

    @classmethod
    def get_params_info(cls):
        return dict(
            latch_params='Latch sizing parameters',
            outputinv_params='Output inverter sizing parameters',
            guard_ring_nf='Width of the guard ring, in number of fingers.  0 to disable guard ring.',
            top_layer='the top routing layer.',
            tr_widths='Track width dictionary.',
            tr_spaces='Track spacing dictionary.',
            show_pins='True to create pin labels.',
            input_reset_high='Are inputs reset high (active low set/reset) = True, or reset low (active high set/reset) =False',
        )

    def draw_layout(self):
        # cell_params = self.params['params'].copy()
        latch_params = self.params['latch_params']
        outputinv_params = self.params['outputinv_params']
        show_pins = self.params['show_pins']
        latch_params['show_pins'] = show_pins
        latch_params['input_reset_high'] = self.params['input_reset_high']
        outputinv_params['show_pins'] = show_pins
        input_reset_high = self.params['input_reset_high']
        top_layer = self.params['top_layer']

        # Define the instance masters
        latch_master = self.new_template(params=latch_params, temp_cls=D2S_latch)
        outputinv_master = self.new_template(params=outputinv_params, temp_cls=D2S_output_inv)

        # Instances are known to have top layer be horizontal
        top_horz_layer = latch_master.top_layer
        top_vert_layer = top_horz_layer+1

        # Derive where to place the latch and output inverter
        # Center the components so that center coordinate is on grid of top_layer
        x_mid = max(latch_master.bound_box.width_unit, outputinv_master.bound_box.width_unit) /2

        x_latch = self.grid.track_to_coord(
            top_vert_layer,
            self.grid.coord_to_nearest_track(
                top_vert_layer, x_mid - latch_master.bound_box.width_unit/2, unit_mode=True
            ),
            unit_mode=True
        )
        x_outputinv = self.grid.track_to_coord(
            top_vert_layer,
            self.grid.coord_to_nearest_track(
                top_vert_layer, x_mid - outputinv_master.bound_box.width_unit/2, unit_mode=True
            ),
            unit_mode=True
        )

        # Output inverter will be flipped, so get the sum of heights
        y0 = latch_master.bound_box.top_unit + outputinv_master.bound_box.height_unit
        outputinv_orient = 'MX'

        latch_inst = self.add_instance(latch_master, 'XLATCH',
                                    loc=(x_latch, 0), unit_mode=True)
        outputinv_inst = self.add_instance(outputinv_master, 'XOUTINV',
                                    loc=(x_outputinv, y0),
                                    unit_mode=True, orient=outputinv_orient)

        # Connect latch output to inverter input
        inv1_vin_warr = outputinv_inst.get_all_port_pins('QM_B')[0]
        inv2_vin_warr = outputinv_inst.get_all_port_pins('QM')[0]
        latch_qm_warr = latch_inst.get_all_port_pins('QM')[0]
        latch_qmb_warr = latch_inst.get_all_port_pins('QM_B')[0]

        # Define vertical connection tracks
        latchout1_vert_track = TrackID(
            top_vert_layer,
            self.grid.coord_to_nearest_track(
                top_vert_layer,
                inv1_vin_warr.middle,
                mode=-1
            )
        )
        latchout2_vert_track = TrackID(
            top_vert_layer,
            self.grid.coord_to_nearest_track(
                top_vert_layer,
                inv2_vin_warr.middle,
                mode=1
            )
        )
        # Actually perform connection
        self.connect_to_tracks(
            [inv1_vin_warr, latch_qmb_warr], latchout1_vert_track,
            track_lower=self.grid.track_to_coord(
                latch_qmb_warr.layer_id,
                min(latch_qm_warr.track_id.base_index, latch_qmb_warr.track_id.base_index),
                unit_mode=True
            ),
            unit_mode=True
        )
        self.connect_to_tracks(
            [inv2_vin_warr, latch_qm_warr], latchout2_vert_track,
            track_lower=self.grid.track_to_coord(
                latch_qm_warr.layer_id,
                min(latch_qm_warr.track_id.base_index, latch_qmb_warr.track_id.base_index),
                unit_mode=True
            ),
            unit_mode=True
        )


        # Define the VSS connections
        latch_vss_warr = latch_inst.get_all_port_pins('VSS')[0]
        outputinv_vss_warr = outputinv_inst.get_all_port_pins('VSS')[0]

        vss_verttrack_1 = TrackID(
            top_vert_layer,
            self.grid.coord_to_nearest_track(
                top_vert_layer,
                outputinv_vss_warr.lower, mode=-1
            )
        )
        vss_verttrack_2 = TrackID(
            top_vert_layer,
            self.grid.coord_to_nearest_track(
                top_vert_layer,
                outputinv_vss_warr.upper, mode=1
            )
        )

        self.connect_to_tracks(
            [latch_vss_warr, outputinv_vss_warr],
            vss_verttrack_1
        )
        self.connect_to_tracks(
            [latch_vss_warr, outputinv_vss_warr],
            vss_verttrack_2
        )

        # Define VDD connections
        latch_vdd_warr = latch_inst.get_all_port_pins('VDD')[0]
        outputinv_vdd_warr = outputinv_inst.get_all_port_pins('VDD')[0]

        # Stitch the VDD wires together
        vdd_tracklist = self.get_available_tracks(
            top_vert_layer,
            #  +1 because range subtracts 1
            range(
                self.grid.coord_to_nearest_track(
                    top_vert_layer,
                    outputinv_vdd_warr.lower, mode=1
                ),
                self.grid.coord_to_nearest_track(
                    top_vert_layer,
                    outputinv_vdd_warr.upper, mode=-1
                )+1
            ),
            lower=self.grid.track_to_coord(latch_vdd_warr.layer_id, latch_vdd_warr.track_id.base_index, unit_mode=True),
            upper=self.grid.track_to_coord(outputinv_vdd_warr.layer_id, outputinv_vdd_warr.track_id.base_index, unit_mode=True),
            unit_mode=True
        )

        for counter in range(len(vdd_tracklist)):
            if counter % 2 == 0:
                self.connect_to_tracks(
                    [latch_vdd_warr, outputinv_vdd_warr],
                    TrackID(top_vert_layer, vdd_tracklist[counter])
                )
            counter += 1


        self.reexport(outputinv_inst.get_port("Q_B"), net_name="Q_B", show=show_pins)
        self.reexport(outputinv_inst.get_port("Q"), net_name="Q", show=show_pins)

        # Extend output pins to top of block
        outputinv_q_warr = outputinv_inst.get_all_port_pins('Q')[0]
        outputinv_qb_warr = outputinv_inst.get_all_port_pins('Q_B')[0]
        q_tid = TrackID(
            outputinv_q_warr.layer_id + 1,
            self.grid.coord_to_nearest_track(
                outputinv_q_warr.layer_id + 1,
                outputinv_q_warr.middle,
                mode=-1
            )
        )
        qb_tid = TrackID(
            outputinv_qb_warr.layer_id + 1,
            self.grid.coord_to_nearest_track(
                outputinv_qb_warr.layer_id + 1,
                outputinv_qb_warr.middle,
                mode=1
            )
        )
        q_warr = self.connect_to_tracks(outputinv_q_warr, q_tid)
        qb_warr = self.connect_to_tracks(outputinv_qb_warr, qb_tid)
        q_warr, qb_warr = self.extend_wires(
            [q_warr, qb_warr], upper=y0, unit_mode=True
        )
        self.add_pin("Q", q_warr, show=show_pins)
        self.add_pin("Q_B", qb_warr, show=show_pins)




        # Export the input ports, and rename according to whether inputs are active low or high
        if input_reset_high:
            self.reexport(latch_inst.get_port("S_B"), show=show_pins)
            self.reexport(latch_inst.get_port("R_B"), show=show_pins)
        else:
            self.reexport(latch_inst.get_port("S"), show=show_pins)
            self.reexport(latch_inst.get_port("R"), show=show_pins)

        # Export the VDD and VSS pins
        self.reexport(latch_inst.get_port("VDD"), show=show_pins)
        self.reexport(outputinv_inst.get_port("VDD"), show=show_pins)
        self.add_pin('VSS',
                     [latch_inst.get_all_port_pins('VSS')[0], outputinv_inst.get_all_port_pins('VSS')[0]],
                     show=show_pins, )

        # Set the bound box. Start by getting block height
        [blk_w, blk_h] = self.grid.get_block_size(top_vert_layer, unit_mode=True)

        top = math.ceil(max(outputinv_inst.bound_box.top_unit, latch_inst.bound_box.top_unit) / blk_h) * blk_h
        right = math.ceil(max(outputinv_inst.bound_box.right_unit, latch_inst.bound_box.right_unit)/ blk_w) * blk_w
        bound_box = latch_inst.bound_box
        bound_box = bound_box.extend(x=0, y=0, unit_mode=True)
        bound_box = bound_box.extend(x=right, y=top, unit_mode=True)
        self.set_size_from_bound_box(top_vert_layer, bound_box)

        self._sch_params = dict(
            latch_params=latch_master.sch_params,
            outputinv_params=outputinv_master.sch_params,
            input_reset_high=input_reset_high
        )