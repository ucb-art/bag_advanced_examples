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

    This is the layout generator for Double Tail Sense Amplifier (DTSA)

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

from typing import Dict, Any, Set, Optional, Union, List

from bag.layout.routing import TrackManager, TrackID, WireArray
from bag.layout.template import TemplateDB

from abs_templates_ec.analog_core import AnalogBase


class DoubleTailSenseAmplifier(AnalogBase):
    """A Double-tail Sense Amplifier with NMOS input pair

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

        # Define what layers will be used for horizontal and vertical connections
        horz_conn_layer = self.mos_conn_layer + 1
        vert_conn_layer = self.mos_conn_layer + 2

        ################################################################################
        # 0:   Floorplan the design
        #
        # Know where transistors go and where the horizontal
        # and vertical connections will be
        # Also determine how transistors will be aligned horizontally relative to eachother
        # and know how things will need to shift as various sizes scale
        ################################################################################

        ################################################################################
        # 1:   Set up track allocations for each row
        #
        # Use TrackManager to create allocate track and spacing based on the floorplan
        #   -  Rather than explicitly allocate a number of gate and source/drain tracks for each row (as is done in
        #      the bootcamp modules), instead define a list of what horizontal connects will be required for each row.
        #   -  Based on spacing and width rules provided in the specification file, BAG/trackmanager will calculate
        #      how many tracks each row needs
        # Finally, initialize the rows using the helper functions above
        ################################################################################
        tr_manager = TrackManager(grid=self.grid, tr_widths=tr_widths, tr_spaces=tr_spaces)

        # Rows are ordered from bottom to top
        # To use TrackManager, an ordered list of wiring types and their locations must be provided.
        # Define two lists, one for the nch rows and one for the pch rows
        # The lists are composed of dictionaries, one per row.
        # Each dictionary has two list entries (g and ds), which are ordered lists of what wire types will be present
        #  in the g and ds sections of that row. Ordering is from bottom to top of the design.
        wire_names = dict(
            nch=[
                # Pre-amp tail row
                dict(
                    g=['clk', ],
                    ds=['bias', ]
                ),
                # Pre-amp input row
                dict(
                    g=['sig'],
                    ds=['sig']
                ),
                # Regen-amp nmos
                dict(
                    ds=['bias'],
                    g=['sig', ]
                ),
            ],
            pch=[
                # Regen amp pmos load
                dict(
                    g=['sig', 'sig'],
                    ds=['bias']
                ),
                # Pre-amp pmos load and pmos tail/enable
                dict(
                    g=['clk', 'clk'],
                    ds=['sig']
                ),
            ]
        )

        # Set up the row information
        # Row information contains the row properties like width/number of fins, orientation, intent, etc.
        # Storing in a row dictionary/object allows for convenient fetching of data in later functions
        row_tail = self.initialize_rows(row_name='tail',
                                        orient='R0',
                                        nch_or_pch='nch',
                                        )
        row_pre_in = self.initialize_rows(row_name='pre_in',
                                          orient='R0',
                                          nch_or_pch='nch',
                                          )
        row_regen_n = self.initialize_rows(row_name='regen_n',
                                           orient='MX',
                                           nch_or_pch='nch',
                                           )
        row_regen_p = self.initialize_rows(row_name='regen_p',
                                           orient='R0',
                                           nch_or_pch='pch',
                                           )
        row_pre_load = self.initialize_rows(row_name='pre_load',
                                            orient='R0',
                                            nch_or_pch='pch',
                                            )

        # Define the order of the rows (bottom to top) for this analogBase cell
        self.set_global_rows(
            [row_tail, row_pre_in, row_regen_n, row_regen_p, row_pre_load]
        )

        ################################################################################
        # 2:
        # Initialize the transistors in the design
        # Storing each transistor's information (name, location, row, size, etc) in a dictionary object allows for
        # convient use later in the code, and also greatly simplifies the schematic generation
        # The initialization sets the transistor's row, width, and source/drain net names for proper dummy creation
        ################################################################################
        tail_1 = self.initialize_tx(name='tail_1', row=row_tail, fg_spec='tail_n', deff_net='PRE_AMP_SOURCE')
        tail_2 = self.initialize_tx(name='tail_2', row=row_tail, fg_spec='tail_n', deff_net='PRE_AMP_SOURCE')
        in_p = self.initialize_tx(name='in_p', row=row_pre_in, fg_spec='in_n',
                                  seff_net='PRE_AMP_SOURCE', deff_net='DIN')
        in_n = self.initialize_tx(name='in_n', row=row_pre_in, fg_spec='in_n',
                                  seff_net='PRE_AMP_SOURCE', deff_net='DIP')
        pre_load_p = self.initialize_tx(name='pre_load_p', row=row_pre_load, fg_spec='pre_load_p', deff_net='DIN')
        pre_load_n = self.initialize_tx(name='pre_load_n', row=row_pre_load, fg_spec='pre_load_p', deff_net='DIP')
        regen_n_p = self.initialize_tx(name='regen_n_p', row=row_regen_n, fg_spec='regen_n', deff_net='VOP')
        regen_n_n = self.initialize_tx(name='regen_n_n', row=row_regen_n, fg_spec='regen_n', deff_net='VON')
        regen_p_p = self.initialize_tx(name='regen_p_p', row=row_regen_p, fg_spec='regen_p',
                                       seff_net='REGEN_SOURCE', deff_net='VOP')
        regen_p_n = self.initialize_tx(name='regen_p_n', row=row_regen_p, fg_spec='regen_p',
                                       seff_net='REGEN_SOURCE', deff_net='VON')
        tail_p_1 = self.initialize_tx(name='tail_p_1', row=row_pre_load, fg_spec='tail_p', deff_net='REGEN_SOURCE')
        tail_p_2 = self.initialize_tx(name='tail_p_2', row=row_pre_load, fg_spec='tail_p', deff_net='REGEN_SOURCE')
        reset_p = self.initialize_tx(name='reset_p', row=row_regen_n, fg_spec='reset_n', deff_net='VOP')
        reset_n = self.initialize_tx(name='reset_n', row=row_regen_n, fg_spec='reset_n', deff_net='VON')

        # Compose a list of all the transistors so it can be iterated over later
        transistors = [
            tail_1, tail_2, in_p, in_n, pre_load_p, pre_load_n, regen_n_p, regen_n_n,
            regen_p_p, regen_p_n, tail_p_1, tail_p_2, reset_p, reset_n
        ]

        # Check that all transistors are even fingered
        for tx in transistors:
            if tx['fg'] % 2 == 1:
                raise ValueError(
                    "Transistors must have even number of fingers. Transistor '{}' has {}".format(tx['name'], tx['fg]'])
                )

        ################################################################################
        # 3:   Calculate transistor locations
        # Based on the floorplan, want the tail, input, nmos regen, pmos regen, and pmos tail to be in a column
        # and for convenience, place the reset and load in a column, but right/left justified
        # Notation:
        #     fg_xxx refers to how wide (in fingers) a transistor or column of transistors is
        #     col_xxx refers to the location of the left most finger of a transistor or a column of transistors
        ################################################################################

        fg_stack = max(tail_1['fg'], in_p['fg'], regen_n_p['fg'], regen_p_p['fg'], tail_p_1['fg'])
        fg_side = max(reset_p['fg'], pre_load_p['fg'])

        # Add an explicit gap in the middle for symmetry. Set to 0 to not have gap
        fg_mid = 0

        # Get the minimum gap between fingers of different transistors
        # This varies between processes, so avoid hard-coding by using the method in self._tech_cls
        fg_space = self._tech_cls.get_min_fg_sep(lch)

        fg_total = fg_dum + fg_side + fg_space + fg_stack + fg_mid + fg_space + fg_stack + fg_space + fg_side + fg_dum

        # Calculate the starting column index for each stack of transistors
        col_side_left = fg_dum
        col_stack_left = col_side_left + fg_side + fg_space
        col_stack_right = col_stack_left + fg_stack + fg_space + fg_mid
        col_side_right = col_stack_right + fg_stack + fg_space

        # Calculate positions of transistors
        # This uses helper functions to place each transistor within a stack/column of a specified starting index and
        # width, and with a certain alignment (left, right, centered) within that column
        self.assign_tx_column(tx=tail_1, offset=col_stack_left, fg_col=fg_stack, align=0)
        self.assign_tx_column(tx=in_n, offset=col_stack_left, fg_col=fg_stack, align=0)
        self.assign_tx_column(tx=regen_n_n, offset=col_stack_left, fg_col=fg_stack, align=0)
        self.assign_tx_column(tx=regen_p_n, offset=col_stack_left, fg_col=fg_stack, align=0)
        self.assign_tx_column(tx=tail_p_1, offset=col_stack_left, fg_col=fg_stack, align=0)

        self.assign_tx_column(tx=tail_2, offset=col_stack_right, fg_col=fg_stack, align=0)
        self.assign_tx_column(tx=in_p, offset=col_stack_right, fg_col=fg_stack, align=0)
        self.assign_tx_column(tx=regen_n_p, offset=col_stack_right, fg_col=fg_stack, align=0)
        self.assign_tx_column(tx=regen_p_p, offset=col_stack_right, fg_col=fg_stack, align=0)
        self.assign_tx_column(tx=tail_p_2, offset=col_stack_right, fg_col=fg_stack, align=0)

        self.assign_tx_column(tx=reset_n, offset=col_side_left, fg_col=fg_side, align=1)
        self.assign_tx_column(tx=pre_load_n, offset=col_side_left, fg_col=fg_side, align=1)

        self.assign_tx_column(tx=reset_p, offset=col_side_right, fg_col=fg_side, align=-1)
        self.assign_tx_column(tx=pre_load_p, offset=col_side_right, fg_col=fg_side, align=-1)

        ################################################################################
        # 4:  Assign the transistor directions (s/d up vs down)
        #
        # Specify the directions that connections to the source and connections to the drain will go (up vs down)
        # Doing so will also determine how the gate is aligned (ie will it be aligned to the source or drain)
        # See the bootcamp for more details
        # The helper functions used here help to abstract away whether the intended source/drain diffusion region of
        # a transistor occurs on the even or odd columns of that device (BAG always considers the even columns of a
        # device to be the 's').
        # These helper functions allow a user to specify whether the even columns should be the transistors effective
        #  source or effective drain, so that the user does not need to worry about BAG's notation.
        ################################################################################

        # Set tail transistor to have source on the leftmost diffusion (arbitrary) and source going down
        self.set_tx_directions(tx=tail_1, seff='s', seff_dir=0)
        # Assign the input to be anti-aligned, so that the input source and tail drain are vertically aligned
        self.assign_tx_matched_direction(target_tx=in_p, source_tx=tail_1, seff_dir=0, aligned=False)

        # Set regen nmos to arbitrarily have source on left
        self.set_tx_directions(tx=regen_n_p, seff='s', seff_dir=0)
        # Set regen pmos to align so sources and drains are vertically aligned. pmos source will go up
        self.assign_tx_matched_direction(target_tx=regen_p_p, source_tx=regen_n_p, seff_dir=2, aligned=True)
        # Set the regen tail stage to anti-align so tail drain aligns vertically with regen pmos source.
        self.assign_tx_matched_direction(target_tx=tail_p_1, source_tx=regen_p_p, seff_dir=2, aligned=False)

        # Arbitrarily set the s/d effective for the reset and pre-amp load, as they will not align to anything
        self.set_tx_directions(tx=reset_p, seff='s', seff_dir=0)
        self.set_tx_directions(tx=pre_load_p, seff='s', seff_dir=2)

        # Do the same alignments for the mirror-symmetric negative half of the circuit
        self.set_tx_directions(tx=tail_2, seff='s', seff_dir=0)
        self.assign_tx_matched_direction(target_tx=in_n, source_tx=tail_2, seff_dir=0, aligned=False)
        self.set_tx_directions(tx=regen_n_n, seff='s', seff_dir=0)
        self.assign_tx_matched_direction(target_tx=regen_p_n, source_tx=regen_n_n, seff_dir=2, aligned=True)
        self.assign_tx_matched_direction(target_tx=tail_p_2, source_tx=regen_p_n, seff_dir=2, aligned=False)
        self.set_tx_directions(tx=reset_n, seff='s', seff_dir=0)
        self.set_tx_directions(tx=pre_load_n, seff='s', seff_dir=2)

        ################################################################################
        # 5:  Draw the transistor rows, and the transistors
        #
        # All the difficult setup has been complete. Drawing the transistors is simple now.
        # Note that we pass the wire_names dictionary defined above so that BAG knows how to space out
        # the transistor rows. BAG uses this to calculate how many tracks to allocate to each
        ################################################################################

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

        ################################################################################
        # 6:  Define horizontal tracks on which connections will be made
        #
        # Based on the wire_names dictionary defined in step 1), create TrackIDs on which horizontal connections will
        #  be made
        ################################################################################

        tid_pream_tail = self.get_wire_id('nch', row_tail['index'], 'ds', wire_name='bias')
        tid_d = self.get_wire_id('nch', row_pre_in['index'], 'ds', wire_name='sig')
        tid_reset_gate = self.get_wire_id('nch', row_regen_n['index'], 'g', wire_name='sig')
        tid_out_p_horz = self.get_wire_id('pch', row_regen_p['index'], 'g', wire_name='sig', wire_idx=0)
        tid_out_n_horz = self.get_wire_id('pch', row_regen_p['index'], 'g', wire_name='sig', wire_idx=1)
        tid_regen_vss = self.get_wire_id('nch', row_regen_n['index'], 'ds', wire_name='bias')
        tid_regen_tail = self.get_wire_id('pch', row_regen_p['index'], 'ds', wire_name='bias')
        tid_tail_regen_clk = self.get_wire_id('pch', row_pre_load['index'], 'g', wire_name='clk', wire_idx=0)
        tid_tail_regen_clk_b = self.get_wire_id('pch', row_pre_load['index'], 'g', wire_name='clk', wire_idx=1)
        tid_tail_preamp_clk = self.get_wire_id('nch', row_tail['index'], 'g', wire_name='clk')
        tid_sig_in_horz = self.get_wire_id('nch', row_pre_in['index'], 'g', wire_name='sig')
        tid_preamp_load_d = self.get_wire_id('pch', row_pre_load['index'], 'ds', wire_name='sig')

        ################################################################################
        # 7:  Perform wiring
        #
        # Use the self.connect_to_tracks, self.connect_differential_tracks, self.connect_wires, etc
        #  to perform connections
        # Note that the drain/source/gate wire arrays of the transistors can be easily accessed as keys in the tx
        # dictionary structure.
        #
        # Best practice:
        #  - Avoid hard-coding widths and pitches
        #    Instead, use TrackManger's get_width, get_space, or get_next_track functionality to make the design
        #    fully portable across process
        #  - Avoid hard-coding which layers connections will be on.
        #    Instead use layers relative to self.mos_conn_layer
        ################################################################################
        # Connect pre-amp tail drain to input source
        self.connect_to_tracks(
            [tail_1['d'], tail_2['d'], in_p['s'], in_n['s']],
            tid_pream_tail,
            min_len_mode=0,
        )

        # connect outputs of pre-amp to gates of regen resets
        # Define vertical tracks for the connections, based on the location of the col_stack_left/right
        tid_d_n_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=self.grid.coord_to_nearest_track(
                layer_id=vert_conn_layer,
                coord=self.layout_info.col_to_coord(
                    col_idx=col_stack_right + fg_stack + 1,
                    unit_mode=True
                ),
                half_track=False,
                mode=1,
                unit_mode=True
            ),
            width=tr_manager.get_width(vert_conn_layer, 'sig')
        )
        # Connect input transistor drain to a horizontal track
        # Note from schematic that D_N connects to IN_P
        warr_d_n = self.connect_to_tracks(
            [in_p['d']],
            tid_d,
            min_len_mode=1
        )
        # Connect the reset gate to a horizontal track
        warr_reset_p_gate = self.connect_to_tracks(
            [reset_p['g']],
            tid_reset_gate,
            min_len_mode=1,
        )

        # Connect the pre-amp pmos load drain to a horizontal track
        warr_pream_load_d_p = self.connect_to_tracks(
            [pre_load_p['d']],
            tid_preamp_load_d,
            min_len_mode=0
        )

        # Perform the vertical connection
        self.connect_to_tracks(
            [warr_d_n, warr_reset_p_gate, warr_pream_load_d_p],
            tid_d_n_vert,
            min_len_mode=0,
        )

        # Do the same for the 'n' input
        tid_d_p_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=self.grid.coord_to_nearest_track(
                layer_id=vert_conn_layer,
                coord=self.layout_info.col_to_coord(
                    col_idx=col_stack_left - 1,
                    unit_mode=True
                ),
                half_track=False,
                mode=-1,
                unit_mode=True
            ),
            width=tr_manager.get_width(vert_conn_layer, 'sig')
        )

        warr_d_p = self.connect_to_tracks(
            [in_n['d']],
            tid_d,
            min_len_mode=1
        )
        warr_reset_n_gate = self.connect_to_tracks(
            [reset_n['g']],
            tid_reset_gate,
            min_len_mode=-1,
        )
        warr_pream_load_d_n = self.connect_to_tracks(
            [pre_load_n['d']],
            tid_preamp_load_d,
            min_len_mode=0
        )
        self.connect_to_tracks(
            [warr_d_p, warr_reset_n_gate, warr_pream_load_d_n],
            tid_d_p_vert,
            min_len_mode=0,
        )

        # Connect the regeneration nmos and reset nmos sources to a common strap to be grounded
        # warr_regen_vss =
        self.connect_to_tracks(
            [regen_n_n['s'], regen_n_p['s'], reset_n['s'], reset_p['s']],
            tid_regen_vss,
        )

        # Connect the regeneration pair
        warr_out_p_horz, warr_out_n_horz = self.connect_differential_tracks(
            pwarr_list=[regen_n_p['d'], regen_p_p['d'], reset_p['d'], regen_n_n['g'], regen_p_n['g']],
            nwarr_list=[regen_n_n['d'], regen_p_n['d'], reset_n['d'], regen_n_p['g'], regen_p_p['g']],
            tr_layer_id=tid_out_p_horz.layer_id,
            ptr_idx=tid_out_p_horz.base_index,
            ntr_idx=tid_out_n_horz.base_index,
            width=tr_manager.get_width(horz_conn_layer, 'sig')
        )

        # Connect the regeneration tail transistor drains to the regen_p sources
        self.connect_to_tracks(
            [tail_p_1['d'], tail_p_2['d'], regen_p_p['s'], regen_p_n['s']],
            tid_regen_tail,
            min_len_mode=0
        )

        # Connect horizontal straps to the gates of the pmos pre-amp load and the clocked pmos regen tail
        warr_tail_regen_clk = self.connect_to_tracks(
            [pre_load_p['g'], pre_load_n['g']],
            tid_tail_regen_clk,
            min_len_mode=0
        )

        warr_tail_regen_clk_b = self.connect_to_tracks(
            [tail_p_1['g'], tail_p_2['g']],
            tid_tail_regen_clk_b,
            min_len_mode=0
        )

        warr_tail_preamp_clk = self.connect_to_tracks(
            [tail_1['g'], tail_2['g']],
            tid_tail_preamp_clk,
            min_len_mode=0
        )

        # Define a vertical track to connect the clock
        tid_clk_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=self.grid.coord_to_nearest_track(
                layer_id=vert_conn_layer,
                coord=self.bound_box.left_unit,
                mode=2,
                unit_mode=True
            ),
            width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='clk')
        )
        tid_clk_b_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=tr_manager.get_next_track(
                layer_id=vert_conn_layer,
                cur_idx=tid_clk_vert.base_index,
                cur_type='clk',
                next_type='clk',
                up=True
            ),
            width=tr_manager.get_width(layer_id=vert_conn_layer, track_type='clk')
        )
        warr_clk_vert = self.connect_to_tracks(
            [warr_tail_regen_clk, warr_tail_preamp_clk],
            tid_clk_vert
        )
        warr_clk_b_vert = self.connect_to_tracks(
            [warr_tail_regen_clk_b],
            tid_clk_b_vert,
            track_lower=warr_clk_vert.lower_unit,
            track_upper=warr_clk_vert.upper_unit,
            unit_mode=True
        )

        # Wire inputs
        # 1: wire each to a horizontal stub
        # 2: wire each one down vertically to bottom of block
        warr_in_p_horz = self.connect_to_tracks(
            [in_p['g']],
            tid_sig_in_horz,
            min_len_mode=0
        )
        warr_in_n_horz = self.connect_to_tracks(
            [in_n['g']],
            tid_sig_in_horz,
            min_len_mode=0
        )

        tid_in_p_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=self.grid.coord_to_nearest_track(
                layer_id=vert_conn_layer,
                coord=warr_in_p_horz.middle_unit,
                mode=0,
                unit_mode=True
            ),
            width=tr_manager.get_width(vert_conn_layer, 'sig')
        )
        warr_in_p_vert = self.connect_to_tracks(
            [warr_in_p_horz],
            tid_in_p_vert,
            track_lower=self.bound_box.bottom_unit,
            unit_mode=True
        )

        tid_in_n_vert = TrackID(
            layer_id=vert_conn_layer,
            track_idx=self.grid.coord_to_nearest_track(
                layer_id=vert_conn_layer,
                coord=warr_in_n_horz.middle_unit,
                mode=0,
                unit_mode=True
            ),
            width=tr_manager.get_width(vert_conn_layer, 'sig')
        )
        warr_in_n_vert = self.connect_to_tracks(
            [warr_in_n_horz],
            tid_in_n_vert,
            track_lower=self.bound_box.bottom_unit,
            unit_mode=True
        )

        ################################################################################
        # 8:  Connections to substrate, and dummy fill
        #
        # - Use the self.connect_to_substrate method to perform wiring to the ntap and ptap where VDD and VSS will be
        # - Use self.fill_dummy() draw the dummy transistor structures, and finalize the VDD and VSS wiring
        #   - This method should be called last!
        ################################################################################
        # Connections to VSS
        self.connect_to_substrate(
            'ptap',
            [tail_1['s'], tail_2['s'], reset_p['s'], reset_n['s']]
        )

        # Connections to VDD
        self.connect_to_substrate(
            'ntap',
            [pre_load_p['s'], pre_load_n['s'], tail_p_1['s'], tail_p_2['s']]
        )

        warr_vss, warr_vdd = self.fill_dummy()

        ################################################################################
        # 9:  Add pins
        #
        ################################################################################
        self.add_pin('VDD', warr_vdd, show=show_pins)
        self.add_pin('VSS', warr_vss, show=show_pins)
        self.add_pin('VIP', warr_in_p_vert, show=show_pins)
        self.add_pin('VIN', warr_in_n_vert, show=show_pins)
        self.add_pin('CLK', warr_clk_vert, show=show_pins)
        self.add_pin('CLK_B', warr_clk_b_vert, show=show_pins)
        self.add_pin('VOP', warr_out_p_horz, show=show_pins)
        self.add_pin('VON', warr_out_n_horz, show=show_pins)
        self.add_pin('DIP', warr_d_p, show=show_pins)
        self.add_pin('DIN', warr_d_n, show=show_pins)

        ################################################################################
        # 10:  Organize parameters for the schematic generator
        #
        # To make the schematic generator very simple, organize the required transistor information (fins/widith,
        # intent, and number of fingers) into a convenient data structure
        # Finally set the self._sch_params property so that these parameters are accessible to the schematic generator
        ################################################################################
        # Define transistor properties for schematic
        tx_info = {}
        for tx in transistors:
            tx_info[tx['name']] = {}
            tx_info[tx['name']]['w'] = tx['w']
            tx_info[tx['name']]['th'] = tx['th']
            tx_info[tx['name']]['fg'] = tx['fg']

        self._sch_params = dict(
            lch=lch,
            dum_info=self.get_sch_dummy_info(),
            tx_info=tx_info,
        )
