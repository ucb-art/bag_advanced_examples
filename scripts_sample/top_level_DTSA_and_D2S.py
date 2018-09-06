# -*- coding: utf-8 -*-

import yaml
import os
import importlib
from bag.core import BagProject
from bag.layout import RoutingGrid
from bag.data import *
from bag.layout.template import TemplateDB


def make_tdb(prj, target_lib, specs):
    grid_specs = specs['routing_grid']
    layers = grid_specs['layers']
    spaces = grid_specs['spaces']
    widths = grid_specs['widths']
    bot_dir = grid_specs['bot_dir']

    routing_grid = RoutingGrid(prj.tech_info, layers, spaces, widths, bot_dir)
    tdb = TemplateDB('template_libs.def', routing_grid, target_lib, use_cybagoa=True)
    return tdb


def generate(prj, specs, gen_layout=True, gen_sch=False, run_lvs=False, run_rcx=False):
    """
    Creates the layout instance.
    Generates the layout and schematic based on the passed arguments. Optionally runs LVS and PEX/extraction.
    Parameters
    ----------
    prj :
        the BAG project
    specs :
        the specs from the yaml file
    gen_layout : bool
        true to generate the layout
    gen_sch : bool
        true to generate the schematic
    run_lvs : bool
        true to run LVS
    run_rcx :
        true to run post layout extraction

    Returns
    -------

    """
    # Get information from YAML
    params = specs['params']

    impl_lib = specs['impl_lib']
    impl_cell = specs['impl_cell']

    layout_package = specs['layout_package']
    layout_class = specs['layout_class']

    layout_module = importlib.import_module(layout_package)
    temp_cls = getattr(layout_module, layout_class)

    temp_db = make_tdb(prj, impl_lib, specs)
    temp = temp_db.new_template(params=params, temp_cls=temp_cls, debug=False)

    if gen_layout:
        print('creating layout')
        temp_db.batch_layout(prj, [temp], [impl_cell])
        print('done')

    if gen_sch:
        sch_gen_lib = specs['sch_gen_lib']
        sch_gen_cell = specs['sch_gen_cell']

        dsn = prj.create_design_module(lib_name=sch_gen_lib, cell_name=sch_gen_cell)
        dsn.design(**temp.sch_params)
        print('creating schematic for %s' % sch_gen_cell)
        dsn.implement_design(impl_lib, top_cell_name=impl_cell)

    if run_lvs:
        print('running lvs')
        lvs_passed, lvs_log = prj.run_lvs(impl_lib, impl_cell)
        if not lvs_passed:
            raise ValueError('LVS failed log %s' % lvs_log)
        else:
            print('lvs passed')
            print('lvs log is' + lvs_log)

    if run_rcx:
        print('running rcx')
        rcx_passed, rcx_log = prj.run_rcx(impl_lib, impl_cell)
        if not rcx_passed:
            raise Exception('rcx died. See RCX log %s' % rcx_log)
        print('rcx passed')


def generate_tb(prj, specs):
    print('Generating tb')
    testbenches = specs['testbenches']
    impl_lib = specs['impl_lib']
    impl_cell = specs['impl_cell']

    # In this script, we assume only one cell is generated
    for name, info in testbenches.items():
        tb_gen_cell = info['tb_cell']
        tb_impl_lib = info['tb_lib']
        print('setting up %s' % tb_gen_cell)

        tb_dsn = prj.create_design_module(tb_impl_lib, tb_gen_cell)
        print('computing %s schematics' % tb_gen_cell)

        tb_dsn.design(dut_lib=impl_lib, dut_cell=impl_cell, layout_params=specs['params'])
        print(tb_gen_cell)
        print('creating %s schematics' % tb_gen_cell)
        tb_dsn.implement_design(impl_lib, top_cell_name=tb_gen_cell)


def configure_and_run(prj, specs, dsn_name, run=False):
    impl_lib = specs['impl_lib']
    # Set which cellview to simulate (schematic vs pex)
    view_name = specs['view_name']
    # Sets up the corners
    sim_envs = specs['sim_envs']

    lib_name = specs['lib_name']
    data_dir = lib_name + '/tb_output_data'
    testbenches = specs['testbenches']

    results_dict = {}
    for name, info in testbenches.items():
        tb_generator_cell = info['tb_cell']
        tb_generated_cell = tb_generator_cell
        tb_params = info['tb_params']
        tb_sweep_params = info['tb_sweep_params']

        print(tb_sweep_params)
        print('setting up %s' % tb_generator_cell)
        tb = prj.configure_testbench(impl_lib, tb_generated_cell)

        for key, val in tb_params.items():
            tb.set_parameter(key, val)

        if tb_sweep_params:
            for key, val in tb_sweep_params.items():
                tb.set_sweep_parameter(key, values=val)

        # Set the config view: simulate schematic vs extracted
        tb.set_simulation_view(impl_lib, tb_generator_cell, view_name)
        # Set process corner
        tb.set_simulation_environments(sim_envs)
        # Send changes of the ADEXL to the database
        tb.update_testbench()

        if run:
            print('running simulation')
            tb.run_simulation()
            print('simulation done, load results')
            results = load_sim_results(tb.save_dir)
            save_sim_results(results, os.path.join(data_dir, '%s.hdf5' % tb_generated_cell))
            results_dict[name] = results


if __name__ == '__main__':
    with open('bag_advanced_examples/specs_sample/DTSA_and_D2S.yaml', 'r') as f:
        block_specs = yaml.load(f)

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        bprj = BagProject()

    else:
        print('loading BAG project')
        bprj = local_dict['bprj']
    print("generating")

    generate(bprj, block_specs, gen_layout=True, gen_sch=True, run_lvs=True, run_rcx=False)
    # generate_tb(bprj, block_specs)
    # configure_and_run(bprj, block_specs, block_specs['impl_cell'], run=False)
