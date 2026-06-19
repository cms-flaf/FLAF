#!/usr/bin/env python

import importlib
import os
import re
import sys
import ROOT

ROOT.gROOT.SetBatch(True)


def get_file_path(file):
    if os.path.exists(file):
        return file
    file_name = os.path.basename(file)
    if os.path.exists(file_name):
        return file_name
    if "ANALYSIS_PATH" in os.environ:
        file_path = os.path.join(os.environ["ANALYSIS_PATH"], file)
        if os.path.exists(file_path):
            return file_path
    if "CMSSW_BASE" in os.environ:
        file_path = os.path.join(os.environ["CMSSW_BASE"], "src", file)
        if os.path.exists(file_path):
            return file_path
    raise RuntimeError(f"Unable to find {file}")


def load_module(module_path):
    module_path = os.path.abspath(get_file_path(module_path))
    module_dir, module_file = os.path.split(module_path)
    module_name, _ = os.path.splitext(module_file)
    package_dir, package_name = os.path.split(module_dir)
    if len(package_dir) > 0 and package_dir not in sys.path:
        sys.path.append(package_dir)
    if len(package_name) == 0:
        package_name = module_name
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.{module_name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def select_items(all_items, filters, verbose=0):
    def name_match(name, pattern):
        if pattern[0] == "^":
            return re.match(pattern, name) is not None
        return name == pattern

    selected_items = {c for c in all_items}
    excluded_items = set()
    keep_prefix = "keep "
    drop_prefix = "drop "
    used_filters = set()
    for item_filter in filters:
        if item_filter.startswith(keep_prefix):
            keep = True
            items_from = excluded_items
            items_to = selected_items
            prefix = keep_prefix
        elif item_filter.startswith(drop_prefix):
            keep = False
            items_from = selected_items
            items_to = excluded_items
            prefix = drop_prefix
        else:
            raise RuntimeError(f'Unsupported filter = "{item_filter}".')
        pattern = item_filter[len(prefix) :]
        if len(pattern) == 0:
            raise RuntimeError(f"Filter with an empty pattern expression.")

        to_move = [item for item in items_from if name_match(item, pattern)]
        if len(to_move) > 0:
            used_filters.add(item_filter)
            for column in to_move:
                items_from.remove(column)
                items_to.add(column)

    unused_filters = set(filters) - used_filters
    if len(unused_filters) > 0 and verbose > 0:
        print("Unused filters: " + " ".join(unused_filters))

    return sorted(selected_items)


def select_inputs(all_inputs, obj_names, ignore_absent, skip_empty, verbose=0):
    inputs = {}
    for obj_name in obj_names:
        inputs[obj_name] = ROOT.vector("string")()
    if not ignore_absent and not skip_empty:
        for input in all_inputs:
            for obj_name in obj_names:
                inputs[obj_name].push_back(input)
    else:
        for input in all_inputs:
            file = ROOT.TFile.Open(input, "READ")
            if not file:
                raise RuntimeError(f"Failed to open file '{input}'.")
            for obj_name in obj_names:
                obj = file.Get(obj_name)
                if obj:
                    if (
                        not skip_empty
                        or not hasattr(obj, "GetEntries")
                        or obj.GetEntries() > 0
                    ):
                        inputs[obj_name].push_back(input)
                else:
                    if ignore_absent:
                        if verbose > 0:
                            print(
                                f"WARNING: object '{obj_name}' is not found in file '{input}'."
                            )
                    else:
                        raise RuntimeError(
                            f"Object '{obj_name}' is not found in file '{input}'."
                        )
            file.Close()
    return inputs


def get_columns(df):
    all_columns = [str(c) for c in df.GetColumnNames()]
    simple_types = ["Int_t", "UInt_t", "Long64_t", "ULong64_t", "int", "long"]
    column_types = {c: str(df.GetColumnType(c)) for c in all_columns}
    all_columns = sorted(
        all_columns, key=lambda c: (column_types[c] not in simple_types, c)
    )
    return all_columns, column_types


processing_modules = {}


def skim_tree(
    inputs,
    input_tree,
    output,
    output_tree,
    input_range,
    output_range,
    snapshot_options,
    column_filters,
    sel,
    invert_sel,
    processing_module,
    processing_function,
    processing_arguments,
    verbose=0,
):
    def create_empty_file(msg):
        if verbose > 0:
            print(msg)
        if not os.path.exists(output):
            if verbose > 0:
                print("Creating an empty output file.")
            compression_settings = (
                snapshot_options.fCompressionAlgorithm * 100
                + snapshot_options.fCompressionLevel
            )
            output_file = ROOT.TFile.Open(output, "RECREATE", "", compression_settings)
            output_file.Close()

    if len(inputs) == 0:
        create_empty_file(f"No inputs files to skim {input_tree}.")
        return

    df = ROOT.RDataFrame(input_tree, inputs)

    if input_range:
        df = df.Range(*input_range)

    if processing_module:
        if verbose > 0:
            arg_str = ", ".join(["df"] + processing_arguments)
            print(f"Running {processing_module}:{processing_function}({arg_str}) ...")
        if processing_module not in processing_modules:
            processing_modules[processing_module] = load_module(processing_module)
        module = processing_modules[processing_module]
        fn = getattr(module, processing_function)
        df = fn(df, *processing_arguments)

    all_columns, column_types = get_columns(df)
    selected_columns = select_items(all_columns, column_filters, verbose=verbose)

    if len(selected_columns) == 0:
        create_empty_file(f"No columns were selected to skim {input_tree}.")
        return

    branches = ROOT.vector("string")()
    for column in all_columns:
        if column in selected_columns:
            branches.push_back(column)
            if verbose > 2:
                print(f"Adding column '{column_types[column]} {column}'...")

    if sel:
        if invert_sel:
            df = df.Define("__passSkimSel", sel).Filter("!__passSkimSel")
        else:
            df = df.Filter(sel)

    if output_range is not None:
        df = df.Range(*output_range)

    if verbose > 0:
        print("Creating a snapshot...")
    df.Snapshot(output_tree, output, branches, snapshot_options)
    if processing_module:
        module = processing_modules[processing_module]
        if hasattr(module, "OnTreeSkimFinish"):
            if verbose > 0:
                print(
                    f"Running {processing_module}:OnTreeSkimFinish({input_tree}, {output_tree}) ..."
                )
            module.OnTreeSkimFinish(input_tree, output_tree)


def copy_tree(inputs, tree_name, output, snapshot_options, verbose=0):
    if verbose > 0:
        print(f"Copying {tree_name}...")
    df = ROOT.RDataFrame(tree_name, inputs)
    branches = ROOT.vector("string")()
    all_columns, column_types = get_columns(df)
    for column in all_columns:
        branches.push_back(column)
        if verbose > 2:
            print(f"\tAdding column '{column_types[column]} {column}'...")
    df.Snapshot(tree_name, output, branches, snapshot_options)


def copy_histograms(
    all_inputs, inputs, hist_names, output, snapshot_options, verbose=0
):
    if verbose > 0:
        print(f"Copying histograms {hist_names}...")
    hists = {}
    for input in all_inputs:
        input_file = None
        for hist_name in hist_names:
            if input in inputs[hist_name]:
                if not input_file:
                    input_file = ROOT.TFile.Open(input, "READ")
                    if not input_file:
                        raise RuntimeError(f"Failed to open file '{input}'.")
                hist = input_file.Get(hist_name)
                if not hist:
                    raise RuntimeError(
                        f"Failed to get histogram '{hist_name}' from file '{input}'."
                    )
                if hist_name in hists:
                    hists[hist_name].Add(hist)
                else:
                    hist_copy = hist.Clone()
                    hist_copy.SetDirectory(0)
                    hists[hist_name] = hist_copy
        if input_file:
            input_file.Close()

    compression_settings = (
        snapshot_options.fCompressionAlgorithm * 100
        + snapshot_options.fCompressionLevel
    )
    output_file = ROOT.TFile.Open(output, "UPDATE", "", compression_settings)
    for hist_name, hist in hists.items():
        output_file.WriteTObject(hist, hist_name)
    output_file.Close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skim tree.")
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="input root file or txt file with a list of files",
    )
    parser.add_argument("--output", required=True, type=str, help="output root file")
    parser.add_argument(
        "--input-tree", required=False, type=str, default=None, help="input tree name"
    )
    parser.add_argument(
        "--output-tree", required=False, type=str, default=None, help="output tree"
    )
    parser.add_argument(
        "--other-trees", required=False, default=None, help="other trees to copy"
    )
    parser.add_argument(
        "--hists", required=False, default=None, help="histograms to copy"
    )
    parser.add_argument(
        "--sel", required=False, type=str, default=None, help="selection"
    )
    parser.add_argument("--invert-sel", action="store_true", help="Invert selection.")
    parser.add_argument(
        "--ignore-absent",
        action="store_true",
        help="Ignore file if target tree or hist is absent.",
    )
    parser.add_argument(
        "--skip-empty", action="store_true", help="Skip skip trees with 0 entries."
    )
    parser.add_argument(
        "--column-filters",
        required=False,
        default=None,
        type=str,
        help="""Comma separated statements to keep and drop columns based on exact names or regex patterns.
                              By default, all columns will be included""",
    )
    parser.add_argument(
        "--input-prefix",
        required=False,
        type=str,
        default="",
        help="prefix to be added to input each input file",
    )
    parser.add_argument(
        "--processing-module",
        required=False,
        type=str,
        default=None,
        help="Python module used to process DataFrame. Should be in form file:method",
    )
    parser.add_argument(
        "--processing-arguments",
        nargs="*",
        required=False,
        type=str,
        default=[],
        help="Argument to pass to processing module.",
    )
    parser.add_argument(
        "--config",
        required=False,
        type=str,
        default=None,
        help="Configuration with the skim setup.",
    )
    parser.add_argument(
        "--setup",
        required=False,
        type=str,
        default=None,
        help="Name of the setup descriptor in the configuration file.",
    )
    parser.add_argument(
        "--comp-algo",
        required=False,
        type=str,
        default="LZMA",
        help="compression algorithm",
    )
    parser.add_argument(
        "--comp-level", required=False, type=int, default=9, help="compression level"
    )
    parser.add_argument(
        "--n-threads", required=False, type=int, default=None, help="number of threads"
    )
    parser.add_argument(
        "--input-range",
        required=False,
        type=str,
        default=None,
        help="read only entries in range begin:end (before any selection)",
    )
    parser.add_argument(
        "--output-range",
        required=False,
        type=str,
        default=None,
        help="write only entries in range begin:end (after all selections)",
    )
    parser.add_argument(
        "--update-output",
        action="store_true",
        help="Update output file instead of overriding it.",
    )
    parser.add_argument(
        "--verbose", required=False, type=int, default=3, help="verbosity level"
    )
    args = parser.parse_args()

    all_inputs = []
    if args.input.endswith(".root"):
        all_inputs.append(args.input_prefix + args.input)
    elif args.input.endswith(".txt"):
        with open(args.input, "r") as input_list:
            for name in input_list.readlines():
                name = name.strip()
                if len(name) > 0 and name[0] != "#":
                    all_inputs.append(args.input_prefix + name)
    elif os.path.isdir(args.input):
        for f in os.listdir(args.input):
            if not f.endswith(".root"):
                continue
            file_name = os.path.join(args.input, f)
            all_inputs.append(file_name)
    else:
        raise RuntimeError("Input format is not supported.")
    if args.verbose > 1:
        print("Input files:\n" + "\n\t".join(all_inputs))

    processing_module = None
    processing_function = None
    processing_arguments = None

    if args.config:
        for arg_name in [
            "input_tree",
            "output_tree",
            "other_trees",
            "hists",
            "sel",
            "invert_sel",
            "column_filters",
            "processing_module",
        ]:
            value = getattr(args, arg_name)
            if value:
                raise RuntimeError(
                    f"--config and --{arg_name} can not be specified together."
                )
        import yaml

        with open(args.config, "r") as f:
            skim_config = yaml.safe_load(f)
        if not isinstance(skim_config, dict) or len(skim_config) == 0:
            raise RuntimeError("Config file is empty or has invalid format.")
        if args.setup:
            if args.setup not in skim_config:
                raise RuntimeError(
                    f"Setup '{args.setup}' is not found in the config file."
                )
            setup = skim_config[args.setup]
        else:
            if len(skim_config) > 1:
                raise RuntimeError(
                    "Multiple setups are found in the config file, but setup name is not specified."
                )
            setup = list(skim_config.values())[0]
        setup_ref = setup
        while "base" in setup_ref:
            base_setup = skim_config[setup_ref["base"]]
            for key in base_setup:
                if key not in setup:
                    setup[key] = base_setup[key]
            setup_ref = base_setup

        input_tree = setup["input_tree"]
        output_tree = setup.get("output_tree", input_tree)
        input_trees = [input_tree]
        output_trees = [output_tree]
        other_trees = setup.get("other_trees", [])
        hists = setup.get("hists", [])
        sel = setup.get("sel", None)
        sel_ref = setup.get("sel_ref", None)
        if sel and sel_ref:
            raise RuntimeError("Both 'sel' and 'sel_ref' are specified.")
        if sel_ref:
            sel = skim_config[sel_ref]
        invert_sel = setup.get("invert_sel", False)
        column_filters = setup.get("column_filters", [])
        if "processing_module" in setup:
            processing_module_entry = setup["processing_module"]["file"]
            config_path = os.path.dirname(args.config)
            if type(processing_module_entry) is str:
                processing_module_entry = [processing_module_entry]
            for file in processing_module_entry:
                if file.startswith("."):
                    file = os.path.join(config_path, file)
                if os.path.exists(file):
                    processing_module = file
                    break
            if processing_module is None:
                raise RuntimeError(
                    f"Processing module file is not found in [ {', '.join(processing_module_entry)} ]."
                )
            processing_function = setup["processing_module"]["function"]
            processing_arguments = setup["processing_module"].get("arguments", [])
    else:
        if args.setup:
            raise RuntimeError("Setup name is specified, but config file is not.")
        if not args.input_tree:
            raise RuntimeError("Input tree is not specified.")
        input_trees = args.input_tree.split(",")
        if len(input_trees) > 1:
            if args.output_tree:
                raise RuntimeError(
                    "Output tree renaming is not supported when multiple input trees are provided."
                )
            output_trees = input_trees
        else:
            output_trees = [args.output_tree if args.output_tree else args.input_tree]
        sel = args.sel
        invert_sel = args.invert_sel

        def get_list(arg):
            arg_value = getattr(args, arg)
            values = []
            if arg_value:
                for v in arg_value.split(","):
                    v = v.strip()
                    if len(v) > 0:
                        values.append(v)
            return values

        other_trees = get_list("other_trees")
        hists = get_list("hists")
        column_filters = get_list("column_filters")
        if args.processing_module:
            processing_module, processing_function = args.processing_module.split(":")
            processing_arguments = args.processing_arguments

    input_range = (
        [int(x) for x in args.input_range.split(":")] if args.input_range else None
    )
    output_range = (
        [int(x) for x in args.output_range.split(":")] if args.output_range else None
    )

    if args.n_threads is None:
        if args.input_range is not None or args.output_range is not None:
            args.n_threads = 1
        else:
            args.n_threads = 4
    if args.n_threads > 1:
        if args.verbose > 0:
            print(f"Using {args.n_threads} threads.")
        ROOT.ROOT.EnableImplicitMT(args.n_threads)

    inputs = select_inputs(
        all_inputs,
        input_trees + other_trees + hists,
        args.ignore_absent,
        args.skip_empty,
        args.verbose,
    )

    opt = ROOT.RDF.RSnapshotOptions()
    opt.fCompressionAlgorithm = getattr(ROOT.ROOT, "k" + args.comp_algo)
    opt.fCompressionLevel = args.comp_level
    if args.update_output:
        opt.fMode = "UPDATE"

    for input_idx in range(len(input_trees)):
        input_tree = input_trees[input_idx]
        output_tree = output_trees[input_idx]
        print(f"Processing {input_tree} -> {output_tree}...")
        skim_tree(
            inputs=inputs[input_tree],
            input_tree=input_tree,
            output=args.output,
            output_tree=output_tree,
            input_range=input_range,
            output_range=output_range,
            column_filters=column_filters,
            sel=sel,
            invert_sel=invert_sel,
            processing_module=processing_module,
            processing_function=processing_function,
            processing_arguments=processing_arguments,
            snapshot_options=opt,
            verbose=args.verbose,
        )
        opt.fMode = "UPDATE"

    for tree_name in other_trees:
        copy_tree(
            inputs=inputs[tree_name],
            tree_name=tree_name,
            output=args.output,
            snapshot_options=opt,
            verbose=args.verbose,
        )

    if len(hists) > 0:
        copy_histograms(
            all_inputs=all_inputs,
            inputs=inputs,
            hists=hists,
            output=args.output,
            snapshot_options=opt,
            verbose=args.verbose,
        )

    if processing_module:
        module = processing_modules.get(processing_module)
        if module and hasattr(module, "OnSkimFinish"):
            if args.verbose > 0:
                print(f"Running {processing_module}:OnSkimFinish() ...")
            module.OnSkimFinish()

    if args.verbose > 0:
        print("Skim successfully finished.")
