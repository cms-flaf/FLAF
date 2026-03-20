import os
import json
import math
import numpy as np


class InputFile:
    def __init__(self, name, nEvents, eraLetter, eraVersion, run_lumi_ranges):
        self.name = name
        self.nEvents = nEvents
        self.eraLetter = eraLetter
        self.eraVersion = eraVersion
        self.run_lumi = {}
        for run, lumi_ranges in run_lumi_ranges.items():
            if run not in self.run_lumi:
                self.run_lumi[run] = set()
            for lumi_range in lumi_ranges:
                for lumi in range(lumi_range[0], lumi_range[1] + 1):
                    self.run_lumi[run].add(lumi)


class InputBlock:
    def __init__(self):
        self.files = set()
        self.eraLetter = None
        self.eraVersion = None
        self.run_lumi = {}

    def add(self, file):
        if self.eraLetter is None and self.eraVersion is None:
            self.eraLetter = file.eraLetter
            self.eraVersion = file.eraVersion
        elif self.eraLetter != file.eraLetter or self.eraVersion != file.eraVersion:
            raise RuntimeError(
                f'Input file "{file.name}" has a different era than the other files in the block.'
            )
        self.files.add(file)
        for run, lumis in file.run_lumi.items():
            if run not in self.run_lumi:
                self.run_lumi[run] = set()
            self.run_lumi[run].update(lumis)

    @property
    def nEvents(self):
        return sum([f.nEvents for f in self.files])

    def hasOverlap(self, other):
        for run, lumis in other.run_lumi.items():
            if run not in self.run_lumi:
                continue
            if len(lumis & self.run_lumi[run]) > 0:
                return True
        return False

    def canMerge(self, other):
        return self.eraLetter == other.eraLetter and self.eraVersion == other.eraVersion

    @staticmethod
    def merge(*blocks):
        merged_block = InputBlock()
        for block in blocks:
            for file in block.files:
                merged_block.add(file)
        return merged_block

    @staticmethod
    def create(input_files):
        blocks = []
        for file in input_files:
            block = InputBlock()
            block.add(file)
            blocks.append(block)

        had_merge = True
        while had_merge:
            new_blocks = []
            processed_indices = set()
            had_merge = False
            for block_idx, block in enumerate(blocks):
                if block_idx in processed_indices:
                    continue
                overlap_idx = []
                overlap_blocks = []
                for other_idx in range(block_idx + 1, len(blocks)):
                    if other_idx in processed_indices:
                        continue
                    other = blocks[other_idx]
                    if block.hasOverlap(other):
                        overlap_blocks.append(other)
                        overlap_idx.append(other_idx)
                processed_indices.add(block_idx)
                if len(overlap_blocks) == 0:
                    new_blocks.append(block)
                else:
                    merged_block = InputBlock.merge(block, *overlap_blocks)
                    new_blocks.append(merged_block)
                    processed_indices.update(overlap_idx)
                    had_merge = True
            blocks = new_blocks

        print(f"Created {len(blocks)} input blocks:")
        processed_inputs = set()
        has_overlaps = False
        for block_idx, block in enumerate(blocks):
            print(
                f"  block #{block_idx}: {len(block.files)} files, {block.nEvents} events total"
            )
            for file in block.files:
                print(f"    {file.name} ({file.nEvents} events)")
                if file in processed_inputs:
                    has_overlaps = True
                processed_inputs.add(file)
        if has_overlaps:
            raise RuntimeError(
                "Error while creating input blocks. Some input files are shared between blocks."
            )
        if processed_inputs != input_files:
            raise RuntimeError(
                "Error while creating input blocks. Some input files are missing."
            )
        return blocks


def ToRunLumiRanges(run_lumi):
    run_lumi_ranges = {}
    for run, lumis in run_lumi.items():
        if len(lumis) == 0:
            continue
        sorted_lumis = sorted(lumis)
        lumi_ranges = []
        current_range = [sorted_lumis[0], sorted_lumis[0]]
        for lumi in sorted_lumis[1:]:
            if lumi == current_range[1] + 1:
                current_range[1] = lumi
            else:
                lumi_ranges.append(current_range)
                current_range = [lumi, lumi]
        lumi_ranges.append(current_range)
        run_lumi_ranges[run] = lumi_ranges
    return run_lumi_ranges


class Output:
    def __init__(self):
        self.inputs = set()
        self.size = 0
        self.n_splits = 1

    def add(self, input_idx, input_size):
        if input_idx in self.inputs:
            raise RuntimeError(f"Input #{input_idx} is already added to the output.")
        self.inputs.add(input_idx)
        self.size += input_size

    def remove(self, input_idx, input_size):
        if input_idx not in self.inputs:
            raise RuntimeError(f"Input #{input_idx} is not part of the output.")
        self.inputs.remove(input_idx)
        self.size -= input_size


def CreateMergeSchema(
    input_sizes, target_output_size, allow_multiple_outputs_per_block
):
    inputs = sorted(range(len(input_sizes)), key=lambda i: -input_sizes[i])
    outputs = []
    processed_inputs = set()
    flexible_inputs = set()

    def output_metric(size, n_splits, n_inputs):
        active = 1 if n_inputs > 0 else 0
        if size == 0:
            delta = 0
        else:
            split_size = int(math.ceil(size / n_splits))
            delta = n_splits * abs(split_size - target_output_size)
        return np.array([delta, delta**2, active], dtype=np.int64)

    def combined_metric():
        cmb = np.array([0, 0, 0], dtype=np.int64)
        for output in outputs:
            cmb += output_metric(output.size, output.n_splits, len(output.inputs))
        return cmb

    while len(processed_inputs) != len(input_sizes):
        output = Output()
        for input_idx in inputs:
            if input_idx in processed_inputs:
                continue
            input_size = input_sizes[input_idx]
            if (
                len(output.inputs) == 0
                or output.size + input_size <= target_output_size
            ):
                output.add(input_idx, input_size)
        if allow_multiple_outputs_per_block:
            prev_metric = output_metric(
                output.size, output.n_splits, len(output.inputs)
            )
            while True:
                new_metric = output_metric(
                    output.size, output.n_splits + 1, len(output.inputs)
                )
                if tuple(new_metric) >= tuple(prev_metric):
                    break
                output.n_splits += 1
                prev_metric = new_metric
        if output.n_splits == 1:
            flexible_inputs.update(output.inputs)
        outputs.append(output)
        processed_inputs.update(output.inputs)

    def optimization_step():
        for input_idx in flexible_inputs:
            original_output = next(
                output for output in outputs if input_idx in output.inputs
            )
            input_size = input_sizes[input_idx]
            old_origin_metric = output_metric(
                original_output.size,
                original_output.n_splits,
                len(original_output.inputs),
            )
            new_origin_metric = output_metric(
                original_output.size - input_size,
                original_output.n_splits,
                len(original_output.inputs) - 1,
            )
            for output in outputs:
                if output == original_output or output.n_splits > 1:
                    continue
                old_target_metric = output_metric(
                    output.size, output.n_splits, len(output.inputs)
                )
                new_target_metric = output_metric(
                    output.size + input_size, output.n_splits, len(output.inputs) + 1
                )
                if (
                    new_origin_metric + new_target_metric
                    < old_origin_metric + old_target_metric
                ):
                    original_output.remove(input_idx, input_size)
                    output.add(input_idx, input_size)
                    return True
        return False

    prev_metric = combined_metric()
    while optimization_step():
        new_metric = combined_metric()
        if tuple(new_metric) >= tuple(prev_metric):
            raise RuntimeError(
                "Error in merge schema optimization. Metric did not improve after modification."
            )
        prev_metric = new_metric
    merge_schema = [
        (output.inputs, output.n_splits) for output in outputs if len(output.inputs) > 0
    ]
    return merge_schema


def CreateMergePlan(setup, local_inputs, n_events_per_file, is_data):
    """Create a merge plan for either data or MC.

    Args:
        setup (Setup): FLAF setup object
        local_inputs (list[str]): list of input report file paths
        n_events_per_file (int): an aproximate number of events per output file. The goal is to have output files with a number of events close to this value, so it is not guaranteed that the actual number of events is less or equal to this value.
        is_data (bool): data or MC

    Returns:
        dict: merge plan and combined reports
    """

    combined_reports = {}
    input_files = set()
    for report in local_inputs:
        with open(report, "r") as file:
            data = json.load(file)
        key = os.path.join(data["dataset_name"], data["anaTuple_file_name"])
        is_valid = data.get("valid", True)
        if not is_valid:
            print(f"{key}: is marked as invalid, skipping", file=os.sys.stderr)
            continue
        if key in combined_reports:
            raise ValueError(f"Duplicate report for file {key}")
        combined_reports[key] = data
        file = InputFile(
            key,
            data["n_events"],
            data.get("eraLetter", ""),
            data.get("eraVersion", ""),
            data["run_lumi_ranges"],
        )
        input_files.add(file)

    input_blocks = InputBlock.create(input_files)
    block_groups = {}
    for block in input_blocks:
        eraLetter = block.eraLetter
        eraVersion = block.eraVersion
        if (eraLetter, eraVersion) not in block_groups:
            block_groups[(eraLetter, eraVersion)] = []
        block_groups[(eraLetter, eraVersion)].append(block)

    merge_plan = []
    input_file_names = set([file.name for file in input_files])
    processed_input_file_names = set()
    for (eraLetter, eraVersion), blocks in block_groups.items():
        block_sizes = [block.nEvents for block in blocks]
        merge_schema = CreateMergeSchema(
            block_sizes, n_events_per_file, allow_multiple_outputs_per_block=is_data
        )
        output_idx = 0
        output_format = "anaTuple_"
        if eraLetter != "" or eraVersion != "":
            output_format += f"{eraLetter}{eraVersion}_"
        output_format += "{}.root"
        for block_indices, n_outputs in merge_schema:
            assert n_outputs > 0
            assert len(block_indices) > 0
            blocks_to_merge = [blocks[i] for i in block_indices]
            block = InputBlock.merge(*blocks_to_merge)
            merge = {
                "inputs": [],
                "outputs": [],
                "n_events": block.nEvents,
                "run_lumi_ranges": ToRunLumiRanges(block.run_lumi),
            }
            for file in block.files:
                if file.name in processed_input_file_names:
                    raise RuntimeError(
                        f'File "{file.name}" is duplicated in the input files.'
                    )
                merge["inputs"].append(file.name)
                processed_input_file_names.add(file.name)
            for i in range(n_outputs):
                output_name = output_format.format(output_idx)
                merge["outputs"].append(output_name)
                output_idx += 1
            merge_plan.append(merge)
    if processed_input_file_names != input_file_names:
        missing_files = input_file_names - processed_input_file_names
        raise RuntimeError(
            f"Some input files were not scheduled for merging: {missing_files}"
        )

    return {"plan": merge_plan, "reports": combined_reports}
