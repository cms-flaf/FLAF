import heapq
import os
import json
import math


class InputFile:
    """An input anaTuple file with its lumi sections expanded into per-run lumi sets."""

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


class RunCluster:
    """Groups (file, run) pairs where files have lumi overlap for the same run.
    A file can belong to multiple clusters (with different runs per cluster)."""

    def __init__(self):
        self.files = set()
        self.eraLetter = None
        self.eraVersion = None
        self.run_lumi = {}
        self._file_runs = {}  # InputFile -> set of run numbers in this cluster

    def add_file_run(self, file, run):
        """Register (file, run) in this cluster, accumulating lumi sections."""
        if self.eraLetter is None and self.eraVersion is None:
            self.eraLetter = file.eraLetter
            self.eraVersion = file.eraVersion
        elif self.eraLetter != file.eraLetter or self.eraVersion != file.eraVersion:
            raise RuntimeError(
                f'Input file "{file.name}" has a different era than the other files in the cluster.'
            )
        self.files.add(file)
        if file not in self._file_runs:
            self._file_runs[file] = set()
        self._file_runs[file].add(run)
        if run not in self.run_lumi:
            self.run_lumi[run] = set()
        self.run_lumi[run].update(file.run_lumi[run])

    def runs_for_file(self, file):
        """Return the set of run numbers that file contributes to this cluster."""
        return self._file_runs.get(file, set())

    @property
    def nEvents(self):
        """Estimated event count: each file's events weighted by its lumi fraction in this cluster."""
        total = 0
        for file in self.files:
            file_runs_in_cluster = self._file_runs.get(file, set())
            total_file_lumis = sum(len(lumis) for lumis in file.run_lumi.values())
            cluster_file_lumis = sum(
                len(file.run_lumi[r])
                for r in file_runs_in_cluster
                if r in file.run_lumi
            )
            if total_file_lumis > 0:
                total += int(file.nEvents * cluster_file_lumis / total_file_lumis)
        return total

    @staticmethod
    def merge(*clusters):
        """Return a new cluster containing all (file, run) pairs from the given clusters."""
        merged = RunCluster()
        for cluster in clusters:
            for file in cluster.files:
                for run in cluster.runs_for_file(file):
                    merged.add_file_run(file, run)
        return merged

    @staticmethod
    def create(input_files):
        """For each run, group files by lumi overlap within that run.
        Files that overlap for a given run go into the same cluster for that run.
        Clusters with identical file sets are merged (accumulating runs).
        A file can appear in multiple clusters with different runs."""
        all_runs = set()
        for file in input_files:
            all_runs.update(file.run_lumi.keys())

        cluster_map = {}  # frozenset(files) -> RunCluster

        for run in sorted(all_runs):
            files_with_run = [f for f in input_files if run in f.run_lumi]
            if not files_with_run:
                continue

            parent = {f: f for f in files_with_run}

            def find(x):
                while parent[x] != x:
                    x = parent[x]
                return x

            def union(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry

            # Build lumi->files map: O(total_lumis) instead of O(n_files^2 * lumi_set_size)
            lumi_to_files = {}
            for f in files_with_run:
                for lumi in f.run_lumi[run]:
                    if lumi not in lumi_to_files:
                        lumi_to_files[lumi] = f  # store first file
                    else:
                        union(
                            lumi_to_files[lumi], f
                        )  # union with first file seen for this lumi

            groups = {}
            for f in files_with_run:
                root = find(f)
                if root not in groups:
                    groups[root] = []
                groups[root].append(f)

            for file_group in groups.values():
                key = frozenset(file_group)
                if key not in cluster_map:
                    cluster_map[key] = RunCluster()
                cluster = cluster_map[key]
                for file in file_group:
                    cluster.add_file_run(file, run)

        clusters = list(cluster_map.values())

        seen_file_runs = set()
        for cluster in clusters:
            for file in cluster.files:
                for run in cluster.runs_for_file(file):
                    pair = (file.name, run)
                    if pair in seen_file_runs:
                        raise RuntimeError(
                            f'(file, run) pair "{pair}" appears in multiple clusters.'
                        )
                    seen_file_runs.add(pair)

        for file in input_files:
            for run in file.run_lumi:
                if (file.name, run) not in seen_file_runs:
                    raise RuntimeError(
                        f'(file, run) pair ("{file.name}", "{run}") is missing from clusters.'
                    )

        print(f"Created {len(clusters)} run clusters:")
        for cluster_idx, cluster in enumerate(clusters):
            print(
                f"  cluster #{cluster_idx}: {len(cluster.files)} files, "
                f"{len(cluster.run_lumi)} runs, {cluster.nEvents} events (estimated)"
            )
        return clusters


def ToRunLumiRanges(run_lumi):
    """Convert {run: set_of_lumis} to the [[start, end], ...] range format used in JSON output."""
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
    """One output group in a merge schema: a set of input indices, their combined size,
    and the number of output files this group will be split into."""

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
    *,
    input_sizes,
    target_output_size,
    allow_multiple_outputs_per_block,
    max_steps,
    cluster_files=None,
    cluster_runs=None,
    cluster_owned=None,
    file_all_runs=None,
):
    """Assign inputs to output groups so that each group's total size is close to target_output_size.

    Both data and MC use LPT (Longest Processing Time First) for initial placement: inputs are
    sorted largest-first and each is assigned to the currently lightest output, producing a
    near-optimal balanced solution in O(n log n).

    When allow_multiple_outputs_per_block=True (data): after LPT, any single-input output whose
    size already exceeds target_output_size may be assigned n_splits > 1 if that improves the
    metric. Multi-input outputs always keep n_splits = 1 — splitting is never used as a way to
    handle combined outputs that happen to be large. n_splits > 1 outputs are excluded from
    coordinate-descent optimization.

    When allow_multiple_outputs_per_block=False (MC): n_splits is always 1; all inputs are
    eligible for optimization.

    The metric minimized is (sum(|delta|), n_groups, sum(delta^2)) where
    delta = n_splits * |ceil(size/n_splits) - target_output_size| for each output group.

    Contamination (data only): when cluster_files/cluster_runs/cluster_owned/file_all_runs are
    provided, both LPT placement and coordinate-descent moves are constrained so that no output
    group can end up with a file F and a run R in its assignments unless (F, R) is owned by one
    of its clusters. This prevents the flat "run == R" filter in MergeAnaTuples from pulling
    extra events from a file that happens to be shared across output groups.

    Returns a list of (input_index_set, n_splits) pairs.
    """
    contamination_aware = cluster_files is not None

    inputs = sorted(range(len(input_sizes)), key=lambda i: -input_sizes[i])

    def output_metric(size, n_splits, n_inputs):
        if n_inputs == 0:
            return (0, 0, 0)
        split_size = math.ceil(size / n_splits)
        delta = n_splits * abs(split_size - target_output_size)
        return (delta, 1, delta**2)

    def add_metrics(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def combined_metric():
        d, act, d2 = 0, 0, 0
        for output in outputs:
            m = output_metric(output.size, output.n_splits, len(output.inputs))
            d += m[0]
            act += m[1]
            d2 += m[2]
        return (d, act, d2)

    def metric_to_str(metric):
        delta, groups, delta2 = metric[0], metric[1], metric[2]
        return f"sum(delta)={delta}, n_groups={groups}, sum(delta^2)={delta2}"

    total_size = sum(input_sizes)
    n_out = max(1, math.ceil(total_size / target_output_size))
    outputs = [Output() for _ in range(n_out)]

    if contamination_aware:
        # Per-output tracking for contamination enforcement.
        # out_owned[i]: (file_name, run) pairs whose events are assigned to output i.
        # out_file_cnt[i]: {file_name: cluster_count} — how many clusters in i reference this file.
        # out_run_cnt[i]: {run: cluster_count} — how many clusters in i reference this run.
        out_owned = [set() for _ in range(n_out)]
        out_file_cnt = [{} for _ in range(n_out)]
        out_run_cnt = [{} for _ in range(n_out)]

        def _register(input_idx, out_idx):
            out_owned[out_idx].update(cluster_owned[input_idx])
            for f in cluster_files[input_idx]:
                out_file_cnt[out_idx][f] = out_file_cnt[out_idx].get(f, 0) + 1
            for r in cluster_runs[input_idx]:
                out_run_cnt[out_idx][r] = out_run_cnt[out_idx].get(r, 0) + 1

        def _unregister(input_idx, out_idx):
            out_owned[out_idx].difference_update(cluster_owned[input_idx])
            for f in cluster_files[input_idx]:
                out_file_cnt[out_idx][f] -= 1
            for r in cluster_runs[input_idx]:
                out_run_cnt[out_idx][r] -= 1

        def _contaminated_add(input_idx, out_idx):
            """Return True if adding cluster input_idx to output out_idx would create contamination.

            Contamination arises when the output would contain both file F and run R but not own
            the (F, R) pair — so the "run == R" filter would incorrectly include F's R events.
            We check two directions:
              (a) new files (from the cluster) against runs already present in the output, and
              (b) new runs (from the cluster) against files already present in the output.
            The new_owned set covers all (file, run) pairs that would be owned after the add.
            """
            new_owned = out_owned[out_idx] | cluster_owned[input_idx]
            for f in cluster_files[input_idx]:
                f_runs = file_all_runs.get(f, frozenset())
                for r, cnt in out_run_cnt[out_idx].items():
                    if cnt > 0 and r in f_runs and (f, r) not in new_owned:
                        return True
            for r in cluster_runs[input_idx]:
                for f, cnt in out_file_cnt[out_idx].items():
                    if (
                        cnt > 0
                        and r in file_all_runs.get(f, frozenset())
                        and (f, r) not in new_owned
                    ):
                        return True
            return False

        def _contaminated_remove(input_idx, out_idx):
            """Return True if removing cluster input_idx from output out_idx would create contamination.

            Removal exposes contamination when a (file, run) pair currently owned by this cluster
            would remain referenced by both a remaining file and a remaining run in the output
            (meaning both that file and that run are contributed by other clusters too).
            """
            for f, r in cluster_owned[input_idx]:
                if (
                    out_file_cnt[out_idx].get(f, 0) > 1
                    and out_run_cnt[out_idx].get(r, 0) > 1
                ):
                    return True
            return False

        # Contamination-aware LPT: assign each cluster (largest first) to the lightest output
        # that does not create contamination. If no existing output is compatible, open a new one.
        for input_idx in inputs:
            best_out_idx = None
            best_size = float("inf")
            for out_idx, output in enumerate(outputs):
                if (
                    not _contaminated_add(input_idx, out_idx)
                    and output.size < best_size
                ):
                    best_out_idx = out_idx
                    best_size = output.size
            if best_out_idx is None:
                best_out_idx = len(outputs)
                outputs.append(Output())
                out_owned.append(set())
                out_file_cnt.append({})
                out_run_cnt.append({})
            outputs[best_out_idx].add(input_idx, input_sizes[input_idx])
            _register(input_idx, best_out_idx)
    else:
        # Standard heap-based LPT for MC.
        heap = [(0, i) for i in range(n_out)]
        for input_idx in inputs:
            _, out_idx = heapq.heappop(heap)
            outputs[out_idx].add(input_idx, input_sizes[input_idx])
            heapq.heappush(heap, (outputs[out_idx].size, out_idx))

    # Build a stable output-index map (outputs list does not change during optimization).
    output_to_idx = {id(o): i for i, o in enumerate(outputs)}

    # For data: a single oversized cluster may be split (n_splits > 1) if that improves the
    # metric. Multi-input outputs always keep n_splits = 1.
    if allow_multiple_outputs_per_block:
        for output in outputs:
            if len(output.inputs) != 1:
                continue
            prev_metric = output_metric(
                output.size, output.n_splits, len(output.inputs)
            )
            while True:
                new_metric = output_metric(
                    output.size, output.n_splits + 1, len(output.inputs)
                )
                if new_metric >= prev_metric:
                    break
                output.n_splits += 1
                prev_metric = new_metric

    flexible_inputs = {
        idx for output in outputs for idx in output.inputs if output.n_splits == 1
    }
    input_to_output = {idx: out for out in outputs for idx in out.inputs}

    def optimization_step():
        """Full coordinate-descent pass: for every flexible input find and make the best move."""
        any_move = False
        for input_idx in sorted(flexible_inputs, key=lambda i: -input_sizes[i]):
            current_output = input_to_output[input_idx]
            cur_idx = output_to_idx[id(current_output)]
            input_size = input_sizes[input_idx]
            # Skip if removing this cluster from its current output would expose contamination.
            if contamination_aware and _contaminated_remove(input_idx, cur_idx):
                continue
            old_origin_metric = output_metric(
                current_output.size, current_output.n_splits, len(current_output.inputs)
            )
            new_origin_metric = output_metric(
                current_output.size - input_size,
                current_output.n_splits,
                len(current_output.inputs) - 1,
            )
            best_target = None
            best_target_idx = None
            best_new_combined = None
            for candidate in outputs:
                if candidate is current_output or candidate.n_splits > 1:
                    continue
                cand_idx = output_to_idx[id(candidate)]
                if contamination_aware and _contaminated_add(input_idx, cand_idx):
                    continue
                old_target_metric = output_metric(
                    candidate.size, candidate.n_splits, len(candidate.inputs)
                )
                new_target_metric = output_metric(
                    candidate.size + input_size,
                    candidate.n_splits,
                    len(candidate.inputs) + 1,
                )
                old_combined = add_metrics(old_origin_metric, old_target_metric)
                new_combined = add_metrics(new_origin_metric, new_target_metric)
                if new_combined < old_combined:
                    if best_new_combined is None or new_combined < best_new_combined:
                        best_target = candidate
                        best_target_idx = cand_idx
                        best_new_combined = new_combined
            if best_target is not None:
                current_output.remove(input_idx, input_size)
                best_target.add(input_idx, input_size)
                input_to_output[input_idx] = best_target
                if contamination_aware:
                    _unregister(input_idx, cur_idx)
                    _register(input_idx, best_target_idx)
                any_move = True
        return any_move

    prev_metric = combined_metric()
    print(
        f"CreateMergeSchema: metric for the initial merge solution: {metric_to_str(prev_metric)}"
    )
    step = 0
    while optimization_step():
        new_metric = combined_metric()
        if new_metric >= prev_metric:
            raise RuntimeError(
                "Error in merge schema optimization. Metric did not improve after modification."
            )
        prev_metric = new_metric
        step += 1
        print(f"CreateMergeSchema: step {step}. metric: {metric_to_str(new_metric)}")
        if step >= max_steps:
            print(
                f"CreateMergeSchema: reached the maximum number of steps ({max_steps})."
            )
            break
    print(
        f"CreateMergeSchema: optimization finished after {step} steps. Final metric: {metric_to_str(prev_metric)}"
    )
    merge_schema = [
        (output.inputs, output.n_splits) for output in outputs if len(output.inputs) > 0
    ]
    return merge_schema


def CreateMergePlan(
    *,
    n_events_per_file,
    is_data,
    setup=None,
    local_inputs=None,
    combined_report_path=None,
    max_steps=1000,
):
    """Create a merge plan for either data or MC.

    Args:
        n_events_per_file (int): target event count per output file (approximate).
        is_data (bool): True for collision data, False for MC.
        setup: FLAF setup object (required when local_inputs is used).
        local_inputs (list[str]): paths to per-file anaTuple report JSONs.
        combined_report_path (str): path to a pre-merged combined report JSON.
            Exactly one of local_inputs or combined_report_path must be supplied.
        max_steps (int): maximum coordinate-descent steps in CreateMergeSchema.

    Returns:
        dict with keys "plan" (list of merge items) and "reports" (combined report dict).
    """

    if local_inputs is None:
        if combined_report_path is None:
            raise ValueError(
                "Either local_inputs or combined_report_path must be provided."
            )
        with open(combined_report_path, "r") as file:
            combined_reports = json.load(file)
    else:
        if combined_report_path is not None:
            raise ValueError(
                "Only one of local_inputs or combined_report_path can be provided."
            )
        if setup is None:
            raise ValueError("Setup must be provided when local_inputs is used.")

        combined_reports = {}
        for report in local_inputs:
            with open(report, "r") as file:
                data = json.load(file)
            key = os.path.join(data["dataset_name"], data["anaTuple_file_name"])
            dataset = setup.datasets[data["dataset_name"]]
            eraLetter = dataset.get("eraLetter", "")
            if len(eraLetter) > 0:
                data["eraLetter"] = eraLetter
            eraVersion = dataset.get("eraVersion", "")
            if len(eraVersion) > 0:
                data["eraVersion"] = eraVersion
            is_valid = data.get("valid", True)
            if not is_valid:
                print(f"{key}: is marked as invalid, skipping", file=os.sys.stderr)
                continue
            if key in combined_reports:
                raise ValueError(f"Duplicate report for file {key}")
            combined_reports[key] = data

    input_files = set()
    for key, data in combined_reports.items():
        file = InputFile(
            key,
            data["n_events"],
            data.get("eraLetter", ""),
            data.get("eraVersion", ""),
            data["run_lumi_ranges"],
        )
        input_files.add(file)

    run_clusters = RunCluster.create(input_files)
    cluster_groups = {}
    for cluster in run_clusters:
        eraLetter = cluster.eraLetter
        eraVersion = cluster.eraVersion
        if (eraLetter, eraVersion) not in cluster_groups:
            cluster_groups[(eraLetter, eraVersion)] = []
        cluster_groups[(eraLetter, eraVersion)].append(cluster)

    # For data, precompute each file's complete run set so the contamination check in
    # CreateMergeSchema can determine which (file, run) combinations must be owned together.
    if is_data:
        file_all_runs_map = {f.name: frozenset(f.run_lumi.keys()) for f in input_files}

    merge_plan = []
    input_file_runs = set()
    for file in input_files:
        for run in file.run_lumi:
            input_file_runs.add((file.name, run))
    processed_file_runs = set()
    for (eraLetter, eraVersion), clusters in cluster_groups.items():
        cluster_sizes = [cluster.nEvents for cluster in clusters]
        report_suffix = (
            f" for era {eraLetter}{eraVersion}"
            if eraLetter != "" or eraVersion != ""
            else ""
        )
        print(f"Creating merge schema{report_suffix}")
        if is_data:
            # Build per-cluster metadata so CreateMergeSchema can enforce the contamination
            # constraint: no output group may contain both file F and run R unless it owns (F, R).
            cf = [frozenset(f.name for f in c.files) for c in clusters]
            cr = [frozenset(c.run_lumi.keys()) for c in clusters]
            co = [
                frozenset((f.name, r) for f in c.files for r in c.runs_for_file(f))
                for c in clusters
            ]
            merge_schema = CreateMergeSchema(
                input_sizes=cluster_sizes,
                target_output_size=n_events_per_file,
                allow_multiple_outputs_per_block=True,
                max_steps=max_steps,
                cluster_files=cf,
                cluster_runs=cr,
                cluster_owned=co,
                file_all_runs=file_all_runs_map,
            )
        else:
            merge_schema = CreateMergeSchema(
                input_sizes=cluster_sizes,
                target_output_size=n_events_per_file,
                allow_multiple_outputs_per_block=False,
                max_steps=max_steps,
            )
        print(f"Merge schema{report_suffix} created")
        output_idx = 0
        output_format = "anaTuple_"
        if eraLetter != "" or eraVersion != "":
            output_format += f"{eraLetter}{eraVersion}_"
        output_format += "{}.root"
        for cluster_indices, n_outputs in merge_schema:
            assert n_outputs > 0
            assert len(cluster_indices) > 0
            clusters_to_merge = [clusters[i] for i in cluster_indices]
            merged_cluster = RunCluster.merge(*clusters_to_merge)
            item_runs = set()
            item_owned = set()
            for file in merged_cluster.files:
                for run in merged_cluster.runs_for_file(file):
                    pair = (file.name, run)
                    if pair in processed_file_runs:
                        raise RuntimeError(
                            f'(file, run) pair "{pair}" is duplicated in the merge plan.'
                        )
                    processed_file_runs.add(pair)
                    item_runs.add(run)
                    item_owned.add(pair)
            if is_data:
                for file in merged_cluster.files:
                    for run in item_runs:
                        if run in file.run_lumi and (file.name, run) not in item_owned:
                            raise RuntimeError(
                                f"Run contamination in merge plan: file {file.name!r} has data "
                                f"for run {run!r} which is in this plan item but not owned by it."
                            )
            merge = {
                "inputs": sorted(file.name for file in merged_cluster.files),
                "runs": sorted(item_runs),
                "outputs": [],
                "n_events": merged_cluster.nEvents,
                "run_lumi_ranges": ToRunLumiRanges(merged_cluster.run_lumi),
            }
            for i in range(n_outputs):
                output_name = output_format.format(output_idx)
                merge["outputs"].append(output_name)
                output_idx += 1
            merge_plan.append(merge)
    if processed_file_runs != input_file_runs:
        missing = input_file_runs - processed_file_runs
        raise RuntimeError(
            f"Some (file, run) pairs were not scheduled for merging: {missing}"
        )

    return {"plan": merge_plan, "reports": combined_reports}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-reports", required=True, type=str)
    parser.add_argument("--n-events-per-file", required=True, type=int)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--max-steps", required=False, type=int, default=1000)
    parser.add_argument("--is-data", action="store_true")
    args = parser.parse_args()

    result = CreateMergePlan(
        n_events_per_file=args.n_events_per_file,
        is_data=args.is_data,
        combined_report_path=args.combined_reports,
        max_steps=args.max_steps,
    )

    with open(args.output, "w") as file:
        json.dump(result["plan"], file, indent=2)
