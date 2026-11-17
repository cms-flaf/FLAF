"""The client side of HistMergerFromHists.py --stream, as HistMergerTask runs it: the inputs
of one process at a time are downloaded, sent to one merger process and deleted once it has
written them; then the steps that need every process. Standard library only."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

# The acknowledgement lines of HistMergerFromHists.py --stream (its STREAM_ACK) on stdout.
MERGER_ACK = "@@HistMerger"


def stream_merge(cmd, processes, fetch, all_dataset_names, log=print):
    """Run the merger ``cmd`` (HistMergerFromHists.py --stream) over ``processes``, a list of
    (process, items). ``fetch(items)`` downloads the inputs of one process and returns (a
    stack that deletes them, [(dataset_name, local path)]). The next process is fetched while
    one is merged; a process's local copies are deleted once the merger has acknowledged
    writing it. Then the merger runs the steps that need every process (the datasets
    ``all_dataset_names``, in the merge order) and exits. Every other line the merger prints
    goes to ``log``."""
    merger = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )

    def send(message, ack):
        merger.stdin.write(json.dumps(message) + "\n")
        merger.stdin.flush()
        for line in merger.stdout:
            if line.startswith(MERGER_ACK + " "):
                got = line[len(MERGER_ACK) + 1 :].strip()
                if got != ack:
                    raise RuntimeError(f"merger: expected '{ack}', got '{got}'")
                return
            log(line, end="")
        raise RuntimeError(f"merger exited with {merger.wait()} before '{ack}'")

    pending = None
    try:
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            if processes:
                pending = prefetch.submit(fetch, processes[0][1])
            for index, (process, _) in enumerate(processes):
                stack, local_inputs = pending.result()
                pending = None
                if index + 1 < len(processes):
                    pending = prefetch.submit(fetch, processes[index + 1][1])
                with stack:
                    send(
                        {"process": process, "inputs": [list(i) for i in local_inputs]},
                        f"merged {process}",
                    )
            send({"postprocess": list(all_dataset_names)}, "postprocessed")
        merger.stdin.close()
        for line in merger.stdout:
            log(line, end="")
        code = merger.wait()
        if code != 0:
            raise RuntimeError(f"merger exited with {code}")
    finally:
        # a download still running when a step failed: delete what it fetched
        if pending is not None:
            try:
                pending.result()[0].close()
            except Exception:
                pass
        if merger.poll() is None:
            merger.kill()
            merger.wait()
