"""ONNX Runtime sessions whose outputs do not depend on the core count of the batch node.

A session left at the defaults of ONNX Runtime sizes its thread pool by the physical cores of the
host, whatever the slot of the job and its affinity mask, and how the work is split across the
threads changes the last bits of some outputs. With one thread the split is the same on every node.
"""

import onnxruntime as ort


def session_options():
    """New `SessionOptions` with one intra-op and one inter-op thread.

    For code that creates its sessions itself (DeepHME); further options may be set on them.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def make_session(model_path):
    """A CPU `InferenceSession` of the model at `model_path`, created with `session_options()`."""
    return ort.InferenceSession(
        model_path, sess_options=session_options(), providers=["CPUExecutionProvider"]
    )
