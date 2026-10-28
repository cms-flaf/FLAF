#!/usr/bin/env python3
"""NN inference in FLAF jobs gives the same outputs on every batch node it can.

TensorFlow runs its Eigen kernels on every CPU: `env.sh` sets `TF_ENABLE_ONEDNN_OPTS=0` for
flaf_env, and `Setup.cmssw_env`, which starts from an empty environment, sets it for CMSSW.
ONNX Runtime sessions are created by `FLAF.Common.OnnxSession` with one thread, whatever the
core count of the host.
"""

import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from FLAF.Common import OnnxSession  # noqa: E402


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# Setup imports ROOT through FLAF.Common.Utilities without using it on import; the unit-test
# runner has no ROOT. A stand-in serves that import alone and is removed again.
_root_placeholder = "ROOT" not in sys.modules and not _importable("ROOT")
if _root_placeholder:
    sys.modules["ROOT"] = mock.MagicMock()
try:
    from FLAF.Common import Setup as flaf_setup  # noqa: E402
finally:
    if _root_placeholder:
        sys.modules.pop("ROOT", None)

#: env.sh is sourced from bash and from zsh; under CI a missing zsh fails the zsh case
_SHELLS = ["bash", "zsh"] if shutil.which("zsh") or os.environ.get("CI") else ["bash"]


class EnvShTurnsOneDnnOff(unittest.TestCase):
    """The real env.sh, sourced in a stand-in analysis whose flaf_env is already built."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_nn_env_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        flaf = os.path.join(self.tmp, "FLAF")
        os.makedirs(os.path.join(flaf, "run_tools"))
        shutil.copy2(os.path.join(flaf_repo, "env.sh"), os.path.join(flaf, "env.sh"))
        mk_flaf_env = os.path.join(flaf, "run_tools", "mk_flaf_env.sh")
        with open(mk_flaf_env, "w") as f:
            f.write("#!/bin/bash\nexit 1\n")
        with open(mk_flaf_env, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        self.ana = os.path.join(self.tmp, "ana")
        env_path = os.path.join(self.ana, "soft", "flaf_env")
        os.makedirs(os.path.join(env_path, "bin"))
        with open(os.path.join(env_path, "bin", "activate"), "w") as f:
            f.write("true\n")
        with open(os.path.join(flaf_repo, "env.sh")) as f:
            env_sh = f.read()
        lcg, arch = (
            re.search(rf'local {name}="([^"]+)"', env_sh).group(1)
            for name in ("FLAF_LCG_VERSION", "FLAF_LCG_ARCH")
        )
        open(os.path.join(env_path, f".{lcg}_{arch}_{digest[:12]}"), "w").close()
        self.env_sh = os.path.join(flaf, "env.sh")

    def sourced_value(self, shell, preset):
        env = {
            "PATH": os.environ["PATH"],
            "HOME": self.tmp,
            "ANALYSIS_PATH": self.ana,
            "FLAF_NO_INSTALL": "1",
        }
        if preset is not None:
            env["TF_ENABLE_ONEDNN_OPTS"] = preset
        proc = subprocess.run(
            [
                shell,
                "-c",
                f'source "{self.env_sh}"\n'
                "alias cmsEnv\n"
                'echo "sourced TF_ENABLE_ONEDNN_OPTS=$(printenv TF_ENABLE_ONEDNN_OPTS)"\n',
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=self.tmp,
        )
        self.assertIn("sourced", proc.stdout, proc.stdout + proc.stderr)
        # cmsEnv starts the CMSSW environment from an empty one as well
        self.assertIn(" TF_ENABLE_ONEDNN_OPTS=0 ", proc.stdout)
        return proc.stdout.split("sourced TF_ENABLE_ONEDNN_OPTS=")[-1].strip()

    def test_exported_whatever_the_shell_had(self):
        for shell in _SHELLS:
            for preset in (None, "1"):
                with self.subTest(shell=shell, preset=preset):
                    self.assertEqual(self.sourced_value(shell, preset), "0")


#: a scratch analysis with one era on top of FLAF's own configuration
_ANALYSIS_FILES = {
    "config/global.yaml": """
phys_model: M
channelSelection: [ eE ]
histTuple_flavor: default
histTuple_flavors:
  default:
    variables: []
    fullResolution_variables: []
""",
    "config/phys_models.yaml": "M:\n  backgrounds: [ TT ]\n  data: [ data ]\n",
    "config/processes.yaml": "TT:\n  datasets: [ TTto2L2Nu ]\n",
    "config/Run3_2022/processes.yaml": "data:\n  datasets: [ Data_A ]\n",
    "config/Run3_2022/datasets.yaml": "Data_A: {}\nTTto2L2Nu: {}\n",
    "config/weights.yaml": "norm: {}\nshape: {}\n",
}


class TheCmsswEnvironmentTurnsOneDnnOff(unittest.TestCase):
    """`Setup.cmssw_env` is built from scram's output and a list of forwarded variables."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_nn_setup_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ana = os.path.join(self.tmp, "ana")
        os.makedirs(self.ana)
        os.symlink(flaf_repo, os.path.join(self.ana, "FLAF"))
        for path, text in _ANALYSIS_FILES.items():
            path = os.path.join(self.ana, path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)

    def cmssw_env(self, scram_env, environ):
        environ = dict(
            environ,
            ANALYSIS_PATH=self.ana,
            ANALYSIS_DATA_PATH=os.path.join(self.tmp, "data"),
        )
        with mock.patch.dict(os.environ, environ, clear=True):
            setup = flaf_setup.Setup(self.ana, "Run3_2022", "v1")
            with mock.patch.object(
                flaf_setup, "get_cmsenv", return_value=dict(scram_env)
            ):
                return setup.cmssw_env

    def test_set_although_the_environment_starts_empty(self):
        env = self.cmssw_env({"PATH": "/cmssw/bin"}, {})
        self.assertEqual(env["TF_ENABLE_ONEDNN_OPTS"], "0")

    def test_neither_the_shell_nor_scram_changes_it(self):
        env = self.cmssw_env(
            {"PATH": "/cmssw/bin", "TF_ENABLE_ONEDNN_OPTS": "1"},
            {"TF_ENABLE_ONEDNN_OPTS": "1"},
        )
        self.assertEqual(env["TF_ENABLE_ONEDNN_OPTS"], "0")

    def test_the_rest_is_kept(self):
        env = self.cmssw_env(
            {"PATH": "/cmssw/bin", "PYTHONPATH": "/cmssw/python"}, {"UNLISTED": "x"}
        )
        self.assertEqual(env["PATH"], "/cmssw/bin")
        self.assertEqual(env["ANALYSIS_PATH"], self.ana)
        self.assertEqual(env["PYTHONPATH"], f"{self.ana}:/cmssw/python")
        self.assertNotIn("UNLISTED", env)


def _write_model(path, n_in=64, n_out=8):
    rng = np.random.default_rng(1)
    weights = numpy_helper.from_array(
        rng.normal(size=(n_in, n_out)).astype(np.float32), "w"
    )
    bias = numpy_helper.from_array(rng.normal(size=n_out).astype(np.float32), "b")
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["xw"]),
            helper.make_node("Add", ["xw", "b"], ["y"]),
        ],
        "dense",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, n_in])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, n_out])],
        initializer=[weights, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


class OnnxSessionsUseOneThread(unittest.TestCase):
    def test_options(self):
        options = OnnxSession.session_options()
        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)

    def test_every_call_gives_options_of_its_own(self):
        """A caller that changes the options it got (DeepHME turns off the memory arena)
        changes no one else's."""
        first = OnnxSession.session_options()
        first.enable_cpu_mem_arena = False
        self.assertTrue(OnnxSession.session_options().enable_cpu_mem_arena)

    def test_session(self):
        tmp = tempfile.mkdtemp(prefix="flaf_onnx_")
        self.addCleanup(shutil.rmtree, tmp, True)
        path = os.path.join(tmp, "dense.onnx")
        _write_model(path)
        session = OnnxSession.make_session(path)
        options = session.get_session_options()
        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(session.get_providers(), ["CPUExecutionProvider"])
        x = np.ones((3, 64), dtype=np.float32)
        (y,) = session.run(None, {"x": x})
        self.assertEqual(y.shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
