#!/usr/bin/env python3
"""FLAF runs with exactly one law release, and every place that installs or ships law agrees.

law 0.1.21 renamed the CRAB job manager's `proxy` keyword, changed which CRAB server states it
accepts and rewrote the run block of its job script. Code adapted to it fails in silence under
0.1.20 and the reverse, so the release is pinned once per recipe (`LAW_VERSION` in
`run_tools/law_customizations.py`, `FLAF_LAW_VERSION` in `env.sh`, the CI workflows), an import
under any other release is refused, an existing environment is brought to the pin when env.sh is
sourced, and the unhashed environment bundle is named after the law it carries.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock

flaf_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
flaf_parent = os.path.dirname(flaf_repo)
if flaf_parent not in sys.path:
    sys.path.insert(0, flaf_parent)

import law  # noqa: E402

# The unit-test runner has no ROOT, and law_customizations reaches FLAF.Common.Utilities,
# which imports it at module level without using it on import. A placeholder stands in for
# that import only and is removed again.
_root_placeholder = (
    "ROOT" not in sys.modules and importlib.util.find_spec("ROOT") is None
)
if _root_placeholder:
    sys.modules["ROOT"] = types.ModuleType("ROOT")
try:
    from FLAF.run_tools import law_customizations as lc  # noqa: E402
finally:
    if _root_placeholder:
        sys.modules.pop("ROOT", None)


def _read(*parts):
    with open(os.path.join(flaf_repo, *parts)) as f:
        return f.read()


class ThePinAgreesEverywhere(unittest.TestCase):
    def test_env_sh_pins_the_release_the_code_requires(self):
        pins = re.findall(r'local FLAF_LAW_VERSION="([^"]+)"', _read("env.sh"))
        self.assertEqual(pins, [lc.LAW_VERSION])

    def test_the_ci_workflows_install_that_release(self):
        for workflow in ("unit-tests.yaml", "test-setup-loading.yaml"):
            with self.subTest(workflow):
                content = _read(".github", "workflows", workflow)
                requirements = [
                    token
                    for line in re.findall(r"pip install ([^\n]*)", content)
                    for token in line.split()
                    if re.match(r"law\b", token)
                ]
                self.assertEqual(requirements, [f"law=={lc.LAW_VERSION}"])


class AnotherReleaseIsRefusedAtImport(unittest.TestCase):
    """A loud refusal that says how to fix it, instead of code that submits and polls wrongly."""

    def import_with(self, version):
        code = textwrap.dedent(f"""
            import importlib.util, sys, types
            sys.path.insert(0, {flaf_parent!r})
            if importlib.util.find_spec("ROOT") is None:
                sys.modules["ROOT"] = types.ModuleType("ROOT")
            import law
            law.__version__ = {version!r}
            from FLAF.run_tools import law_customizations
            """)
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=600,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )

    def test_law_0_1_20_is_refused(self):
        proc = self.import_with("0.1.20")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(f"FLAF requires law {lc.LAW_VERSION}", proc.stderr)
        self.assertIn("law 0.1.20 is installed", proc.stderr)
        self.assertIn(f"pip install law=={lc.LAW_VERSION}", proc.stderr)

    def test_the_pinned_release_imports(self):
        proc = self.import_with(lc.LAW_VERSION)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])


#: a fake `pip` that records what it is asked to install
_FAKE_PIP = """#!/bin/bash
echo "$*" >> "$PIP_LOG"
"""


class MkFlafEnvInstallsThePin(unittest.TestCase):
    """run_tools/mk_flaf_env.sh, executed with pip replaced -- and, for a whole build, the two
    steps that need cvmfs or the network."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_mk_env_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.env_base = os.path.join(self.tmp, "flaf_env")
        os.makedirs(os.path.join(self.env_base, "bin"))
        with open(os.path.join(self.env_base, "bin", "pip"), "w") as f:
            f.write(_FAKE_PIP)
        os.chmod(os.path.join(self.env_base, "bin", "pip"), 0o755)
        with open(os.path.join(self.env_base, "bin", "activate"), "w") as f:
            f.write(f'export PATH="{self.env_base}/bin:$PATH"\n')
        self.pip_log = os.path.join(self.tmp, "pip.log")
        self.step_log = os.path.join(self.tmp, "steps.log")

    def run_script(self, script, *args):
        proc = subprocess.run(
            ["bash", script, *args],
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(os.environ, PIP_LOG=self.pip_log, STEP_LOG=self.step_log),
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(self.pip_log) as f:
            return f.read().splitlines()

    def assert_installs_law(self, installs, version):
        self.assertIn(f"install luigi==3.8.1 law=={version} scinum", installs)
        self.assertFalse(
            [i for i in installs if re.search(rf"\blaw\b(?!=={re.escape(version)})", i)]
        )

    def test_law_is_installed_at_the_version_it_is_given(self):
        installs = self.run_script(
            os.path.join(flaf_repo, "run_tools", "mk_flaf_env.sh"),
            "install",
            self.env_base,
            "9.9.9",
        )
        self.assert_installs_law(installs, "9.9.9")

    def test_a_whole_build_installs_the_law_it_is_given(self):
        """The default action, as env.sh runs it, passes its 4th argument on to `install`.

        The script runs itself for each step, so `create` (the LCG view from cvmfs) and
        `install_gh_cli` (a download) are stubbed in a copy, ahead of its dispatch; every other
        line is the script as written.
        """
        script = _read("run_tools", "mk_flaf_env.sh")
        dispatch = 'if [[ "$1" == "create" ]]; then'
        self.assertEqual(script.count(dispatch), 1)
        stubs = (
            'create() { echo "create $*" >> "$STEP_LOG"; }\n'
            'install_gh_cli() { echo "install_gh_cli $*" >> "$STEP_LOG"; }\n'
        )
        copy = os.path.join(self.tmp, "mk_flaf_env.sh")
        with open(copy, "w") as f:
            f.write(script.replace(dispatch, stubs + dispatch))
        os.chmod(copy, 0o755)
        installs = self.run_script(copy, self.env_base, "LCG_X", "ARCH_Y", "9.9.9")
        self.assert_installs_law(installs, "9.9.9")
        with open(self.step_log) as f:
            self.assertEqual(
                f.read().splitlines(),
                [
                    f"create {self.env_base} LCG_X ARCH_Y",
                    f"install_gh_cli {self.env_base}",
                ],
            )
        self.assertTrue(os.path.exists(os.path.join(self.env_base, ".LCG_X_ARCH_Y")))


def _law_guard_of_env_sh():
    """The part of env.sh's load_flaf_env that builds or checks flaf_env, as written there."""
    content = _read("env.sh")
    start = content.index('  local FLAF_LCG_VERSION="')
    end = content.index("  local os_version=", start)
    return content[start:end]


#: env.sh is sourced from bash and from zsh
_SHELLS = ["bash"] + (["zsh"] if shutil.which("zsh") else [])


class EnvShBringsTheEnvironmentToThePin(unittest.TestCase):
    """env.sh's own lines, run with `run_cmd` recording instead of running."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_env_guard_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.env_path = os.path.join(self.tmp, "flaf_env")
        fake_bin = os.path.join(self.tmp, "bin")
        os.makedirs(os.path.join(self.env_path, "bin"))
        os.makedirs(fake_bin)
        with open(os.path.join(self.env_path, "bin", "activate"), "w") as f:
            f.write(f'export PATH="{fake_bin}:$PATH"\n')
        # the environment's python3: the real interpreter, or one without law
        self.python3 = os.path.join(fake_bin, "python3")
        self.set_python(f'exec {sys.executable} "$@"')
        self.log = os.path.join(self.tmp, "run_cmd.log")

    def set_python(self, body):
        with open(self.python3, "w") as f:
            f.write(f"#!/bin/bash\n{body}\n")
        os.chmod(self.python3, 0o755)

    def without_law(self):
        """A fresh venv's interpreter: a site-packages that holds no law. Returns that venv."""
        venv = os.path.join(self.tmp, "venv")
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", venv], check=True
        )
        self.set_python(f'exec {os.path.join(venv, "bin", "python3")} "$@"')
        return venv

    def law_of_version(self, version):
        """Make importlib.metadata find law `version` ahead of the installed one."""
        site = os.path.join(self.tmp, "site")
        dist = os.path.join(site, f"law-{version}.dist-info")
        os.makedirs(dist)
        with open(os.path.join(dist, "METADATA"), "w") as f:
            f.write(f"Metadata-Version: 2.1\nName: law\nVersion: {version}\n")
        return site

    def run_guard(
        self,
        marker=True,
        no_install=False,
        pythonpath=None,
        law_job=False,
        shell="bash",
    ):
        if marker:
            open(
                os.path.join(self.env_path, ".LCG_110a_x86_64-el9-gcc15-opt"), "w"
            ).close()
        if os.path.exists(self.log):
            os.remove(self.log)
        script = (
            f'run_cmd() {{ echo "$*" >> "{self.log}"; }}\n'
            f"guard() {{\n{_law_guard_of_env_sh()}\n}}\n"
            "guard\n"
            'echo "guard returned $?"\n'
        )
        env = dict(
            os.environ,
            FLAF_ENVIRONMENT_PATH=self.env_path,
            FLAF_PATH="/flaf",
            FLAF_NO_INSTALL="1" if no_install else "0",
        )
        env.pop("PYTHONPATH", None)
        env.pop("LAW_JOB_HOME", None)
        if pythonpath:
            env["PYTHONPATH"] = pythonpath
        if law_job:
            env["LAW_JOB_HOME"] = os.path.join(self.tmp, "job_home")
        proc = subprocess.run(
            [shell, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=self.tmp,
        )
        calls = []
        if os.path.exists(self.log):
            with open(self.log) as f:
                calls = f.read().splitlines()
        return proc, calls

    def assert_refused(self, proc, calls, found):
        self.assertNotIn("guard returned", proc.stdout)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(f"has law {found}, FLAF requires {lc.LAW_VERSION}", proc.stdout)
        self.assertIn("Source env.sh on the submitting machine", proc.stdout)
        self.assertEqual(calls, [])

    def test_an_environment_at_the_pin_is_left_alone(self):
        for shell in _SHELLS:
            with self.subTest(shell):
                proc, calls = self.run_guard(shell=shell)
                self.assertIn("guard returned 0", proc.stdout)
                self.assertEqual(calls, [])

    def test_an_environment_with_another_law_is_brought_to_the_pin(self):
        site = self.law_of_version("0.1.20")
        for shell in _SHELLS:
            with self.subTest(shell):
                proc, calls = self.run_guard(pythonpath=site, shell=shell)
                self.assertIn("guard returned 0", proc.stdout)
                self.assertIn(f"from 0.1.20 to {lc.LAW_VERSION}", proc.stdout)
                self.assertEqual(
                    calls, [f"python3 -m pip install law=={lc.LAW_VERSION}"]
                )

    def test_an_environment_without_law_gets_it(self):
        self.without_law()
        proc, calls = self.run_guard()
        self.assertIn(f"from <none> to {lc.LAW_VERSION}", proc.stdout)
        self.assertEqual(calls, [f"python3 -m pip install law=={lc.LAW_VERSION}"])

    def test_with_installs_forbidden_another_law_stops_the_shell(self):
        proc, calls = self.run_guard(
            no_install=True, pythonpath=self.law_of_version("0.1.20")
        )
        self.assert_refused(proc, calls, "0.1.20")
        self.assertIn("FLAF_NO_INSTALL=1", proc.stdout)

    def test_a_batch_job_never_installs_into_the_shared_environment(self):
        """A job of a driver started before the pin moved sources the shared checkout's env.sh:
        installing from there would have every starting job write into the environment the
        running ones use."""
        site = self.law_of_version("0.1.20")
        for shell in _SHELLS:
            with self.subTest(shell):
                proc, calls = self.run_guard(pythonpath=site, law_job=True, shell=shell)
                self.assert_refused(proc, calls, "0.1.20")

    def test_a_batch_job_without_law_is_refused_as_well(self):
        self.without_law()
        proc, calls = self.run_guard(law_job=True)
        self.assert_refused(proc, calls, "<none>")

    def test_a_batch_job_at_the_pin_goes_on(self):
        proc, calls = self.run_guard(law_job=True)
        self.assertIn("guard returned 0", proc.stdout)
        self.assertEqual(calls, [])

    def test_a_check_that_fails_installs_nothing(self):
        """A check that fails (the environment's storage not answering, say) is not "no law"."""
        self.set_python('echo "OSError: storage did not answer" >&2; exit 1')
        for shell in _SHELLS:
            with self.subTest(shell):
                proc, calls = self.run_guard(shell=shell)
                self.assertNotIn("guard returned", proc.stdout)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("cannot tell which law", proc.stdout)
                self.assertIn("storage did not answer", proc.stderr)
                self.assertEqual(calls, [])

    @unittest.skipIf(os.geteuid() == 0, "root lists any directory")
    def test_site_packages_that_cannot_be_listed_installs_nothing(self):
        """importlib.metadata alone reads a site-packages it cannot list as one without law."""
        venv = self.without_law()
        purelib = subprocess.run(
            [
                os.path.join(venv, "bin", "python3"),
                "-c",
                'import sysconfig; print(sysconfig.get_paths()["purelib"])',
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        os.chmod(purelib, 0)
        self.addCleanup(os.chmod, purelib, 0o755)
        for shell in _SHELLS:
            with self.subTest(shell):
                proc, calls = self.run_guard(shell=shell)
                self.assertNotIn("guard returned", proc.stdout)
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("cannot tell which law", proc.stdout)
                self.assertIn("PermissionError", proc.stderr)
                self.assertEqual(calls, [])

    def test_a_new_environment_is_built_for_the_pin(self):
        proc, calls = self.run_guard(marker=False)
        self.assertIn("guard returned 0", proc.stdout)
        self.assertEqual(
            calls[-1],
            f"/flaf/run_tools/mk_flaf_env.sh {self.env_path} LCG_110a "
            f"x86_64-el9-gcc15-opt {lc.LAW_VERSION}",
        )


class ANonBundleJobRefusesAnotherLaw(unittest.TestCase):
    """bootstrap.sh's non-bundle branch, rendered and sourced as law's job script does it, with
    the real env.sh: such a job sets no FLAF_NO_INSTALL, so only LAW_JOB_HOME tells env.sh that it
    runs on a batch node."""

    def test_the_job_stops_without_running_pip(self):
        tmp = tempfile.mkdtemp(prefix="flaf_bootstrap_law_")
        self.addCleanup(shutil.rmtree, tmp, True)
        ana = os.path.join(tmp, "ana")
        env_path = os.path.join(ana, "soft", "flaf_env")
        fake_bin = os.path.join(tmp, "bin")
        job_home = os.path.join(tmp, "job")
        site = os.path.join(tmp, "site")
        for d in (os.path.join(env_path, "bin"), fake_bin, job_home):
            os.makedirs(d)
        os.makedirs(os.path.join(site, "law-0.1.20.dist-info"))
        with open(os.path.join(site, "law-0.1.20.dist-info", "METADATA"), "w") as f:
            f.write("Metadata-Version: 2.1\nName: law\nVersion: 0.1.20\n")
        open(os.path.join(env_path, ".LCG_110a_x86_64-el9-gcc15-opt"), "w").close()
        with open(os.path.join(env_path, "bin", "activate"), "w") as f:
            f.write(f'export PATH="{fake_bin}:$PATH"\n')
        # pip is recorded and fails, so that an install, if attempted, also ends the job there
        pip_log = os.path.join(tmp, "pip.log")
        with open(os.path.join(fake_bin, "python3"), "w") as f:
            f.write(
                "#!/bin/bash\n"
                'if [[ "$1" == "-m" && "$2" == "pip" ]]; then\n'
                f'  echo "$*" >> "{pip_log}"\n'
                "  exit 1\n"
                "fi\n"
                f'exec {sys.executable} "$@"\n'
            )
        os.chmod(os.path.join(fake_bin, "python3"), 0o755)
        # an analysis env.sh, reduced to what it hands FLAF/env.sh
        with open(os.path.join(ana, "env.sh"), "w") as f:
            f.write(
                f'export ANALYSIS_PATH="{ana}"\n'
                f'source "$FLAF_PATH/env.sh" "{ana}/env.sh"\n'
            )
        values = {
            "run_token_server_host": "",
            "run_token_server_port": "",
            "analysis_path": ana,
            "bundle_list": "",
            "rucio_account": "flaf_test",
            "flaf_path": flaf_repo,
            "corrections_path": "",
        }
        bootstrap = os.path.join(tmp, "bootstrap.sh")
        with open(bootstrap, "w") as f:
            f.write(
                re.sub(
                    r"\{\{(\w+)\}\}",
                    lambda m: values[m.group(1)],
                    _read("bootstrap.sh"),
                )
            )
        env = {
            "PATH": os.environ["PATH"],
            "HOME": tmp,
            "LAW_JOB_HOME": job_home,
            "LAW_JOB_INIT_DIR": job_home,
            "PYTHONPATH": site,
        }
        proc = subprocess.run(
            ["bash", "-c", f'source "{bootstrap}" ""\necho "bootstrap returned $?"\n'],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=job_home,
        )
        self.assertNotIn("bootstrap returned", proc.stdout)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn(
            f"has law 0.1.20, FLAF requires {lc.LAW_VERSION}", proc.stdout, proc.stderr
        )
        self.assertFalse(os.path.exists(pip_log))


class FakeBundleTask(lc.BundleTask):
    """BundleTask without the law machinery: only the naming is exercised."""

    @property
    def global_params(self):
        return {
            "bundles": {
                "core": {"hashed": True, "patterns": ["config"]},
                "soft": {"patterns": ["soft/flaf_env"]},
                "all_soft": {"patterns": ["soft"]},
                "cmssw": {"patterns": ["soft/CMSSW_16_0_6"]},
                "flaf_tools": {"patterns": ["soft/flaf"]},
            }
        }

    def remote_target(self, *parts):
        return "/".join(parts)


def bundle_name(flavour):
    # luigi's metaclass owns __call__, so the instance is built without it
    task = object.__new__(FakeBundleTask)
    task.flavour = flavour
    task.version = "v1"
    task.period = "Era"
    return task.output().rsplit("/", 1)[-1]


class TheEnvironmentBundleIsNamedAfterItsLaw(unittest.TestCase):
    """An unhashed bundle is complete once it exists: built before a law upgrade, the soft
    bundle would ship the old law to jobs whose script comes from the new one."""

    def setUp(self):
        ana = tempfile.mkdtemp(prefix="flaf_bundle_law_")
        self.addCleanup(shutil.rmtree, ana, True)
        os.makedirs(os.path.join(ana, "config"))
        env = mock.patch.dict(
            os.environ,
            {
                "ANALYSIS_PATH": ana,
                "FLAF_ENVIRONMENT_PATH": os.path.join(ana, "soft", "flaf_env"),
            },
        )
        env.start()
        self.addCleanup(env.stop)
        lc.BundleTask._source_hash_cache.clear()
        self.addCleanup(lc.BundleTask._source_hash_cache.clear)

    def test_the_flavour_holding_flaf_env_carries_the_law_version(self):
        self.assertEqual(bundle_name("soft"), f"soft_law{law.__version__}.tar.bz2")
        self.assertEqual(
            bundle_name("all_soft"), f"all_soft_law{law.__version__}.tar.bz2"
        )

    def test_another_law_gives_another_name(self):
        with mock.patch.object(law, "__version__", "0.1.22"):
            self.assertEqual(bundle_name("soft"), "soft_law0.1.22.tar.bz2")

    def test_flavours_without_it_keep_their_names(self):
        self.assertEqual(bundle_name("cmssw"), "cmssw.tar.bz2")
        self.assertRegex(bundle_name("core"), r"^core_[0-9a-f]{12}\.tar\.bz2$")

    def test_a_sibling_whose_name_is_a_prefix_does_not_hold_it(self):
        """`soft/flaf` is a string prefix of `soft/flaf_env`, not a directory above it."""
        self.assertEqual(bundle_name("flaf_tools"), "flaf_tools.tar.bz2")

    def test_an_environment_kept_elsewhere_is_not_in_the_soft_bundle(self):
        with mock.patch.dict(
            os.environ, {"FLAF_ENVIRONMENT_PATH": "/elsewhere/flaf_env"}
        ):
            self.assertEqual(bundle_name("soft"), "soft.tar.bz2")


if __name__ == "__main__":
    unittest.main()
