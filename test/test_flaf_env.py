#!/usr/bin/env python3
"""flaf_env is defined by its installation script, and what identifies it follows that script.

`run_tools/mk_flaf_env.sh` pins every package it installs, and `env.sh` knows none of them: it
marks a built environment with the LCG release and a hash of that script, rebuilds one whose
marker differs on the submitting machine only, and exports that identity, after which the bundle
that packs flaf_env is named. The CI workflows install the versions the script pins.
"""

import glob
import hashlib
import importlib.util
import os
import re
import shutil
import signal
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


#: env.sh is sourced from bash and from zsh. A machine without zsh runs the bash cases only, but
#: not under CI (which sets `CI`), where a missing zsh fails the zsh cases instead.
_SHELLS = ["bash", "zsh"] if shutil.which("zsh") or os.environ.get("CI") else ["bash"]

_LCG = "LCG_110a"
_ARCH = "x86_64-el9-gcc15-opt"


def _ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


#: a requirement pinned to one version
_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[0-9][^\s=]*)$"
)
#: a package installed from the archive of one commit
_COMMIT_ARCHIVE_RE = re.compile(
    r"^https://github\.com/[\w.-]+/[\w.-]+/archive/[0-9a-f]{40}\.zip$"
)

#: a fake `pip` that records what it is asked to install, and fails on `$PIP_FAIL`
_FAKE_PIP = """#!/bin/bash
echo "$*" >> "$PIP_LOG"
if [[ -n "$PIP_FAIL" && "$*" == *"$PIP_FAIL"* ]]; then
    exit 1
fi
"""


class _MkFlafEnvCase(unittest.TestCase):
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

    def run_script(self, script, *args, pip_fail="", sigint_ignored=False):
        proc = subprocess.run(
            ["bash", script, *args],
            capture_output=True,
            text=True,
            timeout=120,
            env=dict(
                os.environ,
                PIP_LOG=self.pip_log,
                STEP_LOG=self.step_log,
                PIP_FAIL=pip_fail,
            ),
            preexec_fn=_ignore_sigint if sigint_ignored else None,
        )
        return proc, self.lines(self.pip_log), self.lines(self.step_log)

    @staticmethod
    def lines(path):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return f.read().splitlines()

    def installs(self):
        proc, installs, _ = self.run_script(
            os.path.join(flaf_repo, "run_tools", "mk_flaf_env.sh"),
            "install",
            self.env_base,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(installs)
        return installs

    def whole_build_script(self):
        """The script as written, with `create` (the LCG view from cvmfs) and `install_gh_cli`
        (a download) stubbed ahead of its dispatch. It runs itself for each step, so the stubs
        are what those steps run."""
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
        return copy


def _pins():
    """{canonical name: version} of what mk_flaf_env.sh installs, read from the script."""
    pins = {}
    for line in re.findall(
        r"^\s*run_cmd pip install (.*)$", _read("run_tools", "mk_flaf_env.sh"), re.M
    ):
        for token in line.split("#")[0].split():
            match = _PIN_RE.match(token)
            if match:
                pins[_canonical(match.group("name"))] = match.group("version")
    return pins


class EveryPackageIsPinned(_MkFlafEnvCase):
    def test_every_requirement_it_installs_is_pinned(self):
        for line in self.installs():
            with self.subTest(line):
                command, *requirements = line.split()
                self.assertEqual(command, "install")
                self.assertTrue(requirements)
                for requirement in requirements:
                    self.assertTrue(
                        _PIN_RE.match(requirement)
                        or _COMMIT_ARCHIVE_RE.match(requirement),
                        f"{requirement!r} is not pinned",
                    )

    def test_pip_itself_is_pinned(self):
        """An upgrade to whatever pip is newest would make two builds of one script differ."""
        self.assertIn("pip", _pins())

    def test_no_install_in_the_script_escapes_the_run(self):
        """Each `pip install` line of the script is one that the run above executed."""
        written = [
            line
            for line in _read("run_tools", "mk_flaf_env.sh").splitlines()
            if re.search(r"\bpip\S*\s+install\b", line)
            and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(written), len(self.installs()))


class AWholeBuild(_MkFlafEnvCase):
    """The default action, as env.sh runs it. env.sh writes the marker, on this script's exit
    status, so a failed step must end the script with a non-zero status -- also where SIGINT is
    ignored (any `cmd &` with job control off), which a SIGINT the script sends itself is not.
    """

    def test_it_builds_and_writes_no_marker(self):
        proc, installs, steps = self.run_script(
            self.whole_build_script(), self.env_base, "LCG_X", "ARCH_Y"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(
            steps,
            [
                f"create {self.env_base} LCG_X ARCH_Y",
                f"install_gh_cli {self.env_base}",
            ],
        )
        os.remove(self.pip_log)
        self.assertEqual(installs, self.installs())
        self.assertEqual(
            [f for f in os.listdir(self.env_base) if f.startswith(".")], []
        )

    def test_a_failed_install_fails_the_build(self):
        for sigint_ignored in (False, True):
            with self.subTest(sigint_ignored=sigint_ignored):
                for log in (self.pip_log, self.step_log):
                    if os.path.exists(log):
                        os.remove(log)
                proc, installs, steps = self.run_script(
                    self.whole_build_script(),
                    self.env_base,
                    "LCG_X",
                    "ARCH_Y",
                    pip_fail="fastcrc",
                    sigint_ignored=sigint_ignored,
                )
                self.assertNotEqual(proc.returncode, 0, proc.stdout)
                self.assertIn("Error while running", proc.stdout)
                self.assertEqual(steps, [f"create {self.env_base} LCG_X ARCH_Y"])
                self.assertTrue(installs[-1].startswith("install fastcrc=="))

    def test_a_failed_step_stops_the_install(self):
        proc, installs, _ = self.run_script(
            os.path.join(flaf_repo, "run_tools", "mk_flaf_env.sh"),
            "install",
            self.env_base,
            pip_fail="law==",
            sigint_ignored=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(len(installs), 2)
        self.assertIn("law==", installs[-1])


def _identity_block_of_env_sh():
    """The part of env.sh's load_flaf_env that identifies, builds and activates flaf_env."""
    content = _read("env.sh")
    start = content.index(f'  local FLAF_LCG_VERSION="{_LCG}"')
    end = content.index("  local os_version=", start)
    return content[start:end]


#: a stand-in for mk_flaf_env.sh: records its arguments and makes an activatable environment
_FAKE_BUILD = """#!/bin/bash
echo "$*" >> "$BUILD_LOG"
if [[ -n "$BUILD_FAIL" ]]; then
    exit 1
fi
mkdir -p "$1/bin" && echo 'export FLAF_ENV_ACTIVE=1' > "$1/bin/activate" || exit 1
if [[ -n "$BUILD_READ_ONLY" ]]; then
    chmod 555 "$1"
fi
"""


class _EnvShCase(unittest.TestCase):
    """env.sh's own lines, run in bash and zsh against a stand-in installation script."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="flaf_env_id_")
        self.addCleanup(self.cleanup)
        self.flaf = self.make_flaf("flaf", _FAKE_BUILD)
        self.env_path = os.path.join(self.tmp, "soft", "flaf_env")
        self.build_log = os.path.join(self.tmp, "build.log")

    def cleanup(self):
        def unlock(path):
            os.chmod(path, 0o755)
            for entry in os.scandir(path):
                if entry.is_dir(follow_symlinks=False):
                    unlock(entry.path)

        unlock(self.tmp)
        shutil.rmtree(self.tmp)

    def make_flaf(self, name, script):
        run_tools = os.path.join(self.tmp, name, "run_tools")
        os.makedirs(run_tools)
        with open(os.path.join(run_tools, "mk_flaf_env.sh"), "w") as f:
            f.write(script)
        os.chmod(os.path.join(run_tools, "mk_flaf_env.sh"), 0o755)
        return os.path.join(self.tmp, name)

    def identity(self, flaf=None):
        with open(
            os.path.join(flaf or self.flaf, "run_tools", "mk_flaf_env.sh"), "rb"
        ) as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        return f"{_LCG}_{_ARCH}_{digest[:12]}"

    def make_env(self, marker):
        """An existing environment holding `sentinel`, marked `.<marker>` (if given)."""
        os.makedirs(os.path.join(self.env_path, "bin"))
        with open(os.path.join(self.env_path, "bin", "activate"), "w") as f:
            f.write("export FLAF_ENV_ACTIVE=1\n")
        open(os.path.join(self.env_path, "sentinel"), "w").close()
        if marker:
            open(os.path.join(self.env_path, f".{marker}"), "w").close()

    def link_soft(self):
        """soft/ a link to a directory elsewhere, as in a clone whose soft/ is kept on other
        storage."""
        real_soft = os.path.join(self.tmp, "real_soft")
        os.makedirs(real_soft)
        os.symlink(real_soft, os.path.dirname(self.env_path))

    def omitting_find(self):
        """A directory holding a `find` that lists nothing and succeeds, as a stale directory
        cache answers."""
        fake_bin = os.path.join(self.tmp, "omitting_bin")
        os.makedirs(fake_bin, exist_ok=True)
        with open(os.path.join(fake_bin, "find"), "w") as f:
            f.write("#!/bin/bash\nexit 0\n")
        os.chmod(os.path.join(fake_bin, "find"), 0o755)
        return fake_bin

    def failing_find(self):
        """A directory holding a `find` that fails with an error and lists nothing, as storage
        that does not answer."""
        fake_bin = os.path.join(self.tmp, "failing_bin")
        os.makedirs(fake_bin, exist_ok=True)
        with open(os.path.join(fake_bin, "find"), "w") as f:
            f.write("#!/bin/bash\necho 'find: Input/output error' >&2\nexit 1\n")
        os.chmod(os.path.join(fake_bin, "find"), 0o755)
        return fake_bin

    def kept(self):
        return os.path.exists(os.path.join(self.env_path, "sentinel"))

    def markers(self):
        if not os.path.isdir(self.env_path):
            return []
        return sorted(f for f in os.listdir(self.env_path) if f.startswith("."))

    def run_block(
        self,
        shell="bash",
        no_install=False,
        law_job=False,
        sigint_ignored=False,
        flaf=None,
        path_prefix=None,
        **build_env,
    ):
        if os.path.exists(self.build_log):
            os.remove(self.build_log)
        # The identity is read by a child process, as BundleTask reads it in law's Python.
        script = (
            f"block() {{\n{_identity_block_of_env_sh()}\n}}\n"
            "block\n"
            'echo "block returned $?"\n'
            'echo "id=$(printenv FLAF_ENVIRONMENT_ID) active=$FLAF_ENV_ACTIVE"\n'
        )
        env = dict(
            os.environ,
            FLAF_ENVIRONMENT_PATH=self.env_path,
            FLAF_PATH=flaf or self.flaf,
            FLAF_NO_INSTALL="1" if no_install else "0",
            BUILD_LOG=self.build_log,
            **build_env,
        )
        for name in ("LAW_JOB_HOME", "FLAF_ENVIRONMENT_ID", "FLAF_ENV_ACTIVE"):
            env.pop(name, None)
        if law_job:
            env["LAW_JOB_HOME"] = os.path.join(self.tmp, "job_home")
        if path_prefix:
            env["PATH"] = f"{path_prefix}:{env['PATH']}"
        proc = subprocess.run(
            [shell, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=self.tmp,
            preexec_fn=_ignore_sigint if sigint_ignored else None,
        )
        builds = []
        if os.path.exists(self.build_log):
            with open(self.build_log) as f:
                builds = f.read().splitlines()
        return proc, builds

    def assert_went_on(self, proc, identity):
        self.assertIn("block returned 0", proc.stdout, proc.stdout + proc.stderr)
        self.assertIn(f"id={identity} active=1", proc.stdout)

    def assert_stopped(self, proc, sigint_ignored=False):
        """Stopped: the shell ends -- or, where SIGINT is ignored, the block returns non-zero
        before the environment is activated."""
        if sigint_ignored:
            self.assertRegex(proc.stdout, r"block returned [1-9]")
            self.assertNotIn("active=1", proc.stdout)
        else:
            self.assertNotIn("block returned", proc.stdout)
            self.assertNotEqual(proc.returncode, 0)


class AMatchingEnvironmentIsLeftAlone(_EnvShCase):
    def test_on_the_submitting_machine_and_in_a_batch_job(self):
        self.make_env(self.identity())
        for shell in _SHELLS:
            for law_job in (False, True):
                with self.subTest(shell=shell, law_job=law_job):
                    proc, builds = self.run_block(shell=shell, law_job=law_job)
                    self.assert_went_on(proc, self.identity())
                    self.assertEqual(builds, [])
                    self.assertTrue(self.kept())

    def test_with_a_backslash_in_the_path_of_the_script(self):
        """The identity is the script's content alone, however its path is spelled."""
        flaf = self.make_flaf("FL\\AF", _FAKE_BUILD)
        self.assertEqual(self.identity(flaf), self.identity())
        self.make_env(self.identity())
        for shell in _SHELLS:
            with self.subTest(shell):
                proc, builds = self.run_block(shell=shell, flaf=flaf)
                self.assert_went_on(proc, self.identity())
                self.assertEqual(builds, [])
                self.assertTrue(self.kept())


class AnotherEnvironmentIsRebuilt(_EnvShCase):
    """On the submitting machine an environment of another LCG release or another installation
    script -- the marker of the release alone that earlier versions wrote included -- is removed
    and built again, and marked once the build succeeded."""

    def check_rebuilt(self, shell):
        proc, builds = self.run_block(shell=shell)
        self.assert_went_on(proc, self.identity())
        self.assertEqual(builds, [f"{self.env_path} {_LCG} {_ARCH}"])
        self.assertFalse(self.kept())
        self.assertEqual(self.markers(), [f".{self.identity()}"])
        self.assertIn("Removing old FLAF environment", proc.stdout)

    def test_the_marker_of_the_release_alone(self):
        for shell in _SHELLS:
            with self.subTest(shell):
                shutil.rmtree(self.env_path, True)
                self.make_env(f"{_LCG}_{_ARCH}")
                self.check_rebuilt(shell)

    def test_a_changed_installation_script(self):
        """A pin moved in the script changes the identity, and with it the environment."""
        built_by = self.identity()
        with open(os.path.join(self.flaf, "run_tools", "mk_flaf_env.sh"), "a") as f:
            f.write("# pip install law==9.9.9\n")
        self.assertNotEqual(self.identity(), built_by)
        for shell in _SHELLS:
            with self.subTest(shell):
                shutil.rmtree(self.env_path, True)
                self.make_env(built_by)
                self.check_rebuilt(shell)

    def test_under_a_linked_parent(self):
        """soft/ is a link: the environment is listed through it, and the link stays."""
        self.link_soft()
        for shell in _SHELLS:
            with self.subTest(shell):
                shutil.rmtree(self.env_path, True)
                self.make_env(f"{_LCG}_{_ARCH}")
                self.check_rebuilt(shell)
                self.assertTrue(os.path.islink(os.path.dirname(self.env_path)))

    def test_an_entry_of_the_marker_name_further_down_is_no_marker(self):
        for shell in _SHELLS:
            with self.subTest(shell):
                shutil.rmtree(self.env_path, True)
                self.make_env(f"{_LCG}_{_ARCH}")
                os.makedirs(os.path.join(self.env_path, "lib"))
                open(
                    os.path.join(self.env_path, "lib", f".{self.identity()}"), "w"
                ).close()
                self.check_rebuilt(shell)

    def test_a_missing_environment_is_built(self):
        """Absent from its parent -- a parent that does not exist yet included, one that holds a
        sibling whose name begins with the environment's or an entry of its name one level
        down, and a listing that omits everything, where no stat sees the environment either.
        """
        soft = os.path.dirname(self.env_path)
        omitting_find = self.omitting_find()
        for shell in _SHELLS:
            for parent in ("missing", "with siblings", "omitted from a listing"):
                with self.subTest(shell=shell, parent=parent):
                    shutil.rmtree(soft, True)
                    if parent == "with siblings":
                        for sibling in ("CMSSW_16_0_6/flaf_env", "flaf_env_old"):
                            os.makedirs(os.path.join(soft, sibling))
                    proc, builds = self.run_block(
                        shell=shell,
                        path_prefix=(
                            omitting_find
                            if parent == "omitted from a listing"
                            else None
                        ),
                    )
                    self.assert_went_on(proc, self.identity())
                    self.assertEqual(builds, [f"{self.env_path} {_LCG} {_ARCH}"])
                    self.assertEqual(self.markers(), [f".{self.identity()}"])
                    self.assertNotIn("Removing old FLAF environment", proc.stdout)

    def test_a_missing_environment_under_a_parent_of_its_name_is_built(self):
        """soft/flaf_env/flaf_env: the parent's own name is no entry of the parent."""
        self.env_path = os.path.join(self.tmp, "soft", "flaf_env", "flaf_env")
        for shell in _SHELLS:
            with self.subTest(shell):
                shutil.rmtree(os.path.join(self.tmp, "soft"), True)
                os.makedirs(os.path.dirname(self.env_path))
                proc, builds = self.run_block(shell=shell)
                self.assert_went_on(proc, self.identity())
                self.assertEqual(builds, [f"{self.env_path} {_LCG} {_ARCH}"])
                self.assertEqual(self.markers(), [f".{self.identity()}"])

    def test_the_identity_follows_the_content_not_the_location(self):
        """A bundle job hashes the copy of the script in its bundle."""
        other = self.make_flaf("bundle/FLAF", _FAKE_BUILD)
        self.assertEqual(self.identity(other), self.identity())
        self.make_env(self.identity())
        proc, builds = self.run_block(flaf=other, no_install=True)
        self.assert_went_on(proc, self.identity())
        self.assertEqual(builds, [])


class ABatchJobNeverBuilds(_EnvShCase):
    """A batch job runs from an environment other jobs share: a mismatch stops it with the fix,
    and nothing is removed or built -- a non-bundle HTCondor job included, which sources the
    shared checkout's env.sh with LAW_JOB_HOME set and FLAF_NO_INSTALL unset."""

    def check_refused(self, proc, builds, had_env=True):
        self.assert_stopped(proc)
        self.assertIn("Source env.sh on the submitting machine", proc.stdout)
        self.assertEqual(builds, [])
        if had_env:
            self.assertTrue(self.kept())
            self.assertEqual(self.markers(), [f".{_LCG}_{_ARCH}"])
        else:
            self.assertFalse(os.path.exists(self.env_path))

    def test_a_law_job_with_another_environment(self):
        self.make_env(f"{_LCG}_{_ARCH}")
        for shell in _SHELLS:
            with self.subTest(shell):
                self.check_refused(*self.run_block(shell=shell, law_job=True))

    def test_installs_forbidden_with_another_environment(self):
        self.make_env(f"{_LCG}_{_ARCH}")
        for shell in _SHELLS:
            with self.subTest(shell):
                proc, builds = self.run_block(shell=shell, no_install=True)
                self.check_refused(proc, builds)
                self.assertIn("FLAF_NO_INSTALL=1", proc.stdout)

    def test_a_law_job_without_an_environment(self):
        self.check_refused(*self.run_block(law_job=True), had_env=False)


class ACheckThatFailsChangesNothing(_EnvShCase):
    """A check that cannot be made says nothing about the environment: it stops, and nothing is
    removed or built -- with SIGINT ignored as well, where the stop is the block returning.
    """

    def check_untouched(self, proc, builds, message, sigint_ignored=False):
        self.assert_stopped(proc, sigint_ignored)
        self.assertIn(message, proc.stdout)
        self.assertEqual(builds, [])
        self.assertTrue(self.kept())

    def test_a_missing_installation_script(self):
        os.remove(os.path.join(self.flaf, "run_tools", "mk_flaf_env.sh"))
        self.make_env(f"{_LCG}_{_ARCH}")
        for shell in _SHELLS:
            for sigint_ignored in (False, True):
                with self.subTest(shell=shell, sigint_ignored=sigint_ignored):
                    proc, builds = self.run_block(
                        shell=shell, sigint_ignored=sigint_ignored
                    )
                    self.check_untouched(
                        proc,
                        builds,
                        "cannot identify the FLAF environment",
                        sigint_ignored,
                    )

    @unittest.skipIf(os.geteuid() == 0, "root reads any file")
    def test_an_unreadable_installation_script(self):
        os.chmod(os.path.join(self.flaf, "run_tools", "mk_flaf_env.sh"), 0)
        self.make_env(f"{_LCG}_{_ARCH}")
        for shell in _SHELLS:
            with self.subTest(shell):
                proc, builds = self.run_block(shell=shell)
                self.check_untouched(
                    proc, builds, "cannot identify the FLAF environment"
                )
                self.assertRegex(proc.stderr, "(?i)permission denied")

    def test_a_hash_that_fails(self):
        """A hashing tool that fails after printing something is not trusted for what it printed."""
        fake_bin = os.path.join(self.tmp, "fake_bin")
        os.makedirs(fake_bin)
        with open(os.path.join(fake_bin, "sha256sum"), "w") as f:
            f.write(
                "#!/bin/bash\necho \"0123456789abcdef  $1\"\necho 'I/O error' >&2\nexit 1\n"
            )
        os.chmod(os.path.join(fake_bin, "sha256sum"), 0o755)
        self.make_env(f"{_LCG}_{_ARCH}")
        for shell in _SHELLS:
            for sigint_ignored in (False, True):
                with self.subTest(shell=shell, sigint_ignored=sigint_ignored):
                    proc, builds = self.run_block(
                        shell=shell, sigint_ignored=sigint_ignored, path_prefix=fake_bin
                    )
                    self.check_untouched(
                        proc,
                        builds,
                        "cannot identify the FLAF environment",
                        sigint_ignored,
                    )

    def check_refused(self, message, still_there, path_prefix=None):
        """In every shell, with SIGINT ignored as well: it stops with `message`, nothing is
        built, and `still_there()` holds."""
        for shell in _SHELLS:
            for sigint_ignored in (False, True):
                with self.subTest(shell=shell, sigint_ignored=sigint_ignored):
                    proc, builds = self.run_block(
                        shell=shell,
                        sigint_ignored=sigint_ignored,
                        path_prefix=path_prefix,
                    )
                    self.assert_stopped(proc, sigint_ignored)
                    self.assertIn(message, proc.stdout)
                    self.assertEqual(builds, [])
                    self.assertTrue(still_there())

    def dangling_marker(self, env):
        """The marker of the current identity in `env`, as a link whose target is gone: the
        listing has it, and its stat fails."""
        os.symlink(
            os.path.join(self.tmp, "gone"), os.path.join(env, f".{self.identity()}")
        )

    def check_marker_refused(self, still_there):
        self.check_refused(
            f"cannot tell whether {self.env_path} has its marker", still_there
        )

    def test_a_marker_that_is_listed_but_cannot_be_stated(self):
        self.make_env(None)
        self.dangling_marker(self.env_path)
        self.check_marker_refused(self.kept)

    def test_a_marker_that_cannot_be_stated_under_a_linked_parent(self):
        """soft/ is a link: the environment and its marker are listed through it."""
        self.link_soft()
        self.make_env(None)
        self.dangling_marker(self.env_path)
        self.check_marker_refused(self.kept)

    def test_a_marker_that_cannot_be_stated_in_a_linked_environment(self):
        """soft/flaf_env is a link to an environment elsewhere: its marker is listed through it,
        and the link stays."""
        self.make_env(None)
        elsewhere = os.path.join(self.tmp, "elsewhere")
        os.makedirs(elsewhere)
        os.rename(self.env_path, os.path.join(elsewhere, "flaf_env"))
        os.symlink(os.path.join(elsewhere, "flaf_env"), self.env_path)
        self.dangling_marker(self.env_path)
        self.check_marker_refused(lambda: os.path.islink(self.env_path) and self.kept())

    def test_an_environment_seen_but_not_listed(self):
        """The listing of its parent omits the environment (a stale directory cache) while a
        stat sees it."""
        self.make_env(f"{_LCG}_{_ARCH}")
        self.check_refused(
            f"cannot tell whether {self.env_path} exists: it is seen but not listed",
            self.kept,
            path_prefix=self.omitting_find(),
        )

    def test_a_link_seen_but_not_listed(self):
        """As above, for a link whose target is gone, which only an lstat sees."""
        os.makedirs(os.path.dirname(self.env_path))
        os.symlink(os.path.join(self.tmp, "gone"), self.env_path)
        self.check_refused(
            f"cannot tell whether {self.env_path} exists: it is seen but not listed",
            lambda: os.path.islink(self.env_path),
            path_prefix=self.omitting_find(),
        )

    def test_a_file_seen_but_not_listed(self):
        """As above, for a file, which a stat sees but not as a directory."""
        os.makedirs(os.path.dirname(self.env_path))
        with open(self.env_path, "w") as f:
            f.write("kept\n")

        def still_there():
            with open(self.env_path) as f:
                return f.read() == "kept\n"

        self.check_refused(
            f"cannot tell whether {self.env_path} exists: it is seen but not listed",
            still_there,
            path_prefix=self.omitting_find(),
        )

    @unittest.skipIf(os.geteuid() == 0, "root lists any directory")
    def test_an_environment_that_cannot_be_listed(self):
        self.make_env(f"{_LCG}_{_ARCH}")
        os.chmod(self.env_path, 0o300)
        for shell in _SHELLS:
            for sigint_ignored in (False, True):
                with self.subTest(shell=shell, sigint_ignored=sigint_ignored):
                    proc, builds = self.run_block(
                        shell=shell, sigint_ignored=sigint_ignored
                    )
                    os.chmod(self.env_path, 0o755)
                    self.check_untouched(
                        proc, builds, "cannot tell whether", sigint_ignored
                    )
                    os.chmod(self.env_path, 0o300)

    @unittest.skipIf(os.geteuid() == 0, "root reaches any directory")
    def test_an_environment_that_cannot_be_reached(self):
        """Its parent can be read but not searched, so neither a listing nor a stat reaches the
        environment."""
        self.make_env(f"{_LCG}_{_ARCH}")
        soft = os.path.dirname(self.env_path)
        os.chmod(soft, 0o600)
        for shell in _SHELLS:
            for sigint_ignored in (False, True):
                with self.subTest(shell=shell, sigint_ignored=sigint_ignored):
                    proc, builds = self.run_block(
                        shell=shell, sigint_ignored=sigint_ignored
                    )
                    os.chmod(soft, 0o755)
                    self.check_untouched(
                        proc,
                        builds,
                        f"cannot tell whether {self.env_path}",
                        sigint_ignored,
                    )
                    os.chmod(soft, 0o600)

    @unittest.skipIf(os.geteuid() == 0, "root lists any directory")
    def test_a_parent_that_cannot_be_listed(self):
        """The environment is reachable, and a stat finds it, but its parent cannot be listed to
        confirm that it is there."""
        self.make_env(f"{_LCG}_{_ARCH}")
        soft = os.path.dirname(self.env_path)
        os.chmod(soft, 0o300)
        for shell in _SHELLS:
            for sigint_ignored in (False, True):
                with self.subTest(shell=shell, sigint_ignored=sigint_ignored):
                    proc, builds = self.run_block(
                        shell=shell, sigint_ignored=sigint_ignored
                    )
                    os.chmod(soft, 0o755)
                    self.check_untouched(
                        proc,
                        builds,
                        f"cannot tell whether {self.env_path} exists ({soft} cannot be listed)",
                        sigint_ignored,
                    )
                    os.chmod(soft, 0o300)

    def test_a_parent_listing_that_fails_where_no_stat_sees_the_environment(self):
        """soft/ exists and holds no environment, but its listing fails: nothing is built."""
        soft = os.path.dirname(self.env_path)
        os.makedirs(soft)
        self.check_refused(
            f"cannot tell whether {self.env_path} exists ({soft} cannot be listed)",
            lambda: not os.path.lexists(self.env_path),
            path_prefix=self.failing_find(),
        )

    def check_entry_kept(self, still_there):
        """The environment's path is listed in its parent, but a stat sees no directory there."""
        self.check_refused(
            f"cannot tell whether {self.env_path} is an environment", still_there
        )

    @unittest.skipIf(os.geteuid() == 0, "root searches any directory")
    def test_a_listed_environment_whose_stat_fails(self):
        """A link to an environment in a directory that cannot be searched: the listing has it,
        and its stat fails, as on storage that blinks."""
        self.make_env(f"{_LCG}_{_ARCH}")
        locked = os.path.join(self.tmp, "locked")
        os.makedirs(locked)
        os.rename(self.env_path, os.path.join(locked, "flaf_env"))
        os.symlink(os.path.join(locked, "flaf_env"), self.env_path)
        os.chmod(locked, 0o600)

        def still_there():
            os.chmod(locked, 0o755)
            try:
                return os.path.islink(self.env_path) and self.kept()
            finally:
                os.chmod(locked, 0o600)

        self.check_entry_kept(still_there)

    def test_a_listed_link_whose_target_is_gone(self):
        os.makedirs(os.path.dirname(self.env_path))
        os.symlink(os.path.join(self.tmp, "gone"), self.env_path)
        self.check_entry_kept(lambda: os.path.islink(self.env_path))

    def test_a_listed_file(self):
        os.makedirs(os.path.dirname(self.env_path))
        with open(self.env_path, "w") as f:
            f.write("kept\n")

        def still_there():
            with open(self.env_path) as f:
                return f.read() == "kept\n"

        self.check_entry_kept(still_there)


class TheMarkerFollowsACompleteBuild(_EnvShCase):
    def test_a_failed_build_is_not_marked(self):
        for shell in _SHELLS:
            for sigint_ignored in (False, True):
                with self.subTest(shell=shell, sigint_ignored=sigint_ignored):
                    proc, builds = self.run_block(
                        shell=shell, sigint_ignored=sigint_ignored, BUILD_FAIL="1"
                    )
                    self.assert_stopped(proc, sigint_ignored)
                    self.assertIn("building the FLAF environment", proc.stdout)
                    self.assertEqual(builds, [f"{self.env_path} {_LCG} {_ARCH}"])
                    self.assertEqual(self.markers(), [])

    @unittest.skipIf(os.geteuid() == 0, "root writes anywhere")
    def test_a_marker_that_cannot_be_written_fails_the_build(self):
        for shell in _SHELLS:
            for sigint_ignored in (False, True):
                with self.subTest(shell=shell, sigint_ignored=sigint_ignored):
                    if os.path.exists(self.env_path):
                        os.chmod(self.env_path, 0o755)
                        shutil.rmtree(self.env_path)
                    proc, builds = self.run_block(
                        shell=shell, sigint_ignored=sigint_ignored, BUILD_READ_ONLY="1"
                    )
                    self.assert_stopped(proc, sigint_ignored)
                    self.assertIn("building the FLAF environment", proc.stdout)
                    self.assertEqual(self.markers(), [])


class EnvShKnowsNoPackage(unittest.TestCase):
    """Which packages flaf_env holds, and at which versions, is the installation script's
    business alone: env.sh neither names nor probes any of them."""

    def test_no_pin_probe_or_install(self):
        env_sh = _read("env.sh")
        pins = _pins()
        self.assertIn("law", pins)
        self.assertNotRegex(env_sh, r"\bpip\b")
        self.assertNotIn("importlib.metadata", env_sh)
        self.assertNotIn("__version__", env_sh)
        self.assertNotIn("FLAF_LAW_VERSION", env_sh)
        for name, version in pins.items():
            with self.subTest(name):
                spelled = r"[-_.]".join(map(re.escape, re.split(r"-", name)))
                self.assertNotRegex(env_sh, rf"(?i)\b{spelled}\s*(==|>=|<=|~=|!=)")
                self.assertNotRegex(env_sh, rf"(?<![\w.]){re.escape(version)}(?![\w.])")


class TheCiInstallsWhatTheScriptPins(unittest.TestCase):
    """A CI job that installs a package the installation script pins installs that version, so
    that it tests, formats and lints with what flaf_env holds."""

    def test_the_versions_agree(self):
        pins = _pins()
        checked = set()
        files = sorted(
            glob.glob(
                os.path.join(flaf_repo, ".github", "**", "*.y*ml"), recursive=True
            )
            + glob.glob(
                os.path.join(flaf_repo, ".github", "**", "*.sh"), recursive=True
            )
        )
        self.assertTrue(files)
        for path in files:
            with open(path) as f:
                content = f.read()
            for line in re.findall(r"\bpip3?\s+install\s+([^\n]*)", content):
                for token in line.split("#")[0].split():
                    if token.startswith("-"):
                        continue
                    name = _canonical(re.split(r"[\[=<>!~;@ ]", token)[0])
                    if name not in pins:
                        continue
                    with self.subTest(
                        file=os.path.relpath(path, flaf_repo), token=token
                    ):
                        match = _PIN_RE.match(token)
                        self.assertIsNotNone(match, f"{token!r} is not pinned")
                        self.assertEqual(match.group("version"), pins[name])
                    checked.add(name)
        self.assertLessEqual({"pip", "law", "luigi"}, checked)


class NoImportRefusesAnotherLaw(unittest.TestCase):
    """The installation script pins law; importing FLAF does not check it again."""

    def test_an_import_under_another_release_goes_through(self):
        code = textwrap.dedent(f"""
            import importlib.util, sys, types
            sys.path.insert(0, {flaf_parent!r})
            if importlib.util.find_spec("ROOT") is None:
                sys.modules["ROOT"] = types.ModuleType("ROOT")
            import law
            law.__version__ = "0.0.1"
            from FLAF.run_tools import law_customizations
            """)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=600,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])


class ANonBundleJobNeverRebuilds(unittest.TestCase):
    """bootstrap.sh's non-bundle branch, rendered and sourced as law's job script does it, with
    the real env.sh: such a job sets no FLAF_NO_INSTALL, so only LAW_JOB_HOME tells env.sh that it
    runs on a batch node, where an environment of another script must not be removed and built
    again under the jobs that run from it."""

    def test_the_job_stops_and_leaves_the_environment(self):
        tmp = tempfile.mkdtemp(prefix="flaf_bootstrap_env_")
        self.addCleanup(shutil.rmtree, tmp, True)
        ana = os.path.join(tmp, "ana")
        flaf = os.path.join(tmp, "FLAF")
        env_path = os.path.join(ana, "soft", "flaf_env")
        job_home = os.path.join(tmp, "job")
        for d in (
            os.path.join(env_path, "bin"),
            os.path.join(flaf, "run_tools"),
            job_home,
        ):
            os.makedirs(d)
        shutil.copy2(os.path.join(flaf_repo, "env.sh"), os.path.join(flaf, "env.sh"))
        build_log = os.path.join(tmp, "build.log")
        with open(os.path.join(flaf, "run_tools", "mk_flaf_env.sh"), "w") as f:
            f.write(f'#!/bin/bash\necho "$*" >> "{build_log}"\nexit 1\n')
        os.chmod(os.path.join(flaf, "run_tools", "mk_flaf_env.sh"), 0o755)
        open(os.path.join(env_path, f".{_LCG}_{_ARCH}"), "w").close()
        open(os.path.join(env_path, "sentinel"), "w").close()
        with open(os.path.join(env_path, "bin", "activate"), "w") as f:
            f.write("true\n")
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
            "flaf_path": flaf,
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
            "Source env.sh on the submitting machine", proc.stdout, proc.stderr
        )
        self.assertTrue(os.path.exists(os.path.join(env_path, "sentinel")))
        self.assertFalse(os.path.exists(build_log))


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


_ENV_ID = f"{_LCG}_{_ARCH}_0123456789ab"


class TheEnvironmentBundleIsNamedAfterTheEnvironment(unittest.TestCase):
    """An unhashed bundle is complete once it exists: built before the environment was rebuilt,
    the soft bundle would ship the old packages to jobs whose script comes from the new ones.
    """

    def setUp(self):
        ana = tempfile.mkdtemp(prefix="flaf_bundle_env_")
        self.addCleanup(shutil.rmtree, ana, True)
        os.makedirs(os.path.join(ana, "config"))
        env = mock.patch.dict(
            os.environ,
            {
                "ANALYSIS_PATH": ana,
                "FLAF_ENVIRONMENT_PATH": os.path.join(ana, "soft", "flaf_env"),
                "FLAF_ENVIRONMENT_ID": _ENV_ID,
            },
        )
        env.start()
        self.addCleanup(env.stop)
        lc.BundleTask._source_hash_cache.clear()
        self.addCleanup(lc.BundleTask._source_hash_cache.clear)

    def test_the_flavour_holding_flaf_env_carries_the_identity(self):
        self.assertEqual(bundle_name("soft"), f"soft_{_ENV_ID}.tar.bz2")
        self.assertEqual(bundle_name("all_soft"), f"all_soft_{_ENV_ID}.tar.bz2")

    def test_another_environment_gives_another_name(self):
        with mock.patch.dict(os.environ, {"FLAF_ENVIRONMENT_ID": "other"}):
            self.assertEqual(bundle_name("soft"), "soft_other.tar.bz2")

    def test_no_single_package_names_it(self):
        with mock.patch.object(law, "__version__", "0.0.1"):
            self.assertEqual(bundle_name("soft"), f"soft_{_ENV_ID}.tar.bz2")

    def test_without_the_identity_it_raises(self):
        for value in (None, ""):
            with self.subTest(value=value):
                with mock.patch.dict(os.environ):
                    if value is None:
                        os.environ.pop("FLAF_ENVIRONMENT_ID")
                    else:
                        os.environ["FLAF_ENVIRONMENT_ID"] = value
                    with self.assertRaisesRegex(RuntimeError, "FLAF_ENVIRONMENT_ID"):
                        bundle_name("soft")

    def test_flavours_without_it_keep_their_names(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("FLAF_ENVIRONMENT_ID")
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
