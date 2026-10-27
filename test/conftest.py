"""pytest configuration of the unit tests, shared with the analyses.

FLAF loads it as the conftest of test/; an analysis loads the copy of its pinned FLAF with
`python -m pytest -p FLAF.test.conftest test`. The Unit Tests workflow does both.

A test that needs what a plain Python environment lacks declares it with one of the markers of
REQUIREMENTS, and `-m "not needs_root and not needs_cvmfs and not needs_proxy"` deselects it
where that is missing. A module that cannot even be imported there declares its marker
module-wide (`pytestmark = pytest.mark.needs_root`): the marker is read from the source, and the
failed import becomes one test that carries it, so `-m` deselects the module while an unmarked
module that fails to import still stops the run. A module that leaves a stand-in for ROOT in
sys.modules fails too: the modules share one process, and the next one would import the stand-in.
The end of the run lists the deselected tests and every test file that ran none.
"""

import ast
import fnmatch
import os
import pathlib
import sys

import pytest

REQUIREMENTS = {
    "needs_root": "needs ROOT or another package of the analysis environment (numpy, "
    "uproot, awkward, Corrections); run it after `source env.sh`",
    "needs_cvmfs": "needs /cvmfs, e.g. the CRAB client or SITECONF",
    "needs_proxy": "needs a valid grid proxy",
}

# Scripts named test_*.py that other workflows and the docs run by path; not unit tests. pytest
# reads collect_ignore from a conftest only, so it does not apply where an analysis loads this
# file with -p.
collect_ignore = [
    "GenProcess",
    "test_era_dimension.py",
    "test_hello_world.py",
    "test_setup_loading.py",
]

_DESELECTED = pytest.StashKey[dict]()
_SELECTED = pytest.StashKey[set]()


def pytest_configure(config):
    for name, description in REQUIREMENTS.items():
        config.addinivalue_line("markers", f"{name}: {description}")
    config.stash[_DESELECTED] = {}
    config.stash[_SELECTED] = set()


def declared_requirements(path):
    """The REQUIREMENTS markers of the module-level `pytestmark` of `path`, without importing it."""
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except SyntaxError:
        return []  # the import reports it
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in node.targets
        ):
            names.update(
                sub.attr
                for sub in ast.walk(node.value)
                if isinstance(sub, ast.Attribute) and sub.attr in REQUIREMENTS
            )
    return sorted(names)


class UnimportableModule(pytest.Item):
    """A module that declares a requirement and cannot be imported here, as one test."""

    def __init__(self, *, error, **kwargs):
        super().__init__(**kwargs)
        self.error = error
        # pytest wraps an ImportError in a CollectError; anything else raised on import is bare
        cause = error.__cause__ or error
        self.reason = f"{type(cause).__name__}: {cause}"

    def runtest(self):
        raise self.error

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, pytest.Collector.CollectError):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, None, self.name


class ModuleWithRequirements(pytest.Module):
    def collect(self):
        try:
            return super().collect()
        except Exception as error:
            item = UnimportableModule.from_parent(self, name="import", error=error)
            for name in declared_requirements(self.path):
                item.add_marker(name)
            return [item]


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    if declared_requirements(module_path):
        return ModuleWithRequirements.from_parent(parent, path=module_path)
    return None


def _is_stand_in(module):
    return module is not None and getattr(module, "__file__", None) is None


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    before = sys.modules.get("ROOT")
    report = yield
    after = sys.modules.get("ROOT")
    if (
        isinstance(collector, pytest.Module)
        and after is not before
        and _is_stand_in(after)
    ):
        report.outcome = "failed"
        report.longrepr = (
            f"{collector.nodeid} leaves a stand-in for ROOT in sys.modules; withdraw it "
            "after the import it serves, as test_crab_credentials.py does"
        )
    return report


def _file_of(item):
    return item.nodeid.split("::")[0]


def pytest_deselected(items):
    for item in items:
        markers = {mark.name for mark in item.iter_markers()} & REQUIREMENTS.keys()
        entry = item.config.stash[_DESELECTED].setdefault(
            _file_of(item),
            {"marked": 0, "markers": set(), "unmarked": 0, "import": None},
        )
        entry["marked" if markers else "unmarked"] += 1
        entry["markers"] |= markers
        if isinstance(item, UnimportableModule):
            entry["import"] = item.reason


def pytest_collection_finish(session):
    session.config.stash[_SELECTED].update(_file_of(item) for item in session.items)


def _test_files(config):
    """Every file under the collected paths that matches python_files, as in node IDs."""
    patterns = config.getini("python_files")
    files = set()
    for arg in config.args:
        base = config.invocation_params.dir / arg.split("::")[0]
        for path in [base] if base.is_file() else base.rglob("*.py"):
            if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
                files.add(
                    pathlib.Path(os.path.relpath(path, config.rootpath)).as_posix()
                )
    return files


def pytest_terminal_summary(terminalreporter, config):
    deselected = config.stash[_DESELECTED]
    if deselected:
        options = " ".join(
            f"-{flag} {value!r}"
            for flag, value in (
                ("m", config.option.markexpr),
                ("k", config.option.keyword),
            )
            if value
        )
        terminalreporter.section(f"deselected by {options}")
        for path, entry in sorted(deselected.items()):
            if not entry["marked"]:
                continue
            if entry["import"]:
                what = f"not importable here ({entry['import']})"
            else:
                what = f"{entry['marked']} test(s)"
            markers = ", ".join(sorted(entry["markers"]))
            terminalreporter.write_line(f"{path}: {what}; {markers}")
        unmarked = [
            entry["unmarked"] for entry in deselected.values() if entry["unmarked"]
        ]
        if unmarked:
            terminalreporter.write_line(
                f"{sum(unmarked)} test(s) in {len(unmarked)} file(s) without a "
                "requirement marker"
            )
    idle = sorted(_test_files(config) - config.stash[_SELECTED] - set(deselected))
    if idle:
        terminalreporter.section("test files that ran no test")
        for path in idle:
            terminalreporter.write_line(path)
