# Prerequisites

Before installing anything, make sure you have access to the CERN computing infrastructure that
FLAF relies on. If you have run any CMS analysis on `lxplus` before, you almost certainly have
all of this already — skip to [Installation](installation.md).

## 1. A CERN account and `lxplus`

FLAF is developed and run on CERN's interactive login service, **`lxplus`**, currently on
**AlmaLinux 9** (`el9`). Connect with:

```sh
ssh <your-cern-username>@lxplus.cern.ch
```

!!! note "Other machines"
    FLAF can run on any machine that provides CVMFS and an `el9` (or compatible) environment,
    but `lxplus` is the supported and tested platform. The instructions throughout assume it.

## 2. CVMFS

FLAF gets its compilers, Python and ROOT from the CERN software distribution service
[**CVMFS**](https://cvmfs.readthedocs.io/). On `lxplus` it is already mounted. Check that the two
areas FLAF uses are visible:

```sh
ls /cvmfs/cms.cern.ch        # CMS software (CMSSW)
ls /cvmfs/sft.cern.ch        # LCG software stacks (Python, ROOT, ...)
```

If those directories are empty or missing, CVMFS is not available and FLAF will not work.

## 3. A grid certificate and CMS VO membership

The pipeline reads CMS data from the grid (WLCG) and writes to grid storage, so you need a
**grid certificate** installed and to be a member of the **CMS Virtual Organisation (VO)**.

- Request a grid certificate and join the CMS VO by following the
  [CMS computing access guide](https://uscms.org/uscms_at_work/computing/getstarted/get_grid_cert.shtml)
  (one-time setup).
- Your certificate (`usercert.pem` / `userkey.pem`) lives in `~/.globus/`.

You will turn this certificate into a short-lived **VOMS proxy** every time you work — that step
is part of [Installation](installation.md).

!!! warning "VOMS membership is not instant"
    Joining the CMS VO requires approval and can take a day or two. Do this early.

## 4. SSH keys for GitHub **and** CERN GitLab

FLAF and the analyses live on **GitHub** (`github.com/cms-flaf/...`); some shared submodules
(the HH `inference` tooling) live on **CERN GitLab** (`gitlab.cern.ch`). Cloning with submodules
pulls from both, so you need an SSH key registered on each:

- GitHub → [github.com/settings/keys](https://github.com/settings/keys)
- CERN GitLab → [gitlab.cern.ch/-/profile/keys](https://gitlab.cern.ch/-/profile/keys)

Verify both work:

```sh
ssh -T git@github.com           # should greet you by username
ssh -T git@gitlab.cern.ch       # should welcome you
```

!!! tip "Why two hosts?"
    Most of FLAF is on GitHub. Only the combine-based HH statistical-inference submodule is on
    CERN GitLab. If a `git clone --recursive` stalls or fails on the `inference` submodule, a
    missing GitLab key is the usual cause.

## 5. Somewhere to work and to store output

- **Code** goes in your AFS work area (e.g. `/afs/cern.ch/work/<initial>/<user>/`), which has
  more quota than your home directory. The first environment build needs a few GB under the
  repository's `soft/` directory (CMSSW + a Python virtual environment).
- **Outputs** (ntuples, histograms) go to grid/EOS storage that you configure in
  `user_custom.yaml` — see the [Configuration guide](../configuration/user-custom.md).

## Checklist

- [ ] Can `ssh` to `lxplus` (`el9`)
- [ ] `/cvmfs/cms.cern.ch` and `/cvmfs/sft.cern.ch` are populated
- [ ] Grid certificate in `~/.globus/`, member of the CMS VO
- [ ] SSH keys registered on **both** GitHub and CERN GitLab
- [ ] A few GB of free quota in your AFS work area

All set? Continue to [Installation](installation.md).
