# Storage & filesystems

FLAF reads CMS data from the grid and writes large outputs (ntuples, histograms) to grid/EOS
storage, while keeping small artifacts locally. It abstracts every location behind a **named
filesystem** — a `fs_*` key you set in [`user_custom.yaml`](../configuration/user-custom.md).

## Named filesystems (`fs_*`)

Each output type has a filesystem name. You only have to set `fs_default`; the others fall back to
it when unset, so a one-line configuration is enough to get going.

| Key | Used for |
|---|---|
| `fs_default` | The fallback for everything below. **The one key you must set.** |
| `fs_anaTuple` | Merged analysis ntuples (anaTuples). |
| `fs_anaCacheTuple` | Cached per-event payloads. |
| `fs_HistTuple` | histTuples (ntuples with analysis observables). |
| `fs_plots` | Plot outputs. |
| `fs_nanoAOD` | Location of NanoAOD inputs for **custom/local** samples (not from DAS). |
| `fs_das` | The DAS-backed filesystem used to resolve official datasets. |

!!! tip "Start with just `fs_default`"
    Set `fs_default` to your personal storage and leave the rest unset. Everything then lands in
    one place, namespaced by `--version` and era. Split outputs across sites later, when you need
    to (e.g. point `fs_anaTuple` at a Tier-2/Tier-3 with lots of space).

## How to write a location

A filesystem value is a storage URL (or a list of them). Two common forms:

```yaml
# EOS via WebDAV (your CERNBox / EOS user area):
fs_default: davs://eoshome-k.cern.ch:8444/eos/user/k/kandroso/FLAF/HH_bbtautau/

# A WLCG site (Tier-3/Tier-2) by its name + logical path:
fs_anaTuple: T3_CH_CERNBOX:/store/user/<user>/HH_bbtautau/
```

- `davs://…` is a direct WebDAV endpoint (good for your EOS user area).
- `T3_CH_CERNBOX:/store/...` names a registered WLCG site and a logical path under it; FLAF/LAW
  resolves it to a real endpoint. `T3_CH_CERNBOX` (CERNBox) and `T3_US_FNALLPC` (FNAL LPC) are
  common choices.
- A **local absolute path** (e.g. `/builds/.../output/HH_bbtautau`) is also valid — CI uses one so
  its outputs stay on the runner.

A value may also be a **list**, in which case the entries are tried in order — useful for
read-fallback across mirrors.

## Local working area: `data/`

Independently of the `fs_*` storage, each analysis checkout has a `data/` directory used for:

- your **VOMS proxy** (`data/voms.proxy`, where `X509_USER_PROXY` points),
- LAW job files and logs,
- small local copies of outputs.

This lives in your AFS checkout, not on grid storage.

## The VOMS proxy is your storage key

All grid/EOS access uses your **VOMS proxy**. If it expires, reads and writes to `fs_*` locations
fail — often with confusing "permission denied" or "file not found" messages. Refresh it with
`voms-proxy-init -voms cms -rfc -valid 192:00` (see [Installation](../getting-started/installation.md)).

## A caveat worth knowing: read-after-write lag on EOS

EOS is eventually consistent. A file you just **wrote** can be briefly **invisible** to a
subsequent existence check (seconds, occasionally longer). FLAF tolerates this in normal
operation, but if you script your own existence checks against freshly written outputs, probe with
a directory listing and a short retry rather than a single `exists()`. See
[Troubleshooting](../troubleshooting.md#eos-read-after-write-lag).

## Keeping I/O off shared production areas

When testing, point `fs_default` at *your own* area (and use a personal `--version`) so you never
write into a shared production path. The CI does the inverse — it points `fs_default` at the local
runner so a test never touches real storage at all.
