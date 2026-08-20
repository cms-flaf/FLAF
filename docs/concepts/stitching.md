# MC stitching

Several backgrounds are generated as a set of overlapping samples — an inclusive one plus
samples enriched in a corner of phase space (jet multiplicity, boson p<sub>T</sub>, a decay
channel, a generator-level filter). Stitching combines them into a single process: every
event is assigned to exactly one **bin**, and the events of that bin are normalised with the
cross-section and the event count of all samples that contribute to it.

## Declaring a stitcher

A process declares a stitching processor in its `processes.yaml` entry:

```yaml
TT:
  processors:
    - name: Stitcher
      module: FLAF.Processors.MCStitchingTT
      class: TTStitcher
      config: FLAF/config/Processors/stitching_TT_decayMode.yaml
      stages: [ AnaTuple, AnaTupleMerge ]
      dependency_level:
        AnaTuple: file
        AnaTupleMerge: process
  genInfo: [ TT ]
  datasets:
    - TTto2L2Nu
    - TTtoLNu2Q
    - TTto4Q
```

The `config` file lists the bins, each with a `selection` (a C++ expression) and a
`crossSection` (an expression over the cross-section database):

```yaml
bins:
  - name: SingleLepton
    selection: "TT_n_leptonic_W == 1"
    crossSection: TTtoLNu2Q
```

The bins must be **orthogonal and exhaustive**: an event that matches no bin aborts the job,
and an event that matches two is counted twice. When `totalCrossSection` is given, the sum of
the bin cross-sections is checked against it.

## Variables the bins select on

`LHE_Vpt`, `LHE_NpNLO` and friends are nanoAOD branches that the anaTuple keeps, so a bin can
select on them directly. Anything else — a gen-level decay mode, a dilepton mass, the
generator filter of a sample — has to be derived from `GenPart`/`LHEPart`, and those
collections are **not** part of the anaTuple.

This matters because the bin selections are evaluated **twice**:

| Stage | Frame | Purpose |
|---|---|---|
| `AnaTupleFileTask` | nanoAOD | sum the denominator of each bin into the report |
| `AnaTupleMergeTask` | merged anaTuple | give each event the cross-section and denominator of its bin |

The merge stage no longer has `GenPart`/`LHEPart`, so a derived variable must be **stored in
the anaTuple** by the analysis. Which gen-level information a process needs is declared as
`genInfo` next to its `processors`, and the analysis-level anaTuple definition turns that into
branches (`DYInfo_*`, `TauTauInfo_*`, `TTInfo_*`). A stitcher then prefers the stored branch
and falls back to the nanoAOD collections only when it is not there:

```python
class TTStitcher(MCStitcher):
    def defineVariables(self, df):
        df = defineFromStoredOrExpression(
            df,
            "TT_n_leptonic_W",
            stored="TTInfo_nLeptonicW",
            expression="gen_process::tt::identify(...).nLeptonicW()",
            prepare=_prepare,
        )
        return super().defineVariables(df)
```

Both stages therefore see the same value, computed once from the nanoAOD.

!!! warning "Adding a variable to a bin selection changes the anaTuple"
    A bin that selects on a new derived variable needs that variable stored, which means the
    affected samples have to be produced again. A stitcher whose variable is missing from the
    anaTuple fails in `AnaTupleMergeTask` with `use of undeclared identifier 'GenPart_…'`.

## Adding a stitcher

1. Put the gen-level identification in a self-contained header under
   `FLAF/include/GenProcess/` and give it a test in `FLAF/test/GenProcess/`.
2. Subclass `MCStitcher` and override `defineVariables` with
   `defineFromStoredOrExpression`, naming the branch the analysis stores.
3. Store that branch in the analysis anaTuple definition and declare the corresponding
   `genInfo` for the process.
4. Cover it in the integration test: a CI process group that uses the stitcher runs the whole
   anaTuple → merge → histogram chain, which is where a missing branch shows up.
