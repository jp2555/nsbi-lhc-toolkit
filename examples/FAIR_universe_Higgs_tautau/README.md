FAIR Universe Dataset
--

The tabular dataset used in this demonstration is hosted on Zenodo (https://zenodo.org/records/15131565), and is created using the particle physics simulation tools Pythia 8.2 and Delphes 3.5.0. The dataset provides events for the $H\to \tau\tau$ analysis, where the signal process is sub-dominant compared to the very large $Z\to \tau\tau$ and other backgrounds - good challenge to test the sensitivty of NSBI techniques.

## Download saved models and processed data 

If you need access to pre-trained ensemble neural networks and preprocessed data, to avoid running each notebook in sequence but rather pick and choose any of them, download the directory from [LINK TBA]().

## Running a HTCondor workflow with DAGMan

If you have access to HTCondor resources, check out the `htcondor/` directory for DAGMan workflow that can be modified to your setup. Here is a preview of the full DAG:

```mermaid
flowchart TD
    classDef stageLabel fill:#2c3e6b,stroke:#1a2a4a,color:#fff,font-weight:bold,font-size:14px
    classDef job fill:#16213e,stroke:#3d5a99,color:#e0e0e0,font-size:13px
    classDef parallel fill:#1a3a5c,stroke:#4a7ab5,color:#e0e0e0,font-size:12px
    classDef pre fill:#7d5a1e,stroke:#c47d0e,color:#fff,font-size:12px
    classDef eval fill:#1e5e3e,stroke:#27ae60,color:#fff,font-weight:bold
    classDef fit fill:#4a235a,stroke:#8e44ad,color:#fff,font-weight:bold

    %% ─── STAGE 1 & 2 ───
    J1["Dataset Loader"]
    J2["Preprocessing Script"]
    J1 -->|"Load data"| J2

    J3["Preselection Network"]
    J2 -->|"Apply feature engineering / processing"| J3

    %% ─── STAGE 3 ───
    DRT["Density Ratio Training"]
    J3 -->|"Extract Signal Region"| DRT

    PRE_NOM["Generate Parallel Training DAG"]
    PRE_SYS["Generate Parallel Training DAG"]

    DRT -->|"Submit parallel training jobs"| PRE_NOM
    DRT -->|"Submit parallel training jobs"| PRE_SYS

    subgraph NOMINAL["Nominal Density Ratios"]
        direction LR
        subgraph P3["Process M"]
            N1["Ensemble member 0"]
            N2["Ensemble member 1"]
            ND["···"]
            NN2["Ensemble member N"]
        end
        subgraph P2["Process 2"]
            Z1["Ensemble member 0"]
            Z2["Ensemble member 1"]
            ZD["···"]
            ZN["Ensemble member N"]
        end
        subgraph P1["Process 1"]
            T1["Ensemble member 0"]
            T2["Ensemble member 1"]
            TD["···"]
            TN["Ensemble member N"]
        end
    end

    subgraph SYSTEMATICS["Systematic Variation Ratios"]
        direction LR
        subgraph SP3["Process M"]
            SN1["NP 1 Up"]
            SN2["NP 1 Down"]
            SND["···"]
            SNN["NP K Up / Down"]
        end
        subgraph SP2["Process 2"]
            SZ1["NP 1 Up"]
            SZ2["NP 1 Down"]
            SZD["···"]
            SZN["NP K Up / Down"]
        end
        subgraph SP1["Process 1"]
            S1["NP 1 Up"]
            S2["NP 1 Down"]
            SD["···"]
            SN["NP K Up / Down"]
        end
    end

    PRE_NOM --> NOMINAL
    PRE_SYS --> SYSTEMATICS

    EVAL["Neural Network Evaluation — Ensemble Aggregation"]
    NOMINAL -->|"Predicted ratios"| EVAL
    SYSTEMATICS -->|"Predicted ratios"| EVAL

    STAT["Statistical Model"]
    EVAL -->|"Aggregated density ratios"| STAT

    FIT["Parameter Fitting"]
    STAT -->|"Model for hypothesis test"| FIT

    class J1,J2,J3 job
    class DRT stageLabel
    class T1,T2,TD,TN,Z1,Z2,ZD,ZN,N1,N2,ND,NN2 parallel
    class S1,S2,SD,SN,SZ1,SZ2,SZD,SZN,SN1,SN2,SND,SNN parallel
    class PRE_NOM,PRE_SYS pre
    class EVAL eval
    class FIT,STAT fit
```

## Systematics-robustness diagnostic

`scripts/systematics_robustness.py` tests whether a discriminant is *robust* under the
real FAIR Universe systematics (TES, JES) — the axis that BCE/AUC cannot see and the one
where a foundation model could help a precision measurement. For each region
(`Nominal`, `JES_Up/Dn`, `TES_Up/Dn`) it scores the signal-region events with the
**nominal** density-ratio ensemble and measures how much the discriminant's output
distribution moves vs nominal (total-variation distance, mean shift). Run it **after**
the nominal density ratios are trained:

```bash
python scripts/systematics_robustness.py --config config.pipeline.yaml \
    --process htautau --out-dir output/robustness
```

Writes `robustness.json` + `robustness.png` (shape shift per systematic; lower = more robust).

The *downstream* impact — whether the profiled μ CI shrinks without systematics — is now
reported by `scripts/parameter_fitting.py` itself: it extracts the ±1σ CI (via
`ci.ci_from_scan`) from the stat+syst and stat-only profile scans it already computes,
logs the **systematic inflation** (stat+syst / stat-only half-width), and saves
`mu_ci.json`. Together these answer "how much do systematics move the discriminant, and
how much do they cost the measurement" — for any discriminant, including an FM once one
that consumes these inputs is plugged in.
