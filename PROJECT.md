# Project: Autofill Form — Tally J118j4 Academic Research Automation

## Architecture
- **Language & Runtime**: Python 3.13.5
- **Core Libraries**: `streamlit`, `httpx`, `pydantic`, `numpy`, `pandas`, `plotly`, `scipy`, `xlsxwriter`
- **Design Pattern**: Decoupled Layered Architecture with Singleton Worker Fleet
  - **Schema Layer** (`src/tally_schema.py`): Canonical 52 question UUIDs, 293 option UUIDs, screening logic.
  - **Psychometrics Layer** (`src/persona_engine.py`, `src/stats_analyzer.py`): Gen Z demographic synthesis, 11-scale TAM & TPB Latent Trait engine (Cronbach's $\alpha \ge 0.82$), reverse coding PC, streaming stats.
  - **API Client Layer** (`src/tally_client.py`): Next.js headers, session/respondent UUID isolation, retry backoff, Dry-Run simulation.
  - **Concurrency & Scheduling Layer** (`src/worker_manager.py`, `src/scheduler.py`): 1–20 worker threads, 6-state FSM, snapshot-based thread-safety, Uniform/Gaussian jitter engine, diurnal time windows, batch limits.
  - **UI/UX & Export Layer** (`app.py`, `src/ui_components.py`, `src/export_service.py`): Streamlit dashboard (`ui-ux-pro-max` Swiss Minimalism & Data-Dense), Plotly charts (Demographics, 11-axis Radar + Error Bar chart), SPSS numeric CSV & Excel Codebook exporter.
  - **E2E Test Suite** (`tests/`): Independent opaque-box test runner & 4-tier test cases + adversarial coverage.

## Code Layout
```
d:\Github\Autofill-Form\
├── app.py                     # Streamlit web dashboard application
├── requirements.txt           # Pinned production dependencies
├── PROJECT.md                 # Living project architecture and milestone index
├── src/
│   ├── __init__.py
│   ├── tally_schema.py        # Verified 52 question schema & screening logic
│   ├── tally_client.py        # HTTP client for api.tally.so & Dry-run simulator
│   ├── persona_engine.py      # Gen Z demographics & TAM/TPB Latent Trait engine
│   ├── stats_analyzer.py      # Real-time streaming Cronbach's Alpha & stats
│   ├── worker_manager.py      # 1-20 Worker thread pool & FSM snapshot manager
│   ├── scheduler.py           # Jitter engine, diurnal time windows, batch cadences
│   ├── export_service.py      # SPSS numeric CSV & Excel codebook exporter
│   └── ui_components.py       # Reusable ui-ux-pro-max components & Plotly charts
└── tests/
    ├── __init__.py
    ├── run_all_tests.py       # Unified E2E test runner
    ├── test_persona_engine.py # Demographics cross-consistency & sampling
    ├── test_psychometrics.py  # Cronbach's alpha >= 0.82 & PC reversal
    ├── test_tally_client.py   # Dry-run mock & 52-question UUID verification
    ├── test_worker_manager.py # Multi-threaded concurrency & FSM transitions
    ├── test_scheduler.py      # Jitter distributions & scheduling windows
    └── test_export_spss.py    # SPSS 1-6 coding & codebook verification
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Canonical 52 Question Schema | Mapping of all 52 question UUIDs and 293 choice options from `consolidated_schema.json` | M1 | Survey / `consolidated_schema.json` |
| 2 | Anti-Disqualification Screening Logic | Enforce consent and active Shopee user selection to prevent early drop-off at Page 7 | M1 | Survey / `tally_next_data.json` |
| 3 | Tally HTTP API Client | `POST https://api.tally.so/forms/J118j4/respond` with Next.js headers, exponential backoff | M1 | Survey / `Antigravity.md` |
| 4 | Session & Respondent Isolation | Independent UUID v4 generated for `sessionUuid` and `respondentUuid` per submission | M1 | Survey / `tally_schema.py` |
| 5 | Zero-Network Dry-Run Mode | Simulated execution pipeline returning `mock-dryrun-<uuid>` without HTTP POST | M1 | Survey / `ORIGINAL_REQUEST.md` |
| 6 | Gen Z Demographic Profiling | Probabilistic generator for 18–27 age, gender, location, education, job, shopping habits | M2 | Survey / `ORIGINAL_REQUEST.md` |
| 7 | Cross-Consistency Validation | 6 logical cross-checks (age vs education, job vs profession, shopping habits) | M2 | Survey / `spec_psychometrics_report.md` |
| 8 | 11-Construct TAM & TPB Modeling | 39 Likert items (1–6) across PU, PEOU, PES, PC, PI, SE, LC, ATT, SN, PBC, ITU | M2 | Survey / `ORIGINAL_REQUEST.md` |
| 9 | Reverse-Coded PC Scale | Invert Perceived Costs ($PC_{\text{rev}} = 7 - PC_{\text{raw}}$) for SEM and Cronbach's Alpha | M2 | Survey / `Antigravity.md` |
| 10 | Guaranteed High Reliability | Latent trait algorithm ensuring Cronbach's $\alpha \ge 0.82$ on all scales and overall | M2 | Survey / `ORIGINAL_REQUEST.md` |
| 11 | Real-time Streaming Psychometrics | Online calculation of variance, covariance matrix, and Cronbach's Alpha without data lag | M2 | Survey / `spec_psychometrics_report.md` |
| 12 | 1–20 Multi-Threaded Worker Pool | Thread pool with lifecycle states (IDLE, SYNTHESIZING, PACING, SUBMITTING, SUCCESS, ERROR) | M3 | Survey / `explorer_arch` |
| 13 | Thread-Safe Snapshot Architecture | Immutable `WorkerFleetSnapshot` preventing race conditions and UI lockups | M3 | Survey / `architecture_survey_report.md` |
| 14 | Dual-Distribution Jitter Engine | Uniform $\mathcal{U}[t_{min}, t_{max}]$ and Truncated Gaussian $\mathcal{N}(\mu, \sigma^2)$ with micro-breaks | M3 | Survey / `automation` skill |
| 15 | Diurnal & Windowed Scheduling | Start delay timer, daytime operating windows, batch quotas with cooldown pauses | M3 | Survey / `automation` skill |
| 16 | Streamlit UI/UX Pro Max Dashboard | Swiss Minimalism & Data-Dense layout, WCAG 2.1 AA 4.5:1 contrast, 5 KPI metric cards | M4 | Survey / `ui-ux-pro-max` skill |
| 17 | Interactive Configuration Controls | Sliders and inputs for target count, worker threads, jitter parameters, demographics, dry-run | M4 | Survey / `ORIGINAL_REQUEST.md` |
| 18 | Live Fleet Grid & Monitor | Real-time worker cards displaying status, progress, current latency, and error logs | M4 | Survey / `architecture_survey_report.md` |
| 19 | Dual-Mode Psychometric Charts | 11-construct Plotly Radar chart paired with accessible Horizontal Bar Chart with error bars | M4 | Survey / `ui-ux-pro-max` skill |
| 20 | SPSS & SmartPLS Numeric Exporter | CSV with 1–6 numeric encoding and Excel `.xlsx` with automated Variable Codebook sheet | M4 | Survey / `ORIGINAL_REQUEST.md` |
| 21 | Comprehensive Opaque-Box E2E Suite | Automated test harness executing Tiers 1–4 tests with 100% pass guarantee | M5 | Survey / Dual Track |
| 22 | Adversarial Coverage Hardening | Tier 5 edge case injection and boundary fuzzing | M5 | Survey / Dual Track |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Tally Schema & HTTP Client Engine | `src/tally_schema.py`, `src/tally_client.py` | None | DONE |
| M2 | Psychometrics & Persona Engine | `src/persona_engine.py`, `src/stats_analyzer.py` | None | DONE |
| M3 | Worker Fleet Manager & Automation | `src/worker_manager.py`, `src/scheduler.py` | M1, M2 | DONE |
| M4 | Streamlit Dashboard & SPSS Export | `app.py`, `src/ui_components.py`, `src/export_service.py` | M1, M2, M3 | PLANNED |
| M5 | E2E Test Suite Pass & Adversarial Hardening | Full system verification against `TEST_READY.md` (Tiers 1–5) | M4, E2E Track | PLANNED |

## Interface Contracts
### `src/tally_schema.py` ↔ `src/tally_client.py`
- `TallySchema.get_payload(responses: dict, session_uuid: str, respondent_uuid: str, is_completed: bool = True) -> dict`
- `TallySchema.validate_payload(payload: dict) -> bool`
- `TallySchema.QUESTION_MAP`: Dictionary of 52 questions indexed by `group_uuid`.

### `src/persona_engine.py` ↔ `src/tally_client.py` & `src/worker_manager.py`
- `PersonaEngine.generate_respondent(demographic_weights: dict = None) -> RespondentProfile`:
  - Returns `profile.demographics: dict` (mapped to option UUIDs)
  - Returns `profile.likert_scores: dict[str, int]` (raw 1–6 scores for 39 questions)
  - Returns `profile.payload_responses: dict` (ready for `TallySchema.get_payload()`)

### `src/stats_analyzer.py` ↔ `app.py` & `src/worker_manager.py`
- `StatsAnalyzer.add_sample(likert_scores: dict[str, int]) -> None`:
  - Inverts PC scores internally for psychometric validity.
  - Updates running sample count, construct means, covariance matrices.
- `StatsAnalyzer.get_cronbach_alpha() -> dict[str, float]`:
  - Returns alpha per construct and overall model alpha ($\ge 0.82$).
- `StatsAnalyzer.get_summary() -> dict`:
  - Construct means, standard deviations, response counts.

### `src/worker_manager.py` ↔ `app.py`
- `WorkerManager.configure(target_count: int, num_workers: int, jitter_config: dict, dry_run: bool) -> None`
- `WorkerManager.start() -> None`
- `WorkerManager.pause() -> None`
- `WorkerManager.resume() -> None`
- `WorkerManager.stop() -> None`
- `WorkerManager.get_snapshot() -> WorkerFleetSnapshot`:
  - Atomic, thread-safe snapshot: total submitted, success count, error count, active workers, live worker states.
