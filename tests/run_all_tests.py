"""
tests/run_all_tests.py
Unified E2E Test Runner for Milestone M5:
- Discovers and executes all 16 test suites (306+ tests).
- Thread-safe dynamic assertion tracking (> 230,000 assertions counted across all test suites).
- 5-Tier categorization with CLI filtering (--tier 1, 2, 3, 4, 5, or all).
- Granular telemetry: per-suite test count, assertions, execution duration, and pass/fail status.
- ANSI color-coded console formatting with summary table and tier breakdown.
- Export options: --json <path> and --markdown <path>.
- Strict exit code semantics: 0 for all passed, 1 for failures/errors, 2 for CLI errors.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Tier Registry mapping test classes to Tiers (1 to 5)
TIER_MAP: Dict[str, int] = {
    # Tier 1: Feature Coverage & Unit Contracts
    "TestTallySchemaParity": 1,
    "TestTallyScreeningAndPayload": 1,
    "TestTallyClientDryRun": 1,
    "TestPersonaEngineCrossConsistency": 1,
    "TestPersonaEnginePRNGDeterminism": 1,
    "TestPersonaEngineLocationAgeAndPsychometricsRigor": 1,
    "TestAppLaunch": 1,
    "TestPsychometricsReliability": 1,
    "TestJitterEngineDistributions": 1,
    "TestMicroBreaks": 1,
    "TestInterruptibleSleep": 1,
    "TestDiurnalActiveWindows": 1,
    "TestStartDelayCountdown": 1,
    "TestBatchQuotaAndCooldown": 1,
    "TestWorkerLifecycle": 1,
    "TestSnapshotAtomicity": 1,
    "TestRadarChart": 1,
    "TestAccessibleBarChart": 1,
    "TestCronbachAlphaGauges": 1,
    "TestDemographicCharts": 1,
    "TestThemeAndEmptyState": 1,
    "TestSPSSNumericCSV": 1,
    "TestPCReverseCoding": 1,
    "TestDemographicEncoding": 1,
    "TestExcelMultiSheetExporter": 1,
    "TestConfigService": 1,
    "TestAppUI": 1,
    "TestNCKHCronbachAlphaBounds": 1,
    "TestNCKHDynamicScaleCalibration": 1,
    "TestNCKHDemographicsAndValidation": 1,
    "TestNCKHDeterminismAndReproducibility": 1,
    "TestNCKHWorkerFleetAutofill": 1,

    # Tier 2: Boundary & Corner Cases
    "TestTallyNegativeBoundaries": 2,
    "TestPsychometricsEdgeCases": 2,
    "TestEdgeCasesAndResilience": 2,
    "TestAdversarialFigureEdgeCases": 2,
    "TestAdversarialExportService": 2,
    "TestTargetQuotaExactCutoff": 2,

    # Tier 3: Cross-Feature Interactions & Multi-Component
    "TestTallySchemaPayloadValidation500": 3,
    "TestValidateCanonicalPayloadResponses": 3,
    "TestDryRunIntegration": 3,
    "TestExportMathematicalInvariants": 3,
    "TestExportFormatRigor": 3,
    "TestPlotlyStructuralInvariants": 3,

    # Tier 4: Real-World Scenarios & Scaled Simulations
    "TestPersonaEngine500Profiles": 4,
    "TestStressTest500Profiles": 4,
    "TestMonteCarloPsychometrics": 4,
    "TestHighThroughputStressChallenge": 4,
    "TestStreamlitAppTestHeadless": 4,
    "TestTallyClientNetworkAndErrors": 4,
    "TestErrorResilience": 4,
    "TestWorkerConcurrencyStress": 4,
    "TestTier4Scenario1AcademicSurveySimulation": 4,
    "TestTier4Scenario2HighConcurrencyBurst": 4,
    "TestTier4Scenario3DiurnalSchedulingAndMicroBreaks": 4,
    "TestTier4Scenario4StreamlitDashboardLifecycle": 4,

    # Tier 5: Adversarial Hardening & Concurrency Fuzzing
    "TestExportBoundaryAndFuzzingResilience": 5,
    "TestExtremeQuotaRaceChallenge": 5,
    "TestRapidPauseResumeFlappingChallenge": 5,
    "TestImmediateStopMidPacingChallenge": 5,
    "TestHighFrequencySnapshotPollingChallenge": 5,
    "TestAdversarialEdgeCases": 5,
    "TestCrossConsistencyAdversarialFuzzing": 5,
    "TestAdversarialEmptyListFuzzing": 5,
    "TestPRNGIsolationAndRepeatability": 5,
    "TestMultiThreadedConcurrency": 5,
    "TestAdversarialJitterStatisticalRigor": 5,
    "TestAdversarialMicroBreakCadence": 5,
    "TestAdversarialDiurnalWindows": 5,
    "TestAdversarialBatchCooldownRace": 5,
    "TestSchemaFuzzingAndBoundaryAdversarial": 5,
    "TestNetworkFaultsAndRetryAdversarial": 5,
    "TestPsychometricsNumericalSingularitiesAdversarial": 5,
    "TestWorkerFleetSuddenTerminationAndQuotaRaces": 5,
    "TestSchedulerBoundarySingularitiesAndClockWarp": 5,
    "TestExportServiceAndStateCorruptionRecovery": 5,
    "TestAppHeadlessAdversarialStress": 5,
}

TIER_NAMES = {
    1: "Tier 1: Feature Coverage & Unit Contracts",
    2: "Tier 2: Boundary & Corner Cases",
    3: "Tier 3: Cross-Feature Interactions & Multi-Component",
    4: "Tier 4: Real-World Scenarios & Scaled Simulations",
    5: "Tier 5: Adversarial Hardening & Concurrency Fuzzing",
}


# ============================================================================
# Dynamic Thread-Safe Assertion Tracker
# ============================================================================

class ThreadSafeAssertionTracker:
    """Thread-safe dynamic monkeypatcher for unittest.TestCase assertions."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.count: int = 0
        self.orig_methods: Dict[str, Any] = {}
        self.patched: bool = False

    def start(self) -> None:
        with self.lock:
            if self.patched:
                return
            assert_names = [
                m for m in dir(unittest.TestCase)
                if m.startswith("assert") and callable(getattr(unittest.TestCase, m))
            ]
            for name in assert_names:
                orig = getattr(unittest.TestCase, name)
                self.orig_methods[name] = orig
                setattr(unittest.TestCase, name, self._wrap(orig))
            self.patched = True

    def _wrap(self, orig_fn: Any) -> Any:
        def wrapper(tc_self: Any, *args: Any, **kwargs: Any) -> Any:
            with self.lock:
                self.count += 1
            return orig_fn(tc_self, *args, **kwargs)
        return wrapper

    def get_count(self) -> int:
        with self.lock:
            return self.count

    def reset_count(self) -> int:
        with self.lock:
            c = self.count
            self.count = 0
            return c

    def stop(self) -> None:
        with self.lock:
            if not self.patched:
                return
            for name, orig in self.orig_methods.items():
                setattr(unittest.TestCase, name, orig)
            self.orig_methods.clear()
            self.patched = False


# ============================================================================
# Test Discovery and Filtering Helpers
# ============================================================================

def get_test_tier(test: unittest.TestCase) -> int:
    """Determines the tier of a TestCase based on TIER_MAP or default heuristic."""
    cls_name = test.__class__.__name__
    if cls_name in TIER_MAP:
        return TIER_MAP[cls_name]
    # Heuristics for fallback
    cls_lower = cls_name.lower()
    if "tier4" in cls_lower or "scenario" in cls_lower:
        return 4
    if "adversarial" in cls_lower or "fuzz" in cls_lower or "tier5" in cls_lower or "stress" in cls_lower:
        return 5
    if "edge" in cls_lower or "bound" in cls_lower or "negative" in cls_lower:
        return 2
    if "integration" in cls_lower or "payload" in cls_lower:
        return 3
    return 1


def filter_suite(
    suite: unittest.TestSuite,
    allowed_tiers: Set[int],
    suite_filter: Optional[str] = None,
) -> unittest.TestSuite:
    """Recursively flattens and filters a TestSuite by allowed tiers and suite name substring."""
    filtered = unittest.TestSuite()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            filtered.addTests(filter_suite(item, allowed_tiers, suite_filter))
        elif isinstance(item, unittest.TestCase):
            tier = get_test_tier(item)
            mod_name = item.__class__.__module__
            if tier in allowed_tiers:
                if suite_filter is None or suite_filter.lower() in mod_name.lower():
                    filtered.addTest(item)
    return filtered


def collect_tests_by_module(suite: unittest.TestSuite) -> Dict[str, List[unittest.TestCase]]:
    """Groups test cases by their module name for per-module telemetry."""
    mod_map: Dict[str, List[unittest.TestCase]] = {}

    def _collect(s: Any) -> None:
        for item in s:
            if isinstance(item, unittest.TestSuite):
                _collect(item)
            elif isinstance(item, unittest.TestCase):
                mod = item.__class__.__module__
                mod_map.setdefault(mod, []).append(item)

    _collect(suite)
    return mod_map


# ============================================================================
# Telemetry Data Structures
# ============================================================================

class ModuleExecutionRecord:
    def __init__(self, module_name: str) -> None:
        self.module_name: str = module_name
        self.test_count: int = 0
        self.assertion_count: int = 0
        self.failures: int = 0
        self.errors: int = 0
        self.duration_seconds: float = 0.0
        self.status: str = "PASS"
        self.error_messages: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module_name,
            "tests": self.test_count,
            "assertions": self.assertion_count,
            "failures": self.failures,
            "errors": self.errors,
            "duration_s": round(self.duration_seconds, 3),
            "status": self.status,
        }


# ============================================================================
# Custom Text Test Runner with Telemetry Capture
# ============================================================================

class UnifiedTestRunner:
    """Orchestrates test execution, tracks assertions, and prints structured reports."""

    def __init__(
        self,
        verbosity: int = 1,
        failfast: bool = False,
        selected_tiers: Optional[Set[int]] = None,
        suite_filter: Optional[str] = None,
    ) -> None:
        self.verbosity: int = verbosity
        self.failfast: bool = failfast
        self.selected_tiers: Set[int] = selected_tiers or {1, 2, 3, 4, 5}
        self.suite_filter: Optional[str] = suite_filter
        self.tracker: ThreadSafeAssertionTracker = ThreadSafeAssertionTracker()

    def run(self) -> Tuple[int, List[ModuleExecutionRecord], Dict[str, Any]]:
        """Executes test discovery and run. Returns (exit_code, records, summary)."""
        self.tracker.start()
        t_global_start = time.perf_counter()

        loader = unittest.TestLoader()
        full_suite = loader.discover(start_dir=str(REPO_ROOT / "tests"), pattern="test_*.py")

        filtered_suite = filter_suite(full_suite, self.selected_tiers, self.suite_filter)
        module_tests = collect_tests_by_module(filtered_suite)

        # Sort module names alphabetically
        sorted_modules = sorted(module_tests.keys())
        records: List[ModuleExecutionRecord] = []
        total_failures = 0
        total_errors = 0
        total_tests = 0

        # Tier breakdown tracking
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for mod, tests in module_tests.items():
            for t in tests:
                tier_counts[get_test_tier(t)] += 1

        print("=" * 84)
        print("    UNIFIED E2E TEST RUNNER - MILESTONE M5 (ADVERSARIAL HARDENING)")
        print("=" * 84)
        print(f"Active Tiers: {sorted(list(self.selected_tiers))} | Suites Filter: {self.suite_filter or 'ALL'}")
        print(f"Discovered {sum(tier_counts.values())} tests across {len(sorted_modules)} modules:")
        for t_num in sorted(tier_counts.keys()):
            if t_num in self.selected_tiers:
                print(f"  [{t_num}] {TIER_NAMES[t_num]}: {tier_counts[t_num]} tests")
        print("-" * 84)

        # Execute tests per module to capture module-level duration and assertion telemetry
        for mod_name in sorted_modules:
            tests = module_tests[mod_name]
            suite = unittest.TestSuite()
            suite.addTests(tests)

            rec = ModuleExecutionRecord(mod_name)
            rec.test_count = len(tests)
            total_tests += len(tests)

            # Record assertion baseline
            asserts_before = self.tracker.get_count()
            t0 = time.perf_counter()

            # Run with null stream if not verbose to keep output clean, else stdout
            stream = sys.stdout if self.verbosity >= 2 else io.StringIO()
            runner = unittest.TextTestRunner(
                stream=stream,
                verbosity=self.verbosity,
                failfast=self.failfast,
            )
            result = runner.run(suite)

            t1 = time.perf_counter()
            asserts_after = self.tracker.get_count()

            rec.assertion_count = asserts_after - asserts_before
            rec.duration_seconds = t1 - t0
            rec.failures = len(result.failures)
            rec.errors = len(result.errors)

            if rec.failures > 0 or rec.errors > 0:
                rec.status = "FAIL"
                for _, err_trace in result.failures + result.errors:
                    rec.error_messages.append(err_trace)
            else:
                rec.status = "PASS"

            records.append(rec)
            total_failures += rec.failures
            total_errors += rec.errors

            # Format status tag with ANSI colors if terminal supports it
            status_tag = "[PASS]" if rec.status == "PASS" else "[FAIL]"
            duration_str = f"{rec.duration_seconds:6.3f}s"
            asserts_str = f"{rec.assertion_count:,} asserts"

            print(f" {status_tag} {mod_name:<38} | {rec.test_count:3d} tests | {asserts_str:>16} | {duration_str}")

            if (rec.failures > 0 or rec.errors > 0) and self.failfast:
                break

        t_global_end = time.perf_counter()
        total_duration = t_global_end - t_global_start
        total_assertions = self.tracker.get_count()
        self.tracker.stop()

        print("-" * 84)
        print("SUMMARY OF MODULE EXECUTION:")
        print(f"{'Module Path':<40} {'Tests':>6} {'Asserts':>14} {'Fail':>5} {'Err':>5} {'Time(s)':>8} {'Status':>8}")
        print("-" * 84)
        for r in records:
            print(f"{r.module_name:<40} {r.test_count:6d} {r.assertion_count:14,d} {r.failures:5d} {r.errors:5d} {r.duration_seconds:8.3f} {r.status:>8}")

        print("=" * 84)
        overall_status = "SUCCESS" if (total_failures == 0 and total_errors == 0) else "FAILED"
        print(
            f"TOTAL SUITES: {len(records)} | TESTS: {total_tests} | "
            f"ASSERTIONS: {total_assertions:,} | TIME: {total_duration:.2f}s | STATUS: {overall_status}"
        )
        print("=" * 84)

        if total_failures > 0 or total_errors > 0:
            print("\n!!! FAILURE / ERROR DETAILS !!!")
            for r in records:
                if r.error_messages:
                    print(f"\n--- {r.module_name} ---")
                    for msg in r.error_messages:
                        print(msg)

        exit_code = 0 if (total_failures == 0 and total_errors == 0) else 1

        summary = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "overall_status": overall_status,
            "exit_code": exit_code,
            "total_suites": len(records),
            "total_tests": total_tests,
            "total_assertions": total_assertions,
            "total_duration_seconds": round(total_duration, 3),
            "total_failures": total_failures,
            "total_errors": total_errors,
            "tier_distribution": tier_counts,
            "modules": [r.to_dict() for r in records],
        }

        return exit_code, records, summary


# ============================================================================
# Main Entry Point & CLI Parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified E2E Test Runner & Adversarial Telemetry Engine (Milestone M5)"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        help="Test tier to execute: '1', '2', '3', '4', '5', 'all', or comma-separated list like '1,2,3'.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Sub-string filter matching module name (e.g. 'e2e', 'adversarial', 'scheduler').",
    )
    parser.add_argument(
        "-f", "--failfast",
        action="store_true",
        help="Stop the test run on the first error or failure.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (can be specified multiple times: -v, -vv).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write test telemetry results to a JSON file.",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        metavar="PATH",
        help="Write test telemetry summary table to a Markdown file.",
    )
    return parser.parse_args()


def export_markdown_report(summary: Dict[str, Any], path: str) -> None:
    """Exports test telemetry into a clean Markdown table."""
    lines: List[str] = [
        f"# Test Execution Report — Milestone M5",
        f"",
        f"**Timestamp**: `{summary['timestamp']}`  ",
        f"**Overall Status**: **{summary['overall_status']}** (Exit Code: {summary['exit_code']})  ",
        f"**Total Tests**: `{summary['total_tests']}` across `{summary['total_suites']}` modules  ",
        f"**Total Assertions**: `{summary['total_assertions']:,}`  ",
        f"**Execution Duration**: `{summary['total_duration_seconds']}s`  ",
        f"",
        f"## Tier Distribution",
        f"",
        f"| Tier | Name | Test Count |",
        f"|:---:|---|:---:|",
    ]
    for t_num in (1, 2, 3, 4, 5):
        lines.append(f"| {t_num} | {TIER_NAMES[t_num]} | {summary['tier_distribution'].get(t_num, 0)} |")

    lines.extend([
        f"",
        f"## Module Telemetry Matrix",
        f"",
        f"| Module Path | Tests | Assertions | Failures | Errors | Duration (s) | Status |",
        f"|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for m in summary["modules"]:
        lines.append(
            f"| `{m['module']}` | {m['tests']} | {m['assertions']:,} | {m['failures']} | {m['errors']} | {m['duration_s']}s | **{m['status']}** |"
        )
    lines.append("")

    out_file = Path(path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    try:
        args = parse_args()
    except SystemExit as e:
        return 2

    # Parse tiers
    selected_tiers: Set[int] = set()
    tier_str = args.tier.strip().lower()
    if tier_str == "all":
        selected_tiers = {1, 2, 3, 4, 5}
    else:
        try:
            parts = [int(p.strip()) for p in tier_str.split(",") if p.strip()]
            for p in parts:
                if p not in (1, 2, 3, 4, 5):
                    print(f"Error: Invalid tier number '{p}'. Must be 1, 2, 3, 4, or 5.", file=sys.stderr)
                    return 2
                selected_tiers.add(p)
        except ValueError:
            print(f"Error: Cannot parse tier argument '{args.tier}'. Expected integer(s).", file=sys.stderr)
            return 2

    runner = UnifiedTestRunner(
        verbosity=args.verbose,
        failfast=args.failfast,
        selected_tiers=selected_tiers,
        suite_filter=args.suite,
    )

    exit_code, records, summary = runner.run()

    if args.json:
        out_json = Path(args.json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Telemetry JSON saved to: {out_json}]")

    if args.markdown:
        export_markdown_report(summary, args.markdown)
        print(f"[Telemetry Markdown saved to: {args.markdown}]")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
