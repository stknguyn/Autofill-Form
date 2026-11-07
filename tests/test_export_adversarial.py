"""
tests/test_export_adversarial.py - Adversarial Stress & Mathematical Invariant Test Suite for SPSS & Excel Export Service.
Milestone M4 Challenger 2.

Covers:
1. Mathematical Verification:
   - Pearson correlation r(PC_raw, PC_rev) = -1.0 across arbitrary sample sizes (N=2, 5, 10, 50, 100, 500, 1000).
   - Strict arithmetic identity: PC_raw + PC_rev == 7 for all observations and items.
   - Strict Likert bounds [1, 6] across standard and adversarial inputs (floats, negative, out-of-range, strings).
   - Composite construct means arithmetic accuracy and bounds [1.0, 6.0].
2. Format Rigor:
   - CSV parsing with standard csv.reader across comma, semicolon, tab delimiters.
   - RFC 4180 escaping, UTF-8 BOM encoding, exact 73 columns, zero NaNs or empty cells.
   - OPC XML inspection of .xlsx: workbook.xml, sheets (Data, Variable Codebook, Psychometric Summary),
     styles.xml, sharedStrings.xml, and purely numeric cell XML types in Sheet 1.
   - Variable Codebook 100% 1-to-1 match with Data matrix columns.
3. Boundary & Malformed Input Fuzzing:
   - Empty input list [] (CSV, Excel, Codebook, Psychometrics).
   - Single respondent [profile] (N=1 degrees of freedom edge case).
   - Incomplete profiles (missing demographic keys, partial Likert maps, custom objects).
   - Extreme inputs (-999, 999, None, string numbers).
"""

from __future__ import annotations

import csv
import io
import sys
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.export_service import DemographicMappings, ExportService
from src.persona_engine import PersonaEngine, RespondentProfile
from src.stats_analyzer import StatsAnalyzer


class TestExportMathematicalInvariants(unittest.TestCase):
    """Adversarial mathematical and psychometric invariant verification."""

    def test_pc_correlation_across_arbitrary_sample_sizes(self):
        """Verify Pearson correlation r(PC_raw, PC_rev) == -1.0 exactly across sample sizes 2 to 1000."""
        sample_sizes = [2, 3, 5, 10, 25, 50, 100, 250, 500, 1000]
        for n in sample_sizes:
            engine = PersonaEngine(seed=n * 13)
            profiles = [engine.generate_respondent() for _ in range(n)]
            df = ExportService.to_numeric_matrix(profiles)

            for i in (1, 2, 3):
                raw = df[f"PC{i}"].to_numpy(dtype=float)
                rev = df[f"PC{i}_rev"].to_numpy(dtype=float)

                # Linear identity
                np.testing.assert_array_equal(
                    raw + rev, 7,
                    err_msg=f"Identity PC{i}_raw + PC{i}_rev == 7 violated for N={n}"
                )

                # Pearson correlation (if variance > 0)
                if np.std(raw) > 0:
                    r = np.corrcoef(raw, rev)[0, 1]
                    self.assertAlmostEqual(
                        r, -1.0, places=10,
                        msg=f"Pearson r != -1.0 for N={n}, item PC{i}: {r}"
                    )

    def test_pc_covariance_equals_negative_variance(self):
        """Verify Cov(PC_raw, PC_rev) == -Var(PC_raw) by algebraic properties of linear transformation."""
        engine = PersonaEngine(seed=888)
        profiles = [engine.generate_respondent() for _ in range(200)]
        df = ExportService.to_numeric_matrix(profiles)

        for i in (1, 2, 3):
            raw = df[f"PC{i}"].to_numpy(dtype=float)
            rev = df[f"PC{i}_rev"].to_numpy(dtype=float)
            cov_mat = np.cov(raw, rev)
            cov_val = cov_mat[0, 1]
            var_raw = np.var(raw, ddof=1)
            var_rev = np.var(rev, ddof=1)

            self.assertAlmostEqual(var_raw, var_rev, places=10)
            self.assertAlmostEqual(cov_val, -var_raw, places=10)

    def test_all_39_items_strictly_in_likert_range_under_fuzz(self):
        """Stress-test clamping of all 39 Likert items against out-of-range and non-integer inputs."""
        fuzz_cases = [
            -9999, -1, 0, 1, 3, 6, 7, 100, 99999,
            1.0001, 2.4999, 3.5, 4.9999, 5.8,
            "1", "3", "6", "5.2",
            None,
        ]
        records = []
        for val in fuzz_cases:
            records.append({
                "likert_scores": {it: val for it in ExportService.ALL_39_ITEMS}
            })

        df = ExportService.to_numeric_matrix(records)
        self.assertEqual(len(df), len(fuzz_cases))

        for it in ExportService.ALL_39_ITEMS:
            vals = df[it].values
            self.assertTrue(
                np.all(vals >= 1),
                f"Item {it} contains values < 1: {vals[vals < 1]}"
            )
            self.assertTrue(
                np.all(vals <= 6),
                f"Item {it} contains values > 6: {vals[vals > 6]}"
            )
            self.assertTrue(
                np.all(vals.astype(int) == vals),
                f"Item {it} contains non-integer values: {vals}"
            )

    def test_composite_means_arithmetic_exactness(self):
        """Verify composite construct means match exact arithmetic average of indicator items."""
        engine = PersonaEngine(seed=555)
        profiles = [engine.generate_respondent() for _ in range(50)]
        df = ExportService.to_numeric_matrix(profiles)

        for c_code, items in ExportService.CONSTRUCT_INDICATORS.items():
            if c_code == "PC":
                mean_col = "MEAN_PC_rev"
                expected_means = df[[f"{it}_rev" for it in items]].mean(axis=1).round(4)
            else:
                mean_col = f"MEAN_{c_code}"
                expected_means = df[list(items)].mean(axis=1).round(4)

            self.assertIn(mean_col, df.columns)
            np.testing.assert_allclose(
                df[mean_col].values,
                expected_means.values,
                atol=1e-4,
                err_msg=f"Composite mean mismatch for {c_code}"
            )
            self.assertTrue(np.all(df[mean_col].values >= 1.0))
            self.assertTrue(np.all(df[mean_col].values <= 6.0))


class TestExportFormatRigor(unittest.TestCase):
    """Rigorous standard library parsing of generated CSV and Excel artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.engine = PersonaEngine(seed=777)
        cls.profiles = [cls.engine.generate_respondent() for _ in range(80)]
        cls.analyzer = StatsAnalyzer()
        for p in cls.profiles:
            cls.analyzer.add_sample(p.likert_scores)
        cls.summary = cls.analyzer.get_summary()

    def test_csv_rfc4180_and_encoding_rigor(self):
        """Verify CSV encoding (UTF-8 BOM), column count (73), cell counts, and absence of NaN."""
        for delimiter in (",", ";", "\t"):
            csv_bytes = ExportService.export_spss_csv(self.profiles, delimiter=delimiter)
            # UTF-8 BOM verification
            self.assertTrue(
                csv_bytes.startswith(b"\xef\xbb\xbf"),
                f"CSV output missing UTF-8 BOM for delimiter '{delimiter}'"
            )
            csv_text = csv_bytes.decode("utf-8-sig")

            reader = csv.reader(io.StringIO(csv_text), delimiter=delimiter)
            rows = list(reader)

            self.assertEqual(len(rows), 81, "Expected 1 header row + 80 data rows")
            header = rows[0]
            self.assertEqual(len(header), 72, f"Expected exactly 72 columns, got {len(header)}")

            for row_idx, row in enumerate(rows[1:], start=1):
                self.assertEqual(
                    len(row), 72,
                    f"Row {row_idx} length mismatch: got {len(row)}, expected 72"
                )
                for col_idx, cell_str in enumerate(row):
                    col_name = header[col_idx]
                    self.assertNotEqual(
                        cell_str, "",
                        f"Empty cell found at row {row_idx}, column '{col_name}'"
                    )
                    self.assertNotIn(
                        cell_str.lower(), {"nan", "null", "none"},
                        f"NaN/null string found at row {row_idx}, column '{col_name}'"
                    )
                    # Verify parseable as numeric
                    try:
                        float(cell_str)
                    except ValueError:
                        self.fail(f"Non-numeric string '{cell_str}' at row {row_idx}, column '{col_name}'")

    def test_excel_zipfile_opc_xml_structure(self):
        """Verify complete Open Packaging Convention (OPC) XML structure in generated .xlsx."""
        excel_bytes = ExportService.export_excel(self.profiles, stats_summary=self.summary)
        self.assertIsInstance(excel_bytes, bytes)

        with zipfile.ZipFile(io.BytesIO(excel_bytes)) as zf:
            namelist = zf.namelist()

            # Required OPC files
            required_members = [
                "[Content_Types].xml",
                "_rels/.rels",
                "xl/_rels/workbook.xml.rels",
                "xl/workbook.xml",
                "xl/styles.xml",
                "xl/sharedStrings.xml",
                "xl/worksheets/sheet1.xml",
                "xl/worksheets/sheet2.xml",
                "xl/worksheets/sheet3.xml",
            ]
            for member in required_members:
                self.assertIn(member, namelist, f"Missing OPC member '{member}' in .xlsx ZIP")

            # Verify workbook.xml sheets
            wb_xml = zf.read("xl/workbook.xml")
            wb_root = ET.fromstring(wb_xml)
            sheet_elements = [el for el in wb_root.iter() if el.tag.endswith("sheet")]
            sheet_names = [el.attrib.get("name") for el in sheet_elements]
            self.assertEqual(
                sheet_names,
                ["Data", "Variable Codebook", "Psychometric Summary"],
                "Sheets in xl/workbook.xml do not match specification"
            )

            # Verify Sheet 1 Data rows are purely numeric cell types
            s1_xml = zf.read("xl/worksheets/sheet1.xml")
            s1_root = ET.fromstring(s1_xml)
            ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            rows = s1_root.findall(".//ns:row", ns)
            self.assertEqual(len(rows), 81, "Sheet 1 should contain 1 header + 80 data rows")

            # Check header row has 72 columns
            header_cells = rows[0].findall("ns:c", ns)
            self.assertEqual(len(header_cells), 72, "Sheet 1 header cell count mismatch")

            # Check data rows cells are numeric ('n' or absent t attribute)
            for r in rows[1:]:
                cells = r.findall("ns:c", ns)
                for c in cells:
                    t_attr = c.attrib.get("t", "n")
                    self.assertIn(
                        t_attr, ("n", ""),
                        f"Non-numeric cell type '{t_attr}' found in Sheet 1: {c.attrib}"
                    )

    def test_codebook_one_to_one_data_columns_parity(self):
        """Verify Variable Codebook contains exactly 1 row for every Data column."""
        df_data = ExportService.to_numeric_matrix(self.profiles)
        df_cb = ExportService.get_variable_codebook()

        self.assertEqual(len(df_data.columns), 72)
        self.assertEqual(len(df_cb), 72)

        data_cols = list(df_data.columns)
        codebook_vars = list(df_cb["Variable Name"].values)
        self.assertEqual(data_cols, codebook_vars, "Column order in Data does not match Codebook")


class TestExportBoundaryAndFuzzingResilience(unittest.TestCase):
    """Stress tests across empty, single-sample, corrupt, and incomplete inputs."""

    def test_empty_dataset_across_all_export_formats(self):
        """Verify empty records input produces valid empty outputs without unhandled exceptions."""
        df_empty = ExportService.to_numeric_matrix([])
        self.assertEqual(df_empty.shape, (0, 72))

        csv_empty = ExportService.export_spss_csv([])
        self.assertTrue(csv_empty.startswith(b"\xef\xbb\xbf"))
        lines = csv_empty.decode("utf-8-sig").strip().splitlines()
        self.assertEqual(len(lines), 1, "Empty CSV should contain only 1 line (headers)")
        self.assertEqual(len(lines[0].split(",")), 72)

        excel_empty = ExportService.export_excel([])
        with zipfile.ZipFile(io.BytesIO(excel_empty)) as zf:
            self.assertIn("xl/workbook.xml", zf.namelist())
            self.assertIn("xl/worksheets/sheet1.xml", zf.namelist())

    def test_single_respondent_degrees_of_freedom_edge_case(self):
        """Verify single respondent (N=1) handles psychometrics without zero-division crashes."""
        p = PersonaEngine(seed=101).generate_respondent()

        df_single = ExportService.to_numeric_matrix([p])
        self.assertEqual(df_single.shape, (1, 72))
        self.assertEqual(df_single["Respondent_ID"].iloc[0], 1)

        csv_single = ExportService.export_spss_csv([p])
        rows = list(csv.reader(io.StringIO(csv_single.decode("utf-8-sig"))))
        self.assertEqual(len(rows), 2)

        excel_single = ExportService.export_excel([p])
        self.assertGreater(len(excel_single), 5000)

        # Psychometric summary with N=1 records
        c_sum, it_sum = ExportService.get_psychometric_summary(records=[p])
        self.assertEqual(len(c_sum), 12)
        self.assertEqual(len(it_sum), 38)

    def test_incomplete_and_malformed_respondents_graceful_handling(self):
        """Verify robust fallbacks when records have missing fields, alien types, or partial keys."""
        class MockBareObject:
            pass

        class MockCorruptObject:
            age = "not_an_age"
            gender = "alien_gender"
            location = None
            education = 999
            employment = "invalid"
            profession = None
            frequency = None
            awareness = None
            categories = None
            pain_points = "not_a_list"
            likert_scores = {"PU1": 5, "PC1": 2}

        records: List[Any] = [
            {},  # completely empty dict
            {"age": None, "likert_scores": {}},  # null demographics
            {"age": "twenty-two", "education": "high_school"},  # non-numeric strings
            {"categories": None, "pain_points": None},  # null lists
            {"likert_scores": None},  # None likert_scores
            {"likert_scores": {"PC1": 1}},  # only 1 likert item provided
            MockBareObject(),
            MockCorruptObject(),
        ]

        df = ExportService.to_numeric_matrix(records)
        self.assertEqual(df.shape, (len(records), 72))
        self.assertFalse(df.isna().any().any(), "Fuzzed matrix contains unexpected NaN values")

        # Verify all cells are numeric
        for col in df.columns:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(df[col]),
                f"Column '{col}' is not numeric"
            )

    def test_high_volume_batch_scale_n2000(self):
        """Stress-test N=2,000 maximum survey quota execution for memory and data integrity."""
        engine = PersonaEngine(seed=2000)
        profiles = [engine.generate_respondent() for _ in range(2000)]

        df = ExportService.to_numeric_matrix(profiles)
        self.assertEqual(df.shape, (2000, 72))
        self.assertEqual(df["Respondent_ID"].iloc[-1], 2000)
        self.assertFalse(df.isna().any().any())

        csv_bytes = ExportService.export_spss_csv(profiles)
        self.assertGreater(len(csv_bytes), 300_000)


if __name__ == "__main__":
    unittest.main()
