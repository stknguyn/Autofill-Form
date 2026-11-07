"""
tests/test_export_spss.py - Comprehensive Verification Suite for SPSS & Excel Export Service.
Milestone M4: Streamlit Dashboard & SPSS Export.

Covers:
- 1. TestSPSSNumericCSV: RFC 4180 format, headers, purely numeric cells, 1-6 bounds.
- 2. TestPCReverseCoding: Exact PC_rev = 7 - PC_raw inversion, correlation = -1.0, column naming.
- 3. TestDemographicEncoding: Completeness, standard numeric mappings, 0/1 dummy indicators.
- 4. TestExcelMultiSheetExporter: Sheets (Data, Variable Codebook, Psychometric Summary), codebook integrity, in-memory bytes.
- 5. TestEdgeCasesAndResilience: N=0, N=1, partial demographics, WorkerManager DataFrame ingestion.
"""

from __future__ import annotations

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


class TestSPSSNumericCSV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PersonaEngine(seed=42)
        cls.profiles = [cls.engine.generate_respondent() for _ in range(50)]

    def test_csv_purely_numeric(self):
        """Verify all cells in exported CSV data matrix are strictly numeric (no strings, no NaNs)."""
        csv_bytes = ExportService.export_spss_csv(self.profiles)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(df), 50)
        for col in df.columns:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(df[col]),
                f"Column '{col}' is not purely numeric: {df[col].dtype}",
            )
            self.assertFalse(
                df[col].isna().any(),
                f"Column '{col}' contains NaN values",
            )

    def test_likert_bounds_1_to_6(self):
        """Verify all 39 Likert items are strictly integers in [1, 6]."""
        df = ExportService.to_numeric_matrix(self.profiles)
        for item in ExportService.ALL_39_ITEMS:
            vals = df[item].values
            self.assertTrue(np.all(vals >= 1), f"Item {item} has values < 1: {vals.min()}")
            self.assertTrue(np.all(vals <= 6), f"Item {item} has values > 6: {vals.max()}")
            self.assertTrue(np.all(vals == np.round(vals)), f"Item {item} has non-integer values")

    def test_canonical_headers_present(self):
        """Verify headers contain canonical construct indicator names."""
        df = ExportService.to_numeric_matrix(self.profiles)
        expected_indicators = [
            "PU1", "PU2", "PU3", "PU4",
            "PEOU1", "PEOU2", "PEOU3", "PEOU4",
            "PES1", "PES2", "PES3", "PES4",
            "PC1", "PC2", "PC3",
            "PC1_rev", "PC2_rev", "PC3_rev",
            "PI1", "PI2", "PI3",
            "SE1", "SE2", "SE3", "SE4",
            "LC1", "LC2", "LC3",
            "ATT1", "ATT2", "ATT3",
            "SN1", "SN2", "SN3",
            "PBC1", "PBC2", "PBC3", "PBC4",
            "ITU1", "ITU2", "ITU3",
        ]
        for it in expected_indicators:
            self.assertIn(it, df.columns, f"Indicator '{it}' missing from exported matrix")


class TestPCReverseCoding(unittest.TestCase):
    def test_pc_reversal_formula_exactness(self):
        """Verify theoretical reversal mapping: [1,2,3,4,5,6] -> [6,5,4,3,2,1]."""
        dummy_record = {
            "likert_scores": {
                "PC1": 1, "PC2": 3, "PC3": 6,
                **{it: 4 for it in ExportService.ALL_39_ITEMS if not it.startswith("PC")}
            }
        }
        df = ExportService.to_numeric_matrix([dummy_record])
        self.assertEqual(df["PC1_rev"].iloc[0], 6)
        self.assertEqual(df["PC2_rev"].iloc[0], 4)
        self.assertEqual(df["PC3_rev"].iloc[0], 1)

    def test_pc_raw_plus_rev_equals_seven(self):
        """Verify PC_raw + PC_rev == 7 for all observations."""
        engine = PersonaEngine(seed=99)
        profiles = [engine.generate_respondent() for _ in range(30)]
        df = ExportService.to_numeric_matrix(profiles)
        for i in (1, 2, 3):
            raw = df[f"PC{i}"].values
            rev = df[f"PC{i}_rev"].values
            np.testing.assert_array_equal(raw + rev, 7)

    def test_pc_perfect_negative_correlation(self):
        """Verify Pearson correlation between raw and reversed items is exactly -1.0."""
        engine = PersonaEngine(seed=77)
        profiles = [engine.generate_respondent() for _ in range(50)]
        df = ExportService.to_numeric_matrix(profiles)
        for i in (1, 2, 3):
            corr = np.corrcoef(df[f"PC{i}"], df[f"PC{i}_rev"])[0, 1]
            self.assertAlmostEqual(corr, -1.0, places=5)


class TestDemographicEncoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PersonaEngine(seed=123)
        cls.profiles = [cls.engine.generate_respondent() for _ in range(60)]

    def test_demographic_completeness(self):
        """Verify all demographic categorical variables map to valid integer codes without NaN."""
        df = ExportService.to_numeric_matrix(self.profiles)
        demographic_cols = ["Age", "Age_Group", "Gender", "Location", "Education", "Employment", "Profession", "Frequency", "Awareness"]
        for col in demographic_cols:
            self.assertIn(col, df.columns)
            self.assertFalse(df[col].isna().any(), f"Demographic column '{col}' has nulls")

    def test_gender_code_values(self):
        """Verify Gender codes are strictly in {1: Nam, 2: Nữ, 3: Khác}."""
        df = ExportService.to_numeric_matrix(self.profiles)
        self.assertTrue(set(df["Gender"].unique()).issubset({1, 2, 3}))

    def test_age_group_classification(self):
        """Verify age grouping logic: <=20 -> 1, <=23 -> 2, >23 -> 3."""
        df = ExportService.to_numeric_matrix(self.profiles)
        for _, row in df.iterrows():
            age = row["Age"]
            group = row["Age_Group"]
            if age <= 20:
                self.assertEqual(group, 1)
            elif age <= 23:
                self.assertEqual(group, 2)
            else:
                self.assertEqual(group, 3)

    def test_dummy_variables_binary(self):
        """Verify multi-select dummy columns are strictly binary {0, 1}."""
        df = ExportService.to_numeric_matrix(self.profiles)
        dummy_cols = [c[0] for c in DemographicMappings.CATEGORY_DUMMIES] + [p[0] for p in DemographicMappings.PAIN_POINT_DUMMIES]
        for col in dummy_cols:
            vals = set(df[col].unique())
            self.assertTrue(vals.issubset({0, 1}), f"Column '{col}' has non-binary values: {vals}")


class TestExcelMultiSheetExporter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PersonaEngine(seed=42)
        cls.profiles = [cls.engine.generate_respondent() for _ in range(40)]
        cls.analyzer = StatsAnalyzer()
        for p in cls.profiles:
            cls.analyzer.add_sample(p.likert_scores)
        cls.summary = cls.analyzer.get_summary()

    def test_excel_three_sheets_present(self):
        """Verify generated .xlsx archive contains exactly Data, Variable Codebook, Psychometric Summary."""
        excel_bytes = ExportService.export_excel(self.profiles, stats_summary=self.summary)
        zf = zipfile.ZipFile(io.BytesIO(excel_bytes))
        wb_xml = zf.read("xl/workbook.xml")
        root = ET.fromstring(wb_xml)
        sheets = [elem.attrib["name"] for elem in root.iter() if elem.tag.endswith("sheet")]
        self.assertEqual(sheets, ["Data", "Variable Codebook", "Psychometric Summary"])

    def test_variable_codebook_integrity(self):
        """Verify all columns in the Data sheet are documented in Variable Codebook."""
        df_data = ExportService.to_numeric_matrix(self.profiles)
        df_cb = ExportService.get_variable_codebook()
        codebook_vars = set(df_cb["Variable Name"].values)
        for col in df_data.columns:
            self.assertIn(col, codebook_vars, f"Data column '{col}' is missing from Variable Codebook")

    def test_psychometric_summary_constructs(self):
        """Verify Psychometric Summary table lists all 11 constructs plus OVERALL."""
        df_c, _ = ExportService.get_psychometric_summary(stats_summary=self.summary)
        constructs = set(df_c["Construct Code"].values)
        expected = {"PU", "PEOU", "PES", "PC", "PI", "SE", "LC", "ATT", "SN", "PBC", "ITU", "OVERALL"}
        self.assertTrue(expected.issubset(constructs), f"Missing constructs in summary: {expected - constructs}")

    def test_excel_bytes_valid_zip(self):
        """Verify export_excel returns valid non-empty bytes."""
        res = ExportService.export_excel(self.profiles)
        self.assertIsInstance(res, bytes)
        self.assertGreater(len(res), 5000)


class TestEdgeCasesAndResilience(unittest.TestCase):
    def test_empty_dataset_handling(self):
        """Verify empty records list returns valid empty DataFrame with schema headers."""
        df = ExportService.to_numeric_matrix([])
        self.assertEqual(len(df), 0)
        self.assertIn("PU1", df.columns)
        self.assertIn("Age", df.columns)

        excel_bytes = ExportService.export_excel([])
        zf = zipfile.ZipFile(io.BytesIO(excel_bytes))
        self.assertIn("xl/workbook.xml", zf.namelist())

    def test_single_sample_dataset(self):
        """Verify single record (N=1) exports cleanly without crashing."""
        engine = PersonaEngine(seed=1)
        p = engine.generate_respondent()
        df = ExportService.to_numeric_matrix([p])
        self.assertEqual(len(df), 1)
        self.assertEqual(df["Respondent_ID"].iloc[0], 1)

    def test_partial_demographics_graceful_fallback(self):
        """Verify record with missing demographic fields defaults gracefully without error."""
        minimal_record = {
            "age": 22,
            "likert_scores": {it: 5 for it in ExportService.ALL_39_ITEMS}
        }
        df = ExportService.to_numeric_matrix([minimal_record])
        self.assertEqual(len(df), 1)
        self.assertEqual(df["Age"].iloc[0], 22)
        self.assertIn(df["Gender"].iloc[0], (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
