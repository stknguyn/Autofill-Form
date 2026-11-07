"""
src/export_service.py - Authoritative SPSS & Multi-Sheet Excel Export Service.
Milestone M4: Streamlit Dashboard & SPSS Export.

Implements:
- Pure numeric matrix transformation for IBM SPSS Statistics, SmartPLS 3/4, and AMOS.
- 39 Likert items bounded strictly in [1, 6].
- Reverse-coded Perceived Costs (PC_rev = 7 - PC_raw) with transparent column naming.
- Canonical demographic integer mapping and multi-select binary dummy variables.
- Multi-sheet Excel (.xlsx) exporter with Data, Variable Codebook, and Psychometric Summary sheets.
- Native xlsxwriter styling engine with freeze panes, column auto-fitting, and alpha highlighting.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.persona_engine import (
    OPT_AWARE_NEVER,
    OPT_AWARE_OTHER,
    OPT_AWARE_SEEN_UNUSED,
    OPT_AWARE_USED_ONCE,
    OPT_CAT_BEAUTY,
    OPT_CAT_BOOKS,
    OPT_CAT_FASHION,
    OPT_CAT_HOUSEHOLD,
    OPT_CAT_OTHER,
    OPT_CAT_TECH,
    OPT_EDU_DAI_HOC,
    OPT_EDU_KHAC,
    OPT_EDU_SAU_DAI_HOC,
    OPT_EDU_THPT,
    OPT_EDU_TRUNG_CAP,
    OPT_EMP_FREELANCER,
    OPT_EMP_FULL_TIME,
    OPT_EMP_KHAC,
    OPT_EMP_KHONG_LAO_DONG,
    OPT_EMP_PART_TIME,
    OPT_EMP_TIM_VIEC,
    OPT_FREQ_2_TO_4,
    OPT_FREQ_4_TO_6,
    OPT_FREQ_6_TO_8,
    OPT_FREQ_OVER_8,
    OPT_FREQ_UNDER_2,
    OPT_GENDER_KHAC,
    OPT_GENDER_NAM,
    OPT_GENDER_NU,
    OPT_LOC_DANANG,
    OPT_LOC_HANOI,
    OPT_LOC_HCM,
    OPT_LOC_KHAC,
    OPT_PAIN_BUILDING,
    OPT_PAIN_NONE,
    OPT_PAIN_OTHER,
    OPT_PAIN_PRIVACY_LOSS,
    OPT_PROF_CNTT,
    OPT_PROF_FREELANCE_CONTENT,
    OPT_PROF_GIAO_DUC,
    OPT_PROF_KINH_DOANH,
    OPT_PROF_KHAC,
    OPT_PROF_SAN_XUAT,
    OPT_PROF_VIEN_CHUC,
    OPT_PROF_VU_TRANG,
    RespondentProfile,
)
from src.stats_analyzer import StatsAnalyzer, StatsSummary
from src.tally_schema import TallySchema


# ---------------------------------------------------------------------------
# Codebook Item Definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SPSSCodebookItem:
    name: str
    label: str
    construct: str
    measurement_level: str  # "Scale", "Nominal", "Ordinal"
    data_type: str          # "Numeric", "Binary", "Integer", "Float"
    value_labels: str
    notes: str


# ---------------------------------------------------------------------------
# Demographic Mapping Registry
# ---------------------------------------------------------------------------
class DemographicMappings:
    GENDER_MAP: Dict[Any, int] = {
        OPT_GENDER_NAM: 1, "Nam": 1, "nam": 1, 1: 1,
        OPT_GENDER_NU: 2, "Nữ": 2, "Nu": 2, "nu": 2, 2: 2,
        OPT_GENDER_KHAC: 3, "Khác": 3, "Khac": 3, "khac": 3, 3: 3,
    }

    LOCATION_MAP: Dict[Any, int] = {
        OPT_LOC_HANOI: 1, "Thành phố Hà Nội": 1, "Hà Nội": 1, "Hanoi": 1, 1: 1,
        OPT_LOC_DANANG: 2, "Thành phố Đà Nẵng": 2, "Đà Nẵng": 2, "Danang": 2, 2: 2,
        OPT_LOC_HCM: 3, "Thành phố Hồ Chí Minh": 3, "TP. Hồ Chí Minh": 3, "TP.HCM": 3, "HCM": 3, 3: 3,
        OPT_LOC_KHAC: 4, "Tỉnh / Thành phố khác": 4, "Khác": 4, 4: 4,
    }

    EDUCATION_MAP: Dict[Any, int] = {
        OPT_EDU_THPT: 1, "Từ THPT trở xuống": 1, 1: 1,
        OPT_EDU_TRUNG_CAP: 2, "Trung cấp / Đào tạo nghề": 2, 2: 2,
        OPT_EDU_DAI_HOC: 3, "Cao đẳng / Đại học": 3, 3: 3,
        OPT_EDU_SAU_DAI_HOC: 4, "Sau Đại học": 4, 4: 4,
        OPT_EDU_KHAC: 5, "Khác": 5, 5: 5,
    }

    EMPLOYMENT_MAP: Dict[Any, int] = {
        OPT_EMP_PART_TIME: 1, "Đang làm việc bán thời gian (Part-time)": 1, 1: 1,
        OPT_EMP_FULL_TIME: 2, "Đang làm việc toàn thời gian (Full-time)": 2, 2: 2,
        OPT_EMP_FREELANCER: 3, "Làm việc tự do (Freelancer)": 3, 3: 3,
        OPT_EMP_TIM_VIEC: 4, "Đang tìm kiếm việc làm": 4, 4: 4,
        OPT_EMP_KHONG_LAO_DONG: 5, "Không trong lực lượng lao động": 5, 5: 5,
        OPT_EMP_KHAC: 6, "Khác": 6, 6: 6,
    }

    PROFESSION_MAP: Dict[Any, int] = {
        OPT_PROF_KINH_DOANH: 1, "Kinh doanh / Thương mại / Marketing / Dịch vụ": 1, 1: 1,
        OPT_PROF_CNTT: 2, "Kỹ thuật / Công nghệ thông tin": 2, 2: 2,
        OPT_PROF_SAN_XUAT: 3, "Sản xuất": 3, 3: 3,
        OPT_PROF_GIAO_DUC: 4, "Giáo dục": 4, 4: 4,
        OPT_PROF_VIEN_CHUC: 5, "Viên chức nhà nước": 5, 5: 5,
        OPT_PROF_VU_TRANG: 6, "Lực lượng vũ trang (Quân đội / Công an)": 6, 6: 6,
        OPT_PROF_FREELANCE_CONTENT: 7, "Lao động tự do / Sáng tạo nội dung": 7, 7: 7,
        OPT_PROF_KHAC: 8, "Lĩnh vực khác": 8, 8: 8,
    }

    FREQUENCY_MAP: Dict[Any, int] = {
        OPT_FREQ_UNDER_2: 1, "Dưới 2 đơn hàng/tháng": 1, 1: 1,
        OPT_FREQ_2_TO_4: 2, "Từ 2 – 4 đơn hàng/tháng": 2, 2: 2,
        OPT_FREQ_4_TO_6: 3, "Từ 4 – 6 đơn hàng/tháng": 3, 3: 3,
        OPT_FREQ_6_TO_8: 4, "Từ 6 – 8 đơn hàng/tháng": 4, 4: 4,
        OPT_FREQ_OVER_8: 5, "Trên 8 đơn hàng/tháng": 5, 5: 5,
    }

    AWARENESS_MAP: Dict[Any, int] = {
        OPT_AWARE_NEVER: 1, "Chưa từng nghe đến trước đây": 1, 1: 1,
        OPT_AWARE_SEEN_UNUSED: 2, "Đã từng nghe nói hoặc nhìn thấy nhưng chưa từng dùng": 2, 2: 2,
        OPT_AWARE_USED_ONCE: 3, "Đã từng sử dụng ít nhất 1 lần": 3, 3: 3,
        OPT_AWARE_OTHER: 4, "Khác": 4, 4: 4,
    }

    CATEGORY_DUMMIES: List[Tuple[str, str, str]] = [
        ("CAT_Fashion", OPT_CAT_FASHION, "Thời trang, quần áo và phụ kiện"),
        ("CAT_Beauty", OPT_CAT_BEAUTY, "Mỹ phẩm, chăm sóc sức khỏe và làm đẹp"),
        ("CAT_Tech", OPT_CAT_TECH, "Thiết bị điện tử và phụ kiện công nghệ"),
        ("CAT_Books", OPT_CAT_BOOKS, "Sách vở, văn phòng phẩm"),
        ("CAT_Household", OPT_CAT_HOUSEHOLD, "Đồ gia dụng, thực phẩm bách hóa"),
        ("CAT_Other", OPT_CAT_OTHER, "Mặt hàng khác"),
    ]

    PAIN_POINT_DUMMIES: List[Tuple[str, str, str]] = [
        ("PAIN_Building", OPT_PAIN_BUILDING, "Chung cư/KTX/Văn phòng không cho gửi ở lễ tân/bảo vệ"),
        ("PAIN_Privacy", OPT_PAIN_PRIVACY_LOSS, "Lo ngại lộ thông tin hoặc sợ mất hàng"),
        ("PAIN_None", OPT_PAIN_NONE, "Không gặp bất tiện nào"),
        ("PAIN_Other", OPT_PAIN_OTHER, "Bất tiện khác"),
    ]


# ---------------------------------------------------------------------------
# Export Service Core Class
# ---------------------------------------------------------------------------
class ExportService:
    """
    Production-grade SPSS & Excel Data Exporter.
    Generates research-ready pure numeric matrices and multi-sheet workbooks.
    """

    CONSTRUCT_INDICATORS: Dict[str, Tuple[str, ...]] = {
        "PU": ("PU1", "PU2", "PU3", "PU4"),
        "PEOU": ("PEOU1", "PEOU2", "PEOU3", "PEOU4"),
        "PES": ("PES1", "PES2", "PES3", "PES4"),
        "PC": ("PC1", "PC2", "PC3"),
        "PI": ("PI1", "PI2", "PI3"),
        "SE": ("SE1", "SE2", "SE3", "SE4"),
        "LC": ("LC1", "LC2", "LC3"),
        "ATT": ("ATT1", "ATT2", "ATT3"),
        "SN": ("SN1", "SN2", "SN3"),
        "PBC": ("PBC1", "PBC2", "PBC3", "PBC4"),
        "ITU": ("ITU1", "ITU2", "ITU3"),
    }

    ALL_38_ITEMS: Tuple[str, ...] = (
        "PU1", "PU2", "PU3", "PU4",
        "PEOU1", "PEOU2", "PEOU3", "PEOU4",
        "PES1", "PES2", "PES3", "PES4",
        "PC1", "PC2", "PC3",
        "PI1", "PI2", "PI3",
        "SE1", "SE2", "SE3", "SE4",
        "LC1", "LC2", "LC3",
        "ATT1", "ATT2", "ATT3",
        "SN1", "SN2", "SN3",
        "PBC1", "PBC2", "PBC3", "PBC4",
        "ITU1", "ITU2", "ITU3",
    )
    ALL_39_ITEMS = ALL_38_ITEMS

    @classmethod
    def to_numeric_matrix(
        cls,
        records: Union[Sequence[RespondentProfile], Sequence[Dict[str, Any]], pd.DataFrame],
        include_pc_rev: bool = True,
        include_raw_pc: bool = True,
        include_construct_means: bool = True,
    ) -> pd.DataFrame:
        """
        Converts raw submission records into a strictly numeric pandas DataFrame.
        Guarantees:
        - 1-6 integer encoding for Likert items.
        - Exact PC reversal (PC_rev = 7 - PC_raw).
        - Integer demographic categories & binary dummy indicators.
        """
        if isinstance(records, pd.DataFrame):
            records_list = records.to_dict(orient="records")
        else:
            records_list = list(records)

        if not records_list:
            return cls._build_empty_dataframe(
                include_pc_rev=include_pc_rev,
                include_raw_pc=include_raw_pc,
                include_construct_means=include_construct_means,
            )

        rows: List[Dict[str, Any]] = []
        for idx, rec in enumerate(records_list, start=1):
            row: Dict[str, Any] = {"Respondent_ID": idx}

            # Extract Likert scores & Demographics
            likert_scores: Dict[str, Any] = {}
            if isinstance(rec, RespondentProfile):
                likert_scores = rec.likert_scores
                age = rec.age
                gender = rec.gender
                loc = rec.location
                edu = rec.education
                emp = rec.employment
                prof = rec.profession
                freq = rec.frequency
                aware = rec.awareness
                cats = rec.categories or []
                pains = rec.pain_points or []
            elif isinstance(rec, dict):
                likert_scores = rec.get("likert_scores", {})
                if not likert_scores:
                    # Direct key fallback
                    likert_scores = {k: rec[k] for k in cls.ALL_39_ITEMS if k in rec}
                age = rec.get("age", 21)
                gender = rec.get("gender", rec.get("gender_uuid"))
                loc = rec.get("location", rec.get("location_uuid"))
                edu = rec.get("education", rec.get("education_uuid"))
                emp = rec.get("employment", rec.get("employment_uuid"))
                prof = rec.get("profession", rec.get("profession_uuid"))
                freq = rec.get("frequency", rec.get("frequency_uuid"))
                aware = rec.get("awareness", rec.get("awareness_uuid"))
                cats = rec.get("categories", rec.get("categories_uuids", [])) or []
                pains = rec.get("pain_points", rec.get("pain_points_uuids", [])) or []
            else:
                age = getattr(rec, "age", 21)
                gender = getattr(rec, "gender", None)
                loc = getattr(rec, "location", None)
                edu = getattr(rec, "education", None)
                emp = getattr(rec, "employment", None)
                prof = getattr(rec, "profession", None)
                freq = getattr(rec, "frequency", None)
                aware = getattr(rec, "awareness", None)
                cats = getattr(rec, "categories", []) or []
                pains = getattr(rec, "pain_points", []) or []
                likert_scores = getattr(rec, "likert_scores", {})

            # 1. Demographics Encoding
            try:
                row["Age"] = int(age) if age is not None else 21
            except (ValueError, TypeError):
                row["Age"] = 21

            row["Age_Group"] = 1 if row["Age"] <= 20 else (2 if row["Age"] <= 23 else 3)
            row["Gender"] = DemographicMappings.GENDER_MAP.get(gender, 1)
            row["Location"] = DemographicMappings.LOCATION_MAP.get(loc, 3)
            row["Education"] = DemographicMappings.EDUCATION_MAP.get(edu, 3)
            row["Employment"] = DemographicMappings.EMPLOYMENT_MAP.get(emp, 2)
            row["Profession"] = DemographicMappings.PROFESSION_MAP.get(prof, 1)
            row["Frequency"] = DemographicMappings.FREQUENCY_MAP.get(freq, 2)
            row["Awareness"] = DemographicMappings.AWARENESS_MAP.get(aware, 2)

            # Category Dummies
            cats_set = set(cats) if isinstance(cats, (list, tuple, set)) else set()
            for col_name, uuid_val, _ in DemographicMappings.CATEGORY_DUMMIES:
                row[col_name] = 1 if (uuid_val in cats_set or col_name in cats_set) else 0

            # Pain Point Dummies
            pains_set = set(pains) if isinstance(pains, (list, tuple, set)) else set()
            for col_name, uuid_val, _ in DemographicMappings.PAIN_POINT_DUMMIES:
                row[col_name] = 1 if (uuid_val in pains_set or col_name in pains_set) else 0

            # 2. Likert Items (PU1 .. ITU4)
            for c_code, items in cls.CONSTRUCT_INDICATORS.items():
                for item in items:
                    raw_val = likert_scores.get(item, 4)
                    try:
                        clamped_val = int(np.clip(round(float(raw_val)), 1, 6))
                    except (ValueError, TypeError):
                        clamped_val = 4

                    if c_code == "PC":
                        if include_raw_pc:
                            row[item] = clamped_val
                        if include_pc_rev:
                            row[f"{item}_rev"] = 7 - clamped_val
                    else:
                        row[item] = clamped_val

            # 3. Composite Means
            if include_construct_means:
                for c_code, items in cls.CONSTRUCT_INDICATORS.items():
                    if c_code == "PC":
                        if include_pc_rev:
                            row["MEAN_PC_rev"] = round(float(np.mean([row[f"{it}_rev"] for it in items])), 4)
                        elif include_raw_pc:
                            row["MEAN_PC"] = round(float(np.mean([row[it] for it in items])), 4)
                    else:
                        row[f"MEAN_{c_code}"] = round(float(np.mean([row[it] for it in items])), 4)

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    @classmethod
    def _build_empty_dataframe(
        cls,
        include_pc_rev: bool = True,
        include_raw_pc: bool = True,
        include_construct_means: bool = True,
    ) -> pd.DataFrame:
        cols = [
            "Respondent_ID", "Age", "Age_Group", "Gender", "Location",
            "Education", "Employment", "Profession", "Frequency", "Awareness"
        ]
        cols.extend([c[0] for c in DemographicMappings.CATEGORY_DUMMIES])
        cols.extend([p[0] for p in DemographicMappings.PAIN_POINT_DUMMIES])
        for c_code, items in cls.CONSTRUCT_INDICATORS.items():
            if c_code == "PC":
                if include_raw_pc:
                    cols.extend(items)
                if include_pc_rev:
                    cols.extend([f"{it}_rev" for it in items])
            else:
                cols.extend(items)
        if include_construct_means:
            for c_code in cls.CONSTRUCT_INDICATORS:
                cols.append(f"MEAN_{c_code}_rev" if c_code == "PC" and include_pc_rev else f"MEAN_{c_code}")
        return pd.DataFrame(columns=cols)

    @classmethod
    def export_spss_csv(
        cls,
        records: Any,
        filepath: Optional[Union[str, Path]] = None,
        delimiter: str = ",",
        include_pc_rev: bool = True,
        include_raw_pc: bool = True,
        include_construct_means: bool = True,
        encoding: str = "utf-8-sig",
    ) -> Union[str, bytes]:
        """
        Exports records to an RFC 4180 pure numeric CSV.
        If filepath is None, returns bytes. If filepath is supplied, writes to disk.
        """
        df = cls.to_numeric_matrix(
            records=records,
            include_pc_rev=include_pc_rev,
            include_raw_pc=include_raw_pc,
            include_construct_means=include_construct_means,
        )

        csv_str = df.to_csv(index=False, sep=delimiter)
        csv_bytes = csv_str.encode(encoding)

        if filepath is not None:
            p = Path(filepath).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(csv_bytes)
            return str(p)

        return csv_bytes

    @classmethod
    def get_variable_codebook(
        cls,
        include_pc_rev: bool = True,
        include_raw_pc: bool = True,
        include_construct_means: bool = True,
    ) -> pd.DataFrame:
        """Constructs the comprehensive data dictionary matching the numeric matrix."""
        TallySchema.initialize()
        items: List[Dict[str, str]] = []

        def add_item(name: str, label: str, construct: str, m_level: str, d_type: str, vals: str, notes: str):
            items.append({
                "Variable Name": name,
                "Variable Label": label,
                "Construct": construct,
                "Measurement Level": m_level,
                "Data Type": d_type,
                "Value Labels": vals,
                "Notes": notes,
            })

        # Metadata & Demographics
        add_item("Respondent_ID", "Mã định danh duy nhất của đáp viên", "Metadata", "Nominal", "Integer", "None", "Khóa chính tự tăng (1..N)")
        add_item("Age", "Tuổi hiện tại của đáp viên (năm)", "Demographics", "Scale", "Integer", "None", "Khoảng tuổi Thế hệ Z (18 - 27 tuổi)")
        add_item("Age_Group", "Nhóm tuổi phân đoạn học thuật", "Demographics", "Ordinal", "Integer", "1 = 18-20 tuổi; 2 = 21-23 tuổi; 3 = 24-27 tuổi", "Phân đoạn giai đoạn học tập / công việc")
        add_item("Gender", "Giới tính", "Demographics", "Nominal", "Integer", "1 = Nam; 2 = Nữ; 3 = Khác", "Mã hóa danh nghĩa chuẩn SPSS")
        add_item("Location", "Khu vực sinh sống hiện tại", "Demographics", "Nominal", "Integer", "1 = Hà Nội; 2 = Đà Nẵng; 3 = TP. HCM; 4 = Khác", "Địa bàn khảo sát chính")
        add_item("Education", "Trình độ học vấn hiện tại", "Demographics", "Ordinal", "Integer", "1 = THPT trở xuống; 2 = Trung cấp/Nghề; 3 = Cao đẳng/Đại học; 4 = Sau Đại học; 5 = Khác", "Trình độ đào tạo")
        add_item("Employment", "Tình trạng việc làm hiện tại", "Demographics", "Nominal", "Integer", "1 = Part-time; 2 = Full-time; 3 = Freelancer; 4 = Đang tìm việc; 5 = Không tham gia LĐ; 6 = Khác", "Tình trạng lao động")
        add_item("Profession", "Lĩnh vực học tập hoặc nghề nghiệp chính", "Demographics", "Nominal", "Integer", "1 = Kinh doanh/Dịch vụ; 2 = CNTT/Kỹ thuật; 3 = Sản xuất; 4 = Giáo dục; 5 = Viên chức; 6 = Vũ trang; 7 = Tự do/Sáng tạo; 8 = Khác", "Nhóm ngành nghề")
        add_item("Frequency", "Tần suất mua hàng Shopee gần đây", "Demographics", "Ordinal", "Integer", "1 = Dưới 2 đơn/tháng; 2 = 2-4 đơn; 3 = 4-6 đơn; 4 = 6-8 đơn; 5 = Trên 8 đơn/tháng", "Thói quen mua sắm")
        add_item("Awareness", "Mức độ nhận biết Shopee Smart Locker", "Demographics", "Ordinal", "Integer", "1 = Chưa từng nghe; 2 = Đã nghe chưa từng dùng; 3 = Đã dùng ít nhất 1 lần; 4 = Khác", "Nhận biết dịch vụ")

        # Dummy sets
        for col, _, desc in DemographicMappings.CATEGORY_DUMMIES:
            add_item(col, f"Mặt hàng thường mua: {desc}", "Shopping Habits", "Nominal", "Binary", "0 = Không chọn; 1 = Có chọn", "Biến giả nhị phân (Multiple Response Set)")
        for col, _, desc in DemographicMappings.PAIN_POINT_DUMMIES:
            add_item(col, f"Bất tiện giao nhận: {desc}", "Delivery Friction", "Nominal", "Binary", "0 = Không chọn; 1 = Có chọn", "Biến giả nhị phân (Multiple Response Set)")

        # Likert items
        likert_values = "1 = Hoàn toàn không đồng ý; 2 = Không đồng ý; 3 = Hơi không đồng ý; 4 = Hơi đồng ý; 5 = Đồng ý; 6 = Hoàn toàn đồng ý"
        for c_code, items_list in cls.CONSTRUCT_INDICATORS.items():
            for it_code in items_list:
                q_def = TallySchema.CODE_MAP.get(it_code)
                q_title = q_def.title if q_def else f"Câu hỏi {it_code}"
                c_name = q_def.construct if q_def else c_code

                if c_code == "PC":
                    if include_raw_pc:
                        add_item(it_code, f"{it_code}: {q_title} (Gốc)", "Perceived Costs (Raw)", "Scale", "Integer", likert_values, "Mục hỏi gốc (Nghịch biến với ý định)")
                    if include_pc_rev:
                        add_item(f"{it_code}_rev", f"{it_code}_rev: {q_title} (Đã đảo chiều)", "Perceived Costs (Reversed)", "Scale", "Integer", "1 = Rất trở ngại (Chi phí rất cao); 6 = Hoàn toàn không trở ngại", "Đảo chiều: PC_rev = 7 - PC_raw")
                else:
                    add_item(it_code, f"{it_code}: {q_title}", c_name, "Scale", "Integer", likert_values, "Thang đo Likert 6 mức độ")

        # Composite means
        if include_construct_means:
            for c_code in cls.CONSTRUCT_INDICATORS:
                if c_code == "PC":
                    if include_pc_rev:
                        add_item("MEAN_PC_rev", "Điểm trung bình nhân tố Chi phí nhận thức (Đã đảo)", "Composite Mean", "Scale", "Float", "1.00 - 6.00", "Trung bình cộng các biến PC1_rev, PC2_rev, PC3_rev")
                    elif include_raw_pc:
                        add_item("MEAN_PC", "Điểm trung bình nhân tố Chi phí nhận thức (Gốc)", "Composite Mean", "Scale", "Float", "1.00 - 6.00", "Trung bình cộng các biến PC1, PC2, PC3")
                else:
                    add_item(f"MEAN_{c_code}", f"Điểm trung bình nhân tố {c_code}", "Composite Mean", "Scale", "Float", "1.00 - 6.00", f"Trung bình cộng các biến quan sát {c_code}")

        return pd.DataFrame(items)

    @classmethod
    def get_psychometric_summary(
        cls,
        stats_summary: Optional[Union[StatsSummary, Dict[str, Any]]] = None,
        records: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds (construct_summary_df, item_diagnostics_df) from StatsSummary or raw records.
        """
        if stats_summary is None and records is not None:
            analyzer = StatsAnalyzer()
            df = cls.to_numeric_matrix(records)
            for _, r in df.iterrows():
                analyzer.add_sample(r.to_dict())
            stats_summary = analyzer.get_summary()

        c_rows: List[Dict[str, Any]] = []
        it_rows: List[Dict[str, Any]] = []

        if stats_summary is not None:
            c_stats_dict = (
                stats_summary.construct_stats
                if isinstance(stats_summary, StatsSummary)
                else stats_summary.get("constructs", {})
            )
            overall_alpha = (
                stats_summary.overall_alpha
                if isinstance(stats_summary, StatsSummary)
                else stats_summary.get("overall_alpha", 0.0)
            )

            for code, cstat in c_stats_dict.items():
                is_obj = hasattr(cstat, "name")
                name = cstat.name if is_obj else cstat.get("name", code)
                k = cstat.item_count if is_obj else cstat.get("item_count", 0)
                mean = cstat.mean if is_obj else cstat.get("mean", 0.0)
                std = cstat.std if is_obj else cstat.get("std", 0.0)
                alpha = cstat.alpha if is_obj else cstat.get("alpha", 0.0)

                eval_str = "Xuất sắc (>= 0.90)" if alpha >= 0.90 else ("Đạt chuẩn NCKH (>= 0.82)" if alpha >= 0.82 else "Cần cải thiện (< 0.82)")
                c_rows.append({
                    "Construct Code": code,
                    "Construct Name": name,
                    "Items (k)": k,
                    "Mean": round(mean, 2),
                    "Std Dev": round(std, 2),
                    "Cronbach's Alpha": round(alpha, 4),
                    "Reliability Evaluation": eval_str,
                })

                # Item level diagnostics
                item_stats_dict = cstat.item_stats if is_obj else cstat.get("items", {})
                for icode, it_obj in item_stats_dict.items():
                    it_is_obj = hasattr(it_obj, "mean")
                    it_mean = it_obj.mean if it_is_obj else it_obj.get("mean", 0.0)
                    it_std = it_obj.std if it_is_obj else it_obj.get("std", 0.0)
                    citc = it_obj.citc if it_is_obj else it_obj.get("citc", 0.0)
                    a_del = it_obj.alpha_if_deleted if it_is_obj else it_obj.get("alpha_if_deleted", 0.0)

                    citc_eval = "Tốt (>= 0.30)" if citc >= 0.30 else "Cần xem xét (< 0.30)"
                    it_rows.append({
                        "Construct": code,
                        "Item Code": icode,
                        "Item Mean": round(it_mean, 2),
                        "Item Std": round(it_std, 2),
                        "Corrected Item-Total Correlation (CITC)": round(citc, 4),
                        "Alpha if Item Deleted": round(a_del, 4),
                        "CITC Status": citc_eval,
                    })

            # Model Overall row
            ov_eval = "Xuất sắc (>= 0.90)" if overall_alpha >= 0.90 else ("Đạt chuẩn (>= 0.82)" if overall_alpha >= 0.82 else "Chưa đạt")
            c_rows.append({
                "Construct Code": "OVERALL",
                "Construct Name": "Toàn bộ Mô hình TAM & TPB (38 biến)",
                "Items (k)": 38,
                "Mean": "-",
                "Std Dev": "-",
                "Cronbach's Alpha": round(overall_alpha, 4),
                "Reliability Evaluation": ov_eval,
            })

        return pd.DataFrame(c_rows), pd.DataFrame(it_rows)

    @classmethod
    def export_excel(
        cls,
        records: Any,
        stats_summary: Optional[Any] = None,
        filepath: Optional[Union[str, Path]] = None,
        include_pc_rev: bool = True,
        include_raw_pc: bool = True,
        include_construct_means: bool = True,
    ) -> Union[str, bytes]:
        """
        Exports a multi-sheet Excel workbook with Data, Variable Codebook, and Psychometric Summary.
        Applies xlsxwriter styling.
        """
        df_data = cls.to_numeric_matrix(
            records=records,
            include_pc_rev=include_pc_rev,
            include_raw_pc=include_raw_pc,
            include_construct_means=include_construct_means,
        )
        df_codebook = cls.get_variable_codebook(
            include_pc_rev=include_pc_rev,
            include_raw_pc=include_raw_pc,
            include_construct_means=include_construct_means,
        )
        df_c_summary, df_it_summary = cls.get_psychometric_summary(
            stats_summary=stats_summary,
            records=records,
        )

        output_buffer = io.BytesIO()
        with pd.ExcelWriter(output_buffer, engine="xlsxwriter") as writer:
            # 1. Sheet 1: Data
            df_data.to_excel(writer, sheet_name="Data", index=False)

            # 2. Sheet 2: Variable Codebook
            df_codebook.to_excel(writer, sheet_name="Variable Codebook", index=False)

            # 3. Sheet 3: Psychometric Summary
            start_row = 0
            df_c_summary.to_excel(writer, sheet_name="Psychometric Summary", startrow=start_row, index=False)
            if not df_it_summary.empty:
                start_row += len(df_c_summary) + 3
                df_it_summary.to_excel(writer, sheet_name="Psychometric Summary", startrow=start_row, index=False)

            # Formatting with XlsxWriter
            workbook = writer.book
            header_fmt = workbook.add_format({
                "bold": True,
                "font_name": "Calibri",
                "font_size": 11,
                "font_color": "#FFFFFF",
                "bg_color": "#1E293B",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            })
            green_fmt = workbook.add_format({
                "bg_color": "#DCFCE7",
                "font_color": "#166534",
                "bold": True,
            })

            # Format Sheet 1
            ws_data = writer.sheets["Data"]
            ws_data.freeze_panes(1, 0)
            for col_num, val in enumerate(df_data.columns.values):
                ws_data.write(0, col_num, val, header_fmt)
                max_w = max(len(str(val)), 8)
                ws_data.set_column(col_num, col_num, max_w + 3)

            # Format Sheet 2
            ws_cb = writer.sheets["Variable Codebook"]
            ws_cb.freeze_panes(1, 0)
            for col_num, val in enumerate(df_codebook.columns.values):
                ws_cb.write(0, col_num, val, header_fmt)
                max_w = min(max(len(str(val)), 12), 50)
                ws_cb.set_column(col_num, col_num, max_w + 4)

            # Format Sheet 3
            ws_sum = writer.sheets["Psychometric Summary"]
            ws_sum.freeze_panes(1, 0)
            if not df_c_summary.empty:
                for col_num, val in enumerate(df_c_summary.columns.values):
                    ws_sum.write(0, col_num, val, header_fmt)
                    max_w = max(len(str(val)), 12)
                    ws_sum.set_column(col_num, col_num, max_w + 3)

        excel_bytes = output_buffer.getvalue()

        if filepath is not None:
            p = Path(filepath).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(excel_bytes)
            return str(p)

        return excel_bytes
