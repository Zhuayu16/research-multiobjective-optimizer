from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatImportResult:
    data: pd.DataFrame
    summary: pd.DataFrame
    files: list[Path]


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文本编码：{path.name}")


def _clean_columns(columns: list[object]) -> list[str]:
    cleaned: list[str] = []
    used: dict[str, int] = {}
    for index, column in enumerate(columns, start=1):
        name = re.sub(r"\s+", " ", str(column).strip().strip('"')) or f"列{index}"
        used[name] = used.get(name, 0) + 1
        cleaned.append(name if used[name] == 1 else f"{name}_{used[name]}")
    return cleaned


def parse_dat_file(path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(path)
    if path.name.lower().endswith(".dat.h5"):
        raise ValueError(
            f"{path.name} 是 Fluent 二进制结果文件。它需要配套 .cas.h5 和 Fluent 报告定义，"
            "不能直接当作数值表读取。请导入 Fluent/CFD-Post 导出的文本 .dat，"
            "或先导出压降、功率和效率报告。"
        )
    if path.suffix.lower() != ".dat":
        raise ValueError(f"不是 .dat 文本文件：{path.name}")

    text = _read_text(path).replace("\x00", "")
    lines = text.splitlines()
    series_name = path.stem
    data_start: int | None = None
    for index, raw in enumerate(lines):
        marker = raw.strip().lower()
        if marker == "[name]":
            for candidate in lines[index + 1 :]:
                if candidate.strip():
                    series_name = candidate.strip()
                    break
        if marker == "[data]":
            data_start = index + 1
            break

    if data_start is not None:
        payload = "\n".join(lines[data_start:]).strip()
        if not payload:
            raise ValueError(f"{path.name} 的 [Data] 区域为空。")
        frame = pd.read_csv(io.StringIO(payload), sep=None, engine="python")
    else:
        useful = [line for line in lines if line.strip() and not line.lstrip().startswith(("#", "//", ";"))]
        if not useful:
            raise ValueError(f"{path.name} 没有可读取的数据。")
        payload = "\n".join(useful)
        try:
            frame = pd.read_csv(io.StringIO(payload), sep=None, engine="python")
        except Exception:
            frame = pd.read_csv(io.StringIO(payload), sep=r"\s+", engine="python")

    frame.columns = _clean_columns(list(frame.columns))
    frame = frame.dropna(how="all").reset_index(drop=True)
    for column in frame.columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() >= max(2, int(0.8 * len(frame))):
            frame[column] = converted
    if frame.empty:
        raise ValueError(f"{path.name} 没有有效数据行。")

    stem = path.stem
    q_match = re.search(r"(?i)(?:^|[^A-Z])Q\s*([0-9]+(?:\.[0-9]+)?)", stem)
    n_match = re.search(r"(?i)N\s*([0-9]+(?:\.[0-9]+)?)", stem)
    metadata: dict[str, object] = {
        "源文件": path.name,
        "曲线名称": series_name,
        "文件名Q": float(q_match.group(1)) if q_match else np.nan,
        "文件名N": float(n_match.group(1)) if n_match else np.nan,
    }
    return frame, metadata


def _profile_summary(frame: pd.DataFrame, metadata: dict[str, object]) -> dict[str, object]:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    result = dict(metadata)
    result["数据行数"] = len(frame)
    if len(numeric) < 2:
        result["状态"] = "数值列不足2列"
        return result
    x_name, y_name = numeric[0], numeric[1]
    valid = frame[[x_name, y_name]].dropna().sort_values(x_name)
    if valid.empty:
        result["状态"] = "无有效XY数据"
        return result
    x = valid[x_name].to_numpy(float)
    y = valid[y_name].to_numpy(float)
    mean = float(np.mean(y))
    result.update(
        {
            "状态": "可用",
            "X列": x_name,
            "Y列": y_name,
            "X最小": float(np.min(x)),
            "X最大": float(np.max(x)),
            "Y最小": float(np.min(y)),
            "Y最大": float(np.max(y)),
            "Y平均": mean,
            "Y标准差": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
            "Y极差": float(np.ptp(y)),
            "变异系数%": float(np.std(y, ddof=1) / abs(mean) * 100.0) if len(y) > 1 and mean != 0 else np.nan,
            "曲线积分": float(np.trapezoid(y, x)) if len(y) > 1 else np.nan,
        }
    )
    return result


def import_dat_files(paths: list[str | Path]) -> DatImportResult:
    if not paths:
        raise ValueError("没有选择 DAT 文件。")
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    resolved: list[Path] = []
    for item in paths:
        path = Path(item)
        frame, metadata = parse_dat_file(path)
        enriched = frame.copy()
        enriched.insert(0, "源文件", metadata["源文件"])
        enriched.insert(1, "曲线名称", metadata["曲线名称"])
        enriched.insert(2, "文件名Q", metadata["文件名Q"])
        enriched.insert(3, "文件名N", metadata["文件名N"])
        frames.append(enriched)
        summaries.append(_profile_summary(frame, metadata))
        resolved.append(path.resolve())
    return DatImportResult(
        data=pd.concat(frames, ignore_index=True, sort=False),
        summary=pd.DataFrame(summaries),
        files=resolved,
    )
