from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


DEFAULT_SHEET_TOTAL = "_____TOTAL_____"


def _norm(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_key(s: str) -> str:
    s = _norm(s).lower()
    repl = {
        "á": "a", "à": "a", "â": "a", "ã": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "ô": "o", "õ": "o",
        "ú": "u",
        "ç": "c",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    s = re.sub(r"[^a-z0-9_ ()+:/-]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_ole(bytes_head: bytes) -> bool:
    # OLE Compound File signature (xls antigo)
    return bytes_head.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1")


def _is_zip(bytes_head: bytes) -> bool:
    return bytes_head.startswith(b"PK")


def excel_engine_from_bytes(xls_bytes: bytes) -> Optional[str]:
    head = xls_bytes[:8]
    if _is_ole(head):
        return "xlrd"
    if _is_zip(head):
        return "openpyxl"
    return None  # deixa pandas tentar


def list_sheets(xls: Union[str, io.BytesIO, bytes]) -> List[str]:
    if isinstance(xls, (bytes, bytearray)):
        eng = excel_engine_from_bytes(bytes(xls))
        bio = io.BytesIO(xls)
        xf = pd.ExcelFile(bio, engine=eng) if eng else pd.ExcelFile(bio)
        return list(xf.sheet_names)
    xf = pd.ExcelFile(xls)
    return list(xf.sheet_names)


def read_excel_any(
    xls: Union[str, io.BytesIO, bytes],
    sheet_name: str,
    header: Union[int, None] = 0,
) -> pd.DataFrame:
    if isinstance(xls, (bytes, bytearray)):
        eng = excel_engine_from_bytes(bytes(xls))
        bio = io.BytesIO(xls)
        return pd.read_excel(bio, sheet_name=sheet_name, header=header, engine=eng) if eng else pd.read_excel(bio, sheet_name=sheet_name, header=header)
    return pd.read_excel(xls, sheet_name=sheet_name, header=header)


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {c: _norm_key(c) for c in cols}
    cand_norm = [_norm_key(x) for x in candidates]
    for c in cols:
        if norm_map[c] in cand_norm:
            return c
    for c in cols:
        nk = norm_map[c]
        for cn in cand_norm:
            if cn and cn in nk:
                return c
    return None


def _to_datetime_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def parse_duration_hours(series: pd.Series) -> pd.Series:
    """
    Converte diferentes representações de tempo para HORAS (float).
    Suporta:
    - número (horas) ou fração de dia (Excel time -> *24)
    - datetime.time
    - string "hh:mm" / "h:mm:ss" / "1,5" / "2.75" / "2h30"
    """
    s = series

    # caso datetime.time em células
    import datetime as _dt

    def _time_to_h(v):
        if isinstance(v, _dt.time):
            return v.hour + v.minute/60 + v.second/3600
        return np.nan

    # tenta capturar time objects
    out_time = s.map(_time_to_h)
    has_time = out_time.notna().any()

    # tenta numérico direto
    num = pd.to_numeric(s, errors="coerce")

    # strings
    txt = s.astype(str).str.strip()
    txt = txt.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    # hh:mm[:ss]
    hhmm = txt.str.extract(r"^(?P<h>\d{1,3})\s*:\s*(?P<m>\d{1,2})(?:\s*:\s*(?P<sec>\d{1,2}))?$")
    hh = pd.to_numeric(hhmm["h"], errors="coerce")
    mm = pd.to_numeric(hhmm["m"], errors="coerce")
    sec = pd.to_numeric(hhmm["sec"], errors="coerce").fillna(0)
    hhmm_hours = hh + mm/60 + sec/3600

    # "2h30" ou "2 h 30"
    hh_h = txt.str.extract(r"^(?P<h>\d{1,3})\s*h\s*(?P<m>\d{1,2})?$")
    hh2 = pd.to_numeric(hh_h["h"], errors="coerce")
    mm2 = pd.to_numeric(hh_h["m"], errors="coerce").fillna(0)
    hhm_hours = hh2 + mm2/60

    # decimal com vírgula
    dec = pd.to_numeric(txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")

    # junta candidatos por prioridade
    out = pd.Series(np.nan, index=s.index, dtype=float)

    # times
    if has_time:
        out = out.combine_first(out_time.astype(float))

    # hh:mm
    out = out.combine_first(hhmm_hours.astype(float))

    # 2h30
    out = out.combine_first(hhm_hours.astype(float))

    # numérico
    out = out.combine_first(num.astype(float))

    # decimal vírgula
    out = out.combine_first(dec.astype(float))

    # heurística: fração de dia (0 < x < 1.1) e muitos valores assim -> *24
    mask = out.notna()
    if mask.any():
        vals = out[mask]
        frac = float(((vals > 0) & (vals < 1.1)).mean())
        # se a maioria parece fração do dia, converte
        if frac >= 0.60:
            out = out * 24.0

    return out


@dataclass
class ColumnMap:
    demandante: Optional[str] = None
    date: Optional[str] = None
    year: Optional[str] = None
    status: Optional[str] = None
    icao_from: Optional[str] = None
    icao_to: Optional[str] = None
    aircraft: Optional[str] = None
    ttv: Optional[str] = None
    asa: Optional[str] = None
    op_name: Optional[str] = None
    op_metrics: List[str] = None


def infer_columns(df: pd.DataFrame) -> ColumnMap:
    cm = ColumnMap(op_metrics=[])

    cm.demandante = _pick_col(df, ["Demandante", "DEMANDANTE", "Órgão", "Orgao", "Solicitante", "Unidade"])
    cm.status = _pick_col(df, ["Situação", "Situacao", "Status", "Resultado", "Andamento"])
    cm.aircraft = _pick_col(df, ["Aeronave", "AERONAVE", "Prefixo", "Matrícula", "Matricula", "ARP", "ARPs"])
    cm.ttv = _pick_col(df, ["TTV", "Horas", "Horas de voo", "Tempo de voo", "Tempo", "HH", "Horas de Voo"])
    cm.date = _pick_col(df, ["Data", "DATA", "Dt", "DT", "Data do voo", "Data da missão", "Data da missao"])
    cm.year = _pick_col(df, ["Ano", "ANO", "Year"])

    cm.icao_from = _pick_col(df, ["Origem", "ICAO Origem", "Icao Origem", "From", "DE", "ICAO1", "ICAO 1"])
    cm.icao_to = _pick_col(df, ["Destino", "ICAO Destino", "Icao Destino", "To", "PARA", "ICAO2", "ICAO 2"])

    cm.asa = _pick_col(df, ["ASA (F ou R)", "ASA", "Asa (F ou R)", "Asa", "F ou R"])

    cm.op_name = _pick_col(df, ["ERR (NOME DA OPERAÇÃO)", "NOME DA OPERAÇÃO", "Nome da Operação"])
    cols = list(df.columns)
    for c in cols:
        k = _norm_key(c)
        if any(tok in k for tok in ["obs", "int", "mobiliza", "destruicao", "erradicacao"]):
            cm.op_metrics.append(c)

    # dedup mantendo ordem
    seen = set()
    cm.op_metrics = [c for c in cm.op_metrics if not (c in seen or seen.add(c))]

    return cm


def load_total_table(
    xls: Union[str, io.BytesIO, bytes],
    sheet_name: str = DEFAULT_SHEET_TOTAL,
) -> Tuple[pd.DataFrame, ColumnMap]:
    df = read_excel_any(xls, sheet_name=sheet_name, header=0)
    df.columns = [_norm(c) for c in df.columns]
    df = df.dropna(how="all")

    cm = infer_columns(df)

    # YEAR auxiliar
    if cm.year and cm.year in df.columns:
        df["_year"] = pd.to_numeric(df[cm.year], errors="coerce").astype("Int64")
    elif cm.date and cm.date in df.columns:
        dt = _to_datetime_series(df[cm.date])
        df["_year"] = dt.dt.year.astype("Int64")
    else:
        df["_year"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    # ASA auxiliar
    if cm.asa and cm.asa in df.columns:
        a = df[cm.asa].astype(str).str.strip().str.upper()
        a = a.where(a.isin(["F", "R"]), other=pd.NA)
        df["_asa"] = a
    else:
        df["_asa"] = pd.Series([pd.NA] * len(df), dtype="object")

    # ICAO normalize
    for col in [cm.icao_from, cm.icao_to]:
        if col and col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    # TTV hours robusto
    if cm.ttv and cm.ttv in df.columns:
        df["_ttv"] = parse_duration_hours(df[cm.ttv])
    else:
        df["_ttv"] = pd.Series([np.nan] * len(df), dtype=float)

    return df, cm


def completion_rate(df: pd.DataFrame, status_col: Optional[str], done_values: List[str]) -> float:
    if not status_col or status_col not in df.columns:
        return float("nan")
    s = df[status_col].astype(str).fillna("")
    done_norm = {_norm_key(x) for x in done_values}
    ok = s.map(lambda x: _norm_key(x) in done_norm)
    if len(ok) == 0:
        return float("nan")
    return float(ok.mean() * 100.0)



def standardize_airports_df(a: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza a base de aeroportos para sempre ter:
      - icao (string, upper)
      - latitude_deg (float)
      - longitude_deg (float)

    Aceita variações comuns:
      - icao_code (OurAirports) -> icao
      - ident (OurAirports) como fallback
      - gps_code / local_code como fallback
    """
    a = a.copy()

    # escolher coluna ICAO
    cand = None
    for c in ["icao", "icao_code", "ident", "gps_code", "local_code"]:
        if c in a.columns:
            cand = c
            break
    if cand is None:
        return a  # deixa o chamador acusar erro com colunas

    if "icao" not in a.columns:
        a["icao"] = a[cand]
    a["icao"] = a["icao"].astype(str).str.strip().str.upper()

    # garantir lat/lon numérico
    if "latitude_deg" in a.columns:
        a["latitude_deg"] = pd.to_numeric(a["latitude_deg"], errors="coerce")
    if "longitude_deg" in a.columns:
        a["longitude_deg"] = pd.to_numeric(a["longitude_deg"], errors="coerce")

    return a

def airports_from_csv(csv_bytes: bytes) -> pd.DataFrame:
    a = pd.read_csv(io.BytesIO(csv_bytes))
    return standardize_airports_df(a)


def load_airports_local(path: str) -> Optional[pd.DataFrame]:
    try:
        a = pd.read_csv(path)
        return standardize_airports_df(a)
    except Exception:
        return None


def count_or_sum(series: pd.Series) -> Tuple[float, str]:
    """
    Se a série é majoritariamente numérica -> soma.
    Senão -> conta linhas preenchidas (marcações).
    """
    s = series.copy()
    num = pd.to_numeric(s, errors="coerce")
    frac_numeric = float(num.notna().mean()) if len(num) else 0.0
    if frac_numeric >= 0.60:
        return float(num.fillna(0.0).sum()), "soma"
    txt = s.astype(str).str.strip()
    txt = txt[txt.ne("") & txt.ne("nan") & txt.ne("None")]
    txt = txt[~txt.isin(["0", "0.0", "0,0"])]
    return float(len(txt)), "contagem"
