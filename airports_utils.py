from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


# Ajuste aqui a lista "fixa" de aeroportos da CAOP.
# Tudo que estiver aqui será marcado com is_caop=True ao carregar a base.
CAOP_AIRPORTS = {
    "SBBR",
    "SBEG",
    "SBMG"
}

# ICAOs que sempre devem ser ignorados no mapa
IGNORED_ICAOS_ALWAYS = {
    "ZZZZ",
}

# Colunas candidatas para ICAO na base de aeroportos
ICAO_CANDIDATES = [
    "icao",
    "icao_code",
    "ident",
    "gps_code",
    "local_code",
]

# Colunas candidatas de trecho no dataframe operacional
TRECHO_FROM_CANDIDATES = [
    "TRECHO (DE)",
    "ICAO Origem",
    "origem",
]

TRECHO_TO_CANDIDATES = [
    "TRECHO (PARA)",
    "ICAO Destino",
    "destino",
]


def _pick_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    colset = list(columns)
    for cand in candidates:
        if cand in colset:
            return cand
    return None


def standardize_airports_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza a base de aeroportos para sempre ter:
      - icao
      - latitude_deg
      - longitude_deg
      - iso_country (quando existir)
      - is_caop
    """
    out = df.copy()

    icao_col = _pick_existing(out.columns, ICAO_CANDIDATES)
    if icao_col is None:
        raise ValueError(
            "Base de aeroportos sem coluna ICAO reconhecível. "
            "Use uma das colunas: " + ", ".join(ICAO_CANDIDATES)
        )

    out["icao"] = out[icao_col].astype(str).str.strip().str.upper()

    if "latitude_deg" not in out.columns or "longitude_deg" not in out.columns:
        raise ValueError("Base de aeroportos precisa ter latitude_deg e longitude_deg.")

    out["latitude_deg"] = pd.to_numeric(out["latitude_deg"], errors="coerce")
    out["longitude_deg"] = pd.to_numeric(out["longitude_deg"], errors="coerce")

    if "iso_country" not in out.columns:
        out["iso_country"] = None

    if "name" not in out.columns:
        out["name"] = None

    if "municipality" not in out.columns:
        out["municipality"] = None

    if "is_caop" not in out.columns:
        out["is_caop"] = False

    out["is_caop"] = out["is_caop"].fillna(False).astype(bool)
    out.loc[out["icao"].isin(CAOP_AIRPORTS), "is_caop"] = True

    out = out.drop_duplicates(subset=["icao"], keep="first")
    return out


def apply_overrides(airports: pd.DataFrame, overrides: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Aplica overrides manuais em cima da base principal.
    Espera colunas como:
      icao, latitude_deg, longitude_deg, iso_country, name, municipality, is_caop, notes
    """
    if overrides is None or overrides.empty:
        return airports

    base = airports.copy()
    ov = overrides.copy()

    if "icao" not in ov.columns:
        raise ValueError("Arquivo de overrides precisa ter a coluna 'icao'.")

    ov["icao"] = ov["icao"].astype(str).str.strip().str.upper()

    # garante colunas opcionais
    for col in ["latitude_deg", "longitude_deg", "iso_country", "name", "municipality", "is_caop", "notes"]:
        if col not in ov.columns:
            ov[col] = None

    ov["latitude_deg"] = pd.to_numeric(ov["latitude_deg"], errors="coerce")
    ov["longitude_deg"] = pd.to_numeric(ov["longitude_deg"], errors="coerce")
    ov["is_caop"] = ov["is_caop"].fillna(False).astype(bool)

    base = base.set_index("icao", drop=False)

    for _, row in ov.iterrows():
        icao = row["icao"]
        if icao in base.index:
            for col in ["latitude_deg", "longitude_deg", "iso_country", "name", "municipality", "is_caop"]:
                val = row[col]
                if pd.notna(val) and val != "":
                    base.loc[icao, col] = val
        else:
            new_row = {col: None for col in base.columns}
            for col in ["icao", "latitude_deg", "longitude_deg", "iso_country", "name", "municipality", "is_caop"]:
                if col in new_row:
                    new_row[col] = row[col]
            base.loc[icao] = new_row

    out = base.reset_index(drop=True)
    out.loc[out["icao"].isin(CAOP_AIRPORTS), "is_caop"] = True
    return out


def load_airports(
    csv_path: Optional[str | Path] = None,
    allow_download: bool = False,
    data_dir: str | Path = "data",
) -> pd.DataFrame:
    """
    Ordem de carregamento:
    1) csv_path explícito
    2) data/airports_master.csv
    3) data/airports.csv
    4) download do OurAirports (somente se allow_download=True)

    Depois aplica, se existir:
    - data/icao_overrides.csv
    """
    data_dir = Path(data_dir)

    base_path = None
    if csv_path:
        base_path = Path(csv_path)
    elif (data_dir / "airports_master.csv").exists():
        base_path = data_dir / "airports_master.csv"
    elif (data_dir / "airports.csv").exists():
        base_path = data_dir / "airports.csv"

    if base_path is not None and base_path.exists():
        airports = pd.read_csv(base_path)
    elif allow_download:
        airports = pd.read_csv("https://ourairports.com/data/airports.csv")
    else:
        raise FileNotFoundError(
            "Nenhuma base de aeroportos encontrada. "
            "Coloque 'airports_master.csv' ou 'airports.csv' na pasta data/."
        )

    airports = standardize_airports_df(airports)

    overrides_path = data_dir / "icao_overrides.csv"
    overrides = pd.read_csv(overrides_path) if overrides_path.exists() else None
    airports = apply_overrides(airports, overrides)

    return airports


def airport_visit_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conta visitas por ICAO somando origem + destino.
    """
    from_col = _pick_existing(df.columns, TRECHO_FROM_CANDIDATES)
    to_col = _pick_existing(df.columns, TRECHO_TO_CANDIDATES)

    if from_col is None or to_col is None:
        return pd.DataFrame(columns=["icao", "visitas"])

    s = pd.concat([df[from_col], df[to_col]], ignore_index=True).dropna()
    s = s.astype(str).str.strip().str.upper()

    # Mantém tudo para auditoria; o filtro real fica em classify_icao_points
    return s.value_counts().rename_axis("icao").reset_index(name="visitas")


def classify_icao_points(
    visits: pd.DataFrame,
    airports: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna:
      - mapped: ICAOs com coordenadas e categoria (Brasil / Exterior / CAOP)
      - ignored: ICAOs ignorados + motivo
    """
    if visits.empty:
        empty_mapped = pd.DataFrame(
            columns=["icao", "visitas", "latitude_deg", "longitude_deg", "iso_country", "is_caop", "point_category"]
        )
        empty_ignored = pd.DataFrame(columns=["icao", "motivo", "visitas"])
        return empty_mapped, empty_ignored

    v = visits.copy()
    v["icao"] = v["icao"].astype(str).str.strip().str.upper()

    ignored_rows = []

    # ZZZZ e afins explicitamente ignorados
    mask_always = v["icao"].isin(IGNORED_ICAOS_ALWAYS)
    if mask_always.any():
        for _, row in v.loc[mask_always].iterrows():
            ignored_rows.append({"icao": row["icao"], "motivo": "ICAO especial ignorado", "visitas": row["visitas"]})

    # formato inválido (qualquer coisa que não tenha exatamente 4 chars)
    mask_invalid = (~mask_always) & (v["icao"].str.len() != 4)
    if mask_invalid.any():
        for _, row in v.loc[mask_invalid].iterrows():
            ignored_rows.append({"icao": row["icao"], "motivo": "Código inválido / não-ICAO", "visitas": row["visitas"]})

    valid = v.loc[~mask_always & ~mask_invalid].copy()

    merged = valid.merge(airports, on="icao", how="left")

    # não encontrado na base
    mask_not_found = merged["latitude_deg"].isna() | merged["longitude_deg"].isna()
    if mask_not_found.any():
        for _, row in merged.loc[mask_not_found, ["icao", "visitas"]].iterrows():
            ignored_rows.append({"icao": row["icao"], "motivo": "Sem coordenadas na base de aeroportos", "visitas": row["visitas"]})

    mapped = merged.loc[~mask_not_found].copy()

    if not mapped.empty:
        mapped["point_category"] = "Brasil"
        mapped.loc[mapped["iso_country"].fillna("").astype(str).str.upper().ne("BR"), "point_category"] = "Exterior"
        mapped.loc[mapped["is_caop"].fillna(False).astype(bool), "point_category"] = "CAOP"

    ignored = pd.DataFrame(ignored_rows)
    if not ignored.empty:
        ignored = (
            ignored.groupby(["icao", "motivo"], dropna=False)["visitas"]
            .sum()
            .reset_index()
            .sort_values(["visitas", "icao"], ascending=[False, True])
        )
    else:
        ignored = pd.DataFrame(columns=["icao", "motivo", "visitas"])

    return mapped, ignored


def save_ignored_report(ignored: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ignored.to_csv(path, index=False)
