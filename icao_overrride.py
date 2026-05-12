import os
import re
import pandas as pd
import numpy as np


DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "icao_overrides.csv")

ARQ_PRIVADOS = "AerodromosPrivados.csv"
ARQ_PUBLICOS = "cadastro-de-aerodromos-civis-publicos.csv"
ARQ_HELIPONTOS = "Helipontos.csv"
ARQ_HELIDECKS = "Helidecks.csv"


def norm_icao(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().upper()


def to_float_br(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s == "-":
        return np.nan
    s = s.replace(".", "").replace(",", ".") if s.count(",") == 1 and s.count(".") > 1 else s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_dms(value: str):
    if pd.isna(value):
        return np.nan

    s = str(value).strip().upper()
    if not s or s == "-":
        return np.nan

    # decimal já pronto
    s_dec = s.replace(",", ".")
    try:
        return float(s_dec)
    except Exception:
        pass

    # exemplos: 8° 20' 55'' S | 23°14'19,4"S
    pattern = r"(\d+)[°º]\s*(\d+)?'?\s*(\d+(?:[.,]\d+)?)?\"?\s*([NSEW])"
    m = re.search(pattern, s)
    if not m:
        return np.nan

    deg = float(m.group(1))
    minute = float(m.group(2) or 0)
    second = float((m.group(3) or "0").replace(",", "."))
    hemi = m.group(4)

    dec = deg + minute / 60 + second / 3600
    if hemi in ("S", "W"):
        dec = -dec
    return dec


def standardize_publicos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", skiprows=1)

    out = pd.DataFrame({
        "icao": df["CÓDIGO OACI"].map(norm_icao),
        "ident": df["CÓDIGO OACI"].map(norm_icao),
        "name": df["NOME"].astype(str).str.strip(),
        "municipality": df["MUNICÍPIO ATENDIDO"].astype(str).str.strip(),
        "state": df["UF"].astype(str).str.strip(),
        "latitude_deg": df["LATITUDE"].map(parse_dms),
        "longitude_deg": df["LONGITUDE"].map(parse_dms),
        "iso_country": "BR",
        "type": "public_aerodrome",
        "point_category": "Brasil",
        "source": "ANAC_publicos",
    })

    return out


def standardize_privados(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="latin1", skiprows=1)

    out = pd.DataFrame({
        "icao": df["Código OACI"].map(norm_icao),
        "ident": df["Código OACI"].map(norm_icao),
        "name": df["Nome"].astype(str).str.strip(),
        "municipality": df["Município"].astype(str).str.strip(),
        "state": df["UF"].astype(str).str.strip(),
        "latitude_deg": df["LATGEOPOINT"].map(to_float_br),
        "longitude_deg": df["LONGEOPOINT"].map(to_float_br),
        "iso_country": "BR",
        "type": "private_aerodrome",
        "point_category": "Brasil",
        "source": "ANAC_privados",
    })

    return out


def standardize_helipontos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="latin1", skiprows=1)

    out = pd.DataFrame({
        "icao": df["Código OACI"].map(norm_icao),
        "ident": df["Código OACI"].map(norm_icao),
        "name": df["Nome"].astype(str).str.strip(),
        "municipality": df["Município"].astype(str).str.strip(),
        "state": df["UF"].astype(str).str.strip(),
        "latitude_deg": df["LATGEOPOINT"].map(to_float_br),
        "longitude_deg": df["LONGEOPOINT"].map(to_float_br),
        "iso_country": "BR",
        "type": "heliport",
        "point_category": "Brasil",
        "source": "ANAC_helipontos",
    })

    return out


def standardize_helidecks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", skiprows=1)

    # arquivo enviado não trouxe coordenadas
    out = pd.DataFrame({
        "icao": df["CÓDIGO OACI"].map(norm_icao),
        "ident": df["CÓDIGO OACI"].map(norm_icao),
        "name": df["NOME"].astype(str).str.strip(),
        "municipality": "",
        "state": "",
        "latitude_deg": np.nan,
        "longitude_deg": np.nan,
        "iso_country": "BR",
        "type": "helideck",
        "point_category": "Brasil",
        "source": "ANAC_helidecks_sem_coordenadas",
    })

    return out


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    frames = [
        standardize_publicos(ARQ_PUBLICOS),
        standardize_privados(ARQ_PRIVADOS),
        standardize_helipontos(ARQ_HELIPONTOS),
        standardize_helidecks(ARQ_HELIDECKS),
    ]

    all_points = pd.concat(frames, ignore_index=True)

    all_points["icao"] = all_points["icao"].astype(str).str.strip().str.upper()
    all_points = all_points[all_points["icao"].ne("")]
    all_points = all_points.drop_duplicates(subset=["icao"], keep="first")

    # overrides úteis para mapa = só o que tem coordenada
    overrides = all_points.dropna(subset=["latitude_deg", "longitude_deg"]).copy()

    overrides = overrides[
        [
            "icao",
            "ident",
            "name",
            "municipality",
            "state",
            "latitude_deg",
            "longitude_deg",
            "iso_country",
            "type",
            "point_category",
            "source",
        ]
    ].sort_values(["icao"])

    overrides.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Arquivo gerado: {OUT_FILE}")
    print(f"Total com coordenadas: {len(overrides)}")
    print(f"Helidecks sem coordenadas ignorados: {all_points['latitude_deg'].isna().sum()}")


if __name__ == "__main__":
    main()