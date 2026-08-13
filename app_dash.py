from __future__ import annotations

import base64
import io
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, MATCH, callback_context, dcc, html, dash_table, no_update
from plotly.colors import qualitative

from data_utils import (
    count_or_sum,
    infer_columns,
    list_sheets,
    load_total_table,
    parse_duration_hours,
)
from airports_utils import load_airports, airport_visit_counts, classify_icao_points
from map_utils import build_grid, contour_traces_from_grid


DEFAULT_SHEET = "_____TOTAL_____"
MAP_STYLES = ["carto-darkmatter", "open-street-map", "carto-positron"]
MAP_MODES = [
    "Zonas de calor (colorido)",
    "Topologia (zonas + contornos)",
    "Somente pontos (todos iguais)",
    "Caminhos por aeronave",
]

POINT_COLORS = {
    "Brasil": "#4cc9f0",
    "Exterior": "#ef476f",
    "CAOP": "#ffb000",
}

UF_LABELS = {
    "AC": "AC - Acre",
    "AL": "AL - Alagoas",
    "AM": "AM - Amazonas",
    "AP": "AP - Amapa",
    "BA": "BA - Bahia",
    "CE": "CE - Ceara",
    "DF": "DF - Distrito Federal",
    "ES": "ES - Espirito Santo",
    "GO": "GO - Goias",
    "MA": "MA - Maranhao",
    "MG": "MG - Minas Gerais",
    "MS": "MS - Mato Grosso do Sul",
    "MT": "MT - Mato Grosso",
    "PA": "PA - Para",
    "PB": "PB - Paraiba",
    "PE": "PE - Pernambuco",
    "PI": "PI - Piaui",
    "PR": "PR - Parana",
    "RJ": "RJ - Rio de Janeiro",
    "RN": "RN - Rio Grande do Norte",
    "RO": "RO - Rondonia",
    "RR": "RR - Roraima",
    "RS": "RS - Rio Grande do Sul",
    "SC": "SC - Santa Catarina",
    "SE": "SE - Sergipe",
    "SP": "SP - Sao Paulo",
    "TO": "TO - Tocantins",
}

FIXED_COLUMNS = {
    "demandante": "DEM.",
    "asa": "ASA (F ou R)",
    "ttv": "TTV",
    "year": "ANO",
    "date": "DATA",
    "icao_from": "TRECHO (DE)",
    "icao_to": "TRECHO (PARA)",
    "op_name": "RESULTADO / OBSERVAÇÕES PERTINENTES",
    "aircraft": "AERONAVE (matrícula)",
    "passengers": [
        "PASSAG.  desem.",
        "PASSAG. desem.",
        "PASSAG. desemb.",
        "PASSAG. desembarq.",
    ],
    "prisoners": [
        "PRESOS",
        "PRESOS desem.",
        "PRESOS desemb.",
        "PRESOS desembarq.",
    ],
    "cargo": [
        "CARGA(Kg) desembarq.",
        "CARGA(Kg) desembarq",
        "CARGA (Kg) desembarq.",
        "CARGA (Kg) desembarq",
    ],
    "op_metrics": [
        "OBS (IMA + REC) (1 por dia)",
        "INT (1 por dia)",
        "MOBILIZAÇÃO (POR NOME DA OPERAÇÃO, unidade apoiada, serviço especializado)",
        "DESTRUIÇÃO + ERRADICAÇÃO (1 POR DIA CADA)",
        "ERR (NOME DA OPERAÇÃO)",
    ],
    "nat": "NAT.",
    "espec": "ESPEC.",
}

NAT_MAP = {
    "CQ": "Cheque / Recheque",
    "TN": "Treinamento",
    "EX": "Voo de experiência",
    "TR": "Traslado",
    "SA": "Serviço aéreo especializado",
}

ESPEC_MAP = {
    "CQ": "Cheque / Recheque",
    "TN": "Treinamento",
    "EX": "Voo de experiência",
    "SPF": "Servidores Polícia Federal",
    "SFN": "Servidores Força Nacional",
    "SDE": "Servidores DEPEN",
    "SOO": "Servidores de outros órgãos",
    "ERR": "Erradicação de cultivos ilícitos",
    "ESC": "Escolta aérea",
    "ESP": "Escolta de presos PF",
    "ESD": "Escolta de presos DEPEN",
    "ESO": "Escolta de presos outro órgão",
    "IMA": "Imageamento",
    "INT": "Intervenção policial",
    "LPQ": "LPQD",
    "REC": "Apoio / reconhecimento - levantamento / evento",
    "RES": "Resgate",
    "TRP": "Traslado sem PAX e carga",
    "TRM": "Traslado para manutenção",
    "TRO": "Treinamento (operadores / outros a bordo)",
    "TCE": "Transporte de carga",
    "DES": "Destruição de máquinas / equipamentos",
    "EVT": "Evento",
}

OPS_SPEC_LABELS = {
    "OBS": "Observação",
    "IMA": "Observação",
    "REC": "Observação",
    "INT": "Intervenção",
    "MOB": "Mobilização",
    "ERR": "Erradicação",
    "DES": "Destruição",
    "TCE": "Mobilização",
    "EVT": "Evento",
}


@dataclass
class ColumnConfig:
    demandante: Optional[str] = None
    asa: Optional[str] = None
    ttv: Optional[str] = None
    year: Optional[str] = None
    date: Optional[str] = None
    icao_from: Optional[str] = None
    icao_to: Optional[str] = None
    op_name: Optional[str] = None
    aircraft: Optional[str] = None
    status: Optional[str] = None
    passengers: Optional[str] = None
    prisoners: Optional[str] = None
    cargo: Optional[str] = None
    nat: Optional[str] = None
    espec: Optional[str] = None
    op_metrics: list[str] | None = None


def _format_int(v: int) -> str:
    return f"{v:,}".replace(",", ".")


def _format_float_br(v: float) -> str:
    return f"{v:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _existing_column(df: pd.DataFrame, name: Optional[str]) -> Optional[str]:
    return name if name and name in df.columns else None


def _existing_columns(df: pd.DataFrame, names: list[str]) -> list[str]:
    return [c for c in names if c in df.columns]


def _existing_first(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.has_duplicates:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _safe_numeric_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)


def build_fixed_config(df: pd.DataFrame, inferred) -> ColumnConfig:
    return ColumnConfig(
        demandante=_existing_column(df, FIXED_COLUMNS["demandante"]) or getattr(inferred, "demandante", None),
        asa=_existing_column(df, FIXED_COLUMNS["asa"]) or getattr(inferred, "asa", None),
        ttv=_existing_column(df, FIXED_COLUMNS["ttv"]) or getattr(inferred, "ttv", None),
        year=_existing_column(df, FIXED_COLUMNS["year"]) or getattr(inferred, "year", None),
        date=_existing_column(df, FIXED_COLUMNS["date"]) or getattr(inferred, "date", None),
        icao_from=_existing_column(df, FIXED_COLUMNS["icao_from"]) or getattr(inferred, "icao_from", None),
        icao_to=_existing_column(df, FIXED_COLUMNS["icao_to"]) or getattr(inferred, "icao_to", None),
        op_name=_existing_column(df, FIXED_COLUMNS["op_name"]) or getattr(inferred, "op_name", None),
        aircraft=_existing_column(df, FIXED_COLUMNS["aircraft"]) or getattr(inferred, "aircraft", None),
        passengers=_existing_first(df, FIXED_COLUMNS["passengers"]),
        prisoners=_existing_first(df, FIXED_COLUMNS["prisoners"]),
        cargo=_existing_first(df, FIXED_COLUMNS["cargo"]),
        nat=_existing_column(df, FIXED_COLUMNS["nat"]),
        espec=_existing_column(df, FIXED_COLUMNS["espec"]),
        status=getattr(inferred, "status", None),
        op_metrics=_existing_columns(df, FIXED_COLUMNS["op_metrics"]) or getattr(inferred, "op_metrics", []),
    )


def apply_column_overrides(df: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    out = _drop_duplicate_columns(df).copy()

    if cfg.year and cfg.year in out.columns:
        out["_year"] = pd.to_numeric(out[cfg.year], errors="coerce").astype("Int64")
    elif cfg.date and cfg.date in out.columns:
        out["_year"] = pd.to_datetime(out[cfg.date], errors="coerce", dayfirst=True).dt.year.astype("Int64")

    if cfg.date and cfg.date in out.columns:
        out["_date"] = pd.to_datetime(out[cfg.date], errors="coerce", dayfirst=True)
        out["_month_dt"] = out["_date"].dt.to_period("M").dt.to_timestamp()
        out["_month"] = out["_month_dt"].dt.strftime("%Y-%m")
    else:
        out["_date"] = pd.NaT
        out["_month_dt"] = pd.NaT
        out["_month"] = pd.NA

    if cfg.asa and cfg.asa in out.columns:
        asa = out[cfg.asa].astype(str).str.strip().str.upper()
        out["_asa"] = asa.where(asa.isin(["F", "R"]), other=pd.NA)
    else:
        out["_asa"] = pd.NA

    if cfg.ttv and cfg.ttv in out.columns:
        out["_ttv"] = parse_duration_hours(out[cfg.ttv])
    else:
        out["_ttv"] = 0.0

    out["_passengers"] = _safe_numeric_series(out, cfg.passengers)
    out["_prisoners"] = _safe_numeric_series(out, cfg.prisoners)
    out["_cargo"] = _safe_numeric_series(out, cfg.cargo)

    if cfg.nat and cfg.nat in out.columns:
        nat = out[cfg.nat].astype(str).str.strip().str.upper()
        out["_nat_code"] = nat
        out["_nat_label"] = nat.map(NAT_MAP).fillna(nat)
    else:
        out["_nat_code"] = pd.NA
        out["_nat_label"] = pd.NA

    if cfg.espec and cfg.espec in out.columns:
        espec = out[cfg.espec].astype(str).str.strip().str.upper()
        out["_espec_code"] = espec
        out["_espec_label"] = espec.map(ESPEC_MAP).fillna(espec)
        out["_op_exec"] = espec.map(OPS_SPEC_LABELS)
    else:
        out["_espec_code"] = pd.NA
        out["_espec_label"] = pd.NA
        out["_op_exec"] = pd.NA

    return out


def get_year_options(df: pd.DataFrame) -> list[int]:
    if "_year" not in df.columns:
        return []
    years = df["_year"].dropna().astype(int).unique().tolist()
    years.sort()
    return years


def apply_filters(df: pd.DataFrame, cfg: ColumnConfig, years: list[int], asa_mode: str, exclude_caop: bool) -> pd.DataFrame:
    out = df.copy()
    if years and "_year" in out.columns:
        out = out[out["_year"].isin(years)]
    if asa_mode and asa_mode != "Todas" and "_asa" in out.columns:
        code = "F" if "F" in asa_mode else "R"
        out = out[out["_asa"] == code]
    if exclude_caop and cfg.demandante and cfg.demandante in out.columns:
        dem = out[cfg.demandante].astype(str).str.strip().str.upper()
        out = out[dem != "CAOP"]
    return out


def aggregate_visits_by_icao(df: pd.DataFrame, icao_from: Optional[str], icao_to: Optional[str]) -> pd.DataFrame:
    if not icao_from or not icao_to or icao_from not in df.columns or icao_to not in df.columns:
        return pd.DataFrame(columns=["icao", "visitas"])
    values = pd.concat([df[icao_from], df[icao_to]], ignore_index=True).dropna()
    values = values.astype(str).str.strip().str.upper()
    values = values[(values.str.len() == 4) & (values != "ZZZZ")]
    return values.value_counts().rename_axis("icao").reset_index(name="visitas")


def get_map_center(df_points: pd.DataFrame) -> dict:
    if df_points.empty:
        return {"lat": -14.2, "lon": -51.9}
    return {"lat": float(df_points["latitude_deg"].mean()), "lon": float(df_points["longitude_deg"].mean())}


@lru_cache(maxsize=1)
def load_airports_cached(data_dir: str = "data") -> pd.DataFrame:
    return load_airports(data_dir=data_dir)


def analyze_icao_usage(df: pd.DataFrame, cfg: ColumnConfig, airports: Optional[pd.DataFrame]):
    empty_valid = pd.DataFrame(columns=["icao", "visitas", "latitude_deg", "longitude_deg", "iso_country", "categoria"])
    empty_ignored = pd.DataFrame(columns=["icao", "visitas", "motivo"])
    if not cfg.icao_from or not cfg.icao_to:
        return empty_valid, empty_ignored, {"total_raw": 0, "validos": 0, "ignorados": 0}
    visits = airport_visit_counts(df)
    if visits.empty:
        return empty_valid, empty_ignored, {"total_raw": 0, "validos": 0, "ignorados": 0}
    if airports is None or airports.empty:
        return empty_valid, empty_ignored, {"total_raw": int(visits["visitas"].sum()), "validos": 0, "ignorados": int(visits["visitas"].sum())}
    mapped, ignored = classify_icao_points(visits, airports)
    if not mapped.empty:
        mapped = mapped.rename(columns={"point_category": "categoria"})
    summary = {
        "total_raw": int(visits["visitas"].sum()),
        "validos": int(mapped["visitas"].sum()) if not mapped.empty else 0,
        "ignorados": int(ignored["visitas"].sum()) if not ignored.empty else 0,
    }
    if mapped.empty:
        mapped = empty_valid
    if ignored.empty:
        ignored = empty_ignored
    return mapped, ignored, summary


def _prepare_aircraft_base(df: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    base = _drop_duplicate_columns(df).copy()
    base[cfg.aircraft] = base[cfg.aircraft].astype(str).str.strip()
    base = base[
        base[cfg.aircraft].ne("")
        & base[cfg.aircraft].ne("nan")
        & base[cfg.aircraft].ne("None")
    ].copy()
    return base


def _normalize_aircraft_selection(selected, options: list[str], default: list[str]) -> list[str]:
    if selected is None:
        return list(default)
    if isinstance(selected, str):
        selected = [selected]
    valid = set(options)
    return [str(item).strip() for item in selected if str(item).strip() in valid]


def _aircraft_options(df: pd.DataFrame, cfg: ColumnConfig) -> list[str]:
    if not cfg.aircraft or cfg.aircraft not in df.columns:
        return []
    base = _prepare_aircraft_base(df, cfg)
    if base.empty:
        return []
    return _aircraft_grouped(base, cfg)[cfg.aircraft].astype(str).tolist()


def _aircraft_grouped(base: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    grouped = (
        base.groupby(cfg.aircraft, dropna=False)
        .agg(
            trechos=(cfg.aircraft, "size"),
            ttv_total=("_ttv", "sum"),
            passageiros_total=("_passengers", "sum"),
            presos_total=("_prisoners", "sum"),
            carga_total_kg=("_cargo", "sum"),
            asa_tipo=("_asa", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().shape[0] else "N/I"),
        )
        .reset_index()
    )
    grouped["asa_label"] = grouped["asa_tipo"].map({"F": "Asa fixa", "R": "Asa rotativa"}).fillna("Não informado")
    return grouped.sort_values(["ttv_total", "trechos"], ascending=[False, False])


def _build_aircraft_color_map(aircraft_names: list[str]) -> dict[str, str]:
    palette = qualitative.Plotly + qualitative.Dark24 + qualitative.Light24 + qualitative.Alphabet + qualitative.Safe + qualitative.Vivid
    unique_names = sorted(set(map(str, aircraft_names)))
    return {name: palette[i % len(palette)] for i, name in enumerate(unique_names)}


PT_MONTHS = {1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR", 5: "MAI", 6: "JUN", 7: "JUL", 8: "AGO", 9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ"}


def _prepare_aircraft_availability_matrix(base: pd.DataFrame, cfg: ColumnConfig, aircraft_order: list[str]):
    work = base.copy()
    work["_date"] = pd.to_datetime(work["_date"], errors="coerce")
    work["_month_dt"] = pd.to_datetime(work["_month_dt"], errors="coerce")
    work = work.dropna(subset=["_date", "_month_dt"]).copy()
    work[cfg.aircraft] = work[cfg.aircraft].astype(str).str.strip()
    work = work[work[cfg.aircraft].isin(aircraft_order)].copy()
    if work.empty:
        return pd.DataFrame(index=aircraft_order), [], "DISPONIBILIDADE MENSAL POR AERONAVE"
    work["_day"] = work["_date"].dt.floor("D")
    daily_presence = work.groupby([cfg.aircraft, "_month_dt", "_day"], dropna=False).size().reset_index(name="n")
    monthly_presence = daily_presence.groupby([cfg.aircraft, "_month_dt"], dropna=False).size().reset_index(name="dias_utilizados")
    years = sorted(work["_year"].dropna().astype(int).unique().tolist()) if "_year" in work.columns else []
    if len(years) == 1:
        all_months = pd.date_range(f"{years[0]}-01-01", f"{years[0]}-12-01", freq="MS")
        title = f"DISPONIBILIDADE MENSAL POR AERONAVE - {years[0]}"
        month_labels = [PT_MONTHS[m.month] for m in all_months]
    else:
        all_months = pd.to_datetime(sorted(work["_month_dt"].dropna().unique()))
        title = "DISPONIBILIDADE MENSAL POR AERONAVE"
        month_labels = [f"{PT_MONTHS[m.month]}/{str(m.year)[2:]}" for m in all_months]
    full_grid = pd.MultiIndex.from_product([aircraft_order, all_months], names=[cfg.aircraft, "_month_dt"]).to_frame(index=False)
    matrix_df = full_grid.merge(monthly_presence, on=[cfg.aircraft, "_month_dt"], how="left")
    matrix_df["dias_utilizados"] = matrix_df["dias_utilizados"].fillna(0)
    matrix_df["dias_mes"] = matrix_df["_month_dt"].dt.days_in_month
    matrix_df["disponibilidade_pct"] = (100.0 * matrix_df["dias_utilizados"] / matrix_df["dias_mes"]).fillna(0.0)
    label_map = {m: lbl for m, lbl in zip(all_months, month_labels)}
    matrix_df["_month_label"] = matrix_df["_month_dt"].map(label_map)
    pivot = matrix_df.pivot(index=cfg.aircraft, columns="_month_label", values="disponibilidade_pct").reindex(index=aircraft_order, columns=month_labels).fillna(0.0)
    return pivot, month_labels, title


def _build_availability_heatmap(pivot: pd.DataFrame, month_labels: list[str], title: str):
    if pivot.empty or len(month_labels) == 0:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title=title)
        return fig
    z = pivot.to_numpy(dtype=float)
    text = np.vectorize(lambda v: f"{int(round(v))}%")(z)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=month_labels,
        y=pivot.index.tolist(),
        colorscale=[[0.00, "#f0f0f0"], [0.15, "#dfe9df"], [0.35, "#bdd8bd"], [0.55, "#8dc28f"], [0.75, "#5fb56b"], [1.00, "#35a853"]],
        zmin=0,
        zmax=100,
        text=text,
        texttemplate="%{text}",
        xgap=1,
        ygap=1,
        colorbar=dict(title=dict(text="Disponibilidade (%)")),
    ))
    fig.update_layout(template="plotly_white", title=title, height=max(520, 48 * len(pivot.index)), yaxis=dict(autorange="reversed"))
    return fig


def _enrich_with_uf_from_airports(base: pd.DataFrame, cfg: ColumnConfig, airports: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    out["_uf_dest"] = pd.NA
    if airports is None or airports.empty or not cfg.icao_to or cfg.icao_to not in out.columns:
        return out
    ap = airports.copy()
    if "icao" not in ap.columns and "ident" in ap.columns:
        ap["icao"] = ap["ident"].astype(str).str.strip().str.upper()
    else:
        ap["icao"] = ap["icao"].astype(str).str.strip().str.upper()
    uf_col = next((c for c in ["state", "iso_region", "municipality_state", "uf"] if c in ap.columns), None)
    if uf_col is None:
        return out
    ap_map = ap[["icao", uf_col]].dropna().drop_duplicates(subset=["icao"], keep="last").copy().rename(columns={uf_col: "_uf_lookup"})
    out["_icao_to_norm"] = out[cfg.icao_to].astype(str).str.strip().str.upper()
    out = out.merge(ap_map, left_on="_icao_to_norm", right_on="icao", how="left")
    if "_uf_lookup" in out.columns:
        if uf_col == "iso_region":
            out["_uf_dest"] = out["_uf_lookup"].astype(str).str.split("-").str[-1]
        else:
            out["_uf_dest"] = out["_uf_lookup"]
    out["_uf_dest"] = out["_uf_dest"].astype("string").str.strip()
    invalid = out["_uf_dest"].isin(["", "nan", "None"])
    out.loc[invalid.fillna(False), "_uf_dest"] = pd.NA
    return out


def _group_ttv(df: pd.DataFrame, group_col: str, output_col: str) -> pd.DataFrame:
    out = df.dropna(subset=[group_col]).groupby(group_col, dropna=False).agg(horas_voo=("_ttv", "sum")).reset_index().rename(columns={group_col: output_col}).sort_values("horas_voo", ascending=False)
    out["horas_voo"] = pd.to_numeric(out["horas_voo"], errors="coerce").fillna(0.0)
    return out


def _monthly_group_for_aircraft_selection(base_sel: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    return base_sel.dropna(subset=["_month_dt"]).groupby(["_month_dt", "_month"], dropna=False).agg(
        trechos=(cfg.aircraft, "size"),
        passageiros_total=("_passengers", "sum"),
        presos_total=("_prisoners", "sum"),
        carga_total_kg=("_cargo", "sum"),
        ttv_total=("_ttv", "sum"),
    ).reset_index().sort_values("_month_dt")


def _operations_summary_for_subset(df: pd.DataFrame, cfg: ColumnConfig):
    if not cfg.op_metrics:
        return {}, 0.0
    raw_values = {}
    for col in cfg.op_metrics:
        if col not in df.columns:
            continue
        value, _ = count_or_sum(df[col])
        raw_values[col] = float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])
    obs_col = "OBS (IMA + REC) (1 por dia)"
    int_col = "INT (1 por dia)"
    mob_col = "MOBILIZAÇÃO (POR NOME DA OPERAÇÃO, unidade apoiada, serviço especializado)"
    destr_err_col = "DESTRUIÇÃO + ERRADICAÇÃO (1 POR DIA CADA)"
    err_col = "ERR (NOME DA OPERAÇÃO)"
    erradicacao = raw_values.get(err_col, 0.0)
    destruicao = max(raw_values.get(destr_err_col, 0.0) - erradicacao, 0.0)
    summary = {
        "Observação": raw_values.get(obs_col, 0.0),
        "Intervenção": raw_values.get(int_col, 0.0),
        "Mobilização": raw_values.get(mob_col, 0.0),
        "Destruição": destruicao,
        "Erradicação": erradicacao,
    }
    return summary, float(sum(summary.values()))


def decode_upload(contents: str) -> bytes:
    _, content_string = contents.split(",", 1)
    return base64.b64decode(content_string)


def df_to_store(df: pd.DataFrame) -> str:
    return df.to_json(date_format="iso", orient="split")


def _restore_store_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = _drop_duplicate_columns(df).copy()
    for col in ["_date", "_month_dt"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "_year" in out.columns:
        out["_year"] = pd.to_numeric(out["_year"], errors="coerce").astype("Int64")
    for col in ["_ttv", "_passengers", "_prisoners", "_cargo"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def df_from_store(value: Optional[str]) -> pd.DataFrame:
    if not value:
        return pd.DataFrame()
    return _restore_store_dtypes(pd.read_json(io.StringIO(value), orient="split"))


def config_from_store(value: Optional[dict]) -> ColumnConfig:
    return ColumnConfig(**(value or {}))


def metric_card(title: str, value: str) -> html.Div:
    return html.Div([
        html.Div(title, style={"fontSize": "0.9rem", "color": "#666"}),
        html.Div(value, style={"fontSize": "2rem", "fontWeight": "700"}),
    ], style={"padding": "0.75rem 1rem", "border": "1px solid #ddd", "borderRadius": "10px", "background": "#fff", "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"})


def make_datatable(df: pd.DataFrame, page_size: int = 15):
    safe = df.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].astype(str)
    return dash_table.DataTable(
        data=safe.to_dict("records"),
        columns=[{"name": c, "id": c} for c in safe.columns],
        page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px", "fontFamily": "Arial", "fontSize": 13, "maxWidth": 320, "whiteSpace": "normal"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f4f6f8"},
    )


PAIR_STYLE = {
    "display": "flex",
    "gap": "12px",
    "alignItems": "stretch",
    "overflowX": "auto",
    "marginBottom": "12px",
}
PIE_GRAPH_STYLE = {"flex": "1 0 420px", "minWidth": "420px"}
BAR_GRAPH_STYLE = {"flex": "1.55 0 680px", "minWidth": "680px"}


def _graph_value_text(values) -> list[str]:
    out = []
    for value in values:
        value = float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])
        if abs(value - round(value)) < 0.05:
            out.append(_format_int(int(round(value))))
        else:
            out.append(_format_float_br(value))
    return out


def _pair_colors(labels: list[str]) -> list[str]:
    palette = qualitative.Plotly + qualitative.Dark24 + qualitative.Light24 + qualitative.Safe + qualitative.Vivid
    return [palette[i % len(palette)] for i, _ in enumerate(labels)]


def _prepare_pair_df(df: pd.DataFrame, label_col: str, value_col: str) -> pd.DataFrame:
    out = df[[label_col, value_col]].copy()
    out[label_col] = out[label_col].astype(str).str.strip()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0)
    out = out[
        out[label_col].ne("")
        & out[label_col].ne("nan")
        & out[label_col].ne("None")
    ].copy()
    return out


def _apply_bar_readability(fig: go.Figure, value_axis: str = "x") -> go.Figure:
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=13, color="#1f2937"),
        marker_line_width=0,
    )
    fig.update_layout(
        template="plotly_white",
        bargap=0.24,
        margin=dict(l=150, r=130, t=72, b=56),
        uniformtext_minsize=11,
        uniformtext_mode="show",
    )
    if value_axis == "x":
        fig.update_xaxes(rangemode="tozero", automargin=True)
        fig.update_yaxes(automargin=True)
    else:
        fig.update_yaxes(rangemode="tozero", automargin=True)
        fig.update_xaxes(automargin=True)
    return fig


def _build_linked_bar_figure(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    value_label: str,
    colors: list[str],
    hidden_labels: set[str] | None = None,
) -> go.Figure:
    hidden_labels = hidden_labels or set()
    visible_df = df[~df[label_col].astype(str).isin(hidden_labels)].copy()
    visible_df = visible_df.sort_values(value_col, ascending=True)
    labels = visible_df[label_col].astype(str).tolist()
    values = visible_df[value_col].astype(float).tolist()
    color_map = {str(label): color for label, color in zip(df[label_col].astype(str).tolist(), colors)}
    bar_colors = [color_map.get(label, "#4c78a8") for label in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        text=_graph_value_text(values),
        customdata=labels,
        marker=dict(color=bar_colors),
        hovertemplate="<b>%{y}</b><br>" + value_label + ": %{x:,.1f}<extra></extra>",
    ))
    max_value = max(values) if values else 1.0
    fig.update_layout(
        title=title,
        height=max(420, 54 + 34 * max(len(labels), 1)),
        xaxis=dict(title=value_label, range=[0, max_value * 1.22 if max_value > 0 else 1]),
        yaxis=dict(title=""),
        showlegend=False,
        meta={
            "linked_bar": {
                "labels": df[label_col].astype(str).tolist(),
                "values": df[value_col].astype(float).tolist(),
                "colors": colors,
                "title": title,
                "value_label": value_label,
            }
        },
    )
    _apply_bar_readability(fig, value_axis="x")
    if not labels:
        fig.update_layout(
            annotations=[dict(
                text="Todos os itens estao ocultos na pizza.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="#666"),
            )]
        )
    return fig


def _linked_pie_bar_pair(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    value_label: str,
    scope: str,
    pie_first: bool = True,
):
    plot_df = _prepare_pair_df(df, label_col, value_col)
    if plot_df.empty:
        return html.Div(f"Sem dados para {title}.")
    labels = plot_df[label_col].astype(str).tolist()
    colors = _pair_colors(labels)
    pie_df = plot_df[plot_df[value_col] > 0].copy()
    if pie_df.empty:
        pie_df = plot_df.copy()
    pie_colors = [colors[labels.index(str(label))] for label in pie_df[label_col].astype(str).tolist()]
    pie = go.Figure(go.Pie(
        labels=pie_df[label_col].astype(str).tolist(),
        values=pie_df[value_col].astype(float).tolist(),
        hole=0.35,
        sort=False,
        textinfo="percent",
        marker=dict(colors=pie_colors),
        hovertemplate="<b>%{label}</b><br>" + value_label + ": %{value:,.1f}<br>%{percent}<extra></extra>",
    ))
    pie.update_layout(
        template="plotly_white",
        title=f"{title} (%)",
        height=max(420, 54 + 26 * min(len(plot_df), 16)),
        margin=dict(l=20, r=20, t=72, b=20),
        legend=dict(font=dict(size=12), itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    bar = _build_linked_bar_figure(plot_df, label_col, value_col, title, value_label, colors)
    pie_graph = dcc.Graph(id={"type": "linked-pie", "scope": scope}, figure=pie, style=PIE_GRAPH_STYLE)
    bar_graph = dcc.Graph(id={"type": "linked-bar", "scope": scope}, figure=bar, style=BAR_GRAPH_STYLE)
    graphs = [pie_graph, bar_graph] if pie_first else [bar_graph, pie_graph]
    return html.Div(graphs, style=PAIR_STYLE)


def _sync_bar_from_hidden_labels(bar_figure: dict, hidden_labels: list[str] | None) -> go.Figure:
    fig = go.Figure(bar_figure)
    meta = fig.layout.meta or {}
    cfg = meta.get("linked_bar", {}) if isinstance(meta, dict) else {}
    labels = cfg.get("labels") or []
    values = cfg.get("values") or []
    colors = cfg.get("colors") or []
    if not labels or not values:
        return fig
    source = pd.DataFrame({"label": labels, "value": values})
    return _build_linked_bar_figure(
        source,
        "label",
        "value",
        cfg.get("title", ""),
        cfg.get("value_label", "Valor"),
        colors,
        set(map(str, hidden_labels or [])),
    )


def overview_layout(df: pd.DataFrame, cfg: ColumnConfig):
    if df.empty:
        return html.Div("Filtro atual retornou 0 linhas.")
    total = len(df)
    ttv = df["_ttv"] if "_ttv" in df.columns else pd.Series(dtype=float)
    ttv_sum = float(np.nansum(ttv.to_numpy())) if len(ttv) else 0.0
    ttv_fill = float(ttv.notna().mean() * 100.0) if len(ttv) else 0.0
    asa_series = df["_asa"].astype(str).str.upper().str.strip() if "_asa" in df.columns else pd.Series(dtype="object")
    asa_f = int((asa_series == "F").sum())
    asa_r = int((asa_series == "R").sum())
    cards = html.Div([
        metric_card("Trechos (linhas)", _format_int(total)),
        metric_card("Horas de voo (TTV)", _format_float_br(ttv_sum)),
        metric_card("TTV preenchido", f"{ttv_fill:.0f}%"),
        metric_card("ASA F / R", f"{asa_f} / {asa_r}"),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginBottom": "16px"})
    children = [cards]
    if "_year" in df.columns and df["_year"].notna().any():
        plot_df = df.dropna(subset=["_year"]).copy()
        plot_df["asa_plot"] = plot_df["_asa"].astype(str).str.strip().str.upper().map({"F": "Asa fixa", "R": "Asa rotativa"}).fillna("Não informado")
        yearly = plot_df.groupby(["_year", "asa_plot"], dropna=True)["_ttv"].sum().reset_index(name="ttv_h")
        fig = px.bar(yearly, x="_year", y="ttv_h", color="asa_plot", barmode="group", title="HORAS DE VOO POR ANO — ASA FIXA X ASA ROTATIVA", labels={"_year": "ANO", "ttv_h": "HORAS DE VOO (TTV)", "asa_plot": "CATEGORIA"}, text_auto=".1f")
        _apply_bar_readability(fig, value_axis="y")
        totals = plot_df.groupby("_year", dropna=True)["_ttv"].sum().reset_index(name="ttv_total")
        fig.add_scatter(x=totals["_year"], y=totals["ttv_total"], mode="lines+markers+text", name="TTV total", text=[f"{v:.1f}" for v in totals["ttv_total"]], textposition="top center")
        children.append(dcc.Graph(figure=fig))
    return html.Div(children)


def operations_layout(df: pd.DataFrame, cfg: ColumnConfig):
    if df.empty:
        return html.Div("Filtro atual retornou 0 linhas.")
    if not cfg.op_metrics:
        return html.Div("As colunas de operações não foram encontradas automaticamente.")
    obs_col = "OBS (IMA + REC) (1 por dia)"
    int_col = "INT (1 por dia)"
    mob_col = "MOBILIZAÇÃO (POR NOME DA OPERAÇÃO, unidade apoiada, serviço especializado)"
    destr_err_col = "DESTRUIÇÃO + ERRADICAÇÃO (1 POR DIA CADA)"
    err_col = "ERR (NOME DA OPERAÇÃO)"
    raw_values = {}
    for col in cfg.op_metrics:
        if col not in df.columns:
            continue
        value, _ = count_or_sum(df[col])
        raw_values[col] = float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])
    erradicacao = raw_values.get(err_col, 0.0)
    destruicao = max(raw_values.get(destr_err_col, 0.0) - erradicacao, 0.0)
    ops = pd.DataFrame([
        ("Observação", raw_values.get(obs_col, 0.0)),
        ("Intervenção", raw_values.get(int_col, 0.0)),
        ("Mobilização", raw_values.get(mob_col, 0.0)),
        ("Destruição", destruicao),
        ("Erradicação", erradicacao),
    ], columns=["tipo", "quantidade"])
    ops = ops[ops["quantidade"] > 0].copy()
    if ops.empty:
        return html.Div("Não há valores positivos de operações no filtro atual.")
    cards = html.Div([metric_card(row["tipo"], _format_int(int(round(row["quantidade"])))) for _, row in ops.iterrows()], style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "12px", "marginBottom": "16px"})
    bar = px.bar(ops.sort_values(["quantidade", "tipo"]), x="quantidade", y="tipo", orientation="h", text="quantidade", title="Tipos de operações realizadas pela CAOP")
    pie = px.pie(ops, names="tipo", values="quantidade", hole=0.35, title="Distribuição percentual dos tipos de operação")
    return html.Div([
        cards,
        _linked_pie_bar_pair(ops.sort_values(["quantidade", "tipo"]), "tipo", "quantidade", "Tipos de operacoes realizadas pela CAOP", "Quantidade", "operations-quantidade", pie_first=True),
        make_datatable(ops.rename(columns={"tipo": "Tipo de operação", "quantidade": "Quantidade"})),
    ])


def _airport_lookup(airports: pd.DataFrame) -> pd.DataFrame:
    cols = ["icao", "latitude_deg", "longitude_deg", "name", "municipality", "iso_country", "iso_region"]
    present = [col for col in cols if col in airports.columns]
    ap = airports[present].copy()
    ap["icao"] = ap["icao"].astype(str).str.strip().str.upper()
    ap["latitude_deg"] = pd.to_numeric(ap["latitude_deg"], errors="coerce")
    ap["longitude_deg"] = pd.to_numeric(ap["longitude_deg"], errors="coerce")
    for col in ["name", "municipality", "iso_country", "iso_region"]:
        if col not in ap.columns:
            ap[col] = pd.NA
    ap = ap.dropna(subset=["icao", "latitude_deg", "longitude_deg"])
    return ap.drop_duplicates(subset=["icao"], keep="first")


def _state_bounds_from_airports(airports: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    ap = _airport_lookup(airports)
    if "iso_region" not in ap.columns:
        return {}
    ap["uf"] = ap["iso_region"].astype(str).str.upper().str.extract(r"^BR-([A-Z]{2})$")[0]
    ap = ap[ap["uf"].isin(UF_LABELS)].copy()
    if ap.empty:
        return {}
    grouped = ap.groupby("uf", dropna=False).agg(
        min_lat=("latitude_deg", "min"),
        max_lat=("latitude_deg", "max"),
        min_lon=("longitude_deg", "min"),
        max_lon=("longitude_deg", "max"),
    )
    margin_deg = 0.55
    return {
        uf: (
            float(row.min_lat) - margin_deg,
            float(row.max_lat) + margin_deg,
            float(row.min_lon) - margin_deg,
            float(row.max_lon) + margin_deg,
        )
        for uf, row in grouped.iterrows()
    }


def _br_uf_from_iso_region(value) -> Optional[str]:
    text = str(value).strip().upper()
    if text.startswith("BR-") and len(text) >= 5:
        uf = text[-2:]
        return uf if uf in UF_LABELS else None
    return None


def _circle_points(lat: float, lon: float, radius_km: float = 100.0, points: int = 96) -> tuple[list[float], list[float]]:
    lat1 = np.radians(float(lat))
    lon1 = np.radians(float(lon))
    angular_distance = float(radius_km) / 6371.0088
    bearings = np.linspace(0, 2 * np.pi, points, endpoint=True)
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(angular_distance)
        + np.cos(lat1) * np.sin(angular_distance) * np.cos(bearings)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(bearings) * np.sin(angular_distance) * np.cos(lat1),
        np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2),
    )
    return np.degrees(lat2).tolist(), np.degrees(lon2).tolist()


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(76,120,168,{alpha})"
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _route_overflown_states(row, bounds: dict[str, tuple[float, float, float, float]]) -> list[str]:
    found = set()
    for state_col in ["from_state", "to_state"]:
        uf = _br_uf_from_iso_region(getattr(row, state_col, ""))
        if uf:
            found.add(uf)
    if bounds:
        if str(getattr(row, "route_kind", "")).lower() == "zone":
            try:
                lats, lons = _circle_points(float(row.zone_center_lat), float(row.zone_center_lon), 100.0, 120)
                lats = np.array(lats + [float(row.zone_center_lat)])
                lons = np.array(lons + [float(row.zone_center_lon)])
            except (TypeError, ValueError):
                lats = np.array([])
                lons = np.array([])
        else:
            lats = np.linspace(float(row.from_lat), float(row.to_lat), 80)
            lons = np.linspace(float(row.from_lon), float(row.to_lon), 80)
        for uf, (min_lat, max_lat, min_lon, max_lon) in bounds.items():
            inside = (lats >= min_lat) & (lats <= max_lat) & (lons >= min_lon) & (lons <= max_lon)
            if bool(inside.any()):
                found.add(uf)
    return sorted(found)


def _filter_routes_by_states(routes: pd.DataFrame, selected_states: list[str] | None) -> pd.DataFrame:
    selected = {str(uf).strip().upper() for uf in (selected_states or []) if str(uf).strip().upper() in UF_LABELS}
    if not selected or "overflown_states" not in routes.columns:
        return routes
    return routes[routes["overflown_states"].map(lambda states: bool(selected.intersection(states or [])))].copy()


def _prepare_aircraft_routes(df: pd.DataFrame, cfg: ColumnConfig, airports: pd.DataFrame) -> pd.DataFrame:
    needed = [cfg.aircraft, cfg.icao_from, cfg.icao_to]
    if any(not col or col not in df.columns for col in needed):
        return pd.DataFrame()
    work_cols = [cfg.aircraft, cfg.icao_from, cfg.icao_to]
    if "_ttv" in df.columns:
        work_cols.append("_ttv")
    if "_date" in df.columns:
        work_cols.append("_date")
    work = df[work_cols].copy()
    work = work.rename(columns={cfg.aircraft: "aircraft", cfg.icao_from: "from_icao", cfg.icao_to: "to_icao"})
    if "_ttv" not in work.columns:
        work["_ttv"] = 0.0
    if "_date" not in work.columns:
        work["_date"] = pd.NaT
    work["aircraft"] = work["aircraft"].astype(str).str.strip()
    work["from_icao"] = work["from_icao"].astype(str).str.strip().str.upper()
    work["to_icao"] = work["to_icao"].astype(str).str.strip().str.upper()
    work["_ttv"] = pd.to_numeric(work["_ttv"], errors="coerce").fillna(0.0)
    work["_date"] = pd.to_datetime(work["_date"], errors="coerce")
    work["unknown_from"] = work["from_icao"].eq("ZZZZ")
    work["unknown_to"] = work["to_icao"].eq("ZZZZ")
    valid = (
        work["aircraft"].ne("")
        & work["aircraft"].ne("nan")
        & work["aircraft"].ne("None")
        & work["from_icao"].str.len().eq(4)
        & work["to_icao"].str.len().eq(4)
        & work["from_icao"].ne(work["to_icao"])
        & ~(work["unknown_from"] & work["unknown_to"])
    )
    work = work[valid].copy()
    if work.empty:
        return pd.DataFrame()

    ap = _airport_lookup(airports)
    from_ap = ap.rename(columns={
        "icao": "from_icao",
        "latitude_deg": "from_lat",
        "longitude_deg": "from_lon",
        "name": "from_name",
        "municipality": "from_municipality",
        "iso_country": "from_country",
        "iso_region": "from_state",
    })
    to_ap = ap.rename(columns={
        "icao": "to_icao",
        "latitude_deg": "to_lat",
        "longitude_deg": "to_lon",
        "name": "to_name",
        "municipality": "to_municipality",
        "iso_country": "to_country",
        "iso_region": "to_state",
    })
    work = work.merge(from_ap, on="from_icao", how="left")
    work = work.merge(to_ap, on="to_icao", how="left")
    work["route_kind"] = np.where(work["unknown_from"] | work["unknown_to"], "zone", "line")
    work["zone_center_lat"] = np.where(work["unknown_from"], work["to_lat"], np.where(work["unknown_to"], work["from_lat"], np.nan))
    work["zone_center_lon"] = np.where(work["unknown_from"], work["to_lon"], np.where(work["unknown_to"], work["from_lon"], np.nan))
    work["zone_known_icao"] = np.where(work["unknown_from"], work["to_icao"], np.where(work["unknown_to"], work["from_icao"], ""))
    work["zone_unknown_side"] = np.where(work["unknown_from"], "Origem ZZZZ", np.where(work["unknown_to"], "Destino ZZZZ", ""))
    work["zone_radius_km"] = np.where(work["route_kind"].eq("zone"), 100.0, np.nan)
    work.loc[work["unknown_from"], ["from_lat", "from_lon"]] = work.loc[work["unknown_from"], ["to_lat", "to_lon"]].to_numpy()
    work.loc[work["unknown_to"], ["to_lat", "to_lon"]] = work.loc[work["unknown_to"], ["from_lat", "from_lon"]].to_numpy()
    work = work.dropna(subset=["from_lat", "from_lon", "to_lat", "to_lon"]).copy()
    if work.empty:
        return pd.DataFrame()
    group_cols = [
        "aircraft", "from_icao", "to_icao", "from_lat", "from_lon", "to_lat", "to_lon",
        "from_name", "to_name", "from_municipality", "to_municipality", "from_country", "to_country",
        "from_state", "to_state",
        "route_kind", "zone_center_lat", "zone_center_lon", "zone_known_icao", "zone_unknown_side", "zone_radius_km",
    ]
    routes = (
        work.groupby(group_cols, dropna=False)
        .agg(
            trechos=("aircraft", "size"),
            ttv_total=("_ttv", "sum"),
            date_start=("_date", "min"),
            date_end=("_date", "max"),
        )
        .reset_index()
        .sort_values(["aircraft", "trechos"], ascending=[True, False])
    )
    bounds = _state_bounds_from_airports(airports)
    routes["overflown_states"] = routes.apply(lambda row: _route_overflown_states(row, bounds), axis=1)
    routes["overflown_state_labels"] = routes["overflown_states"].map(lambda states: ", ".join(states) if states else "N/I")
    routes["route_id"] = (
        routes["aircraft"].astype(str)
        + "|"
        + routes["from_icao"].astype(str)
        + "|"
        + routes["to_icao"].astype(str)
    )
    return routes


def _route_map_center(routes: pd.DataFrame) -> dict:
    if routes.empty:
        return {"lat": -14.2, "lon": -51.9}
    lat = pd.concat([routes["from_lat"], routes["to_lat"]], ignore_index=True)
    lon = pd.concat([routes["from_lon"], routes["to_lon"]], ignore_index=True)
    return {"lat": float(lat.mean()), "lon": float(lon.mean())}


def _route_aircraft_order(routes: pd.DataFrame) -> list[str]:
    return (
        routes.groupby("aircraft", dropna=False)["trechos"]
        .sum()
        .sort_values(ascending=False)
        .index.astype(str)
        .tolist()
    )


def _build_aircraft_routes_map(routes: pd.DataFrame, map_style: str, hidden_aircraft: set[str] | None = None) -> go.Figure:
    hidden_aircraft = hidden_aircraft or set()
    aircraft_order = _route_aircraft_order(routes)
    color_map = _build_aircraft_color_map(aircraft_order)
    fig = go.Figure()
    for aircraft in aircraft_order:
        sub = routes[routes["aircraft"].astype(str) == aircraft].copy()
        if sub.empty:
            continue
        route_kind = sub["route_kind"].astype(str) if "route_kind" in sub.columns else pd.Series("line", index=sub.index)
        line_sub = sub[route_kind.ne("zone")].copy()
        zone_sub = sub[route_kind.eq("zone")].copy()
        total_trechos = int(sub["trechos"].sum())
        visible_state = "legendonly" if str(aircraft) in hidden_aircraft else True
        lat = []
        lon = []
        text = []
        route_ids = []
        for row in line_sub.itertuples(index=False):
            route_id = str(row.route_id)
            hover = (
                f"<b>{aircraft}</b><br>"
                f"{row.from_icao} -> {row.to_icao}<br>"
                f"Trechos: {int(row.trechos)}<br>"
                f"Horas de voo: {float(row.ttv_total):.1f}<br>"
                f"UFs provaveis: {row.overflown_state_labels}"
            )
            route_lats = np.linspace(float(row.from_lat), float(row.to_lat), 8).tolist()
            route_lons = np.linspace(float(row.from_lon), float(row.to_lon), 8).tolist()
            lat.extend(route_lats + [None])
            lon.extend(route_lons + [None])
            text.extend([hover] * len(route_lats) + [None])
            route_ids.extend([route_id] * len(route_lats) + [None])
        if lat:
            fig.add_trace(go.Scattermap(
                lat=lat,
                lon=lon,
                mode="lines",
                name=f"{aircraft} ({total_trechos})",
                legendgroup=aircraft,
                line=dict(color=color_map[aircraft], width=4),
                text=text,
                customdata=route_ids,
                hovertemplate="%{text}<extra></extra>",
                visible=visible_state,
            ))
        zone_lat = []
        zone_lon = []
        zone_text = []
        zone_route_ids = []
        for row in zone_sub.itertuples(index=False):
            route_id = str(row.route_id)
            radius_km = float(getattr(row, "zone_radius_km", 100.0) or 100.0)
            circle_lats, circle_lons = _circle_points(float(row.zone_center_lat), float(row.zone_center_lon), radius_km, 48)
            hover = (
                f"<b>{aircraft}</b><br>"
                f"{row.from_icao} -> {row.to_icao}<br>"
                f"Zona circular: {radius_km:.0f} km em torno de {row.zone_known_icao}<br>"
                f"{row.zone_unknown_side}<br>"
                f"Trechos: {int(row.trechos)}<br>"
                f"Horas de voo: {float(row.ttv_total):.1f}<br>"
                f"UFs provaveis: {row.overflown_state_labels}"
            )
            zone_lat.extend(circle_lats + [None])
            zone_lon.extend(circle_lons + [None])
            zone_text.extend([hover] * len(circle_lats) + [None])
            zone_route_ids.extend([route_id] * len(circle_lats) + [None])
        if zone_lat:
            fig.add_trace(go.Scattermap(
                lat=zone_lat,
                lon=zone_lon,
                mode="lines",
                fill="toself",
                fillcolor=_hex_to_rgba(color_map[aircraft], 0.20),
                name=f"{aircraft} ({total_trechos})",
                legendgroup=aircraft,
                showlegend=not bool(lat),
                line=dict(color=color_map[aircraft], width=2),
                text=zone_text,
                customdata=zone_route_ids,
                hovertemplate="%{text}<extra></extra>",
                visible=visible_state,
            ))
    if routes.empty:
        endpoints = pd.DataFrame(columns=["icao", "lat", "lon"])
    else:
        endpoints = pd.concat([
            routes[["from_icao", "from_lat", "from_lon"]].rename(columns={"from_icao": "icao", "from_lat": "lat", "from_lon": "lon"}),
            routes[["to_icao", "to_lat", "to_lon"]].rename(columns={"to_icao": "icao", "to_lat": "lat", "to_lon": "lon"}),
        ], ignore_index=True).drop_duplicates("icao")
        endpoints = endpoints[endpoints["icao"].astype(str).ne("ZZZZ")].copy()
    fig.add_trace(go.Scattermap(
        lat=endpoints["lat"],
        lon=endpoints["lon"],
        mode="markers",
        name="Aerodromos",
        showlegend=False,
        marker=dict(size=7, color="#111827", opacity=0.75),
        text=endpoints["icao"],
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(
        title="Caminhos realizados por aeronave",
        height=720,
        margin=dict(l=8, r=8, t=56, b=8),
        map=dict(style=map_style, center=_route_map_center(routes), zoom=3.4),
        legend=dict(title="Aeronaves (trechos)", itemclick="toggle", itemdoubleclick="toggleothers", groupclick="togglegroup"),
    )
    if routes.empty:
        fig.update_layout(
            annotations=[dict(
                text="Nenhuma rota visivel na selecao atual.",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=15, color="#666"),
            )]
        )
    return fig


def _format_route_date_series(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.strftime("%d/%m/%Y").fillna("N/I")


def _routes_report_df(routes: pd.DataFrame) -> pd.DataFrame:
    report = routes.copy()
    for col in ["from_municipality", "to_municipality", "from_country", "to_country", "overflown_state_labels"]:
        if col not in report.columns:
            report[col] = "N/I"
        report[col] = report[col].fillna("N/I").astype(str)
    for col in ["date_start", "date_end"]:
        if col not in report.columns:
            report[col] = pd.NaT
    if "route_id" not in report.columns:
        report["route_id"] = (
            report["aircraft"].astype(str)
            + "|"
            + report["from_icao"].astype(str)
            + "|"
            + report["to_icao"].astype(str)
        )
    if "route_kind" not in report.columns:
        report["route_kind"] = "line"
    report["Tipo"] = report["route_kind"].astype(str).map({"zone": "Zona 100 km", "line": "Linha"}).fillna("Linha")
    report["Data inicial"] = _format_route_date_series(report["date_start"])
    report["Data final"] = _format_route_date_series(report["date_end"])
    report["Trechos"] = pd.to_numeric(report["trechos"], errors="coerce").fillna(0).astype(int)
    report["Horas de voo (TTV)"] = pd.to_numeric(report["ttv_total"], errors="coerce").fillna(0.0).round(1)
    report = report.sort_values(["aircraft", "Trechos", "Horas de voo (TTV)"], ascending=[True, False, False])
    report = report.rename(columns={
        "aircraft": "Aeronave",
        "from_icao": "Origem",
        "to_icao": "Destino",
        "from_municipality": "Cidade origem",
        "to_municipality": "Cidade destino",
        "from_country": "Pais origem",
        "to_country": "Pais destino",
        "overflown_state_labels": "UFs provaveis",
    })
    cols = [
        "route_id",
        "Aeronave",
        "Origem",
        "Destino",
        "Tipo",
        "Data inicial",
        "Data final",
        "Cidade origem",
        "Cidade destino",
        "Pais origem",
        "Pais destino",
        "UFs provaveis",
        "Trechos",
        "Horas de voo (TTV)",
    ]
    return report[cols]


def _routes_report_table_from_records(records: list[dict]):
    if not records:
        return html.Div("Nenhuma linha visivel na selecao atual.")
    data = []
    for row in records:
        item = dict(row)
        route_id = item.get("route_id") or item.get("id")
        item["route_id"] = route_id
        item["id"] = route_id
        data.append(item)
    columns = [
        {"name": col, "id": col}
        for col in [
            "Aeronave",
            "Origem",
            "Destino",
            "Tipo",
            "Data inicial",
            "Data final",
            "Cidade origem",
            "Cidade destino",
            "Pais origem",
            "Pais destino",
            "UFs provaveis",
            "Trechos",
            "Horas de voo (TTV)",
        ]
    ]
    return dash_table.DataTable(
        id="routes-report-table-dyn",
        data=data,
        columns=columns,
        page_size=20,
        sort_action="native",
        filter_action="native",
        cell_selectable=True,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px", "fontFamily": "Arial", "fontSize": 13, "maxWidth": 320, "whiteSpace": "normal"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f4f6f8"},
        style_data_conditional=[{"if": {"state": "active"}, "backgroundColor": "#ffe8cc", "border": "1px solid #f59f00"}],
    )


def _routes_report_table(routes: pd.DataFrame):
    return _routes_report_table_from_records(_routes_report_df(routes).to_dict("records"))


def _routes_report_payload(routes: pd.DataFrame) -> dict:
    routes_store = routes.copy()
    for col in ["date_start", "date_end"]:
        if col in routes_store.columns:
            routes_store[col] = pd.to_datetime(routes_store[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return {
        "rows": _routes_report_df(routes).to_dict("records"),
        "routes": routes_store.to_dict("records"),
        "trace_aircraft": _route_aircraft_order(routes),
    }


def _routes_df_from_payload(report_payload: dict) -> pd.DataFrame:
    routes = pd.DataFrame((report_payload or {}).get("routes", []))
    for col in ["from_lat", "from_lon", "to_lat", "to_lon", "zone_center_lat", "zone_center_lon", "zone_radius_km", "trechos", "ttv_total"]:
        if col in routes.columns:
            routes[col] = pd.to_numeric(routes[col], errors="coerce")
    for col in ["date_start", "date_end"]:
        if col in routes.columns:
            routes[col] = pd.to_datetime(routes[col], errors="coerce")
    if "trechos" in routes.columns:
        routes["trechos"] = routes["trechos"].fillna(0).astype(int)
    if "ttv_total" in routes.columns:
        routes["ttv_total"] = routes["ttv_total"].fillna(0.0)
    return routes


def _visible_routes_from_payload(report_payload: dict, hidden_aircraft: set[str], removed_route_ids: set[str]) -> pd.DataFrame:
    routes = _routes_df_from_payload(report_payload)
    if routes.empty:
        return routes
    if hidden_aircraft and "aircraft" in routes.columns:
        routes = routes[~routes["aircraft"].astype(str).isin(hidden_aircraft)].copy()
    if removed_route_ids and "route_id" in routes.columns:
        routes = routes[~routes["route_id"].astype(str).isin(removed_route_ids)].copy()
    return routes


def _filter_route_report_records(records: list[dict], hidden_aircraft: set[str], removed_route_ids: set[str] | None = None) -> list[dict]:
    removed_route_ids = removed_route_ids or set()
    out = records
    if hidden_aircraft:
        out = [row for row in out if str(row.get("Aeronave", "")) not in hidden_aircraft]
    if removed_route_ids:
        out = [row for row in out if str(row.get("route_id", "")) not in removed_route_ids]
    return out


def _route_label_from_record(record: dict) -> str:
    return f"{record.get('Aeronave', '')}: {record.get('Origem', '')} -> {record.get('Destino', '')} ({record.get('Trechos', 0)} trechos)"


def _removed_route_options(report_payload: dict, removed_route_ids: set[str]) -> list[dict]:
    rows = (report_payload or {}).get("rows", [])
    return [
        {"label": _route_label_from_record(row), "value": str(row.get("route_id", ""))}
        for row in rows
        if str(row.get("route_id", "")) in removed_route_ids
    ]


def _route_id_from_map_click(click_data) -> Optional[str]:
    points = (click_data or {}).get("points") or []
    if not points:
        return None
    customdata = points[0].get("customdata")
    if isinstance(customdata, (list, tuple)):
        customdata = customdata[0] if customdata else None
    route_id = str(customdata).strip() if customdata is not None else ""
    return route_id or None


def _render_routes_after_state_change(report_payload: dict, map_style: str, hidden_aircraft: set[str], removed_route_ids: set[str]):
    map_routes = _visible_routes_from_payload(report_payload, set(), removed_route_ids)
    visible_records = _filter_route_report_records(report_payload.get("rows", []), hidden_aircraft, removed_route_ids)
    fig = _build_aircraft_routes_map(map_routes, map_style, hidden_aircraft=hidden_aircraft)
    removed_options = _removed_route_options(report_payload, removed_route_ids)
    removed_value = removed_options[0]["value"] if removed_options else None
    return fig, _routes_report_table_from_records(visible_records), removed_options, removed_value


def _remove_route_id_from_state(route_id: str | None, removed_data, report_payload, hidden_data, map_style):
    if not route_id or not report_payload:
        return no_update, no_update, no_update, no_update, no_update
    removed = set(map(str, (removed_data or {}).get("removed", [])))
    removed.add(str(route_id))
    hidden = set(map(str, (hidden_data or {}).get("hidden", [])))
    fig, table, options, value = _render_routes_after_state_change(report_payload, map_style or MAP_STYLES[0], hidden, removed)
    return {"removed": sorted(removed)}, fig, table, options, value


def _update_hidden_aircraft_from_restyle(restyle_data, current_hidden: list[str] | None, trace_aircraft: list[str]) -> list[str]:
    hidden = set(map(str, current_hidden or []))
    if not restyle_data or len(restyle_data) < 2:
        return sorted(hidden)
    changes, trace_indices = restyle_data[0], restyle_data[1]
    if not isinstance(changes, dict) or "visible" not in changes:
        return sorted(hidden)
    if not isinstance(trace_indices, list):
        trace_indices = [trace_indices]
    visible_values = changes.get("visible")
    if not isinstance(visible_values, list):
        visible_values = [visible_values] * len(trace_indices)
    elif len(visible_values) == 1 and len(trace_indices) > 1:
        visible_values = visible_values * len(trace_indices)

    for trace_idx, visible in zip(trace_indices, visible_values):
        try:
            aircraft = trace_aircraft[int(trace_idx)]
        except (TypeError, ValueError, IndexError):
            continue
        if not aircraft:
            continue
        if visible == "legendonly":
            hidden.add(str(aircraft))
        elif visible is True:
            hidden.discard(str(aircraft))
    return sorted(hidden)


def _trace_aircraft_from_figure(figure: dict | None) -> list[str]:
    out = []
    for trace in (figure or {}).get("data", []):
        group = trace.get("legendgroup")
        if group:
            out.append(str(group))
            continue
        name = str(trace.get("name", ""))
        if name and name != "Aerodromos":
            out.append(name.rsplit(" (", 1)[0])
        else:
            out.append("")
    return out


def map_layout(df: pd.DataFrame, cfg: ColumnConfig, airports: Optional[pd.DataFrame], mode: str, map_style: str, radius: int, clip_q: float, opacity: float, selected_states: list[str] | None = None):
    if df.empty:
        return html.Div("Filtro atual retornou 0 linhas.")
    valid_points, ignored_df, summary = analyze_icao_usage(df, cfg, airports)
    if airports is None or airports.empty:
        return html.Div("Base de aeroportos não carregada automaticamente.")
    if mode == "Caminhos por aeronave":
        routes = _prepare_aircraft_routes(df, cfg, airports)
        if routes.empty:
            return html.Div("Nao ha trechos com origem, destino e aeronave validos para desenhar caminhos.")
        routes = _filter_routes_by_states(routes, selected_states)
        if routes.empty:
            states_label = ", ".join(selected_states or [])
            return html.Div(f"Nao ha caminhos passando pelas UFs selecionadas: {states_label}.")
        fig = _build_aircraft_routes_map(routes, map_style)
        report_payload = _routes_report_payload(routes)
        cards = html.Div([
            metric_card("Trechos com rota", _format_int(int(routes["trechos"].sum()))),
            metric_card("Aeronaves", _format_int(int(routes["aircraft"].nunique()))),
            metric_card("Rotas distintas", _format_int(len(routes))),
            metric_card("UFs filtradas", ", ".join(selected_states or []) or "Todas"),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginTop": "12px"})
        ignored_table = make_datatable(ignored_df.groupby(["icao", "motivo"], dropna=False)["visitas"].sum().reset_index(), 10) if not ignored_df.empty else html.Div("Nenhum ICAO ignorado alem das regras fixas aplicadas.")
        return html.Div([
            dcc.Store(id="routes-report-store-dyn", data=report_payload),
            dcc.Store(id="routes-hidden-aircraft-store-dyn", data={"hidden": []}),
            dcc.Store(id="routes-removed-store-dyn", data={"removed": []}),
            dcc.Graph(
                id="aircraft-routes-map-dyn",
                figure=fig,
                config={"responsive": True},
                style={"height": "72vh", "minHeight": "620px", "width": "100%"},
            ),
            cards,
            html.H4("Relatorio das linhas selecionadas"),
            html.Div(_routes_report_table_from_records(report_payload["rows"]), id="routes-report-content-dyn"),
            html.Div([
                html.Button("Baixar relatorio CSV", id="btn-download-routes-csv-dyn", n_clicks=0),
                html.Button("Baixar relatorio Excel", id="btn-download-routes-xlsx-dyn", n_clicks=0),
            ], style={"display": "flex", "gap": "8px", "marginTop": "10px", "marginBottom": "10px"}),
            html.Div([
                dcc.Dropdown(id="routes-removed-dropdown-dyn", options=[], value=None, placeholder="Rotas apagadas"),
                html.Button("Retornar linha", id="btn-restore-route-dyn", n_clicks=0),
                html.Button("Retornar todas", id="btn-restore-all-routes-dyn", n_clicks=0),
            ], style={"display": "grid", "gridTemplateColumns": "minmax(260px, 1fr) auto auto", "gap": "8px", "alignItems": "center", "marginTop": "10px", "marginBottom": "16px"}),
            html.H4("ICAOs ignorados"),
            ignored_table,
        ])
    if valid_points.empty:
        return html.Div("Nenhum ICAO válido encontrou coordenadas na base de aeroportos.")
    center = get_map_center(valid_points)
    zoom_default = 3.4 if (valid_points["categoria"] == "Exterior").any() else 4.0
    if mode == "Somente pontos (todos iguais)":
        points_df = valid_points.copy()
        cat_order = ["Brasil", "Exterior", "CAOP"]
        points_df["categoria"] = pd.Categorical(points_df["categoria"], categories=cat_order, ordered=True)
        points_df["_marker_size"] = 10
        fig = px.scatter_map(points_df, lat="latitude_deg", lon="longitude_deg", color="categoria", size="_marker_size", size_max=10, zoom=zoom_default, center=center, map_style=map_style, hover_name="icao", hover_data={"visitas": True, "iso_country": True, "categoria": True, "latitude_deg": False, "longitude_deg": False, "_marker_size": False}, color_discrete_map=POINT_COLORS, category_orders={"categoria": cat_order}, title="PONTOS VISITADOS (origem e destino)")
        fig.update_traces(marker=dict(opacity=0.95))
    else:
        zmax = float(np.quantile(valid_points["visitas"].to_numpy(), float(clip_q))) if len(valid_points) else 1.0
        zmax = max(zmax, 1.0)
        fig = px.density_map(valid_points, lat="latitude_deg", lon="longitude_deg", z="visitas", radius=int(radius), zoom=zoom_default, center=center, map_style=map_style, color_continuous_scale="Turbo", range_color=(0, zmax), opacity=float(opacity), title="ZONAS DE CALOR POR AERODROMO VISITADO (origem e destino)")
        if mode == "Topologia (zonas + contornos)":
            cell_km = max(20.0, min(120.0, float(radius) * 1.3))
            grid = build_grid(valid_points, lat_col="latitude_deg", lon_col="longitude_deg", w_col="visitas", cell_km=cell_km)
            for trace in contour_traces_from_grid(grid, levels=14, clip_q=float(clip_q)):
                fig.add_trace(trace)
    cards = html.Div([
        metric_card("Visitas mapeadas", _format_int(summary.get("validos", 0))),
        metric_card("Visitas ignoradas", _format_int(summary.get("ignorados", 0))),
        metric_card("Visitas no exterior", _format_int(int(valid_points.loc[valid_points["categoria"] == "Exterior", "visitas"].sum()) if not valid_points.empty else 0)),
        metric_card("Visitas em aeroportos CAOP", _format_int(int(valid_points.loc[valid_points["categoria"] == "CAOP", "visitas"].sum()) if not valid_points.empty else 0)),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginTop": "12px"})
    ignored_table = make_datatable(ignored_df.groupby(["icao", "motivo"], dropna=False)["visitas"].sum().reset_index(), 10) if not ignored_df.empty else html.Div("Nenhum ICAO ignorado além das regras fixas aplicadas.")
    fig.update_layout(
        height=720,
        margin=dict(l=8, r=8, t=56, b=8),
    )
    return html.Div([
        dcc.Graph(
            figure=fig,
            config={"responsive": True},
            style={"height": "72vh", "minHeight": "620px", "width": "100%"},
        ),
        cards,
        html.H4("ICAOs ignorados"),
        ignored_table,
    ])


def demandantes_layout(df: pd.DataFrame, cfg: ColumnConfig, selected_aircraft: list[str] | None):
    if not cfg.demandante or cfg.demandante not in df.columns:
        return html.Div("A coluna de demandante não foi encontrada automaticamente.")
    if df.empty:
        return html.Div("Filtro atual retornou 0 linhas.")
    base = df.copy()
    base[cfg.demandante] = base[cfg.demandante].astype(str).str.strip()
    base = base[base[cfg.demandante].ne("") & base[cfg.demandante].ne("nan") & base[cfg.demandante].ne("None")].copy()
    if cfg.aircraft and cfg.aircraft in base.columns and selected_aircraft:
        base[cfg.aircraft] = base[cfg.aircraft].astype(str).str.strip()
        base = base[base[cfg.aircraft].isin(selected_aircraft)].copy()
    if base.empty:
        return html.Div("Não há demandantes válidos no filtro atual.")
    agg = base.groupby(cfg.demandante, dropna=False).agg(trechos=(cfg.demandante, "size"), passageiros_total=("_passengers", "sum"), carga_total_kg=("_cargo", "sum")).reset_index().rename(columns={cfg.demandante: "Demandante"}).sort_values(["trechos", "passageiros_total", "carga_total_kg"], ascending=[False, False, False])
    top = agg.head(min(20, len(agg))).copy()
    cards = html.Div([
        metric_card("Demandantes no filtro", _format_int(base[cfg.demandante].nunique())),
        metric_card("Trechos", _format_int(len(base))),
        metric_card("Passageiros", _format_int(int(base["_passengers"].sum()))),
        metric_card("Carga (kg)", _format_float_br(float(base["_cargo"].sum()))),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px", "marginBottom": "16px"})
    def pair(value_col, title, label):
        return _linked_pie_bar_pair(top, "Demandante", value_col, title, label, f"demandantes-{value_col}", pie_first=False)
    children = [cards]
    if selected_aircraft:
        children.append(html.Div(f"Filtro ativo de aeronaves: {', '.join(selected_aircraft)}", style={"marginBottom": "12px", "color": "#666"}))
    children += [pair("trechos", "Trechos por demandante", "Trechos"), pair("passageiros_total", "Passageiros transportados por demandante", "Passageiros"), pair("carga_total_kg", "Carga transportada por demandante", "Carga (kg)"), make_datatable(top.rename(columns={"trechos": "Trechos", "passageiros_total": "Passageiros", "carga_total_kg": "Carga (kg)"}))]
    return html.Div(children)


def aircraft_layout(df: pd.DataFrame, cfg: ColumnConfig, airports: Optional[pd.DataFrame], general_aircraft: list[str] | None, detail_aircraft: list[str] | None):
    if not cfg.aircraft or cfg.aircraft not in df.columns:
        return html.Div("A coluna da aeronave não foi encontrada automaticamente.")
    if df.empty:
        return html.Div("Filtro atual retornou 0 linhas.")
    base = _prepare_aircraft_base(df, cfg)
    if base.empty:
        return html.Div("Não há aeronaves válidas no filtro atual.")
    base = _enrich_with_uf_from_airports(base, cfg, airports)
    grouped = _aircraft_grouped(base, cfg)
    aircraft_options = grouped[cfg.aircraft].astype(str).tolist()
    general_aircraft = _normalize_aircraft_selection(general_aircraft, aircraft_options, aircraft_options)
    detail_aircraft = _normalize_aircraft_selection(detail_aircraft, aircraft_options, aircraft_options[:1])
    if not general_aircraft:
        return html.Div("Selecione ao menos uma aeronave para visualizar a aba.")
    base_general = base[base[cfg.aircraft].astype(str).isin(general_aircraft)].copy()
    grouped_general = _aircraft_grouped(base_general, cfg)
    if grouped_general.empty:
        return html.Div("NÃ£o hÃ¡ dados para as aeronaves selecionadas no filtro atual.")
    total_aeronaves = grouped_general[cfg.aircraft].nunique()
    total_trechos = int(grouped_general["trechos"].sum())
    total_ttv = float(grouped_general["ttv_total"].sum())
    total_passag = float(grouped_general["passageiros_total"].sum())
    total_presos = float(grouped_general["presos_total"].sum())
    total_carga = float(grouped_general["carga_total_kg"].sum())
    cards = html.Div([
        metric_card("Aeronaves no filtro", _format_int(total_aeronaves)),
        metric_card("Trechos", _format_int(total_trechos)),
        metric_card("Horas de voo (TTV)", _format_float_br(total_ttv)),
        metric_card("Passageiros", _format_int(int(total_passag))),
        metric_card("Presos", _format_int(int(total_presos))),
        metric_card("Carga (kg)", _format_float_br(total_carga)),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(6, 1fr)", "gap": "12px", "marginBottom": "16px"})
    top_n = min(10, max(1, len(grouped_general)))
    top_ttv = grouped_general.nlargest(top_n, "ttv_total")[[cfg.aircraft, "ttv_total"]].copy()
    top_pass = grouped_general.nlargest(top_n, "passageiros_total")[[cfg.aircraft, "passageiros_total"]].copy()
    top_carga = grouped_general.nlargest(top_n, "carga_total_kg")[[cfg.aircraft, "carga_total_kg"]].copy()
    def pair(df_top, label_col, value_col, title, value_label):
        scope = f"aircraft-{abs(hash((str(label_col), str(value_col), str(title))))}"
        return _linked_pie_bar_pair(df_top, label_col, value_col, title, value_label, scope, pie_first=True)
    children = [html.H3("1) Visão geral da frota"), cards, pair(top_ttv, cfg.aircraft, "ttv_total", "Horas de voo por aeronave", "Horas de voo (TTV)"), pair(top_pass, cfg.aircraft, "passageiros_total", "Passageiros transportados por aeronave", "Passageiros"), pair(top_carga, cfg.aircraft, "carga_total_kg", "Carga transportada por aeronave", "Carga (kg)")]
    if "_uf_dest" in base_general.columns:
        uf_df = _group_ttv(base_general, "_uf_dest", "UF").head(top_n)
        if not uf_df.empty:
            children.append(pair(uf_df, "UF", "horas_voo", "Unidades da federação atendidas", "Horas de voo (TTV)"))
    if "_nat_label" in base_general.columns:
        nat_df = _group_ttv(base_general, "_nat_label", "Natureza da missão")
        if not nat_df.empty:
            children.append(pair(nat_df, "Natureza da missão", "horas_voo", "Natureza da missão", "Horas de voo (TTV)"))
    if "_espec_label" in base_general.columns:
        espec_df = _group_ttv(base_general, "_espec_label", "Especificação da missão")
        if not espec_df.empty:
            children.append(pair(espec_df, "Especificação da missão", "horas_voo", "Especificação da missão", "Horas de voo (TTV)"))
    children.append(make_datatable(grouped_general.rename(columns={cfg.aircraft: "Aeronave", "asa_label": "Tipo", "trechos": "Trechos", "ttv_total": "Horas de voo (TTV)", "passageiros_total": "Passageiros", "presos_total": "Presos", "carga_total_kg": "Carga (kg)"})))
    children.append(html.H3("2) Foco em uma ou mais aeronaves"))
    sub = base[base[cfg.aircraft].astype(str).isin(detail_aircraft)].copy()
    monthly = _monthly_group_for_aircraft_selection(sub, cfg)
    _, total_ops_exec = _operations_summary_for_subset(sub, cfg)
    detail_cards = html.Div([
        metric_card("Aeronaves selecionadas", _format_int(len(detail_aircraft))),
        metric_card("Trechos", _format_int(len(sub))),
        metric_card("Passageiros", _format_int(int(sub["_passengers"].sum()))),
        metric_card("Presos", _format_int(int(sub["_prisoners"].sum()))),
        metric_card("Carga (kg)", _format_float_br(float(sub["_cargo"].sum()))),
        metric_card("Operações executadas", _format_int(int(round(total_ops_exec)))),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(6, 1fr)", "gap": "12px", "marginBottom": "16px"})
    children.append(detail_cards)
    if not monthly.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["_month"], y=monthly["passageiros_total"], mode="lines+markers", name="Passageiros", line=dict(width=4, shape="spline", smoothing=1.0), marker=dict(size=8), yaxis="y1"))
        fig.add_trace(go.Scatter(x=monthly["_month"], y=monthly["presos_total"], mode="lines+markers", name="Presos", line=dict(width=4, shape="spline", smoothing=1.0), marker=dict(size=8), yaxis="y1"))
        fig.add_trace(go.Scatter(x=monthly["_month"], y=monthly["carga_total_kg"], mode="lines+markers", name="Carga (kg)", line=dict(width=4, shape="spline", smoothing=1.0), marker=dict(size=8), yaxis="y2"))
        fig.add_trace(go.Scatter(x=monthly["_month"], y=monthly["ttv_total"], mode="lines+markers", name="Horas de voo (TTV)", line=dict(width=4, shape="spline", smoothing=1.0), marker=dict(size=8), yaxis="y1"))
        left_max = max(1, float(monthly[["passageiros_total", "presos_total", "ttv_total"]].fillna(0).to_numpy().max()))
        right_max = max(1, float(monthly["carga_total_kg"].fillna(0).max()))
        fig.update_layout(template="plotly_white", title=f"Evolução mensal consolidada — {', '.join(detail_aircraft)}", xaxis=dict(title="Mês", tickangle=-35), yaxis=dict(title="Passageiros / Presos / Horas de voo", side="left", showgrid=True, range=[0, left_max * 1.10], rangemode="tozero"), yaxis2=dict(title="Carga (kg)", overlaying="y", side="right", showgrid=False, range=[0, right_max * 1.10], rangemode="tozero"), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), height=620)
        children.append(dcc.Graph(figure=fig))
        children.append(make_datatable(monthly.rename(columns={"_month": "Mês", "trechos": "Trechos", "passageiros_total": "Passageiros", "presos_total": "Presos", "carga_total_kg": "Carga (kg)", "ttv_total": "Horas de voo (TTV)"})))
    children.append(html.H3("3) Disponibilidade e uso mensal das aeronaves"))
    fleet_monthly = base.dropna(subset=["_month_dt"]).groupby([cfg.aircraft, "_asa", "_month_dt", "_month"], dropna=False).agg(ttv_total=("_ttv", "sum")).reset_index().sort_values(["_month_dt", cfg.aircraft])
    if not fleet_monthly.empty:
        fig_line = go.Figure()
        color_map = _build_aircraft_color_map(aircraft_options)
        for ac in aircraft_options:
            sub_ac = fleet_monthly[fleet_monthly[cfg.aircraft].astype(str) == ac].copy()
            if sub_ac.empty:
                continue
            asa_tipo = str(sub_ac["_asa"].dropna().astype(str).iloc[0]) if sub_ac["_asa"].dropna().shape[0] else ""
            dash_style = "solid" if asa_tipo == "F" else "dash" if asa_tipo == "R" else "dot"
            marker_symbol = "circle" if asa_tipo == "F" else "diamond" if asa_tipo == "R" else "square"
            asa_label = "Asa fixa" if asa_tipo == "F" else "Asa rotativa" if asa_tipo == "R" else "Não informado"
            fig_line.add_trace(go.Scatter(x=sub_ac["_month"], y=sub_ac["ttv_total"], mode="lines+markers", name=f"{ac} ({asa_label})", line=dict(width=4, dash=dash_style, color=color_map[ac], shape="spline", smoothing=1.0), marker=dict(size=7, color=color_map[ac], symbol=marker_symbol)))
        fig_line.update_layout(template="plotly_white", title="Horas de voo por mês e por aeronave", xaxis_title="Mês", yaxis_title="Horas de voo (TTV)", xaxis_tickangle=-35, height=560)
        children.append(dcc.Graph(figure=fig_line))
        pivot, month_labels, heatmap_title = _prepare_aircraft_availability_matrix(base=base, cfg=cfg, aircraft_order=aircraft_options)
        children.append(dcc.Graph(figure=_build_availability_heatmap(pivot, month_labels, heatmap_title)))
    return html.Div(children)


def debug_layout(df: pd.DataFrame, cfg: ColumnConfig):
    return html.Div([html.Div(f"Linhas filtradas: {len(df)}"), html.Pre(str(cfg)), make_datatable(df.head(80))])


def _download_filename(extension: str, prefix: str = "dados_filtrados") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}.{extension}"


def _filtered_download_df(df_json, cfg_json, years, asa_mode, exclude_caop) -> pd.DataFrame:
    if not df_json or not cfg_json:
        return pd.DataFrame()
    df = df_from_store(df_json)
    cfg = config_from_store(cfg_json)
    return apply_filters(df, cfg, years or [], asa_mode or "Todas", "exclude" in (exclude_caop or []))


def _csv_download_payload(df: pd.DataFrame, prefix: str = "dados_filtrados") -> dict:
    csv_text = df.to_csv(index=False, sep=";")
    return {
        "content": "\ufeff" + csv_text,
        "filename": _download_filename("csv", prefix),
        "type": "text/csv",
    }


def _xlsx_download_payload(df: pd.DataFrame, prefix: str = "dados_filtrados", sheet_name: str = "dados_filtrados") -> dict:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return dcc.send_bytes(buffer.getvalue(), _download_filename("xlsx", prefix))


def _routes_report_download_df(report_payload, hidden_data, removed_data) -> pd.DataFrame:
    if not report_payload:
        return pd.DataFrame()
    hidden = set(map(str, (hidden_data or {}).get("hidden", [])))
    removed = set(map(str, (removed_data or {}).get("removed", [])))
    records = _filter_route_report_records(report_payload.get("rows", []), hidden, removed)
    out = pd.DataFrame(records)
    for col in ["route_id", "id"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


external_stylesheets = [{"href": "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap", "rel": "stylesheet"}]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Store(id="df-store"),
    dcc.Store(id="cfg-store"),
    dcc.Download(id="download-filtered-data"),
    dcc.Download(id="download-routes-report"),
    html.H1("CAOP - DASHBOARD ESTATÍSTICO (Dash)"),
    html.Div([
        html.Div([
            html.H3("Arquivo"),
            dcc.Upload(id="upload-excel", children=html.Div(["Arraste ou clique para enviar o Excel"]), style={"padding": "16px", "border": "1px dashed #999", "borderRadius": "8px", "textAlign": "center"}),
            html.Div(id="upload-status", style={"marginTop": "8px", "color": "#666"}),
            dcc.Dropdown(id="sheet-dropdown", placeholder="Escolha a aba de dados"),
        ], style={"display": "grid", "gap": "10px"}),
        html.Div([
            html.H3("Filtros globais"),
            dcc.Dropdown(id="years-dropdown", multi=True, placeholder="Filtro de anos"),
            html.Div([
                html.Button("Último ano", id="btn-last-year", n_clicks=0),
                html.Button("Últimos 2", id="btn-last-two", n_clicks=0),
            ], style={"display": "flex", "gap": "8px"}),
            dcc.Dropdown(id="asa-dropdown", options=[{"label": x, "value": x} for x in ["Todas", "Asa fixa (F)", "Asa rotativa (R)"]], value="Todas", clearable=False),
            dcc.Checklist(id="exclude-caop-check", options=[{"label": "Excluir demandante CAOP", "value": "exclude"}], value=[]),
            html.Div([
                html.Button("Baixar CSV", id="btn-download-csv", n_clicks=0),
                html.Button("Baixar Excel", id="btn-download-xlsx", n_clicks=0),
            ], style={"display": "flex", "gap": "8px"}),
        ], style={"display": "grid", "gap": "10px"}),
        html.Div([
            html.H3("Mapa"),
            dcc.RadioItems(id="map-mode", options=[{"label": x, "value": x} for x in MAP_MODES], value=MAP_MODES[0]),
            dcc.Dropdown(id="map-style", options=[{"label": x, "value": x} for x in MAP_STYLES], value=MAP_STYLES[0], clearable=False),
            dcc.Dropdown(id="map-state-filter", options=[{"label": label, "value": uf} for uf, label in UF_LABELS.items()], value=[], multi=True, placeholder="Filtrar caminhos por UF sobrevoada"),
            dcc.Slider(id="map-radius", min=10, max=90, step=5, value=55),
            dcc.Slider(id="map-clipq", min=0.90, max=0.995, step=0.005, value=0.98),
            dcc.Slider(id="map-opacity", min=0.30, max=1.0, step=0.05, value=0.90),
        ], style={"display": "grid", "gap": "10px"}),
    ], style={"display": "grid", "gridTemplateColumns": "1.2fr 1fr 1fr", "gap": "20px", "marginBottom": "20px"}),
    dcc.Tabs(id="main-tabs", value="overview", children=[
        dcc.Tab(label="Visão geral", value="overview"),
        dcc.Tab(label="Operações", value="operations"),
        dcc.Tab(label="Mapa", value="map"),
        dcc.Tab(label="Demandantes", value="demandantes"),
        dcc.Tab(label="Aeronaves", value="aircraft"),
        dcc.Tab(label="Dados", value="debug"),
    ]),
    html.Div(id="tab-content", style={"marginTop": "16px"}),
], style={"fontFamily": "Inter, Arial, sans-serif", "padding": "20px", "background": "#f7f8fa"})


@app.callback(
    Output({"type": "linked-bar", "scope": MATCH}, "figure"),
    Input({"type": "linked-pie", "scope": MATCH}, "relayoutData"),
    State({"type": "linked-bar", "scope": MATCH}, "figure"),
    prevent_initial_call=True,
)
def sync_pie_hidden_labels_to_bar(pie_relayout, bar_figure):
    if not pie_relayout or "hiddenlabels" not in pie_relayout:
        return no_update
    return _sync_bar_from_hidden_labels(bar_figure, pie_relayout.get("hiddenlabels"))


@app.callback(
    Output("routes-hidden-aircraft-store-dyn", "data"),
    Output("routes-report-content-dyn", "children"),
    Input("aircraft-routes-map-dyn", "restyleData"),
    State("routes-hidden-aircraft-store-dyn", "data"),
    State("routes-report-store-dyn", "data"),
    State("routes-removed-store-dyn", "data"),
    State("aircraft-routes-map-dyn", "figure"),
    prevent_initial_call=True,
)
def sync_route_legend_to_report(restyle_data, hidden_data, report_payload, removed_data, current_figure):
    if not report_payload:
        return no_update, no_update
    hidden = _update_hidden_aircraft_from_restyle(
        restyle_data,
        (hidden_data or {}).get("hidden", []),
        _trace_aircraft_from_figure(current_figure),
    )
    removed = set(map(str, (removed_data or {}).get("removed", [])))
    visible_records = _filter_route_report_records(report_payload.get("rows", []), set(hidden), removed)
    return {"hidden": hidden}, _routes_report_table_from_records(visible_records)


@app.callback(
    Output("routes-removed-store-dyn", "data", allow_duplicate=True),
    Output("aircraft-routes-map-dyn", "figure", allow_duplicate=True),
    Output("routes-report-content-dyn", "children", allow_duplicate=True),
    Output("routes-removed-dropdown-dyn", "options", allow_duplicate=True),
    Output("routes-removed-dropdown-dyn", "value", allow_duplicate=True),
    Input("aircraft-routes-map-dyn", "clickData"),
    State("routes-removed-store-dyn", "data"),
    State("routes-report-store-dyn", "data"),
    State("routes-hidden-aircraft-store-dyn", "data"),
    State("map-style", "value"),
    prevent_initial_call=True,
)
def remove_route_from_map_click(click_data, removed_data, report_payload, hidden_data, map_style):
    route_id = _route_id_from_map_click(click_data)
    return _remove_route_id_from_state(route_id, removed_data, report_payload, hidden_data, map_style)


@app.callback(
    Output("routes-removed-store-dyn", "data", allow_duplicate=True),
    Output("aircraft-routes-map-dyn", "figure", allow_duplicate=True),
    Output("routes-report-content-dyn", "children", allow_duplicate=True),
    Output("routes-removed-dropdown-dyn", "options", allow_duplicate=True),
    Output("routes-removed-dropdown-dyn", "value", allow_duplicate=True),
    Input("routes-report-table-dyn", "active_cell"),
    State("routes-report-table-dyn", "data"),
    State("routes-removed-store-dyn", "data"),
    State("routes-report-store-dyn", "data"),
    State("routes-hidden-aircraft-store-dyn", "data"),
    State("map-style", "value"),
    prevent_initial_call=True,
)
def remove_route_from_report(active_cell, table_rows, removed_data, report_payload, hidden_data, map_style):
    if not active_cell or not report_payload:
        return no_update, no_update, no_update, no_update, no_update
    route_id = active_cell.get("row_id")
    if not route_id:
        try:
            route_id = (table_rows or [])[int(active_cell.get("row", -1))].get("route_id")
        except (TypeError, ValueError, IndexError, AttributeError):
            route_id = None
    if not route_id:
        return no_update, no_update, no_update, no_update, no_update

    removed = set(map(str, (removed_data or {}).get("removed", [])))
    removed.add(str(route_id))
    hidden = set(map(str, (hidden_data or {}).get("hidden", [])))
    fig, table, options, value = _render_routes_after_state_change(report_payload, map_style or MAP_STYLES[0], hidden, removed)
    return {"removed": sorted(removed)}, fig, table, options, value


@app.callback(
    Output("routes-removed-store-dyn", "data", allow_duplicate=True),
    Output("aircraft-routes-map-dyn", "figure", allow_duplicate=True),
    Output("routes-report-content-dyn", "children", allow_duplicate=True),
    Output("routes-removed-dropdown-dyn", "options", allow_duplicate=True),
    Output("routes-removed-dropdown-dyn", "value", allow_duplicate=True),
    Input("btn-restore-route-dyn", "n_clicks"),
    Input("btn-restore-all-routes-dyn", "n_clicks"),
    State("routes-removed-dropdown-dyn", "value"),
    State("routes-removed-store-dyn", "data"),
    State("routes-report-store-dyn", "data"),
    State("routes-hidden-aircraft-store-dyn", "data"),
    State("map-style", "value"),
    prevent_initial_call=True,
)
def restore_removed_routes(n_one, n_all, selected_route_id, removed_data, report_payload, hidden_data, map_style):
    if not report_payload:
        return no_update, no_update, no_update, no_update, no_update
    removed = set(map(str, (removed_data or {}).get("removed", [])))
    trig = callback_context.triggered_id
    if trig == "btn-restore-all-routes-dyn":
        removed.clear()
    elif trig == "btn-restore-route-dyn" and selected_route_id:
        removed.discard(str(selected_route_id))
    else:
        return no_update, no_update, no_update, no_update, no_update

    hidden = set(map(str, (hidden_data or {}).get("hidden", [])))
    fig, table, options, value = _render_routes_after_state_change(report_payload, map_style or MAP_STYLES[0], hidden, removed)
    return {"removed": sorted(removed)}, fig, table, options, value


@app.callback(
    Output("sheet-dropdown", "options"),
    Output("sheet-dropdown", "value"),
    Output("upload-status", "children"),
    Input("upload-excel", "contents"),
    prevent_initial_call=True,
)
def update_sheets(contents):
    if not contents:
        return [], None, ""
    try:
        file_bytes = decode_upload(contents)
        sheets = list_sheets(file_bytes)
        default = DEFAULT_SHEET if DEFAULT_SHEET in sheets else (sheets[0] if sheets else None)
        return [{"label": s, "value": s} for s in sheets], default, "Arquivo carregado. Selecione a aba de dados."
    except Exception as e:
        return [], None, f"Erro ao ler o arquivo: {e}"


@app.callback(
    Output("df-store", "data"),
    Output("cfg-store", "data"),
    Output("years-dropdown", "options"),
    Output("years-dropdown", "value"),
    Input("upload-excel", "contents"),
    Input("sheet-dropdown", "value"),
    prevent_initial_call=True,
)
def parse_file(contents, sheet_name):
    if not contents or not sheet_name:
        return None, None, [], []
    file_bytes = decode_upload(contents)
    loaded = load_total_table(file_bytes, sheet_name=sheet_name)
    if isinstance(loaded, tuple):
        df_raw, inferred_cfg = loaded[0], loaded[1]
    else:
        df_raw, inferred_cfg = loaded, infer_columns(loaded)
    df_raw = _drop_duplicate_columns(df_raw)
    column_cfg = build_fixed_config(df_raw, inferred_cfg)
    df = apply_column_overrides(df_raw, column_cfg)
    years = get_year_options(df)
    return df_to_store(df), asdict(column_cfg), [{"label": str(y), "value": int(y)} for y in years], years


@app.callback(
    Output("years-dropdown", "value", allow_duplicate=True),
    Input("btn-last-year", "n_clicks"),
    Input("btn-last-two", "n_clicks"),
    State("years-dropdown", "options"),
    prevent_initial_call=True,
)
def quick_years(n1, n2, options):
    if not options:
        return []
    years = [opt["value"] for opt in options]
    trig = callback_context.triggered_id
    if trig == "btn-last-year":
        return [years[-1]] if years else []
    if trig == "btn-last-two":
        return years[-2:] if len(years) >= 2 else years
    return no_update


@app.callback(
    Output("download-filtered-data", "data"),
    Input("btn-download-csv", "n_clicks"),
    Input("btn-download-xlsx", "n_clicks"),
    State("df-store", "data"),
    State("cfg-store", "data"),
    State("years-dropdown", "value"),
    State("asa-dropdown", "value"),
    State("exclude-caop-check", "value"),
    prevent_initial_call=True,
)
def download_filtered_data(n_csv, n_xlsx, df_json, cfg_json, years, asa_mode, exclude_caop):
    df_filtered = _filtered_download_df(df_json, cfg_json, years, asa_mode, exclude_caop)
    if df_filtered.empty:
        return no_update
    trig = callback_context.triggered_id
    if trig == "btn-download-csv":
        return _csv_download_payload(df_filtered)
    if trig == "btn-download-xlsx":
        return _xlsx_download_payload(df_filtered)
    return no_update


@app.callback(
    Output("download-routes-report", "data"),
    Input("btn-download-routes-csv-dyn", "n_clicks"),
    Input("btn-download-routes-xlsx-dyn", "n_clicks"),
    State("routes-report-store-dyn", "data"),
    State("routes-hidden-aircraft-store-dyn", "data"),
    State("routes-removed-store-dyn", "data"),
    prevent_initial_call=True,
)
def download_routes_report(n_csv, n_xlsx, report_payload, hidden_data, removed_data):
    report_df = _routes_report_download_df(report_payload, hidden_data, removed_data)
    if report_df.empty:
        return no_update
    trig = callback_context.triggered_id
    if trig == "btn-download-routes-csv-dyn":
        return _csv_download_payload(report_df, prefix="relatorio_rotas")
    if trig == "btn-download-routes-xlsx-dyn":
        return _xlsx_download_payload(report_df, prefix="relatorio_rotas", sheet_name="relatorio_rotas")
    return no_update


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("df-store", "data"),
    Input("cfg-store", "data"),
    Input("years-dropdown", "value"),
    Input("asa-dropdown", "value"),
    Input("exclude-caop-check", "value"),
    Input("map-mode", "value"),
    Input("map-style", "value"),
    Input("map-state-filter", "value"),
    Input("map-radius", "value"),
    Input("map-clipq", "value"),
    Input("map-opacity", "value"),
)
def render_tab(tab, df_json, cfg_json, years, asa_mode, exclude_caop, map_mode, map_style, map_states, radius, clip_q, opacity):
    if not df_json or not cfg_json:
        return html.Div("Faça upload do Excel para começar.")
    df = df_from_store(df_json)
    cfg = config_from_store(cfg_json)
    df_filtered = apply_filters(df, cfg, years or [], asa_mode or "Todas", "exclude" in (exclude_caop or []))
    airports = None
    try:
        airports = load_airports_cached(data_dir="data")
    except Exception:
        airports = None
    if tab == "overview":
        return overview_layout(df_filtered, cfg)
    if tab == "operations":
        return operations_layout(df_filtered, cfg)
    if tab == "map":
        return map_layout(df_filtered, cfg, airports, map_mode, map_style, radius, clip_q, opacity, map_states or [])
    if tab == "demandantes":
        ac_opts = []
        if cfg.aircraft and cfg.aircraft in df_filtered.columns:
            ac_opts = sorted(df_filtered[cfg.aircraft].astype(str).replace(["nan", "None"], "").loc[lambda s: s.ne("")].unique().tolist())
        return html.Div([
            html.Div([
                html.Label("Filtrar estatísticas de demandantes por aeronave"),
                dcc.Dropdown(id="demandantes-aircraft-filter-dyn", options=[{"label": x, "value": x} for x in ac_opts], value=ac_opts, multi=True),
            ], style={"marginBottom": "16px"}),
            html.Div(demandantes_layout(df_filtered, cfg, ac_opts), id="demandantes-content-dyn")
        ])
    if tab == "aircraft":
        ac_opts = _aircraft_options(df_filtered, cfg)
        return html.Div([
            html.Div([
                html.Label("Escolha uma ou mais aeronaves para a visão geral"),
                dcc.Dropdown(id="aircraft-general-dyn", options=[{"label": x, "value": x} for x in ac_opts], value=ac_opts, multi=True),
            ], style={"marginBottom": "12px"}),
            html.Div([
                html.Label("Escolha uma ou mais aeronaves para o foco detalhado"),
                dcc.Dropdown(id="aircraft-detail-dyn", options=[{"label": x, "value": x} for x in ac_opts], value=ac_opts[:1], multi=True),
            ], style={"marginBottom": "12px"}),
            html.Div(aircraft_layout(df_filtered, cfg, airports, ac_opts, ac_opts[:1]), id="aircraft-content-dyn")
        ])
    return debug_layout(df_filtered, cfg)


@app.callback(
    Output("demandantes-content-dyn", "children"),
    Input("demandantes-aircraft-filter-dyn", "value"),
    State("df-store", "data"),
    State("cfg-store", "data"),
    State("years-dropdown", "value"),
    State("asa-dropdown", "value"),
    State("exclude-caop-check", "value"),
    prevent_initial_call=True,
)
def render_demandantes_dynamic(selected_aircraft, df_json, cfg_json, years, asa_mode, exclude_caop):
    df = df_from_store(df_json)
    cfg = config_from_store(cfg_json)
    df_filtered = apply_filters(df, cfg, years or [], asa_mode or "Todas", "exclude" in (exclude_caop or []))
    return demandantes_layout(df_filtered, cfg, selected_aircraft or [])


@app.callback(
    Output("aircraft-content-dyn", "children"),
    Input("aircraft-general-dyn", "value"),
    Input("aircraft-detail-dyn", "value"),
    State("df-store", "data"),
    State("cfg-store", "data"),
    State("years-dropdown", "value"),
    State("asa-dropdown", "value"),
    State("exclude-caop-check", "value"),
    prevent_initial_call=True,
)
def render_aircraft_dynamic(general_ac, detail_ac, df_json, cfg_json, years, asa_mode, exclude_caop):
    df = df_from_store(df_json)
    cfg = config_from_store(cfg_json)
    df_filtered = apply_filters(df, cfg, years or [], asa_mode or "Todas", "exclude" in (exclude_caop or []))
    airports = None
    try:
        airports = load_airports_cached(data_dir="data")
    except Exception:
        airports = None
    return aircraft_layout(df_filtered, cfg, airports, general_ac, detail_ac)


if __name__ == "__main__":
    app.run(debug=True)
