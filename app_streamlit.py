from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative

from data_utils import (
    completion_rate,
    count_or_sum,
    infer_columns,
    list_sheets,
    load_total_table,
    parse_duration_hours,
)
from airports_utils import load_airports, airport_visit_counts, classify_icao_points, save_ignored_report
from map_utils import build_grid, contour_traces_from_grid

st.set_page_config(page_title="CAOP - DASHBOARD DE ESTATÍSTICA", layout="wide")

DEFAULT_SHEET = "_____TOTAL_____"
MAP_STYLES = ["carto-darkmatter", "open-street-map", "carto-positron"]
MAP_MODES = [
    "Zonas de calor (colorido)",
    "Topologia (zonas + contornos)",
    "Somente pontos (todos iguais)",
]

# =========================
# Configuração fixa do mapa
# Edite aqui para todos os usuários do dashboard
# =========================

# ICAOs especiais que não devem entrar no mapa
IGNORED_ICAOS_ALWAYS = {"ZZZZ"}

# Defina aqui os aeroportos/base da CAOP.
# Exemplo: {"SBBR", "SBGO"}
CAOP_AIRPORTS = {"SBBR"}

POINT_COLORS = {
    "Brasil": "#4cc9f0",
    "Exterior": "#ef476f",
    "CAOP": "#ffb000",
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
    "ESO": "Escolta presos outro órgão",
    "IMA": "Imageamento",
    "INT": "Intervenção policial",
    "LPQ": "LPQD",
    "REC": "Apoio / reconhecimento - levantamento / evento",
    "RES": "Resgate",
    "TRP": "Traslado sem PAX e carga",
    "TRM": "Traslado para manutenção",
    "TRO": "Treinamento (operadores / outros a bordo)",
    "TCE": "Transporte de carga",
    # códigos extras encontrados em bases antigas / variações
    "DES": "Destruição de ilícitos",
    "EVT": "Evento",
}

OPS_SPEC_CODES = {"OBS", "IMA", "REC", "INT", "MOB", "ERR", "DES", "TCE", "EVT"}

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

@dataclass
class FilterConfig:
    years: list[int]
    asa_mode: str
    exclude_caop: bool
    granularity: str = "Ano"
    period_start: Optional[date] = None
    period_end: Optional[date] = None

PLOTLY_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "grafico_caop",
        "width": 1800,
        "height": 1000,
        "scale": 2,
    },
}



def _is_missing(value) -> bool:
    try:
        return pd.isna(value)
    except Exception:
        return False


def _stringify_for_streamlit(value):
    if value is None or _is_missing(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat(sep=" ") if hasattr(value, "isoformat") else str(value)
    if isinstance(value, time):
        return value.isoformat()
    return str(value)


def _make_arrow_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    risky_types = (pd.Timestamp, datetime, date, time, list, dict, set, tuple, bytes, bytearray)

    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue

        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_na = s.dropna()
            if non_na.empty:
                continue

            sample = non_na.head(100).tolist()
            types = {type(v) for v in sample}
            should_stringify = len(types) > 1 or any(isinstance(v, risky_types) for v in sample)

            if should_stringify:
                out[col] = s.map(_stringify_for_streamlit)

    return out.infer_objects(copy=False)


_ORIG_ST_DATAFRAME = st.dataframe
_ORIG_ST_PLOTLY_CHART = st.plotly_chart


def _safe_st_dataframe(data=None, *args, **kwargs):
    if kwargs.pop("use_container_width", None) is True and "width" not in kwargs:
        kwargs["width"] = "stretch"
    if isinstance(data, pd.DataFrame):
        data = _make_arrow_safe_dataframe(data)
    return _ORIG_ST_DATAFRAME(data, *args, **kwargs)


def _safe_st_plotly_chart(figure_or_data, *args, **kwargs):
    if kwargs.pop("use_container_width", None) is True and "width" not in kwargs:
        kwargs["width"] = "stretch"
    return _ORIG_ST_PLOTLY_CHART(figure_or_data, *args, **kwargs)


st.dataframe = _safe_st_dataframe
st.plotly_chart = _safe_st_plotly_chart
def style_plotly_figure(fig, height=500):
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(size=16),
        title_font=dict(size=24),
        legend=dict(font=dict(size=14), title_font=dict(size=15)),
        margin=dict(l=30, r=30, t=80, b=30),
    )
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=13))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=13))
    return fig

def style_pie_figure(fig, height=520):
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(size=16),
        title_font=dict(size=24),
        legend=dict(
            font=dict(size=14),
            title_font=dict(size=15),
            orientation="v",
        ),
        margin=dict(l=30, r=30, t=80, b=30),
        uniformtext_minsize=14,
        uniformtext_mode="hide",
    )
    fig.update_traces(
        textinfo="percent",
        textfont=dict(size=16),
        insidetextfont=dict(size=16),
        outsidetextfont=dict(size=15),
        sort=False,
    )
    return fig

def prepare_pie_dataframe(df: pd.DataFrame, label_col: str, value_col: str, others_label: str = "OUTROS") -> pd.DataFrame:
    pie_df = df[[label_col, value_col]].copy()
    pie_df = pie_df[pd.to_numeric(pie_df[value_col], errors="coerce").fillna(0) > 0].copy()

    if pie_df.empty:
        return pie_df

    pie_df["_is_others"] = pie_df[label_col].astype(str).str.upper().eq(others_label.upper())
    pie_df = pie_df.sort_values(
        by=["_is_others", value_col, label_col],
        ascending=[True, False, True],
    ).drop(columns="_is_others")

    return pie_df

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

def _safe_numeric_series(df: pd.DataFrame, col: Optional[str]) -> pd.Series:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype=float)

def apply_global_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 100% !important;
            padding-top: 2.2rem !important;
            padding-bottom: 1rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        h1 {
            font-size: 2.4rem !important;
            margin-top: 0 !important;
            margin-bottom: 0.8rem !important;
            padding-top: 0.2rem !important;
            line-height: 1.25 !important;
        }

        h2, h3 {
            font-size: 1.75rem !important;
            line-height: 1.25 !important;
            margin-top: 0.6rem !important;
        }

        div[data-testid="stMetricLabel"] p {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.95rem !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stCaptionContainer"],
        label {
            font-size: 1rem !important;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stMultiSelect"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stCheckbox"] label {
            font-size: 1rem !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner=False)
def load_main_table(file_bytes: bytes, sheet_name: str):
    return load_total_table(file_bytes, sheet_name=sheet_name)

@st.cache_data(show_spinner=False)
def load_airports_table(_: bytes) -> pd.DataFrame:
    # Mantido apenas por compatibilidade; o app agora usa load_airports(data_dir="data")
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_airports_cached(data_dir: str = "data") -> pd.DataFrame:
    return load_airports(data_dir=data_dir)

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
    out = df.copy()

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

def get_valid_date_range(df: pd.DataFrame) -> tuple[Optional[date], Optional[date]]:
    if "_date" not in df.columns:
        return None, None
    dates = pd.to_datetime(df["_date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().date(), dates.max().date()

def get_month_options(df: pd.DataFrame) -> list[str]:
    start_date, end_date = get_valid_date_range(df)
    if start_date is None or end_date is None:
        return []
    return pd.period_range(start=start_date, end=end_date, freq="M").astype(str).tolist()

def format_month(month_value: str) -> str:
    month = pd.Period(month_value, freq="M")
    return f"{PT_MONTHS[month.month]}/{month.year}"

def ensure_year_session(year_options: list[int]) -> None:
    if "years_sel" not in st.session_state:
        st.session_state["years_sel"] = year_options.copy()
        return
    valid = [y for y in st.session_state["years_sel"] if y in year_options]
    st.session_state["years_sel"] = valid if valid else year_options.copy()

def set_last_year(year_options: list[int]) -> None:
    if year_options:
        st.session_state["years_sel"] = [year_options[-1]]

def set_last_two_years(year_options: list[int]) -> None:
    if year_options:
        st.session_state["years_sel"] = year_options[-2:] if len(year_options) >= 2 else year_options.copy()

def render_global_filters(df: pd.DataFrame) -> FilterConfig:
    years = get_year_options(df)
    month_options = get_month_options(df)
    min_date, max_date = get_valid_date_range(df)

    st.subheader("Período de averiguação")
    available_granularities = ["Ano", "Mês", "Dia"] if month_options else ["Ano"]
    granularity = st.radio(
        "Granularidade",
        options=available_granularities,
        horizontal=True,
        help="Escolha se o período será definido por ano, mês ou dia.",
    )

    period_start: Optional[date] = None
    period_end: Optional[date] = None
    selected_years: list[int] = []

    if granularity == "Ano" and years:
        ensure_year_session(years)
        c1, c2, c3 = st.columns([2, 1, 1])
        with c2:
            st.button("Último ano", on_click=set_last_year, args=(years,), width='stretch')
        with c3:
            st.button("Últimos 2", on_click=set_last_two_years, args=(years,), width='stretch')
        with c1:
            st.multiselect("Filtro de anos", options=years, key="years_sel")
        selected_years = st.session_state["years_sel"]
    elif granularity == "Mês" and month_options:
        selected_months = st.select_slider(
            "Intervalo de meses",
            options=month_options,
            value=(month_options[0], month_options[-1]),
            format_func=format_month,
        )
        start_month, end_month = selected_months
        period_start = pd.Period(start_month, freq="M").start_time.date()
        period_end = pd.Period(end_month, freq="M").end_time.date()
    elif granularity == "Dia" and min_date is not None and max_date is not None:
        selected_dates = st.date_input(
            "Intervalo de datas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="DD/MM/YYYY",
        )
        if isinstance(selected_dates, (tuple, list)):
            if len(selected_dates) >= 1:
                period_start = selected_dates[0]
            if len(selected_dates) >= 2:
                period_end = selected_dates[1]
        else:
            period_start = period_end = selected_dates

    if not month_options:
        st.caption("A base não possui datas válidas; o filtro por mês e dia não está disponível.")

    c1, c2 = st.columns([1, 1])
    with c1:
        asa_mode = st.selectbox("Filtro ASA", options=["Todas", "Asa fixa (F)", "Asa rotativa (R)"], index=0)
    with c2:
        exclude_caop = st.checkbox("Excluir demandante CAOP", value=False)

    return FilterConfig(selected_years, asa_mode, exclude_caop, granularity, period_start, period_end)

def apply_filters(df: pd.DataFrame, cfg: ColumnConfig, filters: FilterConfig) -> pd.DataFrame:
    out = df.copy()

    if filters.granularity == "Ano" and filters.years and "_year" in out.columns:
        out = out[out["_year"].isin(filters.years)]

    if filters.granularity in {"Mês", "Dia"} and "_date" in out.columns:
        dates = pd.to_datetime(out["_date"], errors="coerce").dt.normalize()
        if filters.period_start is not None:
            out = out[dates >= pd.Timestamp(filters.period_start)]
            dates = dates.loc[out.index]
        if filters.period_end is not None:
            out = out[dates <= pd.Timestamp(filters.period_end)]

    if filters.asa_mode != "Todas" and "_asa" in out.columns:
        code = "F" if "F" in filters.asa_mode else "R"
        out = out[out["_asa"] == code]

    if filters.exclude_caop and cfg.demandante and cfg.demandante in out.columns:
        dem = out[cfg.demandante].astype(str).str.strip().str.upper()
        out = out[dem != "CAOP"]

    return out

def style_horizontal_bar_with_labels(fig, height=520, x_pad=0.08):
    max_x = 0.0

    for trace in fig.data:
        try:
            vals = [float(v) for v in trace.x if v is not None]
            if vals:
                max_x = max(max_x, max(vals))
        except Exception:
            pass

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=15, color="#1f1f1f"),
        cliponaxis=False,
        insidetextanchor="middle",
    )

    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(size=16),
        title_font=dict(size=24),
        legend=dict(font=dict(size=14), title_font=dict(size=15)),
        margin=dict(l=30, r=70, t=80, b=30),
    )

    fig.update_xaxes(
        title_font=dict(size=16),
        tickfont=dict(size=13),
        range=[0, max_x * (1 + x_pad)] if max_x > 0 else None,
    )
    fig.update_yaxes(
        title_font=dict(size=16),
        tickfont=dict(size=13),
    )

    return fig

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

def analyze_icao_usage(df: pd.DataFrame, cfg: ColumnConfig, airports: Optional[pd.DataFrame]):
    """
    Usa airports_utils.py para:
      - contar visitas por ICAO,
      - classificar pontos mapeados,
      - separar ignorados com motivo,
      - salvar relatório em data/ignored_icao_report.csv
    """
    empty_valid = pd.DataFrame(columns=["icao", "visitas", "latitude_deg", "longitude_deg", "iso_country", "categoria"])
    empty_ignored = pd.DataFrame(columns=["icao", "visitas", "motivo"])

    if not cfg.icao_from or not cfg.icao_to:
        return empty_valid, empty_ignored, {"total_raw": 0, "validos": 0, "ignorados": 0}

    visits = airport_visit_counts(df)
    if visits.empty:
        return empty_valid, empty_ignored, {"total_raw": 0, "validos": 0, "ignorados": 0}

    if airports is None or airports.empty:
        return empty_valid, empty_ignored, {
            "total_raw": int(visits["visitas"].sum()),
            "validos": 0,
            "ignorados": int(visits["visitas"].sum()),
        }

    mapped, ignored = classify_icao_points(visits, airports)

    if not mapped.empty:
        mapped = mapped.rename(columns={"point_category": "categoria"})

    try:
        save_ignored_report(ignored, "data/ignored_icao_report.csv")
    except Exception:
        pass

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

def render_ignored_icao_section(valid_points: pd.DataFrame, ignored_df: pd.DataFrame, summary: dict) -> None:
    exterior = int(valid_points.loc[valid_points["categoria"] == "Exterior", "visitas"].sum()) if not valid_points.empty else 0
    caop = int(valid_points.loc[valid_points["categoria"] == "CAOP", "visitas"].sum()) if not valid_points.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Visitas mapeadas", _format_int(summary.get("validos", 0)))
    c2.metric("Visitas ignoradas", _format_int(summary.get("ignorados", 0)))
    c3.metric("Visitas no exterior", _format_int(exterior))
    c4.metric("Visitas em aeroportos CAOP", _format_int(caop))

    with st.expander("ICAOs ignorados e conferência do mapa"):
        st.write("ICAOs em `IGNORED_ICAOS_ALWAYS` e códigos sem coordenadas não entram no mapa.")
        if ignored_df.empty:
            st.success("Nenhum ICAO foi ignorado além das regras fixas aplicadas.")
        else:
            ignored_show = ignored_df.groupby(["icao", "motivo"], dropna=False)["visitas"].sum().reset_index()
            ignored_show = ignored_show.sort_values(["visitas", "icao"], ascending=[False, True])
            st.dataframe(ignored_show, width='stretch')

def render_overview(df: pd.DataFrame, cfg: ColumnConfig) -> None:
    st.subheader("Visão geral")
    apply_global_style()

    if df.empty:
        st.warning("Filtro atual retornou 0 linhas.")
        return

    total = len(df)
    ttv = df["_ttv"] if "_ttv" in df.columns else pd.Series(dtype=float)
    ttv_sum = float(np.nansum(ttv.to_numpy())) if len(ttv) else 0.0
    ttv_fill = float(ttv.notna().mean() * 100.0) if len(ttv) else 0.0

    asa_series = df["_asa"].astype(str).str.upper().str.strip() if "_asa" in df.columns else pd.Series(dtype="object")
    asa_f = int((asa_series == "F").sum())
    asa_r = int((asa_series == "R").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trechos (linhas)", _format_int(total))
    c2.metric("Horas de voo (TTV)", _format_float_br(ttv_sum))
    c3.metric("TTV preenchido", f"{ttv_fill:.0f}%")
    c4.metric("ASA F / R", f"{asa_f} / {asa_r}")

    st.divider()

    if "_year" in df.columns and df["_year"].notna().any():
        plot_df = df.dropna(subset=["_year"]).copy()
        plot_df["asa_plot"] = (
            plot_df["_asa"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"F": "Asa fixa", "R": "Asa rotativa"})
            .fillna("Não informado")
        )

        yearly = plot_df.groupby(["_year", "asa_plot"], dropna=True)["_ttv"].sum().reset_index(name="ttv_h")

        fig = px.bar(
            yearly,
            x="_year",
            y="ttv_h",
            color="asa_plot",
            barmode="group",
            title="HORAS DE VOO POR ANO — ASA FIXA X ASA ROTATIVA",
            labels={"_year": "ANO", "ttv_h": "HORAS DE VOO (TTV)", "asa_plot": "CATEGORIA"},
            text_auto=".1f",
        )

        totals = plot_df.groupby("_year", dropna=True)["_ttv"].sum().reset_index(name="ttv_total")
        fig.add_scatter(
            x=totals["_year"],
            y=totals["ttv_total"],
            mode="lines+markers+text",
            name="TTV total",
            text=[f"{v:.1f}" for v in totals["ttv_total"]],
            textposition="top center",
        )

        fig.update_layout(xaxis_title="ANO", yaxis_title="HORAS DE VOO (TTV)", legend_title="CATEGORIA")
        st.plotly_chart(fig, width='stretch')

def render_operations(df: pd.DataFrame, cfg: ColumnConfig) -> None:
    st.subheader("Operações")

    if df.empty:
        st.warning("Filtro atual retornou 0 linhas.")
        return

    if not cfg.op_metrics:
        st.info("As colunas de operações não foram encontradas automaticamente.")
        return

    # Colunas originais
    OBS_COL = "OBS (IMA + REC) (1 por dia)"
    INT_COL = "INT (1 por dia)"
    MOB_COL = "MOBILIZAÇÃO (POR NOME DA OPERAÇÃO, unidade apoiada, serviço especializado)"
    DESTR_ERR_COL = "DESTRUIÇÃO + ERRADICAÇÃO (1 POR DIA CADA)"
    ERR_COL = "ERR (NOME DA OPERAÇÃO)"

    raw_values = {}
    methods = []

    for col in cfg.op_metrics:
        if col not in df.columns:
            continue
        value, method = count_or_sum(df[col])
        value = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
        raw_values[col] = float(value)
        methods.append((col, method, float(value)))

    if not raw_values:
        st.info("Nenhuma métrica de operação foi encontrada nas colunas configuradas.")
        return

    # Regras de negócio
    erradicacao = raw_values.get(ERR_COL, 0.0)
    destruicao_e_erradicacao = raw_values.get(DESTR_ERR_COL, 0.0)

    # Mostrar apenas destruição líquida
    destruicao = max(destruicao_e_erradicacao - erradicacao, 0.0)

    rows = [
        ("Observação", raw_values.get(OBS_COL, 0.0)),
        ("Intervenção", raw_values.get(INT_COL, 0.0)),
        ("Mobilização", raw_values.get(MOB_COL, 0.0)),
        ("Destruição", destruicao),
        ("Erradicação", erradicacao),
    ]

    ops = pd.DataFrame(rows, columns=["tipo", "quantidade"])
    ops["quantidade"] = pd.to_numeric(ops["quantidade"], errors="coerce").fillna(0.0)
    ops["quantidade"] = ops["quantidade"].clip(lower=0)

    if ops["quantidade"].sum() <= 0:
        st.info("Não há valores positivos de operações no filtro atual.")
        return

    def _fmt_num(v: float) -> str:
        if abs(v - round(v)) < 1e-9:
            return _format_int(int(round(v)))
        return _format_float_br(v)

    ops["quantidade_txt"] = ops["quantidade"].map(_fmt_num)

    # KPIs rápidos
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observação", _fmt_num(float(ops.loc[ops["tipo"] == "Observação", "quantidade"].sum())))
    c2.metric("Intervenção", _fmt_num(float(ops.loc[ops["tipo"] == "Intervenção", "quantidade"].sum())))
    c3.metric("Mobilização", _fmt_num(float(ops.loc[ops["tipo"] == "Mobilização", "quantidade"].sum())))
    c4.metric("Destruição", _fmt_num(float(ops.loc[ops["tipo"] == "Destruição", "quantidade"].sum())))
    c5.metric("Erradicação", _fmt_num(float(ops.loc[ops["tipo"] == "Erradicação", "quantidade"].sum())))

    st.divider()

    col_bar, col_pie = st.columns([3, 2], vertical_alignment="top")

    with col_bar:
        ops_bar = ops.sort_values(["quantidade", "tipo"], ascending=[True, True]).copy()

        fig_bar = px.bar(
            ops_bar,
            x="quantidade",
            y="tipo",
            orientation="h",
            text="quantidade_txt",
            title="Tipos de operações realizadas pela CAOP",
            labels={"quantidade": "Quantidade", "tipo": "Tipo de operação"},
        )

        fig_bar.update_traces(
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=15, color="#1f1f1f"),
        )

        fig_bar.update_layout(
            template="plotly_white",
            height=520,
            font=dict(size=16),
            title_font=dict(size=24),
            legend_font=dict(size=14),
            margin=dict(l=30, r=70, t=80, b=30),
        )
        fig_bar.update_xaxes(title_font=dict(size=16), tickfont=dict(size=13))
        fig_bar.update_yaxes(title_font=dict(size=16), tickfont=dict(size=13))

        st.plotly_chart(
            fig_bar,
            width='stretch',
            config=PLOTLY_CONFIG if "PLOTLY_CONFIG" in globals() else None,
        )

    with col_pie:
        ops_pie = ops[ops["quantidade"] > 0].sort_values(
            ["quantidade", "tipo"], ascending=[False, True]
        ).copy()

        fig_pie = px.pie(
            ops_pie,
            names="tipo",
            values="quantidade",
            hole=0.35,
            title="Distribuição percentual dos tipos de operação",
        )

        fig_pie.update_traces(
            textinfo="percent+label",
            textposition="inside",
            sort=False,
        )

        fig_pie.update_layout(
            template="plotly_white",
            height=520,
            font=dict(size=16),
            title_font=dict(size=24),
            legend=dict(font=dict(size=14), title_font=dict(size=15)),
            margin=dict(l=30, r=30, t=80, b=30),
            uniformtext_minsize=14,
            uniformtext_mode="hide",
        )

        st.plotly_chart(
            fig_pie,
            width='stretch',
            config=PLOTLY_CONFIG if "PLOTLY_CONFIG" in globals() else None,
        )

    st.divider()

    tabela = ops.sort_values(["quantidade", "tipo"], ascending=[False, True]).rename(
        columns={
            "tipo": "Tipo de operação",
            "quantidade": "Quantidade",
            "quantidade_txt": "Quantidade formatada",
        }
    )

    st.dataframe(
        tabela[["Tipo de operação", "Quantidade"]],
        width='stretch',
    )

    with st.expander("Conferência da transformação aplicada"):
        methods_df = pd.DataFrame(
            methods,
            columns=["Coluna original", "Método", "Valor bruto"],
        )

        regra_df = pd.DataFrame(
            [
                {
                    "Tipo exibido": "Observação",
                    "Origem": OBS_COL,
                    "Regra": "Valor direto da coluna original",
                },
                {
                    "Tipo exibido": "Intervenção",
                    "Origem": INT_COL,
                    "Regra": "Valor direto da coluna original",
                },
                {
                    "Tipo exibido": "Mobilização",
                    "Origem": MOB_COL,
                    "Regra": "Valor direto da coluna original",
                },
                {
                    "Tipo exibido": "Erradicação",
                    "Origem": ERR_COL,
                    "Regra": "Valor direto da coluna original",
                },
                {
                    "Tipo exibido": "Destruição",
                    "Origem": DESTR_ERR_COL,
                    "Regra": "DESTRUIÇÃO + ERRADICAÇÃO - ERRADICAÇÃO",
                },
            ]
        )

        st.markdown("**Método detectado por coluna**")
        st.dataframe(methods_df, width='stretch')

        st.markdown("**Regras de exibição para o usuário final**")
        st.dataframe(regra_df, width='stretch')

def render_map(df: pd.DataFrame, cfg: ColumnConfig, airports: Optional[pd.DataFrame]) -> None:
    st.subheader("Mapa")

    if df.empty:
        st.warning("Filtro atual retornou 0 linhas.")
        return

    valid_points, ignored_df, summary = analyze_icao_usage(df, cfg, airports)

    if airports is None or airports.empty:
        st.warning("Base de aeroportos não carregada automaticamente. Verifique data/airports_master.csv, data/airports.csv e data/icao_overrides.csv.")
        render_ignored_icao_section(valid_points, ignored_df, summary)
        return

    if valid_points.empty:
        st.warning("Nenhum ICAO válido encontrou coordenadas na base de aeroportos.")
        render_ignored_icao_section(valid_points, ignored_df, summary)
        return

    mode = st.radio("Visualização", options=MAP_MODES, horizontal=True)
    map_style = st.selectbox("Mapa-base", options=MAP_STYLES, index=0)
    center = get_map_center(valid_points)
    zoom_default = 3.4 if (valid_points["categoria"] == "Exterior").any() else 4.0

    if mode == "Somente pontos (todos iguais)":
        points_df = valid_points.copy()

        # garante ordem estável da legenda
        cat_order = ["Brasil", "Exterior", "CAOP"]
        points_df["categoria"] = pd.Categorical(
            points_df["categoria"],
            categories=cat_order,
            ordered=True,
        )

        # marcador de tamanho fixo
        points_df["_marker_size"] = 10

        fig = px.scatter_map(
            points_df,
            lat="latitude_deg",
            lon="longitude_deg",
            color="categoria",
            size="_marker_size",
            size_max=10,
            zoom=zoom_default,
            center=center,
            map_style=map_style,
            hover_name="icao",
            hover_data={
                "visitas": True,
                "iso_country": True,
                "categoria": True,
                "latitude_deg": False,
                "longitude_deg": False,
                "_marker_size": False,
            },
            color_discrete_map=POINT_COLORS,
            category_orders={"categoria": cat_order},
            title="PONTOS VISITADOS (origem e destino)",
        )

        fig.update_traces(
            marker=dict(opacity=0.95),
        )

        fig.update_layout(
            height=720,
            margin=dict(l=0, r=0, t=45, b=0),
            legend_title="CATEGORIA",
        )

        st.plotly_chart(fig, width="stretch")

    else:
        c1, c2, c3 = st.columns(3)
        radius = c1.slider("Raio do calor (px)", min_value=10, max_value=90, value=55, step=5)
        clip_q = c2.slider("Corte de outlier (quantil)", min_value=0.90, max_value=0.995, value=0.98, step=0.005)
        opacity = c3.slider("Opacidade", min_value=0.30, max_value=1.0, value=0.90, step=0.05)

        zmax = float(np.quantile(valid_points["visitas"].to_numpy(), float(clip_q))) if len(valid_points) else 1.0
        zmax = max(zmax, 1.0)

        fig = px.density_map(
            valid_points,
            lat="latitude_deg",
            lon="longitude_deg",
            z="visitas",
            radius=int(radius),
            zoom=zoom_default,
            center=center,
            map_style=map_style,
            color_continuous_scale="Turbo",
            range_color=(0, zmax),
            opacity=float(opacity),
            title="ZONAS DE CALOR POR AERODROMO VISITADO (origem e destino)",
        )
        fig.update_layout(height=720, margin=dict(l=0, r=0, t=45, b=0))

        if mode == "Topologia (zonas + contornos)":
            cell_km = max(20.0, min(120.0, float(radius) * 1.3))
            grid = build_grid(valid_points, lat_col="latitude_deg", lon_col="longitude_deg", w_col="visitas", cell_km=cell_km)
            levels = st.slider("Níveis (contornos)", min_value=6, max_value=26, value=14, step=1)
            for trace in contour_traces_from_grid(grid, levels=int(levels), clip_q=float(clip_q)):
                fig.add_trace(trace)

        st.plotly_chart(fig, width='stretch')

    render_ignored_icao_section(valid_points, ignored_df, summary)

def render_demandantes(df: pd.DataFrame, cfg: ColumnConfig) -> None:
    st.subheader("Demandantes")

    if not cfg.demandante or cfg.demandante not in df.columns:
        st.info("A coluna de demandante não foi encontrada automaticamente.")
        return

    if df.empty:
        st.warning("Filtro atual retornou 0 linhas.")
        return

    base = df.copy()
    base[cfg.demandante] = base[cfg.demandante].astype(str).str.strip()
    base = base[
        base[cfg.demandante].ne("")
        & base[cfg.demandante].ne("nan")
        & base[cfg.demandante].ne("None")
    ].copy()

    if base.empty:
        st.warning("Não há demandantes válidos no filtro atual.")
        return

    aircraft_filter_active = bool(cfg.aircraft and cfg.aircraft in base.columns)
    selected_aircraft = []

    if aircraft_filter_active:
        base[cfg.aircraft] = base[cfg.aircraft].astype(str).str.strip()
        aircraft_options = sorted(
            base.loc[
                base[cfg.aircraft].ne("")
                & base[cfg.aircraft].ne("nan")
                & base[cfg.aircraft].ne("None"),
                cfg.aircraft,
            ].astype(str).unique().tolist()
        )

        if aircraft_options:
            selected_aircraft = st.multiselect(
                "Filtrar estatísticas de demandantes por aeronave",
                options=aircraft_options,
                default=aircraft_options,
                key="demandantes_aircraft_filter",
            )

            if not selected_aircraft:
                st.warning("Selecione ao menos uma aeronave para visualizar as estatísticas de demandantes.")
                return

            base = base[base[cfg.aircraft].astype(str).isin(selected_aircraft)].copy()

            if base.empty:
                st.warning("Não há dados de demandantes para as aeronaves selecionadas.")
                return

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        top_n = st.slider(
            "Top N",
            min_value=1,
            max_value=min(50, max(1, base[cfg.demandante].nunique())),
            value=min(20, max(1, base[cfg.demandante].nunique())),
            step=1,
            key="demandantes_top_n",
        )
    with c2:
        exclude_caop_local = st.checkbox("Excluir CAOP (apenas aqui)", value=True, key="demandantes_excluir_caop")
    with c3:
        group_others = st.checkbox("Agrupar restante em 'OUTROS'", value=True, key="demandantes_group_others")

    total_trechos_full = len(base)
    caop_mask_full = base[cfg.demandante].str.upper() == "CAOP"
    caop_trechos = int(caop_mask_full.sum())
    caop_passageiros = float(base.loc[caop_mask_full, "_passengers"].sum()) if "_passengers" in base.columns else 0.0
    caop_carga = float(base.loc[caop_mask_full, "_cargo"].sum()) if "_cargo" in base.columns else 0.0

    if exclude_caop_local:
        base = base[base[cfg.demandante].str.upper() != "CAOP"].copy()

    agg = (
        base.groupby(cfg.demandante, dropna=False)
        .agg(
            trechos=(cfg.demandante, "size"),
            passageiros_total=("_passengers", "sum"),
            carga_total_kg=("_cargo", "sum"),
        )
        .reset_index()
        .rename(columns={cfg.demandante: "Demandante"})
        .sort_values(["trechos", "passageiros_total", "carga_total_kg"], ascending=[False, False, False])
    )

    if agg.empty:
        st.warning("Não há demandantes após aplicar os filtros locais.")
        return

    top = agg.head(top_n).copy()

    if group_others and len(agg) > top_n:
        others = agg.iloc[top_n:].copy()
        if not others.empty:
            top = pd.concat(
                [
                    top,
                    pd.DataFrame(
                        [
                            {
                                "Demandante": "OUTROS",
                                "trechos": others["trechos"].sum(),
                                "passageiros_total": others["passageiros_total"].sum(),
                                "carga_total_kg": others["carga_total_kg"].sum(),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    total_trechos = int(base.shape[0])
    total_passageiros = float(base["_passengers"].sum()) if "_passengers" in base.columns else 0.0
    total_carga = float(base["_cargo"].sum()) if "_cargo" in base.columns else 0.0

    if selected_aircraft:
        st.caption(f"Filtro ativo de aeronaves: {', '.join(selected_aircraft)}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Demandantes no filtro", _format_int(base[cfg.demandante].nunique()))
    k2.metric("Trechos", _format_int(total_trechos))
    k3.metric("Passageiros", _format_int(int(total_passageiros)))
    k4.metric("Carga (kg)", _format_float_br(total_carga))
    k5.metric("CAOP (%)", f"{(100.0 * caop_trechos / total_trechos_full):.1f}%" if total_trechos_full else "—")

    st.divider()

    # =========================
    # Trechos por demandante
    # =========================
    col_bar, col_pie = st.columns([3, 2], vertical_alignment="top")

    with col_bar:
        plot_df = top.sort_values("trechos", ascending=True)
        fig = px.bar(
            plot_df,
            x="trechos",
            y="Demandante",
            orientation="h",
            title="TRECHOS POR DEMANDANTE",
            labels={"trechos": "Trechos", "Demandante": "Demandante"},
            text="trechos",
        )
        st.plotly_chart(
            style_horizontal_bar_with_labels(fig, height=560, x_pad=0.10),
            width='stretch',
            config=PLOTLY_CONFIG,
        )

    with col_pie:
        pie_df = prepare_pie_dataframe(top, "Demandante", "trechos")
        if not pie_df.empty:
            fig = px.pie(
                pie_df,
                names="Demandante",
                values="trechos",
                hole=0.35,
                title="DISTRIBUIÇÃO DE TRECHOS (%)",
            )
            st.plotly_chart(
                style_pie_figure(fig),
                width='stretch',
                config=PLOTLY_CONFIG,
            )

    # =========================
    # Passageiros por demandante
    # =========================
    col_bar2, col_pie2 = st.columns([3, 2], vertical_alignment="top")

    with col_bar2:
        plot_df = top.sort_values("passageiros_total", ascending=True)
        fig = px.bar(
            plot_df,
            x="passageiros_total",
            y="Demandante",
            orientation="h",
            title="PASSAGEIROS TRANSPORTADOS POR DEMANDANTE",
            labels={"passageiros_total": "Passageiros", "Demandante": "Demandante"},
            text="passageiros_total",
        )
        st.plotly_chart(
            style_horizontal_bar_with_labels(fig, height=560, x_pad=0.10),
            width='stretch',
            config=PLOTLY_CONFIG,
        )

    with col_pie2:
        pie_df = prepare_pie_dataframe(top, "Demandante", "passageiros_total")
        if not pie_df.empty:
            fig = px.pie(
                pie_df,
                names="Demandante",
                values="passageiros_total",
                hole=0.35,
                title="DISTRIBUIÇÃO DE PASSAGEIROS (%)",
            )
            st.plotly_chart(style_pie_figure(fig), width='stretch', config=PLOTLY_CONFIG)

    # =========================
    # Carga por demandante
    # =========================
    col_bar3, col_pie3 = st.columns([3, 2], vertical_alignment="top")

    with col_bar3:
        plot_df = top.sort_values("carga_total_kg", ascending=True)
        fig = px.bar(
            plot_df,
            x="carga_total_kg",
            y="Demandante",
            orientation="h",
            title="CARGAS TRANSPORTADA POR DEMANDANTE",
            labels={"carga_total_kg": "Carga (kg)", "Demandante": "Demandante"},
            text="carga_total_kg",
        )
        st.plotly_chart(
            style_horizontal_bar_with_labels(fig, height=560, x_pad=0.10),
            width='stretch',
            config=PLOTLY_CONFIG,
        )

    with col_pie3:
        pie_df = prepare_pie_dataframe(top, "Demandante", "carga_total_kg")
        if not pie_df.empty:
            fig = px.pie(
                pie_df,
                names="Demandante",
                values="carga_total_kg",
                hole=0.35,
                title="DISTRIBUIÇÃO DE CARGA (%)",
            )
            st.plotly_chart(
                style_pie_figure(fig),
                width='stretch',
                config=PLOTLY_CONFIG,
            )

    st.divider()

    tabela = top.rename(
        columns={
            "trechos": "Trechos",
            "passageiros_total": "Passageiros",
            "carga_total_kg": "Carga (kg)",
        }
    )

    st.dataframe(tabela, width='stretch')

    with st.expander("Resumo de CAOP"):
        c1, c2, c3 = st.columns(3)
        c1.metric("CAOP (trechos)", _format_int(caop_trechos))
        c2.metric("CAOP (passageiros)", _format_int(int(caop_passageiros)))
        c3.metric("CAOP (carga kg)", _format_float_br(caop_carga))

def _prepare_aircraft_base(df: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    base = df.copy()
    base[cfg.aircraft] = base[cfg.aircraft].astype(str).str.strip()
    base = base[
        base[cfg.aircraft].ne("")
        & base[cfg.aircraft].ne("nan")
        & base[cfg.aircraft].ne("None")
    ].copy()
    return base

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

def _monthly_aircraft_metrics(sub: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    monthly = (
        sub.dropna(subset=["_month_dt"])
        .groupby(["_month_dt", "_month"], dropna=False)
        .agg(
            trechos=(cfg.aircraft, "size"),
            passageiros_total=("_passengers", "sum"),
            presos_total=("_prisoners", "sum"),
            carga_total_kg=("_cargo", "sum"),
            ttv_total=("_ttv", "sum"),
        )
        .reset_index()
        .sort_values("_month_dt")
    )
    return monthly

def _update_fig_layout(fig, height=430):
    fig.update_layout(
        height=height,
        title_font_size=22,
        font=dict(size=16),
        legend_font=dict(size=14),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fig.update_xaxes(title_font=dict(size=16), tickfont=dict(size=13))
    fig.update_yaxes(title_font=dict(size=16), tickfont=dict(size=13))
    return fig

PT_MONTHS = {
    1: "JAN", 2: "FEV", 3: "MAR", 4: "ABR", 5: "MAI", 6: "JUN",
    7: "JUL", 8: "AGO", 9: "SET", 10: "OUT", 11: "NOV", 12: "DEZ",
}

def _build_aircraft_color_map(aircraft_names: list[str]) -> dict[str, str]:
    from plotly.colors import qualitative

    palette = (
        qualitative.Plotly
        + qualitative.Dark24
        + qualitative.Light24
        + qualitative.Alphabet
        + qualitative.Safe
        + qualitative.Vivid
    )

    unique_names = sorted(set(map(str, aircraft_names)))
    return {name: palette[i % len(palette)] for i, name in enumerate(unique_names)}

def _prepare_aircraft_availability_matrix(
    base: pd.DataFrame,
    cfg: ColumnConfig,
    aircraft_order: list[str],
) -> tuple[pd.DataFrame, list[str], str]:
    work = base.dropna(subset=["_date", "_month_dt"]).copy()
    work[cfg.aircraft] = work[cfg.aircraft].astype(str).str.strip()
    work = work[work[cfg.aircraft].isin(aircraft_order)].copy()

    if work.empty:
        return pd.DataFrame(index=aircraft_order), [], "DISPONIBILIDADE MENSAL POR AERONAVE"

    work["_day"] = work["_date"].dt.floor("D")

    daily_presence = (
        work.groupby([cfg.aircraft, "_month_dt", "_day"], dropna=False)
        .size()
        .reset_index(name="n")
    )

    monthly_presence = (
        daily_presence.groupby([cfg.aircraft, "_month_dt"], dropna=False)
        .size()
        .reset_index(name="dias_utilizados")
    )

    years = sorted(work["_year"].dropna().astype(int).unique().tolist()) if "_year" in work.columns else []

    if len(years) == 1:
        all_months = pd.date_range(f"{years[0]}-01-01", f"{years[0]}-12-01", freq="MS")
        title = f"DISPONIBILIDADE MENSAL POR AERONAVE - {years[0]}"
        month_labels = [PT_MONTHS[m.month] for m in all_months]
    else:
        all_months = pd.to_datetime(sorted(work["_month_dt"].dropna().unique()))
        title = "DISPONIBILIDADE MENSAL POR AERONAVE"
        month_labels = [f"{PT_MONTHS[m.month]}/{str(m.year)[2:]}" for m in all_months]

    full_grid = pd.MultiIndex.from_product(
        [aircraft_order, all_months],
        names=[cfg.aircraft, "_month_dt"],
    ).to_frame(index=False)

    matrix_df = full_grid.merge(
        monthly_presence,
        on=[cfg.aircraft, "_month_dt"],
        how="left",
    )

    matrix_df["dias_utilizados"] = matrix_df["dias_utilizados"].fillna(0)
    matrix_df["dias_mes"] = matrix_df["_month_dt"].dt.days_in_month
    matrix_df["disponibilidade_pct"] = (
        100.0 * matrix_df["dias_utilizados"] / matrix_df["dias_mes"]
    ).fillna(0.0)

    label_map = {m: lbl for m, lbl in zip(all_months, month_labels)}
    matrix_df["_month_label"] = matrix_df["_month_dt"].map(label_map)

    pivot = (
        matrix_df.pivot(index=cfg.aircraft, columns="_month_label", values="disponibilidade_pct")
        .reindex(index=aircraft_order, columns=month_labels)
        .fillna(0.0)
    )

    return pivot, month_labels, title

def _build_availability_heatmap(
    pivot: pd.DataFrame,
    month_labels: list[str],
    title: str,
) -> go.Figure:
    if pivot.empty or len(month_labels) == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title=title,
            height=450,
            annotations=[
                dict(
                    text="Sem dados suficientes para montar o mapa de disponibilidade.",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=18),
                )
            ],
        )
        return fig

    z = pivot.to_numpy(dtype=float)
    text = np.vectorize(lambda v: f"{int(round(v))}%")(z)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=month_labels,
            y=pivot.index.tolist(),
            colorscale=[
                [0.00, "#f0f0f0"],
                [0.15, "#dfe9df"],
                [0.35, "#bdd8bd"],
                [0.55, "#8dc28f"],
                [0.75, "#5fb56b"],
                [1.00, "#35a853"],
            ],
            zmin=0,
            zmax=100,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 15, "color": "#111111"},
            xgap=1,
            ygap=1,
            colorbar=dict(
                title=dict(
                    text="Disponibilidade (%)",
                    font=dict(size=16),
                ),
                tickfont=dict(size=13),
            ),
            hovertemplate="<b>%{y}</b><br>Mês: %{x}<br>Disponibilidade observada: %{z:.0f}%<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        title_font=dict(size=26),
        font=dict(size=16),
        height=max(520, 48 * len(pivot.index)),
        margin=dict(l=30, r=30, t=80, b=30),
        xaxis=dict(
            title="",
            tickfont=dict(size=13),
            side="bottom",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=14),
            autorange="reversed",
        ),
    )

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

    uf_col = None
    for c in ["state", "iso_region", "municipality_state", "uf"]:
        if c in ap.columns:
            uf_col = c
            break

    if uf_col is None:
        return out

    ap_map = ap[["icao", uf_col]].dropna().drop_duplicates(subset=["icao"], keep="last").copy()
    ap_map.rename(columns={uf_col: "_uf_lookup"}, inplace=True)

    out["_icao_to_norm"] = out[cfg.icao_to].astype(str).str.strip().str.upper()
    out = out.merge(ap_map, left_on="_icao_to_norm", right_on="icao", how="left")

    if "_uf_lookup" in out.columns:
        if uf_col == "iso_region":
            out["_uf_dest"] = out["_uf_lookup"].astype(str).str.split("-").str[-1]
        else:
            out["_uf_dest"] = out["_uf_lookup"]

    uf_series = out["_uf_dest"].astype("string").str.strip()
    uf_series = uf_series.mask(uf_series.isin(["", "nan", "None", "<NA>"]), pd.NA)
    out["_uf_dest"] = uf_series
    return out


def _monthly_group_for_aircraft_selection(base_sel: pd.DataFrame, cfg: ColumnConfig) -> pd.DataFrame:
    return (
        base_sel.dropna(subset=["_month_dt"])
        .groupby(["_month_dt", "_month"], dropna=False)
        .agg(
            trechos=(cfg.aircraft, "size"),
            passageiros_total=("_passengers", "sum"),
            presos_total=("_prisoners", "sum"),
            carga_total_kg=("_cargo", "sum"),
            ttv_total=("_ttv", "sum"),
        )
        .reset_index()
        .sort_values("_month_dt")
    )


def _operations_summary_for_subset(df: pd.DataFrame, cfg: ColumnConfig) -> tuple[dict[str, float], float]:
    if not cfg.op_metrics:
        return {}, 0.0

    raw_values: dict[str, float] = {}
    for col in cfg.op_metrics:
        if col not in df.columns:
            continue
        value, _ = count_or_sum(df[col])
        val = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
        raw_values[col] = float(val)

    obs_col = "OBS (IMA + REC) (1 por dia)"
    int_col = "INT (1 por dia)"
    mob_col = "MOBILIZAÇÃO (POR NOME DA OPERAÇÃO, unidade apoiada, serviço especializado)"
    destr_err_col = "DESTRUIÇÃO + ERRADICAÇÃO (1 POR DIA CADA)"
    err_col = "ERR (NOME DA OPERAÇÃO)"

    erradicacao = raw_values.get(err_col, 0.0)
    destruicao_e_erradicacao = raw_values.get(destr_err_col, 0.0)
    destruicao = max(destruicao_e_erradicacao - erradicacao, 0.0)

    summary = {
        "Observação": raw_values.get(obs_col, 0.0),
        "Intervenção": raw_values.get(int_col, 0.0),
        "Mobilização": raw_values.get(mob_col, 0.0),
        "Destruição": destruicao,
        "Erradicação": erradicacao,
    }
    total = float(sum(summary.values()))
    return summary, total


def _group_ttv(df: pd.DataFrame, group_col: str, output_col: str) -> pd.DataFrame:
    out = (
        df.dropna(subset=[group_col])
        .groupby(group_col, dropna=False)
        .agg(horas_voo=("_ttv", "sum"))
        .reset_index()
        .rename(columns={group_col: output_col})
        .sort_values(["horas_voo", output_col], ascending=[False, True])
    )
    out["horas_voo"] = pd.to_numeric(out["horas_voo"], errors="coerce").fillna(0.0)
    return out


def _pie_bar_pair(
    df_top: pd.DataFrame,
    label_col: str,
    value_col: str,
    title_base: str,
    value_label: str,
    key_prefix: str,
):
    work = df_top.copy()
    work = work[pd.to_numeric(work[value_col], errors="coerce").fillna(0) > 0].copy()

    if work.empty:
        return

    labels = work[label_col].astype(str).tolist()

    selected_labels = st.multiselect(
        f"Categorias visíveis — {title_base}",
        options=labels,
        default=labels,
        key=f"{key_prefix}_visible_labels",
    )

    if not selected_labels:
        st.info(f"Nenhuma categoria selecionada em {title_base}.")
        return

    work = work[work[label_col].astype(str).isin(selected_labels)].copy()

    c1, c2 = st.columns(2)

    with c1:
        pie_df = prepare_pie_dataframe(work, label_col, value_col)
        if not pie_df.empty:
            fig = px.pie(
                pie_df,
                names=label_col,
                values=value_col,
                hole=0.35,
                title=f"{title_base} (%)",
            )
            st.plotly_chart(
                style_pie_figure(fig, 480),
                width="stretch",
                config=PLOTLY_CONFIG,
                key=f"{key_prefix}_pie",
            )

    with c2:
        plot_df = work.sort_values(value_col, ascending=True).copy()
        plot_df["_valor_txt"] = plot_df[value_col].map(lambda v: _format_float_br(float(v)))

        fig = px.bar(
            plot_df,
            x=value_col,
            y=label_col,
            orientation="h",
            title=title_base,
            labels={value_col: value_label, label_col: ""},
            text="_valor_txt",
        )
        st.plotly_chart(
            style_horizontal_bar_with_labels(fig, 480, 0.10),
            width="stretch",
            config=PLOTLY_CONFIG,
            key=f"{key_prefix}_bar",
        )

def render_aircraft(df: pd.DataFrame, cfg: ColumnConfig, airports: Optional[pd.DataFrame] = None) -> None:
    st.subheader("Aeronaves")

    if not cfg.aircraft or cfg.aircraft not in df.columns:
        st.info("A coluna da aeronave não foi encontrada automaticamente.")
        return

    if df.empty:
        st.warning("Filtro atual retornou 0 linhas.")
        return

    base = _prepare_aircraft_base(df, cfg)
    if base.empty:
        st.warning("Não há aeronaves válidas no filtro atual.")
        return

    base = _enrich_with_uf_from_airports(base, cfg, airports)
    grouped = _aircraft_grouped(base, cfg)
    aircraft_options = grouped[cfg.aircraft].astype(str).tolist()

    st.markdown("### 1) Visão geral da frota")

    selected_general = st.multiselect(
        "Escolha uma ou mais aeronaves para a visão geral",
        options=aircraft_options,
        default=aircraft_options,
        key="aircraft_selected_general",
    )

    if not selected_general:
        st.warning("Selecione ao menos uma aeronave na visão geral.")
        return

    base_general = base[base[cfg.aircraft].astype(str).isin(selected_general)].copy()
    grouped_general = _aircraft_grouped(base_general, cfg)

    total_aeronaves = grouped_general[cfg.aircraft].nunique()
    total_trechos = int(grouped_general["trechos"].sum())
    total_ttv = float(grouped_general["ttv_total"].sum())
    total_passag = float(grouped_general["passageiros_total"].sum())
    total_presos = float(grouped_general["presos_total"].sum())
    total_carga = float(grouped_general["carga_total_kg"].sum())

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Aeronaves no filtro", _format_int(total_aeronaves))
    k2.metric("Trechos", _format_int(total_trechos))
    k3.metric("Horas de voo (TTV)", _format_float_br(total_ttv))
    k4.metric("Passageiros", _format_int(int(total_passag)))
    k5.metric("Presos", _format_int(int(total_presos)))
    k6.metric("Carga (kg)", _format_float_br(total_carga))

    top_n = st.slider(
        "Top N de aeronaves na visão geral",
        min_value=1,
        max_value=max(1, len(grouped_general)),
        value=min(10, max(1, len(grouped_general))),
        step=1,
        key="aircraft_top_n_general",
    )

    top_ttv = grouped_general.nlargest(top_n, "ttv_total")[[cfg.aircraft, "ttv_total"]].copy()
    top_pass = grouped_general.nlargest(top_n, "passageiros_total")[[cfg.aircraft, "passageiros_total"]].copy()
    top_carga = grouped_general.nlargest(top_n, "carga_total_kg")[[cfg.aircraft, "carga_total_kg"]].copy()

    _pie_bar_pair(
        top_ttv,
        cfg.aircraft,
        "ttv_total",
        "Horas de voo por aeronave",
        "Horas de voo (TTV)",
        "aircraft_ttv",
    )

    _pie_bar_pair(
        top_pass,
        cfg.aircraft,
        "passageiros_total",
        "Passageiros transportados por aeronave",
        "Passageiros",
        "aircraft_pass",
    )

    _pie_bar_pair(
        top_carga,
        cfg.aircraft,
        "carga_total_kg",
        "Carga transportada por aeronave",
        "Carga (kg)",
        "aircraft_cargo",
    )

    if "_uf_dest" in base_general.columns:
        uf_df = _group_ttv(base_general, "_uf_dest", "UF")
        if not uf_df.empty:
            _pie_bar_pair(
                uf_df,
                "UF",
                "horas_voo",
                "Unidades da federação atendidas",
                "Horas de voo (TTV)",
                "aircraft_uf",
            )

    if "_nat_label" in base_general.columns:
        nat_df = _group_ttv(base_general, "_nat_label", "Natureza da missão")
        if not nat_df.empty:
            _pie_bar_pair(
                nat_df,
                "Natureza da missão",
                "horas_voo",
                "Natureza da missão",
                "Horas de voo (TTV)",
                "aircraft_nat",
            )

    if "_espec_label" in base_general.columns:
        espec_df = _group_ttv(base_general, "_espec_label", "Especificação da missão")
        if not espec_df.empty:
            _pie_bar_pair(
                espec_df,
                "Especificação da missão",
                "horas_voo",
                "Especificação da missão",
                "Horas de voo (TTV)",
                "aircraft_espec",
            )

    show_table = grouped_general.rename(columns={
        cfg.aircraft: "Aeronave",
        "asa_label": "Tipo",
        "trechos": "Trechos",
        "ttv_total": "Horas de voo (TTV)",
        "passageiros_total": "Passageiros",
        "presos_total": "Presos",
        "carga_total_kg": "Carga (kg)",
    })
    st.dataframe(show_table, width="stretch")

    st.divider()

    st.markdown("### 2) Foco em uma ou mais aeronaves")

    selected_detail = st.multiselect(
        "Escolha uma ou mais aeronaves para o foco detalhado",
        options=aircraft_options,
        default=aircraft_options[:1] if aircraft_options else [],
        key="aircraft_selected_detail_multi",
    )

    if not selected_detail:
        st.warning("Selecione ao menos uma aeronave para o detalhamento.")
        return

    sub = base[base[cfg.aircraft].astype(str).isin(selected_detail)].copy()
    monthly_aircraft = _monthly_group_for_aircraft_selection(sub, cfg)
    _, total_ops_exec = _operations_summary_for_subset(sub, cfg)

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Aeronaves selecionadas", _format_int(len(selected_detail)))
    d2.metric("Trechos", _format_int(len(sub)))
    d3.metric("Passageiros", _format_int(int(sub["_passengers"].sum())))
    d4.metric("Presos", _format_int(int(sub["_prisoners"].sum())))
    d5.metric("Carga (kg)", _format_float_br(float(sub["_cargo"].sum())))
    d6.metric("Operações executadas", _format_int(int(round(total_ops_exec))))

    d7, d8 = st.columns(2)
    d7.metric("Horas de voo (TTV)", _format_float_br(float(sub["_ttv"].sum())))
    d8.metric("Média por aeronave (TTV)", _format_float_br(float(sub["_ttv"].sum()) / max(len(selected_detail), 1)))

    if monthly_aircraft.empty:
        st.info("As aeronaves selecionadas não possuem datas válidas para montar a evolução mensal.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_aircraft["_month"], y=monthly_aircraft["passageiros_total"], mode="lines+markers", name="Passageiros", line=dict(width=4, shape="spline", smoothing=1.0), marker=dict(size=8), yaxis="y1", hovertemplate="<b>Passageiros</b><br>Mês: %{x}<br>Total: %{y:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=monthly_aircraft["_month"], y=monthly_aircraft["presos_total"], mode="lines+markers", name="Presos", line=dict(width=4, shape="spline", smoothing=1.0), marker=dict(size=8), yaxis="y1", hovertemplate="<b>Presos</b><br>Mês: %{x}<br>Total: %{y:.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=monthly_aircraft["_month"], y=monthly_aircraft["carga_total_kg"], mode="lines+markers", name="Carga (kg)", line=dict(width=4, shape="spline", smoothing=1.0), marker=dict(size=8), yaxis="y2", hovertemplate="<b>Carga</b><br>Mês: %{x}<br>Total: %{y:.1f} kg<extra></extra>"))
        fig.add_trace(go.Scatter(x=monthly_aircraft["_month"], y=monthly_aircraft["ttv_total"], mode="lines+markers", name="Horas de voo (TTV)", line=dict(width=4, shape="spline", smoothing=1.0), marker=dict(size=8), yaxis="y1", hovertemplate="<b>TTV</b><br>Mês: %{x}<br>Total: %{y:.1f} h<extra></extra>"))

        left_max = max(1, float(monthly_aircraft[["passageiros_total", "presos_total", "ttv_total"]].fillna(0).to_numpy().max()))
        right_max = max(1, float(monthly_aircraft["carga_total_kg"].fillna(0).max()))

        fig.update_layout(
            template="plotly_white",
            title=f"Evolução mensal consolidada — {', '.join(selected_detail)}",
            xaxis=dict(title="Mês", tickangle=-35),
            yaxis=dict(title="Passageiros / Presos / Horas de voo", side="left", showgrid=True, range=[0, left_max * 1.10], rangemode="tozero"),
            yaxis2=dict(title="Carga (kg)", overlaying="y", side="right", showgrid=False, range=[0, right_max * 1.10], rangemode="tozero"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=620,
            title_font=dict(size=24),
            font=dict(size=16),
            margin=dict(l=30, r=30, t=90, b=30),
        )

        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

        detail_table = monthly_aircraft.rename(columns={"_month": "Mês", "trechos": "Trechos", "passageiros_total": "Passageiros", "presos_total": "Presos", "carga_total_kg": "Carga (kg)", "ttv_total": "Horas de voo (TTV)"})[["Mês", "Trechos", "Passageiros", "Presos", "Carga (kg)", "Horas de voo (TTV)"]]
        st.dataframe(detail_table, width="stretch")

    st.divider()
    st.markdown("### 3) Disponibilidade e uso mensal das aeronaves")

    fleet_monthly = (
        base.dropna(subset=["_month_dt"])
        .groupby([cfg.aircraft, "_asa", "_month_dt", "_month"], dropna=False)
        .agg(ttv_total=("_ttv", "sum"))
        .reset_index()
        .sort_values(["_month_dt", cfg.aircraft])
    )

    if fleet_monthly.empty:
        st.info("Não foi possível calcular a evolução mensal da frota porque não há datas válidas.")
        return

    st.markdown("#### 3.1) Horas de voo por mês e por aeronave")
    selected_for_line = st.multiselect("Escolha quais aeronaves aparecem no gráfico de linhas", options=aircraft_options, default=aircraft_options, key="aircraft_visible_monthly_line_chart")

    if selected_for_line:
        fleet_monthly_line = fleet_monthly[fleet_monthly[cfg.aircraft].astype(str).isin(selected_for_line)].copy()
        fig_line = go.Figure()
        color_map = _build_aircraft_color_map(selected_for_line)

        for ac in selected_for_line:
            sub_ac = fleet_monthly_line[fleet_monthly_line[cfg.aircraft].astype(str) == ac].copy()
            if sub_ac.empty:
                continue
            asa_tipo = str(sub_ac["_asa"].dropna().astype(str).iloc[0]) if sub_ac["_asa"].dropna().shape[0] else ""
            dash_style = "solid" if asa_tipo == "F" else "dash" if asa_tipo == "R" else "dot"
            marker_symbol = "circle" if asa_tipo == "F" else "diamond" if asa_tipo == "R" else "square"
            asa_label = "Asa fixa" if asa_tipo == "F" else "Asa rotativa" if asa_tipo == "R" else "Não informado"
            fig_line.add_trace(go.Scatter(x=sub_ac["_month"], y=sub_ac["ttv_total"], mode="lines+markers", name=f"{ac} ({asa_label})", line=dict(width=4, dash=dash_style, color=color_map[ac], shape="spline", smoothing=1.0), marker=dict(size=7, color=color_map[ac], symbol=marker_symbol), hovertemplate="<b>%{fullData.name}</b><br>Mês: %{x}<br>TTV: %{y:.1f}<extra></extra>"))

        fig_line.update_layout(template="plotly_white", title="Horas de voo por mês e por aeronave", xaxis_title="Mês", yaxis_title="Horas de voo (TTV)", xaxis_tickangle=-35, height=560, title_font=dict(size=24), font=dict(size=16), legend=dict(font=dict(size=13)), margin=dict(l=30, r=30, t=80, b=30))
        st.plotly_chart(fig_line, width="stretch", config=PLOTLY_CONFIG)
    else:
        st.warning("Nenhuma aeronave foi selecionada para o gráfico de linhas.")

    st.divider()
    st.markdown("#### 3.2) Heatmap de disponibilidade mensal")
    selected_for_heatmap = st.multiselect("Escolha quais aeronaves aparecem no heatmap", options=aircraft_options, default=aircraft_options, key="aircraft_visible_availability_heatmap")

    if selected_for_heatmap:
        pivot, month_labels, heatmap_title = _prepare_aircraft_availability_matrix(base=base, cfg=cfg, aircraft_order=selected_for_heatmap)
        fig_heat = _build_availability_heatmap(pivot=pivot, month_labels=month_labels, title=heatmap_title)
        st.plotly_chart(fig_heat, width="stretch", config=PLOTLY_CONFIG)
        st.caption("Disponibilidade observada = percentual de dias do mês em que a aeronave apareceu em pelo menos um trecho.")
    else:
        st.warning("Nenhuma aeronave foi selecionada para o heatmap.")

def render_debug(df: pd.DataFrame, cfg: ColumnConfig) -> None:
    st.subheader("Dados (debug)")
    st.write("Linhas filtradas:", len(df))
    st.write("Mapeamento automático usado:", cfg)

    show_debug = st.checkbox("Exibir amostra dos dados filtrados", value=False, key="show_debug_sample")
    if show_debug:
        debug_rows = st.slider("Quantidade de linhas na amostra", min_value=20, max_value=300, value=80, step=20, key="debug_rows")
        st.dataframe(df.head(debug_rows), width="stretch")

def main() -> None:
    st.title("CAOP - DASHBOARD ESTATÍSTICO")
    apply_global_style()

    with st.sidebar:
        st.header("Arquivo")
        uploaded = st.file_uploader("Arquivo Excel (.xls/.xlsm/.xlsx)", type=["xlsm", "xlsx", "xls"])
        if uploaded is None:
            st.info("Faça upload do Excel para começar.")
            return

        file_bytes = uploaded.getvalue()
        sheets = list_sheets(file_bytes)
        default_sheet = sheets.index(DEFAULT_SHEET) if DEFAULT_SHEET in sheets else 0
        sheet_name = st.selectbox("Aba de dados", options=sheets, index=default_sheet)

    loaded = load_main_table(file_bytes, sheet_name)
    if isinstance(loaded, tuple):
        df_raw, inferred_cfg = loaded[0], loaded[1]
    else:
        df_raw, inferred_cfg = loaded, infer_columns(loaded)

    column_cfg = build_fixed_config(df_raw, inferred_cfg)
    df = apply_column_overrides(df_raw, column_cfg)

    with st.sidebar:
        st.header("Mapa")
        try:
            airports = load_airports_cached(data_dir="data")
            st.success("Base de aeroportos carregada automaticamente da pasta data/.")
        except Exception as e:
            airports = None
            st.warning(f"Base de aeroportos não carregada automaticamente: {e}")

        st.caption("Configuração fixa para todos os usuários.")
        st.write({
            "Demandante": column_cfg.demandante,
            "ASA": column_cfg.asa,
            "TTV": column_cfg.ttv,
            "Ano": column_cfg.year,
            "Data": column_cfg.date,
            "ICAO Origem": column_cfg.icao_from,
            "ICAO Destino": column_cfg.icao_to,
            "Operação (texto)": column_cfg.op_name,
            "Aeronave": column_cfg.aircraft,
            "Relatório de ignorados": "data/ignored_icao_report.csv",
            "Overrides": "data/icao_overrides.csv",
        })

    filter_cfg = render_global_filters(df)
    df_filtered = apply_filters(df, column_cfg, filter_cfg)

    tabs = st.tabs(["Visão geral", "Operações", "Mapa", "Demandantes", "Aeronaves", "Dados"])
    with tabs[0]:
        render_overview(df_filtered, column_cfg)
    with tabs[1]:
        render_operations(df_filtered, column_cfg)
    with tabs[2]:
        render_map(df_filtered, column_cfg, airports)
    with tabs[3]:
        render_demandantes(df_filtered, column_cfg)
    with tabs[4]:
        render_aircraft(df_filtered, column_cfg, airports)
    with tabs[5]:
        render_debug(df_filtered, column_cfg)


if __name__ == "__main__":
    main()
