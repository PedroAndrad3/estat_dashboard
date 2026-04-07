from __future__ import annotations

from dataclasses import dataclass
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
CAOP_AIRPORTS = set()

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

    op_metrics: list[str] | None = None

@dataclass
class FilterConfig:
    years: list[int]
    asa_mode: str
    exclude_caop: bool

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

    return out

def get_year_options(df: pd.DataFrame) -> list[int]:
    if "_year" not in df.columns:
        return []
    years = df["_year"].dropna().astype(int).unique().tolist()
    years.sort()
    return years

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

    if years:
        ensure_year_session(years)
        c1, c2, c3 = st.columns([2, 1, 1])
        with c2:
            st.button("Último ano", on_click=set_last_year, args=(years,), use_container_width=True)
        with c3:
            st.button("Últimos 2", on_click=set_last_two_years, args=(years,), use_container_width=True)
        with c1:
            st.multiselect("Filtro de anos", options=years, key="years_sel")
        selected_years = st.session_state["years_sel"]
    else:
        selected_years = []

    c1, c2 = st.columns([1, 1])
    with c1:
        asa_mode = st.selectbox("Filtro ASA", options=["Todas", "Asa fixa (F)", "Asa rotativa (R)"], index=0)
    with c2:
        exclude_caop = st.checkbox("Excluir demandante CAOP", value=False)

    return FilterConfig(selected_years, asa_mode, exclude_caop)

def apply_filters(df: pd.DataFrame, cfg: ColumnConfig, filters: FilterConfig) -> pd.DataFrame:
    out = df.copy()

    if filters.years and "_year" in out.columns:
        out = out[out["_year"].isin(filters.years)]

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
            st.dataframe(ignored_show, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

def render_operations(df: pd.DataFrame, cfg: ColumnConfig) -> None:
    st.subheader("Operações")

    if df.empty:
        st.warning("Filtro atual retornou 0 linhas.")
        return

    if cfg.op_metrics:
        rows = []
        methods = []
        for col in cfg.op_metrics:
            if col not in df.columns:
                continue
            value, method = count_or_sum(df[col])
            rows.append((col, value))
            methods.append((col, method))

        ops = pd.DataFrame(rows, columns=["tipo", "quantidade"]).sort_values("quantidade", ascending=False)
        methods_df = pd.DataFrame(methods, columns=["tipo", "método"]).sort_values("tipo")

        fig = px.bar(ops.head(25), x="quantidade", y="tipo", orientation="h", title="TOTAIS POR TIPO")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Método usado por coluna (soma vs contagem)"):
            st.dataframe(methods_df, use_container_width=True)

        st.dataframe(ops, use_container_width=True)

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
        fig = go.Figure()

        for categoria in ["Brasil", "Exterior", "CAOP"]:
            sub = valid_points[valid_points["categoria"] == categoria].copy()
            if sub.empty:
                continue

            fig.add_trace(
                go.Scattermapbox(
                    lat=sub["latitude_deg"],
                    lon=sub["longitude_deg"],
                    mode="markers",
                    name=categoria,
                    marker=dict(
                        size=11,
                        color=POINT_COLORS[categoria],
                        opacity=0.95,
                        symbol="circle",
                    ),
                    text=sub["icao"],
                    customdata=np.stack(
                        [
                            sub["visitas"].to_numpy(),
                            sub["iso_country"].fillna("").astype(str).to_numpy(),
                        ],
                        axis=1,
                    ),
                    hovertemplate="<b>%{text}</b><br>visitas: %{customdata[0]}<br>país: %{customdata[1]}<br>categoria: " + categoria + "<extra></extra>",
                )
            )

        fig.update_layout(
            mapbox_style=map_style,
            mapbox_center=center,
            mapbox_zoom=zoom_default,
            height=720,
            margin=dict(l=0, r=0, t=35, b=0),
            legend_title="CATEGORIA",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        c1, c2, c3 = st.columns(3)
        radius = c1.slider("Raio do calor (px)", min_value=10, max_value=90, value=55, step=5)
        clip_q = c2.slider("Corte de outlier (quantil)", min_value=0.90, max_value=0.995, value=0.98, step=0.005)
        opacity = c3.slider("Opacidade", min_value=0.30, max_value=1.0, value=0.90, step=0.05)

        zmax = float(np.quantile(valid_points["visitas"].to_numpy(), float(clip_q))) if len(valid_points) else 1.0
        zmax = max(zmax, 1.0)

        fig = px.density_mapbox(
            valid_points,
            lat="latitude_deg",
            lon="longitude_deg",
            z="visitas",
            radius=int(radius),
            zoom=zoom_default,
            center=center,
            mapbox_style=map_style,
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

        st.plotly_chart(fig, use_container_width=True)

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

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        top_n = st.slider(
            "Top N",
            min_value=1,
            max_value=min(50, max(1, base[cfg.demandante].nunique())),
            value=min(20, max(1, base[cfg.demandante].nunique())),
            step=1,
        )
    with c2:
        exclude_caop_local = st.checkbox("Excluir CAOP (apenas aqui)", value=True)
    with c3:
        group_others = st.checkbox("Agrupar restante em 'OUTROS'", value=True)

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
            use_container_width=True,
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
                use_container_width=True,
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
            use_container_width=True,
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
            st.plotly_chart(style_pie_figure(fig), use_container_width=True, config=PLOTLY_CONFIG)

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
            use_container_width=True,
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
                use_container_width=True,
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

    st.dataframe(tabela, use_container_width=True)

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

def _build_aircraft_color_map(aircraft_names: list[str]) -> dict[str, str]:
    palette = (
        qualitative.Plotly
        + qualitative.Dark24
        + qualitative.Light24
        + qualitative.Alphabet
        + qualitative.Safe
        + qualitative.Vivid
    )

    unique_names = sorted(set(map(str, aircraft_names)))

    color_map = {}
    for i, name in enumerate(unique_names):
        color_map[name] = palette[i % len(palette)]

    return color_map

def render_aircraft(df: pd.DataFrame, cfg: ColumnConfig) -> None:
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

    grouped = _aircraft_grouped(base, cfg)

    total_aeronaves = grouped[cfg.aircraft].nunique()
    total_trechos = int(grouped["trechos"].sum())
    total_ttv = float(grouped["ttv_total"].sum())
    total_passag = float(grouped["passageiros_total"].sum())
    total_presos = float(grouped["presos_total"].sum())
    total_carga = float(grouped["carga_total_kg"].sum())

    # =========================================================
    # 1) VISÃO GERAL DA FROTA
    # =========================================================
    st.markdown("### 1) Visão geral da frota")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Aeronaves no filtro", _format_int(total_aeronaves))
    k2.metric("Trechos", _format_int(total_trechos))
    k3.metric("Horas de voo (TTV)", _format_float_br(total_ttv))
    k4.metric("Passageiros", _format_int(int(total_passag)))
    k5.metric("Presos", _format_int(int(total_presos)))
    k6.metric("Carga (kg)", _format_float_br(total_carga))

    top_n = st.slider(
        "Top N de aeronaves na visão geral",
        min_value=5,
        max_value=min(30, max(5, len(grouped))),
        value=min(10, max(5, len(grouped))),
        step=1,
        key="aircraft_top_n_general",
    )

    top_ttv = grouped.nlargest(top_n, "ttv_total")[[cfg.aircraft, "ttv_total"]].copy()
    top_pass = grouped.nlargest(top_n, "passageiros_total")[[cfg.aircraft, "passageiros_total"]].copy()
    top_carga = grouped.nlargest(top_n, "carga_total_kg")[[cfg.aircraft, "carga_total_kg"]].copy()

    c1, c2 = st.columns(2)
    with c1:
        pie_df = prepare_pie_dataframe(top_ttv, cfg.aircraft, "ttv_total")
        fig = px.pie(
            pie_df,
            names=cfg.aircraft,
            values="ttv_total",
            hole=0.35,
            title="AERONAVES COM MAIS HORAS DE VOO",
        )
        st.plotly_chart(
            style_pie_figure(fig, 480),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c2:
        fig = px.bar(
            top_ttv.sort_values("ttv_total", ascending=True),
            x="ttv_total",
            y=cfg.aircraft,
            orientation="h",
            title="HORAS DE VOO POR AERONAVE",
            labels={"ttv_total": "Horas de voo (TTV)", cfg.aircraft: "Aeronave"},
            text_auto=".1f",
        )
        st.plotly_chart(
            style_plotly_figure(fig, 480),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    c3, c4 = st.columns(2)
    with c3:
        pie_df = prepare_pie_dataframe(top_pass, cfg.aircraft, "passageiros_total")
        fig = px.pie(
            pie_df,
            names=cfg.aircraft,
            values="passageiros_total",
            hole=0.35,
            title="AERONAVES COM MAIS PASSAGEIROS TRANSPORTADOS",
        )
        st.plotly_chart(
            style_pie_figure(fig, 480),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c4:
        fig = px.bar(
            top_pass.sort_values("passageiros_total", ascending=True),
            x="passageiros_total",
            y=cfg.aircraft,
            orientation="h",
            title="PASSAGEIROS TRASNPORTADOS POR AERONAVE",
            labels={"passageiros_total": "Passageiros", cfg.aircraft: "Aeronave"},
            text_auto=".0f",
        )
        st.plotly_chart(
            style_plotly_figure(fig, 480),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    c5, c6 = st.columns(2)
    with c5:
        pie_df = prepare_pie_dataframe(top_carga, cfg.aircraft, "carga_total_kg")
        fig = px.pie(
            pie_df,
            names=cfg.aircraft,
            values="carga_total_kg",
            hole=0.35,
            title="AERONAVES COM MAIS CARGAS TRANSPORTADAS",
        )
        st.plotly_chart(
            style_pie_figure(fig, 480),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )
    with c6:
        fig = px.bar(
            top_carga.sort_values("carga_total_kg", ascending=True),
            x="carga_total_kg",
            y=cfg.aircraft,
            orientation="h",
            title="CARGAS TRANSPORTADA POR AERONAVE",
            labels={"carga_total_kg": "Carga (kg)", cfg.aircraft: "Aeronave"},
            text_auto=".1f",
        )
        st.plotly_chart(
            style_plotly_figure(fig, 480),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    show_table = grouped.rename(columns={
        cfg.aircraft: "Aeronave",
        "asa_label": "Tipo",
        "trechos": "Trechos",
        "ttv_total": "Horas de voo (TTV)",
        "passageiros_total": "Passageiros",
        "presos_total": "Presos",
        "carga_total_kg": "Carga (kg)",
    })
    st.dataframe(show_table, use_container_width=True)

    st.divider()

    # =========================================================
    # 2) FOCO EM UMA AERONAVE ESPECÍFICA
    # =========================================================
    st.markdown("### 2) Foco em uma aeronave específica")

    selected = st.selectbox(
        "Selecione a aeronave",
        options=grouped[cfg.aircraft].astype(str).tolist(),
        key="aircraft_selected_detail",
    )

    sub = base[base[cfg.aircraft].astype(str) == str(selected)].copy()
    monthly_aircraft = _monthly_aircraft_metrics(sub, cfg)

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Aeronave", str(selected))
    d2.metric("Trechos", _format_int(len(sub)))
    d3.metric("Passageiros", _format_int(int(sub["_passengers"].sum())))
    d4.metric("Presos", _format_int(int(sub["_prisoners"].sum())))
    d5.metric("Carga (kg)", _format_float_br(float(sub["_cargo"].sum())))
    d6.metric("Horas de voo (TTV)", _format_float_br(float(sub["_ttv"].sum())))

    if monthly_aircraft.empty:
        st.info("Essa aeronave não possui datas válidas para montar a evolução mensal.")
    else:
        fig = go.Figure()

        # 1) Passageiros
        fig.add_trace(
            go.Scatter(
                x=monthly_aircraft["_month"],
                y=monthly_aircraft["passageiros_total"],
                mode="lines+markers+text",
                name="Passageiros",
                text=monthly_aircraft["passageiros_total"].fillna(0).round(0).astype(int),
                textposition="top center",
                hovertemplate="<b>Passageiros</b><br>Mês: %{x}<br>Total: %{y:.0f}<extra></extra>",
                yaxis="y1",
            )
        )

        # 2) Presos
        fig.add_trace(
            go.Scatter(
                x=monthly_aircraft["_month"],
                y=monthly_aircraft["presos_total"],
                mode="lines+markers+text",
                name="Presos",
                text=monthly_aircraft["presos_total"].fillna(0).round(0).astype(int),
                textposition="top center",
                hovertemplate="<b>Presos</b><br>Mês: %{x}<br>Total: %{y:.0f}<extra></extra>",
                yaxis="y1",
            )
        )

        # 3) Carga
        fig.add_trace(
            go.Scatter(
                x=monthly_aircraft["_month"],
                y=monthly_aircraft["carga_total_kg"],
                mode="lines+markers+text",
                name="Carga (kg)",
                text=monthly_aircraft["carga_total_kg"].fillna(0).round(1),
                textposition="top center",
                hovertemplate="<b>Carga</b><br>Mês: %{x}<br>Total: %{y:.1f} kg<extra></extra>",
                yaxis="y2",
            )
        )

        # 4) TTV
        fig.add_trace(
            go.Scatter(
                x=monthly_aircraft["_month"],
                y=monthly_aircraft["ttv_total"],
                mode="lines+markers+text",
                name="Horas de voo (TTV)",
                text=monthly_aircraft["ttv_total"].fillna(0).round(1),
                textposition="top center",
                hovertemplate="<b>TTV</b><br>Mês: %{x}<br>Total: %{y:.1f} h<extra></extra>",
                yaxis="y1",
            )
        )

        fig.update_layout(
            title=f"EVOLULÇÃO MENSAL CONSOLIDADA — {selected}",
            xaxis=dict(
                title="Mês",
                tickangle=-35,
            ),
            yaxis=dict(
                title="Passageiros / Presos / Horas de voo",
                side="left",
                showgrid=True,
            ),
            yaxis2=dict(
                title="Carga (kg)",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            height=560,
            title_font_size=22,
            font=dict(size=16),
            margin=dict(l=20, r=20, t=90, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        detail_table = monthly_aircraft.rename(columns={
            "_month": "Mês",
            "trechos": "Trechos",
            "passageiros_total": "Passageiros",
            "presos_total": "Presos",
            "carga_total_kg": "Carga (kg)",
            "ttv_total": "Horas de voo (TTV)",
        })[["Mês", "Trechos", "Passageiros", "Presos", "Carga (kg)", "Horas de voo (TTV)"]]

        st.dataframe(detail_table, use_container_width=True)

    st.divider()

    # =========================================================
    # 3) DISPONIBILIDADE / COMPARATIVO MENSAL
    # =========================================================
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

    aircraft_options = sorted(fleet_monthly[cfg.aircraft].astype(str).unique().tolist())
    default_aircraft = aircraft_options.copy()

    selected_for_chart = st.multiselect(
        "Escolha quais aeronaves aparecem no gráfico mensal",
        options=aircraft_options,
        default=default_aircraft,
        key="aircraft_visible_monthly_chart",
    )

    fleet_monthly = fleet_monthly[fleet_monthly[cfg.aircraft].astype(str).isin(selected_for_chart)].copy()

    if fleet_monthly.empty:
        st.warning("Nenhuma aeronave foi selecionada para o gráfico.")
        return

    fig = go.Figure()

    aircraft_list_for_colors = sorted(fleet_monthly[cfg.aircraft].astype(str).unique().tolist())
    color_map = _build_aircraft_color_map(aircraft_list_for_colors)

    for ac in aircraft_list_for_colors:
        sub_ac = fleet_monthly[fleet_monthly[cfg.aircraft].astype(str) == ac].copy()
        asa_tipo = str(sub_ac["_asa"].dropna().astype(str).iloc[0]) if sub_ac["_asa"].dropna().shape[0] else ""
        dash_style = "solid" if asa_tipo == "F" else "dash" if asa_tipo == "R" else "dot"
        asa_label = "Asa fixa" if asa_tipo == "F" else "Asa rotativa" if asa_tipo == "R" else "Não informado"

        fig.add_trace(
            go.Scatter(
                x=sub_ac["_month"],
                y=sub_ac["ttv_total"],
                mode="lines+markers",
                name=f"{ac} ({asa_label})",
                line=dict(
                    width=3,
                    dash=dash_style,
                    color=color_map[ac],
                ),
                marker=dict(
                    size=7,
                    color=color_map[ac],
                ),
                hovertemplate="<b>%{fullData.name}</b><br>Mês: %{x}<br>TTV: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="HORAS DE VOO POR AERONAVE POR MÊS",
        xaxis_title="Mês",
        yaxis_title="Horas de voo (TTV)",
        xaxis_tickangle=-35,
    )
    st.plotly_chart(_update_fig_layout(fig, 560), use_container_width=True)

    disp_month = (
        base.dropna(subset=["_month_dt"])
        .groupby(["_month_dt", "_month"], dropna=False)
        .agg(
            aeronaves_ativas=(cfg.aircraft, lambda s: s.astype(str).nunique()),
            trechos=(cfg.aircraft, "size"),
            ttv_total=("_ttv", "sum"),
        )
        .reset_index()
        .sort_values("_month_dt")
    )
    disp_month["disponibilidade_observada_pct"] = disp_month["aeronaves_ativas"] / total_aeronaves * 100.0

    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.bar(
            disp_month,
            x="_month",
            y="aeronaves_ativas",
            title="AERONAVES ATIVAS POR MÊS",
            labels={"_month": "Mês", "aeronaves_ativas": "Aeronaves ativas"},
            text_auto=".0f",
        )
        fig2.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(_update_fig_layout(fig2), use_container_width=True)

    with c2:
        fig3 = px.line(
            disp_month,
            x="_month",
            y="disponibilidade_observada_pct",
            markers=True,
            title="DISPONIBILIDADE OBSERVADA DA FROTA POR MÊS",
            labels={"_month": "Mês", "disponibilidade_observada_pct": "Disponibilidade (%)"},
        )
        fig3.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(_update_fig_layout(fig3), use_container_width=True)

    peak = disp_month.loc[disp_month["aeronaves_ativas"].idxmax()]
    low = disp_month.loc[disp_month["aeronaves_ativas"].idxmin()]
    peak_ttv = disp_month.loc[disp_month["ttv_total"].idxmax()]

    st.markdown(
        f"""
A disponibilidade observada foi calculada a partir das aeronaves que efetivamente apareceram em trechos registrados em cada mês.

O pico de disponibilidade ocorreu em **{peak['_month']}**, com **{int(peak['aeronaves_ativas'])} aeronaves ativas**
(**{peak['disponibilidade_observada_pct']:.1f}%** da frota no filtro).

O menor nível ocorreu em **{low['_month']}**, com **{int(low['aeronaves_ativas'])} aeronaves ativas**
(**{low['disponibilidade_observada_pct']:.1f}%**).

Já o mês de maior intensidade operacional foi **{peak_ttv['_month']}**, com **{peak_ttv['ttv_total']:.1f} horas de voo**
e **{int(peak_ttv['trechos'])} trechos**.
        """
    )

def render_debug(df: pd.DataFrame, cfg: ColumnConfig) -> None:
    st.subheader("Dados (debug)")
    st.write("Linhas filtradas:", len(df))
    st.write("Mapeamento automático usado:", cfg)
    st.dataframe(df.head(200), use_container_width=True)

def main() -> None:
    st.title("CAOP - DASHBOARD ESTATÍSTICO")

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
            airports = load_airports(data_dir="data")
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
        render_aircraft(df_filtered, column_cfg)
    with tabs[5]:
        render_debug(df_filtered, column_cfg)


if __name__ == "__main__":
    main()
