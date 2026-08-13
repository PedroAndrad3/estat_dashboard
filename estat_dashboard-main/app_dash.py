
from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

from data_utils import load_total_table
from airports_utils import load_airports


XLS_PATH_DEFAULT = "NOVO CONTROLE ESTATISTICO TIL 2025, de 31122024.xlsm"
SHEET_DEFAULT = "_____TOTAL_____"

loaded = load_total_table(XLS_PATH_DEFAULT, sheet_name=SHEET_DEFAULT)
df = loaded[0] if isinstance(loaded, tuple) else loaded

# DATA
if "DATA" in df.columns:
    df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce", dayfirst=True)

# demandante_nome
if "demandante_nome" not in df.columns:
    for cand in ["DEM.", "Demandante", "DEMANDANTE", "demandante_nome"]:
        if cand in df.columns:
            df["demandante_nome"] = df[cand].astype(str).str.strip()
            break

# demandante
if "demandante" not in df.columns and "demandante_nome" in df.columns:
    df["demandante"] = df["demandante_nome"]

# demandante_pf
if "demandante_pf" not in df.columns:
    if "_is_pf" in df.columns:
        df["demandante_pf"] = df["_is_pf"].fillna(False).astype(bool)
    else:
        df["demandante_pf"] = False

# ano
if "ano" not in df.columns:
    if "_year" in df.columns:
        df["ano"] = df["_year"]
    elif "ANO" in df.columns:
        df["ano"] = pd.to_numeric(df["ANO"], errors="coerce")

# TTV_h
if "TTV_h" not in df.columns:
    if "_ttv" in df.columns:
        df["TTV_h"] = pd.to_numeric(df["_ttv"], errors="coerce").fillna(0.0)
    elif "TTV" in df.columns:
        df["TTV_h"] = pd.to_numeric(df["TTV"], errors="coerce").fillna(0.0)
    else:
        df["TTV_h"] = 0.0

# aeronave
if "AERONAVE (matrícula)" not in df.columns:
    for cand in ["AERONAVE (matrícula)", "AERONAVE", "Prefixo", "MATRÍCULA", "MATRICULA"]:
        if cand in df.columns:
            df["AERONAVE (matrícula)"] = df[cand].astype(str)
            break

min_d = df["DATA"].min().date()
max_d = df["DATA"].max().date()

demandantes = sorted(df["DEM."].dropna().unique().tolist())
aeronaves = sorted(df["AERONAVE (matrícula)"].dropna().astype(str).unique().tolist())

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard TIL / CAOP"

app.layout = dbc.Container(
    [
        html.H2("Dashboard – Controle Estatístico (TIL)"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6("Filtros"),
                                html.Label("Período"),
                                dcc.DatePickerRange(
                                    id="f-date",
                                    min_date_allowed=min_d,
                                    max_date_allowed=max_d,
                                    start_date=min_d,
                                    end_date=max_d,
                                    display_format="DD/MM/YYYY",
                                ),
                                html.Br(),
                                html.Br(),
                                html.Label("Escopo"),
                                dcc.RadioItems(
                                    id="f-pf",
                                    options=[
                                        {"label": "Todos", "value": "all"},
                                        {"label": "Somente PF", "value": "pf"},
                                        {"label": "Somente fora PF", "value": "out"},
                                    ],
                                    value="all",
                                    inline=True,
                                ),
                                html.Hr(),
                                html.Label("Demandante"),
                                dcc.Dropdown(
                                    id="f-dem",
                                    options=[{"label": d, "value": d} for d in demandantes],
                                    multi=True,
                                    placeholder="Selecione...",
                                ),
                                html.Br(),
                                html.Label("Aeronave (matrícula)"),
                                dcc.Dropdown(
                                    id="f-aer",
                                    options=[{"label": a, "value": a} for a in aeronaves],
                                    multi=True,
                                    placeholder="Selecione...",
                                ),
                            ]
                        )
                    ),
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(dbc.Card(dbc.CardBody([html.Div(id="kpi-atend")]))),
                                dbc.Col(dbc.Card(dbc.CardBody([html.Div(id="kpi-dem")]))),
                                dbc.Col(dbc.Card(dbc.CardBody([html.Div(id="kpi-pf")]))),
                                dbc.Col(dbc.Card(dbc.CardBody([html.Div(id="kpi-ttv")]))),
                            ]
                        ),
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="g-year"), width=6),
                                dbc.Col(dcc.Graph(id="g-topdem"), width=6),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(dcc.Graph(id="g-pfpie"), width=6),
                                dbc.Col(dcc.Graph(id="g-nat"), width=6),
                            ]
                        ),
                        dbc.Row([dbc.Col(dcc.Graph(id="g-map"), width=12)]),
                    ],
                    width=9,
                ),
            ]
        ),
    ],
    fluid=True,
)

def airport_visit_counts(df: pd.DataFrame) -> pd.DataFrame:
    possible_pairs = [
        ("TRECHO (DE)", "TRECHO (PARA)"),
        ("ICAO Origem", "ICAO Destino"),
        ("origem", "destino"),
    ]

    origem_col = destino_col = None
    for o, d in possible_pairs:
        if o in df.columns and d in df.columns:
            origem_col, destino_col = o, d
            break

    if origem_col is None or destino_col is None:
        return pd.DataFrame(columns=["icao", "visitas"])

    s = pd.concat([df[origem_col], df[destino_col]], ignore_index=True).dropna()
    s = s.astype(str).str.strip().str.upper()
    s = s[(s.str.len() == 4) & (s != "ZZZZ")]

    return s.value_counts().rename_axis("icao").reset_index(name="visitas")

def _apply_filters(df: pd.DataFrame, start: date, end: date, pf_mode: str, dem_sel, aer_sel) -> pd.DataFrame:
    dff = df[(df["DATA"].dt.date >= start) & (df["DATA"].dt.date <= end)].copy()

    if pf_mode == "pf":
        dff = dff[dff["demandante_pf"]]
    elif pf_mode == "out":
        dff = dff[~dff["demandante_pf"]]

    if dem_sel:
        dff = dff[dff["demandante_nome"].isin(dem_sel)]
    if aer_sel:
        dff = dff[dff["AERONAVE (matrícula)"].astype(str).isin(aer_sel)]
    return dff


@app.callback(
    Output("kpi-atend", "children"),
    Output("kpi-dem", "children"),
    Output("kpi-pf", "children"),
    Output("kpi-ttv", "children"),
    Output("g-year", "figure"),
    Output("g-topdem", "figure"),
    Output("g-pfpie", "figure"),
    Output("g-nat", "figure"),
    Output("g-map", "figure"),
    Input("f-date", "start_date"),
    Input("f-date", "end_date"),
    Input("f-pf", "value"),
    Input("f-dem", "value"),
    Input("f-aer", "value"),
)
def update(start_date, end_date, pf_mode, dem_sel, aer_sel):
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()

    dff = _apply_filters(df, start, end, pf_mode, dem_sel, aer_sel)

    total_atend = len(dff)
    demandantes_dist = dff["DEM."].nunique()
    pf_share = (dff["demandante_pf"].mean() * 100.0) if total_atend else 0.0
    ttv_h = dff["TTV_h"].sum()

    kpi1 = html.Div([html.H5(f"{total_atend:,}".replace(",", ".")), html.Small("Atendimentos (linhas)")])
    kpi2 = html.Div([html.H5(f"{demandantes_dist}"), html.Small("Demandantes distintos")])
    kpi3 = html.Div([html.H5(f"{pf_share:.1f}%"), html.Small("% PF (por demandante)")])
    kpi4 = html.Div([html.H5(f"{ttv_h:,.1f} h".replace(",", "X").replace(".", ",").replace("X", ".")), html.Small("Horas TTV (soma)")])

    # garante uma coluna ASA padronizada
    if "_asa" in dff.columns:
        asa_col = "_asa"
    elif "ASA (F ou R)" in dff.columns:
        asa_col = "ASA (F ou R)"
    else:
        asa_col = None

    if asa_col:
        dff_plot = dff.copy()
        dff_plot["asa_plot"] = (
            dff_plot[asa_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"F": "Asa fixa", "R": "Asa rotativa"})
        )
    else:
        dff_plot = dff.copy()
        dff_plot["asa_plot"] = "Não informado"

    by_year_ttv = (
        dff_plot.dropna(subset=["ano"])
        .groupby(["ano", "asa_plot"], dropna=True)["TTV_h"]
        .sum()
        .reset_index()
    )

    fig_year = px.bar(
        by_year_ttv,
        x="ano",
        y="TTV_h",
        color="asa_plot",
        barmode="group",
        title="TTV por ano — asa fixa vs asa rotativa",
        labels={
            "ano": "Ano",
            "TTV_h": "Horas de voo (TTV)",
            "asa_plot": "Categoria",
        },
        text_auto=".1f",
    )
    totais_ano = (
        dff_plot.dropna(subset=["ano"])
        .groupby("ano", dropna=True)["TTV_h"]
        .sum()
        .reset_index()
    )

    fig_year.add_scatter(
        x=totais_ano["ano"],
        y=totais_ano["TTV_h"],
        mode="lines+markers+text",
        name="TTV total",
        text=[f"{v:.1f}" for v in totais_ano["TTV_h"]],
        textposition="top center",
    )

    top_dem = (
        dff.groupby(["demandante_nome", "demandante_pf"], dropna=True)
        .size()
        .reset_index(name="atendimentos")
        .sort_values("atendimentos", ascending=False)
        .head(15)
    )
    fig_top = px.bar(top_dem, x="atendimentos", y="demandante_nome", orientation="h", title="Top 15 demandantes")

    share_pf = (
        dff.assign(grupo=dff["demandante_pf"].map({True: "PF", False: "Fora PF"}))
        .groupby("grupo")
        .size()
        .reset_index(name="atendimentos")
    )
    fig_pf = px.pie(share_pf, values="atendimentos", names="grupo", title="PF vs Fora PF (participação)")

    nat = dff["NAT."].astype(str).str.strip().replace({"nan": ""})
    nat = nat[nat != ""]
    nat_cnt = nat.value_counts().head(12).reset_index()
    nat_cnt.columns = ["natureza", "qtd"]
    fig_nat = px.bar(nat_cnt, x="qtd", y="natureza", orientation="h", title="Top naturezas (NAT.)")

    vis = airport_visit_counts(dff)
    try:
        airports = load_airports(None, allow_download=False)
        vis2 = vis.merge(airports, on="icao", how="left").dropna(subset=["latitude_deg", "longitude_deg"])
        if vis2.empty:
            fig_map = px.scatter_mapbox(title="Sem coordenadas para os ICAOs filtrados.")
        else:
            zmax = float(vis2["visitas"].quantile(0.95)) if len(vis2) else 1.0
            zmax = max(zmax, 1.0)
            center = {
                "lat": float(vis2["latitude_deg"].mean()),
                "lon": float(vis2["longitude_deg"].mean()),
            }
            fig_map = px.density_mapbox(
                vis2,
                lat="latitude_deg",
                lon="longitude_deg",
                z="visitas",
                radius=35,
                hover_name="icao",
                hover_data={"visitas": True, "iso_country": True},
                zoom=4,
                center=center,
                mapbox_style="open-street-map",
                title="Densidade de visitas (origem + destino)",
                range_color=(0, zmax),
            )
            fig_map.update_traces(opacity=0.85)
            fig_map.update_layout(height=650, margin=dict(l=0, r=0, t=45, b=0))
    except Exception:
        fig_map = px.scatter_mapbox(title="Falha ao carregar base de aeroportos.")

    return kpi1, kpi2, kpi3, kpi4, fig_year, fig_top, fig_pf, fig_nat, fig_map


if __name__ == "__main__":
    app.run(debug=True)
