from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


R = 6378137.0  # WebMercator radius


def ll_to_merc(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def merc_to_ll(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(x / R)
    lat = np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)
    return lat, lon


def build_grid(df: pd.DataFrame, lat_col: str, lon_col: str, w_col: str, cell_km: float = 60.0) -> Dict:
    cell_m = float(cell_km) * 1000.0
    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    w = df[w_col].to_numpy(dtype=float)

    x, y = ll_to_merc(lat, lon)
    ix = np.floor(x / cell_m).astype(int)
    iy = np.floor(y / cell_m).astype(int)

    g = pd.DataFrame({"ix": ix, "iy": iy, "w": w})
    agg = g.groupby(["ix", "iy"], as_index=False)["w"].sum()

    if agg.empty:
        return {"agg": agg, "cell_m": cell_m, "Z": np.zeros((0, 0)), "ix_min": 0, "iy_min": 0}

    ix_min, ix_max = int(agg["ix"].min()), int(agg["ix"].max())
    iy_min, iy_max = int(agg["iy"].min()), int(agg["iy"].max())
    nx = ix_max - ix_min + 1
    ny = iy_max - iy_min + 1
    Z = np.zeros((ny, nx), dtype=float)

    for r in agg.itertuples(index=False):
        j = int(r.iy - iy_min)
        i = int(r.ix - ix_min)
        Z[j, i] = float(r.w)

    return {"agg": agg, "cell_m": cell_m, "Z": Z, "ix_min": ix_min, "iy_min": iy_min}


def grid_geojson(grid: Dict) -> Tuple[Dict, List[str], List[float]]:
    agg = grid["agg"]
    cell_m = grid["cell_m"]
    features = []
    ids = []
    values = []

    for r in agg.itertuples(index=False):
        cx0 = r.ix * cell_m
        cy0 = r.iy * cell_m
        cx1 = cx0 + cell_m
        cy1 = cy0 + cell_m

        lat0, lon0 = merc_to_ll(np.array([cx0]), np.array([cy0]))
        lat1, lon1 = merc_to_ll(np.array([cx1]), np.array([cy0]))
        lat2, lon2 = merc_to_ll(np.array([cx1]), np.array([cy1]))
        lat3, lon3 = merc_to_ll(np.array([cx0]), np.array([cy1]))

        fid = f"{int(r.ix)}_{int(r.iy)}"
        features.append(
            {
                "type": "Feature",
                "id": fid,
                "properties": {"id": fid, "w": float(r.w)},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [float(lon0[0]), float(lat0[0])],
                        [float(lon1[0]), float(lat1[0])],
                        [float(lon2[0]), float(lat2[0])],
                        [float(lon3[0]), float(lat3[0])],
                        [float(lon0[0]), float(lat0[0])],
                    ]],
                },
            }
        )
        ids.append(fid)
        values.append(float(r.w))

    return {"type": "FeatureCollection", "features": features}, ids, values


def choropleth_zones(grid: Dict, map_style: str, center: Dict, zoom: float, color_scale: str = "Turbo",
                     clip_q: float = 0.98, opacity: float = 0.9, height: int = 720) -> go.Figure:
    geo, ids, values = grid_geojson(grid)
    z = np.array(values, dtype=float) if len(values) else np.array([0.0])
    zmax = float(np.quantile(z, clip_q)) if len(z) else 1.0
    zmax = max(zmax, 1.0)

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=geo,
            locations=ids,
            z=values,
            colorscale=color_scale,
            zmin=0,
            zmax=zmax,
            marker_opacity=opacity,
            marker_line_width=0,
            colorbar_title="visitas",
        )
    )
    fig.update_layout(
        mapbox_style=map_style,
        mapbox_center=center,
        mapbox_zoom=zoom,
        height=height,
        margin=dict(l=0, r=0, t=35, b=0),
    )
    return fig


def contour_traces_from_grid(grid: Dict, levels: int = 12, clip_q: float = 0.98) -> List[go.Scattermapbox]:
    """
    Gera isolinhas (contornos) a partir da matriz Z (no espaço WebMercator) e converte para lat/lon.
    Compatível com mudanças de API do Matplotlib (usa cs.allsegs em vez de cs.collections).
    """
    Z = grid["Z"]
    if Z.size == 0:
        return []

    cell_m = grid["cell_m"]
    ix_min = grid["ix_min"]
    iy_min = grid["iy_min"]

    z = Z.flatten()
    zpos = z[z > 0]
    zmax = float(np.quantile(zpos, clip_q)) if zpos.size else float(np.max(z))
    zmax = max(zmax, 1.0)

    ny, nx = Z.shape
    xs = (np.arange(nx) + 0.5 + ix_min) * cell_m
    ys = (np.arange(ny) + 0.5 + iy_min) * cell_m
    X, Y = np.meshgrid(xs, ys)

    zmin_pos = float(np.min(zpos)) if zpos.size else 1.0
    if zmin_pos <= 0:
        zmin_pos = 1.0
    levs = np.linspace(zmin_pos, zmax, max(3, int(levels)))

    fig, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=levs)
    plt.close(fig)

    traces: List[go.Scattermapbox] = []

    # Matplotlib moderno: use cs.allsegs (lista por nível -> lista de segmentos Nx2)
    allsegs = getattr(cs, "allsegs", None)
    if allsegs is not None:
        for level_segs in allsegs:
            for seg in level_segs:
                if seg is None or len(seg) < 2:
                    continue
                x = np.asarray(seg)[:, 0]
                y = np.asarray(seg)[:, 1]
                lat, lon = merc_to_ll(x, y)
                traces.append(
                    go.Scattermapbox(
                        lat=lat,
                        lon=lon,
                        mode="lines",
                        line=dict(width=1),
                        hoverinfo="skip",
                        opacity=0.85,
                    )
                )
        return traces

    # Fallback antigo (se existir)
    if hasattr(cs, "collections"):
        for col in cs.collections:
            for path in col.get_paths():
                v = path.vertices
                if v.shape[0] < 2:
                    continue
                x = v[:, 0]
                y = v[:, 1]
                lat, lon = merc_to_ll(x, y)
                traces.append(
                    go.Scattermapbox(
                        lat=lat,
                        lon=lon,
                        mode="lines",
                        line=dict(width=1),
                        hoverinfo="skip",
                        opacity=0.85,
                    )
                )
    return traces
