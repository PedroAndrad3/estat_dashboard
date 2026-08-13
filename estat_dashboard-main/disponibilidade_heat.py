import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# =========================
# DADOS
# =========================
meses = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]

dados_brutos = {
    "PRAAC":  ["0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "37%", "52%"],
    "PSIRB*": ["0%", "0%", "0%", "93%", "100%", "100%", "100%", "100%", "97%", "0%", "0%", "0%"],
    "PRBSI":  ["0%", "36%", "71%", "100%", "100%", "100%", "55%", "26%", "7%", "100%", "90%", "19%"],
    "PSCAV":  ["0%", "4%", "0%", "0%", "0%", "0%", "0%", "0%", "33%", "77%", "70%", "100%"],
    "PSDDF":  ["29%", "89%", "13%", "100%", "100%", "100%", "100%", "45%", "0%", "35%", "100%", "39%"],
    "PRHFA":  ["100%", "64%", "0%", "0%", "84%", "100%", "100%", "3%", "0%", "0%", "0%", "0%"],
    "PRHFC":  ["0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%"],
    "PRHFD":  ["0%", "77%", "0%", "43%", "77%", "37%", "52%", "0%", "0%", "0%", "0%", "0%"],
    "PTHZH":  ["0%", "64%", "100%", "100%", "100%", "100%", "35%", "0%", "0%", "0%", "0%", "0%"],
    "PRLEE":  ["0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%", "0%"],
    "PRHFV":  ["100%", "43%", "0%", "0%", "0%", "0%", "0%", "0%", "100%", "0%", "0%", "0%"],
    "PSPFG*": ["-", "-", "-", "-", "61%", "53%", "81%", "100%", "100%", "35%", "100%", "13%"],
    "PSPFH*": ["-", "-", "-", "-", "-", "-", "0%", "100%", "100%", "100%", "100%", "100%"],
}

# =========================
# FUNÇÃO DE CONVERSÃO
# =========================
def converter_percentual(valor):
    """
    Converte valores como:
    '37%', '61', '-', '—', '' em float.
    Tudo que não for numérico vira 0.
    """
    if pd.isna(valor):
        return 0.0

    s = str(valor).strip().replace("%", "").replace(",", ".")

    if s in ["", "-", "–", "—", "None", "nan"]:
        return 0.0

    try:
        return float(s)
    except ValueError:
        return 0.0

# =========================
# DATAFRAME
# =========================
df = pd.DataFrame(
    {aeronave: [converter_percentual(v) for v in valores]
     for aeronave, valores in dados_brutos.items()},
    index=meses
).T  # Linhas = aeronaves | Colunas = meses

# =========================
# HEATMAP
# =========================
fig, ax = plt.subplots(figsize=(16, 7))

# Mapa de cores: branco -> verde claro -> verde médio -> verde mais forte
cmap = mcolors.LinearSegmentedColormap.from_list(
    "disponibilidade_verde",
    ["#f2f2f2", "#d9f2d9", "#8fd19e", "#34a853"]
)

im = ax.imshow(df.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")

# Título
ax.set_title("DISPONIBILIDADE MENSAL POR AERONAVE - 2025", fontsize=18, fontweight="bold", pad=20)

# Rótulos dos eixos
ax.set_xticks(np.arange(len(df.columns)))
ax.set_xticklabels(df.columns, fontsize=11, fontweight="bold")

ax.set_yticks(np.arange(len(df.index)))
ax.set_yticklabels(df.index, fontsize=11, fontweight="bold")

# Linhas de grade entre as células
ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8)
ax.tick_params(which="minor", bottom=False, left=False)

# Escrever os percentuais dentro de cada célula
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        valor = df.iloc[i, j]
        texto = f"{int(round(valor))}%"
        ax.text(
            j, i, texto,
            ha="center", va="center",
            color="black",
            fontsize=10,
            fontweight="bold"
        )

# Barra de cor
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Disponibilidade (%)", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()

# Para salvar:
# plt.savefig("heatmap_disponibilidade_2025.png", dpi=300, bbox_inches="tight")