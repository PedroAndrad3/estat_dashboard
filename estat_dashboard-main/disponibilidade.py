import matplotlib.pyplot as plt
import pandas as pd

# Meses do ano
meses = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]

# Dados extraídos da tabela
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
    "PSPFG*": ["-", "-", "-", "-", "61", "53", "81", "100", "100", "35", "100", "13%"],
    "PSPFH*": ["-", "-", "-", "-", "-", "-", "0%", "100%", "100%", "100%", "100%", "100%"],
}

def converter_percentual(valor):
    """
    Converte strings como '37%', '61', '-', '—' etc. para float.
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

# Monta DataFrame com os dados tratados
df = pd.DataFrame(
    {aeronave: [converter_percentual(v) for v in valores]
     for aeronave, valores in dados_brutos.items()},
    index=meses
)

# Configuração da figura
plt.figure(figsize=(16, 8))

# Gera cores únicas usando tab20
cores = plt.cm.tab20.colors

# Plota cada aeronave
for i, aeronave in enumerate(df.columns):
    plt.plot(
        df.index,
        df[aeronave],
        marker="o",
        linewidth=2,
        markersize=5,
        label=aeronave,
        color=cores[i % len(cores)]
    )

# Ajustes visuais
plt.title("Disponibilidade Mensal por Aeronave - 2025", fontsize=16, fontweight="bold")
plt.xlabel("Mês", fontsize=12)
plt.ylabel("Disponibilidade (%)", fontsize=12)
plt.ylim(0, 105)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(title="Aeronaves", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

# Exibe o gráfico
plt.show()

# Se quiser salvar a imagem, descomente a linha abaixo:
# plt.savefig("disponibilidade_aeronaves_2025.png", dpi=300, bbox_inches="tight")