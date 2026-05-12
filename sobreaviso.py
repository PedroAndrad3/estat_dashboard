import matplotlib.pyplot as plt
import numpy as np

# =========================
# DADOS
# =========================
meses = ["OUT/2025", "NOV/2025", "DEZ/2025", "JAN/2026", "FEV/2026", "MAR/2026"]

dados = {
    "APOIO OPERACIONAL": {
        "Resposta não": [42, 39, 42, 43, 38, 33],
        "Resposta sim": [20, 21, 19, 19, 18, 21],
        "Sem resposta": [0, 0, 0, 0, 0, 8],
    },
    "AVIÕES - JATO - KING e CARAVAN": {
        "Resposta não": [65, 61, 76, 76, 74, 55],
        "Resposta sim": [26, 57, 48, 27, 35, 45],
        "Sem resposta": [0, 0, 0, 0, 3, 24],
    }
    #"HELICÓPTEROS E AEROTÁTICOS": {
    #    "Resposta não": [120, 112, 111, 103, 89, 87],
    #    "Resposta sim": [4, 8, 13, 21, 23, 26],
    #    "Sem resposta": [0, 0, 0, 0, 0, 11],
    #}
}

cores = {
    "Resposta não": "#44964d",   # verde
    "Resposta sim": "#b0b0b7",   # cinza
    "Sem resposta": "#347b9b"    # azul
}

# =========================
# CONFIGURAÇÃO DA FIGURA
# =========================
fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
fig.suptitle("Comparativo de Sobreavisos por Equipe (OUT/2025 a MAR/2026)",
             fontsize=16, fontweight="bold")

for ax, (equipe, valores) in zip(axes, dados.items()):
    x = np.arange(len(meses))

    resposta_nao = np.array(valores["Resposta não"])
    resposta_sim = np.array(valores["Resposta sim"])
    sem_resposta = np.array(valores["Sem resposta"])

    b1 = ax.bar(x, resposta_nao, label="Resposta não", color=cores["Resposta não"])
    b2 = ax.bar(x, resposta_sim, bottom=resposta_nao, label="Resposta sim", color=cores["Resposta sim"])
    b3 = ax.bar(x, sem_resposta, bottom=resposta_nao + resposta_sim, label="Sem resposta", color=cores["Sem resposta"])

    ax.set_title(equipe, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(meses, rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Rótulos nas barras
    for i in range(len(x)):
        if resposta_nao[i] > 0:
            ax.text(x[i], resposta_nao[i] / 2, str(resposta_nao[i]),
                    ha="center", va="center", fontsize=9, color="black")

        if resposta_sim[i] > 0:
            ax.text(x[i], resposta_nao[i] + resposta_sim[i] / 2, str(resposta_sim[i]),
                    ha="center", va="center", fontsize=9, color="black")

        if sem_resposta[i] > 0:
            ax.text(x[i], resposta_nao[i] + resposta_sim[i] + sem_resposta[i] / 2, str(sem_resposta[i]),
                    ha="center", va="center", fontsize=9, color="white")

axes[0].set_ylabel("Quantidade", fontsize=11)

# Legenda única
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.93))

plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.show()

# Para salvar em arquivo, descomente:
# plt.savefig("comparativo_sobreavisos_out2025_mar2026.png", dpi=300, bbox_inches="tight")