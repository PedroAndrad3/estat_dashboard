CAOP Estat Dashboard (Streamlit) — v3 (robusto + layout melhor)

Rodar:
  pip install -r requirements.txt
  streamlit run app_streamlit.py

Correções nesta versão:
- Soma TTV: parser robusto (número, tempo Excel, "hh:mm", "1,5", etc).
- Botões de anos: agora funcionam (usa session_state + callbacks; sem erro de set após widget).
- Layout: KPIs e tabs voltaram a ficar mais limpos.
- Leitura do Excel: suporta .xlsx/.xlsm (ZIP) e também .xls (OLE) via xlrd automaticamente.
- Colunas configuráveis em "Configuração avançada" (sidebar), mas com defaults automáticos.

Mapas:
- Precisa do airports.csv (OurAirports) para coordenadas ICAO.
  Faça upload na sidebar ou coloque em data/airports.csv.

Obs airports.csv: se vier do OurAirports, a coluna pode ser 'icao_code'. O app reconhece automaticamente.

Nota: Contornos (topologia) usam cs.allsegs para compatibilidade com versões novas do Matplotlib.

Novidade: opção global e por aba para excluir o demandante CAOP (evita distorção nos rankings).

Aba Demandantes: inclui gráfico de pizza (donut) com percentuais do Top N + OUTROS.
