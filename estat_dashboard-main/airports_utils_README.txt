Use assim:

from airports_utils import load_airports, airport_visit_counts, classify_icao_points, save_ignored_report

airports = load_airports(data_dir="data")
vis = airport_visit_counts(df_filtrado)
mapped, ignored = classify_icao_points(vis, airports)
save_ignored_report(ignored, "data/ignored_icao_report.csv")

Arquivos esperados:
- data/airports_master.csv  (preferencial)
ou
- data/airports.csv

Opcional:
- data/icao_overrides.csv

Formato sugerido para overrides:
icao,latitude_deg,longitude_deg,iso_country,name,municipality,is_caop,notes
SNSG,-23.123,-46.456,BR,Exemplo,Exemplo,false,Coordenada manual
