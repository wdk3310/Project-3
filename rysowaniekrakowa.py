import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Wczytaj dane NetCDF ===
ds = xr.open_dataset(r"C:\Data\gistemp250_GHCNv4.nc")

# === 2. Wybierz punkt siatki najbliższy do Krakowa ===
krakow = ds['tempanomaly'].sel(lat=50.0, lon=20.0, method='nearest')

# === 3. Konwertuj do DataFrame ===
df_krakow = krakow.to_dataframe().reset_index()

# === 4. Zapisz do pliku CSV ===
df_krakow.to_csv("krakow_tempanomaly.csv", index=False)

# === 5. Rysuj wykres zmian w czasie ===
plt.figure(figsize=(12, 5))
plt.plot(df_krakow['time'], df_krakow['tempanomaly'], label="Kraków (50°N, 20°E)", color='darkred')
plt.grid()
plt.xlabel("Czas")
plt.ylabel("Anomalia temperatury [°C]")
plt.title("Anomalia temperatury w Krakowie")
plt.legend()
plt.tight_layout()
plt.show()
