import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

ds = xr.open_dataset(r"C:\Data\gistemp250_GHCNv4.nc")

krakow = ds['tempanomaly'].sel(lat=50.0, lon=20.0, method='nearest')

df_krakow = krakow.to_dataframe().reset_index()

df_krakow.to_csv("krakow_tempanomaly.csv", index=False)

plt.figure(figsize=(12, 5))
plt.plot(df_krakow['time'], df_krakow['tempanomaly'], label="Kraków (50°N, 20°E)", color='darkred')
plt.grid()
plt.xlabel("Czas")
plt.ylabel("Anomalia temperatury [°C]")
plt.title("Anomalia temperatury w Krakowie")
plt.legend()
plt.tight_layout()
plt.show()
