import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
torch.manual_seed(15)

mpl.rcParams['animation.embed_limit'] = 50

# === 1. Wczytaj dane z pliku NetCDF ===
ds = xr.open_dataset(r"C:\Data\gistemp250_GHCNv4.nc")

# === 2. Wybierz punkty siatki ===
train_point = ds['tempanomaly'].sel(lat=50.0, lon=20.0, method="nearest")
test_point = ds['tempanomaly'].sel(lat=40.0, lon=20.0, method="nearest")

# === 3. Przygotuj dane czasowe ===
X = np.arange(len(ds['time'])).reshape(-1, 1).astype(np.float32)
y_train = train_point.values.astype(np.float32)
y_test = test_point.values.astype(np.float32)

# Usunięcie braków
mask = ~np.isnan(y_train) & ~np.isnan(y_test)
X_clean = X[mask]
y_train_clean = y_train[mask].reshape(-1, 1)
y_test_clean = y_test[mask].reshape(-1, 1)

# === 4. Skalowanie danych ===
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X_clean)
y_scaled = scaler_y.fit_transform(y_train_clean)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# === 5. Sieć neuronowa ===
class TempModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = TempModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# === 6. Trenowanie ===
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

# === 7. Przewidywanie i rozszerzenie do 2050 ===
model.eval()

# Przestrzeń czasowa: oryginalna + przyszłość
start_index = X_clean.flatten()[0]  # pierwszy miesiąc z danymi
n_future_months = (2050 - 1880) * 12
X_full = np.arange(start_index, n_future_months).reshape(-1, 1).astype(np.float32)
x_years = 1880 + X_full.flatten() / 12

X_full_scaled = scaler_x.transform(X_full[:len(X_clean)])  # Skaluj tylko do granicy treningu
X_future_scaled = scaler_x.transform(X_full)               # Cała przyszłość

X_full_tensor = torch.tensor(X_future_scaled, dtype=torch.float32)
with torch.no_grad():
    y_pred_scaled_full = model(X_full_tensor).numpy()
    y_pred_full = scaler_y.inverse_transform(y_pred_scaled_full).flatten()

# === 8. Wielomian ===
y_real = y_test_clean.flatten()
x_train_years = 1880 + X_clean.flatten() / 12

best_r2 = -np.inf
best_poly = None
best_degree = 0

for degree in range(1, 21):
    polyfit = np.poly1d(np.polyfit(x_train_years, y_real, degree))
    y_poly = polyfit(x_train_years)
    r2 = r2_score(y_real, y_poly)
    if r2 > best_r2:
        best_r2 = r2
        best_poly = polyfit
        best_degree = degree

# Prognoza wielomianowa do 2050
y_poly_full = best_poly(x_years)

# === 9. MSE i lokalizacje ===
y_pred_trunc = y_pred_full[:len(y_real)]
mse_nn = mean_squared_error(y_real, y_pred_trunc)

train_coords = (float(train_point.lat.values), float(train_point.lon.values))
test_coords = (float(test_point.lat.values), float(test_point.lon.values))

# === 10. Wykres ===
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(1880, 2050)
ax.set_ylim(min(y_real) - 0.5, max(max(y_pred_full), max(y_poly_full)) + 0.5)
ax.set_xlabel("Rok")
ax.set_ylabel("Anomalia [°C]")
ax.set_title("Prognoza do 2050: Sieć neuronowa vs wielomian")

# Tło: dane rzeczywiste
ax.plot(x_train_years, y_real, label="Dane rzeczywiste", color='black', alpha=0.5)

# Linie animowane
line_nn, = ax.plot([], [], color='blue', label="Sieć neuronowa")
line_poly, = ax.plot([], [], color='green', linestyle='--', label=f"Wielomian st. {best_degree}")
ax.legend()

# Adnotacja pod wykresem
fig.subplots_adjust(bottom=0.25)
fig.text(
    0.5, 0.02,
    f"Trening: [{train_coords[0]:.1f}°N, {train_coords[1]:.1f}°E] | "
    f"Test: [{test_coords[0]:.1f}°N, {test_coords[1]:.1f}°E] | "
    f"MSE sieci NN: {mse_nn:.4f} | R² wielomianu (st. {best_degree}): {best_r2:.4f}",
    ha='center', fontsize=10
)

# Funkcja animacji
def update(frame):
    x_anim = x_years[:frame]
    y_nn_anim = y_pred_full[:frame]
    y_poly_anim = y_poly_full[:frame]
    line_nn.set_data(x_anim, y_nn_anim)
    line_poly.set_data(x_anim, y_poly_anim)
    return line_nn, line_poly

ani = animation.FuncAnimation(fig, update, frames=len(x_years), interval=20, blit=True)

plt.show()
