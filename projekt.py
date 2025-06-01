import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from sklearn.metrics import r2_score
import os
os.system('clear')
 
plik = "dane.csv"
 
dane = pd.read_csv(plik)
 
x = dane['Year']
y = dane['Annual mean']
 
plt.plot(x, y, label='Data')
plt.xlabel('Year')
plt.ylabel('Temperature Anomaly (°C)')
plt.title('Annual global mean temperature')
plt.grid()
plt.legend()
plt.scatter(x,y,s=5,color='red')
plt.show()
 
best_r2 = -np.inf
best_poly = None
best_degree = 0
 
word = input("Czy chcesz pokazać animacje dopasowania prostej do punktów? Jeśli tak napisz Y: ")
if word == "Y":
    for degree in range(1, 21):
        polyfit = np.poly1d(np.polyfit(x, y, degree))
        y_predicted = polyfit(x)
        r2 = r2_score(y, y_predicted)
 
        if r2 > best_r2:
            best_r2 = r2
            best_poly = polyfit
            best_degree = degree
   
 
    # Animacja
    def update(frame):
        przestrzen = np.linspace(1880, 1880 + frame, 171)
        line.set_data(przestrzen, best_poly(przestrzen))
        return line,
 
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label=f'Polynomial Fit (degree {best_degree})', color='green')
    ax.plot(x, y, label='Data')
    ax.scatter(x,y,s=5,color='red')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Anomaly (°C)')
    ax.set_title('Annual global mean temperature')
    ax.grid()
    ax.legend()
    ax.set_xlim(1880, 2050)
    ax.set_ylim(min(y), max(y) + 3)
    ani = animation.FuncAnimation(fig, update, frames=range(171), interval=50, blit=True)
    plt.show()
 
    print(f"Najlepszy stopień wielomianu: {best_degree}, R^2: {best_r2}")
 
    przestrzen = np.linspace(1880, 2050, 171)
    plt.plot(x, y, label='Data')
    plt.plot(przestrzen, polyfit(przestrzen), label='Polynomial Fit', color='green')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (°C)')
    plt.title('Annual global mean temperature')
    plt.grid()
    plt.legend()
    plt.ylim(-0.60, 3)
    plt.scatter(x,y,s=5,color='red')
    plt.show()
 
    przestrzen = np.linspace(1880, 2050, 171)
 
    predicted_values = best_poly(przestrzen)
 
    predictions_df = pd.DataFrame({'Year': przestrzen, 'Predicted Temperature': predicted_values})
    predictions_df.to_csv('predictions.csv')
    print("Żegnaj")
else:
    print("Żegnaj")