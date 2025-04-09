# 24068622.py
# Fundamentals of Data Science - MOHAN MARTHALA - Student ID: 24068622

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df_24068622 = pd.read_csv('sales2.csv')
df_24068622['Date'] = pd.to_datetime(df_24068622['Date'], errors='coerce')
df_24068622 = df_24068622.dropna(subset=['Date'])

# Extract time-based features
df_24068622['Year'] = df_24068622['Date'].dt.year
df_24068622['Month'] = df_24068622['Date'].dt.month
df_24068622['DayOfYear'] = df_24068622['Date'].dt.dayofyear

# Calculate total items sold and revenue
df_24068622['TotalItemsSold'] = (
    df_24068622['NumberGroceryShop'] + df_24068622['NumberGroceryOnline'] +
    df_24068622['NumberNongroceryShop'] + df_24068622['NumberNongroceryOnline']
)
df_24068622['TotalRevenue'] = df_24068622['RevenueGrocery'] + df_24068622['RevenueNongrocery']
df_24068622['AveragePrice'] = df_24068622['TotalRevenue'] / df_24068622['TotalItemsSold']

# === Step 2: Monthly Avg Daily Items Sold (Bar Chart) ===
monthly_avg_24068622 = df_24068622.groupby('Month')['TotalItemsSold'].mean()

plt.figure(figsize=(14, 6))
plt.bar(monthly_avg_24068622.index, monthly_avg_24068622.values, color='cornflowerblue', label='Monthly Avg Items Sold')
plt.title('Figure 1 - Monthly Avg Items Sold (ID: 24068622)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Avg Items Sold per Day')
plt.xticks(ticks=np.arange(1, 13), labels=[
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
])
plt.legend()
plt.tight_layout()
plt.savefig('figure1.png')
plt.close()

# === Step 3: Fourier Series (8 Terms) for 2022 ===
df_2022 = df_24068622[df_24068622['Year'] == 2022]
y_2022 = df_2022['TotalItemsSold'].values
x_2022 = np.arange(1, len(y_2022) + 1)

N = len(y_2022)
a0 = np.mean(y_2022)
fourier_series = np.full(N, a0 / 2)

for n in range(1, 9):
    an = 2 / N * np.sum(y_2022 * np.cos(2 * np.pi * n * x_2022 / N))
    bn = 2 / N * np.sum(y_2022 * np.sin(2 * np.pi * n * x_2022 / N))
    fourier_series += an * np.cos(2 * np.pi * n * x_2022 / N) + bn * np.sin(2 * np.pi * n * x_2022 / N)

plt.figure(figsize=(14, 6))
plt.plot(np.linspace(1, 12, N), fourier_series, color='red', label='Fourier Approx (8 terms)')
plt.title('Figure 2 - Fourier Series Approximation (2022)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Total Items Sold')
plt.legend()
plt.text(11, max(monthly_avg_24068622) + 100, 'Student ID: 24068622', fontsize=12)
plt.tight_layout()
plt.savefig('figure2.png')
plt.close()

# === Step 4: Scatter Plot – Avg Price vs Items Sold ===
plt.figure(figsize=(10, 6))
plt.scatter(df_24068622['TotalItemsSold'], df_24068622['AveragePrice'], alpha=0.6, label='Daily Data')
plt.title('Figure 3 - Avg Price vs Items Sold (ID: 24068622)', fontsize=14)
plt.xlabel('Items Sold')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure3.png')
plt.close()

# === Step 5: Linear Regression Line ===
X = df_24068622['TotalItemsSold'].values.reshape(-1, 1)
y = df_24068622['AveragePrice'].values

model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(df_24068622['TotalItemsSold'], df_24068622['AveragePrice'], alpha=0.6, label='Daily Data')
plt.plot(df_24068622['TotalItemsSold'], y_pred, color='red', label='Linear Regression')
plt.title('Figure 4 - Regression: Avg Price vs Items Sold (ID: 24068622)', fontsize=14)
plt.xlabel('Items Sold')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('figure4.png')
plt.close()

# === Step 6: Revenue X & Y ===
X_value_24068622 = df_24068622[df_24068622['Year'] == 2021]['TotalRevenue'].sum()
Y_value_24068622 = df_24068622[df_24068622['Year'] == 2022]['TotalRevenue'].sum()

# Print results
print("Student ID: 24068622")
print(f"X (Revenue 2021): {round(X_value_24068622, 2)}")
print(f"Y (Revenue 2022): {round(Y_value_24068622, 2)}")
