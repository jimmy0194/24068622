# -*- coding: utf-8 -*-
"""Fundamentals of Data Science - MOHAN MARTHALA - Student ID- 24068622.ipynb

# Fundamentals of Data Science -  MOHAN MARTHALA - Student ID- 24068622

# **Step 1: Load & Prepare Data**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""**Load the Dataset == .csv**"""

df_24068622 = pd.read_csv('/content/Dataset24068622/sales2.csv')

"""**Loads the CSV file into a Pandas DataFrame named df_24068622**"""

df_24068622['Date'] = pd.to_datetime(df_24068622['Date'], errors='coerce')

"""**Converts the Date column to actual datetime objects.**"""

df_24068622 = df_24068622.dropna(subset=['Date'])

"""**Drop any rows where parsing failed**"""

df_24068622['Year'] = df_24068622['Date'].dt.year
df_24068622['Month'] = df_24068622['Date'].dt.month
df_24068622['DayOfYear'] = df_24068622['Date'].dt.dayofyear

"""**Extract time-based features**

**Calculate total items sold**
"""

df_24068622['TotalItemsSold'] = (
    df_24068622['NumberGroceryShop'] + df_24068622['NumberGroceryOnline'] +
    df_24068622['NumberNongroceryShop'] + df_24068622['NumberNongroceryOnline']
)

"""**Total revenue: RevenueGrocery + RevenueNongrocery**"""

df_24068622['TotalRevenue'] = df_24068622['RevenueGrocery'] + df_24068622['RevenueNongrocery']

"""**Average price per item**"""

df_24068622['AveragePrice'] = df_24068622['TotalRevenue'] / df_24068622['TotalItemsSold']

"""**Results (Preview)**"""

df_24068622[['Date', 'TotalItemsSold', 'TotalRevenue', 'AveragePrice']].head()

"""# **Step 2: Monthly Avg Daily Items Sold (Bar Chart)**

**We Needs "Months" to Calculate the Rest**

**Extract useful date parts**
"""

monthly_avg_24068622 = df_24068622.groupby('Month')['TotalItemsSold'].mean()

"""**Figure -Plotting**"""

fig1_24068622 = plt.figure(figsize=(14, 6))
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
plt.show()

"""**Monthly average daily items sold**

# **STEP 3: Fourier Series (8 Terms) for 2022**

***Fourier approximation using first 8 terms for year 2022***
"""

df_2022_24068622 = df_24068622[df_24068622['Year'] == 2022]
y_2022_24068622 = df_2022_24068622['TotalItemsSold'].values
x_2022_24068622 = np.arange(1, len(y_2022_24068622) + 1)

"""***First 8 terms of Fourier series***"""

N_24068622 = len(y_2022_24068622)
a0_24068622 = np.mean(y_2022_24068622)
fourier_series_24068622 = np.full(N_24068622, a0_24068622 / 2)
for n in range(1, 9):
    an = 2 / N_24068622 * np.sum(y_2022_24068622 * np.cos(2 * np.pi * n * x_2022_24068622 / N_24068622))
    bn = 2 / N_24068622 * np.sum(y_2022_24068622 * np.sin(2 * np.pi * n * x_2022_24068622 / N_24068622))
    fourier_series_24068622 += an * np.cos(2 * np.pi * n * x_2022_24068622 / N_24068622) + \
                               bn * np.sin(2 * np.pi * n * x_2022_24068622 / N_24068622)

"""***Overlay Fourier curve***"""

plt.plot(np.linspace(1, 12, N_24068622), fourier_series_24068622, color='red', label='Fourier Approx (8 terms)')
plt.legend()
plt.text(11, max(monthly_avg_24068622) + 100, 'Student ID: 24068622', fontsize=12)
plt.show()

"""# **STEP 4: Scatter Plot – Average Price vs Items Sold**"""

fig2_24068622 = plt.figure(figsize=(10, 6))
plt.scatter(df_24068622['TotalItemsSold'], df_24068622['AveragePrice'], alpha=0.6, label='Daily Data')

plt.title('Figure 1 - Avg Price vs Items Sold (ID: 24068622)', fontsize=14)
plt.xlabel('Items Sold')
plt.ylabel('Average Price')
plt.legend()

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""# **STEP 5: Linear Regression Line**"""

# Prepare data
X_24068622 = df_24068622['TotalItemsSold'].values.reshape(-1, 1)
y_24068622 = df_24068622['AveragePrice'].values

# Train model
model_24068622 = LinearRegression().fit(X_24068622, y_24068622)
y_pred_24068622 = model_24068622.predict(X_24068622)

# Plot
fig2_24068622 = plt.figure(figsize=(10, 6))
plt.scatter(df_24068622['TotalItemsSold'], df_24068622['AveragePrice'], alpha=0.6, label='Daily Data')
plt.plot(df_24068622['TotalItemsSold'], y_pred_24068622, color='red', label='Linear Regression')

plt.title('Figure 2 - Avg Price vs Items Sold (ID: 24068622)', fontsize=14)
plt.xlabel('Items Sold')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""# **STEP 6: Calculate X & Y (Revenue in 2021 and 2022)**"""

# STEP 6 (Updated): Plot with improved size and clearer annotation

fig2_24068622 = plt.figure(figsize=(12, 8))  # Wider and taller for better visibility
plt.scatter(df_24068622['TotalItemsSold'], df_24068622['AveragePrice'], alpha=0.6, label='Daily Data')
plt.plot(df_24068622['TotalItemsSold'], y_pred_24068622, color='red', label='Linear Regression')

plt.title('Figure 3 - Avg Price vs Items Sold (ID: 24068622)', fontsize=16)
plt.xlabel('Items Sold', fontsize=12)
plt.ylabel('Average Price', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# Improved annotation positions (top right corner)
x_text = df_24068622['TotalItemsSold'].max() * 0.88
y_text = df_24068622['AveragePrice'].max() * 0.97

plt.text(x_text, y_text,
         f'Student ID: 24068622\nX (Revenue 2021): {round(X_value_24068622, 2)}\nY (Revenue 2022): {round(Y_value_24068622, 2)}',
         fontsize=11, bbox=dict(facecolor='white', alpha=0.9), verticalalignment='top')

plt.tight_layout()
plt.show()