import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('car.csv') # Source: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data/data

# Display the first five rows of the dataset to understand its structure.
print(data.head())

# Define columns to drop from the dataset
columns_to_drop = ['id', 'url', 'region_url', 'VIN', 'image_url', 'description', 'county', 'lat', 'long', 'posting_date', 'size', 'state']

# Drop the specified columns from the DataFrame
data.drop(columns=columns_to_drop, inplace=True)

# Remove rows with missing values in specific columns and fill others with 'unknown'
data = data.dropna(subset=['year', 'odometer', 'manufacturer', 'model'])
data.fillna('unknown', inplace=True)

# Generate a summary of the dataset: total number of rows, columns, data types, and basic statistics for numerical columns.
print(data.describe())

# Get the number of rows in the dataset
number_of_rows = data.shape[0]

# Display the number of rows
print("Number of rows in the CSV:", number_of_rows)

# Select numerical columns
numerical_columns = ['year', 'odometer', 'price']

# Convert selected columns to a NumPy array
numerical_data = data[numerical_columns].values

# Calculate statistics
mean_values = np.mean(numerical_data, axis=0)
median_values = np.median(numerical_data, axis=0)
std_deviation = np.std(numerical_data, axis=0)

# Print statistics
print("Mean Values for Year / Odometer / Price:")
print(mean_values)

print("\nMedian Values for Year / Odometer / Price:")
print(median_values)

print("\nStandard Deviation for Year / Odometer / Price:")
print(std_deviation)

# Create a scatter plot for the relationship between price and odometer
filtered_data_high_price = data[data['price'] < 100000]

# Group data in filtered_data_high_price by odometer and calculate the average prices for each group
average_price_by_odometer = filtered_data_high_price.groupby('odometer')['price'].mean()

# Create a line plot for the average prices based on odometer readings
plt.figure(figsize=(10, 6))
plt.plot(average_price_by_odometer.index, average_price_by_odometer.values)
plt.title('Average Prices vs. Odometer (Outliers Removed)')
plt.xlabel('Odometer')
plt.ylabel('Average Price')
plt.grid(True)

# Group data in filtered_data_high_price by car year and calculate the average prices for each group
average_price_by_year = filtered_data_high_price.groupby('year')['price'].mean()

# Create a line plot for the average prices based on car year
plt.figure(figsize=(10, 6))
plt.plot(average_price_by_year.index, average_price_by_year.values)
plt.title('Average Prices vs. Car Year (Outliers Removed)')
plt.xlabel('Car Year')
plt.ylabel('Average Price')
plt.grid(True)

# Create a bar plot for price vs. car paint colors, excluding 'green' and 'unknown'
filtered_data_colors = data[~data['paint_color'].isin(['green', 'unknown'])]
colors = filtered_data_colors['paint_color'].unique()
mean_prices_by_color = [data[data['paint_color'] == color]['price'].mean() for color in colors]

plt.figure(figsize=(12, 6))
plt.bar(colors, mean_prices_by_color)
plt.title('Average Prices by Car Paint Color')
plt.xlabel('Car Paint Color')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.show()
