import pandas as pd

# Create a DataFrame with sample data
data = {
    'Amount': [-1000, 500, 700],
    'Date': ['2023-01-01', '2024-01-01', '2025-01-01']
}

df = pd.DataFrame(data)

df.to_csv('dummy_data.csv', index=False)

print("dummy_data.csv has been created.")
