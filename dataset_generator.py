import pandas as pd
import numpy as np


def generate_test_data(num_rows):
    start_date = pd.Timestamp('2000-01-01')
    end_date = pd.Timestamp('2050-01-01')  # Set an end date

    # Generate random dates within the range
    dates = pd.date_range(start=start_date, end=end_date, periods=num_rows)

    # Shuffle the dates to make them non-sequential
    dates = dates.to_series().sample(n=num_rows, replace=True).sort_values().reset_index(drop=True)

    amounts = np.random.uniform(-10000, 10000, num_rows)

    df = pd.DataFrame({
        'Date': dates,
        'Amount': amounts
    })

    return df


# Generate datasets
sizes = [10_000, 1_000_000, 10_000_000, 100_000_000]

for size in sizes:
    print(f"Generating dataset with {size} rows...")
    df = generate_test_data(size)
    filename = f'test_data_{size}_rows.csv'
    df.to_csv(filename, index=False)
    print(f"Dataset saved as {filename}")
    print(f"File size: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print()
