import pandas as pd
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Increase volatility in a CSV data file')
parser.add_argument('input_csv', type=str, help='Path to the input CSV file')
parser.add_argument('output_csv', type=str, help='Path to save the modified CSV file')
args = parser.parse_args()

# Load the CSV file
data = pd.read_csv(args.input_csv)

# Function to increase volatility
# Ignoring drawdown recalculation
def increase_volatility(df, scale_factor=0.01, noise_level=0.002):
    modified_df = df.copy()

    # Adjust the spx column with the increased volatility
    spx_diff = df['spx'].diff().fillna(0) * scale_factor
    spx_noise = np.random.normal(0, noise_level * df['spx'].std(), len(df))
    modified_df['spx'] = df['spx'] + spx_diff + spx_noise

    # Recalculate log_price based on the adjusted spx
    modified_df['log_price'] = np.log(modified_df['spx'].clip(lower=1e-8))  # avoid log(0)

    # Recalculate log_return based on the adjusted log_price
    modified_df['log_return'] = modified_df['log_price'].diff().fillna(0)

    # Recalculate spx_normalised based on the adjusted log_price
    modified_df['spx_normalised'] = modified_df['spx'] / modified_df['spx'].iloc[0]

    return modified_df

# Increase volatility in the data
volatile_data = increase_volatility(data)

# Save the modified data to a new CSV file
volatile_data.to_csv(args.output_csv, index=False)

print(f"Modified data saved to '{args.output_csv}'")
