import pandas as pd
import torch

def calculate_class_distribution():
    # Read the CSV file
    df = pd.read_csv('nlst_actual.csv')
    
    # Count occurrences for first year (canc_yr1) and fifth year (canc_yr5)
    first_year_counts = df['canc_yr1'].value_counts()
    fifth_year_counts = df['canc_yr5'].value_counts()
    
    total_samples = len(df)
    
    print("Calculating class distribution...")
    print(f"Total samples in dataset: {total_samples}")
    
    # Calculate percentages
    first_year_percentages = (first_year_counts / total_samples) * 100
    fifth_year_percentages = (fifth_year_counts / total_samples) * 100
    
    # Calculate recommended pos_weights
    first_year_neg_to_pos = first_year_counts[0] / first_year_counts[1] if 1 in first_year_counts else float('inf')
    fifth_year_neg_to_pos = fifth_year_counts[0] / fifth_year_counts[1] if 1 in fifth_year_counts else float('inf')
    
    # Print results
    print("\nFirst Year Cancer Prediction Distribution:")
    print(f"Negative cases (0): {first_year_counts[0]} ({first_year_percentages[0]:.2f}%)")
    print(f"Positive cases (1): {first_year_counts[1]} ({first_year_percentages[1]:.2f}%)")
    print(f"Recommended pos_weight for first year: {first_year_neg_to_pos:.2f}")
    
    print("\nFifth Year Cancer Prediction Distribution:")
    print(f"Negative cases (0): {fifth_year_counts[0]} ({fifth_year_percentages[0]:.2f}%)")
    print(f"Positive cases (1): {fifth_year_counts[1]} ({fifth_year_percentages[1]:.2f}%)")
    print(f"Recommended pos_weight for fifth year: {fifth_year_neg_to_pos:.2f}")
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'First Year': {
            'Negative Cases': first_year_counts[0],
            'Positive Cases': first_year_counts[1],
            'Negative %': first_year_percentages[0],
            'Positive %': first_year_percentages[1],
            'Recommended pos_weight': first_year_neg_to_pos
        },
        'Fifth Year': {
            'Negative Cases': fifth_year_counts[0],
            'Positive Cases': fifth_year_counts[1],
            'Negative %': fifth_year_percentages[0],
            'Positive %': fifth_year_percentages[1],
            'Recommended pos_weight': fifth_year_neg_to_pos
        }
    })
    
    # Save results to CSV
    summary.to_csv('class_distribution_summary.csv')
    print("\nResults have been saved to 'class_distribution_summary.csv'")
    
    # Return the recommended pos_weights
    return torch.tensor([first_year_neg_to_pos, fifth_year_neg_to_pos])

if __name__ == "__main__":
    recommended_weights = calculate_class_distribution()
    print("\nRecommended pos_weight tensor:", recommended_weights) 