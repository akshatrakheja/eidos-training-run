import pandas as pd
import csv

def extract_corrected_ways_to_test_assumptions(input_file, output_file):
    """
    Extracts 'Ways to Test Assumptions' data from the CSV and tags it under the correct category.
    """
    results = []
    current_category = None

    # Open the input CSV
    with open(input_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        
        for row in reader:
            # Check if the row is a category
            if row and row[0].startswith("Category"):
                current_category = row[0].split(":", 1)[1].strip()  # Update the current category
                continue

            # Skip empty rows
            if not row or not any(row):
                continue

            # Extract 'Ways to Test Assumptions'
            if len(row) > 2 and row[2]:  # Ensure the "Ways to Test Assumptions" column exists
                assumption = row[2].strip()
                if assumption.startswith("-"):  # Ensure it is part of "Ways to Test Assumptions"
                    results.append({
                        "Category": current_category,
                        "Ways to Test Assumptions": assumption
                    })

    # Write to output CSV
    with open(output_file, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["Category", "Ways to Test Assumptions"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Correctly tagged data saved to {output_file}")

# Example usage
input_csv = "../newdata/Copy of Eidos_ AGI question data  - Sheet1.csv"  # Replace with your file path
output_csv = "../newdata/output.csv"     # Replace with your desired output path
extract_corrected_ways_to_test_assumptions(input_csv, output_csv)