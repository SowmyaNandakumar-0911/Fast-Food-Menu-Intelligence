import pandas as pd

def load_and_clean(path="data/raw/fastfood.csv"):
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop rows missing critical nutrition fields
    critical_cols = ["calories", "total_fat", "protein", "sodium", "total_carb"]
    df = df.dropna(subset=critical_cols)

    # Strip whitespace from string columns
    df["restaurant"] = df["restaurant"].str.strip()
    df["item"] = df["item"].str.strip()

    # Cap extreme outliers (above 99th percentile) in calories
    cap = df["calories"].quantile(0.99)
    df = df[df["calories"] <= cap]

    # Reset index
    df = df.reset_index(drop=True)
    df["item_id"] = df.index + 1

    return df

if __name__ == "__main__":
    df = load_and_clean()
    df.to_csv("data/processed/fastfood_clean.csv", index=False)
    print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df["restaurant"].value_counts())
    