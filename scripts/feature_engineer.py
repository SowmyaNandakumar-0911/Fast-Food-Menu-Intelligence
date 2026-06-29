import pandas as pd

def engineer_features(path="data/processed/fastfood_clean.csv"):
    df = pd.read_csv(path)

    # --- Core Efficiency Metrics ---
    df["protein_per_100cal"] = (df["protein"] / df["calories"] * 100).round(2)
    df["fat_per_100cal"] = (df["total_fat"] / df["calories"] * 100).round(2)
    df["carb_per_100cal"] = (df["total_carb"] / df["calories"] * 100).round(2)

    # --- Sodium Load Score (high sodium = bad) ---
    # FDA daily limit = 2300mg
    df["sodium_load_pct"] = (df["sodium"] / 2300 * 100).round(2)

    # --- Health Score (custom composite, 0-100) ---
    # Higher protein efficiency + lower sodium + lower sat fat = healthier
    df["health_score"] = (
        (df["protein_per_100cal"] / df["protein_per_100cal"].max() * 40) +   # 40% weight
        ((1 - df["sodium_load_pct"] / df["sodium_load_pct"].max()) * 30) +   # 30% weight
        ((1 - df["sat_fat"] / df["sat_fat"].max()) * 30)                     # 30% weight
    ).round(2)

    # --- "Healthy Trap" Flag ---
    # Items with "salad", "grilled", "light" in name but health_score < 40
    trap_keywords = ["salad", "grilled", "light", "garden", "veggie", "fresh"]
    pattern = "|".join(trap_keywords)
    df["healthy_trap"] = (
        df["item"].str.lower().str.contains(pattern) &
        (df["health_score"] < 40)
    ).astype(int)

    # --- Meal Category (rule-based) ---
    def categorize(item):
        item = item.lower()
        if any(k in item for k in ["burger", "sandwich", "wrap", "taco", "burrito"]):
            return "Main"
        elif any(k in item for k in ["fries", "nugget", "side", "onion"]):
            return "Side"
        elif any(k in item for k in ["salad", "bowl"]):
            return "Salad/Bowl"
        elif any(k in item for k in ["shake", "mcflurry", "sundae", "cone", "dessert", "pie"]):
            return "Dessert/Drink"
        elif any(k in item for k in ["coffee", "juice", "water", "soda", "tea"]):
            return "Beverage"
        else:
            return "Other"

    df["meal_category"] = df["item"].apply(categorize)

    # --- Calorie Band ---
    df["calorie_band"] = pd.cut(
        df["calories"],
        bins=[0, 300, 600, 900, 9999],
        labels=["Low (<300)", "Medium (300-600)", "High (600-900)", "Very High (900+)"]
    )

    return df

if __name__ == "__main__":
    df = engineer_features()
    df.to_csv("data/processed/fastfood_features.csv", index=False)
    print(f"Feature engineered: {df.shape[0]} rows")
    print(df[["item", "health_score", "healthy_trap", "meal_category"]].head(10))