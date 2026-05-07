import pandas as pd


def load_csv(data_csv):
    df = pd.read_csv(data_csv)
    cols = {c.lower(): c for c in df.columns}
    # Normalize expected coordinate names.
    if "longitude" not in cols or "latitude" not in cols:
        raise ValueError("CSV must include longitude and latitude columns.")
    return df


def validate_element(df, element):
    if element not in df.columns:
        # case-insensitive fallback
        lower_map = {c.lower(): c for c in df.columns}
        if element.lower() in lower_map:
            return lower_map[element.lower()]
        raise ValueError(f"Element '{element}' not found in CSV columns.")
    return element
