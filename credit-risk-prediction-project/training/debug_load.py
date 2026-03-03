from config import CONFIG
from src.load_data import DataLoader

def main():
    dl = DataLoader(CONFIG)

    df = dl.load_and_filter_data()
    df = dl.define_target(df, strategy=CONFIG["target_settings"]["default_strategy"])

    print("\n=== DF preview ===")
    print(df.head(10))
    print("\nshape:", df.shape)
    print("\ncolumns:", len(df.columns))
    print("\ndefault rate:", df["target"].mean())

if __name__ == "__main__":
    main()