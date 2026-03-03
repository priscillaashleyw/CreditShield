from config import CONFIG
from src.load_data import DataLoader

def main():
    dl = DataLoader(CONFIG)

    # pick the method your DataLoader actually exposes:
    # df = dl.load()
    # df = dl.load_raw()
    # df = dl.get_data()
    df = dl.load_data()   # <-- rename this to whatever exists in your class

    print(df.head(10))
    print("\nshape:", df.shape)
    print("\nmissing top 10:\n", df.isna().mean().sort_values(ascending=False).head(10))

if __name__ == "__main__":
    main()