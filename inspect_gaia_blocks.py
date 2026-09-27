import pandas as pd

# ============================================================
# LOAD
# ============================================================

error = pd.read_csv(
    "./log/log semantics anomaly detection/error.csv"
)

info = pd.read_csv(
    "./log/log semantics anomaly detection/info.csv"
)

error["label"] = "error"
info["label"] = "info"

df = pd.concat([error, info], ignore_index=True)

# ============================================================
# DATETIME
# ============================================================

df["datetime"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    utc=True
)

df = df.sort_values("datetime").reset_index(drop=True)

df["gap"] = df["datetime"].diff()

# ============================================================
# ACTIVE BLOCKS
# ============================================================

THRESHOLD = pd.Timedelta(minutes=30)

df["block_id"] = (
    df["gap"] > THRESHOLD
).cumsum()

blocks = (
    df.groupby("block_id")
    .agg(
        start=("datetime", "min"),
        end=("datetime", "max"),
        logs=("datetime", "size"),
        errors=("label", lambda x: (x == "error").sum()),
        infos=("label", lambda x: (x == "info").sum()),
        sources=("source", "nunique"),
        hosts=("host", "nunique"),
    )
    .reset_index()
)

blocks["duration"] = blocks["end"] - blocks["start"]

blocks["error_rate"] = (
    blocks["errors"] / blocks["logs"]
)

# ============================================================
# FILTER
# ============================================================

usable = blocks[
    blocks["logs"] >= 50
].copy()

usable = usable.sort_values(
    "logs",
    ascending=False
)

# ============================================================
# GLOBAL STATISTICS
# ============================================================

print()
print("=" * 90)
print("GAIA - USABLE ACTIVE BLOCKS")
print("=" * 90)

print(f"Total active blocks        : {len(blocks)}")
print(f"Blocks >= 50 logs          : {len(usable)}")
print(f"Blocks >= 100 logs         : {(blocks['logs'] >= 100).sum()}")
print(f"Blocks >= 200 logs         : {(blocks['logs'] >= 200).sum()}")
print(f"Blocks >= 500 logs         : {(blocks['logs'] >= 500).sum()}")
print(f"Blocks >= 1000 logs        : {(blocks['logs'] >= 1000).sum()}")

print()
print("DURATION")
print("-" * 90)
print(usable["duration"].describe())

print()
print("LOG COUNT")
print("-" * 90)
print(usable["logs"].describe())

print()
print("ERROR RATE")
print("-" * 90)
print(usable["error_rate"].describe())

# ============================================================
# TOP BLOCKS
# ============================================================

print()
print("=" * 90)
print("TOP 30 USABLE BLOCKS")
print("=" * 90)

print(
    usable[
        [
            "block_id",
            "start",
            "end",
            "duration",
            "logs",
            "errors",
            "infos",
            "error_rate",
            "sources",
            "hosts",
        ]
    ]
    .head(30)
    .to_string(index=False)
)

# ============================================================
# SAVE
# ============================================================

usable.to_csv(
    "gaia_usable_active_blocks.csv",
    index=False
)

print()
print("Created:")
print("  gaia_usable_active_blocks.csv")