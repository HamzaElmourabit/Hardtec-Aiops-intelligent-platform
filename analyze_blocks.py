import pandas as pd

# ============================================================
# LOAD DATA
# ============================================================

error = pd.read_csv(
    "./log/log semantics anomaly detection/error.csv"
)

info = pd.read_csv(
    "./log/log semantics anomaly detection/info.csv"
)

# Add ground-truth label
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

# Gap between consecutive logs
df["gap"] = df["datetime"].diff()

# ============================================================
# ACTIVE BLOCK ANALYSIS
# ============================================================

print()
print("=" * 80)
print("GAIA - ACTIVE BLOCK ANALYSIS")
print("=" * 80)

thresholds = {
    "5 minutes": pd.Timedelta(minutes=5),
    "15 minutes": pd.Timedelta(minutes=15),
    "30 minutes": pd.Timedelta(minutes=30),
    "1 hour": pd.Timedelta(hours=1),
    "6 hours": pd.Timedelta(hours=6),
}

results = []

for name, threshold in thresholds.items():

    # New block whenever the gap is larger than threshold
    df["block_id"] = (
        df["gap"] > threshold
    ).cumsum()

    block_sizes = df.groupby("block_id").size()

    number_of_blocks = block_sizes.nunique()
    total_blocks = len(block_sizes)

    result = {
        "threshold": name,
        "blocks": total_blocks,
        "mean_logs": block_sizes.mean(),
        "median_logs": block_sizes.median(),
        "max_logs": block_sizes.max(),
        "blocks_ge_10": (block_sizes >= 10).sum(),
        "blocks_ge_50": (block_sizes >= 50).sum(),
        "blocks_ge_100": (block_sizes >= 100).sum(),
    }

    results.append(result)

    print()
    print(f"THRESHOLD: {name}")
    print("-" * 60)
    print(f"Nombre de blocs      : {total_blocks}")
    print(f"Taille moyenne       : {block_sizes.mean():.1f} logs")
    print(f"Taille médiane       : {block_sizes.median():.1f} logs")
    print(f"Plus grand bloc      : {block_sizes.max()} logs")
    print(f"Blocs >= 10 logs     : {(block_sizes >= 10).sum()}")
    print(f"Blocs >= 50 logs     : {(block_sizes >= 50).sum()}")
    print(f"Blocs >= 100 logs    : {(block_sizes >= 100).sum()}")

# ============================================================
# SUMMARY TABLE
# ============================================================

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

summary = pd.DataFrame(results)

print(summary.to_string(index=False))

# ============================================================
# ADDITIONAL ANALYSIS
# ============================================================

# Use 30 minutes temporarily to inspect blocks
df["block_id"] = (
    df["gap"] > pd.Timedelta(minutes=30)
).cumsum()

blocks = (
    df.groupby("block_id")
    .agg(
        start=("datetime", "min"),
        end=("datetime", "max"),
        logs=("datetime", "size"),
        errors=("label", lambda x: (x == "error").sum()),
        infos=("label", lambda x: (x == "info").sum()),
    )
    .reset_index()
)

blocks["duration"] = blocks["end"] - blocks["start"]

print()
print("=" * 80)
print("TOP 20 ACTIVE BLOCKS - 30 MIN THRESHOLD")
print("=" * 80)

print(
    blocks
    .sort_values("logs", ascending=False)
    .head(20)
    .to_string(index=False)
)

# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    "gaia_active_block_summary.csv",
    index=False
)

blocks.to_csv(
    "gaia_active_blocks_30min.csv",
    index=False
)

print()
print("Fichiers créés :")
print("  - gaia_active_block_summary.csv")
print("  - gaia_active_blocks_30min.csv")