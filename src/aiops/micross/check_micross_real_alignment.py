
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FAULT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_fault_events.csv"
)

METRIC_DIR = PROJECT_ROOT / "MicroSS" / "metric_selected"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "micross"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "micross_real_metric_alignment.csv"

OFFSETS_HOURS = [-8, -2, -1, 0, 1, 2, 8]

TOLERANCE_MINUTES = 1


# ============================================================
# HELPERS
# ============================================================

def extract_service(filename):
    """
    Extract service name from metric filename.
    Example:
    dbservice1_0.0.0.4_docker_cpu_total_norm_pct_...
    -> dbservice1
    """
    match = re.match(
        r"(dbservice1|dbservice2|webservice1|webservice2|"
        r"redisservice1|redisservice2|mobservice1|mobservice2|"
        r"logservice1|logservice2)_",
        filename,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).lower()

    return None


def extract_metric_type(filename):
    filename_lower = filename.lower()

    if "cpu_total_norm_pct" in filename_lower:
        return "cpu"

    if "memory_usage_pct" in filename_lower:
        return "memory"

    return None


# ============================================================
# LOAD FAULTS
# ============================================================

print("=" * 90)
print("MICROSS REAL METRIC / FAULT ALIGNMENT")
print("=" * 90)

print(f"\nProject root:\n{PROJECT_ROOT}")
print(f"\nFault file:\n{FAULT_FILE}")
print(f"\nMetric directory:\n{METRIC_DIR}")


print("\n" + "=" * 90)
print("1. LOADING FAULT EVENTS")
print("=" * 90)

faults = pd.read_csv(FAULT_FILE)

faults["fault_start"] = pd.to_datetime(
    faults["fault_start"],
    errors="coerce"
)

faults["fault_end"] = pd.to_datetime(
    faults["fault_end"],
    errors="coerce"
)

# Remove suspicious durations
if "duration_suspicious" in faults.columns:
    faults = faults[
        faults["duration_suspicious"].fillna(False) == False
    ].copy()

faults = faults.dropna(
    subset=["fault_start", "fault_end", "service"]
).copy()

faults["service"] = faults["service"].astype(str).str.lower()

print(f"Valid faults: {len(faults):,}")

print("\nFaults by service:")
print(
    faults["service"]
    .value_counts()
    .sort_index()
)


# ============================================================
# LOAD METRICS
# ============================================================

print("\n" + "=" * 90)
print("2. LOADING METRIC FILES")
print("=" * 90)

metric_files = sorted(METRIC_DIR.glob("*.csv"))

print(f"\nMetric files found: {len(metric_files)}")

metrics = {}

for file in metric_files:

    service = extract_service(file.name)
    metric_type = extract_metric_type(file.name)

    if service is None or metric_type is None:
        print(f"Skipping: {file.name}")
        continue

    print(
        f"\nLoading {service:<15} | {metric_type:<6}"
    )

    df = pd.read_csv(file)

    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="ms",
        utc=True,
    )

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "value"]
    ).copy()

    # Convert to naive UTC for easier comparison
    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_convert(None)
    )

    df = df.sort_values("timestamp")

    metrics[(service, metric_type)] = df

    print(f"  rows  : {len(df):,}")
    print(f"  start : {df['timestamp'].min()}")
    print(f"  end   : {df['timestamp'].max()}")


# ============================================================
# POINT-LEVEL ALIGNMENT
# ============================================================

print("\n" + "=" * 90)
print("3. TESTING REAL POINT-LEVEL ALIGNMENT")
print("=" * 90)

results = []

for offset in OFFSETS_HOURS:

    print(
        f"\nTesting offset: {offset:+d} hour(s)"
    )

    offset_delta = pd.Timedelta(hours=offset)

    total_faults = 0

    cpu_covered = 0
    memory_covered = 0
    both_covered = 0

    cpu_points_total = 0
    memory_points_total = 0

    overlap_minutes_cpu = []
    overlap_minutes_memory = []

    for _, fault in faults.iterrows():

        total_faults += 1

        service = fault["service"]

        # ----------------------------------------------------
        # Apply candidate offset to RUN timestamps
        # ----------------------------------------------------

        fault_start = fault["fault_start"] + offset_delta
        fault_end = fault["fault_end"] + offset_delta

        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        cpu_df = metrics.get(
            (service, "cpu")
        )

        cpu_found = False
        cpu_count = 0

        if cpu_df is not None:

            timestamps = cpu_df["timestamp"]

            mask = (
                (timestamps >= fault_start)
                & (timestamps <= fault_end)
            )

            cpu_count = int(mask.sum())

            if cpu_count > 0:
                cpu_found = True
                cpu_covered += 1
                cpu_points_total += cpu_count

                overlap_minutes_cpu.append(
                    (
                        cpu_df.loc[mask, "timestamp"].max()
                        - cpu_df.loc[mask, "timestamp"].min()
                    ).total_seconds()
                    / 60
                )

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        memory_df = metrics.get(
            (service, "memory")
        )

        memory_found = False
        memory_count = 0

        if memory_df is not None:

            timestamps = memory_df["timestamp"]

            mask = (
                (timestamps >= fault_start)
                & (timestamps <= fault_end)
            )

            memory_count = int(mask.sum())

            if memory_count > 0:
                memory_found = True
                memory_covered += 1
                memory_points_total += memory_count

                overlap_minutes_memory.append(
                    (
                        memory_df.loc[mask, "timestamp"].max()
                        - memory_df.loc[mask, "timestamp"].min()
                    ).total_seconds()
                    / 60
                )

        if cpu_found and memory_found:
            both_covered += 1

    cpu_pct = (
        cpu_covered / total_faults * 100
        if total_faults
        else 0
    )

    memory_pct = (
        memory_covered / total_faults * 100
        if total_faults
        else 0
    )

    both_pct = (
        both_covered / total_faults * 100
        if total_faults
        else 0
    )

    result = {
        "offset_hours": offset,
        "total_faults": total_faults,

        "cpu_faults_with_real_points": cpu_covered,
        "cpu_coverage_pct": cpu_pct,

        "memory_faults_with_real_points": memory_covered,
        "memory_coverage_pct": memory_pct,

        "both_cpu_memory_faults": both_covered,
        "both_cpu_memory_pct": both_pct,

        "cpu_points_inside_faults": cpu_points_total,
        "memory_points_inside_faults": memory_points_total,

        "median_cpu_overlap_minutes": (
            np.median(overlap_minutes_cpu)
            if overlap_minutes_cpu
            else 0
        ),

        "median_memory_overlap_minutes": (
            np.median(overlap_minutes_memory)
            if overlap_minutes_memory
            else 0
        ),
    }

    results.append(result)

    print(
        f"  CPU    : {cpu_covered:,}/{total_faults:,} "
        f"({cpu_pct:.2f}%)"
    )

    print(
        f"  Memory : {memory_covered:,}/{total_faults:,} "
        f"({memory_pct:.2f}%)"
    )

    print(
        f"  Both   : {both_covered:,}/{total_faults:,} "
        f"({both_pct:.2f}%)"
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "both_cpu_memory_pct",
    ascending=False
)

print("\n" + "=" * 90)
print("4. ALIGNMENT RESULTS")
print("=" * 90)

print(
    results_df[
        [
            "offset_hours",
            "cpu_coverage_pct",
            "memory_coverage_pct",
            "both_cpu_memory_pct",
            "cpu_points_inside_faults",
            "memory_points_inside_faults",
        ]
    ].to_string(index=False)
)


# ============================================================
# BEST OFFSET
# ============================================================

best = results_df.iloc[0]

print("\n" + "=" * 90)
print("5. BEST CANDIDATE OFFSET")
print("=" * 90)

print(
    f"\nBest offset: {best['offset_hours']:+.0f} hour(s)"
)

print(
    f"CPU coverage    : "
    f"{best['cpu_coverage_pct']:.2f}%"
)

print(
    f"Memory coverage : "
    f"{best['memory_coverage_pct']:.2f}%"
)

print(
    f"Both coverage   : "
    f"{best['both_cpu_memory_pct']:.2f}%"
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 90)
print("6. OUTPUT")
print("=" * 90)

print(f"\nSaved to:\n{OUTPUT_FILE}")

print("\nDone.")

