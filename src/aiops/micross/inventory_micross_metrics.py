from pathlib import Path
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LISTING_FILE = (
    PROJECT_ROOT.parent
    / "GAIA-DataSet"
    / "data"
    / "processed"
    / "micross"
    / "metric_archive_listing.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_metric_inventory.csv"
)


SERVICES = [
    "dbservice1",
    "dbservice2",
    "webservice1",
    "webservice2",
    "redisservice1",
    "redisservice2",
    "mobservice1",
    "mobservice2",
    "logservice1",
    "logservice2",
]


def detect_service(filename):
    name = filename.lower()

    for service in SERVICES:
        if service.lower() in name:
            return service

    return None


def detect_metric_type(filename):
    name = filename.lower()

    if "docker_cpu_total_norm_pct" in name:
        return "docker_cpu_total_norm_pct"

    if "docker_memory_usage_pct" in name:
        return "docker_memory_usage_pct"

    if "docker_memory_usage_total" in name:
        return "docker_memory_usage_total"

    if "docker_memory_stats_active_anon" in name:
        return "docker_memory_stats_active_anon"

    if "docker_cpu" in name:
        return "docker_cpu"

    if "docker_memory" in name:
        return "docker_memory"

    if "system_cpu" in name:
        return "system_cpu"

    if "system_memory" in name:
        return "system_memory"

    if "system_load" in name:
        return "system_load"

    if "load" in name:
        return "load"

    if "network" in name or "net_" in name:
        return "network"

    if "disk" in name or "diskio" in name or "io_" in name:
        return "disk_io"

    if "filesystem" in name:
        return "filesystem"

    if "cpu" in name:
        return "cpu"

    if "memory" in name:
        return "memory"

    return "other"


def detect_scope(filename):
    name = filename.lower()

    if "docker_" in name:
        return "docker"

    if "system_" in name:
        return "system"

    return "service"


def detect_period(filename):
    match = re.search(
        r"(2021-\d{2}-\d{2})_(2021-\d{2}-\d{2})",
        filename
    )

    if match:
        return f"{match.group(1)}__{match.group(2)}"

    return "unknown"


def read_listing(path):

    raw = path.read_bytes()

    # UTF-16 avec BOM
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")

    # UTF-16 LE sans BOM
    if b"\x00" in raw[:500]:
        try:
            return raw.decode("utf-16-le")
        except UnicodeDecodeError:
            pass

    return raw.decode(
        "utf-8",
        errors="ignore"
    )


def main():

    print("=" * 70)
    print("MICROSS - METRIC INVENTORY")
    print("=" * 70)

    print()
    print("Listing :")
    print(LISTING_FILE)

    if not LISTING_FILE.exists():
        raise FileNotFoundError(
            f"Listing introuvable : {LISTING_FILE}"
        )

    text = read_listing(LISTING_FILE)

    lines = text.splitlines()

    print()
    print(f"Lignes du listing : {len(lines):,}")

    records = []

    for line in lines:

        line = line.strip()

        # Exemple réel :
        #
        # 2022-05-12 10:17:58 ..... 1195456 171433
        # metric\\dbservice1_0.0.0.4_docker_cpu_core_5_norm_pct_2021-08-01_2021-08-31.csv
        #
        # On cherche donc simplement la partie
        # metric\\...csv à la fin de la ligne.

        match = re.search(
            r"(metric[\\/].*?\.csv)\s*$",
            line,
            flags=re.IGNORECASE
        )

        if not match:
            continue

        archive_path = match.group(1)

        # Uniformisation des séparateurs
        archive_path = archive_path.replace("/", "\\")

        filename = Path(
            archive_path.replace("\\", "/")
        ).name

        if not filename.lower().endswith(".csv"):
            continue

        service = detect_service(filename)

        metric_type = detect_metric_type(filename)

        scope = detect_scope(filename)

        period = detect_period(filename)

        records.append({
            "path_in_archive": archive_path,
            "filename": filename,
            "service": service,
            "metric_type": metric_type,
            "scope": scope,
            "period": period,
        })

    df = pd.DataFrame(records)

    if df.empty:

        print()
        print("ERREUR : aucune métrique trouvée.")
        print()
        print("Quelques lignes contenant 'metric' :")
        print("-" * 70)

        count = 0

        for line in lines:

            if "metric" in line.lower():

                print(repr(line))

                count += 1

                if count >= 20:
                    break

        return

    # Suppression des doublons
    df = df.drop_duplicates(
        subset=["path_in_archive"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("INVENTORY RESULT")
    print("=" * 70)

    print()
    print(f"Métriques trouvées : {len(df):,}")

    print()
    print("Services détectés :")
    print(
        df["service"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print()
    print("Types de métriques :")
    print(
        df["metric_type"]
        .value_counts()
        .to_string()
    )

    print()
    print("Scopes :")
    print(
        df["scope"]
        .value_counts()
        .to_string()
    )

    print()
    print("Périodes :")
    print(
        df["period"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Fichier créé :")
    print(OUTPUT_FILE)

    print()
    print("Premières métriques :")
    print(
        df[
            [
                "service",
                "metric_type",
                "scope",
                "period",
                "filename",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()