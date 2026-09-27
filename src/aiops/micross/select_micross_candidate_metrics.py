from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INVENTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_metric_inventory.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_candidate_metrics.csv"
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


def main():

    print("=" * 70)
    print("MICROSS - CANDIDATE METRIC SELECTION")
    print("=" * 70)

    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(
            f"Inventaire introuvable : {INVENTORY_FILE}"
        )

    df = pd.read_csv(INVENTORY_FILE)

    print()
    print(f"Métriques dans l'inventaire : {len(df):,}")

    # ------------------------------------------------------------
    # Nettoyage
    # ------------------------------------------------------------

    df["filename_lower"] = (
        df["filename"]
        .fillna("")
        .str.lower()
    )

    df["service"] = df["service"].fillna("SYSTEM")

    # ------------------------------------------------------------
    # Période principale
    # ------------------------------------------------------------
    #
    # Notre dataset de prédiction actuel utilise :
    # 2021-08-01 -> 2021-08-31
    #
    # On privilégie donc les fichiers August.
    # ------------------------------------------------------------

    august = df[
        df["filename_lower"].str.contains(
            "2021-08-01_2021-08-31",
            regex=False
        )
    ].copy()

    print(
        f"Métriques August 2021 disponibles : {len(august):,}"
    )

    # ------------------------------------------------------------
    # Fonction pour sélectionner un pattern
    # ------------------------------------------------------------

    def select_pattern(data, pattern):

        return data[
            data["filename_lower"].str.contains(
                pattern,
                regex=False
            )
        ].copy()

    candidates = []

    # ============================================================
    # 1. NETWORK ERRORS
    # ============================================================

    network_errors = august[
        august["filename_lower"].str.contains(
            "docker_network_",
            regex=False
        )
        &
        august["filename_lower"].str.contains(
            "error",
            regex=False
        )
    ].copy()

    network_errors["selection_reason"] = "docker_network_errors"

    candidates.append(network_errors)

    # ============================================================
    # 2. NETWORK PACKETS
    # ============================================================

    network_packets = august[
        august["filename_lower"].str.contains(
            "docker_network_",
            regex=False
        )
        &
        august["filename_lower"].str.contains(
            "packets",
            regex=False
        )
    ].copy()

    network_packets["selection_reason"] = "docker_network_packets"

    candidates.append(network_packets)

    # ============================================================
    # 3. DISK I/O
    # ============================================================

    disk = august[
        august["filename_lower"].str.contains(
            "docker_diskio",
            regex=False
        )
    ].copy()

    # Priorité aux métriques globales/agrégées
    disk_priority = disk[
        disk["filename_lower"].str.contains(
            "summary",
            regex=False
        )
    ].copy()

    if disk_priority.empty:
        disk_priority = disk.copy()

    disk_priority["selection_reason"] = "docker_diskio"

    candidates.append(disk_priority)

    # ============================================================
    # 4. SYSTEM LOAD
    # ============================================================

    system_load = august[
        august["filename_lower"].str.contains(
            "system_load",
            regex=False
        )
    ].copy()

    system_load["selection_reason"] = "system_load"

    candidates.append(system_load)

    # ============================================================
    # 5. SYSTEM CPU
    # ============================================================

    system_cpu = august[
        august["filename_lower"].str.contains(
            "system_cpu",
            regex=False
        )
    ].copy()

    # On privilégie les métriques normalisées
    system_cpu_norm = system_cpu[
        system_cpu["filename_lower"].str.contains(
            "norm",
            regex=False
        )
    ].copy()

    if system_cpu_norm.empty:
        system_cpu_norm = system_cpu.copy()

    system_cpu_norm["selection_reason"] = "system_cpu"

    candidates.append(system_cpu_norm)

    # ============================================================
    # 6. SYSTEM MEMORY
    # ============================================================

    system_memory = august[
        august["filename_lower"].str.contains(
            "system_memory",
            regex=False
        )
    ].copy()

    # On évite les métriques extrêmement spécialisées.
    # On privilégie usage / used / free / available / utilization.

    system_memory_priority = system_memory[
        system_memory["filename_lower"].str.contains(
            "usage|used|free|available|util",
            regex=True
        )
    ].copy()

    if system_memory_priority.empty:
        system_memory_priority = system_memory.copy()

    system_memory_priority["selection_reason"] = "system_memory"

    candidates.append(system_memory_priority)

    # ============================================================
    # Fusion
    # ============================================================

    if not candidates:
        raise RuntimeError(
            "Aucune métrique candidate trouvée."
        )

    selected = pd.concat(
        candidates,
        ignore_index=True
    )

    # Suppression doublons
    selected = selected.drop_duplicates(
        subset=["path_in_archive"]
    ).copy()

    # ------------------------------------------------------------
    # Limiter les métriques système
    # ------------------------------------------------------------

    # On ne veut pas sélectionner des centaines de métriques
    # système sans justification.

    system_rows = selected[
        selected["service"] == "SYSTEM"
    ].copy()

    service_rows = selected[
        selected["service"] != "SYSTEM"
    ].copy()

    # ------------------------------------------------------------
    # Pour chaque service :
    # maximum 4 network + 4 disk
    # ------------------------------------------------------------

    service_selected = []

    for service in SERVICES:

        sub = service_rows[
            service_rows["service"] == service
        ].copy()

        network = sub[
            sub["selection_reason"].str.startswith(
                "docker_network"
            )
        ].copy()

        disk = sub[
            sub["selection_reason"] == "docker_diskio"
        ].copy()

        # Priorité aux erreurs réseau
        network_errors_sub = network[
            network["selection_reason"] ==
            "docker_network_errors"
        ].head(4)

        network_packets_sub = network[
            network["selection_reason"] ==
            "docker_network_packets"
        ].head(2)

        disk_sub = disk.head(4)

        service_selected.append(
            pd.concat(
                [
                    network_errors_sub,
                    network_packets_sub,
                    disk_sub,
                ]
            )
        )

    service_selected = pd.concat(
        service_selected,
        ignore_index=True
    )

    # ------------------------------------------------------------
    # SYSTEM : maximum 15
    # ------------------------------------------------------------

    system_load_selected = system_rows[
        system_rows["selection_reason"] == "system_load"
    ].head(5)

    system_cpu_selected = system_rows[
        system_rows["selection_reason"] == "system_cpu"
    ].head(5)

    system_memory_selected = system_rows[
        system_rows["selection_reason"] == "system_memory"
    ].head(5)

    system_selected = pd.concat(
        [
            system_load_selected,
            system_cpu_selected,
            system_memory_selected,
        ],
        ignore_index=True
    )

    # ------------------------------------------------------------
    # Résultat final
    # ------------------------------------------------------------

    final = pd.concat(
        [
            service_selected,
            system_selected,
        ],
        ignore_index=True
    )

    final = final.drop_duplicates(
        subset=["path_in_archive"]
    ).copy()

    # Supprimer colonne technique
    final = final.drop(
        columns=["filename_lower"],
        errors="ignore"
    )

    # ------------------------------------------------------------
    # Ajouter priorité
    # ------------------------------------------------------------

    def priority(reason):

        if reason == "docker_network_errors":
            return 1

        if reason == "docker_network_packets":
            return 2

        if reason == "docker_diskio":
            return 3

        if reason == "system_load":
            return 4

        if reason == "system_cpu":
            return 5

        if reason == "system_memory":
            return 6

        return 99

    final["priority"] = (
        final["selection_reason"]
        .map(priority)
    )

    final = final.sort_values(
        [
            "priority",
            "service",
            "filename",
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    # ------------------------------------------------------------
    # Rapport
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CANDIDATE METRICS")
    print("=" * 70)

    print()
    print(
        f"Métriques candidates sélectionnées : {len(final):,}"
    )

    print()
    print("Par type de signal :")
    print(
        final["selection_reason"]
        .value_counts()
        .to_string()
    )

    print()
    print("Par service :")
    print(
        final["service"]
        .value_counts()
        .to_string()
    )

    print()
    print("Liste des métriques :")
    print("-" * 70)

    for _, row in final.iterrows():

        print(
            f"[P{int(row['priority'])}] "
            f"{row['service']:15} | "
            f"{row['selection_reason']:25} | "
            f"{row['filename']}"
        )

    print()
    print("=" * 70)
    print("FICHIER CRÉÉ")
    print("=" * 70)

    print()
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()