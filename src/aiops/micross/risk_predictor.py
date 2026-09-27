
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

# Project root:
# HARDTEC/
# ├── src/
# ├── models/
# └── data/

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "micross"
    / "micross_multisource_risk_xgboost_v3.pkl"
)

FEATURES_PATH = (
    PROJECT_ROOT
    / "models"
    / "micross"
    / "micross_multisource_risk_features_v3.json"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "micross"
    / "micross_multisource_features_5m.csv"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_VERSION = "V3.1"

TARGET = "incident_next_15m"

# Threshold selected during validation
ALERT_THRESHOLD = 0.25

# The model was trained on a 5-minute temporal grid
GRID_MINUTES = 5

# Expected V3.1 feature count
EXPECTED_FEATURES = 1380

# Expected number of base MicroSS metrics
EXPECTED_BASE_METRICS = 115


# ============================================================
# MICROSS RISK PREDICTOR
# ============================================================

class MicroSSRiskPredictor:
    """
    MicroSS AIOps Incident Risk Predictor.

    The predictor:

        1. Loads the trained XGBoost model.
        2. Loads the exact feature metadata.
        3. Loads the MicroSS base metrics.
        4. Reconstructs the 1,380 temporal features.
        5. Preserves the exact feature order used during training.
        6. Predicts the risk of an incident during the next 15 minutes.

    V3.1 feature construction:

        115 current metrics

        + 115 lag 5m
        + 115 lag 10m
        + 115 lag 15m

        + 115 delta 5m
        + 115 delta 10m
        + 115 delta 15m

        + 115 pct change 5m
        + 115 pct change 10m
        + 115 pct change 15m

        + 115 rolling mean 15m
        + 115 rolling std 15m

        = 1,380 features

    Memory optimization:

        - float32 instead of float64
        - NumPy operations for temporal features
        - preallocated final feature matrix
        - avoids keeping all intermediate DataFrames in memory
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        features_path: Path = FEATURES_PATH,
        data_path: Path = DATA_PATH,
    ):
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.data_path = Path(data_path)

        self.model = None
        self.metadata = None
        self.base_metrics = None
        self.feature_names = None
        self.feature_matrix = None
        self.timestamps = None
        self.data = None

        self._load_model()
        self._load_metadata()
        self._load_data()
        self._build_feature_matrix()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self):
        """Load the trained XGBoost model."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found:\n{self.model_path}"
            )

        self.model = joblib.load(self.model_path)

    # ========================================================
    # LOAD FEATURE METADATA
    # ========================================================

    def _load_metadata(self):
        """Load the JSON describing the training features."""

        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Feature metadata file not found:\n"
                f"{self.features_path}"
            )

        with open(self.features_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.base_metrics = self.metadata.get(
            "base_metrics",
            []
        )

        self.feature_names = self.metadata.get(
            "features",
            []
        )

        if not self.base_metrics:
            raise ValueError(
                "No base_metrics found in feature metadata."
            )

        if not self.feature_names:
            raise ValueError(
                "No feature names found in feature metadata."
            )

        expected_features = self.metadata.get(
            "n_features"
        )

        if expected_features is not None:
            if len(self.feature_names) != expected_features:
                raise ValueError(
                    "Feature metadata inconsistency: "
                    f"n_features={expected_features}, "
                    f"but {len(self.feature_names)} "
                    "features were found."
                )

        # V3.1 validation
        if len(self.base_metrics) != EXPECTED_BASE_METRICS:
            raise ValueError(
                "Unexpected number of base metrics. "
                f"Expected {EXPECTED_BASE_METRICS}, "
                f"found {len(self.base_metrics)}."
            )

        if len(self.feature_names) != EXPECTED_FEATURES:
            raise ValueError(
                "Unexpected number of model features. "
                f"Expected {EXPECTED_FEATURES}, "
                f"found {len(self.feature_names)}."
            )

    # ========================================================
    # LOAD MICROSS DATA
    # ========================================================

    def _load_data(self):
        """Load the preprocessed 5-minute MicroSS dataset."""

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"MicroSS feature dataset not found:\n"
                f"{self.data_path}"
            )

        # Read timestamp + required columns.
        df = pd.read_csv(
            self.data_path,
            low_memory=False,
        )

        if "timestamp" not in df.columns:
            raise ValueError(
                "The MicroSS feature dataset must contain "
                "a 'timestamp' column."
            )

        # Parse timestamp.
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        # Remove invalid timestamps.
        df = df.dropna(
            subset=["timestamp"]
        )

        # Sort chronologically.
        df = df.sort_values(
            "timestamp"
        )

        # Remove duplicate timestamps.
        df = df.drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )

        # Use timestamp as index.
        df = df.set_index(
            "timestamp"
        )

        # Check all base metrics.
        missing_metrics = [
            metric
            for metric in self.base_metrics
            if metric not in df.columns
        ]

        if missing_metrics:
            preview = missing_metrics[:10]

            raise ValueError(
                f"{len(missing_metrics)} base metrics are missing "
                "from the MicroSS dataset.\n"
                f"Examples: {preview}"
            )

        # Keep only metrics used by the model.
        self.data = df[
            self.base_metrics
        ].copy()

        # Convert metrics to numeric.
        self.data = self.data.apply(
            pd.to_numeric,
            errors="coerce",
        )

        # IMPORTANT:
        # Keep base data in float32 to reduce memory usage.
        self.data = self.data.astype(
            np.float32
        )

        # Store timestamps.
        self.timestamps = self.data.index

    # ========================================================
    # FEATURE NAME GENERATION
    # ========================================================

    def _generate_expected_feature_names(self):
        """
        Generate the canonical V3.1 feature names.

        The order here matches the original training logic:

            current
            lag
            delta
            pct_change
            rolling_mean
            rolling_std
        """

        names = []

        # ----------------------------------------------------
        # 1. CURRENT VALUES
        # ----------------------------------------------------

        names.extend(
            self.base_metrics
        )

        # ----------------------------------------------------
        # 2. LAGS
        # ----------------------------------------------------

        for minutes in (5, 10, 15):
            names.extend(
                [
                    f"{column}__lag_{minutes}m"
                    for column in self.base_metrics
                ]
            )

        # ----------------------------------------------------
        # 3. ABSOLUTE DELTAS
        # ----------------------------------------------------

        for minutes in (5, 10, 15):
            names.extend(
                [
                    f"{column}__delta_{minutes}m"
                    for column in self.base_metrics
                ]
            )

        # ----------------------------------------------------
        # 4. PERCENTAGE CHANGES
        # ----------------------------------------------------

        for minutes in (5, 10, 15):
            names.extend(
                [
                    f"{column}__pct_change_{minutes}m"
                    for column in self.base_metrics
                ]
            )

        # ----------------------------------------------------
        # 5. ROLLING MEAN
        # ----------------------------------------------------

        names.extend(
            [
                f"{column}__rolling_mean_15m"
                for column in self.base_metrics
            ]
        )

        # ----------------------------------------------------
        # 6. ROLLING STD
        # ----------------------------------------------------

        names.extend(
            [
                f"{column}__rolling_std_15m"
                for column in self.base_metrics
            ]
        )

        return names

    # ========================================================
    # BUILD 1,380 FEATURES
    # ========================================================

    def _build_feature_matrix(self):
        """
        Reconstruct the exact V3.1 temporal feature matrix.

        This implementation is memory optimized.

        Instead of creating:

            many Pandas DataFrames
                ↓
            pd.concat()
                ↓
            astype()
                ↓
            reorder

        we directly allocate one float32 NumPy matrix and fill it
        block by block.

        This significantly reduces peak RAM consumption.
        """

        print(
            "Building MicroSS V3.1 feature matrix..."
        )

        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        base = self.data

        n_rows = len(base)
        n_base = len(self.base_metrics)

        if n_base != EXPECTED_BASE_METRICS:
            raise ValueError(
                f"Expected {EXPECTED_BASE_METRICS} base metrics, "
                f"found {n_base}."
            )

        # We expect:
        #
        # 12 blocks × 115 metrics
        # = 1,380 features
        #
        expected_columns = (
            n_base * 12
        )

        if expected_columns != EXPECTED_FEATURES:
            raise ValueError(
                f"Internal feature configuration error. "
                f"Expected {EXPECTED_FEATURES}, "
                f"calculated {expected_columns}."
            )

        # ----------------------------------------------------
        # BASE VALUES
        # ----------------------------------------------------

        # Convert once to NumPy float32.
        base_values = base.to_numpy(
            dtype=np.float32,
            copy=True,
        )

        # ----------------------------------------------------
        # PREALLOCATE FINAL MATRIX
        # ----------------------------------------------------

        # For 17,640 rows × 1,380 float32:
        #
        # approximately 97 MB.
        #
        # This avoids having many complete DataFrames
        # simultaneously in memory.

        feature_array = np.empty(
            (
                n_rows,
                EXPECTED_FEATURES,
            ),
            dtype=np.float32,
        )

        column_position = 0

        # ----------------------------------------------------
        # 1. CURRENT VALUES
        # ----------------------------------------------------

        feature_array[
            :,
            column_position:column_position + n_base,
        ] = base_values

        column_position += n_base

        # ----------------------------------------------------
        # LAG CONFIGURATION
        # ----------------------------------------------------

        lag_steps = {
            5: 1,
            10: 2,
            15: 3,
        }

        # ====================================================
        # 2. LAGS
        # ====================================================

        for minutes, steps in lag_steps.items():

            lag_values = np.full_like(
                base_values,
                np.nan,
                dtype=np.float32,
            )

            if steps < n_rows:
                lag_values[steps:] = (
                    base_values[:-steps]
                )

            feature_array[
                :,
                column_position:column_position + n_base,
            ] = lag_values

            column_position += n_base

            # Release temporary array.
            del lag_values

        # ====================================================
        # 3. ABSOLUTE DELTAS
        # ====================================================

        for minutes, steps in lag_steps.items():

            delta_values = np.full_like(
                base_values,
                np.nan,
                dtype=np.float32,
            )

            if steps < n_rows:
                delta_values[steps:] = (
                    base_values[steps:]
                    - base_values[:-steps]
                )

            feature_array[
                :,
                column_position:column_position + n_base,
            ] = delta_values

            column_position += n_base

            del delta_values

        # ====================================================
        # 4. PERCENTAGE CHANGES
        # ====================================================

        for minutes, steps in lag_steps.items():

            pct_values = np.full_like(
                base_values,
                np.nan,
                dtype=np.float32,
            )

            if steps < n_rows:

                current_values = (
                    base_values[steps:]
                )

                lag_values = (
                    base_values[:-steps]
                )

                # Work on a temporary copy so the original
                # MicroSS values are never modified.
                safe_lag = lag_values.copy()

                # Avoid division by zero.
                safe_lag[
                    safe_lag == 0
                ] = np.nan

                with np.errstate(
                    divide="ignore",
                    invalid="ignore",
                    over="ignore",
                ):
                    pct_block = (
                        current_values
                        / safe_lag
                    ) - np.float32(1.0)

                # Replace +/- infinity with NaN.
                pct_block[
                    ~np.isfinite(pct_block)
                ] = np.nan

                pct_values[
                    steps:
                ] = pct_block

                del current_values
                del lag_values
                del safe_lag
                del pct_block

            feature_array[
                :,
                column_position:column_position + n_base,
            ] = pct_values

            column_position += n_base

            del pct_values

        # ====================================================
        # 5. ROLLING MEAN 15 MIN
        # ====================================================

        # Pandas rolling is retained here because it preserves
        # the original training semantics:
        #
        # window=3
        # min_periods=1

        rolling_mean = (
            base
            .rolling(
                window=3,
                min_periods=1,
            )
            .mean()
            .to_numpy(
                dtype=np.float32
            )
        )

        feature_array[
            :,
            column_position:column_position + n_base,
        ] = rolling_mean

        column_position += n_base

        del rolling_mean

        # ====================================================
        # 6. ROLLING STD 15 MIN
        # ====================================================

        # Pandas std uses ddof=1 by default, which matches
        # the original implementation.

        rolling_std = (
            base
            .rolling(
                window=3,
                min_periods=1,
            )
            .std()
            .to_numpy(
                dtype=np.float32
            )
        )

        feature_array[
            :,
            column_position:column_position + n_base,
        ] = rolling_std

        column_position += n_base

        del rolling_std

        # ====================================================
        # INTERNAL VALIDATION
        # ====================================================

        if column_position != EXPECTED_FEATURES:
            raise ValueError(
                "Feature construction error. "
                f"Expected {EXPECTED_FEATURES} columns, "
                f"but constructed {column_position}."
            )

        # ----------------------------------------------------
        # GENERATE CANONICAL NAMES
        # ----------------------------------------------------

        generated_names = (
            self._generate_expected_feature_names()
        )

        if len(generated_names) != EXPECTED_FEATURES:
            raise ValueError(
                "Generated feature name count mismatch. "
                f"Expected {EXPECTED_FEATURES}, "
                f"found {len(generated_names)}."
            )

        # ----------------------------------------------------
        # CHECK AGAINST METADATA
        # ----------------------------------------------------

        generated_name_set = set(
            generated_names
        )

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in generated_name_set
        ]

        if missing_features:

            preview = missing_features[:20]

            raise ValueError(
                f"{len(missing_features)} model features "
                "could not be reconstructed.\n"
                f"Examples:\n{preview}"
            )

        # Check that no unexpected metadata features exist.
        unexpected_features = [
            feature
            for feature in generated_names
            if feature not in set(self.feature_names)
        ]

        if unexpected_features:

            preview = unexpected_features[:20]

            raise ValueError(
                f"{len(unexpected_features)} generated features "
                "are not present in the model metadata.\n"
                f"Examples:\n{preview}"
            )

        # ----------------------------------------------------
        # MAP METADATA ORDER
        # ----------------------------------------------------

        generated_position = {
            name: index
            for index, name in enumerate(
                generated_names
            )
        }

        reorder_indices = np.array(
            [
                generated_position[name]
                for name in self.feature_names
            ],
            dtype=np.int32,
        )

        # If the metadata order already matches the generated
        # order, avoid creating another complete copy.
        identity_order = np.array_equal(
            reorder_indices,
            np.arange(
                EXPECTED_FEATURES,
                dtype=np.int32,
            ),
        )

        if not identity_order:

            feature_array = (
                feature_array[
                    :,
                    reorder_indices,
                ]
                .copy()
            )

        # ----------------------------------------------------
        # FINAL DATAFRAME
        # ----------------------------------------------------

        self.feature_matrix = pd.DataFrame(
            feature_array,
            index=self.data.index,
            columns=self.feature_names,
        )

        # feature_array is now owned by the DataFrame.
        del feature_array

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        actual_features = (
            self.feature_matrix.shape[1]
        )

        if actual_features != EXPECTED_FEATURES:
            raise ValueError(
                "Feature count mismatch.\n"
                f"Expected: {EXPECTED_FEATURES}\n"
                f"Actual: {actual_features}"
            )

        if actual_features != len(
            self.feature_names
        ):
            raise ValueError(
                "Feature metadata/order mismatch.\n"
                f"Metadata: {len(self.feature_names)}\n"
                f"Matrix: {actual_features}"
            )

        # Ensure final dtype is float32.
        if self.feature_matrix.dtypes.unique().tolist() != [
            np.dtype("float32")
        ]:
            self.feature_matrix = (
                self.feature_matrix.astype(
                    np.float32
                )
            )

        print(
            "MicroSS feature matrix ready:"
        )

        print(
            f"  Rows     : {n_rows}"
        )

        print(
            f"  Features : {actual_features}"
        )

        print(
            f"  Dtype    : {self.feature_matrix.dtypes.iloc[0]}"
        )

        print(
            f"  Memory   : "
            f"{self.feature_matrix.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB"
        )

    # ========================================================
    # AVAILABLE TIMESTAMPS
    # ========================================================

    def get_available_timestamps(
        self,
        remove_initial_history: bool = True,
    ):
        """
        Return timestamps available for prediction.

        The first 15 minutes do not contain complete lag history,
        therefore they can optionally be removed.
        """

        timestamps = self.feature_matrix.index

        if remove_initial_history:

            # 15 minutes / 5 minutes = 3 points
            if len(timestamps) > 3:
                timestamps = timestamps[3:]

        return list(timestamps)

    # ========================================================
    # GET DATA ROW
    # ========================================================

    def get_feature_row(self, timestamp):
        """
        Return the 1,380 features for a specific timestamp.
        """

        timestamp = pd.Timestamp(
            timestamp
        )

        if timestamp not in self.feature_matrix.index:
            raise ValueError(
                "Timestamp not found in dataset: "
                f"{timestamp}"
            )

        row = self.feature_matrix.loc[
            [timestamp]
        ]

        return row

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, timestamp):
        """
        Predict incident risk for the next 15 minutes.

        Returns:

            - timestamp
            - risk_score
            - risk_percentage
            - threshold
            - threshold_percentage
            - alert
            - risk_level
            - target
            - model_version
        """

        row = self.get_feature_row(
            timestamp
        )

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        probability = self.model.predict_proba(
            row
        )[0, 1]

        probability = float(
            probability
        )

        # ----------------------------------------------------
        # ALERT DECISION
        # ----------------------------------------------------

        alert = (
            probability
            >= ALERT_THRESHOLD
        )

        # ----------------------------------------------------
        # UI RISK LEVEL
        # ----------------------------------------------------

        if probability >= ALERT_THRESHOLD:
            risk_level = "HIGH"

        elif probability >= 0.10:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        return {
            "timestamp": pd.Timestamp(
                timestamp
            ),
            "risk_score": probability,
            "risk_percentage": probability * 100,
            "threshold": ALERT_THRESHOLD,
            "threshold_percentage": (
                ALERT_THRESHOLD * 100
            ),
            "alert": bool(alert),
            "risk_level": risk_level,
            "target": TARGET,
            "model_version": MODEL_VERSION,
        }

    # ========================================================
    # PREDICT BY INDEX
    # ========================================================

    def predict_by_index(
        self,
        index: int,
    ):
        """
        Predict using a timestamp index.
        """

        timestamps = (
            self.get_available_timestamps()
        )

        if index < 0 or index >= len(
            timestamps
        ):
            raise IndexError(
                "Timestamp index out of range: "
                f"{index}"
            )

        return self.predict(
            timestamps[index]
        )

    # ========================================================
    # GET RECENT DATA
    # ========================================================

    def get_recent_metrics(
        self,
        timestamp,
        minutes: int = 15,
    ):
        """
        Return recent base metrics before a timestamp.

        Useful for displaying temporal context
        in Streamlit.
        """

        timestamp = pd.Timestamp(
            timestamp
        )

        start = (
            timestamp
            - pd.Timedelta(
                minutes=minutes
            )
        )

        return self.data.loc[
            start:timestamp
        ].copy()

    # ========================================================
    # GET BASE METRICS
    # ========================================================

    def get_base_metrics(self):
        """Return the 115 base metric names."""

        return list(
            self.base_metrics
        )

    # ========================================================
    # GET FEATURE NAMES
    # ========================================================

    def get_feature_names(self):
        """Return the exact 1,380 feature names."""

        return list(
            self.feature_names
        )

    # ========================================================
    # GET DATA INFO
    # ========================================================

    def get_data_info(self):
        """
        Return useful dataset/model information.
        """

        return {
            "model_version": MODEL_VERSION,
            "target": TARGET,
            "threshold": ALERT_THRESHOLD,
            "base_metrics": len(
                self.base_metrics
            ),
            "features": len(
                self.feature_names
            ),
            "rows": len(
                self.data
            ),
            "start": self.data.index.min(),
            "end": self.data.index.max(),
        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "HARDTEC - MICROSS AIOPS RISK PREDICTOR"
    )
    print("=" * 70)

    predictor = (
        MicroSSRiskPredictor()
    )

    info = (
        predictor.get_data_info()
    )

    print()
    print(
        "Model version :",
        info["model_version"],
    )
    print(
        "Target        :",
        info["target"],
    )
    print(
        "Threshold     :",
        info["threshold"],
    )
    print(
        "Base metrics  :",
        info["base_metrics"],
    )
    print(
        "Features      :",
        info["features"],
    )
    print(
        "Rows          :",
        info["rows"],
    )
    print(
        "Start         :",
        info["start"],
    )
    print(
        "End           :",
        info["end"],
    )

    print()
    print(
        "Checking feature matrix..."
    )

    print(
        "Feature matrix shape:",
        predictor.feature_matrix.shape,
    )

    print(
        "Feature matrix dtype:",
        predictor.feature_matrix.dtypes.iloc[0],
    )

    memory_mb = (
        predictor.feature_matrix.memory_usage(
            deep=True
        ).sum()
        / (1024 ** 2)
    )

    print(
        f"Feature matrix memory: {memory_mb:.2f} MB"
    )

    timestamps = (
        predictor.get_available_timestamps()
    )

    print(
        "Available timestamps:",
        len(timestamps),
    )

    # Test the first valid timestamp.
    if timestamps:

        test_timestamp = timestamps[0]

        result = predictor.predict(
            test_timestamp
        )

        print()
        print(
            "TEST PREDICTION"
        )
        print(
            "-" * 70
        )

        print(
            "Timestamp       :",
            result["timestamp"],
        )

        print(
            "Risk score      :",
            result["risk_score"],
        )

        print(
            "Risk percentage :",
            f"{result['risk_percentage']:.2f}%",
        )

        print(
            "Alert threshold :",
            f"{result['threshold_percentage']:.0f}%",
        )

        print(
            "Risk level      :",
            result["risk_level"],
        )

        print(
            "Alert           :",
            result["alert"],
        )

        print(
            "Target          :",
            result["target"],
        )

        print(
            "Model version   :",
            result["model_version"],
        )

    print()
    print("=" * 70)
    print(
        "PREDICTOR READY"
    )
    print("=" * 70)

