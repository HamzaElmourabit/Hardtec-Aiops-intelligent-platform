resource "kubernetes_namespace_v1" "hardtec" {
  metadata {
    name = var.namespace

    labels = {
      "app.kubernetes.io/part-of" = "hardtec-platform"
      "managed-by"                = "terraform"
    }
  }
}

resource "kubernetes_config_map_v1" "hardtec" {
  metadata {
    name      = "hardtec-config"
    namespace = kubernetes_namespace_v1.hardtec.metadata[0].name
  }

  data = {
    ENVIRONMENT             = "local"
    PYTHONPATH              = "/app"
    DATABASE_PATH           = "/app/data/lake/tickets_predictions.db"
    FASTAPI_URL             = "http://api:8000"
    MLFLOW_TRACKING_URI     = "http://mlflow:5000"
    STREAMLIT_SERVER_ADDRESS = "0.0.0.0"
    STREAMLIT_SERVER_PORT   = "8501"
  }
}

resource "kubernetes_persistent_volume_claim_v1" "hardtec_data" {
  metadata {
    name      = "hardtec-data"
    namespace = kubernetes_namespace_v1.hardtec.metadata[0].name
  }

  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = "standard"

    resources {
      requests = {
        storage = var.data_storage_size
      }
    }
  }
}

resource "kubernetes_persistent_volume_claim_v1" "mlflow_data" {
  metadata {
    name      = "mlflow-data"
    namespace = kubernetes_namespace_v1.hardtec.metadata[0].name
  }

  spec {
    access_modes       = ["ReadWriteOnce"]
    storage_class_name = "standard"

    resources {
      requests = {
        storage = var.mlflow_storage_size
      }
    }
  }
}
