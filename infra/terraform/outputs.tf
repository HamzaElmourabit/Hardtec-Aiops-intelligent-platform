output "namespace" {
  value = kubernetes_namespace_v1.hardtec.metadata[0].name
}

output "data_pvc" {
  value = kubernetes_persistent_volume_claim_v1.hardtec_data.metadata[0].name
}

output "mlflow_pvc" {
  value = kubernetes_persistent_volume_claim_v1.mlflow_data.metadata[0].name
}
