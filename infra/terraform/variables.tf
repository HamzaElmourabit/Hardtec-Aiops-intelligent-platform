variable "kubeconfig_path" {
  description = "Path to the kubeconfig used by Terraform."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Kubernetes context, for example minikube or kind-kind."
  type        = string
  default     = ""
}

variable "namespace" {
  description = "Namespace where the platform resources are created."
  type        = string
  default     = "hardtec"
}

variable "data_storage_size" {
  description = "Persistent volume size for application data."
  type        = string
  default     = "20Gi"
}

variable "mlflow_storage_size" {
  description = "Persistent volume size for MLflow."
  type        = string
  default     = "20Gi"
}
