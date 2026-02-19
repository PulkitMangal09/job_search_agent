# config/taxonomy.py
"""
Predefined skill taxonomy for DevOps and AI/ML job analysis.
Used for skill normalization and categorization.
Extend this dict to support new fields (e.g., Security, Data Engineering).
"""

# ─── DEVOPS TAXONOMY ──────────────────────────────────────────────────────────

DEVOPS_SKILLS = {
    # Containers & Orchestration
    "containers": ["docker", "podman", "containerd", "lxc"],
    "orchestration": ["kubernetes", "k8s", "helm", "openshift", "rancher", "nomad"],
    
    # CI/CD
    "ci_cd": [
        "jenkins", "github actions", "gitlab ci", "circleci", "travis ci",
        "argocd", "tekton", "spinnaker", "bamboo", "teamcity"
    ],
    
    # Infrastructure as Code
    "iac": ["terraform", "pulumi", "cloudformation", "ansible", "chef", "puppet", "saltstack"],
    
    # Cloud Platforms
    "cloud_aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "eks", "ecs", "rds"],
    "cloud_gcp": ["gcp", "google cloud", "gke", "bigquery", "cloud run", "pub/sub"],
    "cloud_azure": ["azure", "aks", "azure devops", "blob storage", "azure functions"],
    
    # Monitoring & Observability
    "monitoring": [
        "prometheus", "grafana", "datadog", "new relic", "splunk", "elk stack",
        "elasticsearch", "logstash", "kibana", "jaeger", "zipkin", "opentelemetry"
    ],
    
    # Networking
    "networking": ["nginx", "haproxy", "istio", "envoy", "consul", "vpc", "dns", "cdn"],
    
    # Security
    "security": ["vault", "siem", "devsecops", "sonarqube", "snyk", "trivy", "falco"],
    
    # Source Control
    "version_control": ["git", "github", "gitlab", "bitbucket", "svn"],
    
    # Languages for DevOps
    "scripting": ["bash", "shell", "python", "go", "ruby", "powershell", "yaml", "json"],
    
    # Databases / Messaging
    "databases": ["postgresql", "mysql", "mongodb", "redis", "kafka", "rabbitmq", "nats"],
}

# ─── AI/ML TAXONOMY ───────────────────────────────────────────────────────────

AIML_SKILLS = {
    # ML Frameworks
    "ml_frameworks": [
        "pytorch", "tensorflow", "keras", "jax", "mxnet", "paddle",
        "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost"
    ],
    
    # LLMs & GenAI
    "llm_genai": [
        "llm", "gpt", "chatgpt", "openai", "claude", "gemini", "llama",
        "langchain", "llamaindex", "rag", "fine-tuning", "prompt engineering",
        "transformer", "bert", "hugging face", "diffusion models", "stable diffusion"
    ],
    
    # MLOps
    "mlops": [
        "mlflow", "kubeflow", "airflow", "prefect", "dagster", "sagemaker",
        "vertex ai", "azure ml", "wandb", "dvc", "bentoml", "seldon", "triton"
    ],
    
    # Data Processing
    "data_processing": [
        "spark", "pyspark", "dask", "pandas", "numpy", "polars",
        "kafka", "flink", "databricks", "snowflake", "dbt"
    ],
    
    # Data Science / Statistics
    "data_science": [
        "statistics", "machine learning", "deep learning", "nlp",
        "computer vision", "reinforcement learning", "time series",
        "a/b testing", "feature engineering", "data analysis"
    ],
    
    # Visualization
    "visualization": ["matplotlib", "seaborn", "plotly", "tableau", "power bi", "superset"],
    
    # Programming Languages
    "languages": ["python", "r", "julia", "scala", "sql", "spark sql"],
    
    # Vector / Search
    "vector_db": ["faiss", "pinecone", "weaviate", "chroma", "milvus", "qdrant", "pgvector"],
    
    # Cloud AI Services
    "cloud_ai": [
        "aws sagemaker", "google vertex ai", "azure cognitive services",
        "bedrock", "rekognition", "comprehend", "textract"
    ],
    
    # Hardware / Infrastructure for ML
    "ml_infra": ["gpu", "cuda", "tensorrt", "onnx", "tpu", "distributed training"],
}

# ─── COMBINED FLAT MAPS ──────────────────────────────────────────────────────

def build_alias_map(taxonomy: dict) -> dict[str, str]:
    """
    Returns {alias: canonical_category} for quick lookup.
    Example: {"k8s": "orchestration", "kubernetes": "orchestration"}
    """
    alias_map = {}
    for category, skills in taxonomy.items():
        for skill in skills:
            alias_map[skill.lower()] = category
    return alias_map


DEVOPS_ALIAS_MAP = build_alias_map(DEVOPS_SKILLS)
AIML_ALIAS_MAP = build_alias_map(AIML_SKILLS)

# Skills that strongly indicate DevOps jobs
DEVOPS_INDICATOR_SKILLS = {
    "kubernetes", "docker", "terraform", "jenkins", "ansible", "helm",
    "ci/cd", "devops", "site reliability", "sre", "platform engineering"
}

# Skills that strongly indicate AI/ML jobs
AIML_INDICATOR_SKILLS = {
    "machine learning", "deep learning", "pytorch", "tensorflow", "mlops",
    "data scientist", "ml engineer", "llm", "nlp", "computer vision", "ai"
}

# Job title patterns for category detection
DEVOPS_TITLE_PATTERNS = [
    "devops", "sre", "site reliability", "platform engineer", "infrastructure",
    "cloud engineer", "devsecops", "release engineer", "build engineer"
]

AIML_TITLE_PATTERNS = [
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "research scientist", "nlp engineer", "computer vision", "mlops",
    "ai/ml", "deep learning", "applied scientist"
]

# Standard skill name normalization (alias -> canonical)
SKILL_NORMALIZATION = {
    "k8s": "kubernetes",
    "gh actions": "github actions",
    "tf": "terraform",
    "sklearn": "scikit-learn",
    "aws lambda": "lambda",
    "psql": "postgresql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "gpt-4": "gpt",
    "chatgpt": "gpt",
}
