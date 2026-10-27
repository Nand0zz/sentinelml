"""Central configuration: paths, column groups, defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_FILE = "UNSW_NB15_training-set.csv"
TEST_FILE = "UNSW_NB15_testing-set.csv"

# --- DVC pipeline artifacts ---
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_TRAIN = "train.parquet"
PROCESSED_TEST = "test.parquet"
FEATURE_SCHEMA = "feature_schema.json"
MODELS_DIR = ROOT / "models"            # models/{candidate,champion,previous_champion}/
MODEL_MANIFEST = "model_manifest.json"  # models/model_manifest.json, committed to git
MODEL_META = "export_meta.json"         # metadata file inside every model folder

TARGET = "label"  # 0 = benign, 1 = potentially malicious

# attack_cat is derived from the label (leakage); id is a row counter.
LEAKAGE_COLS = ["id", "attack_cat"]
# Raw TCP initial sequence numbers: random per connection, no behavioural signal,
# and a known source of dataset-specific artifacts that trees can memorise.
ARTIFACT_COLS = ["stcpb", "dtcpb"]

CATEGORICAL = ["proto", "service", "state"]

NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]

ENGINEERED = [
    "total_bytes", "total_pkts", "bytes_ratio", "pkts_ratio",
    "src_bytes_per_pkt", "dst_bytes_per_pkt",
]

FEATURES = CATEGORICAL + NUMERIC  # model input columns (raw, before engineering)

DEFAULT_MAX_FPR = 0.05  # business constraint: at most 5% of benign flows flagged
DEFAULT_SEED = 42
DEFAULT_MODEL_NAME = "sentinelml-ids"
DEFAULT_EXPERIMENT = "sentinelml-training"
RUN_NAME_FORMAT = "run-%Y%m%d-%H%M%S"  # MLflow parent run per training session

# --- Day 2: registry aliases, serving, artifacts ---
ALIAS_CANDIDATE = "candidate"          # newest model awaiting the quality gate
ALIAS_CHAMPION = "champion"            # model currently approved for serving
ALIAS_PREVIOUS = "previous_champion"   # kept for one-step rollback

DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPORT_DIR = ROOT / "exported_model"   # self-contained copy of the champion (used by Docker)
PREDICTION_LOG = ROOT / "logs" / "predictions.jsonl"  # read by Day 3 drift monitoring
TOP_K_EXPLANATIONS = 5
MAX_BATCH = 500
