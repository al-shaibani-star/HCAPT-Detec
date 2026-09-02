"""
HCAPT-Defense Framework — Configuration
Central config for all modules.
"""
import sys
# Force UTF-8 stdout so Unicode symbols (→, ∈, etc.) don't break on Windows cp1256
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from pathlib import Path
import torch

# ============================================================
#  Paths
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Per-dataset paths (scientific naming, post-reorg May 2026)
NSL_KDD_DIR        = DATA_DIR / "nsl-kdd"
UNSW_NB15_DIR      = DATA_DIR / "unsw-nb15"
CIC_IDS_2017_DIR   = DATA_DIR / "cic-ids-2017"
TON_IOT_DIR        = DATA_DIR / "ton-iot"
TON_IOT_NETWORK    = TON_IOT_DIR / "network" / "train_test_network.csv"
TON_IOT_TELEMETRY  = TON_IOT_DIR / "iot-telemetry"
EDGE_IIOTSET_DIR   = DATA_DIR / "edge-iiotset"
EDGE_IIOT_ATTACK   = EDGE_IIOTSET_DIR / "attack"
EDGE_IIOT_NORMAL   = EDGE_IIOTSET_DIR / "normal"
EDGE_IIOT_ML_DL    = EDGE_IIOTSET_DIR / "ml-dl-selected"

for d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
#  NSL-KDD Dataset
# ============================================================
# Official column names (NSL-KDD has 41 features + label + difficulty)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Attack category mapping (5-class NSL-KDD standard)
ATTACK_CATEGORIES = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L (Remote to Local)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R (User to Root)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "xterm": "U2R", "ps": "U2R", "sqlattack": "U2R",
}

LABEL_MAP = {"Normal": 0, "DoS": 1, "Probe": 2, "R2L": 3, "U2R": 4}
NUM_CLASSES = len(LABEL_MAP)

# ============================================================
#  Training
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 256
NUM_WORKERS = 4

# GAN hyperparameters (Innovation #1)
GAN_LATENT_DIM = 100
GAN_EPOCHS = 100
GAN_LR = 2e-4
GAN_BETA1 = 0.5

# Self-Learning (Innovation #2)
SELF_LEARN_WINDOW = 1000       # samples per online update
SELF_LEARN_THRESHOLD = 0.05    # retrain if accuracy drops by 5%
CONFIDENCE_THRESHOLD = 0.75    # min confidence for pseudo-labeling

# Target performance (from HCAPT-Defense spec)
TARGET_ACCURACY = 0.97
TARGET_FPR = 0.015
TARGET_RESPONSE_TIME_S = 1.0

import multiprocessing as _mp
# Only print once in main process (avoids spam from DataLoader workers on Windows)
if _mp.current_process().name == "MainProcess":
    print(f"[config] Device: {DEVICE} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[config] GPU: {torch.cuda.get_device_name(0)}")
