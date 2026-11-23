from pathlib import Path

# Dizinler
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Sabitler ve etiket sütunu
RANDOM_STATE = 42
LABEL_COL = "Label"

# Dosya bazlı ayrık bölme (File-Disjoint Split — dosyalar arası sızıntıyı önler)
TRAIN_FILES = ["Merged01.csv", "Merged02.csv", "Merged03.csv", "Merged04.csv"]
VAL_FILES = ["Merged07.csv", "Merged08.csv"]
TEST_FILES_LOCKED = ["Merged09.csv", "Merged10.csv"]  # Yalnızca final testte (exp4) açılır

# Pencere-boyutu artefaktı taşıyan sütunlar (Number=10 vs 100 ayrımı ve kümülatif sayaçlar)
WINDOW_DEPENDENT_COLS = [
    "Number",
    "Tot sum",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
]

# Pencere-boyutundan bağımsız 33 sayısal özellik (CICIoT2023 39 özellik - 6 pencere/sayaç sütunu)
FEATURES_33 = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Variance",
]

# HTTPS dışarıda bırakılan 32 özellik (ablation referansı)
FEATURES_32_NO_HTTPS = [c for c in FEATURES_33 if c != "HTTPS"]

# 8-Sınıf kategori eşlemesi (Benign + 7 ana saldırı ailesi)
CLASS_8_ORDER = [
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web",
]


def map_label_to_8_classes(label: str) -> str:
    s = str(label).strip().upper()
    if s == "BENIGN":
        return "Benign"
    if s.startswith("DDOS"):
        return "DDoS"
    if s.startswith("DOS"):
        return "DoS"
    if s.startswith("MIRAI"):
        return "Mirai"
    if s.startswith("RECON") or s == "VULNERABILITYSCAN":
        return "Recon"
    if s in ("MITM-ARPSPOOFING", "DNS_SPOOFING"):
        return "Spoofing"
    if s == "DICTIONARYBRUTEFORCE":
        return "BruteForce"
    return "Web"

# Exp 2 sonunda elenen 9 sutun haric kalan 24 cekirdek ozellik
FEATURES_24 = [
    c for c in FEATURES_33
    if c not in {
        "Tot size", "IPv", "LLC", "Variance",
        "cwr_flag_number", "SMTP", "Telnet", "IRC", "IGMP",
    }
]

# Exp 4b Checkpoint 1: Sifir kayipli (ROC=0.99808, @0.1% FPR=0.091%, Rec=98.44%)
FEATURES_TOP15 = [
    "HTTPS",
    "AVG",
    "IAT",
    "Header_Length",
    "ack_flag_number",
    "Max",
    "syn_flag_number",
    "Time_To_Live",
    "Min",
    "psh_flag_number",
    "rst_flag_number",
    "fin_flag_number",
    "Std",
    "HTTP",
    "ARP",
]

# Exp 4b Checkpoint 2: Ultra kompakt (ROC=0.99795, @0.1% FPR=0.097%, Rec=98.38%)
FEATURES_TOP10 = FEATURES_TOP15[:10]