"""
Reorganise data/ into a scientific naming convention.

Usage:
    python reorg.py             # dry-run (prints actions, does nothing)
    python reorg.py --apply     # actually move/delete

Convention:
  - kebab-case for folder names
  - snake_case for file names
  - no spaces, no special characters
  - one folder per dataset; sub-folders by modality
  - one README.md per dataset folder
"""
from __future__ import annotations
import os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data"
APPLY = "--apply" in sys.argv


# ----------------------------------------------------------------------
# Action log
# ----------------------------------------------------------------------
class Log:
    def __init__(self):
        self.moves = []
        self.deletes = []
        self.mkdirs = []
    def move(self, src, dst):
        self.moves.append((src, dst))
    def delete(self, path):
        self.deletes.append(path)
    def mkdir(self, path):
        self.mkdirs.append(path)
    def summary(self):
        print(f"\n{'='*70}")
        print(f"PLAN  ({'APPLY' if APPLY else 'DRY-RUN'})")
        print(f"{'='*70}")
        print(f"  mkdir : {len(self.mkdirs)}")
        print(f"  move  : {len(self.moves)}")
        print(f"  delete: {len(self.deletes)}")

LOG = Log()


def mkdir(p: Path):
    LOG.mkdir(p)
    print(f"  MKDIR  {p.relative_to(ROOT)}")
    if APPLY:
        p.mkdir(parents=True, exist_ok=True)


def move(src: Path, dst: Path):
    if not src.exists():
        print(f"  SKIP   {src.relative_to(ROOT)} (not found)")
        return
    LOG.move(src, dst)
    rel_src = src.relative_to(ROOT)
    rel_dst = dst.relative_to(ROOT)
    print(f"  MOVE   {rel_src}  ->  {rel_dst}")
    if APPLY:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def delete(path: Path):
    if not path.exists():
        return
    LOG.delete(path)
    print(f"  DELETE {path.relative_to(ROOT)}")
    if APPLY:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


# ----------------------------------------------------------------------
# Reorganisation plan
# ----------------------------------------------------------------------
def reorg_nsl_kdd():
    print("\n[1/5] NSL-KDD")
    target = ROOT / "nsl-kdd"
    mkdir(target)
    move(ROOT / "KDDTrain+.txt",            target / "train.txt")
    move(ROOT / "KDDTrain+_20Percent.txt",  target / "train_20pct.txt")
    move(ROOT / "KDDTest+.txt",             target / "test.txt")
    move(ROOT / "KDDTest-21.txt",           target / "test_21.txt")


def reorg_unsw_nb15():
    print("\n[2/5] UNSW-NB15")
    src = ROOT / "UNSW-NB15 dataset"
    target = ROOT / "unsw-nb15"
    mkdir(target)
    if (src / "ReadMe.pdf").exists():
        move(src / "ReadMe.pdf", target / "ReadMe.pdf")
    # Argus
    argus_src = src / "Argus Files"
    if argus_src.exists():
        argus_dst = target / "argus"
        mkdir(argus_dst)
        for sub in argus_src.iterdir():
            if sub.is_dir():
                # rename: "Argus Files 22-1-2015" -> "2015-01-22"
                name = sub.name
                if "22-1-2015" in name:
                    move(sub, argus_dst / "2015-01-22")
                elif "17-2-2015" in name:
                    move(sub, argus_dst / "2015-02-17")
                else:
                    safe = name.lower().replace(" ", "_").replace("-", "_")
                    move(sub, argus_dst / safe)
    # BRO
    bro_src = src / "BRO Files"
    if bro_src.exists():
        bro_dst = target / "bro"
        mkdir(bro_dst)
        for sub in bro_src.iterdir():
            if sub.is_dir():
                name = sub.name
                if "17-2-2015" in name:
                    move(sub, bro_dst / "2015-02-17")
                else:
                    safe = name.lower().replace(" ", "_").replace("-", "_")
                    move(sub, bro_dst / safe)
    # Delete now-empty old folders
    delete(src)
    delete(ROOT / "__UNSW-NB15 dataset")


def reorg_cic_ids_2017():
    print("\n[3/5] CIC-IDS2017")
    target = ROOT / "cic-ids-2017"
    mkdir(target)
    # Renamed file mapping
    rename_map = {
        "Monday-WorkingHours.pcap_ISCX.csv":                            "monday-benign.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv":                           "tuesday-bruteforce.csv",
        "Wednesday-workingHours.pcap_ISCX.csv":                         "wednesday-dos-heartbleed.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv":       "thursday-morning-webattacks.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv":  "thursday-afternoon-infiltration.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv":                    "friday-morning-botnet.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv":         "friday-afternoon-portscan.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv":             "friday-afternoon-ddos.csv",
    }
    ml_src = ROOT / "MachineLearningCVE"
    if ml_src.exists():
        ml_dst = target / "ml-csv"
        mkdir(ml_dst)
        for old, new in rename_map.items():
            move(ml_src / old, ml_dst / new)
        delete(ml_src)
    # Labelled flows -> labelled-flows
    lab_src = ROOT / "GeneratedLabelledFlows" / "TrafficLabelling"
    if lab_src.exists():
        move(lab_src, target / "labelled-flows")
    delete(ROOT / "GeneratedLabelledFlows")


def reorg_ton_iot():
    print("\n[4/5] ToN-IoT")
    target = ROOT / "ton-iot"
    mkdir(target)
    # Network: deeply nested at ToN-IoT/Train_Test_Network_dataset/Train_Test_Network_dataset/train_test_network.csv
    nested = ROOT / "ToN-IoT" / "Train_Test_Network_dataset" / "Train_Test_Network_dataset" / "train_test_network.csv"
    nested_alt = ROOT / "ToN-IoT" / "Train_Test_Network_dataset" / "train_test_network.csv"
    network_dst = target / "network"
    mkdir(network_dst)
    if nested.exists():
        move(nested, network_dst / "train_test_network.csv")
    elif nested_alt.exists():
        move(nested_alt, network_dst / "train_test_network.csv")
    delete(ROOT / "ToN-IoT")
    # IoT telemetry: 7 device files
    iot_src = ROOT / "Train_Test_IoT_dataset"
    iot_dst = target / "iot-telemetry"
    mkdir(iot_dst)
    iot_rename = {
        "Train_Test_IoT_Fridge.csv":        "fridge.csv",
        "Train_Test_IoT_GPS_Tracker.csv":   "gps_tracker.csv",
        "Train_Test_IoT_Garage_Door.csv":   "garage_door.csv",
        "Train_Test_IoT_Modbus.csv":        "modbus.csv",
        "Train_Test_IoT_Motion_Light.csv":  "motion_light.csv",
        "Train_Test_IoT_Thermostat.csv":    "thermostat.csv",
        "Train_Test_IoT_Weather.csv":       "weather.csv",
    }
    for old, new in iot_rename.items():
        move(iot_src / old, iot_dst / new)
    delete(iot_src)
    # Stale ZIPs
    delete(ROOT / "Train_Test_IoT_dataset (1).zip")
    delete(ROOT / "Train_Test_Network_dataset.zip")


def reorg_edge_iiotset():
    print("\n[5/5] Edge-IIoTset")
    src = ROOT / "Edge-IIoTset dataset"
    target = ROOT / "edge-iiotset"
    mkdir(target)
    # Normal
    normal_src = src / "Normal traffic"
    if normal_src.exists():
        move(normal_src, target / "normal")
    # Selected (ML-DL ready)
    selected_src = src / "Selected dataset for ML and DL"
    if selected_src.exists():
        move(selected_src, target / "ml-dl-selected")
    # Attack — separate CSVs from PCAPs, rename CSVs
    attack_src = src / "Attack traffic"
    attack_dst = target / "attack"
    pcap_dst = target / "pcap-attack"
    if attack_src.exists():
        mkdir(attack_dst)
        mkdir(pcap_dst)
        rename_csv = {
            "Backdoor_attack.csv":              "backdoor.csv",
            "DDoS_HTTP_Flood_attack.csv":       "ddos_http_flood.csv",
            "DDoS_ICMP_Flood_attack.csv":       "ddos_icmp_flood.csv",
            "DDoS_TCP_SYN_Flood_attack.csv":    "ddos_tcp_syn_flood.csv",
            "DDoS_UDP_Flood_attack.csv":        "ddos_udp_flood.csv",
            "MITM_attack.csv":                  "mitm.csv",
            "OS_Fingerprinting_attack.csv":     "os_fingerprinting.csv",
            "Password_attack.csv":              "password.csv",
            "Port_Scanning_attack.csv":         "port_scanning.csv",
            "Ransomware_attack.csv":            "ransomware.csv",
            "SQL_injection_attack.csv":         "sql_injection.csv",
            "Uploading_attack.csv":             "uploading.csv",
            "Vulnerability_scanner_attack.csv": "vulnerability_scanner.csv",
            "XSS_attack.csv":                   "xss.csv",
        }
        for f in attack_src.iterdir():
            n = f.name
            if n in rename_csv:
                move(f, attack_dst / rename_csv[n])
            elif f.suffix.lower() == ".pcap":
                # Keep PCAPs but rename to snake_case
                safe = (n.lower()
                        .replace(" ", "_")
                        .replace("(", "").replace(")", "")
                        .replace("+", "_plus_")
                        .replace(".pcap", ""))
                # collapse multiple underscores
                while "__" in safe:
                    safe = safe.replace("__", "_")
                move(f, pcap_dst / f"{safe}.pcap")
            else:
                # unknown extension — leave with safe rename
                safe = n.lower().replace(" ", "_")
                move(f, attack_dst / safe)
        delete(attack_src)
    delete(src)


def main():
    if not ROOT.exists():
        print(f"ERROR: data root not found: {ROOT}")
        return 1
    reorg_nsl_kdd()
    reorg_unsw_nb15()
    reorg_cic_ids_2017()
    reorg_ton_iot()
    reorg_edge_iiotset()
    LOG.summary()
    if not APPLY:
        print("\n  --> DRY-RUN only. Re-run with `--apply` to execute.")
    else:
        print("\n  --> Reorganisation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
