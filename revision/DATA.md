# Data

Raw data are not redistributed. Download each corpus from its source and place the files at
the paths below (relative to the package root). Hashes are those of the files used for every result.

| Corpus | Source |
|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html (KDDTrain+.txt -> train.txt, KDDTest+.txt -> test.txt) |
| DAPT 2020 | https://gitlab.com/asu22/dapt2020 (folder csv/) |
| ToN-IoT network | https://research.unsw.edu.au/projects/toniot-datasets (Train_Test_Network dataset) |
| ToN-IoT telemetry | https://research.unsw.edu.au/projects/toniot-datasets (Train_Test_IoT dataset) |
| Edge-IIoTset | https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications (ML-EdgeIIoT-dataset.csv) |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset (official training/testing partitions) |

| File | Bytes | SHA-256 |
|---|---|---|
| `data/nsl-kdd/train.txt` | 19,109,424 | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` |
| `data/nsl-kdd/test.txt` | 3,441,513 | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` |
| `data/UNSW-NB15/UNSW_NB15_training-set.csv` | 15,380,800 | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` |
| `data/UNSW-NB15/UNSW_NB15_testing-set.csv` | 32,293,018 | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` |
| `data/ton-iot/network/train_test_network.csv` | 29,902,775 | `26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974` |
| `data/edge-iiotset/ml-dl-selected/ML-EdgeIIoT-dataset.csv` | 82,184,390 | `53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a` |
| `data/dapt-2020/csv/enp0s3-monday-pvt.pcap_Flow.csv` | 1,787,717 | `0be551840a27961354941026fe20f52ae0f93f8db5fb210085732701416d8df6` |
| `data/dapt-2020/csv/enp0s3-monday.pcap_Flow.csv` | 4,671,687 | `30c43e0259b86beac835ad8a5bb3d1758aacd37dfea1bb6f12e5153265e4b716` |
| `data/dapt-2020/csv/enp0s3-public-thursday.pcap_Flow.csv` | 4,792,113 | `a51b9a974558557922b6912f6e532ab2a2f236beb193f5e0abeddfc1ab39baac` |
| `data/dapt-2020/csv/enp0s3-public-tuesday.pcap_Flow.csv` | 14,985,262 | `efa02b13d4da2e9e5e3f5ee3ff9e2f49b9d2815e8a1e553935b0fdd117cfc61f` |
| `data/dapt-2020/csv/enp0s3-public-wednesday.pcap_Flow.csv` | 9,336,784 | `f03823e611654950e31d87d0f700941697664150f5b6c25a7289e55cf03e4e8c` |
| `data/dapt-2020/csv/enp0s3-pvt-thursday.pcap_Flow.csv` | 2,049,946 | `0a927ed1c6554f77cd578c70832a51295c4bfa90d13d5b93436a73dbf4561401` |
| `data/dapt-2020/csv/enp0s3-pvt-tuesday.pcap_Flow.csv` | 1,413,634 | `4e3643de6ccd0259329a106ed59ec5e52a3b87ea2b4a4270c896166b37d6d14c` |
| `data/dapt-2020/csv/enp0s3-pvt-wednesday.pcap_Flow.csv` | 793,982 | `a20ebb7ab998500fab6593c1110b65692aed3df16269ab26303a4ae72948f7a2` |
| `data/dapt-2020/csv/enp0s3-tcpdump-friday.pcap_Flow.csv` | 3,671,983 | `c75d7ae54014cce59b1bc0fee7de756c7ad249d5d8fcb5c24e7630d4f0e36e7e` |
| `data/dapt-2020/csv/enp0s3-tcpdump-pvt-friday.pcap_Flow.csv` | 1,443,514 | `fdf4a09646cf6e43003bcc15192d2bca275033c3794846fe9b372c7951d8d92c` |
| `data/ton-iot/iot-telemetry/fridge.csv` | 1,678,744 | `56377a00b97204ea65f8912eadea9e626fe05d091f975f5be64c94c85efe3149` |
| `data/ton-iot/iot-telemetry/garage_door.csv` | 1,739,414 | `44b2d63f233ed239d604815f53e2b2ba6ec2b9bc604a9f6fbed6b98d55e53188` |
| `data/ton-iot/iot-telemetry/gps_tracker.csv` | 2,122,967 | `71c6cc302b14822aea699cf41a02dad36089ea59b87a28000b38a44b2d23cb9b` |
| `data/ton-iot/iot-telemetry/modbus.csv` | 1,694,191 | `78345a857244e671b0c255ca65aac619049632448fc5aa736f1cea255f308cbb` |
| `data/ton-iot/iot-telemetry/motion_light.csv` | 1,469,248 | `e0e145ff3d8145d4ad713380a291248dba74e75a70bd75525719ffaaaf2f5242` |
| `data/ton-iot/iot-telemetry/thermostat.csv` | 1,464,768 | `b475654a43ec885fd8c3409d9e85b7902ecc3ad47440948466f4fe3559fd1f6d` |
| `data/ton-iot/iot-telemetry/weather.csv` | 2,587,701 | `10c49fe8a6eae98db688e277e96218c9a9bf587cc5aee5962833470366dafdba` |

Notes:
* DAPT 2020: `enp0s3-pvt-thursday.pcap_Flow.csv` has no stage labels and is excluded by `r08`.
* UNSW-NB15: in our copy the 175,341-row official *training* partition is named `UNSW_NB15_testing-set.csv`; `r17` assigns partitions by row count.
* ToN-IoT telemetry is audited (`r15`) but not used for modelling.
* Re-analysis from the shipped predictions (`reproduce_from_predictions.ps1`) needs none of these files.
