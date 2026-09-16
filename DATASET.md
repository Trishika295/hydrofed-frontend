# Dataset Profile & Partitioning

This document outlines the Kaggle chest X-ray dataset profiles and our patient-level separation protocols.

---

## 1. Class & Subclass Profile
The original Kaggle pediatric chest X-ray images are cataloged into two main directories: `NORMAL` and `PNEUMONIA`.
Our dataset preprocessor and loader parse pneumonia images into subclasses based on file naming identifiers:
- **NORMAL**: Healthy controls.
- **BACTERIA**: Bacterial pneumonia cases.
- **VIRUS**: Viral pneumonia cases.

---

## 2. Patient Identification Methodology
To prevent patient-level data leakage, patient IDs are parsed directly from file name structures:
- **PNEUMONIA (Bacteria / Virus)**: Filenames follow `person[ID]_[type]_[image_num].jpeg`. The extracted patient ID is `person[ID]` (e.g. `person1000`).
- **NORMAL**: Filenames follow `IM-[ID]-[image_num].jpeg` or `NORMAL2-IM-[ID]-[image_num].jpeg`. The extracted patient ID is `normal_[ID]`.

---

## 3. Patient-Level Splitting
Dividing images randomly results in severe data leakage as multiple chest scans of the same patient would populate both training and validation sets.
We combined all images, grouped them by patient ID, and performed a strict 70% Train, 15% Val, 15% Test partition.

| Split | Unique Patients | Total Images | NORMAL Class | PNEUMONIA (BACTERIA / VIRUS) |
|---|---|---|---|---|
| **TRAIN** | 1,952 | 4,112 | 1,108 | 3,004 (1,969 / 1,035) |
| **VAL** | 418 | 860 | 244 | 616 (401 / 215) |
| **TEST** | 420 | 852 | 227 | 625 (390 / 235) |
| **TOTAL** | **2,790** | **5,824** | **1,579** | **4,245 (2,760 / 1,485)** |

---

## 4. Dirichlet Partitioning (Non-IID Setup)
The 4,112 training samples are partitioned across 60 simulated clinic nodes using a Dirichlet distribution ($\alpha = 0.5$) based on subclass labels. This creates a non-IID clinic distribution where some clinics observe high bacterial prevalence, some viral, and some mostly healthy controls, replicating real-world multi-demographic edge clinic profiles.
