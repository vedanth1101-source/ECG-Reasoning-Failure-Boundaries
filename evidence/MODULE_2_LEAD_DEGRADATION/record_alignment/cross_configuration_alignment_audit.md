# Module 2 — Cross-Configuration Record Alignment Audit

**Module:** Module 2 (Progressive ECG Lead Masking and Availability-Mask Generation)  
**Governing Manifest:** [`runs/module2_v1/MODULE2_FROZEN.json`](file:///e:/ECG/runs/module2_v1/MODULE2_FROZEN.json)  
**Master Configuration:** [`runs/module2_v1/reports/module2_variant_config.json`](file:///e:/ECG/runs/module2_v1/reports/module2_variant_config.json)  
**Audit Purpose:** Verify 1-to-1 record ID alignment, split alignment, and label invariance across all seven lead degradation configurations ($258,216$ physical arrays).

---

## 1. Workload Scope and Matrix

Module 2 systematically generates seven lead configurations from each of the 36,888 Module 1 standardized recordings:

$$\text{Total Physical Arrays} = 36,888 \text{ records} \times 7 \text{ configurations} = 258,216 \text{ physical arrays}$$

### Cross-Configuration Record Inventory
| Lead Configuration | Active Leads | Train Records (Folds 1–7) | Validation Records (Fold 8) | Test Records (Folds 9–10) | Total Physical Files | Directory Location |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **12-lead** | 12 | 25,825 | 3,688 | 7,375 | 36,888 | `runs/module2_v1/masked_leads/12-lead/` |
| **10-lead** | 10 | 25,825 | 3,688 | 7,375 | 36,888 | `runs/module2_v1/masked_leads/10-lead/` |
| **8-lead** | 8 | 25,825 | 3,688 | 7,375 | 36,888 | `runs/module2_v1/masked_leads/8-lead/` |
| **6-lead** | 6 | 25,825 | 3,688 | 7,375 | 36,888 | `runs/module2_v1/masked_leads/6-lead/` |
| **4-lead** | 4 | 25,825 | 3,688 | 7,375 | 36,888 | `runs/module2_v1/masked_leads/4-lead/` |
| **2-lead** | 2 | 25,825 | 3,688 | 7,375 | 36,888 | `runs/module2_v1/masked_leads/2-lead/` |
| **1-lead** | 1 | 25,825 | 3,688 | 7,375 | 36,888 | `runs/module2_v1/masked_leads/1-lead/` |
| **Pooled Total** | — | **180,775** | **25,816** | **51,625** | **258,216** | **7 Configuration Directories** |

---

## 2. Record-ID Alignment and Invariance Verification

A complete set-difference and hash audit across all 7 directories verified:
1. **Exact Set Equality:** Every record ID present in `runs/module1_v2/dataset/` (`PTBXL_1` through `PTBXL_21837`, `CHAPMAN_JS00001` through `CHAPMAN_JS10631`, `MIMIC_40004943` through `MIMIC_42999435`) exists in each of the seven Module 2 subdirectories.
   $$\text{Set}(12\text{-lead}) = \text{Set}(10\text{-lead}) = \dots = \text{Set}(1\text{-lead}) = \text{Set}(\text{Module 1})$$
2. **Zero Missing or Dropped Records:** Exactly 0 missing files were detected across all 258,216 file locations.
3. **Split Partition Invariance:** A record assigned to `train`, `validation`, or `test` in Module 1 maintains the exact same split assignment across all seven configurations.
4. **Label Invariance:** Clinical ground truth labels ($y \in \{0, 1, \text{NaN}\}^8$) and observation masks ($m_{\text{obs}} \in \{0, 1\}^8$) are held completely fixed. Degrading the sensor montage does not alter the underlying physiological ground truth.

---

## 3. Scientific Purpose of Cartesian Alignment

Holding the records and patient clusters strictly constant across all seven configurations enables **counterfactual channel deprivation analysis**:
- A performance decline from 12-lead down to 1-lead represents purely the loss of spatial electrocardiographic information, rather than cohort shift, patient variability, or label noise.
- Every recording serves as its own internal control across the degradation ladder.
