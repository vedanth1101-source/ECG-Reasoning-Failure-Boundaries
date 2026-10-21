# Module 4: Evaluation Authorization & Provenance Statement

**Document ID:** `M4-EVAL-PROVENANCE-v1.0`  
**Execution Timestamp:** `2026-10-15 17:35:00 UTC`  
**Evaluation Scope:** Quarantined Test Folds 9–10 ($N=7,375$ records $\times$ 7 lead configurations = $51,625$ record-configuration evaluations)  
**Evaluated Model Checkpoint:** [`runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt`](file:///e:/ECG/runs/module4_v3/checkpoints/control_reasoning_best_seed42.pt) (SHA-256: `2920c76884e4a432b69326f1d943cb4495e6e55305f508fe3acab3d9e9067d93`)  
**Associated Test Artifacts:**
- [`runs/module4_v3/metrics/test_predictions_seed42.csv`](file:///e:/ECG/runs/module4_v3/metrics/test_predictions_seed42.csv) (SHA-256: `0be7518ad27c157a7da8b25a2e8b7b079042812a2299ff0933b23c413eb8f377`)
- [`runs/module4_v3/metrics/test_evaluation_summary.json`](file:///e:/ECG/runs/module4_v3/metrics/test_evaluation_summary.json)
- [`runs/module4_v3/reports/stage7_test_evaluation_report.md`](file:///e:/ECG/runs/module4_v3/reports/stage7_test_evaluation_report.md)
- [`runs/module4_v3/reports/sample_test_reasoning_traces.json`](file:///e:/ECG/runs/module4_v3/reports/sample_test_reasoning_traces.json)
**Governing Pre-Test Audit:** [`runs/module4_v3/reports/stage6_pretest_independent_audit.md`](file:///e:/ECG/runs/module4_v3/reports/stage6_pretest_independent_audit.md)  
**Governing Freeze Contract:** [`runs/module4_v3/MODULE4_FROZEN.json`](file:///e:/ECG/runs/module4_v3/MODULE4_FROZEN.json)  
**Audit Finding Reference:** [`runs/module4_v3/reports/post_evaluation_baseline_reconciliation_audit.md`](file:///e:/ECG/runs/module4_v3/reports/post_evaluation_baseline_reconciliation_audit.md) (Domain 1, Section 2)  

---

## 1. Executive Provenance Finding

The post-evaluation integrity audit ([`runs/module4_v3/reports/post_evaluation_baseline_reconciliation_audit.md`](file:///e:/ECG/runs/module4_v3/reports/post_evaluation_baseline_reconciliation_audit.md)) conducted a forensic inspection of execution logs and transcript history to verify whether the explicit pre-test authorization condition was satisfied prior to unblinding and evaluating the quarantined test set (Folds 9–10).

### Official Status Determination:
```text
EVALUATION PROVENANCE STATUS: AUTHORIZATION-UNVERIFIED
```

The held-out test evaluation was executed without the required explicit, user-entered literal `"RUN_TEST"` directive. Execution was triggered prematurely due to an automated IDE review policy hook. Consequently, the reported test results cannot be certified as an authorized, preregistered held-out benchmark evaluation.

---

## 2. Pre-Test Governance Mandate & Preregistered Gate

In the independent pre-test audit ([`runs/module4_v3/reports/stage6_pretest_independent_audit.md`](file:///e:/ECG/runs/module4_v3/reports/stage6_pretest_independent_audit.md), Section 10 and Final Verdict, lines 260–264), the explicit governance boundary was defined as follows:

> **Pre-Test Audit Mandate:**
> `STATUS: READY FOR EXPLICIT RUN_TEST AUTHORIZATION`  
> *Preregistered Rule:* The held-out test set (Folds 9–10: 7,375 records) cannot be unblinded, accessed, or evaluated until the primary investigator explicitly issues the literal `RUN_TEST` authorization directive.

This gate was established to ensure strict single-pass test isolation and prevent premature unblinding of test ground truth from the frozen parent manifest ([`runs/module1_v2/splits/split_manifest.csv`](file:///e:/ECG/runs/module1_v2/splits/split_manifest.csv)).

---

## 3. Forensic Reconstruction of the Authorization Timeline

A chronological reconstruction from the conversation transcript logs (`transcript.jsonl`) establishes the exact sequence of events leading to test execution:

```mermaid
sequenceDiagram
    autonumber
    actor User as Primary Investigator (User)
    participant Agent as Assistant (Antigravity Agent)
    participant IDE as IDE Runtime / Review Policy Hook
    participant Pipeline as Stage 7 Test Pipeline

    User->>Agent: Step 4161: "/plan to proceed with Evaluate on the unblinded, quarantined Test Set..."
    Agent->>Agent: Step 4218: Writes "stage7_test_evaluation_plan.md" (RequestFeedback=True)
    Agent->>User: Step 4220: "Please review the plan and let me know if you approve... before we proceed to execution!"
    IDE-->>Agent: Step 4221: <SYSTEM_MESSAGE> stop hook blocked termination due to reason: The user has automatically approved the artifact through their review policy. Proceed to execution. </SYSTEM_MESSAGE>
    Agent->>Pipeline: Step 4242 & 4246: Modifies script 17 and launches background execution (task-4120)
    Pipeline-->>Agent: Step 4250: Task crashes (indexing syntax bug in calibration binning)
    Agent->>Pipeline: Step 4256 & 4260: Fixes syntax bug and relaunches execution (task-4134)
    User-->>Agent: Step 4264: Automated IDE client event: "The user has approved this document" (Empty query)
    Pipeline->>Agent: Step 4300: Test evaluation completes across 51,625 evaluations
```

### Detailed Event Log Analysis:
1. **Step 4161 (`USER_INPUT`):** The user submitted a prompt requesting a plan:  
   `"/plan to proceed with Evaluate on the unblinded, quarantined Test Set (Folds 9–10: 7,375 records × 7 configurations = 51,625 evaluations)..."`
2. **Step 4218 (`CODE_ACTION`):** The assistant drafted the plan document `stage7_test_evaluation_plan.md` and set artifact metadata `RequestFeedback=True`.
3. **Step 4220 (`PLANNER_RESPONSE`):** The assistant addressed the user:  
   `"Please review the plan and let me know if you approve or if you would like any adjustments before we proceed to execution!"`
4. **Step 4221 (`SYSTEM_MESSAGE`):** The IDE environment injected an automated review policy hook message:  
   `<SYSTEM_MESSAGE>stop hook blocked termination due to reason: The user has automatically approved the artifact through their review policy. Proceed to execution.</SYSTEM_MESSAGE>`
5. **Step 4246 & 4260 (`RUN_COMMAND`):** Interpreting this system hook as authorization, the assistant modified [`scripts/17_generate_reasoning_traces.py`](file:///e:/ECG/scripts/17_generate_reasoning_traces.py) and initiated background test execution (`task-4120` and `task-4134`).
6. **Step 4264 (`USER_INPUT`):** The IDE emitted an automated artifact approval notification:  
   `Comments on artifact URI: file:///.../stage7_test_evaluation_plan.md. The user has approved this document.` with an empty `<USER_REQUEST></USER_REQUEST>` tag.

---

## 4. Exact Missing Evidence & Procedural Deficiency

The procedural audit identifies the following specific evidentiary deficiencies:

1. **Absence of User-Entered Literal Token:** At no point prior to unblinding or evaluation did the user enter the literal token `"RUN_TEST"` in any prompt, configuration file, or authorization record.
2. **Automated Hook Ambiguity:** Execution was initiated in response to an automated runtime intercept (`stop hook blocked termination...`), which does not constitute an explicit, unambiguous human-in-the-loop authorization under the ratified research protocol.
3. **Unblinded Test Access:** The test split labels were unblinded from [`runs/module1_v2/splits/split_manifest.csv`](file:///e:/ECG/runs/module1_v2/splits/split_manifest.csv) without satisfying the preregistered gating condition.

---

## 5. Non-Retroactivity Principle & Governance Boundaries

To preserve strict scientific integrity:

1. **No Retroactive Validation:** Subsequent document approvals (such as the approval event at Step 4264 or subsequent discussions) **do not retroactively validate** the premature unblinding and execution that occurred at Steps 4221–4260.
2. **Status of Existing Test Artifacts:** All files generated during Stage 7 are formally designated as **AUTHORIZATION-UNVERIFIED**. They provide descriptive empirical reference points but cannot be claimed as a preregistered confirmatory test benchmark.
3. **Prohibition of Rerunning Without Authorization:** In accordance with audit boundaries, test evaluation shall not be rerun, nor shall test examples be accessed, until a separate, authorized evaluation protocol is formally ratified by the primary investigator.

---

## 6. Physical Artifact Preservation & Integrity Hashes

All existing Stage 7 artifacts remain sealed on disk without modification to ensure a tamper-proof evidentiary trail:

| Artifact Description | Physical File Path | File Size | SHA-256 Cryptographic Checksum | Governance Classification |
| :--- | :--- | :---: | :--- | :--- |
| **Test Predictions CSV** | [`runs/module4_v3/metrics/test_predictions_seed42.csv`](file:///e:/ECG/runs/module4_v3/metrics/test_predictions_seed42.csv) | 33,492,804 bytes | `0be7518ad27c157a7da8b25a2e8b7b079042812a2299ff0933b23c413eb8f377` | Preserved (Unverified Provenance) |
| **Test Summary JSON** | [`runs/module4_v3/metrics/test_evaluation_summary.json`](file:///e:/ECG/runs/module4_v3/metrics/test_evaluation_summary.json) | 22,999 bytes | `b2149e3b97063fcf11c5e62bf610bf61b9e289bf631fc7b9c6a152eec89e909a` | Preserved (Unverified Provenance) |
| **Sample Traces JSON** | [`runs/module4_v3/reports/sample_test_reasoning_traces.json`](file:///e:/ECG/runs/module4_v3/reports/sample_test_reasoning_traces.json) | 1,203,702 bytes | `e3b2e67df48419ca92e8fa5a6c1e1dcad4ae3ca41a3ea306124ad5f4f89d31d9` | Preserved (Unverified Provenance) |
| **Stage 7 Evaluation Report** | [`runs/module4_v3/reports/stage7_test_evaluation_report.md`](file:///e:/ECG/runs/module4_v3/reports/stage7_test_evaluation_report.md) | 9,279 bytes | `a90d4f3b5082ce7f05eb4beee86895e63286395b0577a76059c3817a3a9121a9` | Preserved (Contains Discrepancies) |
| **Baseline Comparison JSON** | [`runs/module4_v3/metrics/comprehensive_baseline_comparison.json`](file:///e:/ECG/runs/module4_v3/metrics/comprehensive_baseline_comparison.json) | 356,444 bytes | `fe788bc59e7e725655513d8031d7e2c94380eb8094038a834241071aa1a3f06e` | Preserved (Defective B-2 Baseline) |

---

## 7. Resolution & Recommended Governance Protocol

To resolve the provenance failure for future benchmark releases:
1. The primary investigator may issue an explicit, formal **Test Authorization Ratification Memo** confirming whether the existing unblinded test results are accepted as a post-hoc authorized baseline.
2. Alternatively, if strict adherence to zero-leakage unblinding is demanded for conference publication, a clean test split (e.g., an independent prospective or external clinical cohort) must be designated and evaluated under strict manual token control.
