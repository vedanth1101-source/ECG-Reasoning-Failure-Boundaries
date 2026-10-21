# CHAPTER 6

# CONCLUSIONS AND FUTURE WORK

In conclusion, this project examined whether the diagnoses of an ECG foundation model remain supported by the evidence left after leads are removed. A frozen ECG-FM encoder [7] and a Control Reasoning Transformer were evaluated on eight benchmark conditions across seven zero-masked lead configurations. The test macro-AUROC stayed between 0.9479 and 0.9540 down to two leads and fell to 0.9215 with lead II alone, and performance depended on which leads remained rather than how many.

These are descriptive results from one seed of an authorisation-unverified test run, and they do not establish clinical suitability or evidence grounding. The grounding gate, its metrics, the Reasoning Failure Boundary, the DFB and the AURDC are not yet implemented. The observation extractor cannot measure wide QRS complexes, morphology or Q waves, and lead removal is simulated rather than recorded.

As future work, the extractor and evidence schema will be repaired, the grounding gate implemented and the Reasoning Failure Boundary measured in an authorised, versioned evaluation. The SNOMED CT-based Clinical Knowledge Prior [16] will then be added, and the system validated on independent cohorts.


<!-- CITATION MAP (not rendered). Global numbering continues from earlier chapters; numbers shared with earlier chapters are unchanged. Keys refer to REFERENCES.bib / REFERENCE_LIBRARY.md.
[7] MCKEEN2025ECGFM
[16] SNOMED2026CDSGUIDE
-->
