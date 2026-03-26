---
name: Notebook is single source of truth
description: All report content must live in the notebook markdown cells — no external .md files. Word report is exported directly from notebook.
type: feedback
---

The notebook is the single source of truth for the report. When exporting to Word, only notebook markdown cells are used — no external .md files. All report text must be written directly into the notebook's markdown cells with enough substance to stand alone without code cells visible.

**Why:** User deleted the separate .md report files because they expected all content to be in the notebook. The Word export strips code cells, so markdown cells must be self-contained and readable as a report.

**How to apply:** Always write report-quality text directly into notebook markdown cells. Never create separate report files unless explicitly asked. Ensure markdown cells have enough context to be understood without the adjacent code.
