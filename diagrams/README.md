# VoiceAffect Architecture & Engineering Diagrams

This directory contains the formal architectural, lifecycle, and data modeling diagrams for the **Bilingual Voice Translation System With Emotion Awareness (VoiceAffect)**.

---

## Diagram Index

| Document | Description | Format & Standards |
| :--- | :--- | :--- |
| **[1. SDLC Specification & Diagram](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/diagrams/sdlc-diagram.md)** | Complete Software Development Life Cycle (Agile-V-Model hybrid) detailing all phases from PRD and mathematical design to scientific auditing and MPS deployment. | Mermaid Flowchart & Gantt Timeline |
| **[2. Project Workflow Diagram](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/diagrams/project-workflow-diagram.md)** | Comprehensive runtime dataflows covering Live Speech Streaming (WebSocket), Batch File Upload, and Direct Text Translation across the 6-stage neural pipeline. | Mermaid Sequence & Process Flowcharts |
| **[3. Entity-Relationship (ER) Diagram](file:///Users/hardikmishra/Data%20science%20projects/live%20translation%20system/diagrams/er-diagram.md)** | Full Entity-Relationship schema, conceptual/logical data dictionary, session tracking, audio artifacts, emotion/VAD telemetry, and privacy retention rules. | Mermaid `erDiagram` Schema & Data Dictionary |

---

## Architectural Highlights

* **Model & Pipeline Pipeline**: ASR (`Whisper-Tiny`), SER (`Wav2Vec2-XLSR`), NMT (`MarianMT`), Latent-Depth Prosody Gating ($z_l$), and Expressive TTS.
* **Execution Environment**: 100% offline local inference on Apple Silicon (MPS) and CPU with zero paid cloud APIs.
* **Privacy & Security**: Ephemeral audio handling with automatic TTL garbage collection and consent-gated prosodic modulation.

---

## Viewing Diagrams

All diagrams are authored in GitHub Flavored Markdown (GFM) using native **Mermaid** syntax. They render natively in:
- Antigravity IDE Markdown preview
- GitHub / GitLab web preview
- VS Code (with Markdown Preview Mermaid Support)
- Any standard Mermaid renderer (`mermaid.live`)
