# SIH 2026: Satellite-Based 3D Ocean Subsurface Parameter Reconstruction (PS 2)
### Repository ID: `sih26066-two-ps` | Team OceanVision AI

Welcome to the central research and engineering repository for **Smart India Hackathon 2026 (SIH PS 2)**: **"Satellite-Based 3D Ocean Subsurface Parameter Reconstruction using Physics-Informed Deep Learning"**.

---

## 📑 Core Documentation & Pitch Deliverables

* **👉 [Comprehensive Research Synthesis & SIH 2026 Pitch Deck Guide](file:///c:/sih26066/SIH_2026_PS2_MASTER_PLAYBOOK_AND_SLIDES.md)**
  * **Part 1:** Deep Technical Research Synthesis (35+ Academic Papers, 4 Architectural Paradigms, Loss Equations, 3D SHAP Analysis, Benchmarks).
  * **Part 2:** SIH 2026 Official Playbook Strategy (5-Step PS Scorecard [25/25], 36-Hour Hackathon Roadmap, 6-Member Role Matrix, Google Developer Integration, Offline Demo Fallback).
  * **Part 3:** Strict 6-Slide SIH Presentation Deck Content (Bulleted, diagrammatic, and 100% template-compliant for immediate copy-pasting into PPT/Gamma/Canva).

---

## 🌊 Overview of Problem Statement (PS 2)

Conventional satellite remote sensing (infrared and microwave) is physically restricted to the uppermost "skin" layer of the ocean (< a few millimeters). However, critical maritime phenomena—such as **cyclone rapid intensification (Tropical Cyclone Heat Potential / $D_{26}$)**, **pelagic fisheries (Potential Fishing Zones / PFZ)**, and **underwater acoustic propagation**—are governed by the 3D subsurface thermal and haline structure down to 2000 meters.

This repository implements a **Physics-Informed Deep Learning Pipeline (Observation-Guided PINN + Hybrid ConvNeXt-Transformer)** that inverts satellite surface observations (SST, SSS, SLA, Wind Stress Curl) into continuous 3D temperature and salinity voxel grids ($0\text{ to }2000\text{ m}$) across the North Indian Ocean (Arabian Sea and Bay of Bengal).

---

## 🏛️ Model Taxonomy in Research Corpus

| Category | Architectures | Core Mechanism | Key References |
| :--- | :--- | :--- | :--- |
| **(a) Pure Data-Driven DL** | CNN, U-Net, ConvLSTM | Convolutions mapping 2D surface patches to depth layers. | [Smith et al., 2023](file:///c:/sih26066/fmars-10-1218514.pdf); [Mao et al., 2023](file:///c:/sih26066/jmse-11-01030.pdf); [Su et al., 2022](file:///c:/sih26066/remotesensing-14-03198-v2.pdf) |
| **(b) Hybrid Attention/Transformers** | **Convformer**, **CSSP-ConvLSTM**, **3DV-Unet** | Depthwise convolutions for local eddies + Swin self-attention for planetary waves. | [Song et al., 2024](file:///c:/sih26066/Convformer_A_Model_for_Reconstructing_Ocean_Subsur%20(1).pdf); [Sun et al., 2026](file:///c:/sih26066/Reconstructing_Subsurface_Ocean_Temperature_From_Sea_Surface_Multivariate_Remote_Sensing_Data_Using_CSSP-ConvLSTM.pdf); [Zhu et al., 2025](file:///c:/sih26066/remotesensing-17-03394.pdf) |
| **(c) Physics-Guided / PINN** | **OG-PINN**, **3D-MOPGCBANN**, **PGTransNet** | Hydrostatic balance, TEOS-10 density coupling, and Brunt-Väisälä stability ($N^2 \ge 0$). | [Xiao et al., 2026](file:///c:/sih26066/v1_covered_7a88a1c7-7d00-4ffa-82a2-2ab9ebc366b4.pdf); [Shao et al., 2025](file:///c:/sih26066/Optimized_Attention-Enhanced_Physics-Guided_Neural_Network_for_Satellite-Based_Ocean_Subsurface_Temperature_Predicting.pdf); [Wu et al., 2024](file:///c:/sih26066/fmars-11-1477710.pdf) |
| **(d) Climatology Ensembles** | **TS-Cast**, **LightGBM**, **Random Forest** | Anomaly prediction over WOA/INCOIS climatological background. | [Chae et al., 2026](file:///c:/sih26066/os-22-2161-2026.pdf); [Meng et al., 2021](file:///c:/sih26066/JGR%20Oceans%20-%202021%20-%20Meng%20-%20Reconstruction%20of%20Three%E2%80%90Dimensional%20Temperature%20and%20Salinity%20Fields%20From%20Satellite%20Observations.pdf) |

---

## 🛠️ Google Developer & Cloud Integration

* **Data Ingestion:** **Google Earth Engine (GEE) Python API** for automated cloud-masked compositing of MODIS/VIIRS SST, SMAP SSS, and ERA5 winds.
* **Model Training & Serving:** **Google Cloud Vertex AI** with `torch-model-archiver` and low-latency gRPC prediction endpoints ($<45\text{ ms}$ inference).
* **Interactive Dashboard:** Containerized FastAPI on **Google Cloud Run** with MapLibre GL for dynamic 3D vertical slicing and maritime heat content visualization.

---

## 📚 Key Research Papers in this Repository
All PDF papers are located in the repository root for offline reference, covering mathematical loss derivations, empirical Argo/RAMA/KEO validations, and remote sensing methodologies.