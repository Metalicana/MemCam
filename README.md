<div align="center">
  <h1>🧠MemCam🎥</h1>
  <h3>Memory-Augmented Camera Control for Consistent Video Generation</h3>
  <p>🎉<strong>IJCNN 2026</strong>🎉</p>
  <br>
  <p>
    <img src="https://img.shields.io/badge/status-coming%20soon-blue?style=for-the-badge&logo=github" alt="Status">
    <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=for-the-badge" alt="License">
    <img src="https://img.shields.io/badge/conference-IJCNN%202026-red?swtyle=for-the-badge" alt="Conference">
  </p>
  <br>
  <p>
    <i> Code Coming Soon... </i>
  </p>
</div>

---

## 🚀 Overview

Existing interactive video generation methods struggle to maintain scene consistency under large camera rotations over long time horizons — they either rely on fixed-length context windows that cannot cover distant viewpoints (e.g., DFoT), or introduce 3D reconstruction that inevitably accumulates errors (e.g., GeometryForcing).

**MemCam** addresses this by treating previously generated frames as **dynamically retrievable external memory**, enabling long-range scene consistency without 3D reconstruction. The framework is built on the Wan2.1 1.3B DiT and introduces two key designs:

- A **Context Compression Module** that encodes historical frames into compact representations via spatial 2× downsampling, reducing token count to 1/4 and achieving ~5× inference speedup with minimal quality loss.
- A **Co-Visibility-Based Context Retrieval** strategy that uses Monte Carlo FOV overlap estimation to dynamically select the most viewpoint-relevant historical frames for each predicted frame, rather than simply using the most recent ones.

<p align="center">
  <img src="assets/overview.jpg" width="90%" alt="MemCam Overview">
</p>

---

## ✨ Key Features

- 🧠 **External Memory Mechanism** – Maintains all historical frames as retrievable memory, enabling faithful scene reconstruction even after 360° camera rotations.
- 🎯 **Co-Visibility Retrieval** – Dynamically selects context frames based on camera FOV overlap, ensuring each predicted frame is conditioned on the most relevant history.
- ⚡ **Efficient Context Compression** – Compresses historical frame tokens to 1/4 via spatial downsampling, achieving ~5× speedup over uncompressed baselines at comparable quality.
- 📊 **Strong Results** – ~80% FVD reduction over the strongest baseline on 360° round-trip benchmarks; significant zero-shot gains on RealEstate10K.

---

## 📄 Paper

> **MemCam: Memory-Augmented Camera Control for Consistent Video Generation**  
> *Xinhang Gao, et al.*  
> *International Joint Conference on Neural Networks (IJCNN), 2026*

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org/) *(link will be updated upon release)*  
[![IJCNN](https://img.shields.io/badge/IJCNN-2026-0077b5)](https://www.ijcnn.org/)

---

## 🧪 Demo & Examples

> 🎬 *Demo videos and visual results will be added after the code release.*

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=Demo+Coming+Soon" width="80%">
</p>

---

## 🛠️ Installation

The code will be made publicly available soon. Installation instructions will be provided upon release.

```bash
# Coming soon
git clone https://github.com/newhorizon2005/MemCam.git
cd MemCam
pip install -r requirements.txt
```
