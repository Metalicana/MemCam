<div align="center">
  <h1>🧠MemCam🎥</h1>
  <h3>Memory-Augmented Camera Control for Consistent Video Generation</h3>
  <p>🎉<strong>IJCNN 2026</strong>🎉</p>
  <br>
  <p>
    <a href="https://newhorizon2005.github.io/MemCam/">
      <img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github" alt="Project Page">
    </a>
    <img src="https://img.shields.io/badge/license-Apache--2.0-green?style=for-the-badge" alt="License">
    <a href="#">
      <img src="https://img.shields.io/badge/🤗-HuggingFace-yellow?style=for-the-badge" alt="HuggingFace">
    </a>
  </p>
  <br>
</div>

## 📄 Paper
> **MemCam: Memory-Augmented Camera Control for Consistent Video Generation**  
> *Xinhang Gao, et al.*  
> *International Joint Conference on Neural Networks (IJCNN), 2026*

---

<p align="center">
   <img src="assets/teaser.png" width="100%" alt="MemCam Teaser">
</p>

---

## 🚀 Overview

Existing interactive video generation methods struggle to maintain scene consistency under large camera rotations over long time horizons — they either rely on fixed-length context windows that cannot cover distant viewpoints (e.g., DFoT), or introduce 3D reconstruction that inevitably accumulates errors (e.g., GeometryForcing).

**MemCam** addresses this by treating previously generated frames as **dynamically retrievable external memory**, enabling long-range scene consistency without 3D reconstruction. The framework is built on the Wan2.1 1.3B DiT and introduces two key designs:

- A **Context Compression Module** that encodes historical frames into compact representations via spatial 2× downsampling, reducing token count to 1/4 and achieving ~5× inference speedup with minimal quality loss.
- A **Co-Visibility-Based Context Retrieval** strategy that uses Monte Carlo FOV overlap estimation to dynamically select the most viewpoint-relevant historical frames for each predicted frame, rather than simply using the most recent ones.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="MemCam Overview">
</p>

---

## ✨ Key Features

- 🧠 **External Memory Mechanism** – Maintains all historical frames as retrievable memory, enabling faithful scene reconstruction even after 360° camera rotations.
- 🎯 **Co-Visibility Retrieval** – Dynamically selects context frames based on camera FOV overlap, ensuring each predicted frame is conditioned on the most relevant history.
- ⚡ **Efficient Context Compression** – Compresses historical frame tokens to 1/4 via spatial downsampling, achieving ~5× speedup over uncompressed baselines at comparable quality.
- 📊 **Strong Results** – ~80% FVD reduction over the strongest baseline on 360° round-trip benchmarks; significant zero-shot gains on RealEstate10K.

---

## 🛠️ Installation

The code will be made publicly available soon. Installation instructions will be provided upon release.
```bash
# Coming soon
git clone https://github.com/newhorizon2005/MemCam.git
cd MemCam
pip install -r requirements.txt
```
