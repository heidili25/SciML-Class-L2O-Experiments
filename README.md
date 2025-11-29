# SciML Class — L2O Surrogate OPF Experiments
Author: Heidi Li  
Course: EN.560.652 Scientific Machine Learning (Fall 2025)  
Instructor: Prof. Jan Drgoňa  
Institution: Johns Hopkins University

## Overview

This repository contains all code for a SciML class project on Learning-to-Optimize (L2O) surrogates for AC Optimal Power Flow (AC-OPF) using the IEEE 14-bus network.  

The objective is to learn fast neural surrogates that approximate AC-OPF solutions while preserving feasibility and physical consistency.

The project is organized into two model families:

### A-Series (Fully AC-OPF Supervised Surrogates)

- **A1: Direct Mapping Network (Supervised)**  
  A feedforward neural network is trained on A0 labels to map loads → optimal dispatch.  
  Importantly, A1 also **generates and saves the AC-OPF dataset** (`acopf_dataset.pt`), which all models A2, B1, and B2 use for consistent benchmarking.

- **A2: Physics-Informed Mapping**  
  A2 begins from the A1 architecture and introduces Neuromancer penalty layers (generator limits, voltage limits, power balance) to improve physical feasibility.  
  A2 optionally projects predictions through a Pandapower PF layer for evaluation.

### B-Series (DC Hot-Start + Residual Refinement)
- **B1: DC Hot-Start Model**  
  A learned DC approximation producing a physically reasonable initial guess for AC-OPF.

- **B2: L2O Refinement Model**  
  A small residual neural layer refines the B1 initialization using an L2O-style training objective and a feasibility layer.

The goal is fast surrogate OPF with reduced constraint violations.


## Performance Metrics

Models A1–B2 are compared along several OPF-style metrics:

- **Constraint Violations**  
  - Generator active power limits  
  - Bus voltage magnitude limits  
  - Line loading percentage limits  

- **Solution Error vs AC-OPF**  
  - L2 norm of generator dispatch error  
  - L2 norm of voltage magnitude error  
  - L2 norm of voltage angle error  

- **Optimality Gap**  
 `J_model - J_ACOPF` evaluated using linear generator costs.

- **Runtime**  
  Wall-clock forward-pass time (no PF/OPF inside timing).

`figures.py` produces a four-panel summary figure visualizing these metrics.

---

## Repository Structure

