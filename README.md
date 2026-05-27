# Autonomous Mobile Robot — Fundamentals & Algorithms

> Based on coursework in Autonomous Mobile Robots  
> B.E. Robotics and Automation — Sri Ramakrishna Engineering College | Anna University

---

## Overview

Core algorithms and implementations for **Autonomous Mobile Robot (AMR)** navigation — covering sensor-based obstacle avoidance, path planning, probabilistic localization, and basic SLAM concepts. These are the foundational techniques used in mobile robotics research and directly applicable to real-world AMR systems.

---

## Contents

| File | Algorithm | Description |
|------|-----------|-------------|
| `a_star_path_planning.py` | A* Search | Grid-based optimal path planning |
| `particle_filter.py` | Monte Carlo Localization | Probabilistic robot localization |

---

## Algorithms Explained

### 1. A* Path Planning
Finds the shortest collision-free path on a grid map from start to goal using a heuristic search (Manhattan/Euclidean distance).

### 2. Particle Filter (MCL)
Monte Carlo Localization — represents robot position as a set of weighted particles. Updates weights using sensor observations. Core of probabilistic robotics.

---

## Tools & Requirements

```bash
pip install numpy matplotlib scipy
```
