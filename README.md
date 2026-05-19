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
| `obstacle_avoidance.py` | Bug Algorithm | Sensor-based reactive obstacle avoidance |
| `pid_controller.py` | PID Control | Line following / heading control |
| `a_star_path_planning.py` | A* Search | Grid-based optimal path planning |
| `particle_filter.py` | Monte Carlo Localization | Probabilistic robot localization |
| `kalman_filter.py` | Extended Kalman Filter | Sensor fusion (odometry + IMU) |

---

## Algorithms Explained

### 1. Obstacle Avoidance (Bug Algorithm)
Reactive navigation — robot moves toward goal, follows obstacle boundary when blocked, resumes toward goal when clear.

### 2. PID Controller
Used for line following, heading control, and speed regulation. Proportional-Integral-Derivative control keeps error near zero.

### 3. A* Path Planning
Finds the shortest collision-free path on a grid map from start to goal using a heuristic search (Manhattan/Euclidean distance).

### 4. Particle Filter (MCL)
Monte Carlo Localization — represents robot position as a set of weighted particles. Updates weights using sensor observations. Core of probabilistic robotics (Bonn's research legacy).

### 5. Kalman Filter
Sensor fusion combining noisy odometry and IMU data to produce a better state estimate of robot position and velocity.

---

## Tools & Requirements

```bash
pip install numpy matplotlib scipy
```
