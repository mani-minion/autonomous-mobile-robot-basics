"""
particle_filter.py
──────────────────────────────────────────────────────────────────────────────
Particle Filter (Monte Carlo Localization — MCL) for mobile robot localization.

The University of Bonn is the birthplace of probabilistic robotics. MCL is a
core algorithm that represents the robot's belief about its position as a set
of weighted particles, updated using motion and sensor models.

This simulation:
  - Robot moves in a 2D environment with noisy motion
  - Particles represent possible robot positions
  - Weights updated based on how well particles match sensor readings
  - Resampling focuses particles near the true position
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Parameters ────────────────────────────────────────────────────────────────
NUM_PARTICLES   = 500
MAP_WIDTH       = 100.0   # meters
MAP_HEIGHT      = 100.0
NUM_STEPS       = 30

# Noise parameters
MOTION_NOISE_XY  = 0.5    # std dev in x,y (meters)
MOTION_NOISE_YAW = 0.05   # std dev in yaw (radians)
SENSOR_NOISE     = 1.5    # std dev of sensor measurement noise

# Landmarks (known positions on map — like beacons or map features)
LANDMARKS = np.array([
    [20, 20], [20, 80], [50, 50],
    [80, 20], [80, 80], [35, 60]
])

# ── Particle Filter Class ─────────────────────────────────────────────────────
class ParticleFilter:
    def __init__(self, n_particles):
        self.N = n_particles
        # Initialize particles uniformly across the map
        self.particles = np.zeros((n_particles, 3))
        self.particles[:, 0] = np.random.uniform(0, MAP_WIDTH,  n_particles)  # x
        self.particles[:, 1] = np.random.uniform(0, MAP_HEIGHT, n_particles)  # y
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, n_particles)  # yaw
        self.weights = np.ones(n_particles) / n_particles

    def predict(self, dx, dy, dyaw):
        """Motion model — move particles with noise."""
        self.particles[:, 0] += dx   + np.random.normal(0, MOTION_NOISE_XY,  self.N)
        self.particles[:, 1] += dy   + np.random.normal(0, MOTION_NOISE_XY,  self.N)
        self.particles[:, 2] += dyaw + np.random.normal(0, MOTION_NOISE_YAW, self.N)

        # Keep within map bounds
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, MAP_WIDTH)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, MAP_HEIGHT)

    def update(self, observed_distances):
        """
        Sensor model — weight particles by how well they explain observations.
        observed_distances: measured distances from robot to each landmark (with noise).
        """
        self.weights = np.ones(self.N)

        for i, landmark in enumerate(LANDMARKS):
            # Expected distance from each particle to this landmark
            dx = self.particles[:, 0] - landmark[0]
            dy = self.particles[:, 1] - landmark[1]
            expected_dist = np.sqrt(dx**2 + dy**2)

            # Gaussian likelihood: how close is expected to observed?
            diff = observed_distances[i] - expected_dist
            self.weights *= np.exp(-0.5 * (diff / SENSOR_NOISE)**2)

        self.weights += 1e-300   # avoid zeros
        self.weights /= self.weights.sum()

    def resample(self):
        """Systematic resampling — draw N particles proportional to weights."""
        indices = np.random.choice(self.N, self.N, replace=True, p=self.weights)
        self.particles = self.particles[indices]
        self.weights   = np.ones(self.N) / self.N

    def estimate(self):
        """Weighted mean position estimate."""
        x_est = np.average(self.particles[:, 0], weights=self.weights)
        y_est = np.average(self.particles[:, 1], weights=self.weights)
        return x_est, y_est

# ── Simulate Robot & Run Filter ───────────────────────────────────────────────
def simulate():
    pf = ParticleFilter(NUM_PARTICLES)

    # True robot starting position
    true_x, true_y, true_yaw = 10.0, 10.0, 0.0

    errors = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    snapshots = [0, NUM_STEPS//2, NUM_STEPS-1]
    snap_data = {}

    for step in range(NUM_STEPS):
        # True motion (straight line with slight turn)
        dx    = 2.0 + np.random.normal(0, 0.1)
        dy    = 1.5 + np.random.normal(0, 0.1)
        dyaw  = 0.02

        true_x   += dx
        true_y   += dy
        true_yaw += dyaw

        # Simulate sensor: measure distances to all landmarks (with noise)
        observed = []
        for lm in LANDMARKS:
            true_dist = np.sqrt((true_x - lm[0])**2 + (true_y - lm[1])**2)
            observed.append(true_dist + np.random.normal(0, SENSOR_NOISE))

        # Particle Filter steps
        pf.predict(dx, dy, dyaw)
        pf.update(observed)
        pf.resample()

        est_x, est_y = pf.estimate()
        error = np.sqrt((true_x - est_x)**2 + (true_y - est_y)**2)
        errors.append(error)

        if step in snapshots:
            snap_data[step] = {
                "particles": pf.particles.copy(),
                "true": (true_x, true_y),
                "est":  (est_x, est_y),
                "error": error,
                "step": step
            }

    # Plot snapshots
    for idx, (step, ax) in enumerate(zip(snapshots, axes)):
        d = snap_data[step]
        ax.set_xlim(0, MAP_WIDTH)
        ax.set_ylim(0, MAP_HEIGHT)
        ax.set_facecolor("#f8f8f8")

        ax.scatter(d["particles"][:, 0], d["particles"][:, 1],
                   s=3, c="cornflowerblue", alpha=0.4, label="Particles")
        ax.scatter(*zip(*LANDMARKS), s=80, c="black", marker="^", zorder=5, label="Landmarks")
        ax.plot(*d["true"], "g*", markersize=14, zorder=6, label=f"True pos")
        ax.plot(*d["est"],  "r+", markersize=12, mew=2.5, zorder=6, label=f"Estimate")

        ax.set_title(f"Step {step+1}  |  Error: {d['error']:.2f}m", fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Particle Filter — Monte Carlo Localization (MCL)\n"
                 "Probabilistic Robotics | University of Bonn Research Legacy",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("particle_filter_mcl.png", dpi=150)
    plt.show()

    print(f"\nParticle Filter Results:")
    print(f"  Particles     : {NUM_PARTICLES}")
    print(f"  Steps         : {NUM_STEPS}")
    print(f"  Initial error : {errors[0]:.2f} m")
    print(f"  Final error   : {errors[-1]:.2f} m")
    print(f"  Mean error    : {np.mean(errors):.2f} m")
    print("  Saved: particle_filter_mcl.png")

if __name__ == "__main__":
    print("Monte Carlo Localization (Particle Filter)")
    print("=" * 45)
    simulate()
