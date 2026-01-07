# fql_self_organizing.py
# ------------------------------------------------------------
# Self-Organizing Neuro-Fuzzy Q-Network for Trajectory Tracking
# Based on: Hostetter et al. (AAMAS 2023) FCQL
# - CLIP: Categorical Learning-Induced Partitioning
# - ECM: Evolving Clustering Method  
# - Wang-Mendel: Fuzzy Rule Generation
# - FCQL: Fuzzy Conservative Q-Learning
# ------------------------------------------------------------

from __future__ import annotations
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ----------------------------- 
# Utils (same as before)
# ----------------------------- 
def angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ----------------------------- 
# Gaussian Membership Function (for CLIP)
# ----------------------------- 
@dataclass
class GaussianMF:
    """Gaussian membership function: μ(x) = exp(-((x-c)^2)/(2*σ^2))"""
    center: float
    width: float  # sigma
    
    def membership(self, x: float) -> float:
        """Calculate membership degree."""
        return math.exp(-((x - self.center) ** 2) / (2.0 * self.width ** 2 + 1e-12))

# ----------------------------- 
# CLIP: Categorical Learning-Induced Partitioning
# ----------------------------- 
class CLIP:
    """
    CLIP algorithm from FCQL paper (Section 2.2).
    Automatically discovers linguistic terms (fuzzy sets) for each dimension.
    """
    def __init__(
        self, 
        n_dims: int,
        epsilon: float = 0.2,  # minimum membership threshold
        kappa: float = 0.6,    # contrasting threshold
    ):
        self.n_dims = n_dims
        self.epsilon = epsilon
        self.kappa = kappa
        
        # List of membership functions for each dimension
        self.mfs: List[List[GaussianMF]] = [[] for _ in range(n_dims)]
        
        # Track min/max for regulator function
        self.bounds: List[Tuple[float, float]] = [(1e9, -1e9) for _ in range(n_dims)]
    
    def _regulator(self, upper: float, lower: float) -> float:
        """Regulator function Φ from paper."""
        return 0.5 * (upper + lower)
    
    def _create_initial_mf(self, dim: int, value: float) -> GaussianMF:
        """Create first membership function for a dimension."""
        min_val, max_val = self.bounds[dim]
        
        # Calculate width using epsilon and bounds
        width_upper = math.sqrt(-((max_val - value) ** 2) / math.log(self.epsilon + 1e-12))
        width_lower = math.sqrt(-((min_val - value) ** 2) / math.log(self.epsilon + 1e-12))
        width = self._regulator(width_upper, width_lower)
        
        return GaussianMF(center=value, width=width)
    
    def fit(self, states: np.ndarray):
        """
        Fit CLIP to discover membership functions.
        states: (N, n_dims) array of state observations
        """
        # Update bounds
        for d in range(self.n_dims):
            self.bounds[d] = (float(np.min(states[:, d])), float(np.max(states[:, d])))
        
        # Process each state
        for state in states:
            self._process_state(state)
        
        print(f"CLIP discovered membership functions:")
        for d in range(self.n_dims):
            print(f"  Dimension {d}: {len(self.mfs[d])} linguistic terms")
    
    def _process_state(self, state: np.ndarray):
        """Process a single state to update membership functions."""
        for dim, value in enumerate(state):
            # If no MF exists, create the first one
            if len(self.mfs[dim]) == 0:
                self.mfs[dim].append(self._create_initial_mf(dim, value))
                continue
            
            # Find best matching MF
            memberships = [mf.membership(value) for mf in self.mfs[dim]]
            best_match_idx = int(np.argmax(memberships))
            best_membership = memberships[best_match_idx]
            
            # If similarity exceeds threshold, no new MF needed
            if best_membership > self.kappa:
                continue
            
            # Create new MF
            new_mf = self._create_new_mf(dim, value, best_match_idx)
            self.mfs[dim].append(new_mf)
            
            # Refine neighboring MFs
            self._refine_neighbors(dim, len(self.mfs[dim]) - 1, value)
    
    def _create_new_mf(self, dim: int, value: float, best_idx: int) -> GaussianMF:
        """Create new membership function (Eq. 1-3 in paper)."""
        best_mf = self.mfs[dim][best_idx]
        
        # Find left and right neighbors
        left_neighbor = self._find_left_neighbor(dim, value)
        right_neighbor = self._find_right_neighbor(dim, value)
        
        # Calculate width based on neighbors
        if left_neighbor is None and right_neighbor is None:
            width_right = math.sqrt(-((best_mf.center - value) ** 2) / math.log(self.epsilon + 1e-12))
            width = self._regulator(width_right, best_mf.width)
        elif left_neighbor is None:
            width_right = math.sqrt(-((right_neighbor.center - value) ** 2) / math.log(self.epsilon + 1e-12))
            width = self._regulator(width_right, right_neighbor.width)
        elif right_neighbor is None:
            width_left = math.sqrt(-((left_neighbor.center - value) ** 2) / math.log(self.epsilon + 1e-12))
            width = self._regulator(width_left, left_neighbor.width)
        else:
            width_left = math.sqrt(-((left_neighbor.center - value) ** 2) / math.log(self.epsilon + 1e-12))
            width_right = math.sqrt(-((right_neighbor.center - value) ** 2) / math.log(self.epsilon + 1e-12))
            width = self._regulator(width_left, width_right)
        
        return GaussianMF(center=value, width=width)
    
    def _find_left_neighbor(self, dim: int, value: float) -> GaussianMF | None:
        """Find left neighbor (Eq. 4 in paper)."""
        left_mfs = [mf for mf in self.mfs[dim] if mf.center < value]
        if not left_mfs:
            return None
        return max(left_mfs, key=lambda mf: mf.center)
    
    def _find_right_neighbor(self, dim: int, value: float) -> GaussianMF | None:
        """Find right neighbor (Eq. 5 in paper)."""
        right_mfs = [mf for mf in self.mfs[dim] if mf.center > value]
        if not right_mfs:
            return None
        return min(right_mfs, key=lambda mf: mf.center)
    
    def _refine_neighbors(self, dim: int, new_idx: int, value: float):
        """Refine neighboring MFs after creating a new one."""
        new_mf = self.mfs[dim][new_idx]
        
        left_neighbor = self._find_left_neighbor(dim, value)
        right_neighbor = self._find_right_neighbor(dim, value)
        
        # Adjust neighbors (simplified version)
        if left_neighbor is not None:
            left_neighbor.width = abs(new_mf.center - left_neighbor.center) / 2.0
        if right_neighbor is not None:
            right_neighbor.width = abs(right_neighbor.center - new_mf.center) / 2.0
    
    def fuzzify(self, state: np.ndarray) -> np.ndarray:
        """
        Fuzzify a state into membership degrees.
        Returns: (n_dims, n_terms_per_dim) ragged array - use list of arrays
        """
        memberships = []
        for dim, value in enumerate(state):
            dim_memberships = np.array([mf.membership(value) for mf in self.mfs[dim]])
            memberships.append(dim_memberships)
        return memberships

# ----------------------------- 
# ECM: Evolving Clustering Method
# ----------------------------- 
@dataclass
class Cluster:
    center: np.ndarray
    radius: float
    support: int

class ECM:
    """
    ECM algorithm from FCQL paper (Section 2.3).
    Identifies regions of interest / exemplars for fuzzy rule generation.
    """
    def __init__(self, dthr: float = 0.4):
        self.dthr = dthr  # distance threshold
        self.clusters: List[Cluster] = []
    
    def fit(self, states: np.ndarray) -> np.ndarray:
        """
        Fit ECM to identify cluster centers.
        Returns: (M, n_dims) array of cluster centers
        """
        for state in states:
            self._process_state(state)
        
        print(f"ECM identified {len(self.clusters)} clusters")
        return np.array([c.center for c in self.clusters])
    
    def _process_state(self, state: np.ndarray):
        """Process a single state (5 steps from paper)."""
        # Step 1: Create first cluster
        if len(self.clusters) == 0:
            self.clusters.append(Cluster(center=state.copy(), radius=0.0, support=1))
            return
        
        # Step 2: Calculate distances to all clusters
        distances = [np.linalg.norm(state - c.center) for c in self.clusters]
        
        # Step 3: Check if belongs to existing cluster
        for i, (dist, cluster) in enumerate(zip(distances, self.clusters)):
            if dist <= cluster.radius:
                # Belongs to cluster with minimum distance
                min_idx = int(np.argmin(distances))
                if i == min_idx:
                    return  # No update needed
        
        # Step 4: Find cluster to potentially update
        criteria = [dist + c.radius for dist, c in zip(distances, self.clusters)]
        best_idx = int(np.argmin(criteria))
        best_cluster = self.clusters[best_idx]
        best_criterion = criteria[best_idx]
        
        # Step 5: Create new cluster or update existing
        if best_criterion > 2.0 * self.dthr:
            # Create new cluster
            self.clusters.append(Cluster(center=state.copy(), radius=0.0, support=1))
        else:
            # Update cluster
            best_cluster.radius = best_criterion / 2.0
            best_cluster.center = (best_cluster.support * best_cluster.center + state) / (best_cluster.support + 1)
            best_cluster.support += 1

# ----------------------------- 
# Wang-Mendel Method: Fuzzy Rule Generation
# ----------------------------- 
class WangMendel:
    """
    Wang-Mendel Method from FCQL paper (Section 2.4).
    Generates fuzzy logic rules from cluster centers.
    """
    def __init__(self, clip: CLIP):
        self.clip = clip
        self.rules: List[Tuple[Tuple[int, ...], int]] = []  # (antecedents, rule_id)
    
    def generate_rules(self, cluster_centers: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Generate fuzzy logic rules from cluster centers.
        Returns: List of rule antecedents (tuples of MF indices)
        """
        unique_rules = set()
        
        for center in cluster_centers:
            # Step 2 from paper (main step we use)
            # For each dimension, find MF with highest membership
            antecedents = []
            for dim, value in enumerate(center):
                memberships = [mf.membership(value) for mf in self.clip.mfs[dim]]
                best_mf_idx = int(np.argmax(memberships))
                antecedents.append(best_mf_idx)
            
            # Add unique rule
            rule_key = tuple(antecedents)
            unique_rules.add(rule_key)
        
        self.rules = list(unique_rules)
        print(f"Wang-Mendel generated {len(self.rules)} unique fuzzy logic rules")
        
        return self.rules

# ----------------------------- 
# Self-Organizing FQL Agent
# ----------------------------- 
class SelfOrganizingFQLAgent:
    """
    Self-organizing Fuzzy Q-Learning agent using CLIP + ECM + Wang-Mendel.
    """
    def __init__(
        self,
        n_state_dims: int,
        n_actions: int,
        alpha: float = 0.08,
        gamma: float = 0.98,
        clip_epsilon: float = 0.2,
        clip_kappa: float = 0.6,
        ecm_dthr: float = 0.4,
    ):
        self.n_state_dims = n_state_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        
        # Components
        self.clip = CLIP(n_state_dims, epsilon=clip_epsilon, kappa=clip_kappa)
        self.ecm = ECM(dthr=ecm_dthr)
        self.wang_mendel = WangMendel(self.clip)
        
        # Q-table (initialized after rule generation)
        self.Q: np.ndarray | None = None
        self.rules: List[Tuple[int, ...]] = []
    
    def build_structure(self, states: np.ndarray):
        """
        Build the neuro-fuzzy network structure from data.
        This is the self-organizing phase.
        """
        print("\n=== Self-Organizing Neuro-Fuzzy Network ===")
        
        # Step 1: Discover membership functions with CLIP
        print("\nStep 1: CLIP - Discovering membership functions...")
        self.clip.fit(states)
        
        # Step 2: Identify cluster centers with ECM
        print("\nStep 2: ECM - Identifying cluster centers...")
        cluster_centers = self.ecm.fit(states)
        
        # Step 3: Generate fuzzy logic rules with Wang-Mendel
        print("\nStep 3: Wang-Mendel - Generating fuzzy logic rules...")
        self.rules = self.wang_mendel.generate_rules(cluster_centers)
        
        # Step 4: Initialize Q-table
        n_rules = len(self.rules)
        self.Q = np.zeros((n_rules, self.n_actions), dtype=float)
        
        print(f"\nStructure built: {n_rules} rules x {self.n_actions} actions")
        print("=" * 45)
    
    def firing_strengths(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate firing strength for each rule (Eq. 9 in paper).
        Uses product-inference engine.
        """
        memberships_per_dim = self.clip.fuzzify(state)
        
        firing = np.zeros(len(self.rules))
        for rule_idx, antecedents in enumerate(self.rules):
            # Product of memberships
            strength = 1.0
            for dim, mf_idx in enumerate(antecedents):
                strength *= memberships_per_dim[dim][mf_idx]
            firing[rule_idx] = strength
        
        # Normalize
        total = np.sum(firing)
        if total > 1e-12:
            firing /= total
        else:
            firing = np.ones(len(self.rules)) / len(self.rules)
        
        return firing
    
    def effective_q(self, w: np.ndarray) -> np.ndarray:
        """Aggregate Q-values across rules using firing weights."""
        return w @ self.Q
    
    def blended_action(
        self, 
        w: np.ndarray, 
        omega_actions: np.ndarray,
        tau: float = 0.2
    ) -> Tuple[float, int]:
        """Softmax action blending."""
        q = self.effective_q(w)
        
        # Numerical stability
        q_max = np.max(q)
        logits = (q - q_max) / tau
        probs = np.exp(logits)
        probs /= np.sum(probs)
        
        # Blended continuous control
        omega = float(np.sum(probs * omega_actions))
        
        # Representative discrete action
        a_idx = int(np.argmax(q))
        
        return omega, a_idx
    
    def update(
        self, 
        w: np.ndarray, 
        a: int, 
        r: float, 
        w_next: np.ndarray, 
        done: bool,
        use_cql: bool = True,
        alpha_cql: float = 0.5
    ):
        """
        Update Q-values using Fuzzy Q-Learning or Fuzzy Conservative Q-Learning.
        
        Args:
            use_cql: If True, use CQL augmentation (Eq. in Section 2.5)
            alpha_cql: CQL trade-off factor
        """
        q_eff = self.effective_q(w)
        q_curr = float(q_eff[a])
        
        # Standard Bellman update
        if done:
            target = float(r)
        else:
            q_eff_next = self.effective_q(w_next)
            target = float(r + self.gamma * np.max(q_eff_next))
        
        td = target - q_curr
        
        # FQL update: Q_i(a) += alpha * w_i * TD
        self.Q[:, a] += self.alpha * w * td
        
        # Optional: CQL augmentation (conservative update)
        if use_cql:
            # Minimize: log sum exp Q(s,a) - Q(s, a_data)
            # This prevents overestimation in offline setting
            all_q = self.effective_q(w)
            logsumexp = np.log(np.sum(np.exp(all_q)) + 1e-12)
            cql_penalty = alpha_cql * (logsumexp - q_curr)
            
            # Apply penalty to all actions
            self.Q[:, :] -= self.alpha * w[:, None] * cql_penalty

# ----------------------------- 
# Reference trajectories (same as before)
# ----------------------------- 
def ref_circle(t: float, R: float = 2.0, T: float = 40.0) -> Tuple[float, float]:
    w = 2.0 * math.pi / T
    return (R * math.cos(w * t), R * math.sin(w * t))

def ref_velocity(t: float, ref_fn, dt: float = 1e-3) -> float:
    x1, y1 = ref_fn(t)
    x2, y2 = ref_fn(t + dt)
    return math.hypot(x2 - x1, y2 - y1) / dt

# ----------------------------- 
# Path-relative functions (same as before)
# ----------------------------- 
def closest_point(x: float, y: float, traj: np.ndarray) -> int:
    d = np.sum((traj - np.array([x, y])) ** 2, axis=1)
    return int(np.argmin(d))

def tracking_errors_ed_eh(
    x: float, y: float, theta: float, xg: float, yg: float
) -> Tuple[float, float, float, float]:
    dx = xg - x
    dy = yg - y
    c = math.cos(theta)
    s = math.sin(theta)
    xR = c * dx + s * dy
    yR = -s * dx + c * dy
    ed = math.sqrt(xR * xR + yR * yR)
    eh = math.atan2(yR, xR)
    eh = angle_wrap(eh)
    return ed, eh, xR, yR

# ----------------------------- 
# Main demonstration
# ----------------------------- 
def main():
    np.random.seed(0)
    
    print("=" * 60)
    print("Self-Organizing Neuro-Fuzzy Q-Network")
    print("Based on: Hostetter et al. (AAMAS 2023)")
    print("=" * 60)
    
    # ---- Trajectory ----
    T_circle = 60.0
    R_circle = 6.0
    ref_fn = lambda tt: ref_circle(tt, R=R_circle, T=T_circle)
    
    # ---- Simulation params ----
    dt = 0.05
    T_total = 2.0 * T_circle
    steps = int(T_total / dt)
    
    # Reference trajectory
    N_REF = 2000
    ref_traj = np.array([ref_fn(T_circle * i / N_REF) for i in range(N_REF)])
    
    # ---- Collect training data for structure learning ----
    print("\n" + "=" * 60)
    print("Phase 1: Collecting training data for self-organization")
    print("=" * 60)
    
    n_data_episodes = 5
    collected_states = []
    
    for ep in range(n_data_episodes):
        x0, y0 = ref_fn(0.0)
        x = x0 + np.random.uniform(-1.0, 1.0)
        y = y0 + np.random.uniform(-1.0, 1.0)
        theta = np.random.uniform(-math.pi, math.pi)
        
        for k in range(steps):
            idx = closest_point(x, y, ref_traj)
            xg, yg = ref_traj[idx]
            ed, eh, _, _ = tracking_errors_ed_eh(x, y, theta, xg, yg)
            
            # Store state (ed, eh) for structure learning
            collected_states.append([ed, eh])
            
            # Random exploration for data collection
            omega = np.random.uniform(-1.0, 1.0)
            v = 0.3
            
            theta = angle_wrap(theta + omega * dt)
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
    
    collected_states = np.array(collected_states)
    print(f"Collected {len(collected_states)} state samples")
    
    # ---- Build self-organizing structure ----
    print("\n" + "=" * 60)
    print("Phase 2: Building self-organizing neuro-fuzzy network")
    print("=" * 60)
    
    # Action set
    omega_max = 1.2
    n_actions = 11
    omega_actions = np.linspace(-omega_max, omega_max, n_actions)
    
    # Create agent and build structure
    agent = SelfOrganizingFQLAgent(
        n_state_dims=2,  # (ed, eh)
        n_actions=n_actions,
        alpha=0.10,
        gamma=0.98,
        clip_epsilon=0.2,
        clip_kappa=0.6,
        ecm_dthr=0.3,  # Tune this to control number of rules
    )
    
    agent.build_structure(collected_states)
    
    # ---- Training ----
    print("\n" + "=" * 60)
    print("Phase 3: Training with Fuzzy Conservative Q-Learning")
    print("=" * 60)
    
    episodes = 100
    reward_hist = []
    
    for ep in range(1, episodes + 1):
        x0, y0 = ref_fn(0.0)
        x = x0 + np.random.uniform(-0.5, 0.5)
        y = y0 + np.random.uniform(-0.5, 0.5)
        theta = np.random.uniform(-math.pi, math.pi)
        
        ep_return = 0.0
        
        # Initial state
        idx = closest_point(x, y, ref_traj)
        xg, yg = ref_traj[idx]
        ed, eh, _, _ = tracking_errors_ed_eh(x, y, theta, xg, yg)
        state = np.array([ed, eh])
        w = agent.firing_strengths(state)
        
        tau = max(0.08, 0.4 * (0.998 ** ep))
        prev_omega = 0.0
        DELTA_MAX = 0.15
        
        for k in range(steps):
            # Action selection
            omega_raw, a = agent.blended_action(w, omega_actions, tau=tau)
            omega = clamp(omega_raw, prev_omega - DELTA_MAX, prev_omega + DELTA_MAX)
            prev_omega = omega
            
            # Velocity
            v_ref = ref_velocity(k * dt, ref_fn)
            v = max(0.25, v_ref * math.exp(-abs(eh)))
            
            # Step
            theta = angle_wrap(theta + omega * dt)
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            
            # Next state
            idx = closest_point(x, y, ref_traj)
            xg, yg = ref_traj[idx]
            ed2, eh2, _, yR2 = tracking_errors_ed_eh(x, y, theta, xg, yg)
            state_next = np.array([ed2, eh2])
            w_next = agent.firing_strengths(state_next)
            
            # Reward
            r = (ed - ed2) - 0.15 * abs(eh2) - 0.4 * (yR2 ** 2)
            
            # Update
            done = (abs(ed2) > 7.0) or (abs(yR2) > 3.0)
            agent.update(w, a, r, w_next, done, use_cql=True, alpha_cql=0.5)
            
            ep_return += r
            w = w_next
            ed = ed2
            eh = eh2
            
            if done:
                break
        
        reward_hist.append(ep_return)
        
        if ep % 25 == 0:
            print(f"Episode {ep:3d}/{episodes} | Return: {ep_return:9.2f} | Rules: {len(agent.rules)}")
    
    # ---- Final evaluation ----
    print("\n" + "=" * 60)
    print("Phase 4: Final evaluation")
    print("=" * 60)
    
    x0, y0 = ref_fn(0.0)
    x = x0 + 1.0
    y = y0 + 1.0
    theta = 0.0
    
    traj = np.zeros((steps, 2))
    ed_log = np.zeros(steps)
    eh_log = np.zeros(steps)
    omega_log = np.zeros(steps)
    
    idx = closest_point(x, y, ref_traj)
    xg, yg = ref_traj[idx]
    ed, eh, _, _ = tracking_errors_ed_eh(x, y, theta, xg, yg)
    state = np.array([ed, eh])
    prev_omega = 0.0
    
    for k in range(steps):
        traj[k] = [x, y]
        ed_log[k] = ed
        eh_log[k] = eh
        
        w = agent.firing_strengths(state)
        omega_raw, _ = agent.blended_action(w, omega_actions, tau=0.4)
        omega = clamp(omega_raw, prev_omega - 0.2, prev_omega + 0.2)
        prev_omega = omega
        omega_log[k] = omega
        
        v_ref = ref_velocity(k * dt, ref_fn)
        v = max(0.25, v_ref * math.exp(-abs(eh)))
        
        theta = angle_wrap(theta + omega * dt)
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        
        idx = closest_point(x, y, ref_traj)
        xg, yg = ref_traj[idx]
        ed, eh, _, _ = tracking_errors_ed_eh(x, y, theta, xg, yg)
        state = np.array([ed, eh])
    
    # ---- Visualization ----
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], "--", linewidth=2, label="reference")
    plt.plot(traj[:, 0], traj[:, 1], "-", linewidth=2, label="tracking")
    plt.scatter([traj[0, 0]], [traj[0, 1]], c="k", s=60, label="start")
    plt.axis("equal")
    plt.grid(True)
    plt.title("Self-Organizing FQL Tracking")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(reward_hist, linewidth=1.5)
    plt.grid(True)
    plt.title("Training Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    
    plt.subplot(2, 2, 3)
    plt.plot(ed_log, linewidth=1.5, label="ed")
    plt.plot(eh_log, linewidth=1.5, label="eh")
    plt.grid(True)
    plt.title("Tracking Errors")
    plt.xlabel("Step")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(omega_log, linewidth=1.5)
    plt.grid(True)
    plt.title("Control Command ω")
    plt.xlabel("Step")
    plt.ylabel("rad/s")
    
    plt.tight_layout()
    plt.savefig("/home/claude/self_organizing_fql_result.png", dpi=150)
    print("\nPlot saved to: self_organizing_fql_result.png")
    
    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Number of linguistic terms per dimension:")
    for d in range(agent.n_state_dims):
        print(f"  Dimension {d}: {len(agent.clip.mfs[d])} terms")
    print(f"Number of clusters identified: {len(agent.ecm.clusters)}")
    print(f"Number of fuzzy logic rules: {len(agent.rules)}")
    print(f"Q-table shape: {agent.Q.shape}")
    print(f"Final RMSE ed: {np.sqrt(np.mean(ed_log**2)):.4f}")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    main()
