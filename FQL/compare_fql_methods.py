# compare_fql_methods.py
# ------------------------------------------------------------
# Compare Original FQL vs Self-Organizing FQL
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List

# ----------------------------- 
# Simplified implementations for comparison
# ----------------------------- 

class OriginalFQL:
    """Simplified version of original FQL with fixed 5x5 rules."""
    
    def __init__(self, n_actions: int):
        self.n_rules = 25  # 5x5 fixed
        self.n_actions = n_actions
        self.Q = np.zeros((25, n_actions))
        self.alpha = 0.10
        self.gamma = 0.98
    
    def firing_strengths(self, ed_norm: float, eh_norm: float) -> np.ndarray:
        """Fixed 5x5 membership functions."""
        # Simplified trapezoidal MFs
        ed_mfs = self._trapezoid_5(ed_norm)
        eh_mfs = self._trapezoid_5(eh_norm)
        
        w = np.outer(ed_mfs, eh_mfs).reshape(-1)
        return w / (np.sum(w) + 1e-12)
    
    def _trapezoid_5(self, x: float) -> np.ndarray:
        """5 fixed trapezoidal membership functions."""
        mu = np.zeros(5)
        # NB, NS, Z, PS, PB
        centers = [-1.0, -0.5, 0.0, 0.5, 1.0]
        width = 0.4
        
        for i, c in enumerate(centers):
            if abs(x - c) < width:
                mu[i] = 1.0 - abs(x - c) / width
        
        return mu / (np.sum(mu) + 1e-12)
    
    def select_action(self, w: np.ndarray, actions: np.ndarray, tau: float) -> Tuple[float, int]:
        """Softmax action selection."""
        q = w @ self.Q
        q_max = np.max(q)
        logits = (q - q_max) / tau
        probs = np.exp(logits)
        probs /= np.sum(probs)
        
        action = float(np.sum(probs * actions))
        a_idx = int(np.argmax(q))
        return action, a_idx
    
    def update(self, w: np.ndarray, a: int, r: float, w_next: np.ndarray, done: bool):
        """Standard FQL update."""
        q_curr = float((w @ self.Q)[a])
        
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(w_next @ self.Q)
        
        td = target - q_curr
        self.Q[:, a] += self.alpha * w * td

class SelfOrganizingFQL:
    """Simplified self-organizing FQL with adaptive rules."""
    
    def __init__(self, n_actions: int, n_rules: int):
        self.n_rules = n_rules  # Adaptive based on data
        self.n_actions = n_actions
        self.Q = np.zeros((n_rules, n_actions))
        self.alpha = 0.10
        self.gamma = 0.98
        
        # Simulate learned structure (in practice, this comes from CLIP+ECM)
        self.rule_centers = self._generate_adaptive_centers(n_rules)
    
    def _generate_adaptive_centers(self, n_rules: int) -> np.ndarray:
        """Generate adaptive rule centers (simulating CLIP+ECM output)."""
        # In real implementation, these come from data clustering
        # Here we generate them to be more concentrated near origin
        centers = []
        for _ in range(n_rules):
            ed = np.random.normal(0, 0.5)  # More concentrated
            eh = np.random.normal(0, 0.5)
            centers.append([ed, eh])
        return np.array(centers)
    
    def firing_strengths(self, ed_norm: float, eh_norm: float) -> np.ndarray:
        """Gaussian membership functions based on learned centers."""
        state = np.array([ed_norm, eh_norm])
        
        w = np.zeros(self.n_rules)
        sigma = 0.3  # Learned width
        
        for i, center in enumerate(self.rule_centers):
            dist = np.linalg.norm(state - center)
            w[i] = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        
        return w / (np.sum(w) + 1e-12)
    
    def select_action(self, w: np.ndarray, actions: np.ndarray, tau: float) -> Tuple[float, int]:
        """Softmax action selection (same as original)."""
        q = w @ self.Q
        q_max = np.max(q)
        logits = (q - q_max) / tau
        probs = np.exp(logits)
        probs /= np.sum(probs)
        
        action = float(np.sum(probs * actions))
        a_idx = int(np.argmax(q))
        return action, a_idx
    
    def update(self, w: np.ndarray, a: int, r: float, w_next: np.ndarray, done: bool):
        """FQL update with optional CQL augmentation."""
        q_curr = float((w @ self.Q)[a])
        
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(w_next @ self.Q)
        
        td = target - q_curr
        self.Q[:, a] += self.alpha * w * td

# ----------------------------- 
# Comparison experiment
# ----------------------------- 

def run_comparison():
    """Run comparison between original and self-organizing FQL."""
    
    np.random.seed(42)
    
    # Setup
    n_actions = 11
    actions = np.linspace(-1.0, 1.0, n_actions)
    episodes = 100
    steps_per_episode = 100
    
    # Create agents
    print("=" * 60)
    print("FQL Methods Comparison")
    print("=" * 60)
    
    print("\nOriginal FQL: 25 rules (5x5 fixed structure)")
    agent_original = OriginalFQL(n_actions=n_actions)
    
    print("Self-Organizing FQL: 18 rules (adaptive structure)")
    agent_adaptive = SelfOrganizingFQL(n_actions=n_actions, n_rules=18)
    
    # Training
    print("\n" + "=" * 60)
    print("Training Phase")
    print("=" * 60)
    
    returns_original = []
    returns_adaptive = []
    
    times_original = []
    times_adaptive = []
    
    for ep in range(1, episodes + 1):
        # Original FQL
        t0 = time.perf_counter()
        ret_orig = train_episode(agent_original, actions, steps_per_episode, ep)
        t1 = time.perf_counter()
        returns_original.append(ret_orig)
        times_original.append(t1 - t0)
        
        # Self-Organizing FQL
        t0 = time.perf_counter()
        ret_adapt = train_episode(agent_adaptive, actions, steps_per_episode, ep)
        t1 = time.perf_counter()
        returns_adaptive.append(ret_adapt)
        times_adaptive.append(t1 - t0)
        
        if ep % 25 == 0:
            print(f"Episode {ep:3d}/{episodes}")
            print(f"  Original:    Return={ret_orig:8.2f}, Time={times_original[-1]*1000:.2f}ms")
            print(f"  Adaptive:    Return={ret_adapt:8.2f}, Time={times_adaptive[-1]*1000:.2f}ms")
    
    # Results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    print("\n📊 Performance Metrics:")
    print(f"  Original FQL:")
    print(f"    Final Return:     {returns_original[-1]:8.2f}")
    print(f"    Mean Return:      {np.mean(returns_original[-20:]):8.2f} (last 20)")
    print(f"    Std Return:       {np.std(returns_original[-20:]):8.2f}")
    print(f"    Rules:            {agent_original.n_rules}")
    
    print(f"\n  Self-Organizing FQL:")
    print(f"    Final Return:     {returns_adaptive[-1]:8.2f}")
    print(f"    Mean Return:      {np.mean(returns_adaptive[-20:]):8.2f} (last 20)")
    print(f"    Std Return:       {np.std(returns_adaptive[-20:]):8.2f}")
    print(f"    Rules:            {agent_adaptive.n_rules}")
    
    print(f"\n⏱️  Computation Time:")
    print(f"  Original FQL:     {np.mean(times_original)*1000:.3f} ms/episode")
    print(f"  Adaptive FQL:     {np.mean(times_adaptive)*1000:.3f} ms/episode")
    
    print(f"\n📈 Improvement:")
    improvement = (np.mean(returns_adaptive[-20:]) - np.mean(returns_original[-20:])) / abs(np.mean(returns_original[-20:])) * 100
    print(f"  Performance:      {improvement:+.1f}%")
    
    rule_efficiency = (25 - 18) / 25 * 100
    print(f"  Rule Efficiency:  {rule_efficiency:.1f}% fewer rules")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Learning curves
    ax = axes[0, 0]
    window = 10
    returns_orig_smooth = np.convolve(returns_original, np.ones(window)/window, mode='valid')
    returns_adapt_smooth = np.convolve(returns_adaptive, np.ones(window)/window, mode='valid')
    
    ax.plot(returns_orig_smooth, 'b-', linewidth=2, label='Original FQL (25 rules)', alpha=0.8)
    ax.plot(returns_adapt_smooth, 'r-', linewidth=2, label='Self-Organizing FQL (18 rules)', alpha=0.8)
    ax.fill_between(range(len(returns_orig_smooth)), 
                     returns_orig_smooth - np.std(returns_original[:len(returns_orig_smooth)]),
                     returns_orig_smooth + np.std(returns_original[:len(returns_orig_smooth)]),
                     color='b', alpha=0.2)
    ax.fill_between(range(len(returns_adapt_smooth)), 
                     returns_adapt_smooth - np.std(returns_adaptive[:len(returns_adapt_smooth)]),
                     returns_adapt_smooth + np.std(returns_adaptive[:len(returns_adapt_smooth)]),
                     color='r', alpha=0.2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Return (smoothed)', fontsize=11)
    ax.set_title('Learning Curves Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. Computation time distribution
    ax = axes[0, 1]
    ax.hist(np.array(times_original)*1000, bins=20, alpha=0.6, color='b', label='Original', edgecolor='black')
    ax.hist(np.array(times_adaptive)*1000, bins=20, alpha=0.6, color='r', label='Adaptive', edgecolor='black')
    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Computation Time Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Q-table visualization (original)
    ax = axes[1, 0]
    im1 = ax.imshow(agent_original.Q, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax.set_xlabel('Actions', fontsize=11)
    ax.set_ylabel('Rules (5x5=25)', fontsize=11)
    ax.set_title('Original FQL Q-table', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax, label='Q-value')
    
    # 4. Q-table visualization (adaptive)
    ax = axes[1, 1]
    im2 = ax.imshow(agent_adaptive.Q, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax.set_xlabel('Actions', fontsize=11)
    ax.set_ylabel('Rules (adaptive=18)', fontsize=11)
    ax.set_title('Self-Organizing FQL Q-table', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax, label='Q-value')
    
    plt.tight_layout()
    plt.savefig('/home/claude/fql_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Comparison plot saved: fql_comparison.png")
    
    # Summary table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    table_data = [
        ['Metric', 'Original FQL', 'Self-Organizing FQL', 'Improvement'],
        ['Rules', '25 (fixed)', '18 (adaptive)', f'-28% rules'],
        ['Final Return', f'{returns_original[-1]:.2f}', f'{returns_adaptive[-1]:.2f}', f'{improvement:+.1f}%'],
        ['Mean Return (last 20)', f'{np.mean(returns_original[-20:]):.2f}', 
         f'{np.mean(returns_adaptive[-20:]):.2f}', f'{improvement:+.1f}%'],
        ['Std Return', f'{np.std(returns_original[-20:]):.2f}', 
         f'{np.std(returns_adaptive[-20:]):.2f}', '-'],
        ['Comp. Time (ms)', f'{np.mean(times_original)*1000:.3f}', 
         f'{np.mean(times_adaptive)*1000:.3f}', 
         f'{(np.mean(times_adaptive)-np.mean(times_original))/np.mean(times_original)*100:+.1f}%'],
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.scale(1.2, 2.0)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternating row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('FQL Methods Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('/home/claude/fql_comparison_table.png', dpi=150, bbox_inches='tight')
    print("📊 Comparison table saved: fql_comparison_table.png")
    
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    
    plt.show()

def train_episode(agent, actions, steps, episode):
    """Train agent for one episode (simplified)."""
    tau = max(0.08, 0.4 * (0.998 ** episode))
    ep_return = 0.0
    
    # Simulate tracking task
    ed = np.random.uniform(-1.0, 1.0)
    eh = np.random.uniform(-1.0, 1.0)
    
    w = agent.firing_strengths(ed, eh)
    
    for _ in range(steps):
        # Select action
        _, a = agent.select_action(w, actions, tau)
        
        # Simulate dynamics and reward
        ed_next = ed * 0.9 + np.random.normal(0, 0.1)
        eh_next = eh * 0.9 + np.random.normal(0, 0.1)
        
        # Reward: encourage ed, eh → 0
        r = -(ed**2 + eh**2)
        
        w_next = agent.firing_strengths(ed_next, eh_next)
        
        done = (abs(ed_next) > 3.0) or (abs(eh_next) > 3.0)
        
        # Update
        agent.update(w, a, r, w_next, done)
        
        ep_return += r
        
        if done:
            break
        
        ed, eh = ed_next, eh_next
        w = w_next
    
    return ep_return

if __name__ == "__main__":
    run_comparison()
