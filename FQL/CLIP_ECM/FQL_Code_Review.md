# FQL Trajectory Tracking Code Review

## 📋 개요
- **파일명**: `fql_basic_tracking.py`
- **목적**: Fuzzy Q-Learning 기반 로봇 궤적 추적
- **알고리즘**: FQL + Pure Pursuit + Adaptive Lookahead
- **전체 평가**: ⭐⭐⭐⭐☆ (4/5)

---

## 🔴 Critical Issues (즉시 수정 필요)

### 1. 초기 Target 설정 불명확 (Line 345-365)
**문제:**
```python
idx = closest_point(x, y, ref_traj)
target_idx = target_index_by_distance(idx, ref_s, L_MIN)
xg_init, yg_init = ref_traj[idx]  # idx 사용
ed_init, _, _, _ = tracking_errors_ed_eh(x, y, theta, xg_init, yg_init)

if ed_init > OUTSIDE_ED:
    target_idx = idx  # 조건부 재할당
xg, yg = ref_traj[target_idx]  # 최종 할당
```

**이슈:**
- `xg_init`은 `idx` (closest point)에서 가져옴
- `ed_init` 계산에 `xg_init` 사용
- 그러나 실제 목표 `xg, yg`는 조건에 따라 다른 `target_idx`에서 가져옴
- 초기 상태 계산과 실제 목표가 불일치할 수 있음

**수정 제안:**
```python
# initial state
idx = closest_point(x, y, ref_traj)
idx_prev = idx

# Determine initial target based on distance from path
x_closest, y_closest = ref_traj[idx]
ed_init, _, _, _ = tracking_errors_ed_eh(x, y, theta, x_closest, y_closest)

if ed_init > OUTSIDE_ED:
    # Outside: target is closest point (capture mode)
    target_idx = idx
else:
    # Inside: use lookahead
    target_idx = target_index_by_distance(idx, ref_s, L_MIN)

xg, yg = ref_traj[target_idx]
```

---

### 2. 변수 초기화 누락 위험 (Line 405)
**문제:**
```python
if outside:
    omega_raw = -K_OUT * eh
    a = None
    omega_ref = 0.0  # ← 여기서만 초기화
else:
    omega_raw, a = agent.blended_action(w, omega_actions, tau=tau)
```

**이슈:**
- `omega_ref`가 `outside` 블록에서만 명시적으로 초기화됨
- 이후 코드에서 `outside`가 아닐 때 `omega_ref` 계산 전에 참조될 수 있음 (Line 485)

**수정 제안:**
```python
# 루프 시작 전 초기화
omega_ref = 0.0

for k in range(steps):
    ...
    if outside:
        omega_raw = -K_OUT * eh
        a = None
        # omega_ref는 이미 0.0
    else:
        omega_raw, a = agent.blended_action(w, omega_actions, tau=tau)
        # omega_ref는 나중에 curvature로 계산
```

---

## 🟡 Medium Priority Issues (개선 권장)

### 3. 중복 계산 (Line 565-610)
**문제:**
```python
# 첫 번째 루프: rec_step 계산
rec_step = None
stable_cnt = 0
for k in range(len(yR_log)):
    if yR_log[k] < EPS_CTE:
        stable_cnt += 1
        if stable_cnt >= M_STABLE:
            rec_step = k - M_STABLE + 1
            break
    else:
        stable_cnt = 0

# 두 번째 루프: rec_step_success 계산 (거의 동일)
rec_step_success = None
success = 0
stable_cnt = 0
for k in range(len(yR_log)):
    if (yR_log[k] < EPS_CTE) and (abs(eh_log[k]) < EPS_EH):
        stable_cnt += 1
        if stable_cnt >= M_STABLE:
            rec_step_success = k - M_STABLE + 1
            break
    else:
        stable_cnt = 0
```

**이슈:**
- 두 루프가 거의 동일한 로직을 수행
- 조건만 약간 다름 (`eh_log` 추가 체크)
- 비효율적인 계산

**수정 제안:**
```python
def find_stabilization_time(
    yR_log: np.ndarray, 
    eh_log: np.ndarray | None, 
    eps_cte: float, 
    eps_eh: float | None, 
    m_stable: int
) -> int | None:
    """Find the first time step where the system stabilizes."""
    stable_cnt = 0
    for k in range(len(yR_log)):
        cte_ok = yR_log[k] < eps_cte
        eh_ok = (eh_log is None) or (abs(eh_log[k]) < eps_eh)
        
        if cte_ok and eh_ok:
            stable_cnt += 1
            if stable_cnt >= m_stable:
                return k - m_stable + 1
        else:
            stable_cnt = 0
    return None

# 사용
rec_step = find_stabilization_time(yR_log, None, EPS_CTE, None, M_STABLE)
rec_step_success = find_stabilization_time(yR_log, eh_log, EPS_CTE, EPS_EH, M_STABLE)
```

---

### 4. Near-Convergence 로직 복잡도 (Line 455-475)
**문제:**
```python
if not near_converged:
    if ed2 < 0.4 and abs(eh2) < math.radians(15.0):
        near_converged = True
else:
    if ed2 > 0.6 or abs(eh2) > math.radians(20.0):
        near_converged = False

# 이후 near_converged이면 다시 target 재계산
if near_converged:
    L_dyn = L_MIN
    target_idx = target_index_by_distance(idx, ref_s, L_dyn)
    xg, yg = ref_traj[target_idx]
    ed2, eh2, _, yR2 = tracking_errors_ed_eh(x, y, theta, xg, yg)
```

**이슈:**
- 히스테리시스 로직은 좋지만 복잡함
- `near_converged` 상태가 되면 target을 재계산하고 오차도 재계산
- 이는 다음 iteration에 영향을 줄 수 있음
- 코드 흐름이 명확하지 않음

**수정 제안:**
```python
# 상수를 명확히 정의
CONVERGE_ED_ENTER = 0.4  # Enter convergence zone
CONVERGE_ED_EXIT = 0.6   # Exit convergence zone
CONVERGE_EH_ENTER = math.radians(15.0)
CONVERGE_EH_EXIT = math.radians(20.0)

# 히스테리시스 체크
def update_convergence_state(near_converged, ed, eh):
    """Update convergence state with hysteresis."""
    if not near_converged:
        # Try to enter convergence zone
        if ed < CONVERGE_ED_ENTER and abs(eh) < CONVERGE_EH_ENTER:
            return True
    else:
        # Check if we should exit convergence zone
        if ed > CONVERGE_ED_EXIT or abs(eh) > CONVERGE_EH_EXIT:
            return False
    return near_converged

near_converged = update_convergence_state(near_converged, ed2, eh2)
```

---

### 5. 긴 Main 함수
**문제:**
- `main()` 함수가 600+ 라인
- 여러 역할 수행: 초기화, 학습, 평가, 시각화
- 유지보수 어려움

**수정 제안:**
```python
def train_fql_agent(agent, ref_traj, ref_s, episodes, steps, dt, ...):
    """Train FQL agent."""
    reward_hist = []
    # ... 학습 로직 ...
    return agent, reward_hist

def evaluate_agent(agent, ref_traj, ref_s, steps, dt, ...):
    """Evaluate trained agent."""
    # ... 평가 로직 ...
    return traj, metrics_dict

def plot_results(traj, ref_traj, ed_log, eh_log, omega_log, ...):
    """Create all plots."""
    # ... 플롯 로직 ...

def compute_metrics(traj, ed_log, eh_log, omega_log, v_log, yR_log, dt):
    """Compute all performance metrics."""
    # ... 메트릭 계산 ...
    return metrics_dict

def main():
    # Setup
    ref_traj, ref_s = generate_reference_trajectory(...)
    agent = FQLAgent(...)
    
    # Train
    agent, reward_hist = train_fql_agent(agent, ...)
    
    # Evaluate
    traj, metrics = evaluate_agent(agent, ...)
    
    # Visualize
    plot_results(traj, ref_traj, ...)
```

---

## 🟢 Good Practices (잘된 점)

### 1. ✅ 명확한 퍼지 로직
```python
MF_5 = {
    "NB": (-1.50, -1.00, -0.80, -0.40),
    "NS": (-0.80, -0.40, -0.20, 0.00),
    "Z": (-0.20, 0.00, 0.00, 0.20),
    "PS": ( 0.00, 0.20, 0.40, 0.80),
    "PB": ( 0.40, 0.80, 1.00, 1.50),
}
```
- 5개 linguistic term으로 깔끔하게 정의
- Overlapping이 적절함

### 2. ✅ 포괄적인 성능 메트릭
15개 이상의 메트릭 계산:
- Tracking error (RMSE, MAE, Max CTE)
- Control effort (Energy, Smoothness)
- Time metrics (TTS, T_settle, T_recover)
- Computational time

### 3. ✅ 적절한 Rate Limiting
```python
DELTA_MAX = 0.15  # rad/s per step
omega = clamp(omega_raw, prev_omega - DELTA_MAX, prev_omega + DELTA_MAX)
```
- Slew-rate 제한으로 급격한 변화 방지
- 실제 로봇 제약 반영

### 4. ✅ Adaptive Lookahead
```python
L_dyn = L_MIN + K_V * v
L_dyn = clamp(L_dyn, L_MIN, L_MAX)
```
- 속도에 따라 lookahead 거리 조정
- Pure Pursuit의 변형으로 적절

### 5. ✅ Outside/Inside Mode 분리
- 경로 밖에 있을 때: radial capture
- 경로 안에 있을 때: FQL tracking
- 초기 수렴 문제 해결

### 6. ✅ 수치 안정성
```python
return (x - a) / (b - a + 1e-12)  # 0으로 나누기 방지
return 2.0 * math.sin(alpha) / max(L, 1e-3)  # 최소값 보장
```

---

## 📊 코드 품질 점수

| 카테고리 | 점수 | 비고 |
|---------|------|------|
| 알고리즘 설계 | 9/10 | FQL + adaptive lookahead 우수 |
| 코드 구조 | 6/10 | 긴 main 함수, 리팩토링 필요 |
| 가독성 | 7/10 | 주석은 있으나 복잡한 로직 설명 부족 |
| 견고성 | 7/10 | 일부 초기화 이슈 |
| 효율성 | 8/10 | 합리적인 계산 복잡도 |
| 테스트/검증 | 8/10 | 포괄적인 메트릭과 시각화 |
| **전체** | **7.5/10** | 실용적이고 작동하는 코드 |

---

## 🔧 Priority Action Items

### High Priority (즉시)
1. ✅ Line 345-365: 초기 target 설정 명확화
2. ✅ Line 405: `omega_ref` 초기화 보장

### Medium Priority (다음 버전)
3. ✅ Line 565-610: 중복 계산 함수화
4. ✅ 전체: Main 함수 리팩토링 (600+ lines → 여러 함수로 분할)
5. ✅ Line 455-475: Near-convergence 로직 명확화

### Low Priority (시간이 될 때)
6. ⚪ 설정값을 별도 dataclass나 config 파일로 분리
7. ⚪ Type hints 추가 (이미 일부 있음)
8. ⚪ Unit tests 작성
9. ⚪ Docstring 보완

---

## 💡 추가 제안

### 1. 설정 관리
```python
from dataclasses import dataclass

@dataclass
class TrackingConfig:
    """Configuration for FQL tracking."""
    # Lookahead
    L_MIN: float = 0.4
    L_MAX: float = 1.5
    K_V: float = 0.6
    
    # Velocity
    V_MIN: float = 0.25
    V_MIN_CONV: float = 0.15
    
    # Outside capture
    OUTSIDE_ED: float = 1.8  # R_circle * 0.3
    K_OUT: float = 1.5
    
    # Control
    OMEGA_MAX: float = 1.2
    DELTA_MAX: float = 0.15
    
    # FQL
    N_ACTIONS: int = 11
    ALPHA: float = 0.10
    GAMMA: float = 0.98
```

### 2. 로깅 개선
```python
import logging

logger = logging.getLogger(__name__)
logger.info(f"Episode {ep}: return={ep_return:.2f}, converged={near_converged}")
```

### 3. 체크포인트 저장
```python
# Best model 저장
if ep_return > best_score:
    best_score = ep_return
    np.savez('best_fql_model.npz', 
             Q=agent.Q, 
             score=best_score, 
             episode=ep)
```

---

## 📈 성능 특성

### 계산 복잡도
- **Fuzzification**: O(5) per input → O(10) total
- **Firing strengths**: O(25) = 5×5 rules
- **Q aggregation**: O(25 × 11) = O(275)
- **전체**: O(1) - 상수 시간, 실시간 제어 가능

### 메모리 사용
- **Q-table**: 25 rules × 11 actions = 275 floats ≈ 2.2 KB
- **Trajectory**: 2000 points × 2 coords = 32 KB
- **Logs** (2400 steps): ~100 KB
- **전체**: <1 MB - 매우 효율적

### 실시간성
```
Mean inference: ~0.001 ms (CPU only)
Max inference: ~0.010 ms
Control rate: 20 Hz (dt=0.05s) 충분히 가능
```

---

## ✅ 결론

**장점:**
- 🎯 알고리즘적으로 우수한 설계 (FQL + Adaptive Lookahead)
- 📊 매우 포괄적인 성능 평가
- ⚡ 실시간 제어 가능한 효율성
- 🛡️ 적절한 안전 장치 (rate limiting, clamping)

**단점:**
- 🔧 코드 구조 개선 필요 (긴 main 함수)
- 🐛 몇 가지 초기화 이슈
- 📝 복잡한 로직에 대한 설명 부족

**최종 평가:**
✅ **Production-ready with minor fixes**

이 코드는 논문 구현이나 실험용으로 충분히 사용 가능합니다. 위의 Critical Issues만 수정하면 즉시 사용해도 됩니다. 전체 리팩토링은 시간이 날 때 천천히 진행하면 됩니다.
