# 점성 Burgers 방정식의 MPM-STD 기반 이산화 및 시스템 구성 보고서

**작성일:** 2026-04-25
**대상 프레임워크:** `/shared/home/wel1come1234/workspace/MPM-STD`
**참고 문헌:** Pan, Kim, Choi, *J. Comput. Phys.* **463** (2022) 111238; Kim, Kang, Pan, Choi, *Comput. Phys. Commun.* **290** (2023) 108779

---

## 1. 목적

본 보고서는 **점성 Burgers 방정식**을 MPM-STD(Monolithic Projection-based Method with Staggered Time Discretization) 프레임워크로 풀 때 다음을 정리한다.

1. 어떤 수식을 시간·공간에서 이산화하는가
2. 어떤 형태의 선형 시스템이 만들어지는가
3. 각 시스템의 계수가 어떻게 정의되는가

NOB 자연대류용 MPM-STD에서 **압력 보정·푸아송·에너지 단계를 모두 제거**하고, 모멘텀 시스템만 남긴 형태로 환원된다는 점이 핵심이다.

---

## 2. 지배 방정식

### 2.1 점성 Burgers 방정식 (3D 벡터 형식)

$$
\frac{\partial u_i}{\partial t} + u_j\,\frac{\partial u_i}{\partial x_j} \;=\; \nu\,\frac{\partial^2 u_i}{\partial x_j\,\partial x_j},\qquad i=1,2,3
$$

또는 벡터 형식으로

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u}\cdot\nabla)\mathbf{u} \;=\; \nu\,\nabla^2\mathbf{u}.
$$

NOB Navier–Stokes(논문 식 (8))와 비교했을 때 사라지는 항은 다음과 같다.

| 항목 | NOB-NS | Burgers |
|---|---|---|
| 압력 구배 $-\dfrac{1}{\hat\rho}\nabla p$ | 있음 | **없음** |
| 부력 $\mathbf{F}(\theta)$ | 있음 | **없음** |
| 비압축성 $\nabla\!\cdot\!\mathbf{u}=0$ | 있음 | **없음** |
| 에너지 방정식 | 있음 | **없음** |
| 점성 계수 | $\hat\mu(\theta)$ | 상수 $\nu$ |

따라서 풀어야 할 방정식은 **벡터 모멘텀 방정식 단 하나**이며, 압력에 의한 비국소 결합이 사라져 시스템은 본질적으로 *국소 ADI* 구조로 환원된다.

### 2.2 무차원 계수

Burgers 방정식 자체에는 Pr·Ra 같은 무차원수가 없고, 점성 계수 $\nu$ 하나만 존재한다. 본 보고서에서는 일반성을 위해 $\nu$ 그대로 두며, 필요시 Reynolds 수 $\mathrm{Re}=UL/\nu$ 도입이 가능하다. $\hat\rho=1$, $\hat C_p=1$, $\hat\kappa=0$ 으로 두면 NOB-MPM-STD 코드의 분기로도 즉시 환원된다.

---

## 3. 시간 이산화 (Crank–Nicolson)

논문 식 (12)에서 압력·부력 항을 제거한 형태:

$$
\frac{u_i^{n+1}-u_i^n}{\Delta t}
+ \tfrac{1}{2}\!\left[u_j^{n+1}\partial_j u_i^{n+1} + u_j^n \partial_j u_i^n\right]
= \tfrac{\nu}{2}\,\partial_{jj}\!\left(u_i^{n+1}+u_i^n\right).
$$

여기서 $\partial_j \equiv \partial/\partial x_j$, $\partial_{jj}\equiv \partial^2/\partial x_j\partial x_j$ 이다.

**중요한 점 — staggered 시간 이산화 불필요.** NOB-MPM-STD는 속도(정수 시간 $n+1$)와 스칼라(반정수 $n+\tfrac12$)를 분리해 운동량–에너지 결합을 끊었지만, Burgers에는 결합할 스칼라가 없다. 모든 변수를 정수 시간 레벨에서 다루며, **half-integer 인덱스는 등장하지 않는다**.

---

## 4. 비선형 항 선형화 (Beam–Warming)

논문 식 (15) 그대로:

$$
\tfrac12\!\left[u_j^{n+1}\partial_j u_i^{n+1} + u_j^n\partial_j u_i^n\right]
\;\approx\;
\tfrac12\!\left[\underbrace{u_j^n\,\partial_j u_i^{n+1}}_{\mathcal{N}^n\,u_i^{n+1}}
+ \underbrace{u_j^{n+1}\,\partial_j u_i^n}_{\mathbb{N}^n\,u_i^{n+1}}\right]
+ O(\Delta t^2).
$$

두 implicit 기여의 의미는 서로 다르다.

- **$\mathcal{N}^n$ — 동일 변수 대류항.** 동결된 속도장 $u_j^n$ 이 새 변수 $u_i^{n+1}$ 의 도함수에 작용. ADI sweep에서 **방향 $j$ 별로 자연스럽게 분리**된다.
- **$\mathbb{N}^n$ — Jacobian-타입 결합항.** 새 속도 $u_j^{n+1}$ 이 동결된 도함수 $\partial_j u_i^n$ 에 곱해진다.
  - $j=i$ (self-derivative, **own-direction**): $u_i^{n+1}\cdot\partial_i u_i^n$ → 방향 $i$ 의 1D 시스템에 *반응항*(reaction-like, 도함수 없는 대각 기여)으로 흡수.
  - $j\neq i$ (cross-component coupling): $u_j^{n+1}\cdot\partial_j u_i^n$ → $u_j^{n+1}$ 은 다른 성분 방정식의 미지수이므로 RHS로 lagged.

> **불변 규칙 (Beam–Warming own vs cross — 메모리 기록 일치):**
> *self-derivative* $\tfrac12\,\partial u_i/\partial x_d$ 는 **own-direction operator $M_d$ ($d=i$) 에만** 들어가고, cross-direction operator($d\neq i$)에는 절대 포함하지 않는다. 이를 어기면 Poiseuille 평형이 깨진다.

---

## 5. δ-Form 시스템

$\delta u_i^{n+1} = u_i^{n+1}-u_i^n$ 로 두고 정리하면 NOB 식 (17)에서 $-\nabla p/\hat\rho$, $F(\theta)$ 를 제거한 형태:

$$
\boxed{\;\mathbb{A}^n\,\delta u_i^{n+1} = \mathbf{R}_i^{\,n}\;}
$$

### 5.1 시스템 행렬 $\mathbb{A}^n$

$$
\mathbb{A}^n \;=\; \frac{1}{\Delta t}\,\mathbb{I}
\;+\;\tfrac12\,\mathcal{N}^n
\;+\;\tfrac12\,\mathbb{N}^n_{\text{own}}
\;-\;\tfrac{\nu}{2}\,\mathcal{L},
$$

각 부분의 의미:

| 기호 | 형태 (성분 $u_i$ 기준) | 작용 |
|---|---|---|
| $\dfrac{1}{\Delta t}\mathbb{I}$ | 대각 | 시간 미분 |
| $\tfrac12\mathcal{N}^n$ | $\tfrac12\,u_j^n\,\partial_j$ — 모든 방향 $j$ 합 | 동일 변수 대류 |
| $\tfrac12\mathbb{N}^n_{\text{own}}$ | $\tfrac12\,(\partial_i u_i^n)\cdot$ — **방향 $i$ 만** | 자기-도함수 반응항 |
| $-\tfrac{\nu}{2}\mathcal{L}$ | $-\tfrac{\nu}{2}\,\partial_{jj}$ | 점성 확산 |

### 5.2 RHS $\mathbf{R}_i^{\,n}$

$$
\mathbf{R}_i^{\,n}
= \underbrace{-\,u_j^n\,\partial_j u_i^n}_{\text{명시적 대류항(시점 }n\text{)}}
\;+\;\underbrace{\nu\,\partial_{jj} u_i^n}_{\text{명시적 확산항}}
\;+\;\underbrace{\mathbf{R}_i^{\,\text{cross}}}_{\text{lagged }j\neq i\text{ 항}}
\;+\;\mathbf{mbc}_i^{\,n+1}.
$$

여기서

$$
\mathbf{R}_i^{\,\text{cross}}
= -\,\tfrac12\,\sum_{j\neq i}\,\big(u_j^n\big)\big(\partial_j u_i^n\big)
$$

는 $\mathbb{N}^n$ 의 cross-coupling을 explicit으로 처리한 부분(혹은 outer iteration에서 갱신).

$\mathbf{mbc}_i^{n+1}$ 은 Dirichlet 경계값을 RHS로 옮긴 항(논문 부록 B와 동일 구조).

---

## 6. 근사 인수분해 (ADI, 3-Stage Sweep)

논문 식 (20)의 무인수분해 패턴을 그대로 적용:

$$
\frac{1}{\Delta t}\,\big(\mathbb{I}+\Delta t\,M_x\big)\,\big(\mathbb{I}+\Delta t\,M_y\big)\,\big(\mathbb{I}+\Delta t\,M_z\big)\,\delta u_i^{n+1}
\;=\;\mathbf{R}_i^{\,n},
$$

오차 $O(\Delta t^2)$ 로 시간 정확도 손실 없음.

각 1D 연산자 $M_d$ 는 (성분 $u_i$, 방향 $d$):

$$
M_d^{(i)} \;=\; \tfrac12\,u_d^n\,\partial_d \;-\; \tfrac{\nu}{2}\,\partial_{dd}
\;+\; \underbrace{\tfrac12\,(\partial_i u_i^n)\,\delta_{di}}_{\text{own-direction에만}}.
$$

Kronecker 델타 $\delta_{di}$ 는 자기-도함수 반응항이 **오직 $d=i$ 일 때만** 켜짐을 명시한다.

### Sweep 절차

각 시간 스텝마다 성분 $u_i$ 별로:

1. RHS 조립: $\mathbf{R}_i^{\,n}$
2. **x-sweep:** $(\mathbb{I}+\Delta t\,M_x^{(i)})\,\mathbf{u}_i^{*} = \Delta t\,\mathbf{R}_i^{\,n}$ → x-방향 삼중대각, PaScaL_TDMA
3. **y-sweep:** $(\mathbb{I}+\Delta t\,M_y^{(i)})\,\mathbf{u}_i^{**} = \mathbf{u}_i^{*}$ → y-방향 삼중대각, PaScaL_TDMA
4. **z-sweep:** $(\mathbb{I}+\Delta t\,M_z^{(i)})\,\delta u_i^{n+1} = \mathbf{u}_i^{**}$ → z-방향 삼중대각, PaScaL_TDMA
5. 갱신: $u_i^{n+1} = u_i^n + \delta u_i^{n+1}$

**압력 projection 단계 없음.** ∇·u=0 구속이 없으므로 단계 5에서 끝.

세 성분 $u_1,u_2,u_3$ 는 독립적으로 위 절차를 적용한다(Cross 결합은 RHS에 lagged).

---

## 7. 1D 삼중대각 시스템의 명시적 계수

격자점 $j$ 에서 방향 $d$ 의 1D 시스템

$$
a_M\,\delta u_{j-1} + a_C\,\delta u_j + a_P\,\delta u_{j+1} \;=\; r_j
$$

의 계수를 정리한다. 격자 간격을 균일 $h_d$ 로 가정하고, 비균일이면 $h_d \to h_{d,j\pm1/2}$ 로 일반화.

### 7.1 중심차분 이산화

$$
\partial_d v\big|_j \approx \frac{v_{j+1}-v_{j-1}}{2h_d},\qquad
\partial_{dd} v\big|_j \approx \frac{v_{j+1}-2v_j+v_{j-1}}{h_d^2}.
$$

### 7.2 계수 표 (성분 $u_i$, 방향 $d$ sweep)

다음 약식 기호를 도입:

- 대류 계수: $\mathsf{C}_d \equiv \dfrac{u_d^{\,n}}{4 h_d}$  (½·½ = ¼ from CN average × central diff)
- 확산 계수: $\mathsf{D}_d \equiv \dfrac{\nu}{2 h_d^{\,2}}$
- 자기-도함수 반응 계수: $\mathsf{S}_d \equiv \tfrac12\,\big(\partial_d u_i^{\,n}\big)\big|_j \cdot \delta_{di}$

| 위치 | 계수 |
|---|---|
| $a_M$ (subdiagonal, $j-1$) | $\Delta t\;\big[\,-\mathsf{C}_d - \mathsf{D}_d\,\big]$ |
| $a_C$ (diagonal, $j$) | $1 + \Delta t\;\big[\,2\,\mathsf{D}_d + \mathsf{S}_d\,\big]$ |
| $a_P$ (superdiagonal, $j+1$) | $\Delta t\;\big[\,+\mathsf{C}_d - \mathsf{D}_d\,\big]$ |

### 7.3 부호·대각우월성 분석

- **확산 기여:** $a_C^{\text{diff}} = 1 + 2\Delta t\,\mathsf{D}_d > 0$, $a_M^{\text{diff}}=a_P^{\text{diff}}=-\Delta t\,\mathsf{D}_d < 0$ — 대각 우월.
- **대류 기여:** $a_M^{\text{conv}}=-\Delta t\,\mathsf{C}_d$, $a_P^{\text{conv}}=+\Delta t\,\mathsf{C}_d$ — 비대칭, 부호는 $u_d^n$ 부호에 의존. 대각에는 기여 없음.
- **반응 기여 ($d=i$ 만):** $a_C$ 에 $\Delta t\,\mathsf{S}_d$ 추가. $\mathsf{S}_d$ 부호는 $\partial_i u_i^n$ 부호에 의존하나, 실제 진폭은 보통 $O(1)$ 이며 $2\mathsf{D}_d$ 가 충분히 크면 dominated.
- **TDMA 안정 조건:** $|a_C| > |a_M|+|a_P|$. 확산이 대류를 압도하면 (즉 격자 Péclet $\mathrm{Pe}_d=|u_d^n|h_d/\nu < 2$) 자동 보장.

### 7.4 기존 코드 매핑

[src/core/core_momentum.f90](src/core/core_momentum.f90) 의 NOB 모멘텀 계수에서 다음을 0으로 두면 Burgers와 동일해진다.

| 코드의 항 | Burgers 처리 |
|---|---|
| `Gp/ρ̂` (압력 구배) | 제거 |
| `F(θ)` (부력) | 제거 |
| `μ̂(θ)` 가변 점성 | 상수 $\nu$ 로 대체 |
| `(∇u)^T` 추가 점성항 | 제거(스칼라 라플라시안만) |
| `1/ρ̂` 곱 | 1로 대체 |

[core_pressure.f90](src/core/core_pressure.f90), [core_energy.f90](src/core/core_energy.f90) 의 호출은 entrypoint에서 모두 제거.

---

## 8. RHS $\mathbf{R}_i^{\,n}$ 명시적 형태

격자점 $(i,j,k)$ 에서 (uniform 격자, 성분 $u_1$ 예시):

$$
R_{1,(i,j,k)}^{\,n} = -\,(\text{conv})_{(i,j,k)} + \nu\,(\text{lap})_{(i,j,k)} + R_1^{\,\text{cross}} + \mathsf{mbc}
$$

with

$$
(\text{conv})_{(i,j,k)} = u_1^n\,\frac{u_{1,i+1}^n-u_{1,i-1}^n}{2h_x} + u_2^n\,\frac{u_{1,j+1}^n-u_{1,j-1}^n}{2h_y} + u_3^n\,\frac{u_{1,k+1}^n-u_{1,k-1}^n}{2h_z}
$$

$$
(\text{lap})_{(i,j,k)} = \frac{u_{1,i+1}^n-2u_{1,i}^n+u_{1,i-1}^n}{h_x^2} + \frac{u_{1,j+1}^n-2u_{1,j}^n+u_{1,j-1}^n}{h_y^2} + \frac{u_{1,k+1}^n-2u_{1,k}^n+u_{1,k-1}^n}{h_z^2}.
$$

Cross-coupling lag(선택):

$$
R_1^{\,\text{cross}} = -\,\tfrac12\,\Big[u_2^n\,(\partial_y u_1^n) + u_3^n\,(\partial_z u_1^n)\Big].
$$

(주의: 이 항은 이미 $-(\text{conv})$ 안에 부분적으로 포함되므로, 완전 lagged Picard 형식을 쓸지 Beam–Warming explicit 형식을 쓸지에 따라 부호·계수 조정 필요. 깨끗한 구현은 $\mathbf{R}^n=-\mathcal{N}^n u^n + \nu\mathcal{L}u^n$ 만 두고 $\mathbb{N}^n$ 항은 평균-CN 식에서 자연 흡수되도록 하는 것이다.)

---

## 9. 경계 조건 처리

### 9.1 Dirichlet (벽경계)

$u_i^{n+1}|_{\partial\Omega}$ 가 알려져 있으므로 $\delta u_i^{n+1}|_{\partial\Omega}$ 도 결정. 1D sweep의 경계 인접 행에서 알려진 $\delta u$ 를 RHS로 옮기고 sub/super-diagonal을 0 처리. 논문 부록 B의 `mbc` 와 동일.

### 9.2 주기 (homogeneous direction)

`PaScaL_TDMA_many_solve_cycle` 사용. x, z 등 주기 방향에서.

### 9.3 Neumann (압력 없음 → 단순화)

Burgers는 압력 Neumann 경계가 없으므로 처리 항목 자체가 사라진다.

---

## 10. 풀이 절차 요약

각 시간 스텝 $n \to n+1$:

```
1. 세 성분 RHS 조립:
     R_i^n = −(u^n·∇)u_i^n + ν∇²u_i^n + cross_lag + mbc_i^{n+1}    (i=1,2,3)

2. 성분별 ADI 3-stage sweep (각 성분 독립):
     for i = 1, 2, 3:
        x-sweep: PaScaL_TDMA  →  u_i*
        y-sweep: PaScaL_TDMA  →  u_i**
        z-sweep: PaScaL_TDMA  →  δu_i^{n+1}

3. 갱신:
     u_i^{n+1} = u_i^n + δu_i^{n+1}
```

NOB-MPM-STD의 `solve_energy → solve_momentum → solve_pressure → projection` 4단계 중 **모멘텀 단계 1개만 살아남는다**.

---

## 11. 시간 정확도

다음 근사들이 모두 $O(\Delta t^2)$ 만 누적시키므로 **시간 2차 정확도 유지**:

- Crank–Nicolson 자체: $O(\Delta t^2)$
- Beam–Warming 선형화 (논문 식 (15)): $O(\Delta t^2)$
- Cross-coupling lag: 이를 inner Picard 로 보완하지 않으면 $O(\Delta t)$ 로 떨어질 수 있으므로, 한 step에서 비반복 처리하려면 $\mathbb{N}^n_{\text{own}}$ 만 implicit로 두고 cross는 Beam–Warming explicit($\propto u_j^n \partial_j u_i^n$) 로 평균화에 흡수해 $O(\Delta t^2)$ 유지.
- ADI 인수분해 (논문 식 (20)): $O(\Delta t^2)$
- 공간: 2차 중심차분 → $O(h^2)$

Burgers analytic solution(Cole–Hopf) 또는 manufactured solution으로 $\mathit{l}^2$ 수렴률 검증 가능.

---

## 12. 안정성과 시간 스텝

비선형 대류항·확산항을 모두 음해(implicit)로 처리하므로 **CFL 무제한**(strict한 의미에서). 다만 정확도 측면에서는 다음을 권장.

- 격자 Péclet $\mathrm{Pe}_d = |u_d^n|\,h_d/\nu \lesssim 2$ — 중심차분이 진동 없이 동작하는 영역.
- $\mathrm{Pe}_d > 2$ 영역에서는 격자 진동 발생 가능 → 메모리에 기록된 **Filtered_TDMA 적용 후보**.
- 충격(shock) 형성 직후엔 중심차분이 wiggle 발생 → upwind 또는 limiter 필요.

---

## 13. 검증 케이스 추천

1. **1D viscous Burgers (Cole–Hopf 정해)** — 시간·공간 2차 정확도 확인.
2. **2D 회전 패치(rotating patch)** — 대류 정확도와 자기-도함수 항 검증.
3. **3D Taylor–Green 유사 점성 감쇠** — 비선형 확산 dissipation 확인.
4. **벽경계 + 강제항** — 부록 B 형식의 `mbc` 가 정확히 들어가는지 확인.

---

## 14. 결론

Burgers 방정식은 MPM-STD 모멘텀 풀이의 **가장 단순한 부분집합**이다.

- 시간이산화: Crank–Nicolson (논문 식 (12) 그대로).
- 비선형 처리: Beam–Warming 선형화 (식 (15)) — own/cross 구분 규칙 준수.
- 시스템: $\mathbb{A}^n\,\delta u_i^{n+1} = \mathbf{R}_i^{\,n}$ 의 hepta-diagonal 시스템을 ADI로 3개의 삼중대각 시스템으로 분해.
- 1D 계수: $a_M=\Delta t(-\mathsf{C}_d - \mathsf{D}_d)$, $a_C=1+\Delta t(2\mathsf{D}_d+\mathsf{S}_d)$, $a_P=\Delta t(+\mathsf{C}_d - \mathsf{D}_d)$.
- 풀이 단계: PaScaL_TDMA 3회 sweep × 3성분.
- **압력 단계·에너지 단계 모두 제거** — 코드 측면에선 entrypoint에서 두 호출만 빼고 모멘텀 RHS에서 $\nabla p/\hat\rho$, $F(\theta)$ 두 항만 제거하면 충분.

Filtered_TDMA 적용 가능성도 그대로 이식 가능하며, 격자 Péclet 영역에서 격자 스케일 진동을 억제하는 용도로 즉시 활용할 수 있다.
