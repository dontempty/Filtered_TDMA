# PaScaL_TCS 모멘텀 방정식 분석 및 Filtered_TDMA 적용 가능성 검토

**작성일:** 2026-04-22  
**대상 코드:** `/shared/home/wel1come1234/workspace/PaScaL_TCS`

---

## 1. 코드 개요

PaScaL_TCS는 **비 Oberbeck-Boussinesq(NOB) 자연대류**를 3차원 직교 격자에서 시뮬레이션하는 MPI 병렬 CFD 코드다. Rayleigh–Bénard 대류와 같이 수평 가열면 사이의 유동을 주 대상으로 한다.

### 1.1 지배 방정식 (무차원화)

코드는 다음 세 방정식 계를 풀어낸다.

**연속 방정식:**
```
∇·u = 0   (질량 보존, NOB 근사에서 수정항 포함)
```

**모멘텀 방정식 (NOB Boussinesq):**
```
∂u/∂t + (u·∇)u = -Cmp·(1/ρ)∇p + Cmu·(1/ρ)∇·[μ(∇u + (∇u)ᵀ)] + Cmt·(θ + a₁₂/a₁₁·θ²·ΔT)·(1/ρ)·ê_y
```

**열 방정식:**
```
∂T/∂t + (u·∇)T = Ct·(1/ρCp)∇·(κ∇T)
```

**무차원 계수 정의** (`module_global.f90:139`):

| 계수 | 정의 | 의미 |
|------|------|------|
| `Cmu` | `sqrt(Pr/Ra)` | 점성 계수 |
| `Cmt` | `1.0` | 부력(온도) 계수 |
| `Ct`  | `1/sqrt(Ra·Pr)` | 열 확산 계수 |
| `Cmp` | `1.0` | 압력 계수 |

**물성치 모델** (온도 의존 다항식):
```
ρ(T) = a₁₀·[1 + a₁₁·θ + a₁₂·θ² + ...]
μ(T) = ρ(T)·ν₀·[1 + d₁₁·θ + d₁₂·θ² + ...]   / μ₀
κ(T) = c₁₀·[1 + c₁₁·θ + c₁₂·θ² + ...]
```
여기서 `θ = ΔT·T` (정규화 온도).  
현재 기본 설정(`a₁₁=0, a₁₂=0, ...`)은 Oberbeck-Boussinesq(OB) 근사이며, 주석 처리된 물(water)·글리세롤(glycerol) 계수로 NOB 계산 가능.

### 1.2 격자 및 이산화

- **격자 타입:** Staggered Grid (교엇갈린 격자)
  - 스칼라(P, T): 체심(cell center)
  - 속도 성분(U, V, W): 각 방향 면 중심(face center)
- **공간 정확도:** 2차 중심차분 (Central Difference)
- **시간 전진:** 1차 암묵적 Euler (각 시간 스텝이 수렴 반복 없이 단번 진행)

---

## 2. 모멘텀 방정식 풀이 알고리즘

### 2.1 전체 시간 진행 순서 (main.f90)

```
시간 스텝 루프:
  1. 열 방정식 풀이 → T^{n+1}
  2. 모멘텀 방정식 풀이 → u^{*}  (압력 미보정 중간 속도)
  3. 압력 포아송 방정식 풀이 → δp
  4. 속도 보정 (Pressure Projection) → u^{n+1}
```

### 2.2 3단계 연산자 분리(Dimensional Splitting)

모멘텀 방정식은 **3단계 교대방향 암묵법(ADI)**으로 분리된다.  
각 속도 성분 dU, dV, dW 모두 동일한 3-단계 구조를 따른다 (`module_solve_momentum.f90`).

**대상 방정식 (증분 형식):**
```
(δu) / Δt = RHS_explicit + [implicit terms along each direction] × δu
```

여기서 `δu = u^{n+1} - u^n`이고, RHS는 현재 시간 스텝의 알려진 값으로 구성.

---

## 3. 각 방향 선형 시스템 구조

### 3.1 U-방정식 (mpi_momentum_solvedU)

#### **1단계: Z-방향 삼중대각 시스템**

격자점 `(i, j, k)`에서 z-방향 계수 (`module_solve_momentum.f90:389-413`):

```fortran
mACK =  invRhocCmu_half/dx3(k)*(mu6/dmx3(kp) + mu5/dmx3(k))
        + 0.25d0*(-w6/dmx3(kp) + w5/dmx3(k))

mAPK = -invRhocCmu_half/dx3(k)*mu6/dmx3(kp)
        + 0.25d0*( w6/dmx3(kp))

mAMK = -invRhocCmu_half/dx3(k)*mu5/dmx3(k)
        + 0.25d0*(-w5/dmx3(k))
```

행렬 계수:
```
a_m = mAMK · Δt          (하삼각, 서쪽, U(i,j,k-1) 계수)
a_c = mACK · Δt + 1      (대각, 중심, U(i,j,k) 계수)
a_p = mAPK · Δt          (상삼각, 동쪽, U(i,j,k+1) 계수)
```

각 계수의 물리적 구성:
- **확산(점성) 기여:** `±invRhocCmu_half/dx3(k)·μ_{k±1/2}/dmx3(k±1)` → 항상 `a_c > 0`, `a_m, a_p < 0` (대각 지배)
- **대류 기여:** `±0.25·w_{k±1/2}/dmx3(k±1)` → 이류 속도에 따라 부호 변동
- **z-방향은 주기 경계**: `PaScaL_TDMA_many_solve_cycle` 사용

#### **2단계: X-방향 삼중대각 시스템**

```fortran
mACI =  invRhocCmu_half/dmx1(i)*(Mu(i,j,k)/dx1(i) + Mu(i,j,k)/dx1(im))*2.0
        + 0.25d0*(dudx2·0.5 + dudx1·0.5 - u2/dx1(i) + u1/dx1(im))

mAPI = -invRhocCmu_half/dmx1(i)*Mu(i,j,k)/dx1(i)*2.0
        + 0.25d0*(u2/dx1(i) + dudx2·0.5)

mAMI = -invRhocCmu_half/dmx1(i)*Mu(i,j,k)/dx1(im)*2.0
        + 0.25d0*(-u1/dx1(im) + dudx1·0.5)
```

행렬 계수:
```
a_m = mAMI · Δt
a_c = mACI · Δt + 1
a_p = mAPI · Δt
```

- 대류항이 `∂(uu)/∂x`의 완전한 선형화: `0.5·u·∂u/∂x + 0.5·u²/∂x` 형태 (skew-symmetric 이산화)
- x-방향은 주기 경계: `PaScaL_TDMA_many_solve_cycle` 사용

#### **3단계: Y-방향 삼중대각 시스템**

```fortran
mACJ =  invRhocCmu_half/dx2(j)*(mu4/dmx2(jp) + mu3/dmx2(j))
        + 0.25d0*(-v4/dmx2(jp) + v3/dmx2(j))

mAPJ = -invRhocCmu_half/dx2(j)*mu4/dmx2(jp)
        + 0.25d0*(v4/dmx2(jp))

mAMJ = -invRhocCmu_half/dx2(j)*mu3/dmx2(j)
        + 0.25d0*(-v3/dmx2(j))

! 벽면 경계 처리 (j=1, j=n2m에서 off-diagonal 항 0으로 강제)
mAPJ = mAPJ * dble(jup)    ! jup=0 at upper wall
mAMJ = mAMJ * dble(jum)    ! jum=0 at lower wall
```

행렬 계수:
```
a_m = mAMJ · Δt
a_c = mACJ · Δt + 1
a_p = mAPJ · Δt
```

- **y-방향은 벽면 경계(Dirichlet BC)**: `PaScaL_TDMA_many_solve` 사용 (비주기)
- 벽면에서 `mAPJ`, `mAMJ`를 0으로 설정하여 경계 조건 적용

### 3.2 V-방정식 (mpi_momentum_solvedV)

V-방정식은 y-방향이 주 확산 방향이므로 계수 구성이 다르다.

| 방향 | 점성항 형태 | 비고 |
|------|------------|------|
| X | `μ_{i±1/2}·∂V/∂x` | 교차 점성 포함 |
| Y (대각) | `2·Mu(i,j,k)·∂V/∂y` | 주 점성항 (계수 2 곱) |
| Z | `μ_{k±1/2}·∂V/∂z` | |

Y-방향 주대각 계수 (`module_solve_momentum.f90:734`):
```fortran
mACJ = invRhocCmu_half/dmx2(j)*(Mu(i,j,k)/dx2(j) + Mu(i,jm,k)/dx2(jm))*2.0
       + 0.25d0*(dvdy4·0.5 + dvdy3·0.5 - v4/dx2(j) + v3/dx2(jm))
```

### 3.3 W-방정식 (mpi_momentum_solvedW)

W-방정식은 z-방향이 주 확산 방향. V-방정식과 대칭적인 구조. z-방향:
```fortran
mACK = invRhocCmu_half/dx3(k)*(mu6/dmx3(kp) + mu5/dmx3(k))*2.0 + 대류항
```

---

## 4. RHS 구성 상세

### 4.1 U-방정식 RHS

`module_solve_momentum.f90:334-413`에서 RHS는 다음 항들로 구성:

```
RHS = [점성항] + [압력 기울기] + [부력항] + [벽면 BC 기여] - [암묵 행렬 × u^n]
```

**① 점성항 (Viscous Stress):**
```fortran
viscous_u1  = (Mu(i,j,k)·dudx2 - Mu(im,j,k)·dudx1) / dmx1(i)    ! ∂(2μ·∂u/∂x)/∂x
viscous_u2  = (mu4·dudy4 - mu3·dudy3) / dx2(j)                    ! ∂(μ·∂u/∂y)/∂y
viscous_u3  = (mu6·dudz6 - mu5·dudz5) / dx3(k)                    ! ∂(μ·∂u/∂z)/∂z
viscous_u12 = (mu4·dvdx4 - mu3·dvdx3) / dx2(j)                   ! ∂(μ·∂v/∂x)/∂y  (교차항)
viscous_u13 = (mu6·dwdx6 - mu5·dwdx5) / dx3(k)                   ! ∂(μ·∂w/∂x)/∂z  (교차항)

RHS_visc = invRhocCmu_half · (2·viscous_u1 + viscous_u2 + viscous_u3 + viscous_u12 + viscous_u13)
```

**② 압력 기울기:**
```fortran
RHS_press = -Cmp · invRhoc · (P(i,j,k) - P(im,j,k)) / dmx1(i)
```

**③ 부력항 (x-방향 U에만 적용, Boussinesq):**
```fortran
Tc = 0.5·(dx1(im)·T(i,j,k) + dx1(i)·T(im,j,k)) / dmx1(i)   ! 보간 온도
RHS_buoy = Cmt · (Tc + a12pera11·Tc²·ΔT) · invRhoc
```
> **주의:** x-방향 부력은 코드에 있으나, 실제 자연대류 문제에서는 y-방향(V-방정식)에 주 부력이 작용한다. V-방정식 코드에서 부력항이 주석 처리되어 있어 현재는 Boussinesq 부력을 U에만 적용 중임.

**④ 암묵 잔류 (Implicit Residual):**
```fortran
RHS = RHS - (mAPI·U(ip,j,k) + mACI·U(i,j,k) + mAMI·U(im,j,k)
           + mAPJ·U(i,jp,k) + mACJ·U(i,j,k) + mAMJ·U(i,jm,k)
           + mAPK·U(i,j,kp) + mACK·U(i,j,k) + mAMK·U(i,j,km))
```
이 잔류항은 3-방향 암묵 항의 합 - 즉, 분리 오차 없이 완전한 3D 암묵 시스템을 근사하기 위해 반드시 필요.

---

## 5. 압력 포아송 방정식 풀이

### 5.1 알고리즘 개요

속도 발산 조건에서 유도된 포아송 방정식:
```
∇·(∇p/ρ) = (1/Δt)·∇·u^*
```

**풀이 방법:** 2D FFT (x-z 방향) + 1D TDMA (y-방향)

```
1. RHS = ∇·u^* / Δt + NOB 보정항
2. x-방향 실수 FFT (r2c)
3. MPI All-to-All 통신으로 재배열
4. z-방향 복소수 FFT (c2c)
5. 파동수 공간에서 y-방향 삼중대각 시스템 풀이 (TDMA)
6. 역 FFT
7. 평균 압력 제거 + 경계 조건 갱신
```

### 5.2 y-방향 TDMA 계수 (파동수 공간)

```fortran
fft_amj = 1.0 / (dx2(j) · dmx2(j))
fft_apj = 1.0 / (dx2(j) · dmx2(jp))
fft_acj = -fft_amj - fft_apj

! 파동수 항 추가
Ac(k,i,j) = fft_acj - dxk1(i) - dzk(k)   ! 대각
Am(k,i,j) = fft_amj                        ! 하삼각
Ap(k,i,j) = fft_apj                        ! 상삼각
```

여기서:
- `dxk1(i) = 2·(1-cos(2π·i·Δx/L1))/(Δx²)`: x-방향 파동수 항
- `dzk(k)  = 2·(1-cos(2π·k·Δz/L3))/(Δz²)`: z-방향 파동수 항

---

## 6. 각 방정식에서 TDMA 시스템 요약

| 방정식 | 방향 | 경계 조건 | TDMA 유형 | 계수 특성 |
|--------|------|-----------|-----------|-----------|
| U-방정식 | Z | 주기 | `solve_cycle` | 점성+대류 계수 |
| U-방정식 | X | 주기 | `solve_cycle` | 점성+대류 계수 |
| U-방정식 | Y | 벽면(Dirichlet) | `solve` | 점성+대류, 벽면에서 off-diag=0 |
| V-방정식 | Z | 주기 | `solve_cycle` | 점성+대류 계수 |
| V-방정식 | X | 주기 | `solve_cycle` | 점성+대류 계수 |
| V-방정식 | Y | 벽면(Dirichlet) | `solve` | **주 확산 방향**, 계수 2배 |
| W-방정식 | Z | 주기 | `solve_cycle` | **주 확산 방향**, 계수 2배 |
| W-방정식 | X | 주기 | `solve_cycle` | 점성+대류 계수 |
| W-방정식 | Y | 벽면(Dirichlet) | `solve` | 점성+대류 계수 |
| T-방정식 | Z | 주기 | `solve_cycle` | 열확산+대류 계수 |
| T-방정식 | X | 주기 | `solve_cycle` | 열확산+대류 계수 |
| T-방정식 | Y | 벽면(Dirichlet) | `solve` | 열확산+대류 계수 |
| 압력(Poisson) | Y | 노이만 | `solve` | 파동수 공간 계수 |

---

## 7. 행렬 계수 정리

모든 모멘텀 방정식에서 각 방향 계수는 **확산 기여 + 대류 기여**로 구성된다.

### 7.1 일반 형태 (방향 d에 대하여)

격자 간격: `δ` (셀 중심간 거리), `Δ` (면간 거리), 유체 속도: `w_d`

```
a_p (상삼각) = -ν_half · μ_{d+1/2} / (Δ · δ_{+})  +  0.25 · w_{d+1/2} / δ_{+}
a_m (하삼각) = -ν_half · μ_{d-1/2} / (Δ · δ_{-})  -  0.25 · w_{d-1/2} / δ_{-}
a_c (대각)   = -(a_p + a_m) + [대류의 발산 기여]
```

여기서 `ν_half = 0.5 · Cmu · invRhoc`.

**대각 지배 조건:**  
확산항은 항상 `a_c > 0`, `a_p, a_m < 0`을 보장한다. 대류항은 `|대류 기여| < |확산 기여|`일 때 (낮은 셀 Pe 수) 대각 지배가 유지된다.

### 7.2 실제 계수 값 분석

주 방향(V-방정식의 y-방향, W-방정식의 z-방향)에서는 계수에 **2배** 인수 (`*2.0`)가 붙는다.  
이는 완전 응력 텐서 `∇·[μ(∇u + (∇u)ᵀ)]`에서 주 대각 성분 `2μ·∂²u_d/∂x_d²`의 정확한 처리.

---

## 8. ρ < 1/2 조건 분석

### 8.1 균등 격자 기준 계수 전개

U-방정식 z-방향 1단계(`module_solve_momentum.f90:389-413`)를 균등 격자 h로 단순화한다.  
`ν_h ≡ invRhocCmu_half · μ / h²` (절반 점성 계수) 로 정의하면:

```
mACK = +2ν_h  +  0.25·(w5 − w6)/h      ← 확산 + 대류 발산 (∇·u ≈ 0)
mAPK = −ν_h   +  0.25·w6/h
mAMK = −ν_h   −  0.25·w5/h
```

TDMA에 전달되는 최종 계수 (`ac = mACK·Δt + 1`, `ap = mAPK·Δt`, `am = mAMK·Δt`):

| 계수 | 확산 기여 | 대류 기여 | 부호 |
|------|-----------|-----------|------|
| `ac` | `+2ν_h·Δt + 1` | `+0.25·(w5−w6)/h·Δt ≈ 0` | 항상 양수 |
| `ap` | `−ν_h·Δt` | `+0.25·w6/h·Δt` | 가변 |
| `am` | `−ν_h·Δt` | `−0.25·w5/h·Δt` | 항상 음수 |

### 8.2 ρ 정의 및 조건 유도

`α ≡ ν_h·Δt` (확산수), `γ ≡ 0.25·|w|·Δt/h = CFL/4` 로 정의 (비압축: w5 ≈ w6 ≈ w).

```
|ac| = 2α + 1
|ap| = |α − γ|        (CFL < 4α 이면 확산 지배, 음수)
|am| = α + γ          (항상 양수)
```

ρ를 다음과 같이 정의:

```
ρ ≡ max(|am|/|ac|, |ap|/|ac|) = (α + γ) / (2α + 1)
```

**ρ < 1/2 조건 유도:**

```
(α + γ) / (2α + 1) < 1/2
  2α + 2γ < 2α + 1
        2γ < 1
         γ < 1/2
   CFL/4   < 1/2
   ──────────────
      CFL   < 2
```

### 8.3 결론

| 경우 | ρ 표현 | ρ < 1/2 조건 | PaScaL_TCS 성립 여부 |
|------|--------|-------------|----------------------|
| 순수 확산 (w=0) | `α/(2α+1)` → 1/2 점근 | **항상 성립** (등호 미달) | ✅ 무조건 |
| 대류 포함 | `(α+γ)/(2α+1)` | `CFL < 2` | ✅ `MaxCFL ≤ 1` 로 강제됨 |

- **순수 확산 한계:** α → ∞ 일 때 ρ → 1/2 이지만 절대 도달하지 않는다. 즉 Δt가 아무리 커도 확산만으로는 ρ < 1/2가 자동 보장된다.
- **대류 포함 시:** ρ < 1/2 ⟺ CFL < 2. PaScaL_TCS는 `MaxCFL` 파라미터(`PARA_INPUT.dat`)로 CFL < 1을 강제하므로, 정상 운용 조건에서 ρ < 1/2는 **항상 성립**한다.

### 8.4 x, y 방향 계수에 대한 동일 분석

x-방향(`mACI/mAPI/mAMI`)과 y-방향(`mACJ/mAPJ/mAMJ`)도 구조가 동일하므로 같은 결론이 성립한다.  
단, **V-방정식의 y-방향, W-방정식의 z-방향**은 주 확산 방향이므로 점성 계수에 2배 인수가 붙어 확산수 α가 2배 → ρ는 더 작아진다 (더 유리).

---

## 9. Filtered_TDMA 적용 가능성 분석

### 9.1 Filtered_TDMA란

Filtered_TDMA는 표준 Thomas 알고리즘(TDMA)에 고주파 필터링을 결합한 방법이다. 고 Rayleigh 수 자연대류에서는 격자 스케일 진동(wiggles)이 발생할 수 있으며, 이를 억제하기 위해 TDMA 해에 컴팩트 필터를 순차적으로 적용한다.

### 9.2 적용 대상 시스템

PaScaL_TCS의 각 방향 삼중대각 시스템은 구조적으로 Filtered_TDMA 적용에 **적합**하다.

**적용 가능한 이유:**

1. **ρ < 1/2 조건 충족:** 8절에서 분석한 바와 같이 `CFL < 2`이면 ρ < 1/2가 보장되며, PaScaL_TCS는 `MaxCFL ≤ 1`을 강제하므로 항상 만족된다.

2. **독립적인 1D 시스템:** 3단계 분리 구조에서 각 단계는 독립적인 1D 삼중대각 시스템이므로 필터를 각 방향에 독립적으로 적용 가능하다.

3. **경계 조건 처리:**
   - **주기 경계** (x, z 방향): `solve_cycle`에 해당하는 Filtered 버전 구현 필요
   - **벽면 경계** (y 방향): `solve`에 해당하는 Filtered 버전 구현 필요, 경계 근방에서 필터 계수 조정 필요

4. **병렬화 호환:** PaScaL_TDMA는 이미 MPI 병렬화된 TDMA이므로, Filtered_TDMA도 동일한 병렬 파이프라인에 통합 가능하다.

### 9.3 적용 우선순위

| 우선순위 | 방향/방정식 | ρ 특성 | 이유 |
|----------|-------------|--------|------|
| **높음** | Y-방향 모멘텀 (V) | 확산 2배 → ρ 더 작음 | 주 대류 방향, 격자 스케일 진동 발생 빈도 최고 |
| **높음** | Y-방향 열 방정식 | 동일 구조 | 주 온도 기울기 방향 |
| **중간** | X, Z-방향 모멘텀 | 표준 ρ | 주기 경계, 큰 Ra에서 진동 가능 |
| **낮음** | 압력 포아송 Y-방향 | 파동수 공간, 구조 다름 | 필터 효과 불명확 |

### 9.4 적용 시 주의사항

1. **ρ 마진:** `α → ∞` 극한에서 ρ → 1/2 점근. 큰 Δt·ν_h 조합(고점도·큰 시간스텝)에서 ρ가 1/2에 가까워지므로 필터 계수 α_f 선택 시 여유 확보 필요.

2. **온도 의존 물성치:** NOB 계산에서 `μ(T)`, `κ(T)`, `ρ(T)`가 변하므로 매 시간 스텝 ρ가 달라진다. 필터 계수도 매 스텝 갱신이 필요하다.

3. **주 방향 계수 2배:** V-방정식의 y-방향, W-방정식의 z-방향에서 점성 계수가 2배 → 동일 조건에서 ρ가 더 작아 필터 안정성이 높다.

4. **경계 처리:** 벽면 경계에서 `mAPJ = mAPJ*dble(jup)` 처리로 경계 인접 행의 상삼각/하삼각 성분이 소거된다. 필터 적용 시 해당 인덱스에서 예외 처리 필요.

5. **V-방정식의 dU 의존성:** V-방정식 RHS에 `dU` 항이 포함되어 있어 U-방정식을 먼저 풀어야 한다. 시스템 간 순서 의존성은 변하지 않는다.

### 9.5 현재 PaScaL_TDMA 사용 인터페이스

```fortran
! 주기 경계 (x, z)
call PaScaL_TDMA_many_solve_cycle(ptdma_plan, am, ac, ap, RHS, nsys, n)

! 벽면 경계 (y)
call PaScaL_TDMA_many_solve(ptdma_plan, am, ac, ap, RHS, nsys, n)
```

Filtered_TDMA로 교체 시 동일한 인터페이스를 유지하되 내부에서 필터 단계를 추가하는 형태가 가장 현실적이다.

---

## 10. 결론

PaScaL_TCS는 3단계 ADI 분리법으로 모멘텀 방정식을 방향별 독립 삼중대각 시스템으로 분해하여 PaScaL_TDMA로 풀어낸다.

각 방향 계수의 구조는 다음과 같이 정리된다:

```
ac = (2ν_h + ∇·w 기여) · Δt + 1      → 항상 양수
ap = (−ν_h + 0.25·w_{+}/h) · Δt      → 부호 가변
am = (−ν_h − 0.25·w_{−}/h) · Δt      → 항상 음수
```

**ρ < 1/2 여부:** `ρ = (α + CFL/4) / (2α + 1)` 로 표현되며, `CFL < 2` 조건과 동치다. PaScaL_TCS는 `MaxCFL ≤ 1`을 강제하므로 **모든 방향·모든 방정식에서 ρ < 1/2가 항상 성립**하며 Filtered_TDMA 적용 조건을 만족한다.

특히 고 Rayleigh 수 조건에서 y-방향 모멘텀(V) 및 열 방정식의 삼중대각 시스템에 Filtered_TDMA를 우선 적용하면 격자 스케일 비물리 진동을 억제하면서도 기존 MPI 병렬 인프라를 그대로 활용할 수 있다.

---

## 부록: 관련 소스 파일 위치

| 파일 | 역할 | 핵심 서브루틴 |
|------|------|---------------|
| `src/module_solve_momentum.f90` | 모멘텀 풀이 | `mpi_momentum_solvedU/V/W` |
| `src/module_solve_pressure.f90` | 압력 포아송 | `mpi_pressure_Poisson_FFT1/2`, `mpi_pressure_Projection` |
| `src/module_solve_thermal.f90` | 열 방정식 | `mpi_thermal_solver` |
| `src/module_global.f90` | 전역 파라미터 | `global_inputpara` |
| `PaScaL_TDMA/src/pascal_tdma.f90` | TDMA 래퍼 | `PaScaL_TDMA_many_solve`, `_solve_cycle` |
| `PaScaL_TDMA/src/tdmas.f90` | Thomas 알고리즘 | `tdma_many`, `tdma_many_cycle` |
