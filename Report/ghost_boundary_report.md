# Ghost-cell 경계 처리로 전 영역 ρ < 1/2 만들기 — Heat ADI 솔버 분석/검증 리포트

작성일: 2026-06-22 · 대상: `/scratch/x3319a05/Filtered_TDMA`

---

## 0. 요약 (TL;DR)

- **목표**: cell-center 격자에서 경계값을 격자점으로 포함시키던(half-cell, 경계 위 노드)
  기존 방식을 버리고, **경계를 ghost cell로 처리**하여 모든 행(boundary 인접 행 포함)의
  정규화 off-diagonal ρ가 입력 rho(=0.25)와 같아지도록(= 전 영역 **ρ < 1/2**) 바꾼다.
- **결과 (CPU)**: heat_cpu는 working tree에 이미 ghost-cell 방식이 들어와 있었고, 이를
  **검증**한 결과 `pascal`, `filtered`(v1), `filtered_v2` 세 백엔드 모두 **공간 2차 수렴**
  (asymptotic ratio → 4.0, order → 2.0), 그리고 모든 행에서 **ρ = 0.25 < 1/2** 확인.
- **핵심 이론**: ghost-cell 균일 stencil에서는 정규화 off-diagonal이 **모든 행에서 정확히 rho**.
  기존 방식의 벽 인접 행은 `(8/3)·rho / (1+2·rho)` 로, **rho > 0.3 이면 1/2 초과**
  (rho→1/2 에서 2/3) → Filtered truncation이 벽 근처에서 무효화됨. ghost 방식이 이를 해소.
- **GPU**: heat_gpu도 동일하게 변환 **완료** + **V100 검증 완료**(§7) — pascal/filtered/filtered_v2
  모두 2차(order 1.996), ρ=0.25, **CPU와 L2 자릿수까지 일치**.

---

## 1. 프로젝트 구조

모놀레포. 두 종류의 분산 TDMA 라이브러리 + 이를 동일 인터페이스로 호출하는 ADI 솔버.

```
Filtered_TDMA/
├── Makefile / Makefile.inc        # 최상위 오케스트레이션, 컴파일 플래그(USE_CUDA, CUDA_ARCH)
├── libs/
│   ├── filtered_tdma/             # libfiltered_tdma.a
│   │   ├── filtered_tdma.{cpp,hpp}        # CPU: solve_filtered_v1 / v2 (truncated 전파)
│   │   ├── filtered_tdma_profile.cpp      # per-phase timing 변형
│   │   └── filtered_tdma_cuda.{cu,hpp}    # GPU FilteredTDMACUDA
│   └── pascal_tdma/               # libpascal_tdma.a
│       ├── pascal_tdma_many.{cpp,hpp}     # alltoallv reduced-system
│       ├── tdma_local.{cpp,hpp}           # 직렬 Thomas (nprocs==1), tdma_many/single
│       └── *_cuda.{cu,cuh}                # GPU 커널
├── apps/
│   ├── heat_cpu/                  # ← 본 리포트의 1차 대상 (CPU ADI heat)
│   ├── heat_gpu/                  # GPU ADI heat (heat_cpu 미러)
│   └── channel/                   # 채널 유동 (FFTW + TDMA)
├── results/ , Report/ , scripts/
└── build/                         # lib/ bin/ include/ obj/
```

### 1.1 두 TDMA 백엔드

| | **FilteredTDMA** | **PaScaLTDMAMany** |
|---|---|---|
| 통신 | 인접 rank와 face 2회 교환 (Isend/Irecv) | 전 rank alltoallv |
| reduced 풀이 | 양 끝 J행만 truncated 전파 (`eps` cutoff) | `2·nprocs`행 정확 풀이 |
| 추가 입력 | `eps_constant`(=dt), per-row `A_rho`/`C_rho` | 없음 |
| 변형 | `filtered`(=v1, 해 기반 J), `filtered_v2`(보수적 RHS-bound J, **기본값**) | — |

`apps/heat_cpu/tdma_backend.{hpp,cpp}` 가 `tdma_backend` 입력으로 런타임 디스패치
(`pascal` | `filtered`(=v1) | `filtered_v2`). FilteredTDMA만 `set_rho()`(행별 ρ 갱신)와
`set_eps_constant(dt)`를 사용.

### 1.2 Heat manufactured solution

- PDE: ∂ₜθ = ∇²θ + f, `f = 3π²·cos(πx)cos(πy)cos(πz)`
- Exact: `θ = sin(πx)sin(πy)sin(πz)·exp(-3π²t) + cos(πx)cos(πy)cos(πz)`, 도메인 `[-1,1]³`
- 시간 적분: factored Crank–Nicolson ADI (Z→Y→X). 각 방향 implicit 행렬 `[-a, 1-b, -c]`,
  `a=c=base`, `b=-2·base`, `base = dt/(2·dx²)`.
- `option=order`: `dt = rho/(1-2rho)·2dx²`, `Tmax = dt_512·128` 고정 →
  모든 N에서 T_final 동일 → **공간 수렴 차수만** 측정.

  | N | dt(=dx², rho=0.25) | Nt | T_final |
  |---|---|---|---|
  | 64 | 9.77e-04 | 2 | 1.953e-03 |
  | 128 | 2.44e-04 | 8 | 〃 |
  | 256 | 6.10e-05 | 32 | 〃 |
  | 512 | 1.53e-05 | 128 | 〃 |

---

## 2. 경계 처리 — 무엇이 문제였나

### 2.1 기존(舊) 방식: "경계값을 격자에 포함" (= HEAD 커밋 상태)

`mesh()`가 양 끝 노드를 **경계 위에** 둠:

```cpp
if      (rankx==0     && i==0)      x_sub[i] = x0;     // 경계 노드 = 벽 위
else if (rankx==npx-1 && i==nx_sub) x_sub[i] = xN;
else                                x_sub[i] = x0 + dx/2 + (ista-2+i)*dx;  // cell center
```

⇒ 좌표열이 `x = [-1, -1+dx/2, -1+3dx/2, …]` 형태(사용자가 말한 `[0, 0.5, 1.5, 2.5, …]`).
경계 노드와 첫 셀 중심 사이만 **간격 dx/2**(half cell), 내부는 dx로 **비균일**.
이 half-cell을 2차로 맞추려고 벽 인접 행에만 보정 stencil을 씀(`stencil_coeffs.hpp`):

```
a = base·(1 + 5/3·lb + 1/3·rb)
b = base·(-2 - 2·lb  - 2·rb )
c = base·(1 + 1/3·lb + 5/3·rb)
```

그리고 ADI cross-derivative 보정항(`Z/Y/X-boundary correction`)을 RHS에 더했음.

### 2.2 왜 ρ ≥ 1/2 문제가 생기나 (핵심)

Filtered TDMA의 truncation은 행별 **정규화 off-diagonal** ρ_n = |a/B|, |c/B| 이 작아야
(geometric decay) 성립한다. cutoff 인덱스 `J = ⌊log(eps/·)/log(q)⌋+1`, `q = ρ_n/λ₊`,
`λ₊ = (1+√(1-4ρ_n²))/2`. 즉

- **ρ_n < 1/2** ⇒ q < 1 ⇒ 지수 감쇠 ⇒ 유한 J로 절단 가능 (Filtered가 의미 있음).
- **ρ_n = 1/2** ⇒ q = 1 ⇒ 감쇠 없음.
- **ρ_n > 1/2** ⇒ 복소근, 감쇠 없음 ⇒ **절단 불가/무효**.

기존 방식 벽 인접 행(lb=1)은 `B = 1 - b = 1 + 4·base`, `a = (8/3)·base` 이므로
`base = rho/(1-2rho)` 를 대입하면

```
ρ_n(벽쪽)  = |a/B| = (8/3)·base/(1+4·base) = (8/3)·rho / (1 + 2·rho)
```

| rho(입력) | 내부 행 ρ_n | 기존 벽 인접 행 ρ_n |
|---:|---:|---:|
| 0.10 | 0.10 | 0.222 |
| 0.25 | 0.25 | **0.444** |
| 0.30 | 0.30 | **0.500** ← 임계 |
| 0.40 | 0.40 | **0.593** |
| 0.49 | 0.49 | **0.659** |

⇒ **rho > 0.3 이면 벽 인접 행의 ρ_n > 1/2** 가 되어 Filtered truncation이 벽 근처에서 깨진다
(rho=0.25에서도 0.444로 비대칭·임계 근접). 사용자의 요구 "전 계산 영역에서 ρ < 1/2"가
기존 방식으로는 보장되지 않는 이유.

---

## 3. 새 방식: Ghost-cell Dirichlet

### 3.1 격자 — 완전 균일 cell-center + ghost

`mesh()`에서 끝 노드 특수처리를 제거, **모든 셀을 균일 cell center**로:

```cpp
x_sub[i] = x0 + dx/2 + (ista-2+i)·dx;   // i=0..nx_sub 전부 동일 식
```

⇒ 내부 셀 i=1..nx_sub-1, ghost는 i=0 (`x0 - dx/2`), i=nx_sub (`xN + dx/2`).
물리 경계(벽)는 ghost와 첫 내부 셀의 **정확히 중점**(x0, xN)에 위치 → 간격 dx **완전 균일**.

### 3.2 Dirichlet를 ghost로 (2차 선형 외삽)

벽 위 값 u_BC를 ghost와 인접 내부 셀의 평균으로 강제:

```
(u_ghost + u_inner)/2 = u_BC   ⇒   u_ghost = 2·u_BC − u_inner
```

본 문제의 정상(steady) 경계값: x=±1 면에서 sin(±π)=0 이라
`u_BC = -cos(πy)cos(πz)` (시간 무관). `boundary()`가 매 step 6면 ghost를 이 식으로 채움.

### 3.3 균일 stencil + implicit ghost 보정

stencil은 전 행 동일: `{a,b,c} = {base, -2·base, base}`.
RHS 커널은 ghost가 채워진 theta를 그대로 읽어 **explicit 절반**을 처리(별도 boundary 보정항 불필요).
각 방향 implicit 풀이에서 알려진 ghost 값을 **RHS로 이항**:

```cpp
// Z-solve, 벽 소유 rank만:
D[kk=0]      += sz.a · theta[k=0    ghost];   // 좌벽
D[kk=iz-1]   += sz.c · theta[k=nz-1 ghost];   // 우벽
```

여기서 `Azz[0] = -sz.a`, `Czz[last] = -sz.c` 를 그대로 두어도 안전한 이유:
직렬 Thomas(`tdma_many`)의 forward-elim은 `A[j]`(j≥1)만, back-subst는 `C[j]`(j≤n-2)만
사용 → 첫 행의 `A[0]`, 마지막 행의 `C[n-1]`은 미사용. 분산 백엔드도 같은 글로벌 경계 규약.

### 3.4 결과: 모든 행에서 ρ_n = rho

균일 행 `B = 1+2·base`, `a = base` 이므로

```
ρ_n = |a/B| = base/(1+2·base) = [rho/(1-2rho)] / [1/(1-2rho)] = rho   (정확히)
```

⇒ 벽 인접 행 포함 **전 영역 ρ_n = 0.25 < 1/2**. (코드의 `[rho]` 출력이 이 값.)
Filtered가 전 영역에서 동일·유효한 truncation을 수행.

---

## 4. CPU 검증 결과 (본 세션, 2026-06-22)

빌드: `module load gcc/15.2.0 mpi/openmpi-4.1.8; make heat`. 드라이버:
`Report/run_order_cpu.sh "NPX NPY NPZ" "backends" "nx-list"`.
`option=order`, rho=0.25, domain `[-1,1]³`, L2 = √(Σ(θ−exact)²/N³) (T_final 고정).

### 4.1 단일 랭크 (np=1)

| backend | nx=64 | 128 | 256 | ratio(64→128, 128→256) | order |
|---|---:|---:|---:|---|---|
| pascal | 1.24416e-04 | 3.43421e-05 | 8.61189e-06 | 3.623, 3.988 | +1.86 → **+2.00** |
| filtered (v1) | 동일 | 동일 | 동일 | 동일 | 동일 |

`[rho] = 0.25` (전 행 균일).

### 4.2 분산 (np = 2×2×2 = 8) — 모든 백엔드

| backend | nx=64 | 128 | 256 | 512 | ratio (asymp) | order |
|---|---:|---:|---:|---:|---|---|
| `pascal`      | 1.24416e-04 | 3.43421e-05 | 8.61189e-06 | 2.16478e-06 | 3.62→3.99→**3.98** | → **+1.99** |
| `filtered`(v1)| 1.24418e-04 | 3.43425e-05 | 8.61194e-06 | 2.16478e-06 | 3.62→3.99→**3.98** | → **+1.99** |
| `filtered_v2` | 1.24417e-04 | 3.43422e-05 | 8.61190e-06 | 2.16478e-06 | 3.62→3.99→**3.98** | → **+1.99** |

- **세 백엔드 모두 공간 2차 수렴 확인** (asymptotic ratio 3.98 ≈ 4.0).
- 세 백엔드 **상호 일치**(6 유효숫자 내, 분산 truncation 차이는 ~1e-6 상대). ρ<1/2 이므로
  Filtered truncation이 정상 동작하고도 PaScaL(정확 풀이)과 동일 정확도.
- nx=64의 첫 ratio 3.62는 Nt=2(coarse·pre-asymptotic) 때문이며, 격자 정제와 함께 4.0으로 수렴.

> 비고: 기존(舊) 방식 대비 같은 격자에서 절대 오차 상수는 약간 큼(예 nx=64 pascal:
> 舊 4.21e-05 vs ghost 1.24e-04). 이는 균일·대칭 ghost 외삽이 벽 근처에서 약간 다른(그러나
> 균일하고 ρ<1/2를 보장하는) 오차 상수를 갖기 때문이며, **수렴 차수는 동일하게 2차**. 사용자의
> 목표(전 영역 ρ<1/2 + 2차)와 정확히 맞교환됨.

---

## 5. 변경 파일 (heat_cpu, working tree)

| 파일 | 변경 |
|---|---|
| `stencil_coeffs.hpp` | `compute_stencil(dt,dd)` 균일판으로 단순화 ((5/3,1/3) 보정 제거) |
| `mpi_subdomain.cpp::mesh()` | 끝 노드 특수처리 제거 → 전부 균일 cell center (ghost 포함) |
| `mpi_subdomain.cpp::boundary()` | 면값 저장 → **ghost 채우기** `u_ghost = 2·u_BC − u_inner` (6면) |
| `solve_theta.cpp` | RHS 균일 stencil; 舊 cross-correction 블록 삭제; 각 방향 **implicit ghost 보정**(D로 이항) |
| `solve_theta_profile.cpp` | 동일 패턴(타이밍판) |

---

## 6. 재현 방법

```bash
cd /scratch/x3319a05/Filtered_TDMA
module purge && module load gcc/15.2.0 mpi/openmpi-4.1.8
rm -f build/obj/ht_*.o build/bin/heat.out && make heat

# np=1, pascal/filtered, nx=64/128/256
bash Report/run_order_cpu.sh "1 1 1" "pascal filtered" "64 128 256"
# np=8, 세 백엔드, nx=64..512
bash Report/run_order_cpu.sh "2 2 2" "pascal filtered filtered_v2" "64 128 256 512"
```

각 run 의 `[rho] = 0.25` 출력으로 전 영역 ρ<1/2 확인, `Global L2 error =` 로 차수 측정.

---

## 7. GPU (heat_gpu) — 변환 완료 + V100 검증

heat_gpu 도 CPU와 동일한 ghost-cell 방식으로 **변환 완료**. 적용한 변경:

| 파일 | 변경 |
|---|---|
| `stencil_coeffs.hpp` + `solve_theta.cu::d_stencil` | (5/3,1/3) 보정 제거 → 균일 `{base,-2base,base}` |
| `mpi_subdomain.cpp::mesh()` | 끝 노드 특수처리 제거 → 균일 cell center (ghost 포함) |
| `mpi_subdomain.cpp::boundary()` | 면값 저장 → ghost 채우기 `2·u_BC − u_inner` (초기 H2D용, CPU와 동일) |
| `solve_theta.cu` **신규** `ghost_x/y/z_kernel` | 매 step RHS 전 d_theta 6면 물리 ghost를 `2·u_BC−u_inner`로 갱신 (comm 단계에서 halo 교환 직후) |
| `solve_theta.cu::rhs_kernel` | 균일 stencil (ghost는 d_theta에서 그대로 읽음 = explicit 절반) |
| `solve_theta.cu` 舊 `z/y/x_boundary_kernel` | **제거** |
| `solve_theta.cu::build_lhs_z/y/x_kernel` | 균일 stencil + **implicit ghost 보정 fold-in**: 벽 소유 경계 행 `d_D`에 `sa·θ_ghost`(좌)·`sc·θ_ghost`(우) 가산 (d_theta 인자 추가) |

CN 정합성: explicit 절반(rhs_kernel)은 ghost를 old-time 기여로, implicit 절반(build_lhs)은
new-time 기여로 사용 → **double-count 아님** (정상 BC라 두 시각의 ghost 값 동일).

빌드: `module load nvhpc/25.11_cuda12; export CUDA_LIBDIR=$NVHPC_ROOT/cuda/lib64;
USE_CUDA=1 CUDA_ARCH=70 make heat_gpu` → 에러 없이 `build/bin/heat_gpu.out` 생성.

### 7.1 V100 검증 결과 (gpu03, Tesla V100-SXM2-32GB, np=1, job 785880)

| backend | nx=64 | 128 | 256 | ratio(asymp) | order | rho |
|---|---:|---:|---:|---:|---:|---:|
| `pascal`      | 1.24416e-04 | 3.43421e-05 | 8.61189e-06 | 3.988 | **1.996** | 0.25 |
| `filtered`(v1)| 1.24416e-04 | 3.43421e-05 | 8.61189e-06 | 3.988 | **1.996** | 0.25 |
| `filtered_v2` | 1.24416e-04 | 3.43421e-05 | 8.61189e-06 | 3.988 | **1.996** | 0.25 |

- **세 백엔드 모두 GPU에서 공간 2차 수렴 확인**, ρ=0.25 전 영역.
- GPU L2 값이 **CPU(np=1)와 자릿수까지 완전 일치** (1.24416e-04 / 3.43421e-05 / 8.61189e-06)
  → CPU·GPU 구현이 동일 이산화를 정확히 재현.

재현: `sbatch Report/sbatch_order_gpu_v100nv8.sh` (또는 `_v100.sh` = cas_v100_2,
`_v100_4.sh` = cas_v100_4). 멀티-GPU(NP>1) 분산 검증은 후속(동일 코드 경로, CPU에서 이미
np=2×2×2 2차 확인됨).
```

---

## 8. FilteredTDMA reduced-system 통신 병합 (2 round → 1 round)

작성일: 2026-06-23. 대상: `libs/filtered_tdma/filtered_tdma.{cpp,hpp}` (CPU),
`libs/filtered_tdma/filtered_tdma_cuda.{cu,hpp}` (GPU), v1·v2 공통.

### 8.1 핵심 관찰 — reduced system이 interface별 독립 2×2로 분해됨

로컬 forward/backward 소거 + pack 후, rank별 reduced 미지수는 `x0`(첫 셀)·`xN`(끝 셀).
pack이 row0의 `C0`(→자기 xN) 결합을 제거하고, 마지막 행은 자기 `x0` 결합이 없어서, 전역
reduced 행렬이 **인터페이스별 독립 2×2 블록**으로 분해된다:

```
... [ xN^{r-1}, x0^r ] ...   ← 인터페이스(r-1,r) 2×2 (강결합 ~ρ)
```
rank 내부 `x0^r ↔ xN^r` 결합은 `q^{n_row}`로 사실상 0.

### 8.2 기존(2 round) vs 병합(1 round)

- **기존**: 각 인터페이스를 **오른쪽 rank가** 풀고(자기 row0 + 받은 왼쪽 rowN), 푼 값을
  **다시 왼쪽으로 통신**. → forward 1 + back-communication 1 = **통신 2 round**,
  rank당 reduced 풀이 = n_sys.
- **병합**: 한 번의 양방향 교환으로 **왼쪽 rowN + 오른쪽 row0**을 동시에 받아, 각 rank가
  **자기 양쪽 인터페이스 2×2를 직접** 푼다. → **통신 1 round**, rank당 풀이 = 2·n_sys(중복).
  - LEFT : `x0 = (D0 − A0·D_left)/(1 − C_left·A0)`
  - RIGHT: `xN = (DN − CN·D0_right)/(1 − A_right·CN)`

**대수적으로 완전 동일** (검증): 기존
`DN = D_N − C_N·(D0' − A0'·D_N)/(1−C_N·A0') = (D_N − C_N·D0')/(1−A0'·C_N)` = 병합 RIGHT 공식.

### 8.3 속도 관점

- 통신 **round 절반**(2α→α): 작은 boundary-row 메시지라 **latency 지배** → PCIe V100에서 이득.
- GPU: `cudaStreamSynchronize`+MPI 시퀀스가 **2→1**로 줄어 GPU idle bubble 하나 제거.
- 비용: reduced 풀이 2·n_sys vs n_sys는 column당 2×2라 **무시 가능**(중복 계산이지만 µs급).
- 통신 volume은 거의 동일(같은 boundary row를 동시 송수신). row0(A,D)은 연속이라 직접 송신,
  rowN(C,D)은 기존 pack 버퍼 재사용(GPU CUDA-aware fast path 유지).

### 8.4 검증 결과 (2026-06-23, option=order, rho=0.25)

모두 **변경 전 2-round 값과 일치 + 2차 수렴** → 동치 확인.

**CPU np=2×2×2** (`bash Report/run_order_cpu.sh "2 2 2" "pascal filtered filtered_v2" "64 128 256 512"`):

| backend | 64 | 128 | 256 | 512 | order |
|---|---:|---:|---:|---:|---:|
| pascal | 1.24416e-4 | 3.43421e-5 | 8.61189e-6 | 2.16478e-6 | 1.99 |
| filtered(v1) | 1.24418e-4 | 3.43425e-5 | 8.61194e-6 | 2.16478e-6 | 1.99 |
| filtered_v2 | 1.24417e-4 | 3.43422e-5 | 8.61190e-6 | 2.16478e-6 | 1.99 |

**GPU 2×V100** (cas_v100nv_8, npz=2, `sbatch Report/sbatch_order_gpu2_v100nv8.sh`, job 785915):

| backend | 64 | 128 | 256 | order |
|---|---:|---:|---:|---:|
| pascal | 1.24416e-4 | 3.43421e-5 | 8.61189e-6 | 1.996 |
| filtered(v1) | 1.24416e-4 | 3.43422e-5 | 8.61190e-6 | 1.996 |
| filtered_v2 | 1.24416e-4 | 3.43422e-5 | 8.61190e-6 | 1.996 |

> 구현 메모: 우측 이웃 row0 수신 버퍼 `A_right_recv_`/`D0_right_recv_`(GPU `d_*`) 추가,
> 통신을 8-request 단일 `Waitall`로 병합, 양쪽 2×2를 푸는 `k_solve_both`(GPU) 커널 추가.
> 구 `k_solve_D0_left`/`k_unpack_DN`·`D_right_send/recv` 버퍼는 미사용(잔존, 후속 제거 가능).
