# Filtered_TDMA — 분산 TDMA 라이브러리 2종 + ADI Heat / Channel 솔버 (CPU + GPU)

> 이 문서는 사람과 **다른 AI 에이전트**가 이 저장소 전체를 빠르게 파악하도록 쓴
> 오리엔테이션 문서입니다. "무엇을/왜/어떻게"를 위에서 아래로, 핵심 아이디어 →
> 구조 → 알고리즘 → 빌드/실행 → 결과 → 함정 → 파일 지도 순으로 정리합니다.

---

## 0. 한 문단 요약

분산-메모리(MPI) 환경에서 **많은 삼중대각(tridiagonal) 시스템을 동시에 푸는** 두
가지 라이브러리 — `PaScaLTDMAMany`(정확해)와 `FilteredTDMA`(절단 근사) — 와, 이
둘을 **같은 인터페이스로 런타임 교체**하며 호출하는 ADI 응용 솔버(3D heat,
turbulent channel)를 한 곳에 담은 모놀레포입니다. **CPU(MPI)** 와 **GPU(CUDA +
CUDA-aware MPI)** 빌드를 모두 지원하며, 입력 파일의 `tdma_backend` 한 줄로
두 알고리즘을 바꿔 성능·정확도를 직접 비교할 수 있습니다.

핵심 질문은 하나입니다: **"병렬 삼중대각 풀이의 전역 통신을, 정확도를 거의
잃지 않으면서 근접-이웃 통신 + 국소 절단으로 바꿀 수 있는가?"** — `FilteredTDMA`
가 그 시도이고, `PaScaLTDMAMany`가 정확해 기준선입니다.

---

## 1. 핵심 아이디어 (왜 이 프로젝트가 존재하나)

### 1.1 병렬 TDMA의 본질적 어려움

삼중대각 풀이(Thomas 알고리즘)는 **행 방향으로 순차 의존성**이 있어 본질적으로
직렬입니다. N개의 행을 P개의 rank에 분할(z-slab)하면 각 rank는 자기 블록을
국소적으로 소거(**modified Thomas**)할 수 있지만, 그 결과 **인접 rank의 경계
행(row 0, row N-1)들끼리만 결합된 "reduced system"**(크기 2P)이 남습니다. 이
reduced system을 어떻게 푸느냐가 두 라이브러리를 가릅니다.

### 1.2 두 가지 전략

```
        분할(z-slab) + 국소 modified-Thomas  →  2P개 경계 미지수의 reduced system
                                                 │
              ┌──────────────────────────────────┴───────────────────────────────┐
   PaScaL: reduced system을 정확히 풀이                Filtered: reduced system을 근사
   - 전역 all-to-all(transpose)로 2P행을 모음          - 결합이 기하급수적으로 감쇠함을 이용
   - 2P 크기 Thomas로 정확해                            - 근접-이웃 교환 + J행 절단
   - 직렬 Thomas와 bit-identical                        - 통신/연산 ↓, 오차 ≤ eps
```

### 1.3 Filtered의 통찰 — 기하급수적 감쇠 → 절단

대각우세(diagonally-dominant) 삼중대각에서는, 한 경계 미지수가 이웃 블록 **안쪽
깊은 행에 미치는 영향이 비율 `q = rho/λ₊` (q<1) 로 기하급수적으로 감쇠**합니다.
(`rho` = 정규화 비대각 = 대각우세도, `λ₊ = (1+√(1−4·rho²))/2`.) 여기서 두 가지가
따라옵니다:

1. **reduced system이 인접 2×2 블록으로 사실상 분리** → 전역 all-to-all 대신
   **근접-이웃 1회 교환**이면 충분 (각 rank가 자기 양쪽 인터페이스를 국소적으로
   풀이). → `filtered_tdma.cpp`의 "single bidirectional exchange" (2라운드 통신을
   1라운드로 병합).
2. **경계 보정이 tolerance `eps` 아래로 떨어지기까지 `J`행만 전파하면 됨**
   → 보정(그리고 v2에서는 소거까지)을 `O(N)`이 아니라 `O(J)`로 절단.

   ```
   J = ⌊ log(eps_ / B) / log(q) ⌋ + 1      (상한 n_row−1)
   eps_ = eps_constant / N_global²          (heat에서 매 step eps_constant = dt)
   B    = 경계 데이터(D0, DN, 또는 max|D|)에서 만든 보수적 상한
   ```

   `J`는 격자 크기와 (거의) 무관한 작은 상수(보통 한 자리~십몇)이며 `N_global`에
   대해 **로그적으로만** 증가합니다. `rho ≥ 0.5`(대각우세 깨짐)면 절단을 끄고
   full solve로 폴백합니다.

**트레이드오프**: Filtered는 *전역 all-to-all*을 *근접-이웃 통신 + 절단 국소
연산*으로 바꾸되, ≤ `eps`의 통제된 절단 오차를 감수합니다. 암시적 확산처럼
대각우세가 강한 문제에서는 `J`가 매우 작아 오차가 무시 가능 — 실제로 heat
manufactured solution에서 **PaScaL과 자릿수까지 동일**합니다(§6).

---

## 2. 3계층 아키텍처

```
  [응용 계층]   apps/heat_cpu  apps/heat_gpu  apps/channel_cpu  apps/channel_gpu
                     │  ADI Z→Y→X sweep, 매 방향마다 "한 방향 = 많은 삼중대각계"
                     ▼
  [디스패치]    tdma_backend(.cpp/.hpp)  ·  tdma_backend_gpu(.cu/.hpp)
                     │  "pascal" | "filtered"(=v1) | "filtered_v2"  ← 런타임 문자열
                     ▼
  [솔버 계층]   libfiltered_tdma.a        libpascal_tdma.a
                FilteredTDMA(CUDA)        PaScaLTDMAMany(CUDA)
                     └──── 공통: tdma_local(Thomas), para_range(분할) ────┘
```

- **솔버 계층**은 응용을 모릅니다 — `solve(A,B,C,D, n_sys, n_row)` 한 인터페이스로
  "`n_row` 길이의 삼중대각계 `n_sys`개"를 풉니다.
- **응용 계층**은 알고리즘을 모릅니다 — `TdmaBackend`/`TdmaBackendGPU`가 enum으로
  실제 솔버를 골라줍니다. 그래서 **재컴파일 없이** 입력 한 줄로 백엔드 교체.
- 같은 소스가 `USE_CUDA=1`이면 `.cu`까지 묶어 GPU 바이너리를, 아니면 CPU
  바이너리를 만듭니다.

---

## 3. 저장소 레이아웃

```
Filtered_TDMA/
├── Makefile, Makefile.inc                # 최상위 오케스트레이션 + 컴파일러/CUDA 플래그
├── README.md                             # (이 문서)
├── libs/
│   ├── filtered_tdma/                     → build/lib/libfiltered_tdma.a
│   │   ├── filtered_tdma.{cpp,hpp}        #   CPU: solve_filtered_v1 / _v2, cal_J_*
│   │   ├── filtered_tdma_cycl.cpp         #   주기(cyclic) 경계 변형
│   │   ├── filtered_tdma_profile.cpp      #   per-phase 타이밍 변형
│   │   └── filtered_tdma_cuda.{cu,hpp}    #   GPU FilteredTDMACUDA (USE_CUDA=1)
│   └── pascal_tdma/                       → build/lib/libpascal_tdma.a
│       ├── pascal_tdma_many.{cpp,hpp}     #   CPU: reduced system + alltoall(w) transpose
│       ├── pascal_tdma_single.{cpp,hpp}   #   단일 시스템 변형
│       ├── tdma_local.{cpp,hpp}           #   직렬 Thomas (nprocs==1 폴백, 공통)
│       ├── para_range.{cpp,hpp}           #   블록 인덱스 분할 (공통)
│       ├── pascal_tdma_many_cuda.{cu,hpp} #   GPU PaScaLTDMAManyCUDA
│       ├── tdma_local_cuda.{cu,cuh}       #   GPU Thomas / modified-Thomas 커널
│       └── nvtx_util.hpp                  #   NVTX 프로파일 매크로
├── apps/
│   ├── heat_cpu/                          → build/bin/heat.out   (참조 벤치마크)
│   │   ├── main.cpp                       #   드라이버 + L2 error (vs manufactured)
│   │   ├── solve_theta.cpp                #   ADI Z→Y→X sweep 루프
│   │   ├── tdma_backend.{cpp,hpp}         #   "filtered"|"pascal" 디스패처(host)
│   │   ├── global.{cpp,hpp}               #   입력 파싱 + option(dt/Tmax/Nt) 정책
│   │   ├── mpi_topology / mpi_subdomain   #   3D Cartesian + z-slab + ghost-cell
│   │   └── inputs/{PARA_INPUT_*,scaling/,spatial/}
│   ├── heat_gpu/                          → build/bin/heat_gpu.out  (heat_cpu의 CUDA 미러)
│   │   ├── solve_theta.cu                 #   커널: rhs/ghost/build_lhs/update_theta
│   │   ├── ghostcell_cuda.cu              #   ghost-cell pack/unpack 커널
│   │   ├── tdma_backend_gpu.{cu,hpp}      #   디스패처(device)
│   │   ├── timing_csv.hpp                 #   per-rank, per-event 타이밍 CSV
│   │   └── inputs/{PARA_INPUT_*gpu_*,scaling/}
│   ├── channel_cpu/                      → build/bin/channel.out      (난류 채널 CPU, 검증 기준)
│   │   ├── main.cpp, TimeIntegrator, MomentumSolver, PressureSolver, TdmaSolver
│   │   ├── HaloExchanger, MpiTopology, Subdomain, Grid, Field, ChannelForcing
│   │   ├── Statistics, FieldOutput, RestartIO, Config, BoundaryCondition
│   │   ├── input/PARA_INPUT_re{180,550}_{pascal,filtered}.dat  (key=value 형식, §8)
│   │   ├── run_re{180,550}_{filtered,pascal}.sh   #   sbatch 드라이버
│   │   └── tests/test_halo.cpp            #   halo 교환 왕복 검증 (make tests)
│   └── channel_gpu/                      → build/bin/channel_gpu.out  (channel_cpu의 CUDA 미러)
│       ├── TimeIntegratorGPU.cu, MomentumSolverGPU.cu, PressureSolverGPU.cu
│       ├── StatisticsGPU.cu, HaloExchangerGPU.cu, ChannelForcingGPU.cu, DeviceField.cu
│       ├── TdmaSolverGPU.{cpp,hpp}        #   백엔드 디스패치(FILTERED/PASCAL)
│       ├── DeviceBuffer.hpp, GpuUtils.cuh, BoundaryConditionGPU.cu
│       ├── input/PARA_INPUT_re{180,550}_{pascal,filtered}.dat
│       └── run_re{180,550}_both.sh        #   2-GPU(np3=2) 실행 드라이버
├── scripts/                              # check_kisti.sh(노드 가용량), run_heat_order.sh
├── results/                             # heat_gpu 기본 타이밍 CSV(흩어진 timing_*.csv) 위치
│                                         #   ※ 정식 스케일링/플롯은 apps/heat_*/results/ 아래
├── Report/                              # performance_analysis.md, ghost_boundary_report.md,
│                                         #   progress_2026-06-24.md
└── build/                              # 생성물(BUILDDIR, 기본 build): lib/ bin/ include/ obj/
                                         #   GPU는 보통 BUILDDIR=build_sm90 로 분리 빌드
```

---

## 4. 두 TDMA 알고리즘

두 솔버 모두 시그니처가 동일합니다 — `solve(A,B,C,D, n_sys, n_row)`. `A,B,C`는
하·주·상 대각, `D`는 RHS이자 출력. 메모리 레이아웃은 **row-major(행이 느리게,
n_sys 시스템이 빠르게)** 라 안쪽 루프가 `#pragma omp simd`(CPU)/coalesced(GPU)로
벡터화됩니다. `nprocs == 1`이면 둘 다 즉시 직렬 `tdma_many`로 폴백합니다.

### 4.1 비교표

| 항목 | **PaScaLTDMAMany** (정확해) | **FilteredTDMA** (절단 근사) |
|---|---|---|
| reduced system 풀이 | 2·nprocs 전역계를 **정확히** 풀이 | 인접 2×2로 분리 + **J행 절단 전파** |
| 통신 패턴 | 전역 **MPI_Alltoall(w/v)** (transpose) | **근접-이웃 1회** (Isend/Irecv) |
| 추가 입력 | 없음 | `eps_constant`(보통 dt), per-row `A_rho/C_rho` |
| 비용 (분할 방향) | `O(n_row)` + 전역 all-to-all | `O(n_row + 2J)` + 근접-이웃 교환 |
| 정확도 | 직렬 Thomas와 **bit-identical** | 절단 한계(≤eps) 내 정확; 대각우세 시 사실상 동일 |
| 강점 영역 | 정확도 최우선, 검증 기준선 | 통신 latency가 큰 환경, 많은 rank |

### 4.2 PaScaLTDMAMany — 정확한 reduced solve

1. 각 rank가 자기 z-slab을 **modified-Thomas**로 전방 소거 → 경계 2행 추출
2. 경계 2행을 reduced 버퍼 `[2 × n_sys]`로 pack
3. **Alltoallw(transpose)** → 각 rank가 `[2·nprocs × n_sys_rt]`의 열 일부 소유
   (CPU는 A/C/D 3회 alltoallw; GPU는 device 연속버퍼에 pack 후 **Alltoallv**)
4. 국소 reduced 시스템을 Thomas로 풀이
5. 역방향 Alltoall(w/v)로 D 회수 → 6) 국소 후방 대입으로 내부 행 갱신

GPU(`PaScaLTDMAManyCUDA`)는 `modified_thomas`/`tdma_many`/pack·unpack 커널 +
`cudaEvent` 5단계 타이밍(`last_step_times_ms()`)을 제공.
> **GPU 통신 주의**: device 포인터에 *derived datatype*을 직접 거는 대신, 명시적
> pack/unpack 커널로 **연속 버퍼**를 만들어 alltoallv — strided DDT는 OpenMPI/UCX의
> 느린 D2H 폴백을 강제하기 때문(§7과 동일 원리).

### 4.3 FilteredTDMA — 절단 근사

- **`solve_filtered_v1`** (백엔드명 `filtered`/`filtered_v1`): **전체** modified-Thomas
  전·후방 소거 → 경계 2×2 인터페이스를 **근접-이웃 1회 교환**으로 풀이 → **최종
  보정만 J행으로 절단** (`J = cal_J_v1(D0,DN)`, 해(solution) 기반 상한).
- **`solve_filtered_v2`** (백엔드명 `filtered_v2`): `J = cal_J_rhs_bound(D)`(RHS 기반)
  를 **미리** 계산 → 전방 소거를 **Phase 1: 2..J 전체 갱신 / Phase 2: J+1.. A-갱신
  생략**으로 절단, 후방·보정도 J 기준 절단. J가 작을수록(빠른 감쇠) 더 빠름.
- **`filtered_tdma_cycl.cpp`**: 주기 경계 변형. **`_profile`**: 단계별 타이밍.
- `cal_J_*` (cpp:56–141): `q = rho/λ₊`, `λ₊=(1+√(1−4·rho²))/2`, `eps_=eps_constant/N²`.
  `rho`(per-row `A_rho/C_rho`의 최대 절댓값)가 `≥0.5`거나 `0`이면 절단 없이 full.
- **경계 rank 처리**: `left_rank_/right_rank_ == MPI_PROC_NULL`이면 해당 인터페이스
  보정 루프를 `if (has_left/has_right)`로 건너뜀 — `MPI_PROC_NULL` recv가 버퍼를
  채우지 않으므로 **정확성 장치**이자 일 절약(루프 밖 1회 분기, 비용 무시).

---

## 5. Heat ADI 응용 (heat_cpu / heat_gpu)

### 5.1 무엇을 푸나 — manufactured solution

```
PDE   :  ∂θ/∂t = ∇²θ + f ,   f = 3π²·cos(πx)cos(πy)cos(πz)
정확해:  θ(x,y,z,t) = sin(πx)sin(πy)sin(πz)·exp(−3π²t) + cos(πx)cos(πy)cos(πz)
도메인:  [−1,1]³ ,  모든 면 Dirichlet (정확해로 BC 부여)
```

`main.cpp`가 솔브 후 정확해와의 **L2 error**를 계산(`option=order`). 시간 감쇠항
(sin·exp)과 정상항(cos)의 합이라, 정상항이 공간 이산화 오차를 노출시켜 **2차
정확도**를 측정하게 합니다.

### 5.2 수치 기법

- **factored Crank–Nicolson ADI**, 방향 순서 **Z → Y → X**. 각 방향이 "많은
  삼중대각계"가 되어 위 TDMA 백엔드로 풀림.
- 방향마다: 명시적 7점 스텐실로 RHS 구성 → 삼중대각 LHS 계수 build → TDMA solve
  → 해를 다음 방향으로 전치/복사. Dirichlet 경계는 ghost-cell 보정으로 RHS에 흡수.

### 5.3 `option` 모드 — dt/Tmax/Nt 정책 (`global.cpp`)

| option | dt | Tmax / Nt | 용도 |
|---|---|---|---|
| `order` | `dt = ρ/(1−2ρ)·2dx²` (ρ=0.25 ⇒ dt=dx²) | `Tmax = dt_512·128` 고정, `Nt=round(Tmax/dt)` ⇒ 64/128/256/512 → Nt 2/8/32/128 | 공간 수렴(모든 N에서 T_final 동일) |
| `strong` | `dt = ρ/(1−2ρ)·2dz²` (ρ_z 고정) | `Nt`(+warmup) 입력, `Tmax=Nt·dt` | 고정-스텝 타이밍/스케일링 |
| (그 외) | order와 동일 | — | 폴백 |

> **strong/weak/refine "연구"**는 별도 코드 분기가 아니라, `inputs/scaling/`의
> **입력 파일 패밀리 + 드라이버 스크립트**로 구성되며 내부적으로 위 정책(주로
> strong, Nt=30·warmup=10)을 씁니다.

### 5.4 분해 · 통신

- **1D z-slab 분해** `(npx,npy,npz)=(1,1,np)` (스케일링 기본). → **x·y 방향은
  nprocs=1이라 두 백엔드가 동일한 직렬 Thomas로 폴백; 차이는 분해 방향 `solve_z`
  에서만 발생** (§6.2의 결론으로 직결).
- ghost-cell 교환: CPU는 MPI subarray datatype, **GPU는 면(slab)을 연속 버퍼로
  `pack_{x,y,z}` → Isend/Irecv → `unpack_{x,y,z}`** (`ghostcell_cuda.cu`).

### 5.5 GPU 특이사항 (heat_gpu)

- **rank↔GPU 1:1**: `cudaSetDevice(local_rank % nDevices)` (`main.cpp`) — 안 하면
  모든 rank가 GPU 0에 몰림.
- 커널(`solve_theta.cu`): `rhs_kernel`, `ghost_{x,y,z}_kernel`,
  `build_lhs_{z,y,x}_kernel`, `update_theta_kernel`, 방향 전치 copy 커널.
  `build_lhs_x`는 32×32 shared-memory tile-transpose로 read/write 모두 coalesced.
- **타이밍 CSV**(`timing_csv.hpp`): `rank,t_step,event,time_sec` long-format, event =
  `rhs,solve_x,solve_y,solve_z,comm`. 경로는 `TIMING_CSV` 환경변수 또는 기본
  `results/timing_<nx>_<npxnpynpz>_<backend>.csv`.

---

## 6. 검증 · 핵심 결과

### 6.1 정확도 — 2차, 백엔드 무관 동일

heat manufactured solution, `pascal`, np=2×2×2 (CPU; nx=64/128/256/512). 세 백엔드
× NP × 격자 전 조합에서 결과가 자릿수까지 일치(filtered 절단 오차 ≤ round-off),
CPU↔GPU bit-identical, V100(sm_70)·A100(sm_80) 동일.

**Dirichlet** (`periodic` 미설정):

| nx | 64 | 128 | 256 | 512 | order |
|---|---:|---:|---:|---:|---:|
| L2 | 1.24416e-04 | 3.43421e-05 | 8.61189e-06 | 2.16478e-06 | → **2.0** (1.86→2.00→1.99) |

**Periodic** (`periodic = 1`):

| nx | 64 | 128 | 256 | 512 | order |
|---|---:|---:|---:|---:|---:|
| L2 | 4.39360e-05 | 1.10226e-05 | 2.76621e-06 | 6.93228e-07 | → **2.0** (1.99→1.99→2.00) |

> 과거 표의 Dirichlet `4.21191e-05`(N=64)는 cell-center 격자에 경계를 노드로
> 포함시키던 반-셀 방식의 값. 현재는 ghost-cell 경계(전 영역 ρ<1/2,
> [`Report/ghost_boundary_report.md`](Report/ghost_boundary_report.md))로 바뀌어
> `1.24416e-04`이며, 둘 다 공간 2차.

### 6.2 스케일링 — Filtered 이득은 어디서, 언제 나오나 (CPU 데이터)

한 스텝 비용 분해: **rhs ~68% (백엔드 무관) · solve_x+y ~17–20% (분해 안 됨 → 동일)
· solve_z ~13–15% (유일한 차이) · comm ~0.05%**. 따라서:

- 두 방법 **전체 시간 차이는 ~0–3%**. Amdahl상 솔버 교체가 건드릴 수 있는 건
  solve_z(~14%)뿐이고, 그 안에서 Filtered가 ~20–30% 빠르지만 전체로는 희석됨.
- **이득은 분해 방향 프로세스 수에 비례** (solve_z 가속 np=8에서 PaScaL ~2.5×
  vs Filtered ~3.1×). **격자만 키우는 refine(np=2 고정)에서는 거의 안 보임**
  (J가 상수 → solve_z 비 ~1.2배에서 평평).

플롯/원자료: `apps/heat_cpu/results/scaling_cpu/` + `plot_scaling_cpu.py`,
`plot_scaling_z.py`(solve_z 단독), GPU는 `scaling_gpu/<hardware>/` (§아래 주의).

> **scaling_gpu 출력은 하드웨어별 하위폴더**(`scaling_gpu/v100`, `/a100`, …)로
> 자동 분리됨 — run/sbatch 스크립트가 `nvidia-smi`로 GPU명을 감지. V100/A100을
> 같이 돌려도 덮어쓰지 않음.

### 6.3 GPU ghost-cell 최적화

면을 연속 버퍼로 pack/unpack(↔ device 포인터 + derived datatype)로 바꿔
N=256에서 **~111×**, N=512에서 **>50×** 가속(V100 PCIe). 원리: CUDA-aware
OpenMPI/UCX는 연속 버퍼만 빠르게 처리하고 strided DDT+device포인터는 느린
폴백으로 빠짐. (PaScaL GPU alltoall도 같은 이유로 명시 pack 사용.)

### 6.4 Channel 검증 · Filtered 이득 (GPU)

- **정확도**: GPU(`channel_gpu`)가 검증 기준 `channel_cpu`와 발달 u_tau **<1%** 일치
  (Re180·Re550, pascal·filtered 모두; 상세 §8.4).
- **Filtered 이득은 채널에서 작다** — (1) 압력 z-TDMA가 두 백엔드 모두 PaScaL 고정,
  (2) 모멘텀 z만 백엔드 차이인데 측정상 ~13%(re550, np3=2) · momentum 전체로 ~3.6%,
  (3) GPU 국소 sweep은 memory-bound라 J 절단의 *연산* 절감이 작음(skip-A 행도 A,B,C,D를
  거의 다 읽음). **이득의 근원은 연산이 아니라 통신**(전역 all-to-all→근접이웃)이고
  **분해 방향 rank 수(np3)에 비례** → np3=2에선 거의 안 보임(§6.2와 동일 결론).
- **모니터 진단 주의(과거 버그)**: `channel_gpu`의 `maxDivU`/`WSS`/`u_tau`는 블록 리덕션을
  쓰는데, 한때 `max_div_host_`↔`wss_host_`의 `k_reduce_max`/`k_reduce_sum`이 뒤바뀌어
  div가 ~256배 과대·u_tau가 ~16배 과소로 보였음(흐름 자체는 정상). 현재 수정됨.

---

## 7. 빌드 & 실행

### 7.1 빌드 타깃 (최상위 `Makefile`)

| 타깃 | 산출물 | 비고 |
|---|---|---|
| `make` (=`all`) | libs + `channel.out` | CPU, FFTW 필요 |
| `make heat` | `build/bin/heat.out` | CPU ADI 참조 |
| `make heat_gpu` | `build/bin/heat_gpu.out` | `USE_CUDA=1 CUDA_ARCH=…` 필수 |
| `make channel` | `build/bin/channel.out` | 난류 채널 CPU(FFT+TDMA) |
| `make channel_gpu` | `build/bin/channel_gpu.out` | 난류 채널 GPU, `USE_CUDA=1 CUDA_ARCH=…` 필수 |
| `make tests` | `build/bin/test_halo.out` | halo 교환 검증 |
| `make clean` / `rm` | — | 산출물 / 런타임 출력 제거 |

`Makefile.inc` 핵심: `CXX:=mpicxx`, `CXXFLAGS:=-O3 -std=c++17 -fPIC -march=x86-64-v3`,
`FFTW_DIR`, `USE_CUDA?=0`, `CUDA_ARCH?=80`, `NVCC:=nvcc`,
`NVCCFLAGS:=… -arch=sm_$(CUDA_ARCH) -ccbin $(CXX)`, `CUDA_LIBDIR:=$(NVHPC_ROOT)/cuda/lib64`.
`USE_CUDA=1`일 때만 `.cu`가 컴파일되어 `.a`에 함께 archive됨.

### 7.2 CPU (KISTI Neuron)

```bash
module load gcc/15.2.0 mpi/openmpi-4.1.8 fftw3/3.3.10
make            # libs + channel
make heat       # + heat.out
mpirun -np 8 build/bin/heat.out apps/heat_cpu/inputs/PARA_INPUT_256.txt
```

### 7.3 GPU (CUDA-aware MPI, KISTI Neuron)

GPU 툴체인은 **nvhpc만** 로드해야 함(gcc/openmpi 동시 로드 시 hpcx mpirun 충돌).
최상위 `env_gpu.sh`가 이를 처리(module purge → nvhpc/25.11_cuda12 + PATH/LD_LIBRARY_PATH).

```bash
source env_gpu.sh

# heat_gpu (기본 build/):
USE_CUDA=1 CUDA_ARCH=70 make heat_gpu        # V100=70, A100=80, H100/H200=90

# channel_gpu (보통 build_sm90 로 분리; H200=sm_90):
make -C libs/filtered_tdma BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90
make -C libs/pascal_tdma   BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90
make -C apps/channel_gpu   BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90

# 할당 노드에서 (예: heat_gpu 1-GPU):
srun --jobid=<JID> --overlap -n1 \
  bash -c 'mpirun -np 1 build/bin/heat_gpu.out apps/heat_gpu/inputs/PARA_INPUT_1gpu_256.txt'
# channel_gpu 2-GPU 실행은 apps/channel_gpu/run_re{180,550}_both.sh 참고
```
> `.cu` 규칙엔 `-MMD` 의존성 추적이 없음 — **헤더에 멤버를 추가/변경했으면** `make clean`
> (또는 `rm $BUILDDIR/obj/chgpu_*.o`) 후 재빌드. 안 그러면 객체 레이아웃이 어긋나
> 세그폴트(§9의 stale-object 함정).

### 7.4 입력 형식

```ini
nx, ny, nz            # 격자 점 수(코드에서 +1 후 node-centered cell로 사용)
npx, npy, npz         # MPI 분할 (NP = npx·npy·npz; 스케일링은 1,1,np)
rho   = 0.25          # dt = rho/(1−2ρ)·2dx² 안정성/정확도
eps   = 0.005         # FilteredTDMA 초기 cutoff (heat는 매 step set_eps_constant(dt)로 덮어씀)
Tmax, dt  또는  Nt+warmup   # option에 따라 코드가 정함
option = order        # order | strong (그 외 = order 폴백)
tdma_backend = pascal # pascal | filtered(=filtered_v1) | filtered_v2
periodic = 0          # 0=Dirichlet 벽(기본) | 1=전 3방향 주기
                      #   방향별 지정: periodic_x / periodic_y / periodic_z = 0|1
```

**경계 조건** — 기본은 전면 Dirichlet 벽. `periodic = 1`이면 x·y·z 모두 주기,
또는 `periodic_x/y/z`로 방향별 선택(mixed BC 가능). 주기 방향은 해당 TDMA가
**cyclic 솔버**(`solve_cyclic`)로 전환되고 ghost는 wrap으로 채워짐. CPU·GPU 모두,
세 백엔드 모두 지원하며 manufactured solution으로 2차 검증됨(§6.1). 멀티-GPU 주기
실행은 **런타임에 nvhpc만** 로드할 것(openmpi 동시 로드 시 hpcx mpirun 충돌).

---

## 8. Channel 솔버 (CPU 검증 기준 + GPU 포트)

### 8.1 무엇을 푸나
**난류 채널 유동** 솔버. 좌표 규약: **x(주류)·y(스팬)은 주기 → FFT로, z(벽수직)는
무활주 벽 → TDMA**(가장 비싼 1D 풀이). Arakawa C-grid 스태거링, projection형 모멘텀
적분(AB2 대류 + Crank–Nicolson 점성), z 격자 stretching, mass-flow 또는 dP/dx forcing.
대표 케이스: **Re_τ≈180**(Re_b≈2857) · **Re_τ≈550**(Re_b≈10000).

### 8.2 두 구현
- **`apps/channel_cpu/`** (→ `channel.out`) — **검증 기준**. MPI 분해 np1·np2·np3 자유.
  TDMA는 `TdmaSolver::parse_backend()`로 Filtered/PaScaL 선택(heat와 동일 구조).
- **`apps/channel_gpu/`** (→ `channel_gpu.out`) — **CUDA 미러**(rank↔GPU 1:1, z-slab np3).
  모듈명에 `…GPU` 접미사(`MomentumSolverGPU`, `PressureSolverGPU`, `TimeIntegratorGPU`,
  `StatisticsGPU`, `HaloExchangerGPU`, `ChannelForcingGPU`). 백엔드는 `TdmaSolverGPU`.

> **주의**: **압력 Poisson의 z-TDMA는 두 백엔드 모두 PaScaL 고정**(`PressureSolverGPU`).
> Filtered/PaScaL 차이가 나는 곳은 **모멘텀 z-solve(`fdma_z_`)뿐**이다. x·y는 nprocs=1
> 직렬 폴백. → 채널에서 Filtered 이득을 보려면 np3를 키우고 tdma_z로 비교(§6.2 원리).

### 8.3 입력(.dat, key=value) · 모니터
입력은 heat의 .txt와 **다른 key=value `.dat`** 형식(`input/PARA_INPUT_re*_{pascal,filtered}.dat`):
`n1m/n2m/n3m`(격자), `np1/np2/np3`(MPI), `pbc1/2/3`(주기), `uniform*/gamma*`(stretch),
`Re_b`, `MaxCFL`, `dtStart`, `Timestepmax`, `forcing_mode`(MASS_FLOW|…),
`target_bulk_velocity`/`target_dPdx`, `init_mode`(vortex), `tdma_backend`(pascal|filtered),
출력/restart 디렉터리. 매 `nmonitor` step 콘솔에 `Timestep,Time,dt,maxDivU,WSS,u_tau,U_b,
rho_max,rho_min` 출력(+ `statistics/wss_history.dat`).
> u_tau=√WSS, WSS=ν·⟨∂U/∂z⟩_wall, Re_τ=u_tau/ν=u_tau·Re_b.

### 8.4 검증
`tests/test_halo.cpp`가 halo 교환 왕복 정합성 검증(`make tests`). 물리 검증은 GPU를
**검증 기준 channel_cpu와 같은 timestep에서 대조** — 발달 u_tau가 **<1%** 일치:
Re180 GPU 0.0635(pascal)/0.0634(filtered) ↔ CPU 0.0631/0.0636;
Re550 GPU 0.0543/0.0540 ↔ CPU 0.0544; maxDivU ~1e-13. 두 GPU 백엔드끼리도 일치(§6.4).

---

## 9. 함정 / 설계 메모

1. **경계는 입력 `periodic`으로 선택** (기본 Dirichlet). 주기 방향은 Dirichlet 인덱스
   플래그가 꺼지고 `solve_cyclic`로 전환되며 ghost가 wrap으로 채워짐 — 두 처리가
   짝이어야 함(한쪽만 켜면 깨짐). 과거엔 토폴로지가 `{false,false,false}` 하드코딩이라
   periodic을 켜면 Dirichlet ghost fill이 wrap을 덮어써 BC가 한 스텝에 깨졌음(현재 해결).
   GPU는 `ghostcellUpdateDevice`가 `nprocs>1`이 아니라 이웃 존재(`west/east != PROC_NULL`)
   기준으로 교환해야 단일-rank 주기 wrap이 동작(§Report/progress_2026-06-24).
2. **`set_eps_constant(dt)`** 를 매 시간 스텝 호출 → lib 내부 `eps_ = dt/N²`. 입력의
   `eps`는 초기값일 뿐 실행 중 덮어쓰임.
3. **`if (has_left/has_right)` 분기는 제거 대상이 아님** — `MPI_PROC_NULL` recv의
   미초기화 버퍼 사용을 막는 정확성 장치이며, 루프 밖 상수 분기라 비용 무시.
4. **GPU에서 derived-datatype + device 포인터 금지** — 연속 버퍼 pack/unpack으로
   (heat ghost-cell, PaScaL alltoall 모두 이 패턴).
5. **GPU 빌드 시 `CUDA_LIBDIR`** = `$NVHPC_ROOT/cuda/lib64` (nvhpc는 libcudart를
   기본 경로에 두지 않음).
6. **stale object 주의** — GPU 앱 헤더(`*GPU.hpp`, `mpi_subdomain.*` 등)에 멤버를
   추가/변경하면 `.cu` 규칙에 `-MMD`가 없어 그 헤더를 포함하는 다른 TU가 재컴파일되지
   않음 → 객체 레이아웃이 어긋나 **세그폴트/illegal memory access**(예: main.cpp가 옛
   크기로 객체 생성). `make clean` 또는 해당 obj 삭제 후 재빌드
   (heat_gpu=`$BUILDDIR/obj/exgpu_*.o`, channel_gpu=`$BUILDDIR/obj/chgpu_*.o`).

---

## 10. AI를 위한 "무엇을 보려면 어디" 지도

| 알고 싶은 것 | 볼 파일 |
|---|---|
| Filtered 핵심 알고리즘 / J 절단 | `libs/filtered_tdma/filtered_tdma.cpp` (`solve_filtered_v1/_v2`, `cal_J_*`) |
| PaScaL reduced solve / alltoall | `libs/pascal_tdma/pascal_tdma_many.cpp` |
| GPU TDMA 커널 / 통신 | `*_cuda.cu`, `tdma_local_cuda.cu` |
| 백엔드가 어떻게 선택되나 | `apps/*/tdma_backend*.{cpp,cu,hpp}` (`Kind` enum, `parse()`) |
| ADI 시간적분 루프 | `apps/heat_cpu/solve_theta.cpp`, `apps/heat_gpu/solve_theta.cu` |
| dt/Tmax/Nt 정책(option) | `apps/heat_*/global.cpp` |
| 무엇을 푸나(정확해/L2) | `apps/heat_*/main.cpp` |
| 분해/ghost-cell | `apps/heat_*/mpi_subdomain.*`, `ghostcell_cuda.cu` |
| 채널 유동 구조 (CPU 기준) | `apps/channel_cpu/` (`MomentumSolver`, `PressureSolver`, `TdmaSolver`, …) |
| 채널 GPU 포트 / 백엔드 디스패치 | `apps/channel_gpu/*GPU.cu`, `TdmaSolverGPU.{cpp,hpp}` |
| 채널 GPU 모니터·타이밍 (maxDivU/WSS/u_tau, tdma_z) | `apps/channel_gpu/TimeIntegratorGPU.cu`, `MomentumSolverGPU.cu` |
| 빌드 규칙 | `Makefile`, `Makefile.inc`, `libs/*/Makefile`, `apps/*/Makefile` |
| 스케일링/수렴 결과·플롯 | `apps/heat_*/results/`, `*/plot_scaling_*.py` |

---

## 11. 참고 자료

- 성능 분석: [`Report/performance_analysis.md`](Report/performance_analysis.md),
  ghost 경계: [`Report/ghost_boundary_report.md`](Report/ghost_boundary_report.md)
- 원본 PaScaL_TDMA Fortran(NVHPC+cuSPARSE 비교): `/scratch/x3319a05/PaScaL_TDMA_F`
- KISTI Neuron 빌드/실행 메모: `~/.claude/projects/-scratch-x3319a05/memory/` 의
  `pascal_tdma_f_kisti_neuron.md`, `filtered_tdma_*` 항목들
