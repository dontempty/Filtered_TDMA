# Filtered_TDMA — Parallel TDMA Libraries + ADI Heat / Channel Solvers (CPU + GPU)

이 저장소는 분산-메모리 환경에서 동작하는 두 종류의 TDMA(Tridiagonal Matrix
Algorithm) 라이브러리와, 그 라이브러리들을 같은 인터페이스로 호출하는 ADI
heat / channel 솔버를 함께 담은 모놀레포입니다. **CPU(MPI)** 와 **GPU(CUDA +
CUDA-aware MPI)** 두 가지 빌드를 모두 지원하며, 두 알고리즘(`FilteredTDMA`,
`PaScaLTDMAMany`)의 결과가 같은 실행 파일 안에서 `tdma_backend` 입력값으로
즉시 교체 가능합니다.

---

## 1. 빠른 시작

```bash
# CPU 빌드
make             # libfiltered_tdma.a + libpascal_tdma.a + channel.out
make heat        # + Heat ADI 솔버 (build/bin/heat.out)

# GPU 빌드 (CUDA-aware MPI 필요)
module load nvhpc/25.11_cuda12          # KISTI Neuron
USE_CUDA=1 CUDA_ARCH=80 make heat_gpu   # A100=80, V100=70, H100=90
```

실행 예시 (Heat, CPU, np=8):
```bash
mpirun -np 8 build/bin/heat.out apps/heat_cpu/inputs/PARA_INPUT_256.txt
```

Heat 입력에서 `tdma_backend = pascal` 또는 `tdma_backend = filtered` 로
백엔드 선택. 같은 옵션이 GPU 바이너리(`heat_gpu.out`)에도 그대로 적용됨.

---

## 2. 저장소 레이아웃

```
Filtered_TDMA/
├── Makefile, Makefile.inc                # 최상위 오케스트레이션 + 컴파일러/CUDA 플래그
├── libs/
│   ├── filtered_tdma/                    # libfiltered_tdma.a
│   │   ├── filtered_tdma.{cpp,hpp}       #   CPU FilteredTDMA (DistD2 + truncation)
│   │   ├── filtered_tdma_cycl.cpp        #   CPU cyclic variant
│   │   ├── filtered_tdma_profile.cpp     #   per-phase timing variant
│   │   └── filtered_tdma_cuda.{cu,hpp}   #   GPU FilteredTDMACUDA (USE_CUDA=1)
│   └── pascal_tdma/                      # libpascal_tdma.a
│       ├── pascal_tdma_many.{cpp,hpp}    #   CPU PaScaLTDMAMany (alltoall transpose)
│       ├── pascal_tdma_single.{cpp,hpp}  #   CPU single-system variant
│       ├── tdma_local.{cpp,hpp}          #   CPU sequential Thomas (used at nprocs==1)
│       ├── para_range.{cpp,hpp}          #   Block range partitioning
│       ├── pascal_tdma_many_cuda.{cu,hpp}#   GPU PaScaLTDMAManyCUDA (USE_CUDA=1)
│       └── tdma_local_cuda.{cu,cuh}      #   GPU Thomas + modified-Thomas kernels
├── apps/
│   ├── heat_cpu/                         # CPU heat ADI example
│   │   ├── main.cpp                      #   Driver (topology=non-periodic walls)
│   │   ├── solve_theta.cpp               #   ADI Z→Y→X sweep loop
│   │   ├── tdma_backend.{cpp,hpp}        #   "filtered" | "pascal" dispatcher (host)
│   │   └── inputs/PARA_INPUT_{64..512}.txt
│   ├── heat_gpu/                         # GPU heat ADI example (mirrors heat_cpu/)
│   │   ├── solve_theta.cu                #   GPU kernels (RHS, boundary, build_LHS, …)
│   │   ├── tdma_backend_gpu.{cu,hpp}     #   "filtered" | "pascal" dispatcher (device)
│   │   ├── inputs/PARA_INPUT_{1,2,4,8}gpu_{64..512}.txt
│   │   ├── run_one_np.sh                 #   sbatch (NP=1|2|4|8 env switch)
│   │   └── run_convergence_a100.sh       #   Full sweep on amd_a100nv_8
│   └── channel/                          # Channel flow solver (CPU, FFTW + TDMA)
├── scripts/                              # Utility scripts (check_kisti.sh, etc.)
├── results/                              # Timing CSV results
└── build/                                # All artifacts: lib/, bin/, include/, obj/
```

---

## 3. 두 알고리즘 한눈에

| 항목                  | **FilteredTDMA**                              | **PaScaLTDMAMany**                                    |
|-----------------------|----------------------------------------------------------|-------------------------------------------------------|
| 분산 통신 패턴        | 인접 rank와 boundary 행 2회 교환 (Isend/Irecv)           | 모든 rank가 boundary 행을 alltoallv로 교환            |
| reduced-system 풀이   | 양 끝점 J행만 truncated 전파 (`eps` cutoff 기반)         | `2·nprocs` 행의 reduced system을 정확히 풀이          |
| 추가 입력             | `eps_constant` (보통 dt), `A_rho`/`C_rho` per row        | 없음                                                  |
| 비용                  | O(n_row + 2J) per axis, 통신 = 2 face exchanges          | O(n_row) per axis, 통신 = 1 alltoallv                 |
| 정확도                | truncation 한계 안에서 정확; multi-rank 시 작은 오차 누적 가능 | 모든 rank 수에서 직렬 Thomas와 bit-identical          |
| 적합한 곳             | 통신 latency 큰 환경, channel flow의 ADI sweep          | 정확도 우선, smooth manufactured solution 검증 등     |

Heat 예제에서 두 백엔드가 결과적으로 동일한 L2 error를 만들어내는 게 검증
완료(아래 §5). 알고리즘 차이는 통신 패턴과 reduced-system 처리에 있고,
**Heat 같은 smooth solution + 작은 격자에서는 동일 결과로 수렴**합니다.

**Filtered backend 두 가지 변형** (GPU 빌드 한정):
- `filtered` (=`filtered_v1`): full forward+backward sweep 후 실제 D0/DN에서
  J를 계산해 부분 final correction.
- `filtered_v2`: 보수적 D0/DN bound로 J를 미리 계산 → forward/backward를 J까지만
  full update, 그 너머는 skip-A 또는 D-only로 처리. J가 작을 때(빠른 감쇠) 빠름.

---

## 4. Heat manufactured solution

PDE: ∂t θ = ∇²θ + f,  `f = 3π²·cos(πx)cos(πy)cos(πz)`

Exact: `θ(x,y,z,t) = sin(πx)sin(πy)sin(πz)·exp(-3π²t) + cos(πx)cos(πy)cos(πz)`

도메인 `[-1,1]³`, 모든 면에서 Dirichlet 벽 (`topo.init({...}, {false,false,false})`).
**x를 periodic으로 켜면 ghost cell이 우측 끝 값을 wrap-around해 좌측 BC를
덮어쓰므로 반드시 `false`로 둘 것.**

시간 적분: factored Crank-Nicolson ADI (Z → Y → X 순). `rho = 0.25`로 두면
`dt = rho/(1-2ρ)·2·dx² = dx²`이고, `Tmax = dt_N · 128` (reference grid
N=512)을 고정해 `Nt = Tmax/dt ∝ 1/dx²`. 따라서 모든 N에서 **T_final이 동일**
→ 격자 정제만의 spatial order를 측정 가능 (`option = order`).

| N    | dt (=dx²)  | Nt   | T_final = 1.953e-3 |
|------|-----------:|-----:|--------------------|
|  64  | 9.77e-04   |   2  | ✓                  |
| 128  | 2.44e-04   |   8  | ✓                  |
| 256  | 6.10e-05   |  32  | ✓                  |
| 512  | 1.53e-05   | 128  | ✓                  |

---

## 5. Convergence 검증 결과

L2 error vs exact, 모든 결과 bit-identical (round-off 안에서 일치).

### CPU 기준값 (`apps/heat_cpu/`, np=2×2×2)

| Backend     | nx=64       | nx=128      | nx=256      | nx=512      | order |
|-------------|------------:|------------:|------------:|------------:|------:|
| `pascal`    | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 | 2.00  |
| `filtered`  | 4.21241e-05 | 1.03733e-05 | 2.58592e-06 | 6.46432e-07 | 2.00  |

asymptotic ratio E(h)/E(h/2) = 4.000, order = **+2.000**.

### GPU 결과 (`apps/heat_gpu/`, KISTI Neuron cas_v100nv_8)

3개 backend × NP=1,2,4 × 4 grid sizes = **36 케이스 모두 bit-identical**:

| Backend       | NP | nx=64       | nx=128      | nx=256      | nx=512      |
|---------------|---:|------------:|------------:|------------:|------------:|
| `pascal`      |  1 | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 |
| `pascal`      |  2 | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 |
| `pascal`      |  4 | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 |
| `filtered`    |  1 | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 |
| `filtered`    |  2 | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 |
| `filtered`    |  4 | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 |
| `filtered_v2` |  1 | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 |
| `filtered_v2` |  2 | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 |
| `filtered_v2` |  4 | 4.21191e-05 | 1.03721e-05 | 2.58574e-06 | 6.46409e-07 |

Asymptotic ratio E(h)/E(h/2) ≈ 4.06 → 4.01 → 4.00 → **2차 정확도 확인**.

**검증 환경**: cas_v100nv_8 (V100 ×4, sm_70 빌드), nvhpc/25.11_cuda12,
CUDA-aware HPC-X OpenMPI. job 723253. A100 (`amd_a100nv_8`)에서도 동일
결과 확인됨.

### GPU 코드에 적용된 주요 변경

1. **`cudaSetDevice(local_rank)` 추가** (`apps/heat_gpu/main.cpp`): MPI rank를 GPU에
   1:1 매핑. 이전엔 모든 rank가 GPU 0에 몰려 NP=4에서 ~4× 오버서브스크립션.
2. **PaScaL_TDMA solver를 PaScaL_TDMA_F 최신과 동기화**: shared-memory
   pipeline (`tdma_many_kernel`, `modified_thomas_kernel`) + bank-conflict
   padding, MPI_Alltoallv on contiguous device buffer.
3. **Filtered backend 멀티-rank 경로를 PaScaL-스타일로 재작성**:
   - 단일-launch 통합 커널 (`k_fwd_pass`, `k_bwd_pass`, `k_final_pass`) — row마다
     별도 kernel 호출하던 loop 제거.
   - v2 전용 통합 커널 (`k_fwd_pass_v2`, `k_bwd_pass_v2`).
   - 마지막 row를 dedicated contiguous send buffer로 pack 후
     `MPI_Isend/Irecv` (plain `MPI_DOUBLE`, CUDA-aware fast path).
   - **`cal_J_v1`/`cal_J_v2`를 GPU 커널로 이전** — reduction과 `sqrt/log`까지
     디바이스에서 처리, host는 결과 int 1개만 받음. (host libm 호출이
     NVHPC nvc++ 환경에서 MPI 후 SIGILL을 유발하던 문제 해결.)
4. **build_lhs_x tile-transpose** (`apps/heat_gpu/solve_theta.cu`): 32×32 shared
   memory tile로 d_rhs(ii-fastest) read와 d_X(jj-fastest) write 양쪽 모두
   coalesced 만듦.

> 참고: PaScaL_TDMA GPU의 별도 sweep 보고서 — [/scratch/x3319a05/PaScaL_TDMA/run/CONVERGENCE_GPU.md](../PaScaL_TDMA/run/CONVERGENCE_GPU.md)

---

## 6. 빌드 매트릭스

### CPU only

```bash
module load gcc/15.2.0 mpi/openmpi-4.1.8 fftw3/3.3.10   # channel은 FFTW 필요
make            # 라이브러리 + channel
make heat       # + Heat (CPU)
make clean
```

### GPU (CUDA-aware MPI)

KISTI Neuron 환경 (CUDA 12 / HPC-X OpenMPI 4):
```bash
module purge
module load nvhpc/25.11_cuda12          # nvcc + HPC-X + mpicxx 모두 제공
export CUDA_LIBDIR=$NVHPC_ROOT/cuda/lib64

# 라이브러리(.cu 포함) + Heat_gpu 빌드
USE_CUDA=1 CUDA_ARCH=80 make heat_gpu   # A100=80, V100=70, H100=90

# V100 노드(cas_v100nv_8)에서 실행하려면 sm_70로 재빌드:
USE_CUDA=1 CUDA_ARCH=70 make heat_gpu
```

`USE_CUDA=1` 일 때만 `Filtered_TDMA/filtered_tdma_cuda.cu`,
`PaScaL_TDMA/pascal_tdma_many_cuda.cu`, `PaScaL_TDMA/tdma_local_cuda.cu` 가
컴파일되어 라이브러리 `.a` 안에 함께 archived됨. CPU만 빌드(`USE_CUDA=0`,
기본값)일 땐 CUDA 의존성 없음.

### sbatch (KISTI Neuron, A100 노드)

```bash
# 단일 NP × 4 grid 시리즈 (~1분):
NP=1 sbatch -p amd_a100nv_8 --gres=gpu:1 -J pascal_1g apps/heat_gpu/run_one_np.sh
NP=2 sbatch -p amd_a100nv_8 --gres=gpu:2 -J pascal_2g apps/heat_gpu/run_one_np.sh

# 또는 전체 스윕(3 backends × 3 분할 × 4 grid = 36 runs):
sbatch apps/heat_gpu/run_convergence_a100.sh
```

V100 노드(`cas_v100nv_8`)에서도 동작하지만 PCIe 토폴로지라 GPU↔GPU 통신이
NVLink 노드보다 훨씬 느림. 알고리즘 정확도는 동일 (multi-rank 결과
bit-identical 확인됨).

---

## 7. 동작 가능한 주요 입력 옵션

`apps/heat_cpu/inputs/PARA_INPUT_*.txt`, `apps/heat_gpu/inputs/PARA_INPUT_*.txt`:

```ini
nx = 256                # 격자 점 수 (입력); 코드 내부에서 nx++ 후 cell 개수로 사용
ny = 256
nz = 256

npx = 2                 # MPI 분할 (총 NP = npx · npy · npz)
npy = 2
npz = 2

rho   = 0.25            # dt = rho/(1-2ρ)·2·dx² 안정성·정확도 조절
eps   = 0.005           # FilteredTDMA 초기 cutoff (매 step solve_eps=dt로 덮어씀)
Tmax  = 0.003           # option=order에서는 코드가 dt_N·128로 덮어씀
dt    = 0.001           # (마찬가지로 코드가 덮어씀)

option = order          # "order"=convergence sweep, "strong"=Nt=3 fixed for scaling
tdma_backend = pascal   # "pascal" | "filtered" (=filtered_v1) | "filtered_v2"
```

---

## 8. 알려진 버그 / 주의 사항

1. **Heat main.cpp의 topology**는 **`{false, false, false}`** 이어야 합니다.
   `{true, …}` (x periodic)로 두면 ghost-cell update가 우측 cell 값을
   좌측 boundary로 wrap-around하여 Dirichlet BC가 한 step 만에 깨집니다 —
   convergence 실패의 과거 원인이었음. 현재 코드는 수정됨.
2. **`set_eps_constant(dt)`** 는 Heat의 시간 step마다 호출해 매번 lib의 `eps_`
   를 `dt`로 갱신해야 channel과 같은 패턴. 입력 파일의 `eps = 0.005`는
   초기값일 뿐 실행 중 덮어쓰입니다.
3. **Filtered backend의 multi-rank 경로**는 cutoff J에 의존해 매우 작은
   truncation을 적용. smooth manufactured solution에서는 결과가 PaScaL과
   거의 동일하지만, 매우 가파른 RHS가 등장하는 케이스에서는 양쪽 결과를
   별도로 검증할 것.
4. **`heat_gpu`의 mpirun-2-GPU 성능**은 V100 PCIe 노드에서 통신 latency가
   dominant. A100 NVLink 노드 (`amd_a100nv_8`)에서 훨씬 빠르며 알고리즘
   정확도는 동일.
5. **GPU 빌드 시 `CUDA_LIBDIR`** 명시 필요 (KISTI Neuron의 nvhpc는
   libcudart를 `$NVHPC_ROOT/cuda/lib64`에 둠 — 기본 `/usr/local/cuda/lib64`에
   없음). sbatch 스크립트에서 자동 설정.

---

## 9. Channel solver

`apps/channel/` 디렉터리는 PaScaL_TCS 자연대류 Fortran 코드를 C++17 + FFTW3 +
Filtered_TDMA로 재구성한 별도 솔버입니다. wall-normal 방향 TDMA에
`FilteredTDMA`를 사용하며, x/y는 FFT, z는 TDMA. Build target: `make channel`.

PaScaL_TCS와의 모듈 대응 + 좌표 규약 + 빌드 패턴은 이 파일의 이전 버전
(channel 중심 README)에 정리되어 있고, 채널-specific 내용은 별도
`channel/README.md` 로 분리 예정.

---

## 10. 성능 최적화 로그 — ghost-cell pack/unpack

### 배경

초기 GPU 포팅은 `MPI_Type_create_subarray` 로 만든 **derived datatype**
을 **device pointer** 에 적용해서 `MPI_Isend/Irecv` 를 호출하는
방식이었음. HPC-X (OpenMPI) 의 CUDA-aware 경로는 **연속 device 버퍼**
에 대해서는 UCX `cuda_ipc` / `cuda_copy` 로 빠르게 처리하지만,
**strided derived type + device pointer** 조합에서는 폴백 (셀 단위
`cudaMemcpy` 또는 전체 배열 D2H/H2D 스테이징) 으로 빠지면서 연속
전송 대비 100×–300× 까지 느려짐.

### 변경 사항 (`apps/heat_gpu/ghostcell_cuda.cu`)

각 면의 slab 을 2D CUDA kernel (`pack_x/y/z`) 로 **연속 device buffer**
에 packing → 연속 포인터로 `MPI_Isend/Irecv` → 받은 buffer 를 `unpack_x/y/z`
kernel 로 ghost slot 에 scatter. 12 개의 device buffer (각 면당 send/recv
1 쌍 × 6 면) 는 time loop 시작에서 `MPISubdomain::allocGhostBufsDevice()`
가 한 번 할당, 끝에서 해제.

### 측정 결과 (V100 PCIe, 2-GPU NP=2,1,1, backend=pascal)

| N    | Before (DDT, job 721173) | After (pack/unpack, job 721218) | Speedup  |
|------|-------------------------:|--------------------------------:|---------:|
|  64  |         3 s              |        2 s                      |  1.5×    |
| 128  |        22 s              |        2 s                      |   11×    |
| 256  |       332 s              |        3 s                      | **111×** |
| 512  |    > 1800 s (안 끝남)    |       36 s                      | **>50×** |

`backend=filtered` 도 동일 ghost-cell 경로를 쓰므로 같은 정도의 가속.
전체 8-run sweep (2 backends × 4 grid) 이 이전엔 1시간 안에 끝나지
않았는데 최적화 후 **약 80 초** 안에 완료.

### 시도했지만 도움 안 된 환경 변수

다음 UCX / OpenMPI 옵션은 시도했지만 V100 PCIe 노드에서는 의미 있는
차이가 없었음 (UCX 가 `gdr_copy is not available` 경고를 띄움 — V100
PCIe 에는 GPUDirect RDMA 없음; `cuda_copy`/`cuda_ipc` 는 이미 활성화):
```bash
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib,uct
export OMPI_MCA_pml_ucx_opal_cuda=1
export OMPI_MCA_opal_cuda_support=true
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy
export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_THRESH=8192
```
**병목은 transport 가 아니라**, `MPI_Isend/Irecv` 안에서 derived-datatype
+ device-pointer 조합이 OpenMPI 를 느린 fallback 으로 강제하는 것이
었음. 동일 패턴이 `PaScaL_TDMA_F examples/solve_theta.f90` 의
`ghostcell_update_cuda` 에서도 사용됨.

### 빌드 캐시 주의

`mpi_subdomain.{hpp,cpp}` 에 device-buffer 멤버를 추가했다면 반드시
**`make` 전에 `rm build/obj/exgpu_*.o`** — 헤더만 바뀌고 `.cpp`
의 object 가 stale 한 채로 남으면 struct layout 이 어긋나서 packing
kernel 이 host pointer 를 dereferencing → `illegal memory access` 로
실패함. 디버깅 중 PaScaL_TDMA repo 에서 같은 증상으로 한 차례 겪음.

---

## 11. 참고 자료

- PaScaL_TDMA GPU convergence 보고서: [`/scratch/x3319a05/PaScaL_TDMA/run/CONVERGENCE_GPU.md`](../PaScaL_TDMA/run/CONVERGENCE_GPU.md)
- 원본 PaScaL_TDMA Fortran (`PaScaL_TDMA_F`): NVHPC + cuSPARSE 비교 — `/scratch/x3319a05/PaScaL_TDMA_F`
- 모멘텀 ρ<1/2 분석: [`claude/PaScaL_TCS_momentum_analysis.md`](claude/PaScaL_TCS_momentum_analysis.md)
- KISTI Neuron 모듈 가이드: [`/home01/x3319a05/.claude/projects/-scratch-x3319a05/memory/pascal_tdma_f_kisti_neuron.md`](file:///home01/x3319a05/.claude/projects/-scratch-x3319a05/memory/pascal_tdma_f_kisti_neuron.md)
