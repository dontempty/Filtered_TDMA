# Heat_gpu 확장성(Scalability) 측정 매뉴얼

> 이 문서는 **사람과 다른 AI 에이전트**가 `apps/heat_gpu`의 GPU 확장성(strong /
> weak / refine) 측정을 처음부터 끝까지 재현할 수 있도록 쓴 사용 가이드입니다.
> "무엇을 / 왜 / 어떻게 → 빌드 → 실행 → 출력 해석 → 함정" 순서로 정리합니다.
>
> 대상 코드: `apps/heat_gpu` (factored Crank–Nicolson ADI heat 솔버, GPU).
> 측정 백엔드: **`pascal`** (정확해 기준선) 과 **`filtered_v2`** (절단 근사) 두 가지만.
> 표준 실행 스크립트: [`run_scaling_profile_h200.sh`](run_scaling_profile_h200.sh).

---

## 0. 한 문단 요약

`heat_gpu.out`를 입력 파일 한 개당 한 번 실행하면, 매 시간스텝·매 rank·이벤트별
타이밍이 long-format CSV로 떨어집니다. 입력 파일 이름이 곧 측정 시나리오
(strong/weak/refine × 격자 × NP × 백엔드 × ρ)를 결정합니다. 8-GPU 노드(H200)
한 대를 SLURM으로 잡고, `run_scaling_profile_h200.sh`를 `srun`으로 그 노드에
들여보내면 `inputs/scaling/`의 입력들을 백엔드 `pascal`/`filtered_v2`로 전부
돌려 `results/scaling_profile/h200/{strong,weak,refine}/`에 CSV를 남깁니다.

---

## 1. 세 가지 확장성 실험이 측정하는 것

| 실험 | 고정 | 변화 | 목적 | 이상적 결과 |
|---|---|---|---|---|
| **strong** | 전체 문제 크기 | NP = 1,2,4,8 | 같은 일을 더 많은 GPU로 → 가속비 | T(1)/T(NP) → NP |
| **weak** | rank당 부분영역 크기 D | NP = 1,2,4,8 (전체 격자가 NP배 커짐) | rank당 일정 일감일 때 통신 오버헤드 | 벽시계 시간 일정(평평) |
| **refine** | NP = 2 | 격자만 키움 | 격자 무관성(특히 filtered의 절단 길이 J) | solve_z 비율 평평 |

- **분해는 z-slab 1D**: 모든 scaling 입력이 `(npx,npy,npz) = (1,1,NP)`. 즉 **z 방향만
  여러 rank로 분해**되고 x·y는 `nprocs=1`(직렬 폴백)입니다. → 백엔드(`pascal`
  vs `filtered_v2`) 차이는 **오직 `solve_z`(=분해 방향)에서만** 납니다. 이게
  CSV에서 `tdma_y_*`, `tdma_x_*`가 항상 0인 이유입니다(버그 아님, 설계).
- **NP = npx·npy·npz** 이고 입력 파일에서 자동 추출합니다(아래 실행 스크립트 참고).
- **ρ(대각우세도)** 두 가지: `rho025`(ρ=0.25, dt=dx²; 기본) 과 `rho040`(ρ=0.40;
  filtered의 절단 길이 J가 더 커져 근사 부담이 큰 케이스). 둘 다 측정합니다.

---

## 2. 입력 파일 규약 (`apps/heat_gpu/inputs/scaling/`)

파일명이 곧 시나리오입니다:

```
strong_<gx>x<gy>x<gz>_np<NP>_<backend>_rho<NNN>.txt
weak_<D>cube_np<NP>_<backend>_rho<NNN>.txt
refine_<gx>x<gy>x<gz>_np2_<backend>_rho<NNN>.txt
```

- `<backend>` = `pascal` | `filtered_v1` | `filtered_v2`  ← **본 측정은 `pascal`,
  `filtered_v2`만 사용**(filtered_v1은 무시).
- `rho<NNN>` = `rho025` | `rho040`.

현재 존재하는 격자 패밀리:

| 실험 | 격자 / D | NP |
|---|---|---|
| strong | `64x64x512`, `128x128x1024`, `256x256x2048` | 1, 2, 4, 8 |
| strong (대형 단독, §4.4) | `1024x1024x1024` | 1, 2, 4, 8 |
| weak | `128cube`, `256cube`, `512cube` (rank당 부분영역) | 1, 2, 4, 8 |
| refine | `64x64x64`,`64x64x128`,`64x64x256`,`128x128x128`,`128x128x256`,`128x128x512`,`256x256x256`,`256x256x512`,`256x256x1024` | 2 (고정) |

입력 파일 핵심 키 (예시):

```ini
nx = 128            # 격자 점 수
ny = 128
nz = 1024
npx = 1             # MPI 분해 (scaling은 항상 1,1,NP)
npy = 1
npz = 8
rho = 0.40          # dt = rho/(1-2rho)*2dz²
eps = 0.005         # filtered 초기 cutoff (heat는 매 step dt로 덮어씀)
Nt     = 30         # 총 스텝 (= warmup + 측정)
warmup = 10         # 앞 10스텝은 CSV에서 제외 → 측정 20스텝
option = strong     # order가 아니면 타이밍 CSV 기록
tdma_backend = filtered_v2
```

> `option != "order"` 이면 타이밍 CSV가 기록됩니다. scaling 입력은 모두
> `option = strong`(고정 dt·고정 스텝) 이라 CSV가 생성됩니다. `option = order`
> (공간 수렴/정확도 측정)는 CSV를 쓰지 않습니다.

---

## 3. 빌드

### 3.1 GPU 아키텍처별 `CUDA_ARCH`

| GPU | `CUDA_ARCH` | 권장 빌드 디렉터리 |
|---|---|---|
| V100 | 70 | `build` 또는 `build_sm70` |
| A100 | 80 | `build_sm80` |
| H100 / **H200** | 90 | **`build_sm90`** |

### 3.2 빌드 절차 (H200 = sm_90 예시, 로그인 노드)

```bash
cd /scratch/x3319a05/Filtered_TDMA
source env_gpu.sh                      # nvhpc만 로드(gcc/openmpi 동시 로드 금지)

# 라이브러리 (보통 캐시됨; profile .cu가 libfiltered_tdma.a에 포함되어야 함)
make -C libs/filtered_tdma BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90
make -C libs/pascal_tdma   BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90

# heat_gpu (앱 .cu는 -MMD 의존성 추적이 없음 → 헤더/디스패처 수정 시 강제 재컴파일)
rm -f build_sm90/obj/exgpu_*.o build_sm90/obj/exgpu_*.d
make -C apps/heat_gpu BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90
# 산출물: build_sm90/bin/heat_gpu.out
```

> ⚠️ **stale object 함정**: `.cu` 규칙엔 `-MMD`가 없어, 헤더(`*.hpp`)나
> `tdma_backend_gpu.cu`를 고쳤는데 객체가 재컴파일되지 않으면 객체 레이아웃이
> 어긋나 **세그폴트/illegal memory access**가 납니다. 위처럼 `exgpu_*.o`를 지우고
> 다시 빌드하세요.

### 3.3 타이밍이 올바로 채워지려면 (중요)

`heat_gpu`는 TDMA 솔브 내부를 `tdma_z_comm`/`tdma_z_gpu` 이벤트로 분해해 기록합니다.
이 값들은 솔버의 `last_comm_ms()` / `last_gpu_ms()` 타이머에서 읽는데:

- **PaScaL**: `solve()`가 내부적으로 cudaEvent + MPI_Wtime 타이밍을 하므로 자동으로 채워짐.
- **Filtered**: 운영용 `solve_filtered_v2`는 **sync-free(타이머 미갱신)**. 그래서
  `apps/heat_gpu/tdma_backend_gpu.cu`의 `TdmaBackendGPU::solve()`는 filtered 백엔드를
  **`solve_filtered_v2_profile`**(= cudaEvent/MPI_Wtime 계측판, 수치 동일)로
  라우팅합니다. 이 라우팅이 없으면 filtered의 `tdma_*_comm/gpu`가 **0으로 고정**됩니다.

확인: 빌드 후 작은 케이스 1개를 돌려 filtered_v2의 `tdma_z_comm`/`tdma_z_gpu`가
**0이 아닌지** 보세요(§6 스모크 테스트).

---

## 4. 실행 — 표준 워크플로

### 4.1 8-GPU 노드 할당

```bash
# 예: H200 8-GPU 노드 4시간
salloc -p amd_h200nv_8 -N 1 --gres=gpu:8 --ntasks-per-node=8 \
       --time=04:00:00 --comment=inhouse
# 현재 살아있는 내 할당 확인:
squeue -u "$USER" -o "%.10i %.16P %.8T %.10M %.6D %R"
#  → JOBID, NODELIST(예: gpu56) 확인
```

### 4.2 측정 실행 (할당 노드로 `srun` 들여보내기)

```bash
cd /scratch/x3319a05/Filtered_TDMA
srun --jobid=<JOBID> --gres=gpu:8 -n1 --overlap \
     bash apps/heat_gpu/scripts/run_scaling_profile_h200.sh \
     > apps/heat_gpu/results/scaling_profile/h200/run_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

스크립트가 하는 일:
1. `env_gpu.sh` 로드(노드에서 자체적으로 nvhpc 세팅) + **`unset PMIX_INSTALL_PREFIX`**
   (hpcx mpirun을 srun 안에서 띄울 때의 PMIx 충돌 회피 — §7 함정).
2. `inputs/scaling/`에서 `strong_* → weak_* → refine_*`를 백엔드 `pascal`,
   `filtered_v2`로 순회. 입력별 `NP = npx·npy·npz`를 awk로 추출해 `mpirun -np NP`.
3. CSV를 `results/scaling_profile/h200/{strong,weak,refine}/timing_<입력파일명>.csv`로 저장.

> 측정 규모: strong 48 + weak 48 + refine 36 = **약 132개 런** (2 backend × 2 rho 포함).
> 작은 격자는 수 초, 큰 격자(256x256x2048, 512cube np8)는 수십 초. 총 30~60분 수준.

### 4.3 다른 하드웨어/배치 큐로 돌리려면

- **A100 / V100**: `CUDA_ARCH`와 빌드 디렉터리를 바꾸고(§3.1), 스크립트 안의
  `BIN`/파티션을 맞추세요. 배치 제출형 템플릿은 이 디렉터리의
  `sbatch_scaling_{h200,a100,v100}.sh`, salloc형은
  `run_scaling_{h200_salloc,a100,v100}.sh` 참고(아래 §8 표).
- 핵심은 어느 경로든 동일: **8-GPU 노드 + CUDA-aware MPI(UCX) + `TIMING_CSV` 환경변수로
  출력 경로 지정 + `mpirun -np NP --host <node>:8`**.

### 4.4 대형 격자(1024³) strong 단독 측정

표준 스윕(§4.2)과 **별개**로, 고정 1024³에서 z-slab np=1,2,4,8 strong 측정을
돌리는 전용 경로입니다. 입력은 `inputs/scaling/strong_1024x1024x1024_np{1,2,4,8}_{pascal,filtered_v2}_rho{025,040}.txt`
(16개), 드라이버는 [`run_strong_1024_h200.sh`](run_strong_1024_h200.sh), 출력은
별도 폴더 `results/scaling_profile/h200/strong_1024/`.

```bash
cd /scratch/x3319a05/Filtered_TDMA
srun --jobid=<JOBID> --gres=gpu:8 -n1 --overlap \
     bash apps/heat_gpu/scripts/run_strong_1024_h200.sh \
     > apps/heat_gpu/results/scaling_profile/h200/strong_1024/run_$(date +%F_%H%M%S).log 2>&1 &
```

- 드라이버는 np **8→4→2→1**(GPU당 가벼운 것부터) 순서로 돌려 결과가 빨리 나오고
  무거운 np=1을 마지막에 둡니다.
- 새 입력을 다시 만들려면(예: 다른 격자) 같은 키 포맷으로 `nx=ny=nz` 와 `npz`만
  바꾸면 됩니다(나머지는 strong 템플릿과 동일: `Nt=30, warmup=10, option=strong`).

**메모리 한계 메모**: 1024³ **np=1은 단일 GPU에 약 51 GB**(6개 풀그리드 배열)
잡혀 H200(≈140 GB)에 문제없이 들어갑니다(실측 ~10 s/run). 다만 **더 큰 격자
(예: 2048³)는 np=1이 단일 GPU 메모리를 초과**해 OOM이 납니다 — 이는 자연스러운
한계로 받아들이고, 그 경우 **np=1 baseline을 빼고 np≥2부터** 측정(가속비 기준점을
np=2로) 하면 됩니다.

**대표 결과 (1024³, rho025, rank0·20스텝 평균, ms):**

| backend | np | rhs | solve_z | tdma_z_comm | tdma_z_gpu | step_tot |
|---|--:|--:|--:|--:|--:|--:|
| pascal | 1 | 15.13 | 35.60 | 0 | 0 | 128.5 |
| pascal | 2 | 7.57 | 27.26 | 0.147 | 18.56 | 74.2 |
| pascal | 4 | 3.80 | 15.21 | 1.619 | 9.24 | 39.4 |
| pascal | 8 | 1.91 | 8.01 | 1.160 | 4.60 | 19.8 |
| filtered_v2 | 1 | 15.13 | 35.60 | 0 | 0 | 128.5 |
| filtered_v2 | 2 | 7.57 | 20.46 | 0.057 | 11.96 | 67.4 |
| filtered_v2 | 4 | 3.79 | 11.40 | 0.108 | 7.08 | 35.6 |
| filtered_v2 | 8 | 1.91 | 6.77 | 0.089 | 4.56 | 18.5 |

- **filtered 통신이 압도적으로 쌈**: `tdma_z_comm` @np8 ≈ pascal 1.16 ms vs
  filtered **0.089 ms (~13×)**, @np4 ~15× (전역 all-to-all → 근접이웃).
- **solve_z 강확장 가속비** @np8: filtered **5.26×** vs pascal 4.45×.
- `rhs`는 백엔드 무관·완벽히 2배씩 스케일(전체 최대 비중). rho040도 거의 동일 경향.

---

## 5. 출력(CSV) 형식과 해석

### 5.1 파일 구조 (long-format)

```
# grid=128x128x1024, np=8 (1,1,8), dt= 1.831E-03, Nt=30, solver_kind=filtered_v2
rank,t_step,event,time_sec
0,1,rhs, 1.2259E-05
0,1,solve_z, 1.3637E-04
...
```

- 첫 줄(`#`)은 메타(격자/분해/dt/Nt/백엔드).
- 본문: `rank, t_step(1..20), event, time_sec`. **모든 시간 단위는 초(sec)**.
- `t_step`은 warmup 제외 후 1부터(측정 20스텝).

### 5.2 이벤트 12종

| event | 의미 | 비고 |
|---|---|---|
| `rhs` | 명시적 RHS 커널 | 백엔드 무관, 보통 최대 비중 |
| `solve_z` | **z 방향 한 방향 전체** (build_lhs_z + set_rho + TDMA + copy + sync) | **백엔드 차이가 나는 유일한 방향** |
| `solve_y` | y 방향 전체 | x·y는 nprocs=1 직렬 폴백 → 백엔드 무관 |
| `solve_x` | x 방향 전체 | 〃 |
| `etc` | filtered eps 갱신 등 부대 | 미미 |
| `comm` | ghost-cell 교환(스텝 시작) | z-slab 이웃 교환 |
| `tdma_z_comm` | z TDMA **내부 MPI 통신** 시간 | filtered=근접이웃, pascal=alltoallv |
| `tdma_y_comm` / `tdma_x_comm` | 〃 (y/x) | **항상 0**(nprocs=1 폴백) |
| `tdma_z_gpu` | z TDMA **GPU 커널** 시간(cudaEvent) | |
| `tdma_y_gpu` / `tdma_x_gpu` | 〃 (y/x) | **항상 0**(nprocs=1 폴백) |

관계: `tdma_z_comm + tdma_z_gpu ⊂ solve_z`. `solve_z`는 build_lhs/copy/동기화까지
포함한 **방향 전체 벽시계**이고, `tdma_z_*`는 그 안의 TDMA 솔버 내부 분해입니다.

> **filtered vs pascal 비교 포인트**: `solve_z`(또는 `tdma_z_comm`)를 NP에 대해
> 보세요. 통신을 전역 all-to-all → 근접-이웃으로 바꾼 filtered의 이득은 **분해
> 방향 rank 수(NP)에 비례**해서 커집니다(refine처럼 NP=2 고정이면 거의 안 보임).

### 5.3 한 런의 대표값 만들기

각 CSV에서 보통 **rank별·이벤트별 20스텝 평균(또는 중앙값)** 을 취하고, 필요시
rank 간 최댓값(부하 불균형 보수적 추정)을 씁니다. 간단 예:

```bash
# rank0, filtered_v2 한 런의 이벤트 평균(sec)
awk -F, 'NR>2 && $1==0 {s[$3]+=$4; n[$3]++} END{for(e in s) printf "%-14s %.3e\n", e, s[e]/n[e]}' \
    results/scaling_profile/h200/strong/timing_strong_128x128x1024_np8_filtered_v2_rho025.csv
```

---

## 6. 빌드 검증용 스모크 테스트

전체 132런 전에 작은 케이스로 **타이밍 배선이 맞는지** 확인하세요(특히 filtered):

```bash
cd /scratch/x3319a05/Filtered_TDMA
srun --jobid=<JOBID> --gres=gpu:8 -n1 --overlap bash -c '
  source /scratch/x3319a05/Filtered_TDMA/env_gpu.sh; unset PMIX_INSTALL_PREFIX
  export OMPI_MCA_opal_warn_on_missing_libcuda=0 UCX_TLS=cuda_copy,cuda_ipc,sm,self UCX_MEMTYPE_CACHE=n
  cd /scratch/x3319a05/Filtered_TDMA; N=$(hostname)
  TIMING_CSV=/scratch/x3319a05/Filtered_TDMA/_smoke.csv \
    mpirun --mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe --host $N:8 -np 8 \
    build_sm90/bin/heat_gpu.out apps/heat_gpu/inputs/scaling/strong_64x64x512_np8_filtered_v2_rho025.txt'
# tdma_z_comm / tdma_z_gpu 가 0이 아니어야 정상:
awk -F, '$1==0 && $2==1 && /tdma_z_/' /scratch/x3319a05/Filtered_TDMA/_smoke.csv
```

> `TIMING_CSV`는 **공유 파일시스템(/scratch)** 경로로 주세요. 로그인 노드의
> `/tmp`나 스크래치패드는 **계산 노드에서 안 보여** "cannot open ... for write"가 납니다.

---

## 7. 함정 / 체크리스트

1. **PMIx 충돌** — `module load nvhpc`가 `PMIX_INSTALL_PREFIX`를 설정하는데, 이게
   hpcx OMPI 내장 PMIx와 충돌해 srun 안에서 mpirun이 안 뜹니다
   (`conflicting directives regarding OPAL vs PMIx`). → 실행 전 **`unset PMIX_INSTALL_PREFIX`**
   (표준 스크립트에 이미 포함).
2. **nvhpc만 로드** — gcc/openmpi를 동시에 로드하면 hpcx mpirun과 충돌. `env_gpu.sh`가 처리.
3. **stale object** — 헤더/디스패처 수정 후 `exgpu_*.o` 삭제 안 하면 세그폴트(§3.2).
4. **filtered 타이머 0** — `tdma_backend_gpu.cu`가 filtered를 `_profile` 솔브로
   라우팅하는지 확인(§3.3). 안 그러면 filtered의 `tdma_*_comm/gpu`가 0.
5. **출력 경로** — `TIMING_CSV`는 /scratch 등 공유 FS로(§6).
6. **rank↔GPU 1:1** — `main.cpp`가 `cudaSetDevice(local_rank % nDevices)`. `mpirun`에
   `--oversubscribe`로 NP<8도 8-GPU 노드에서 동작.
7. **할당 시간** — 큰 weak(512cube np8 = 512×512×4096) 포함 시 총 30~60분. 할당
   `--time` 여유 있게. 스크립트는 케이스 실패해도 계속 진행(`set -u`, 개별 `|| true`).
8. **GPU 메모리** — weak 512cube np1은 단일 GPU에 매우 큼(OOM 가능). 실패해도
   스킵하고 다음으로 넘어갑니다.

---

## 8. 이 디렉터리의 스크립트 지도

| 스크립트 | 형태 | 용도 |
|---|---|---|
| **`run_scaling_profile_h200.sh`** | salloc/srun | **(표준)** profile 타이밍으로 strong+weak+refine, pascal+filtered_v2, 양쪽 rho. 출력 `results/scaling_profile/h200/`. |
| **`run_strong_1024_h200.sh`** | salloc/srun | **(대형 단독, §4.4)** 1024³ strong, np 8→4→2→1, pascal+filtered_v2, 양쪽 rho. 출력 `results/scaling_profile/h200/strong_1024/`. |
| `run_scaling_h200_salloc.sh` | salloc/srun | 빌드(sm_90)까지 포함한 strong/weak 스윕. 출력 `results/scaling_1comm/h200/`. |
| `run_scaling_pascal_h200_salloc.sh` / `run_scaling_filtered_h200_salloc.sh` | salloc/srun | 백엔드 한쪽만 분리 실행. |
| `run_refine_h200_salloc.sh` | salloc/srun | refine(np=2)만. |
| `run_order_h200_salloc.sh`, `run_order_np8_222.sh` | salloc/srun | 공간 수렴/정확도(`option=order`, 타이밍 CSV 없음). |
| `run_comm_breakdown_h200_salloc.sh` | salloc/srun | 통신 분해 측정. |
| `sbatch_scaling_{h200,a100,v100}.sh` | sbatch 제출 | 배치 큐로 동일 스윕(파티션별). `_1comm`, `_opt` 변형은 통신 병합/블록최적화 버전. |
| `sbatch_comm_pure*_h200.sh` | sbatch 제출 | 순수 통신 시간 측정 변형. |
| `run_scaling_{a100,v100}.sh`, `run_scaling_all_norebuild.sh` | 헬퍼 | 타 하드웨어/재빌드 생략 실행. |

> 새로 측정한다면 **`run_scaling_profile_h200.sh`** 를 출발점으로 쓰세요. 다른
> 하드웨어면 그 안의 `BIN`(build_sm90)·`CUDA_ARCH`·파티션만 바꾸면 됩니다.

---

## 9. 플로팅

CSV 모음을 받아 그래프를 그리는 스크립트:

- `apps/heat_gpu/results/plot_scaling_gpu.py` — strong/weak 벽시계·가속비.
- `apps/heat_gpu/results/plot_scaling_z.py` — `solve_z` 단독(백엔드 차이 부각).
- `apps/heat_gpu/results/scaling_1comm/plot_event_decomp.py` — 이벤트별 분해 막대.

플롯 입력 디렉터리를 `results/scaling_profile/h200/`로 지정하거나, 스크립트 상단의
경로 변수를 새 출력 폴더로 맞춰 실행하세요(스크립트별 인자/경로 규약 확인).

---

## 10. 빠른 시작 (TL;DR)

```bash
cd /scratch/x3319a05/Filtered_TDMA
# 1) 빌드 (H200=sm_90)
source env_gpu.sh
rm -f build_sm90/obj/exgpu_*.o
make -C apps/heat_gpu BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90
# 2) 노드 할당 후 JOBID 확인
squeue -u "$USER"
# 3) 측정 실행
srun --jobid=<JOBID> --gres=gpu:8 -n1 --overlap \
     bash apps/heat_gpu/scripts/run_scaling_profile_h200.sh \
     > apps/heat_gpu/results/scaling_profile/h200/run_$(date +%F_%H%M%S).log 2>&1 &
# 4) 결과 확인
ls apps/heat_gpu/results/scaling_profile/h200/{strong,weak,refine}/*.csv | wc -l   # ≈132
```
