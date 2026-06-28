# channel_gpu — 실행 매뉴얼 (KISTI Neuron, H200)

> 난류 채널 유동 GPU 솔버(`channel_gpu.out`)를 **빌드 → GPU 할당 → 실행 → 결과 해석**까지
> 처음 보는 사람/AI가 그대로 따라 할 수 있게 정리한 런북입니다.
> 좌표: **x(주류)·y(스팬)=주기(FFT), z(벽수직)=무활주 벽(TDMA, z-slab 분해)**.
> TDMA 백엔드는 입력 한 줄(`tdma_backend = pascal | filtered`)로 교체.
> 라이브러리/알고리즘 개요는 상위 [`../../README.md`](../../README.md) 참고.

---

## 0. TL;DR (이미 빌드돼 있다면)

```bash
cd /scratch/x3319a05/Filtered_TDMA
source env_gpu.sh                                   # nvhpc-only 환경

# (1) 8-GPU 4시간 할당 받기 (no-shell = 셸 안 띄우고 jobid만)
salloc --no-shell -J ch -p amd_h200nv_8 -N 1 -n 8 \
       --cpus-per-task=8 --gres=gpu:8 -t 04:00:00 --comment etc
squeue -u $USER                                     # JOBID 확인 (예: 794488)

# (2) 실행 (re550: filtered=GPU0-3, pascal=GPU4-7 동시, 각 np3=4)
srun --jobid=<JID> --overlap -n1 -c 64 \
     bash apps/channel_gpu/run_re550_both.sh

# (3) 모니터
tail -f apps/channel_gpu/log/re550_filtered.log
```

---

## 1. 사전 요구

- KISTI Neuron, 파티션 `amd_h200nv_8` (노드당 H200 8장, 약 140 GiB/GPU, 64 core).
- 빌드 환경은 **nvhpc만** 로드 (`source env_gpu.sh`). gcc/openmpi를 같이 로드하면 hpcx mpirun과 충돌.
- 작업 디렉터리는 항상 프로젝트 루트 `/scratch/x3319a05/Filtered_TDMA` 기준.

---

## 2. 빌드

GPU 빌드는 보통 하드웨어별 디렉터리 `build_sm90`에 넣습니다 (H200 = `sm_90`).

```bash
cd /scratch/x3319a05/Filtered_TDMA
source env_gpu.sh

# 의존성 순서(PaScaL → FilteredTDMA → channel_gpu)는 top Makefile이 처리해 줌:
make channel_gpu BUILDDIR=build_sm90 USE_CUDA=1 CUDA_ARCH=90
#   산출물: build_sm90/bin/channel_gpu.out
```

per-디렉터리로 직접 빌드한다면 **반드시 pascal을 먼저** (filtered가 pascal의 `tdma_local_cuda.cuh`에 의존):

```bash
make -C libs/pascal_tdma   BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90
make -C libs/filtered_tdma BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90
make -C apps/channel_gpu   BUILDDIR=../../build_sm90 USE_CUDA=1 CUDA_ARCH=90
```

> **함정 — stale object**: `.cu` 규칙엔 `-MMD` 의존성 추적이 없습니다. **헤더(`*GPU.hpp` 등)에 멤버를
> 추가/변경했으면** 다른 TU가 재컴파일되지 않아 객체 레이아웃이 어긋나 **세그폴트**가 납니다.
> 그럴 땐 `rm build_sm90/obj/chgpu_*.o` 후 다시 빌드하세요.
> CUDA_ARCH가 다른 빌드를 같은 `build_sm90`에 섞지 마세요(라이브러리 .a가 한 arch로 고정됨). 아치별로
> `build_sm70/80/90` 분리.

---

## 3. GPU 할당 (salloc --no-shell)

```bash
salloc --no-shell -J ch -p amd_h200nv_8 -N 1 -n 8 \
       --cpus-per-task=8 --gres=gpu:8 -t 04:00:00 --comment etc
# 특정 노드 지정: -w gpu56
squeue -u $USER -o "%i %P %N %b %C %L %T"     # JOBID, GPU수, 남은시간 확인
```

- `--no-shell`: 셸을 안 띄우고 **할당만 생성**, jobid 반환 → 이후 `srun --jobid=<JID> ...`로 사용.
- `--gres=gpu:8`은 salloc에선 되지만, **이미 만든 step에 `srun --gres=...`를 다시 주면 "Invalid gres" 에러** → srun에는 `--gres` 빼고 `--overlap`만.
- `--comment etc`: KISTI에서 요구될 수 있음(있어서 나쁠 것 없음).
- 4 GPU만 필요하면 `-n 4 --gres=gpu:4 --cpus-per-task=8`.
- GPU 보이는지 확인: `srun --jobid=<JID> --overlap -n1 nvidia-smi -L`

---

## 4. 입력 파일 (`input/PARA_INPUT_re*_{pascal,filtered}.dat`)

key=value `.dat` 형식(heat의 .txt와 다름). 같은 케이스의 pascal/filtered 파일은 **`tdma_backend`와
출력 디렉터리만 다르고 나머지는 동일**해야 공정한 비교가 됩니다.

| 키 | 의미 | 비고 |
|---|---|---|
| `n1m,n2m,n3m` | 격자 셀 수 (x,y,z) | re550=768×512×384, re180=256×256×256 |
| `np1,np2,np3` | MPI 분해 | GPU는 보통 **1,1,np3** = z-slab. **np3 = 사용 GPU 수** |
| `pbc1,pbc2,pbc3` | 주기 경계 | x,y=true, z=false(벽) |
| `uniform1/2/3`,`gamma3` | 격자 stretch | z만 stretch(gamma3=1.5) |
| `Re_b` | 벌크 레이놀즈수 | re550=10000, re180=2857 |
| `dtStart`,`MaxCFL`,`Timestepmax` | 시간적분 | CFL 적응 dt |
| `forcing_mode` | `MASS_FLOW`(벌크속도 고정) 또는 dP/dx | `target_bulk_velocity=1.0` |
| `tdma_backend` | **`pascal` 또는 `filtered`** | 런타임 교체 지점 |
| `nmonitor` | 모니터 출력 간격(step) | 콘솔/wss_history |
| `nstat_start`,`nstat`,`nout_stats` | 통계 누적 시작/간격 | tdma_timing도 이 이후 누적 |
| `nfield_start`,`nout`,`out_field` | 3D 필드 덤프 | **타이밍만 볼 거면 `out_field=0`** (아래 §7 참고) |
| `dir_*` | 출력/restart 경로 | 케이스별 분리(re550_p, re550_f, …) |

**np3 바꾸기** (strong scaling 등): pascal·filtered 두 파일의 `np3`를 같은 값(2/4/8)으로.
n3m이 np3로 나눠떨어져야 함(384 → 2/4/8 OK).

---

## 5. 실행

런처 스크립트 2개:
- **`run_re180_both.sh`** — re180, **순차** 실행 (2 GPU에서 pascal 다음 filtered, 각 np3=2).
- **`run_re550_both.sh`** — re550, **동시** 실행 (filtered=GPU0-3, pascal=GPU4-7, 각 np3=4, 8 GPU).

```bash
cd /scratch/x3319a05/Filtered_TDMA
source env_gpu.sh
srun --jobid=<JID> --overlap -n1 -c 64 bash apps/channel_gpu/run_re550_both.sh \
     > apps/channel_gpu/log/re550_both_driver.log 2>&1 &
# (백그라운드로 두고 tail로 모니터링)
```

스크립트 내부 핵심 (직접 한 백엔드만 돌릴 때 참고):
```bash
unset PMIX_INSTALL_PREFIX                                   # PMIX 충돌 방지
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self UCX_MEMTYPE_CACHE=n
MPI_FLAGS="--bind-to none --mca pml ucx --mca osc ucx --mca coll ^hcoll --oversubscribe"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
  mpirun ${MPI_FLAGS} --host $(hostname):4 -np 4 \
  build_sm90/bin/channel_gpu.out apps/channel_gpu/input/PARA_INPUT_re550_filtered.dat
```

- **`-np`(mpirun) = np3(입력) = `CUDA_VISIBLE_DEVICES`의 GPU 개수** — 셋이 일치해야 함.
- 동시에 두 백엔드를 돌릴 땐 `CUDA_VISIBLE_DEVICES`로 GPU를 갈라 줌(0-3 vs 4-7).
- **별도 `make`/`mpirun`을 같은 build에 동시 2개 실행 금지**(라이브러리 .a/header 레이스). 빌드는 한 번만.

---

## 6. 모니터 읽기

콘솔/로그에 `nmonitor` step마다 한 줄:

```
   Timestep   Time   dt   maxDivU   WSS   u_tau   U_b   rho_max   rho_min
```

- **maxDivU**: 속도장 발산 최댓값. **~1e-12~1e-13이면 정상**(압력 projection이 비발산 강제). 1e-10↑면 의심.
- **WSS**: 벽 전단응력 = ν·⟨∂U/∂z⟩_wall.  **u_tau = √WSS**.
- **Re_τ = u_tau / ν = u_tau · Re_b**.
- **U_b**: 벌크속도 (MASS_FLOW면 1.0 고정).
- **rho_max/min**: TDMA 대각우세도(필터드 절단 J 결정에 사용).

**정상값(발달 후)**:
| 케이스 | 목표 u_tau | Re_τ |
|---|---|---|
| re180 | ~0.063 | ~180 |
| re550 | ~0.054 | ~543 |

초기엔 u_tau가 층류값(re550 ~0.017)에서 출발 → 난류 천이 overshoot(~0.07) → 정상상태로 수렴.

> **"멈춘 것 같다"는 보통 I/O입니다**: `out_field=1`이면 step이 `nout`의 배수일 때 각 백엔드가
> **수~수십 GB 필드를 기록**하는 동안 모니터가 안 올라갑니다. 로그 mtime이나
> `instant/<case>/Output_field_*.plt` 크기가 커지는 중이면 정상. (필드 쓰기 끝나면 재개)

---

## 7. 결과 위치 / 출력

케이스 접미사: `_p`=pascal, `_f`=filtered.

| 경로 | 내용 |
|---|---|
| `log/re550_{filtered,pascal}.log` | 모니터 테이블 (u_tau·maxDivU 등) |
| `statistics/re550_{p,f}/tdma_timing_rank*.csv` | **방향별 TDMA 시간**(타이밍 비교 핵심) |
| `statistics/re550_{p,f}/wss_history.dat` | step별 wss/u_tau/div 이력 |
| `statistics/re550_{p,f}/stats_final_*.dat` | 발달 평균 프로파일(Z⁺, U_mean, u/v/w_rms, uw, P) |
| `instant/re550_{p,f}/Output_field_*.plt` | 3D 순간장 (대용량) |
| `restart_out/re550_{p,f}/cont_*.bin` | 재시작 체크포인트(binary double) |

**instant 필드 포맷(중요)**: 현재 `Output_field_*.plt`는 **ASCII 헤더 3줄 + binary float64** (점당 9변수
X,Y,Z,U,V,W,P,Q,Lambda2, POINT 순서)입니다. **Tecplot/ParaView로 직접 못 엽니다** —
`read_field.py`로 읽으세요:
```bash
python3 apps/channel_gpu/read_field.py instant/re550_f/Output_field_00040000.plt
```
- 필드 1개가 768×512×384에서 ~11 GB. 3회(20000/30000/40000)×2백엔드면 디스크 큼.
- **타이밍만 목적이면 입력에 `out_field = 0`** (필드 덤프 끔) → 빠르고 디스크 절약. 통계/타이밍 CSV는 그대로 나옴.

---

## 8. filtered vs pascal 타이밍 비교

`tdma_timing_rank0000.csv` 마지막 행 컬럼:
```
step, timed_steps, tdma_z_sec_cum, tdma_y_sec_cum, tdma_x_sec_cum,
      tdma_total_sec_cum, momentum_sec_cum, tdma_over_momentum
```
(누적은 `step > nstat_start`부터 시작 → 비교하려면 nstat_start를 넘겨 충분히 돌려야 함.)

해석 포인트:
- **압력 Poisson의 z-TDMA는 두 백엔드 모두 PaScaL 고정** → Filtered/PaScaL 차이는 **모멘텀 z-solve(`tdma_z`)에서만** 발생. x·y는 nprocs=1 직렬 폴백이라 동일.
- 따라서 **`tdma_z`와 `momentum`만 비교**(총 wall-time은 I/O·압력에 희석됨).
- **Filtered 이득은 분해 방향 rank 수(np3)에 비례** (전역 all-to-all → 근접이웃). 실측 예:
  - np3=2: tdma_z −12.7%, momentum −3.6%
  - np3=4: tdma_z −21.6%, momentum −7.2%  (np3 키울수록 격차 ↑)

빠른 비교:
```bash
for b in f p; do echo "== $b =="; tail -1 statistics/re550_$b/tdma_timing_rank0000.csv; done
```

> **성능/프로파일 분리**: 채널 hot path는 동기화 없는 `solve_filtered_v2`(perf)를 호출합니다.
> 시간측정용 `*_profile`(cudaEvent/MPI_Wtime)은 라이브러리에 따로 있고 **채널에선 쓰지 않습니다**.
> 채널의 tdma 타이밍은 `MomentumSolverGPU`가 host `MPI_Wtime`으로 잽니다.

---

## 9. 함정 / 트러블슈팅

1. **세그폴트 직후 빌드함** → stale object. `rm build_sm90/obj/chgpu_*.o` 후 재빌드(§2).
2. **`cannot open tdma_local_cuda.cuh`** → 빌드 순서. pascal을 먼저(또는 top Makefile `make channel_gpu`).
3. **`Invalid gres`** → `srun`에 `--gres` 주지 말 것. `--overlap`만.
4. **hpcx/PMIX 충돌** → 런타임 env에 `unset PMIX_INSTALL_PREFIX`, **nvhpc만** 로드.
5. **모든 rank가 GPU 0에 몰림** → `CUDA_VISIBLE_DEVICES`로 GPU 가르고 `-np`=GPU수 일치.
6. **모니터가 멈춘 듯** → 거의 항상 `out_field` 대용량 I/O(§6). 디스크/quota도 확인(`df -h /scratch`, `lfs quota -u $USER /scratch`).
7. **이전 결과와 섞임** → 새 런 전에 해당 `instant/restart_out/statistics/<case>/` 비우기. wss_history·Monitor는 append 모드라 특히.

---

## 10. 검증 기준값 (channel_cpu 대비)

GPU를 **검증 기준 `apps/channel_cpu`(CPU)와 같은 timestep에서 대조**하면 발달 u_tau가 **<1%** 일치:

| | GPU pascal | GPU filtered | CPU 참조 |
|---|---|---|---|
| re180 u_tau | 0.0635 | 0.0634 | 0.0631–0.0636 |
| re550 u_tau | 0.0543 | 0.0540 | 0.0544 |

maxDivU ~1e-13. 두 GPU 백엔드끼리도 일치. → 수치 정확성 OK, 차이는 **성능(타이밍)뿐**.

---

### 새 케이스/스케일링 만들기 (요약)
1. `input/PARA_INPUT_<case>_{pascal,filtered}.dat` 복사 후 `n*m`,`np3`,`dir_*`,`tdma_backend` 수정.
2. n3m이 np3(2/4/8)로 나눠지게.
3. 타이밍 목적이면 `out_field=0`, `nstat_start`를 충분히 큰 값으로 두고 그 이상 돌리기.
4. `run_*_both.sh`를 복사해 `-np`/`CUDA_VISIBLE_DEVICES`/입력경로만 맞추기.
