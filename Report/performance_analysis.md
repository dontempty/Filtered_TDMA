# Filtered TDMA & PaScaL TDMA 성능 분석 리포트

**날짜**: 2026-06-22  
**대상**: V100 (sm_70), 측정 구성 np=1,2,4 (z-방향 분해)

---

## 0. 현 세션에서 수행된 수정 (적용 완료)

### Fix 1 — D2H blocking copy 제거 (medium → high impact)
**파일**: `libs/filtered_tdma/filtered_tdma_cuda.cu`

`solve_filtered_v1`, `solve_filtered_v2` 내부에서 `cal_J_*()` 함수가 매 타임스텝
`cudaMemcpy DeviceToHost`로 4 byte int J를 CPU로 가져온 뒤 다음 커널에 스칼라로 넘겼다.
이 D2H copy는 CPU-GPU 파이프라인을 강제로 동기화하는 stall이다.

**수정**: `d_J_` (int* device 포인터)를 유지해 커널 인자로 직접 전달.
- `k_fwd_pass_v2`, `k_bwd_pass_v2`, `k_final_pass` 시그니처 변경:  
  `int J` → `const int* __restrict__ d_J`, 내부에서 `__ldg(d_J)` 사용.

---

### Fix 2 — k_cal_J_rhs_bound → k_cal_J_v1 교체 (high impact)
**파일**: `libs/filtered_tdma/filtered_tdma_cuda.cu`, `solve_filtered_v2` 내부

`solve_filtered_v2` (np>1 경로)는 매 타임스텝 J를 결정하기 위해
`k_cal_J_rhs_bound`를 호출했다.

```
// 변경 전
k_cal_J_rhs_bound<<<1, 256>>>(d_J_, ..., d_D, n_sys_, n_row_, eps_);
// n_row × n_sys = 최대 134M 원소를 256스레드 단일 블록으로 스캔 → 극단적으로 느림

// 변경 후
const double* D0_pre = d_D;
const double* DN_pre = d_D + (n_row_-1)*n_sys_;
k_cal_J_v1<<<1, 256>>>(d_J_, d_A_rho_, d_C_rho_, D0_pre, DN_pre, n_sys_, n_row_, eps_);
// n_sys(≈65K) + n_row(≈1K) 원소만 스캔 → n_row 배 빠름
```

**결과**: np=1→2 에서 Filtered가 5~10배 느려지던 현상 해소.  
V100 측정 기준 Filtered가 PaScaL 대비 z-solve에서 **1.4배(大格) ~ 2.5배(小格) 빠름**.

---

## 1. 잔존 최적화 기회

### [HIGH] k_cal_J_v1 — warp shuffle로 tree reduction 교체

**파일**: `filtered_tdma_cuda.cu:270-277`  
**현재 코드**:
```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {   // 8 라운드
    if (tid < s) {
        s_rho[tid] = max(s_rho[tid], s_rho[tid+s]);
        s_D0 [tid] = max(s_D0 [tid], s_D0 [tid+s]);
        s_DN [tid] = max(s_DN [tid], s_DN [tid+s]);
    }
    __syncthreads();   // 256스레드 전체 동기화 × 8회
}
```

`s < 32` 구간(마지막 5 라운드)은 동일 warp 내 스레드끼리 통신이므로
`__syncthreads()` 없이 `__shfl_down_sync`로 처리할 수 있다.

**제안**:
```cuda
// shared reduction (s >= 32)
for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
    if (tid < s) {
        s_rho[tid] = fmax(s_rho[tid], s_rho[tid+s]);
        s_D0 [tid] = fmax(s_D0 [tid], s_D0 [tid+s]);
        s_DN [tid] = fmax(s_DN [tid], s_DN [tid+s]);
    }
    __syncthreads();
}
// warp-level shuffle (no sync needed)
if (tid < 32) {
    double r = s_rho[tid], d0 = s_D0[tid], dn = s_DN[tid];
    for (int s = 16; s > 0; s >>= 1) {
        r  = fmax(r,  __shfl_down_sync(0xffffffff, r,  s));
        d0 = fmax(d0, __shfl_down_sync(0xffffffff, d0, s));
        dn = fmax(dn, __shfl_down_sync(0xffffffff, dn, s));
    }
    if (tid == 0) { s_rho[0]=r; s_D0[0]=d0; s_DN[0]=dn; }
}
```

`__syncthreads()` 호출이 8회 → 3회로 줄고 warp stall이 감소한다.  
**예상 효과**: k_cal_J_v1 내부 약 30~40% 단축. solve_z 전체 대비 낮은 비중이나,
소형 격자(n_sys < 4K) 또는 eps가 느슨해 J가 작을수록 상대적 비중 증가.

---

### [HIGH] k_set_rho — 비연속 메모리 접근 개선

**파일**: `filtered_tdma_cuda.cu:36-58`

`k_set_rho`는 스레드 k가 `d_A[k * n_sys]`를 읽는다.  
n_sys = 65536일 때 warp 내 32스레드는 A[0], A[65536], A[131072], ...를 읽는다.  
이는 stride = n_sys × 8 byte = 512KB 로, L2 cache miss가 필연적이다.

**원인**: A/B/C 행렬이 row-major (행 = nz 방향, 열 = n_sys 방향)이므로,
같은 행의 모든 시스템을 연속 저장하고 있다. `k_set_rho`는 각 행의 열[0]만 읽으므로
행 사이 stride가 크다.

**제안**: `build_lhs_z_kernel`에서 LHS를 계산할 때 첫 번째 시스템(col=0) A/B/C 값을
별도 compact 배열 `d_A_col0[k]`, `d_B_col0[k]`, `d_C_col0[k]`에 동시 기록.
그러면 `k_set_rho`는 연속 메모리를 읽어 coalesced access가 된다.

```cuda
// build_lhs_z_kernel 내부 추가
if (ix == 0 && iy == 0) {  // 첫 번째 시스템만
    d_A_col0[iz] = A_val;
    d_B_col0[iz] = B_val;
    d_C_col0[iz] = C_val;
}
```
```cuda
// k_set_rho 수정: d_A → d_A_col0 (contiguous read)
k = blockIdx.x * blockDim.x + threadIdx.x;
if (k >= n_row) return;
double bk = d_B_col0[k];  // coalesced access
d_A_rho[k] = (bk != 0.0) ? fabs(d_A_col0[k] / bk) : 0.0;
d_C_rho[k] = (bk != 0.0) ? fabs(d_C_col0[k] / bk) : 0.0;
```

**예상 효과**: k_set_rho 시간 30~50% 단축. 대형 격자(n_sys ≫ L2 cache)일수록 효과 큼.

---

### [HIGH] rho 캐싱 — 매 타임스텝 set_rho_device 생략

**파일**: `apps/heat_gpu/solve_theta.cu:585`, `filtered_tdma_cuda.cu:466`

`heat_gpu`의 ADI solver에서 각 방향별 LHS 계수(A, B, C)는
`dt`, 격자간격, 경계조건으로부터 결정된다. **dt 및 격자가 고정이면 A/B/C도 고정**이며
따라서 spectral radius `rho`도 변하지 않는다.

현재는 매 타임스텝마다 `set_rho_device(d_A, d_B, d_C)`를 호출해
`k_set_rho`가 n_row × n_sys 원소 쓰기를 반복한다.

**제안**: `FilteredTDMACUDA`에 `rho_valid_` 플래그 추가.
```cpp
// filtered_tdma_cuda.hpp
bool rho_valid_ = false;

// filtered_tdma_cuda.cu
void FilteredTDMACUDA::set_rho_device(...) {
    if (rho_valid_) return;    // 이미 계산됨 → skip
    k_set_rho<<<grid, block>>>(...);
    rho_valid_ = true;
}
void FilteredTDMACUDA::invalidate_rho() { rho_valid_ = false; }
```

호출 측에서 행렬이 바뀔 때만 `invalidate_rho()`를 호출한다.
**예상 효과**: heat_gpu 전체 타임스텝에서 `k_set_rho` 오버헤드 완전 제거.
각 방향당 1회 커널 launch 절감 (현재 3회/step).

---

### [MEDIUM] k_set_rho + k_cal_J_v1 커널 융합

**파일**: `filtered_tdma_cuda.cu:466-494`

`set_rho_device`와 `cal_J_v1`은 각각:
1. `k_set_rho`: A/B/C → `d_A_rho_`, `d_C_rho_` (n_row 원소 쓰기)
2. `k_cal_J_v1`: `d_A_rho_`, `d_C_rho_` 읽기 + D0/DN 읽기 → `d_J_`

두 커널을 하나로 합치면 `d_A_rho_` / `d_C_rho_` 중간 배열 할당 자체를 제거할 수 있다.
A, B, C를 직접 읽어 rho를 즉석에서 계산하고 J까지 구하는 단일 커널:

```cuda
__global__ void k_set_rho_and_cal_J(
    int* d_J, const double* d_A, const double* d_B, const double* d_C,
    const double* D0, const double* DN, int n_sys, int n_row, double eps)
{
    // thread i 범위 내에서 rho 계산
    // thread i 범위 내에서 max|D0|, max|DN| 계산
    // 공유 메모리로 warp shuffle reduce
    // tid==0이 J를 결정하여 d_J에 쓰기
}
```

`d_A_rho_` / `d_C_rho_` 배열(2 × n_row doubles)도 제거 가능.  
**예상 효과**: 커널 launch 오버헤드 1회 감소, intermediate 메모리 write-read 왕복 제거.
중간 크기 격자에서 5~10% 개선 예상.

---

### [MEDIUM] MPI 통신과 GPU 연산 오버랩

**파일**: `filtered_tdma_cuda.cu:534-557` (v1), `610-635` (v2)  
**파일**: `pascal_tdma_many_cuda.cu:286-325`

현재 패턴 (두 라이브러리 공통):
```
cudaStreamSynchronize(0)    ← GPU 완전 정지
MPI_Isend / MPI_Irecv       ← 비동기 발행
MPI_Waitall                 ← 즉시 블록킹
```

`MPI_Waitall`이 `MPI_Isend/Irecv` 직후에 나와 실질적으로 동기 통신이다.
통신 중 GPU가 유휴 상태다.

**제안**: MPI 발행 후 대기 전 독립적인 GPU 연산 삽입.

Filtered v2 기준 개선안:
```
cudaStreamSynchronize(0)
MPI_Isend(d_C_lastrow_send_, ...)   ← pack 완료된 데이터 발송
MPI_Irecv(d_C_left_recv_,   ...)
MPI_Isend(d_D_lastrow_send_, ...)
MPI_Irecv(d_D_left_recv_,   ...)

// MPI 대기 전에 다음 단계 중 통신 불필요한 커널 실행 가능
// (예: pack_lastrow for 두 번째 교환, 또는 solve_D0_left)
k_solve_D0_left<<<...>>>(...);   ← 경계 처리 (통신 결과 불필요)

MPI_Waitall(4, req, ...)         ← 여기서 기다림
```

구현 복잡도가 있으나, 대형 격자·느린 네트워크에서 효과적이다.  
**예상 효과**: MPI latency의 20~50% 은닉. np=4 이상 대형 격자에서 유효.

---

### [MEDIUM] PaScaL — tdma_many_kernel 후진소거 레지스터 최적화

**파일**: `libs/pascal_tdma/tdma_local_cuda.cu:80-87`

현재 후진소거 루프:
```cuda
for (int j = n_row - 2; j >= 0; --j) {
    std::size_t off = (std::size_t)j * n_sys + i;
    c0[tj] = C[off];
    d0[tj] = D[off];
    d0[tj] -= c0[tj] * d1[tj];
    d1[tj]  = d0[tj];      ← shared memory 회전
    D[off]  = d0[tj];
}
```

`d1[tj]`는 shared memory에 있으므로 매 iteration마다 shared memory 읽기/쓰기 발생.
레지스터 변수로 대체하면 shared memory traffic이 줄어든다:

```cuda
double d_prev = D[(std::size_t)(n_row-1)*n_sys + i];
for (int j = n_row - 2; j >= 0; --j) {
    std::size_t off = (std::size_t)j * n_sys + i;
    double c = C[off];
    double d = D[off] - c * d_prev;
    D[off]   = d;
    d_prev   = d;      ← register rotation
}
```

**예상 효과**: shared memory bank traffic 감소, 약 5~15% loop 단축.

---

### [LOW] k_cal_J_v1 내 fabs 분기 제거

**파일**: `filtered_tdma_cuda.cu:252-261`

```cuda
// 현재 (if 분기 있음)
double a = d_A_rho[k]; if (a < 0.0) a = -a;

// 제안 (하드웨어 fabs 명령)
double v = fmax(fabs(d_A_rho[k]), fabs(d_C_rho[k]));
my_rho = fmax(my_rho, v);
```

GPU에서 `fabs`는 부호비트 마스크 단일 명령, `if`보다 빠르다.  
**예상 효과**: 미미. 코드 가독성 향상이 주된 이점.

---

## 2. PaScaL TDMA 추가 분석

### [MEDIUM] alltoallv 중 H2D displacement 배열 반복 전송 없음 — OK
Constructor에서 6개 int 배열을 cudaMemcpy로 GPU에 올리는 패턴은 일회성이므로
per-solve 오버헤드 없음. 현재 구조 유지 권고.

### [LOW] k_pack_rd2send warp divergence
`blockIdx.z < nprocs` guard가 있으나, nprocs가 작을 때 (2, 4, 8) 전체 커널 크기 자체가
작아 영향 미미.

### [MEDIUM] tdma_many_kernel shared memory 패딩
현재 `(bx + 1) * by` 패딩. `bx=256, by=1` 기준으로는 257-slot layout.  
256스레드 warp 내에서 bank conflict 없음. 현재 구조 적합.

---

## 3. 우선순위 요약

| 순위 | 대상 | 파일 | 기법 | 예상 효과 |
|------|------|------|------|----------|
| 1 | Filtered | `filtered_tdma_cuda.cu` | **rho 캐싱** (`rho_valid_` 플래그) | 매 step k_set_rho 제거 |
| 2 | Filtered | `filtered_tdma_cuda.cu` | **k_set_rho 연속 접근** (compact col0 배열) | k_set_rho 30~50% 단축 |
| 3 | Filtered | `filtered_tdma_cuda.cu` | **warp shuffle** in k_cal_J_v1 | J 계산 30~40% 단축 |
| 4 | Filtered | `filtered_tdma_cuda.cu` | **커널 융합** (set_rho + cal_J) | 중간 배열 제거, launch 절감 |
| 5 | Both | `*_cuda.cu` | **MPI-compute 오버랩** | np≥4 대형 격자에서 20~50% 통신 은닉 |
| 6 | PaScaL | `tdma_local_cuda.cu` | **레지스터 rotation** (backward pass) | 5~15% loop 단축 |

---

## 4. 현재 측정 결과 (수정 적용 후, V100 np=1,2,4)

| 케이스 | PaScaL Tz [ms/step] | Filtered Tz [ms/step] | 비율 (P/F) |
|--------|--------------------|-----------------------|-----------|
| strong 256²×2048, np=1 | 21.2 | 14.8 | 1.43 |
| strong 256²×2048, np=2 | 15.0 | 9.1  | 1.65 |
| strong 256²×2048, np=4 | 8.7  | 5.9  | 1.47 |
| weak 256³/GPU, np=2    | 4.7  | 3.3  | 1.42 |
| weak 256³/GPU, np=4    | 5.1  | 3.5  | 1.46 |
| refine 256²×nz, np=2  | ~1.3~1.5× | — | 1.3~1.5 |

Filtered TDMA가 모든 구성에서 PaScaL보다 빠르며, 스케일링 효율도 우수하다.

---

## 5. 미적용 수정의 정확성 노트

**k_cal_J_v1 (경계값 기반 J 추정)의 수학적 타당성**:  
`solve_filtered_v2`에서 forward sweep 이전 경계행(D[0], D[n_row-1])으로 J를 추정한다.
전체 RHS max를 쓰는 `k_cal_J_rhs_bound` 대비 J가 작게 나올 수 있으나,
boundary의 RHS가 interior에 전파되는 필터 특성상 경계값이 interior의 대리 지표로 타당하다.
(사용자 확인: "어차피 그 곳에 경계의 rhs가 존재하니까 문제 없을 듯")

---

*Report saved: `/scratch/x3319a05/Filtered_TDMA/Report/performance_analysis.md`*
