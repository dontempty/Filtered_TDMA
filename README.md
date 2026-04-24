# Filtered_TDMA — C++ CFD Examples

Filtered_TDMA 라이브러리(`src/`)와 그것을 사용하는 응용 예제들이 형제 디렉토리로
배치된 모놀레포. 첫 예제는 **channel** (PaScaL_TCS Fortran 자연대류 코드를 온도
결합 제거 + 좌표 z=wall-normal로 재구성한 C++17 채널 플로우 솔버) 이며, wall-
normal TDMA 에 Filtered_TDMA 가 사용된다. 새 예제(Heat, cylinder, …)는 channel
옆에 추가하면 된다.

## 디렉토리 구조

[Filtered_TDMAv2](../TDMA/Filtered_TDMAv2/) 와 같이, **라이브러리** 와 각
**예제(application)** 는 형제 디렉토리로 분리. 새 예제는 channel/ 옆에
디렉토리 하나 추가하고 최상위 `Makefile` 에 타겟 한 줄 추가.

| 경로                       | 내용                                              |
|----------------------------|---------------------------------------------------|
| `Makefile`                 | 최상위 오케스트레이션 (`make`, `make tests`)      |
| `Makefile.inc`             | 컴파일러/플래그 (g++ 기본, nvhpc/Intel 주석)      |
| `src/`                     | **Filtered_TDMA 라이브러리** → `libfiltered_tdma.a` |
| `channel/`                 | **예제: 채널 플로우 솔버** (자급자족)             |
| `channel/tests/`           | 채널 단위 테스트 (test_halo 등)                   |
| `channel/PARA_INPUT.dat`   | 채널 입력 파일                                    |
| `claude/`                  | 분석 보고서 (PaScaL_TCS 모멘텀, ρ<1/2 등)         |
| `build/`                   | 산출물 (`obj/`, `lib/`, `include/`, `bin/`)       |
| `(Heat/, cylinder/, …)`    | 향후 추가될 예제 슬롯                              |

## 빌드

```sh
make            # libfiltered_tdma.a + channel.out
make tests      # 단위 테스트 (channel/tests/)
make clean
```

기본 컴파일러는 `mpicxx`(g++ 백엔드). nvhpc/Intel 사용 시 `Makefile.inc` 의 주석된
블록을 활성화. FFTW3 경로는 `Makefile.inc` 의 `FFTW_DIR` 변수에서 조정.

## 새 예제 추가 방법

1. `mkdir <NewExample>` 후 `channel/Makefile` 을 모방한 자체 Makefile 작성
   (`include ../Makefile.inc`, `BUILDDIR ?= ../build`).
2. 최상위 `Makefile` 에 타겟 추가:
   ```makefile
   newexample: TDMA
       $(MAKE) -C NewExample all BUILDDIR=../$(BUILDDIR)
   ```
3. `all:` 타겟에 `newexample` 추가.

## 실행

```sh
mpirun -np 1 build/bin/channel.out channel/PARA_INPUT.dat
```

`PARA_INPUT.dat` 는 PaScaL_TCS Fortran namelist 와 1:1 대응되는 INI 형식
(같은 변수 이름, `key = value` + `#` 주석).

## 좌표 규약 (고정)

| 축 | 의미        | 경계        | Poisson 처리   |
|----|-------------|-------------|---------------|
| x  | streamwise  | periodic    | FFT (FFTW3 r2c) |
| y  | spanwise    | periodic    | FFT (FFTW3 c2c) |
| z  | wall-normal | no-slip wall | TDMA (Filtered_TDMA) |

dP/dx forcing, MPM-STD 의 mass-flow 보정 패턴 적용.

## 모듈 (channel/*)

| 클래스             | 역할                                                | PaScaL_TCS 대응                       |
|--------------------|-----------------------------------------------------|---------------------------------------|
| `Field<T>`         | 1-halo 3D 배열 (x-fastest, 0/n+1 ghost)            | 인라인 array                          |
| `Config`           | `PARA_INPUT.dat` 파싱 + Bcast                      | `module_global.f90` namelists        |
| `MpiTopology`      | 3D Cart + 1D 부분통신자                            | `module_mpi_topology.f90`            |
| `Subdomain`        | rank-local 인덱스 범위 (`para_range` 사용)         | `module_mpi_subdomain.f90` (인덱스 부분) |
| `Grid`             | 1D 좌표 (uniform/tanh stretch)                     | `module_mpi_subdomain.f90:160-295`    |
| `HaloExchanger`    | DDT 기반 6면 ghost 교환                            | `mpi_subdomain_ghostcell_update`     |
| `BoundaryCondition`| z 벽에서 no-slip Dirichlet                         | `module_solve_momentum.f90`의 wall BC |
| `FilteredTdmaSolver`| Filtered_TDMA 라이브러리 RAII 래퍼                | (신규)                                |
| `MomentumSolver`   | Crank-Nicolson + 3-stage ADI (x→y→z)              | `module_solve_momentum.f90`          |
| `PressureSolver`   | 2D FFT(x,y) + 1D TDMA(z) Poisson + projection     | `module_solve_pressure.f90`           |
| `ChannelForcing`   | mass-flow 또는 constant dP/dx (MPM-STD 패턴)      | `cuda_momentum_masscorrection`       |
| `Statistics`       | z-프로파일 누적 평균 (Reynolds 응력 포함)         | `cuda_post_stat.f90`                 |
| `RestartIO`        | binary 재시작 파일 (Gather/Scatterv)              | `cuda_post.f90:1333-1416`             |
| `TimeIntegrator`   | 메인 시간 루프 + ρ-진단                           | `main.f90`                           |

## 단계별 진행 상태

| Phase | 항목                                          | 상태  |
|-------|-----------------------------------------------|-------|
| P0    | 디렉토리 + 라이브러리 복사 + Makefile         | ✓     |
| P1    | Field/Config/MPI/Subdomain/Grid/Halo + test  | ✓ (test_halo 8랭크 PASS) |
| P2    | BC + Filtered_TDMA wrapper                    | ✓     |
| P3    | PressureSolver (FFT+TDMA, np=1 동작)          | ✓     |
| P4    | MomentumSolver (3-stage ADI, np=1 동작)       | ✓     |
| P5    | ChannelForcing + TimeIntegrator               | ✓     |
| P6    | Statistics + RestartIO (round-trip 확인)      | ✓     |
| P7    | DNS 검증 (Re_τ=180, Kim-Moin-Moser)           | ⌛    |

### 현재 한계 (다음 작업)

1. **MPI pencil transpose 미구현**: PressureSolver/MomentumSolver 는 `mpirun -np 1` 에서만
   수치적으로 정확. `np > 1` 사용 시 FFT 길이가 rank-local 가 되어 결과가 잘못됨.
   PaScaL_TCS `module_solve_pressure.f90:304-320` 의 C→I→K alltoallw 체인 이식 필요.
2. **수치 안정성**: 현재 단순 중심차분 convection + 단순 Strang split ADI. 채널 평균
   유동 속도가 상승하면 CFL 적응이 mass-flow 보정과 양의 피드백을 만들어 `dPdx` 가 폭주.
   - skew-symmetric convection
   - Beam-Warming linearization (PaScaL_TCS 처럼)
   - convection-diffusion 분리 (Adams-Bashforth + CN)
   중 하나로 교체 필요.
3. **Statistics z-좌표**: 현재 0으로 출력. rank0 가 z 좌표를 모아서 함께 기록하도록
   확장 필요.
4. **격자 stretch**: tanh 적용은 됐지만 PaScaL_TCS 의 정확한 stretch 함수와 1:1 대응
   여부는 비교 필요.
5. **DNS 검증**: 위 (1)(2) 가 해결된 후 Re_τ=180 비교.

## 참고

- 모멘텀 계수 + ρ<1/2 분석: [`../claude/PaScaL_TCS_momentum_analysis.md`](../claude/PaScaL_TCS_momentum_analysis.md)
- 빌드 패턴 원본: `/shared/home/wel1come1234/workspace/TDMA/Filtered_TDMAv2/Makefile`
