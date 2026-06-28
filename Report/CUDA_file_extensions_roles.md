# CUDA 파일 확장자 역할 정리

CUDA 프로젝트에서 자주 쓰는 확장자는 `.cu`, `.cuh`, `.hpp`이다.  
세 파일은 모두 C/C++ 계열 코드에서 사용되지만, 보통 역할을 나누어 관리한다.

---

## 1. `.cu`

`.cu`는 **CUDA 구현 파일**이다.

보통 다음 내용을 포함한다.

- CUDA kernel 구현
- kernel launch 코드
- GPU 연산을 실행하는 host wrapper 함수
- CUDA API 호출 코드

예시:

```cpp
// solver.cu

#include "solver.cuh"

__global__
void tdma_kernel(double* a, double* b, double* c, double* d, int n) {
    // GPU에서 실행되는 kernel 코드
}

void launch_tdma(double* a, double* b, double* c, double* d, int n) {
    tdma_kernel<<<128, 256>>>(a, b, c, d, n);
}
```

정리하면:

| 항목 | 내용 |
|---|---|
| 목적 | CUDA 구현 파일 |
| 주 컴파일러 | `nvcc` |
| 포함 가능 | `__global__`, `__device__`, kernel launch `<<< >>>` |
| 예시 | `filtered_tdma.cu`, `kernel.cu`, `benchmark.cu` |

---

## 2. `.cuh`

`.cuh`는 **CUDA용 헤더 파일**이다.

보통 다음 내용을 포함한다.

- CUDA kernel 선언
- `__device__` 함수 선언
- `__host__ __device__` 함수 선언
- CUDA 내부 구현에서 공유할 inline 함수
- CUDA kernel에서 사용할 구조체 또는 상수

예시:

```cpp
// solver.cuh

#pragma once

__global__
void tdma_kernel(double* a, double* b, double* c, double* d, int n);

__device__
double compute_pivot(double b, double a, double c);
```

정리하면:

| 항목 | 내용 |
|---|---|
| 목적 | CUDA kernel/device 함수 선언용 헤더 |
| 주 컴파일러 | 주로 `nvcc`가 include |
| 포함 가능 | `__global__`, `__device__`, `__host__ __device__` |
| 예시 | `filtered_tdma_kernel.cuh` |

즉, `.cuh`는 **CUDA 내부 구현을 위한 헤더**라고 보면 된다.

---

## 3. `.hpp`

`.hpp`는 **C++용 헤더 파일**이다.

보통 다음 내용을 포함한다.

- 외부 사용자에게 공개할 API
- C++ `namespace`
- `class`, `struct`
- template
- solver 설정 구조체
- CPU/GPU wrapper 함수 선언

예시:

```cpp
// filtered_tdma.hpp

#pragma once

namespace filtered_tdma {

struct SolverConfig {
    int n;
    int batch;
    int J;
};

void solve(
    double* a,
    double* b,
    double* c,
    double* d,
    const SolverConfig& config
);

}
```

정리하면:

| 항목 | 내용 |
|---|---|
| 목적 | C++ public API 헤더 |
| 주 컴파일러 | `g++`, `mpicxx`, `nvcc` 모두 가능 |
| 포함 가능 | `class`, `struct`, `namespace`, `template` |
| 예시 | `filtered_tdma.hpp`, `solver_config.hpp` |

즉, `.hpp`는 **사용자가 include하는 입구**라고 보면 된다.

---

## 4. 한 줄 요약

```text
.hpp  = 외부에 공개하는 C++ API 헤더
.cuh  = CUDA kernel/device 함수 선언용 헤더
.cu   = CUDA kernel 구현 및 kernel launch 코드
```

---

## 5. Filtered_TDMA 기준 추천 구조

Filtered_TDMA 같은 CUDA 기반 라이브러리라면 다음처럼 나누는 것이 깔끔하다.

```text
include/
└── filtered_tdma.hpp          # 사용자가 include하는 public API

src/
├── filtered_tdma.cu           # solve() 구현, kernel launch
├── filtered_tdma_kernel.cuh   # CUDA kernel 선언
└── filtered_tdma_kernel.cu    # CUDA kernel 실제 구현
```

역할은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `filtered_tdma.hpp` | 외부 공개 API |
| `filtered_tdma_kernel.cuh` | CUDA kernel 및 device 함수 선언 |
| `filtered_tdma.cu` | host wrapper, kernel launch |
| `filtered_tdma_kernel.cu` | 실제 GPU kernel 구현 |

---

## 6. 실전에서 주의할 점

### `.cpp`에서 kernel launch를 직접 쓰면 안 됨

다음 문법은 CUDA 문법이므로 `nvcc`가 컴파일하는 `.cu` 파일 안에 있어야 한다.

```cpp
kernel<<<blocks, threads>>>(...);
```

따라서 kernel launch가 들어가는 코드는 보통 `.cu`에 둔다.

---

### public API와 CUDA 내부 구현을 분리하는 것이 좋음

사용자는 다음처럼 단순하게 호출하게 만들고,

```cpp
filtered_tdma::solve(...);
```

내부에서는 `.cu` 파일에서 CUDA kernel을 launch하는 구조가 좋다.

---

### 이름 충돌 방지를 위해 namespace를 사용

다른 TDMA 라이브러리와 함께 비교할 수 있으므로 public API는 namespace로 감싸는 것이 좋다.

```cpp
namespace filtered_tdma {
    void solve(...);
}
```

이렇게 하면 PaScaL_TDMA, DistD2-TDS 등 다른 라이브러리와 함께 링크할 때 symbol 충돌 가능성을 줄일 수 있다.
