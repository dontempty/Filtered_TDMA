scontrol show node -o | awk '
BEGIN { 
    # 헤더 출력 (CPU 노드 포함)
    printf "%-10s | %-15s | %-12s | %-16s | %-16s\n", "Node", "Partition", "Model", "GPU (Free/Tot)", "CPU (Free/Tot)";
    print "-----------+-----------------+--------------+------------------+------------------";
}
{
    # 변수 초기화
    node="N/A"; part="N/A"; gpu_tot=0; gpu_use=0; cpu_tot=0; cpu_alloc=0; gpu_model="-";

    # 각 필드 파싱
    for(i=1;i<=NF;i++){
        if($i ~ /^NodeName=/)   { split($i,a,"="); node=a[2] }
        if($i ~ /^Partitions=/) { split($i,a,"="); part=a[2] }
        if($i ~ /^CPUTot=/)     { split($i,a,"="); cpu_tot=a[2] }
        if($i ~ /^CPUAlloc=/)   { split($i,a,"="); cpu_alloc=a[2] }
        
        # GPU 총 개수 파싱
        if($i ~ /^Gres=gpu/) { 
            n=split($i, g, ":"); 
            if (g[n] ~ /^[0-9]+$/) gpu_tot=g[n]; 
            else if (g[n-1] ~ /^[0-9]+$/) gpu_tot=g[n-1];
        }

        # 사용 중인 GPU 개수 파싱
        if($i ~ /^AllocTRES=/) {
            if(match($i, /gres\/gpu=([0-9]+)/, m)) {
                gpu_use = substr($i, RSTART+9, RLENGTH-9);
            }
        }
    }

    # --- 모델 판별 로직 ---
    if (gpu_tot > 0) {
        gpu_model = "Unknown";
        if (node ~ /^gpu/) {
            match(node, /[0-9]+/, n_arr);
            n_num = n_arr[0] + 0;
            if (n_num >= 1 && n_num <= 9)        gpu_model = "V100 (PCIe)";
            else if (n_num >= 10 && n_num <= 24) gpu_model = "V100 (SXM)";
            else if (n_num == 25 || n_num == 26 || n_num == 29) gpu_model = "V100 (SXM)";
            else if (n_num >= 30 && n_num <= 45) gpu_model = "A100 (SXM)";
            else if (n_num >= 46 && n_num <= 50) gpu_model = "H200 (SXM)";
            else if (n_num >= 51 && n_num <= 52) gpu_model = "H100 (SXM)";
            else if (n_num >= 53 && n_num <= 56) gpu_model = "H200 (SXM)";
            else if (n_num >= 57 && n_num <= 59) gpu_model = "H100 (SXM)";
        }
        else if (node ~ /^gdebug/ || node ~ /^jupyter/) {
             gpu_model = "V100 (SXM)";
        }
    } else {
        # GPU가 없는 경우
        gpu_model = "CPU Only";
    }

    # 잔여량 계산
    gpu_free = gpu_tot - gpu_use;
    cpu_free = cpu_tot - cpu_alloc;

    # 모든 노드 출력 (GPU가 있으면 수치 표시, 없으면 - 표시)
    if (gpu_tot > 0) {
        gpu_disp = sprintf("%2d / %2d", gpu_free, gpu_tot);
    } else {
        gpu_disp = "     -     ";
    }

    printf "%-10s | %-15s | %-12s | %-16s | %3d / %3d\n", \
    node, part, gpu_model, gpu_disp, cpu_free, cpu_tot;

}'