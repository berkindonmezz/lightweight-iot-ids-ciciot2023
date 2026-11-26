# --- ASAMA 1: Derleme (Builder) ---
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY models/final_edge/ /build/models_final_edge/
COPY deploy/edge_docker/linux_c_bench.c /build/linux_c_bench.c

RUN mkdir -p /build/compiled_c && \
    gcc --version | head -n 1 > /build/compiled_c/gcc_version.txt && \
    for tag in top15_l0_baseline top15_l1_balanced top10_l2_light; do \
      gcc -O3 -fPIC -shared "/build/models_final_edge/${tag}_model.c" -o "/build/compiled_c/lib${tag}.so" -lm && \
      gcc -O3 "/build/models_final_edge/${tag}_model.c" /build/linux_c_bench.c -o "/build/compiled_c/${tag}_bench_linux" -lm ; \
    done

# --- ASAMA 2: Minimal Edge Calisma Imaji (gcc yok!) ---
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

RUN pip install --no-cache-dir numpy onnxruntime

WORKDIR /app

COPY models/final_edge/ /app/models_final_edge/
COPY --from=builder /build/compiled_c/ /app/compiled_c/
COPY deploy/edge_docker/edge_benchmark_sample.npz /app/edge_benchmark_sample.npz
COPY deploy/edge_docker/run_edge_benchmark.py /app/run_edge_benchmark.py

CMD ["python", "/app/run_edge_benchmark.py"]