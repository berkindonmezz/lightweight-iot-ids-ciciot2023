#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

extern double ids_predict_prob(const float *raw_x);

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static void read_linux_rss_mb(double *cur_mb, double *peak_mb) {
    *cur_mb = -1.0;
    *peak_mb = -1.0;
    FILE *fp = fopen("/proc/self/status", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            unsigned long kb = 0;
            if (sscanf(line, "VmRSS: %lu kB", &kb) == 1) {
                *cur_mb = (double)kb / 1024.0;
            } else if (sscanf(line, "VmHWM: %lu kB", &kb) == 1) {
                *peak_mb = (double)kb / 1024.0;
            }
        }
        fclose(fp);
    }
    if (*peak_mb < 0.0) {
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            *peak_mb = (double)ru.ru_maxrss / 1024.0;
            if (*cur_mb < 0.0) *cur_mb = *peak_mb;
        }
    }
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Kullanim: bench <sample.bin> <n_samples> <n_features> [loops]\n");
        return 1;
    }
    const char *bin_path = argv[1];
    int n_samples = atoi(argv[2]);
    int n_feat = atoi(argv[3]);
    int loops = (argc >= 5) ? atoi(argv[4]) : 50;

    FILE *fp = fopen(bin_path, "rb");
    if (!fp) {
        fprintf(stderr, "Dosya acilamadi: %s\n", bin_path);
        return 2;
    }

    float *data = (float *)malloc((size_t)n_samples * n_feat * sizeof(float));
    double *lats = (double *)malloc((size_t)n_samples * sizeof(double));
    if (!data || !lats) {
        fclose(fp);
        return 3;
    }

    size_t got = fread(data, sizeof(float), (size_t)n_samples * n_feat, fp);
    fclose(fp);
    if ((int)got < n_samples * n_feat) {
        n_samples = (int)(got / n_feat);
    }

    /* Warm-up */
    volatile double sink = 0.0;
    int warmup = n_samples < 300 ? n_samples : 300;
    for (int i = 0; i < warmup; ++i) {
        sink += ids_predict_prob(data + (size_t)i * n_feat);
    }

    /* Tekil akis (batch=1) gecikme olcumu */
    for (int i = 0; i < n_samples; ++i) {
        double t0 = now_us();
        double p = ids_predict_prob(data + (size_t)i * n_feat);
        double t1 = now_us();
        lats[i] = t1 - t0;
        sink += p;
    }

    qsort(lats, (size_t)n_samples, sizeof(double), cmp_double);
    double p50_us = lats[(int)(0.50 * (n_samples - 1))];
    double p95_us = lats[(int)(0.95 * (n_samples - 1))];
    double p99_us = lats[(int)(0.99 * (n_samples - 1))];

    /* Throughput olcumu */
    double tb0 = now_us();
    long long total_inf = (long long)loops * n_samples;
    for (int l = 0; l < loops; ++l) {
        for (int i = 0; i < n_samples; ++i) {
            sink += ids_predict_prob(data + (size_t)i * n_feat);
        }
    }
    double tb1 = now_us();
    double elapsed_sec = (tb1 - tb0) / 1e6;
    double pps = elapsed_sec > 0.0 ? ((double)total_inf / elapsed_sec) : 0.0;

    double cur_mb = -1.0, peak_mb = -1.0;
    read_linux_rss_mb(&cur_mb, &peak_mb);

    printf("{\"c_rss_cur_mb\": %.3f, \"c_rss_peak_mb\": %.3f, \"c_lat_p50_us\": %.3f, \"c_lat_p95_us\": %.3f, \"c_lat_p99_us\": %.3f, \"c_throughput_pps\": %.1f, \"checksum\": %.6f}\n",
           cur_mb, peak_mb, p50_us, p95_us, p99_us, pps, sink);

    free(data);
    free(lats);
    return 0;
}