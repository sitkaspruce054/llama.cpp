#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-cpu.h"

// Measures round-trip relative MSE: E[||x - dequant(quant(x))||^2] / E[||x||^2]
// For 3-bit TurboQuant the paper guarantees distortion <= 0.03 (normalized).
static float roundtrip_rel_mse(const ggml_type_traits * qfns, int n_elements, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int blk  = (int)qfns->blck_size;
    const int n_blocks = n_elements / blk;
    const size_t q_bytes = (size_t)n_blocks * qfns->type_size;

    std::vector<float>   x(n_elements), out(n_elements);
    std::vector<uint8_t> q(q_bytes);

    double total_mse = 0.0, total_norm2 = 0.0;

    for (int v = 0; v < n_blocks; v++) {
        float * xb = x.data() + v * blk;
        for (int j = 0; j < blk; j++) xb[j] = dist(rng);
    }

    qfns->from_float_ref(x.data(), q.data(), n_elements);
    qfns->to_float(q.data(), out.data(), n_elements);

    for (int i = 0; i < n_elements; i++) {
        double diff = x[i] - out[i];
        total_mse   += diff * diff;
        total_norm2 += (double)x[i] * x[i];
    }

    return (float)(total_mse / total_norm2);
}

int main(void) {
    ggml_cpu_init();

    const ggml_type type = GGML_TYPE_TBQ3_0;
    const ggml_type_traits * qfns = ggml_get_type_traits(type);

    if (!qfns->from_float_ref || !qfns->to_float) {
        fprintf(stderr, "tbq3_0 missing from_float_ref or to_float\n");
        return 1;
    }

    const int n_elements = 32 * qfns->blck_size; // 32 blocks
    const float rel_mse = roundtrip_rel_mse(qfns, n_elements, 42);

    printf("tbq3_0 round-trip relative MSE over %d elements (%d blocks): %.5f\n",
           n_elements, n_elements / (int)qfns->blck_size, rel_mse);

    const float threshold = 0.05f;
    if (rel_mse > threshold) {
        fprintf(stderr, "FAILED: relative MSE %.5f exceeds threshold %.5f\n", rel_mse, threshold);
        return 1;
    }

    printf("PASSED\n");
    return 0;
}
