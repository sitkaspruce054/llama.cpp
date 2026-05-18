#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"
#include "ggml-cpu.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants-tbq3.h"

static int g_failures = 0;

#define CHECK(cond, ...) \
    do { if (!(cond)) { fprintf(stderr, "FAILED: " __VA_ARGS__); fputc('\n', stderr); g_failures++; } } while(0)

// Verify TBQ3_SIGNS matches the canonical xorshift32(seed=42) generation.
// If this fails the CPU and CUDA backends will silently produce inconsistent results.
static void test_sign_consistency(void) {
    const int8_t signs[QK_TBQ3_0] = TBQ3_SIGNS_INIT;

    uint32_t state = 42;
    for (int i = 0; i < QK_TBQ3_0; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        int8_t expected = (state >> 31) ? (int8_t)1 : (int8_t)-1;
        CHECK(signs[i] == expected,
              "sign[%d]: got %d, xorshift32 says %d", i, signs[i], expected);
    }
    printf("sign consistency:    %s\n", g_failures == 0 ? "ok" : "FAILED");
}

// Verify TBQ3_CODEBOOK is sorted ascending and symmetric around zero.
static void test_codebook(void) {
    const float cb[8] = TBQ3_CODEBOOK_INIT;
    for (int i = 1; i < 8; i++) {
        CHECK(cb[i] > cb[i-1], "codebook not sorted at index %d", i);
    }
    for (int i = 0; i < 4; i++) {
        CHECK(fabsf(cb[i] + cb[7-i]) < 1e-6f,
              "codebook not symmetric: cb[%d]=%f cb[%d]=%f", i, cb[i], 7-i, cb[7-i]);
    }
    printf("codebook:            ok\n");
}

// Verify the 3-bit pack/unpack round-trips correctly at every position,
// including the boundary cases where a 3-bit field spans two bytes
// (bit_idx == 6 or bit_idx == 7, i.e. j % 8 == 2 or j % 8 == 5).
static void test_bit_packing(void) {
    int prev_failures = g_failures;

    for (int j = 0; j < QK_TBQ3_0; j++) {
        for (int idx = 0; idx < 8; idx++) {
            uint8_t qs[48] = {0};

            const int bit_pos  = j * 3;
            const int byte_idx = bit_pos >> 3;
            const int bit_idx  = bit_pos & 7;

            qs[byte_idx] |= (uint8_t)(idx << bit_idx);
            if (bit_idx > 5) {
                qs[byte_idx + 1] |= (uint8_t)(idx >> (8 - bit_idx));
            }

            int unpacked = (qs[byte_idx] >> bit_idx) & 0x7;
            if (bit_idx > 5) {
                unpacked |= (qs[byte_idx + 1] << (8 - bit_idx)) & 0x7;
            }

            CHECK(unpacked == idx,
                  "bit packing j=%d idx=%d: unpacked=%d (bit_idx=%d)",
                  j, idx, unpacked, bit_idx);
        }
    }
    printf("bit packing:         %s\n", g_failures == prev_failures ? "ok" : "FAILED");
}

// Measure round-trip relative MSE: E[||x - dequant(quant(x))||^2 / ||x||^2]
static float roundtrip_rel_mse(const ggml_type_traits * qfns, int n_elements, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const int blk     = (int)qfns->blck_size;
    const int n_blocks = n_elements / blk;
    const size_t q_bytes = (size_t)n_blocks * qfns->type_size;

    std::vector<float>   x(n_elements), out(n_elements);
    std::vector<uint8_t> q(q_bytes);

    for (int i = 0; i < n_elements; i++) x[i] = dist(rng);

    qfns->from_float_ref(x.data(), q.data(), n_elements);
    qfns->to_float(q.data(), out.data(), n_elements);

    double mse = 0.0, norm2 = 0.0;
    for (int i = 0; i < n_elements; i++) {
        double diff = x[i] - out[i];
        mse   += diff * diff;
        norm2 += (double)x[i] * x[i];
    }
    return (float)(mse / norm2);
}

int main(void) {
    ggml_cpu_init();

    test_sign_consistency();
    test_codebook();
    test_bit_packing();

    const ggml_type_traits * qfns = ggml_get_type_traits(GGML_TYPE_TBQ3_0);
    if (!qfns->from_float_ref || !qfns->to_float) {
        fprintf(stderr, "FAILED: tbq3_0 missing from_float_ref or to_float\n");
        return 1;
    }

    const int n_elements = 32 * (int)qfns->blck_size;
    const float rel_mse  = roundtrip_rel_mse(qfns, n_elements, 42);

    // Paper guarantees <= 0.03 distortion for 3-bit MSE quantizer.
    const float threshold = 0.05f;
    printf("round-trip rel MSE:  %.5f  (threshold %.2f)\n", rel_mse, threshold);
    CHECK(rel_mse < threshold, "relative MSE %.5f exceeds threshold %.5f", rel_mse, threshold);

    if (g_failures > 0) {
        fprintf(stderr, "%d test(s) FAILED\n", g_failures);
        return 1;
    }
    printf("ALL PASSED\n");
    return 0;
}
