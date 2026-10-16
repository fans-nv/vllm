// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
#pragma once

namespace vllm::minimax_m3_fused_ops {

// ICP_DEVICE_PLAN_ABI 4: the unsplit OnlyScoreIcp prefill work plan, one CTA
// per chunk slot, derived from the same exact query_start_loc/seq_lens.
// segments rows: 0 qo_segment_offsets, 1 kv_segment_offsets, 2 qo_segment_lens,
// 3 kv_segment_lens, 4 qo_offset. work row: [range | info | scratch x2].
// Items are unsplit, so the scorer takes no KV tile ranges.
// header[3] is the slot origin t0.
constexpr int kIcpPlanThreads = 256;
constexpr int kIcpPlanQoTile = 128;
constexpr int kIcpPlanKvTile = 256;

struct IcpDevicePlanParams {
  int32_t* segments = nullptr;  // [slots, 5, max_segments + 1]
  uint64_t* work = nullptr;     // [slots, num_ctas + 3 * max_work]
  int32_t* header = nullptr;    // [slots, 4]: n, first request, work items, t0
  int64_t seg_slot_stride = 0;
  int64_t work_slot_stride = 0;
  int32_t seg_row_stride = 0;
  int32_t max_segments = 0;
  int32_t max_work = 0;
  int32_t num_ctas = 0;
  int32_t num_heads = 0;
  int32_t blocks = 0;
};

// The first sparse producer also refreshes the retained live metadata. Extra
// CTAs in this SAME grid own it; there is no auxiliary kernel or ATen path.
// Inputs are scheduler buffers prepared before this launch. In particular,
// metadata positions are NOT the RoPE positions this grid reads.
struct IcpLiveMetadataParams {
  int32_t const* query_start_loc = nullptr;
  int32_t const* seq_lens = nullptr;
  int64_t* positions = nullptr;
  uint8_t* active = nullptr;
  int32_t* local_nvalid = nullptr;
  int32_t* local_forced = nullptr;
  int32_t* global_nvalid = nullptr;
  int32_t* forced = nullptr;
  int32_t* n_ordinary = nullptr;
  int32_t* candidates = nullptr;  // [rows, 4, 16, 2], bitcast C4 carrier
  int32_t* qo_offsets = nullptr;  // [slots, qo_stride], unused cells not read
  int32_t num_reqs = 0;
  int32_t num_rows = 0;
  int32_t rank = 0;
  int32_t chunk_width = 0;
  int32_t qo_stride = 0;
  int32_t qo_slots = 0;
  int32_t meta_blocks = 0;
  IcpDevicePlanParams plan;
};

#ifdef VLLM_MINIMAX_M3_NVFP4
__device__ __forceinline__ int icpRequestForRow(IcpLiveMetadataParams const& p,
                                                int row) {
  // upper_bound of request ends, including repeated ends of empty requests.
  int lo = 0, hi = p.num_reqs;
  while (lo < hi) {
    int const mid = (lo + hi) >> 1;
    if (__ldg(p.query_start_loc + mid + 1) <= row)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ __forceinline__ void icpLiveMetadataBlock(
    IcpLiveMetadataParams const& p) {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
  #endif
  int const lane = threadIdx.x % 32;
  int const warps = blockDim.x / 32;
  int const stride = p.meta_blocks * warps;
  for (int row = blockIdx.x * warps + threadIdx.x / 32; row < p.num_rows;
       row += stride) {
    int live = 0;
    if (lane == 0) {
      int64_t position = 0;
      int local = 0, global = 0, forced = -1, local_forced = -1;
      int const req = icpRequestForRow(p, row);
      if (req < p.num_reqs) {
        int const begin = __ldg(p.query_start_loc + req);
        int const end = __ldg(p.query_start_loc + req + 1);
        int const seq = __ldg(p.seq_lens + req);
        int const query = end - begin;
        int64_t const pos = static_cast<int64_t>(seq) - query + row - begin;
        live =
            query > 0 && seq >= query && row >= begin && row < end && pos >= 0;
        if (live) {
          position = pos;
          int const visible = static_cast<int>(pos + 1 < seq ? pos + 1 : seq);
          forced = static_cast<int>(pos / 128);
          global = forced + 1;  // Q8KV4 consumes the uncapped global count.
          local = visible / 128 + (visible % 128 > p.rank * 64);
          local_forced = forced < local ? forced : -1;
          if (p.qo_offsets != nullptr) {
            int const slot = row / p.chunk_width;
            int const t0 = slot * p.chunk_width;
            if (row == max(begin, t0)) {
              int const first_req = icpRequestForRow(p, t0);
              if (!(slot < p.qo_slots && req - first_req >= 0 &&
                    req - first_req < p.qo_stride))
                __trap();
              p.qo_offsets[static_cast<int64_t>(slot) * p.qo_stride + req -
                           first_req] = seq - (end - row);
            }
          }
        }
      }
      p.positions[row] = position;
      p.active[row] = static_cast<uint8_t>(live);
      p.local_nvalid[row] = local;
      p.local_forced[row] = local_forced;
      p.global_nvalid[row] = global;
      p.forced[row] = forced;
      p.n_ordinary[row] = live ? min(15, forced) : 0;
    }
    live = __shfl_sync(FINAL_MASK, live, 0);
    if (!live) {
      // Only the ID word defines validity. Preserve arbitrary score payloads,
      // including NaNs, and never allow an old valid ID in skipped tail waves.
      for (int candidate = lane; candidate < 4 * 16; candidate += 32) {
        p.candidates[(static_cast<int64_t>(row) * 4 * 16 + candidate) * 2 + 1] =
            -1;
      }
    }
  }
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
  #endif
}

// Same packing as plan.cuh pack_work_info / pack_work_range.
__device__ __forceinline__ uint64_t icpPackWorkInfo(int qo_tile, int head,
                                                    int batch) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(qo_tile)) << 32) |
         (static_cast<uint64_t>(head & 0xFFFF) << 16) |
         static_cast<uint64_t>(batch & 0xFFFF);
}

__device__ __forceinline__ uint64_t icpPackWorkRange(int start, int end) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(end)) << 32) |
         static_cast<uint64_t>(static_cast<uint32_t>(start));
}

// Inclusive scan of one int per thread; every thread of the block must call.
__device__ __forceinline__ int icpBlockInclusiveScan(int value, int* tmp,
                                                     int* total) {
  int const tid = threadIdx.x;
  tmp[tid] = value;
  __syncthreads();
  for (int d = 1; d < kIcpPlanThreads; d <<= 1) {
    int const add = tid >= d ? tmp[tid - d] : 0;
    __syncthreads();
    tmp[tid] += add;
    __syncthreads();
  }
  int const result = tmp[tid];
  *total = tmp[kIcpPlanThreads - 1];
  __syncthreads();
  return result;
}

// One chunk slot's plan: plan.cuh direct_greedy (causal, pack 1, qo tile 128)
// with the second pass replaced by a scatter.
__device__ __forceinline__ void icpDevicePlanBlock(
    IcpLiveMetadataParams const& p, int slot) {
  #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
  #endif
  __shared__ int s_cost[kIcpPlanThreads];
  __shared__ int s_count[kIcpPlanThreads];
  __shared__ int s_scan[kIcpPlanThreads];
  __shared__ int s_first, s_n, s_max_tiles;

  IcpDevicePlanParams const& q = p.plan;
  int const tid = threadIdx.x;
  int const nb = q.num_ctas;
  int32_t* const seg =
      q.segments + static_cast<int64_t>(slot) * q.seg_slot_stride;
  int32_t* const qo_off = seg;
  int32_t* const kv_off = seg + q.seg_row_stride;
  int32_t* const qo_len = seg + 2 * q.seg_row_stride;
  int32_t* const kv_len = seg + 3 * q.seg_row_stride;
  int32_t* const causal = seg + 4 * q.seg_row_stride;
  uint64_t* const range =
      q.work + static_cast<int64_t>(slot) * q.work_slot_stride;
  uint64_t* const info = range + nb;
  uint64_t* const scratch_info = info + q.max_work;
  uint64_t* const scratch_slot = scratch_info + q.max_work;
  int const t0 = slot * p.chunk_width;
  int const t1 = min((slot + 1) * p.chunk_width, p.num_rows);

  if (tid == 0) {
    int const live_end =
        p.num_reqs > 0 ? __ldg(p.query_start_loc + p.num_reqs) : 0;
    int const end = min(t1, live_end);
    int first = 0, n = 0;
    if (t0 < end) {
      first = icpRequestForRow(p, t0);
      n = icpRequestForRow(p, end - 1) - first + 1;
    }
    if (n > q.max_segments) __trap();
    s_first = first;
    s_n = n;
    s_max_tiles = 0;
    qo_off[0] = 0;
    kv_off[0] = 0;
  }
  if (tid < nb) {
    s_cost[tid] = 165;  // plan.cuh kSMGlobalOverhead
    s_count[tid] = 0;
  }
  __syncthreads();
  int const first = s_first;
  int const n = s_n;

  int carry = 0;
  for (int base = 0; base < n; base += kIcpPlanThreads) {
    int const b = base + tid;
    int seq = 0;
    if (b < n) {
      int const req = first + b;
      int const begin = __ldg(p.query_start_loc + req);
      int const end = __ldg(p.query_start_loc + req + 1);
      seq = __ldg(p.seq_lens + req);
      int const lo = max(begin, t0);
      int const hi = min(end, t1);
      qo_off[b + 1] = hi - t0;
      qo_len[b] = hi - lo;
      kv_len[b] = seq;
      causal[b] = seq - (end - lo);
      atomicMax(&s_max_tiles, (hi - lo + kIcpPlanQoTile - 1) / kIcpPlanQoTile);
    }
    int total = 0;
    int const inclusive = icpBlockInclusiveScan(seq, s_scan, &total);
    if (b < n) kv_off[b + 1] = carry + inclusive;
    carry += total;
  }
  __syncthreads();

  int const max_tiles = s_max_tiles;
  int work = 0;
  for (int qt = max_tiles - 1; qt >= 0; --qt) {
    for (int b = 0; b < n; ++b) {
      if (qt >= (qo_len[b] + kIcpPlanQoTile - 1) / kIcpPlanQoTile) continue;
      int eff = (qt + 1) * kIcpPlanQoTile + causal[b];
      int const kv = kv_len[b];
      if (eff > kv) eff = kv;
      if (eff <= 0) continue;
      int const iters = (eff + kIcpPlanKvTile - 1) / kIcpPlanKvTile;
      // plan.cuh kPerIterOverhead / kTileGlobalOverhead, float as there.
      int const tile_cost = static_cast<int>(43.0f * iters + 110.0f);
      for (int h_off = 0; h_off < q.num_heads;) {
        int const batch = min(q.num_heads - h_off, nb);
        int rank = 0;
        if (tid < nb) {
          int const mine = s_cost[tid];
          for (int i = 0; i < nb; ++i) {
            int const other = s_cost[i];
            rank += (other < mine || (other == mine && i < tid)) ? 1 : 0;
          }
        }
        bool const take = tid < nb && rank < batch;
        __syncthreads();
        if (take) {
          int const item = work + rank;
          if (item >= q.max_work) __trap();
          int const pos = s_count[tid]++;
          s_cost[tid] += tile_cost;
          scratch_info[item] = icpPackWorkInfo(qt, h_off + rank, b);
          scratch_slot[item] =
              (static_cast<uint64_t>(tid) << 32) |
              static_cast<uint64_t>(static_cast<uint32_t>(pos));
        }
        __syncthreads();
        work += batch;
        h_off += batch;
      }
    }
  }

  int const count = tid < nb ? s_count[tid] : 0;
  int total = 0;
  int const exclusive = icpBlockInclusiveScan(count, s_scan, &total) - count;
  if (tid < nb) range[tid] = icpPackWorkRange(exclusive, exclusive + count);
  s_cost[tid] = exclusive;
  __syncthreads();
  for (int item = tid; item < work; item += kIcpPlanThreads) {
    uint64_t const where = scratch_slot[item];
    int const dst = s_cost[where >> 32] + static_cast<int>(where & 0xFFFFFFFFu);
    info[dst] = scratch_info[item];
  }
  if (tid == 0) {
    int32_t* const header = q.header + static_cast<int64_t>(slot) * 4;
    header[0] = n;
    header[1] = first;
    header[2] = work;
    header[3] = t0;
  }
}

#endif

struct IcpWriterParams {
  int32_t block_tokens = 0;
  int32_t rows_per_rank = 0;
  int32_t rank = 0;
  int32_t world_size = 0;
  int64_t page_stride = 0;
  int64_t row_stride = 0;
  IcpLiveMetadataParams metadata;
};

}  // namespace vllm::minimax_m3_fused_ops
