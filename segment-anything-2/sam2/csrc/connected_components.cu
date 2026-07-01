// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// adapted from https://github.com/zsef123/Connected_components_PyTorch
// with license found in the LICENSE_cctorch file in the root directory.

// 引入 PyTorch CUDA 相關的標頭檔，用於取得當前的 CUDA Stream
#include <ATen/cuda/CUDAContext.h>
// 引入基本的 CUDA 核心定義
#include <cuda.h>
// 引入 CUDA 執行時期 API (如記憶體管理、核心啟動等)
#include <cuda_runtime.h>
// 引入 PyTorch C++ 擴充套件介面，允許 C++ 與 Python 互動
#include <torch/extension.h>
//提供與 TorchScript（PyTorch 的 JIT 即時編譯器）系統的相容性，讓這個自訂的 CUDA 連通域標記演算法能夠被打包進不需要 Python 環境就能執行的獨立模型檔
#include <torch/script.h>
// 引入 C++ 標準函式庫的 vector 容器
#include <vector>

// 2d 影像處理設定
// 定義 GPU 執行緒區塊 (Thread Block) 的高度為 16 個執行緒
#define BLOCK_ROWS 16
// 定義 GPU 執行緒區塊的寬度為 16 個執行緒
#define BLOCK_COLS 16

// 定義命名空間 cc2d ( 二維連通元件Connected Components 2D)，封裝 GPU 上的函式
namespace cc2d {
// 樣板函式：檢查某個整數 (bitmap) 在指定位置 (pos) 的位元是否為 1
template <typename T>
__device__ __forceinline__ unsigned char hasBit(T bitmap, unsigned char pos) {
// 將 bitmap 往右移 pos 位，然後與 1 進行 AND 運算，取出該位元的值
  return (bitmap >> pos) & 1;
}
// 尋找所屬群組的根節點 (並查集演算法中的 Find 操作)
__device__ int32_t find(const int32_t* s_buf, int32_t n) {
// 如果節點 n 不等於它的父節點，代表它不是根節點，繼續往上找
  while (s_buf[n] != n)
    n = s_buf[n];
  return n;
}
// 尋找根節點，並同時進行「路徑壓縮 (Path Compression)」，加速未來的查詢
//修飾詞 回傳型別 函式名稱(參數)
__device__ int32_t find_n_compress(int32_t* s_buf, int32_t n) {
  const int32_t id = n; // 記錄最初查詢的節點
  // 迴圈往上找根節點
  while (s_buf[n] != n) {
    n = s_buf[n]; // 往上走一層
    s_buf[id] = n; // 直接將最初節點的父節點更新為當前找到的祖先節點
  }
  return n; // 回傳最終的根節點
}
//void = 沒有回傳值
//s_buf=並查集的父節點表（parent buffer），用來記錄每個像素目前歸屬哪個連通元件。
// 將兩個節點 a 和 b 所在的群組合併 (Union-Find 演算法中的 Union 操作)
__device__ void union_(int32_t* s_buf, int32_t a, int32_t b) {
//宣告一個布林變數 done
  bool done;
  do {
  // 先找出 a 和 b(兩個集合) 各自的根節點
    a = find(s_buf, a);
    b = find(s_buf, b);
// 為了避免循環，規定編號較小的節點作為根節點
    if (a < b) {
	// atomicMin 去看 s_buf[b] 目前是多少，和 a 比較，然後把較小值寫回 s_buf[b]。回傳修改前的舊值old
      int32_t old = atomicMin(s_buf + b, a);
	  // 如果原本的值就是 b，代表合併成功；否則代表有其他執行緒改了它，需要重試
      done = (old == b);
      b = old; // 更新 b 繼續檢查
    } else if (b < a) {
	// 邏輯同上，只是方向相反 (a 比較大，把 a 指向 b)
      int32_t old = atomicMin(s_buf + a, b);
      done = (old == a);
      a = old;
    } else
      done = true; // a == b，代表已經在同一個群組內，不需合併

  } while (!done); // 直到合併成功為止
}

//__global__：GPU kernel（從 CPU 呼叫，在 GPU 上執行）
//__device__：GPU 裡的普通函數（只能被 GPU 內部呼叫）

// 步驟 1：初始化每個像素的標籤
__global__ void
init_labeling(int32_t* label, const uint32_t W, const uint32_t H) {
// 計算當前執行緒負責的像素在影像中的 2D 座標 (這裡一次跳 2 格，是用於區塊優化的演算法)
//thread（執行緒）指的是 GPU 上最基本的計算單位。
//thread（執行緒）＝一個工人
//block（執行緒區塊）＝一組工人小組
//grid（網格）＝整間工廠的所有小組集合
//CUDA runtime 自動提供的內建變數（built-in variable）。

//blockIdx.y現在在哪個 block
//blockDim.y每個 block 有幾列 threads
//threadIdx.y 在自己的 block 裡面是第幾列

  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2; //算出這個 thread 在整個 grid 裡的 全域 y 座標。把座標間隔放大 2 倍。
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  // 將 2D 座標轉換為 1D 陣列的索引值
  //在 CUDA / PyTorch 的底層，資料通常是 平鋪成一條長陣列，GPU 只知道要讀寫哪個「線性位置」
  const uint32_t idx = row * W + col;
// 確保沒有超出影像邊界
  if (row < H && col < W)
    label[idx] = idx; // 初始狀態下，每個像素的標籤就是它自己的位置索引
}
// 步驟 2：掃描鄰居並合併相連的區塊
__global__ void
merge(uint8_t* img, int32_t* label, const uint32_t W, const uint32_t H) {
// 計算像素座標，一樣是 2x2 的步長設計
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t idx = row * W + col;
// 超出邊界就直接返回
  if (row >= H || col >= W)
    return;
//周圍沒有任何可連通的前景所有 bit 都是 0
  uint32_t P = 0; // 用一個整數來當作位元遮罩，記錄周圍像素的狀態

  if (img[idx])
    P |= 0x777;
  if (row + 1 < H && img[idx + W])
    P |= 0x777 << 4;
  if (col + 1 < W && img[idx + 1])
    P |= 0x777 << 1;

  if (col == 0)
    P &= 0xEEEE;
  if (col + 1 >= W)
    P &= 0x3333;
  else if (col + 2 >= W)
    P &= 0x7777;

  if (row == 0)
    P &= 0xFFF0;
  if (row + 1 >= H)
    P &= 0xFF;

  if (P > 0) {
    // If need check about top-left pixel(if flag the first bit) and hit the
    // top-left pixel
    if (hasBit(P, 0) && img[idx - W - 1]) {
      union_(label, idx, idx - 2 * W - 2); // top left block
    }

    if ((hasBit(P, 1) && img[idx - W]) || (hasBit(P, 2) && img[idx - W + 1]))
      union_(label, idx, idx - 2 * W); // top bottom block

    if (hasBit(P, 3) && img[idx + 2 - W])
      union_(label, idx, idx - 2 * W + 2); // top right block

    if ((hasBit(P, 4) && img[idx - 1]) || (hasBit(P, 8) && img[idx + W - 1]))
      union_(label, idx, idx - 2); // just left block
  }
}

__global__ void compression(int32_t* label, const int32_t W, const int32_t H) {
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t idx = row * W + col;

  if (row < H && col < W)
    find_n_compress(label, idx);
}

__global__ void final_labeling(
    const uint8_t* img,
    int32_t* label,
    const int32_t W,
    const int32_t H) {
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  const uint32_t idx = row * W + col;

  if (row >= H || col >= W)
    return;

  int32_t y = label[idx] + 1;

  if (img[idx])
    label[idx] = y;
  else
    label[idx] = 0;

  if (col + 1 < W) {
    if (img[idx + 1])
      label[idx + 1] = y;
    else
      label[idx + 1] = 0;

    if (row + 1 < H) {
      if (img[idx + W + 1])
        label[idx + W + 1] = y;
      else
        label[idx + W + 1] = 0;
    }
  }

  if (row + 1 < H) {
    if (img[idx + W])
      label[idx + W] = y;
    else
      label[idx + W] = 0;
  }
}

__global__ void init_counting(
    const int32_t* label,
    int32_t* count_init,
    const int32_t W,
    const int32_t H) {
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y);
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x);
  const uint32_t idx = row * W + col;

  if (row >= H || col >= W)
    return;

  int32_t y = label[idx];
  if (y > 0) {
    int32_t count_idx = y - 1;
    atomicAdd(count_init + count_idx, 1);
  }
}

__global__ void final_counting(
    const int32_t* label,
    const int32_t* count_init,
    int32_t* count_final,
    const int32_t W,
    const int32_t H) {
  const uint32_t row = (blockIdx.y * blockDim.y + threadIdx.y);
  const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x);
  const uint32_t idx = row * W + col;

  if (row >= H || col >= W)
    return;

  int32_t y = label[idx];
  if (y > 0) {
    int32_t count_idx = y - 1;
    count_final[idx] = count_init[count_idx];
  } else {
    count_final[idx] = 0;
  }
}

} // namespace cc2d

std::vector<torch::Tensor> get_connected_componnets(
    const torch::Tensor& inputs) {
  AT_ASSERTM(inputs.is_cuda(), "inputs must be a CUDA tensor");
  AT_ASSERTM(inputs.ndimension() == 4, "inputs must be [N, 1, H, W] shape");
  AT_ASSERTM(
      inputs.scalar_type() == torch::kUInt8, "inputs must be a uint8 type");

  const uint32_t N = inputs.size(0);
  const uint32_t C = inputs.size(1);
  const uint32_t H = inputs.size(2);
  const uint32_t W = inputs.size(3);

  AT_ASSERTM(C == 1, "inputs must be [N, 1, H, W] shape");
  AT_ASSERTM((H % 2) == 0, "height must be an even number");
  AT_ASSERTM((W % 2) == 0, "width must be an even number");

  // label must be uint32_t
  auto label_options =
      torch::TensorOptions().dtype(torch::kInt32).device(inputs.device());
//建立一個形狀為 [batch size, C, H, W] 的 4D tensor，並初始化為 0
//宣告一個 PyTorch Tensor 變數
//torch::PyTorch C++ library（LibTorch）的 namespace
  torch::Tensor labels = torch::zeros({N, C, H, W}, label_options); 
  torch::Tensor counts_init = torch::zeros({N, C, H, W}, label_options);
  torch::Tensor counts_final = torch::zeros({N, C, H, W}, label_options);
//由 <cuda_runtime.h>調用
  dim3 grid = dim3(
      ((W + 1) / 2 + BLOCK_COLS - 1) / BLOCK_COLS,
      ((H + 1) / 2 + BLOCK_ROWS - 1) / BLOCK_ROWS);
  dim3 block = dim3(BLOCK_COLS, BLOCK_ROWS);
  dim3 grid_count =
      dim3((W + BLOCK_COLS) / BLOCK_COLS, (H + BLOCK_ROWS) / BLOCK_ROWS);
  dim3 block_count = dim3(BLOCK_COLS, BLOCK_ROWS);
  // 取得目前 PyTorch 正在使用的 CUDA 執行流 (Stream)，確保非同步執行的正確性include <ATen/cuda/CUDAContext.h>
  //<cuda_runtime.h> 提供 cudaStream_t 這個資料型態（Data Type）的定義。
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 

  for (int n = 0; n < N; n++) {
    uint32_t offset = n * H * W;
//nvcc 編譯器（Compiler） 結合 <cuda_runtime.h> 提供的環境定義所共同實現
    cc2d::init_labeling<<<grid, block, 0, stream>>>( 
        labels.data_ptr<int32_t>() + offset, W, H);
    cc2d::merge<<<grid, block, 0, stream>>>(
        inputs.data_ptr<uint8_t>() + offset,
        labels.data_ptr<int32_t>() + offset,
        W,
        H);
    cc2d::compression<<<grid, block, 0, stream>>>(
        labels.data_ptr<int32_t>() + offset, W, H);
    cc2d::final_labeling<<<grid, block, 0, stream>>>(
        inputs.data_ptr<uint8_t>() + offset,
        labels.data_ptr<int32_t>() + offset,
        W,
        H);

    // get the counting of each pixel
    cc2d::init_counting<<<grid_count, block_count, 0, stream>>>(
        labels.data_ptr<int32_t>() + offset,
        counts_init.data_ptr<int32_t>() + offset,
        W,
        H);
    cc2d::final_counting<<<grid_count, block_count, 0, stream>>>(
        labels.data_ptr<int32_t>() + offset,
        counts_init.data_ptr<int32_t>() + offset,
        counts_final.data_ptr<int32_t>() + offset,
        W,
        H);
  }

  // returned values are [labels, counts]
  std::vector<torch::Tensor> outputs;
  outputs.push_back(labels);
  outputs.push_back(counts_final);
  return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "get_connected_componnets",
      &get_connected_componnets,
      "get_connected_componnets");
}
