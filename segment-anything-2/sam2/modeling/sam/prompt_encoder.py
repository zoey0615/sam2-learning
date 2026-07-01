# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#將人類提供的「提示」（Prompts）轉換為模型可以理解的高維向量（Embeddings），以便與圖像特徵進行互動並最終解碼出目標遮罩。
from typing import Optional, Tuple, Type # 從 typing 匯入型別註記工具：
# Optional：表示可能為 None；Tuple：表示固定長度的元組；Type：表示類別型別。

import torch
from torch import nn #匯入神經網路模組，方便使用 nn.Module / nn.Conv2d / nn.Embedding 等層。

from sam2.modeling.position_encoding import PositionEmbeddingRandom #將「純數學的二維平面座標」升維為「高維頻率表徵」。

from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(nn.Module): # 定義 PromptEncoder 類別，繼承 nn.Module，表示它是 PyTorch 模型的一部分。
    def __init__( # 建構子：初始化 prompt encoder 的所有子模組與超參數。
        self,
        embed_dim: int, #輸出提示的嵌入維度
        image_embedding_size: Tuple[int, int], #影像編碼後的空間大小 (H, W)。
        input_image_size: Tuple[int, int],#輸入到 image encoder 前的影像大小 (H, W)。
        mask_in_chans: int, #mask 編碼時中間層通道數。
        activation: Type[nn.Module] = nn.GELU, #mask 下採樣後所使用的啟動函數，預設 GELU。
    ) -> None: #不會傳回（return）任何東西
        super().__init__() # 呼叫父類別 nn.Module 的初始化，讓 PyTorch 能正確註冊參數與子模組。
        self.embed_dim = embed_dim #在 Hiera 骨幹網路的第一個階段（Stage），每個影像區塊（Patch）會被映射成一個長度為 144 的向量
        self.input_image_size = input_image_size # 儲存輸入影像的尺寸，之後座標位置編碼會用到。
        self.image_embedding_size = image_embedding_size # 儲存 image encoder 輸出的 feature map 空間尺寸。
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2) #初始化物件類別(256維/2):將「純數學的二維平面座標」升維為「高維頻率表徵」。
        self.num_point_embeddings: int = 4     # 定義點提示會使用 4 種可學習 embedding：正點、負點、框角點1、框角點2。
        point_embeddings = [  # 建立 4 個torch.nn.modules.sparse.Embedding 物件，可訓練的提示類別向量。(提示微調)
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)  #PyTorch 專門設計用來包裝「子模組列表」的特殊容器。
        self.not_a_point_embed = nn.Embedding(1, embed_dim) #為了湊齊矩陣長度而塞進來的『假點』

        self.mask_input_size = (  # mask 輸入所需空間大小，通常是 image embedding size 的 4 倍。
            4 * image_embedding_size[0],#影像編碼後的空間大小 (H, W)。
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential( #把多個神經網路層串聯
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),   # 第一層卷積：把單通道 mask 下採樣，同時提升到較少的中間通道。
            LayerNorm2d(mask_in_chans // 4), #mask 編碼時中間層通道數。
            activation(),
            
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2), # 第二層卷積：再次下採樣並提升到 mask_in_chans。
            LayerNorm2d(mask_in_chans),#為每一個通道準備一組可訓練的縮放參數（Gamma）和偏移參數 （Beta），用來在規範化後重新微調數據。
            activation(), # 啟動函數，例如 GELU。
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)   # 當沒有提供 mask prompt 時，使用這個可學習向量作為「沒有 mask」的表示。
#回傳與 image embedding 尺寸對齊的密集位置編碼。再加上 batch 維度，變成 1 x C x H x W。
    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)  #初始化物件類別(256維/2):將「純數學的二維平面座標」升維為「高維頻率表徵」。
                                                                    #傳回形狀為 1x(embedding_dim)x(embedding_h)x(embedding_w) 的位置編碼
    #把滑鼠點擊的「畫面上 2D 座標 (X, Y)」以及「你點的是左鍵（前景）還是右鍵（背景）的標籤」，融合成一組具備幾何位置與幾何語意的「高維度特徵密碼（Embedding）」
    def _embed_points(
        self,
        points: torch.Tensor, #points: [B, N, 2] -> 批次大小 x 點的數量 x (X, Y 座標)
        labels: torch.Tensor, #labels: [B, N]    -> 批次大小 x 每個點的標籤類型 (-1, 0, 1, 2, 3)
        pad: bool, #是否要補一個 dummy point
    ) -> torch.Tensor:
        """Embeds point prompts.點提示的編碼"""
        points = points + 0.5  # # 把整數像素座標平移 0.5，讓座標落在像素中心而不是左上角邊界。
        if pad:  # 如果有 box 且 point 數不足，補一個 dummy point，讓 batch 維度一致。
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device) # 建立一個全 0 的點座標，shape 是 [B, 1, 2]。
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device) # 補一個 label = -1，代表這不是有效點。
            points = torch.cat([points, padding_point], dim=1) # 在點序列尾端接上 padding point。
            labels = torch.cat([labels, padding_label], dim=1)    # 在 label 序列尾端接上 padding label。
        point_embedding = self.pe_layer.forward_with_coords( #特定的幾個點 (局部)互動式分割: 位置編碼器 B x 特徵數量N x C:負責將已經歸一化的座標，轉化為模型能理解的高維頻率特徵。
            points, self.input_image_size  # 儲存輸入影像的尺寸
        )

        point_embedding = torch.where(#torch.where(條件, 條件成立時的值, 條件不成立時的值)
            (labels == -1).unsqueeze(-1), #檢查標籤矩陣裡，哪些位置的數字等於 -1。這會得到一個由布林值（True / False）組成的矩陣。
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,  #為了湊齊矩陣長度而塞進來的『假點』
            #複製一模一樣的矩陣形狀，但把裡面的數值全部清空歸零。
            point_embedding, #幾何座標的特徵矩陣
        )
        point_embedding = torch.where(
            (labels == 0).unsqueeze(-1), #背景點（負點）在最後一個維度（Dim -1）強行擠出一個大小為 1 的全新軸，將形狀從 [B, N] 擴展為 [B, N, 1]。
            point_embedding + self.point_embeddings[0].weight, #背景點:空間幾何特徵（它在畫面的什麼位置）』，與『背景點（負點）的專屬語意徽章』進行加法融合
            point_embedding,
        )
        point_embedding = torch.where( # label == 1 表示正點，加入正點 embedding。
            (labels == 1).unsqueeze(-1),
            point_embedding + self.point_embeddings[1].weight, #前景點
            point_embedding,
        )
        point_embedding = torch.where(  # label == 2 表示 box 的第一個角點，加入對應 embedding。
            (labels == 2).unsqueeze(-1),
            point_embedding + self.point_embeddings[2].weight, #提示框左上角點
            point_embedding,
        )
        point_embedding = torch.where(  # label == 3 表示 box 的第二個角點，加入對應 embedding。
            (labels == 3).unsqueeze(-1),
            point_embedding + self.point_embeddings[3].weight, #提示框右下角點
            point_embedding,
        )
        return point_embedding   # 回傳點提示的最終 embedding。

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords( #特定的幾個點 (局部)互動式分割: 位置編碼器 B x 特徵數量N x C:負責將已經歸一化的座標，轉化為模型能理解的高維頻率特徵。
            coords, self.input_image_size # 儲存輸入影像的尺寸
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else: # 當沒有提供 mask prompt 時，提取權重矩陣，扭轉成四維幾何柱，複製空間維度的函數
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
