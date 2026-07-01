# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#先將影像轉換成特徵向量(Image Embedding)，之後利用點(Point)、框(Box)、Mask等提示(Prompt)快速產生分割結果(Segmentation Mask)。
import logging #日誌
from typing import List, Optional, Tuple, Union #型別提示:列表、可以是某型別也可以是 None、.固定長度資料組、多種型別擇一
import numpy as np
import torch #建立張量、將資料送到 GPU/CPU、做模型推論、關閉梯度計算
from PIL.Image import Image
from sam2.modeling.sam2_base import SAM2Base #影像特徵提取 + prompt 編碼 + mask 解碼
from sam2.utils.transforms import SAM2Transforms
# SAM2資料轉換模組
# 1. 將輸入影像轉換為模型可接受的Tensor格式。
# 2. 執行影像尺寸調整（Resize）與標準化（Normalization）。
# 3. 將點座標(Point Prompt)與邊界框(Box Prompt)轉換至模型座標系統。
# 4. 對模型輸出的遮罩(Mask)進行後處理，包括補洞(Hole Filling)、
#    去除小型連通區(Small Component Removal)及還原至原始影像解析度。

class SAM2ImagePredictor:
    def __init__( #建構子:建立物件時，自動執行的初始化程序
        self,
        sam_model: SAM2Base,  #型別提示
        mask_threshold=0.0, #把模型輸出的 mask logits 轉成 binary mask 時，要不要算成前景(mask) 或背景
        max_hole_area=0.0, #預設不處理小洞
        max_sprinkle_area=0.0, #預設不處理小雜點
        **kwargs, #其他你沒有明講的額外參數，全部先收起來
    ) -> None: #沒有回傳值
        """
        Uses SAM-2 to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam-2): The model to use for mask prediction.
          mask_threshold (float): The threshold to use when converting mask logits
            to binary masks. Masks are thresholded at 0 by default.
          max_hole_area (int): If max_hole_area > 0, we fill small holes in up to
            the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (int): If max_sprinkle_area > 0, we remove small sprinkles up to
            the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__() #繼承機制，呼叫父類別的初始化函式。
        self.model = sam_model #SAM2 模型
        self._transforms = SAM2Transforms( #建立影像轉換器
            resolution=self.model.image_size, #設定輸入圖片要被 resize 成多大。
            mask_threshold=mask_threshold, #設定 mask 二值化門檻。
            max_hole_area=max_hole_area, #設定補洞的上限大小。
            max_sprinkle_area=max_sprinkle_area, #設定去除小雜點的上限大小。
        )

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None #儲存一張或多張圖片的「原始高度與寬度」。
        # Whether the predictor is set for single image or a batch of images
        self._is_batch = False

        # Predictor config
        self.mask_threshold = mask_threshold

        # Spatial dim for backbone feature maps
        self._bb_feat_sizes = [ #模型 Backbone 輸出的各層特徵圖尺寸列表
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "SAM2ImagePredictor":
        """
        Load a pretrained model from the Hugging Face hub.

        Arguments:
          model_id (str): The Hugging Face repository ID.
          **kwargs: Additional arguments to pass to the model constructor.

        Returns:
          (SAM2ImagePredictor): The loaded model.
        """
        from sam2.build_sam import build_sam2_hf

        sam_model = build_sam2_hf(model_id, **kwargs)
        return cls(sam_model, **kwargs)

    @torch.no_grad() #不須計算梯度
    def set_image( #執行影像編碼（Image Encoder），將圖片轉換特徵張量（Embeddings）包含了圖片中所有物件的形狀、輪廓、紋理資訊。
        self, #為了存取或修改類別內部設定的變數
        image: Union[np.ndarray, Image], #image 這個變數，既可以是 np.ndarray (NumPy 陣列，通常來自 OpenCV)，也可以是 Image (來自 PIL/Pillow 的影像物件)。
    ) -> None: # 宣告這個函式不會回傳任何值 (None)，它的作用是「改變物件內部的狀態」。
        """
        計算給定圖像的圖像嵌入，允許使用 `predict` 方法預測掩碼。
        參數：
        image（np.ndarray 或 PIL Image）：要嵌入的輸入影像，RGB 格式。如果是 np.ndarray，影像應為 HWC 格式；
        如果是 PIL Image，影像應為 WHC 格式。像素值範圍為 [0, 255]。
        image_format（str）：影像的顏色格式，取值範圍為 ['RGB', 'BGR']。
        """
        self.reset_predictor() # 將「推論器 (Predictor)」回到最原始、乾淨的狀態，清除之前所有殘留的記憶。
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray): #判斷一個物件（image）是否屬於型別（np.ndarray ）。
            logging.info("For numpy array image, we assume (HxWxC) format") #對於 numpy 陣列影像，我們假設其格式為 (HxWxC)。
            self._orig_hw = [image.shape[:2]] #儲存多張圖片的尺寸（例如在批次處理時），所以統一用 [H, W] 的形式儲存
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)] #將其重組為一個列表，內容為一個元組 (h, w)。
        else:
            raise NotImplementedError("Image format not supported") #拋出錯誤

# image（np.ndarray 或 PIL Image）：要嵌入的輸入影像，RGB 格式。
        input_image = self._transforms(image) #將任意尺寸、格式的原始影像，轉換成模型能接受的格式
        #ToTensor:將像素值（0-255 的整數）轉為 float32 浮點數，並縮放到 [0.0, 1.0] 之間。同時將通道順序調整為 PyTorch 標準的 (Channels, Height, Width)。
        #Resize：強行將影像調整為 (1024, 1024)
        #Normalize：將像素值根據模型訓練時使用的平均值（0.485, 0.456, 0.406）與標準差（0.229, 0.224, 0.225）進行標準化。
        input_image = input_image[None, ...].to(self.device) #在最前面新增一個維度batch
#assert (條件), "錯誤訊息"
        assert ( #回傳影像張量的維度必須是 4 維（符合 Batch, Channel, Height, Width 的規範）。
                #張量的第 2 個維度:輸入通常是 RGB 三原色，所以必須是 3。
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...") #計算所提供影像的影像嵌入向量
        backbone_out = self.model.forward_image(input_image)  #字典物件（Dictionary），儲存了編碼器處理後的輸出(FPN)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out) #vision_feats:將 CNN 骨幹網路輸出之多維空間特徵圖，轉換為 Transformer 運算所需的『序列化特徵向量，以利於進行全局注意力機制的計算。
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed: #FALSE
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [  #將 Backbone 的輸出依照解析度分成了三層，並存入 list
                 #PyTorch 的 view() 函數中，-1 代表「自動計算該維度的數值」。
                 #view() 改變張量維度（形狀）:(1,C,H,W) 
                  #將 (H, W) 兩個數字拆開（解包），填入 view 的最後兩個維度，定義影像的高度與寬度
            feat.permute(1, 2, 0).view(1, -1, *feat_size) #重新排列維度順序(H*W,B,C)->(B,C,H*W)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1] #將整個結果列表再次反轉，變回原本的順序。
        #打包(序列化特徵向量 &模型 Backbone 輸出的各層特徵圖尺寸列表)
        #[::-1] 讓兩者都變成從「最深層（解析度最低）」開始，一路往「最淺層（解析度最高）」排列。
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        #列表的最後一個元素（即最深層的特徵，解析度最低但語義最強）。
        #從第一個開始，直到倒數第二個為止。(高解析度的淺層特徵)
        self._is_image_set = True #影像特徵已經計算完畢
        logging.info("Image embeddings computed.")

    @torch.no_grad()
    def set_image_batch(
        self,
        image_list: List[Union[np.ndarray]],
    ) -> None:
        """
        Calculates the image embeddings for the provided image batch, allowing
        masks to be predicted with the 'predict_batch' method.

        Arguments:
          image_list (List[np.ndarray]): The input images to embed in RGB format. The image should be in HWC format if np.ndarray
          with pixel values in [0, 255].
        """
        self.reset_predictor()
        assert isinstance(image_list, list)
        self._orig_hw = []
        for image in image_list:
            assert isinstance(
                image, np.ndarray
            ), "Images are expected to be an np.ndarray in RGB format, and of shape  HWC"
            self._orig_hw.append(image.shape[:2])
        # Transform the image to the form expected by the model
        img_batch = self._transforms.forward_batch(image_list)
        img_batch = img_batch.to(self.device)
        batch_size = img_batch.shape[0]
        assert (
            len(img_batch.shape) == 4 and img_batch.shape[1] == 3
        ), f"img_batch must be of size Bx3xHxW, got {img_batch.shape}"
        logging.info("Computing image embeddings for the provided images...")
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True
        self._is_batch = True
        logging.info("Image embeddings computed.")

    def predict_batch(
        self,
        point_coords_batch: List[np.ndarray] = None,
        point_labels_batch: List[np.ndarray] = None,
        box_batch: List[np.ndarray] = None,
        mask_input_batch: List[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords=True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """This function is very similar to predict(...), however it is used for batched mode, when the model is expected to generate predictions on multiple images.
        It returns a tuple of lists of masks, ious, and low_res_masks_logits.
        """
        assert self._is_batch, "This function should only be used when in batched mode"
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image_batch(...) before mask prediction."
            )
        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        for img_idx in range(num_images):
            # Transform input prompts
            point_coords = (
                point_coords_batch[img_idx] if point_coords_batch is not None else None
            )
            point_labels = (
                point_labels_batch[img_idx] if point_labels_batch is not None else None
            )
            box = box_batch[img_idx] if box_batch is not None else None
            mask_input = (
                mask_input_batch[img_idx] if mask_input_batch is not None else None
            )
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts( 
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )
            masks, iou_predictions, low_res_masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )
            masks_np = masks.squeeze(0).float().detach().cpu().numpy()
            iou_predictions_np = (
                iou_predictions.squeeze(0).float().detach().cpu().numpy()
            )
            low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
            all_masks.append(masks_np)
            all_ious.append(iou_predictions_np)
            all_low_res_masks.append(low_res_masks_np)

        return all_masks, all_ious, all_low_res_masks


    def predict( # 型別註解（Type Annotation） 與 預設值（Default Value）
        self,
        point_coords: Optional[np.ndarray] = None,#點提示的座標陣列
        point_labels: Optional[np.ndarray] = None, #點提示的標籤陣列
        box: Optional[np.ndarray] = None,  #邊界框提示
        mask_input: Optional[np.ndarray] = None, #前一輪輸出的低解析度遮罩，用於迭代式預測：將上次結果作為輸入，進一步優化遮罩品質。
        multimask_output: bool = True, #是否輸出多個候選遮罩
        return_logits: bool = False, #是否回傳未經閾值處理的 logits
        normalize_coords=True, #是否將 point_coords 與 box 的座標正規化到 [0,1] 範圍。
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: #型別註解：表示回傳值是1個元組(包含3個多維陣列)
        """
        Predict masks for the given input prompts, using the currently set image.

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.
          normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] and point_coords is expected to be wrt. image dimensions.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set: #記錄目前是否已經成功設定好一張圖片並計算出影像嵌入特徵（image embedding）。
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )
        # Transform input prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(#轉換後的遮罩張量、轉換後的點座標張量、轉換後的點標籤張量、轉換後的邊界框張量
            point_coords, point_labels, box, mask_input, normalize_coords
        )

        masks, iou_predictions, low_res_masks = self._predict( #SAM 2 模型中專門負責處理「提示（Prompt）」的模組。
            unnorm_coords,
            labels,
            unnorm_box,
            mask_input,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def _prep_prompts( #點提示的座標陣列、點提示的標籤陣列、邊界框提示、前一輪輸出的低解析度遮罩、是否將 point_coords 與 box 的座標正規化到 [0,1] 範圍。
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1 #列表的最後一個元素
    ):
#轉換後的點座標張量。轉換後的點標籤張量。轉換後的邊界框張量。轉換後的遮罩張量。NumPy 陣列轉換為 PyTorch 張量
        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None: #點提示的座標陣列
            assert ( #assert 條件, "错误信息"
                point_labels is not None  #點提示的標籤陣列
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(  # 要轉換的資料:點提示的座標陣列換為新的 PyTorch 張量
                point_coords, dtype=torch.float, device=self.device 
            )
            unnorm_coords = self._transforms.transform_coords( #點提示座標
                point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx] #取列表的最後一個 (height, width)元素。
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device) #將點提示的標籤陣列轉換為新的張量
            if len(unnorm_coords.shape) == 2: #元組，表示張量各維度的大小。
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...] #增加一個大小為 1 的新維度。
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
            )  # Bx2x2
        if mask_logits is not None: #迭代預測
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box #轉換後的遮罩張量、轉換後的點座標張量、轉換後的點標籤張量、轉換後的邊界框張量

    @torch.no_grad()
    def _predict( #將已經轉換好的提示（點、框、遮罩）與圖片特徵結合，產出最終的分割遮罩。
        self,
        point_coords: Optional[torch.Tensor],#已轉換的點座標張量，形狀 (B, N, 2)。
        point_labels: Optional[torch.Tensor], #已轉換的點標籤張量，形狀 (B, N)。
        boxes: Optional[torch.Tensor] = None, #已轉換的邊界框張量，形狀 (B, 2, 2)。
        mask_input: Optional[torch.Tensor] = None, #前一輪的低解析度遮罩，形狀 (B, 1, 256, 256)。
        multimask_output: bool = True, #是否輸出多個遮罩（3個）。
        return_logits: bool = False, #是否回傳原始 logits（預設 False → 回傳二值遮罩）
        img_idx: int = -1, #在批次特徵中選擇第幾張圖片（預設 -1 代表最後一張）。
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:#回傳一個元組，內含三個 PyTorch 張量（遮罩、IoU分數、低解析遮罩）。
        if not self._is_image_set: #記錄目前是否已經成功設定好一張圖片並計算出影像嵌入特徵（image embedding）。
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction."
            )

        if point_coords is not None:
            concat_points = (point_coords, point_labels) #將點座標張量與點標籤張量結合成一個元組（tuple
        else:
            concat_points = None

        # 提示融合邏輯 (Embed prompts)
        if boxes is not None:  # 如果使用者有給邊界框，SAM2 習慣把邊界框視為特殊的「兩個點」來與一般的點提示做拼接
            box_coords = boxes.reshape(-1, 2, 2) # 改變張量形狀，確保其格式為 (Batch, 2點, X及Y)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device) # 建立邊界框專屬的虛擬標籤：2 代表box左上角點，3 代表box右下角點，型態為整數
            box_labels = box_labels.repeat(boxes.size(0), 1) # 語法：`repeat` 沿著 Batch 軸複製，確保與輸入的框數量對齊
            if concat_points is not None:  #如果使用者同時給了「框」和「點」(將點座標張量與點標籤張量結合成一個元組（tuple
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1) # 語法：`torch.cat([...], dim=1)` 在維度 1（點的數量軸）上進行拼接，把框的兩個點塞在一般點的前面
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels) # 更新拼接後的點提示元組
            else:
                concat_points = (box_coords, box_labels)# 若沒有其他點，則拼接提示只包含邊界框轉換來的點
# 呼叫提示編碼器：將點、框、以及先前的遮罩轉換成稀疏（Sparse）與稠密（Dense）兩種提示嵌入向量
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self._is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert (
            self._features is not None
        ), "Features must exist if an image has been set."
        return self._features["image_embed"]

    @property
    def device(self) -> torch.device:
        return self.model.device
#將「推論器 (Predictor)」回到最原始、乾淨的狀態，清除之前所有殘留的記憶。
    def reset_predictor(self) -> None:
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
