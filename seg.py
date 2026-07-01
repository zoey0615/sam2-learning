# -*- coding: utf-8 -*-
import cv2              # 匯入 OpenCV模組；影像讀寫、顯示、輪廓、二值化、滑桿、滑鼠事件等
import torch            # 匯入 PyTorch模組；檢查 GPU、建立裝置 device
import sys              # 匯入系統模組:修改 Python 搜尋路徑 sys.path、結束程式
import numpy as np      # 匯入 NumPy模組；陣列、遮罩矩陣、座標處理
import os               # 匯入作業系統模組：檔案路徑、資料夾建立、檔案列舉

# --- 1. 路徑設定 ---
HOME = r"D:\segment-anything2\segment-anything-2"      #SAM2 專案根目錄
INPUT_DIR = r"D:\DIBAS\cocci_tile"                     #輸入影像資料夾

ROOT_OUTPUT = r"D:\DIBAS\segmentation_results_cocci"   #輸出根資料夾
IMG_OUTPUT_DIR = os.path.join(ROOT_OUTPUT, "images")   #輸出影像子資料夾路徑
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)             # 建立資料夾；若已存在不報錯

LBL_OUTPUT_DIR = os.path.join(ROOT_OUTPUT, "labels")   #輸出標註 txt 子資料夾路徑
os.makedirs(LBL_OUTPUT_DIR, exist_ok=True)             # 建立資料夾；若已存在不報錯

os.chdir(HOME)  # 將目前工作目錄切換到 SAM2 專案根目錄
sys.path.append(HOME) if HOME not in sys.path else None

from sam2.build_sam import build_sam2 #  匯入函式 build_sam2；建立sam2模型
from sam2.sam2_image_predictor import SAM2ImagePredictor  # 匯入類別 SAM2ImagePredictor；SAM2 推論管理器

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                       # torch.device 物件；若有 CUDA 就用 GPU，否則用 CPU
CHECKPOINT = os.path.join(HOME, "checkpoints", "sam2_hiera_large.pt")                       # SAM2 模型權重檔案路徑
CONFIG_PATH = os.path.join(HOME, "sam2", "configs", "sam2", "sam2_hiera_l.yaml")            # SAM2 模型設定檔路徑
sam2_model = build_sam2(CONFIG_PATH, CHECKPOINT, device=DEVICE, apply_postprocessing=False) # 初始化SAM2 模型，輸入：config 路徑、checkpoint 路徑、device、是否後處理
predictor = SAM2ImagePredictor(sam2_model) #初始化SAM2 推論管理器物件類別，可進行 set_image() / predict() 的預測器(紀錄目前影像特徵)

# --- 2. 全域變數 ---
all_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))] # 列出輸入影像資料夾中所有 jpg / png 圖檔名稱
input_points, input_labels, manual_colony_points = [], [], []
# input_points: list[list[int|float]]；SAM 點提示座標
# input_labels: list[int]；SAM 點提示標籤，1 通常代表正點
# manual_colony_points: list[tuple[int,int]]；手動切換類別用的點座標
manual_delete_points = []  # list[tuple[int,int]]；右鍵點擊後標記為刪除的點

current_img, current_gray, sam_mask, auto_mask = None, None, None, None
# current_img: np.ndarray 或 None；目前顯示給 SAM 的彩色影像
# current_gray: np.ndarray 或 None；目前的灰階影像
# sam_mask: np.ndarray 或 None；SAM 產生的遮罩
# auto_mask: np.ndarray 或 None；透過二值化/連通元件(傳統算法)自動產生的遮罩

first_click_contour, current_base_area = None, 0 #SAM 產出的黑白遮罩，經過 OpenCV 加工
# first_click_contour: cv2 contour 或 None；第一次點擊得到的輪廓，作為形狀比較基準
# current_base_area: float/int；第一次點擊輪廓面積，作為後續判斷門檻基準
should_save, should_reset, should_delete, should_quit = False, False, False, False # bool 旗標；控制主迴圈中的儲存、重設、刪除、離開

#================函式1.判斷輪廓類別(
# cnt:單一輪廓點集， shape 為 (N, 1, 2)
# base_area:第一次點到的輪廓面積
# first_contour: contour 或 None第一個輪廓，用於形狀比對   
# r_val:標準細菌面積*比例值  ColonyRatio      
# a_val單個細菌長寬比值  AspectLimit  
# s_val:兩顆細菌的不相似度 ShapeMatch                                                      
def get_class_id(cnt, base_area, first_contour, r_val, a_val, s_val):
    # --- 1. 絕對優先判定：檢查單一輪廓點集是否在手動「刪除」名單中 ---
    for pt in manual_delete_points: 
        if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0: #   >0 在輪廓內 0 在邊界上 <0 在輪廓外
            return None  # 返回 None 代表此輪廓被消掉

    # --- 2. 檢查單一輪廓點集是否在手動「切換類別」名單中 ---
    is_manual = False  # bool；標記此輪廓是否被手動切換
    for pt in manual_colony_points: #list[tuple[int,int]]；手動切換類別用的點座標
        if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0: #滑鼠點擊的座標（pt），是不是『點在這隻細菌（cnt）裡面或邊界上
            is_manual = True # 若點在輪廓內，表示此輪廓需手動反轉類別
            break
            
    # --- 3. 基礎自動判定邏輯 ---
    area = cv2.contourArea(cnt)  # float單一輪廓點集面積
    if area < 5: return None # 面積太小直接忽略，避免雜訊
    
    auto_cid = 0 # 預設類別為 0單株 (藍綠色)
    if first_contour is not None:  # 若存在第一次點擊的輪廓
        shape_diff = cv2.matchShapes(cnt, first_contour, 1, 0.0) #形狀不相似度得分=cv2.matchShapes(contour1, contour2, 胡氏矩絕對倒數差, 預留參數)
        x, y, bw, bh = cv2.boundingRect(cnt) #計算最正外接矩形:左上角座標與寬高
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6) #矩形長寬比
        ref_area = base_area if base_area > 0 else 350 #第一次點到的輪廓面積
        #條件a:形狀不相似度大於ShapeMatch滑桿設定值
        #條件b:當前單一輪廓點集面積>第一次點到的輪廓面積*ColonyRatio（菌落比例門檻）的滑桿
        #條件c:矩形長寬比>AspectLimit（長寬比極限）
        if shape_diff >= (s_val / 100.0) or area > ref_area * r_val or aspect_ratio > a_val:
            auto_cid = 1 #菌落
    # --- 4. 邏輯切換：手動左鍵修正 ---
    if is_manual: # 若點在輪廓內，表示此輪廓需手動反轉類別
        return 1 if auto_cid == 0 else 0
    
    return auto_cid  # 回傳自動判定單株或菌落標籤，型態 int

#================函式2.更新視窗畫面
def update_window():
    #全域變數:傳統演算法遮罩、SAM 產生的遮罩、顯示給 SAM 的彩色影像、
    #        灰階影像、第一次點擊輪廓面積、第一次點擊得到的輪廓
    global auto_mask, sam_mask, current_img, current_gray, current_base_area, first_click_contour
    if current_img is None: return #顯示給 SAM 的彩色影像。若尚未載入影像，直接結束
    h, w = current_img.shape[:2] #顯示給 SAM 的彩色影像
    # 讀取滑桿值cv2.getTrackbarPos(trackbar_name, window_name)
    t_val = cv2.getTrackbarPos('Thresh', "Bacteria Segmenter")                 # 二值化閾值
    b_val = cv2.getTrackbarPos('Blur', "Bacteria Segmenter") | 1               # 模糊核大小，|1 保證(二進位)為奇數
    m_val = cv2.getTrackbarPos('MinArea', "Bacteria Segmenter")                # 過濾最小面積
    r_val = cv2.getTrackbarPos('ColonyRatio', "Bacteria Segmenter") / 10.0     # 菌落比例門檻
    a_val = cv2.getTrackbarPos('AspectLimit', "Bacteria Segmenter") / 10.0     # 單個細菌長寬比值
    s_val = cv2.getTrackbarPos('ShapeMatch', "Bacteria Segmenter")             # 兩顆細菌的不相似度
    #cv2.GaussianBlur(src, ksize, sigmaX) np.ndarray；高斯模糊後影像，降低雜訊
    blurred = cv2.GaussianBlur(current_gray, (b_val, b_val), 0) #灰階影像(模糊核大小)
    _, binary = cv2.threshold(blurred, t_val, 255, cv2.THRESH_BINARY_INV) # binary: np.ndarray；二值化影像，輸出值為 0 或 255
    # num: int；連通元件數量（含背景）
    # labels: np.ndarray；每個像素所屬的連通元件標籤
    # stats: np.ndarray；每個元件的統計資料（[Left, Top(左上角xy座標), Width, Height, Area]等）
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary) #Connected Components With Stats
    auto_mask = np.zeros_like(binary)     # np.ndarray；與 binary 同 shape 的全 0 遮罩
    for i in range(1, num): #連通元件數量(不包含背景)
        if stats[i, cv2.CC_STAT_AREA] >= m_val: auto_mask[labels == i] = 255
        #面積>=過濾最小面積，全 0 影像其中只有屬於編號 i 的那些像素位置是(白色) True。 保留大於最小面積的元件
    combined = cv2.bitwise_or(auto_mask, sam_mask if sam_mask is not None else np.zeros_like(auto_mask))
    #combined = cv2.bitwise_or(傳統影像演算法遮罩, SAM 產生的遮罩):合併遮罩
    right_view = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR) #色彩空間轉換(Height, Width, 3)
    overlay = right_view.copy() #疊圖層，用來畫輪廓顏色
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #輪廓點集清單
    # Retrieval(檢索/提取) External(外部):只回傳階層結構中 最高層級 (Level 0) 的輪廓。
    # Chain Approximation(逼近) Simple:將輪廓鏈碼進行線性逼近的簡單壓縮模式
    for cnt in contours:  #輪廓點集清單
        # 函式1回傳自動判定單株或菌落標籤，型態 int(cnt,第一次點擊輪廓面積,第一次點擊得到的輪廓, ColonyRatio, AspectLimit ,ShapeMatch)
        cid = get_class_id(cnt, current_base_area, first_click_contour, r_val, a_val, s_val)
        if cid is None: continue  #下一輪迴圈
        color = (255, 255, 0) if cid == 0 else (255, 0, 255)
        #BGR (Blue-Green-Red)：OpenCV 的預設標準。
        # cid==0 ->青色系
        # cid==1 -> 紫紅色系
        cv2.drawContours(overlay, [cnt], -1, color, -1) #疊圖層，將該輪廓內部區域塗滿指定的顏色
    
    right_view = cv2.addWeighted(right_view, 0.4, overlay, 0.6, 0) #將原圖與 overlay 加權混合，讓輪廓顏色半透明
    #cv2.addWeighted(src1, alpha顯色度, src2, beta顯色度, gamma偏移量)
    
    # ======= 【點位視覺化：清晰標註點擊軌跡】 =======
    # 1. SAM 引導點：亮綠色實心圓點 (外加深綠邊框)
    for pt in input_points: #SAM 點提示座標
        px, py = int(pt[0]), int(pt[1])
        #cv2.circle(img, center, radius, color, thickness)
        cv2.circle(right_view, (px, py), 4, (0, 255, 0), -1) #鮮綠色(填充)
        cv2.circle(right_view, (px, py), 5, (0, 100, 0), 1) #深綠色(邊框)

    # 2. 類別切換點（左鍵點在物件上）：黃色實心圓點
    for pt in manual_colony_points: #手動切換類別用的點座標
        px, py = int(pt[0]), int(pt[1])
        cv2.circle(right_view, (px, py), 4, (0, 255, 255), -1) #鮮黃色(填充)
        cv2.circle(right_view, (px, py), 5, (0, 150, 150), 1) #較暗的黃色(邊框)

    # 3. 刪除屏蔽點（右鍵點擊）：鮮紅色叉叉 'X'
    for pt in manual_delete_points: #右鍵點擊後標記為刪除的點
        px, py = int(pt[0]), int(pt[1])
        #cv2.line(img, start_point, end_point, color, thickness)
        cv2.line(right_view, (px - 5, py - 5), (px + 5, py + 5), (0, 0, 255), 2) #斜率為正,紅色
        cv2.line(right_view, (px - 5, py + 5), (px + 5, py - 5), (0, 0, 255), 2) #斜率為負,紅色
    # =============================畫布排版與 GUI（圖形使用者介面）繪製
    panel_w = 200  # int；右側控制面板寬度
    canvas_w = (w * 2) + panel_w  # int；整個畫布總寬度 = 左圖 + 右圖 + 控制面板
    canvas = np.zeros((h, canvas_w, 3), dtype=np.uint8)  # np.ndarray；建立黑色畫布，shape = (高度, 寬度, 3通道)
    canvas[:h, :w] = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)  # 左半邊：顯示原始灰階圖轉 BGR 後的影像
    canvas[:h, w:w*2] = right_view  # 中間：顯示目前處理結果（疊圖、輪廓、點位）:將原圖與 overlay 加權混合，讓輪廓顏色半透明
    cv2.rectangle(canvas, (w*2, 0), (canvas_w, h), (35, 35, 35), -1) #深灰色
    #cv2.rectangle(img, 左上角頂點座標, 矩形的右下角頂點座標, color, 填充)
    px = w * 2 + 20  # int；控制區文字起始 x 座標
    #cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
    cv2.putText(canvas, "CONTROL", (px, 40), 0, 0.6, (200, 200, 200), 1)  # 在畫布上寫入控制面板標題(淺灰色)
    cv2.putText(canvas, f"Base: {current_base_area:.0f}", (px, 70), 0, 0.5, (200, 255, 200), 1) #第一次點擊輪廓面積
    cv2.putText(canvas, "[ SAVE (S) ]", (px, 150), 0, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, "[ RESET (R) ]", (px, 230), 0, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "[ DELETE (D) ]", (px, 310), 0, 0.7, (0, 0, 255), 2)
    cv2.putText(canvas, "[ EXIT (Q) ]", (px, 390), 0, 0.7, (150, 150, 150), 2)
    
    cv2.imshow("Bacteria Segmenter", canvas) #顯示整個互動視窗 cv2.imshow(winname, mat)

# event (整數)：滑鼠當下觸發的「動作類型」
# x:滑鼠指標在視窗畫布上的水平坐標。
# y:滑鼠指標在視窗畫布上的垂直坐標
# flags:事件發生時的「輔助狀態

# 函式3. 使用者在螢幕上的滑鼠互動
def mouse_callback(event, x, y, flags, param):
    #全域變數:SAM 點提示座標、SAM 點提示標籤、SAM 產生的遮罩、
    #目前顯示給 SAM 的彩色影像、第一次點擊輪廓面積、第一次點擊得到的輪廓、
    #手動切換類別用的點座標、右鍵點擊後標記為刪除的點、儲存
    #、重設、刪除、離開
    global input_points, input_labels, sam_mask, current_img, current_base_area, first_click_contour, manual_colony_points, manual_delete_points, should_save, should_reset, should_delete, should_quit
    if current_img is None: return  #目前顯示給 SAM 的彩色影像
    h, w = current_img.shape[:2] #影像高與寬
    
    # 右鍵：刪除 / 取消刪除輪廓
    #如果不把它們清空，下一次想點擊別的地方來分割新細菌時，模型會同時參考「舊的點」和「新的點」，常會導致分割結果出現無法預期的變形或錯誤。
    if event == cv2.EVENT_RBUTTONDOWN:
        img_x = x if x < w else x - w   # 將滑鼠點到的 x 座標轉換到影像區域座標，因為視窗左邊與中間各有影像區，因此若點到右邊顯示區，要扣掉 w
        if img_x < w: # 若點擊位置在影像範圍內 (SAM 模型的運作邏輯是累加式的（Prompt-based）：)
            if sam_mask is not None and sam_mask[y, img_x] > 0: #SAM 產生的遮罩分割區域
                input_points, input_labels = [], []  # 重置 SAM 點提示
                sam_mask = np.zeros((512, 512), dtype=np.uint8) #建立空遮罩
                if len(input_points) == 0:      # 若沒有任何點，清空基準面積與基準輪廓
                    current_base_area = 0
                    first_click_contour = None
                update_window() #函式2.更新視窗畫面
                return # 跳出整個 mouse_callback 函式
            #針對兩個矩陣中的每一個像素，執行「邏輯 OR」運算。
            combined = cv2.bitwise_or(auto_mask, sam_mask if sam_mask is not None else np.zeros_like(auto_mask))    # 合併傳統算法產生的遮罩與 SAM 遮罩
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 找出所有輪點集清單
            for cnt in contours: # 遍歷輪廓點集清單
                if cv2.pointPolygonTest(cnt, (float(img_x), float(y)), False) >= 0: # 判定「點」與「多邊形邊界關係:  >0 在輪廓內 0 在邊界上 <0 在輪廓外
                    exists = False #是否存在於刪除名單
                    for i, pt in enumerate(manual_delete_points):  #(回傳索引與內容)右鍵點擊後標記為刪除的點
                        if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0: # 判定「點」與「多邊形邊界關係:  >0 在輪廓內 0 在邊界上 <0 在輪廓外
                            manual_delete_points.pop(i) #從刪除名單中移出
                            exists = True #存在於刪除名單
                            break #強制讓目前的 for 迴圈立即停止執行，跳到迴圈之外的下一行程式碼。
                    if not exists:  # 若不在刪除名單，則加入刪除名單
                        manual_delete_points.append((img_x, y)) #turple
                    update_window() #函式2.更新視窗畫面
                    break
            return  # 跳出整個 mouse_callback 函式

    # 左鍵：切換類別 / 加入 SAM 點 / 面板按鈕
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > w * 2: #右側區域（控制面板區）。
            if 120 < y < 170: should_save = True    # 儲存  判斷使用者點在面板的哪個高度 (y 座標)。
            elif 200 < y < 250: should_reset = True  #、重設
            elif 280 < y < 330: should_delete = True #刪除
            elif 360 < y < 410: should_quit = True  #、離開
            return  # 跳出整個 mouse_callback 函式

        img_x = x if x < w else x - w  # 將滑鼠座標轉回影像座標(處理結果)
        combined = cv2.bitwise_or(auto_mask, sam_mask if sam_mask is not None else np.zeros_like(auto_mask))  # 合併傳統算法產生的遮罩與 SAM 遮罩
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 找出所有輪廓點集清單
        
        hit_bacteria = False # bool；是否點到細菌輪廓
        for cnt in contours: # 遍歷輪廓點集清單
            if cv2.pointPolygonTest(cnt, (float(img_x), float(y)), False) >= 0:  # 滑鼠點中了哪顆細菌
                is_deleted = False #預設輪廓未被刪除
                for pt in manual_delete_points: #左鍵點擊後標記為刪除的點
                    if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0: #這顆細菌是否在黑名單（刪除清單）內
                        is_deleted = True
                        break #跳出當前所在的那個 for  迴圈，不再執行該迴圈剩下的次數
                if is_deleted:
                    continue  #停止當前這一輪迴圈，直接跳到下一輪（下一個 cnt）繼續執行。
                
                hit_bacteria = True  # bool；是否點到細菌輪廓
                exists = False #使否存在切換標籤類別清單
                for i, pt in enumerate(manual_colony_points): #切換至菌落類別的點座標
                    if cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0: #這顆細菌是否在（切換清單）內
                        manual_colony_points.pop(i)
                        exists = True # 若已存在，則移除，等同取消菌落標記
                        break #跳出當前所在的那個 for  迴圈，不再執行該迴圈剩下的次數
                if not exists: # 若不存在，等同切換回菌落類別
                    manual_colony_points.append((img_x, y))
                update_window() #函式2.更新視窗畫面
                break  #跳出當前所在的那個 for  迴圈，不再執行該迴圈剩下的次數
  # 若沒有點到任何細菌輪廓，則把這次左鍵視為 SAM 點提示(其實那裡有細菌，但程式沒自動偵測到)
        if not hit_bacteria and img_x < w:
            # list[list[int,int]]；加入座標、 # list[int]；加入正點標籤 1
            input_points.append([img_x, y]); input_labels.append(1) #點提示的座標陣列
            
            # predictor.predict() 輸入：
            #   np.array(input_points): shape (N, 2) 點提示的座標陣列
            #   np.array(input_labels): shape (N,) 點提示的標籤陣列
            #   multimask_output=False：只輸出單一最可能遮罩
            masks, _, _ = predictor.predict(np.array(input_points), np.array(input_labels), multimask_output=False)
            #回傳:masks_np(分割遮罩二值化陣列), iou_predictions_np, low_res_masks_np
            #iou_predictions_np 遮罩（Mask）」與「真實目標（Ground Truth）」重疊程度的預測信心分數。
            #low_res_masks_np以 NumPy 陣列格式存在的低解析度遮罩(logits值)
            #檢查 masks[0] 這個陣列的維度數量是否為 3。取出該圖片中第一個候選遮罩的二維平面(顯示為 0 (背景) 和 1 (前景)
            m = masks[0][0] if len(masks[0].shape)==3 else masks[0]
            # 如果是第一次點擊
            if len(input_points) == 1: 
                cnts, _ = cv2.findContours((m>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:   #所有找到的輪廓(前景)
                    current_base_area = cv2.contourArea(cnts[0])  # float；記錄第一次輪廓面積
                    first_click_contour = cnts[0] # contour；記錄第一次輪廓(n,1,2)
            
            sam_mask = (m > 0).astype(np.uint8) * 255 # 將布林遮罩轉成 0/255 的 uint8 遮罩
            update_window() #函式2.更新視窗畫面

# --- 3. 視窗與滑桿初始化 ---建立一個可以讓使用者調整參數（滑桿）並與滑鼠互動（點擊）的視窗
cv2.namedWindow("Bacteria Segmenter", cv2.WINDOW_NORMAL) #允許使用者用滑鼠拖拉視窗邊框來放大縮小
# 建立可調整大小的視窗，名稱為 Bacteria Segmenter
#cv2.createTrackbar(滑桿名稱, 視窗名稱, 起始值, 最大值, 回呼函式)
cv2.createTrackbar('Thresh', "Bacteria Segmenter", 127, 255, lambda x: update_window()) #灰階門檻
cv2.createTrackbar('Blur', "Bacteria Segmenter", 3, 21, lambda x: update_window()) #高斯模糊程度
cv2.createTrackbar('MinArea', "Bacteria Segmenter", 100, 5000, lambda x: update_window()) #面積濾波門檻。
cv2.createTrackbar('ColonyRatio', "Bacteria Segmenter", 16, 50, lambda x: update_window()) #菌落判定門檻
cv2.createTrackbar('AspectLimit', "Bacteria Segmenter", 18, 50, lambda x: update_window()) #  單個細菌長寬比值
cv2.createTrackbar('ShapeMatch', "Bacteria Segmenter", 10, 100, lambda x: update_window()) # 與第一次點輪廓相比兩顆細菌的不相似度
# 事件(滑鼠)監聽:設定視窗的滑鼠事件回呼函式
cv2.setMouseCallback("Bacteria Segmenter", mouse_callback) # 函式3. 使用者在螢幕上的滑鼠互動

# --- 4. 主迴圈 ---
remaining_files = [f for f in all_files if not os.path.exists(os.path.join(IMG_OUTPUT_DIR, f))] #從『所有影像檔案清單 (all_files)』中，篩選出那些『在輸出資料夾 (IMG_OUTPUT_DIR) 裡還找不到』的檔案，並把這些檔案存成一個新的清單
# all_files:列出輸入影像資料夾中所有 jpg / png 圖檔名稱
# IMG_OUTPUT_DIR:輸出影像子資料夾路徑
for f_name in remaining_files: #遍歷在輸出資料夾 (IMG_OUTPUT_DIR) 裡還找不到』的檔案清單
   # 第一次輪廓面積 、第一次輪廓、SAM 點提示座標、SAM 點提示標籤、手動切換類別用的點座標
    current_base_area, first_click_contour, input_points, input_labels, manual_colony_points = 0, None, [], [], [] 
    manual_delete_points = []   # list[tuple[int,int]]；右鍵點擊後標記為刪除的點
    raw_img = cv2.imread(os.path.join(INPUT_DIR, f_name))#在輸出資料夾 (IMG_OUTPUT_DIR) 裡還找不到』的檔案清單
    if raw_img is None: continue #跳過當前這一輪的剩餘程式碼，直接進入迴圈的下一輪（處理下一個檔案）。
    raw_img = cv2.resize(raw_img, (512, 512)) #影像縮放為512*512
    
    # ======= 【LAB 空間 CLAHE 局部對比度增強】 =======
    lab = cv2.cvtColor(raw_img, cv2.COLOR_BGR2LAB) #將影像的色彩空間從 BGR 轉換為 LAB
    l_channel, a_channel, b_channel = cv2.split(lab) #lab 影像矩陣，根據通道（Channel）拆開成三個獨立的 2D 陣列。
    #Contrast Limited Adaptive Histogram Equalization，限制對比度自適應直方圖均衡化:局部光影不均
    # 建立 CLAHE 物件 (clipLimit=3.0 控制增強強度，grid 8x8 切分局部區域)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel) #將設定好的 CLAHE 濾鏡，實際應用到亮度通道 (L) 上，生成一張對比度增強後的影像
    
    # 重新結合回 BGR 給 SAM 2 使用，這樣模型更容易辨識邊緣
    current_img = cv2.cvtColor(cv2.merge((cl, a_channel, b_channel)), cv2.COLOR_LAB2BGR) #將處理完亮度細節的圖層，與原始色彩圖層重新合併，並轉回顯示用的 BGR 格式」
    # 直接把增強後的 L 通道做為灰階圖，二值化（Thresh）效果會極度清晰！
    current_gray = cl #對比度增強後的灰階影像
    # ========================================================   
    current_img_rgb = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
    #執行影像編碼（Image Encoder），將圖片轉換特徵張量（Embeddings）包含了圖片中所有物件的形狀、輪廓、紋理資訊。
    predictor.set_image(current_img_rgb ) #將處理完亮度細節的圖層，與原始色彩圖層重新合併，並轉回顯示用的 rgb 格式
    sam_mask = np.zeros((512, 512), dtype=np.uint8) # np.ndarray；初始 SAM 遮罩為全 0
    update_window() #函式2.更新視窗畫面
    
    while True: #事件迴圈
    # cv2.waitKey(1)回傳該按鍵的 ASCII 碼（一個整數）。如果什麼都沒按，它會回傳 -1。
    # 無論前面 24 個位元是什麼，都只取最後 8 個位元
        key = cv2.waitKey(1) & 0xFF #處理鍵盤互動
        # Ordinal（序數）:將一個「字元 (Character)」轉換為它對應的「Unicode/ASCII 編碼（整數）」。
        if key == ord('s') or should_save:  #鍵盤觸發或滑鼠觸發儲存
            r = cv2.getTrackbarPos('ColonyRatio', "Bacteria Segmenter") / 10.0
            a = cv2.getTrackbarPos('AspectLimit', "Bacteria Segmenter") / 10.0
            s = cv2.getTrackbarPos('ShapeMatch', "Bacteria Segmenter")
            # 重新取得當前滑桿值
            #針對兩個矩陣中的每一個像素，執行「邏輯 OR」運算。
            combined_mask = cv2.bitwise_or(auto_mask, sam_mask) #傳統演算法遮罩、SAM 產生的遮罩
            cnts, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 找出所有輪廓點集清單
            clean_save_mask = np.zeros_like(combined_mask) #np.ndarray；準備輸出的乾淨遮罩
            #輸出標註 txt 子資料夾路徑
            #os.path.splitext 會將檔名拆分成「檔名主體」與「副檔名」兩個部分
            # 開啟標註檔（覆蓋寫入模式）
            with open(os.path.join(LBL_OUTPUT_DIR, os.path.splitext(f_name)[0]+".txt"), 'w') as f: 
                for c in cnts: # 遍歷輪廓點集清單
                    # 函式1回傳自動判定單株或菌落標籤，型態 int(cnt,第一次點擊輪廓面積,第一次點擊得到的輪廓, ColonyRatio, AspectLimit ,ShapeMatch)
                    cid = get_class_id(c, current_base_area, first_click_contour, r, a, s)
                    if cid is not None:  
                        #cv2.drawContours(image, contours, contourIdx, color, thickness)
                        #在畫布 (clean_save_mask) 上，找到輪廓 (c)， 將所有點(-1) 用白色 (255) 將它們填滿 (-1)。
                        cv2.drawContours(clean_save_mask, [c], -1, 255, -1)
                        bx, by, bw, bh = cv2.boundingRect(c) # 輪廓外接矩形座標與寬高(左上角 xy、寬度、高度)
                        f.write(f"{cid} {(bx+bw/2.0)/512:.6f} {(by+bh/2.0)/512:.6f} {bw/512.0:.6f} {bh/512.0:.6f}\n")
                        # YOLO 格式輸出：
                        # class_id x_center y_center width height
                        # 全部皆為 0~1 正規化座標
            cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, f_name), clean_save_mask)
            # 儲存二值遮罩影像到 images 資料夾
            should_save = False #儲存動作已經完成，請回到等待狀態，不要再重複執行儲存了。
            print(f"Saved: {f_name}"); break  # 印出儲存完成訊息，跳出目前檔案的 while 迴圈，進入下一張
        # 若按下 r 或滑鼠點擊觸發重設
        if key == ord('r') or should_reset:
            input_points, input_labels, sam_mask, manual_colony_points, current_base_area = [], [], np.zeros((512,512), dtype=np.uint8), [], 0
            #重置SAM 點提示座標、SAM 點提示標籤、SAM 產生的遮罩、手動切換類別用的點座標、第一次輪廓面積
            manual_delete_points = [] # list[tuple[int,int]]；右鍵點擊後標記為刪除的點
            should_reset = False; update_window() #函式2.更新視窗畫面
        # 若按下 d 或滑鼠點擊觸發刪除
        if key == ord('d') or should_delete:
            img_p = os.path.join(IMG_OUTPUT_DIR, f_name)   # 輸出影像路徑
            lbl_p = os.path.join(LBL_OUTPUT_DIR, os.path.splitext(f_name)[0]+".txt")   # 輸出標註檔路徑
            if os.path.exists(img_p): os.remove(img_p)  # 若存在則刪除影像檔
            if os.path.exists(lbl_p): os.remove(lbl_p)   # 若存在則刪除標註檔
            should_delete = False; print(f"Deleted: {f_name}"); break
            # 刪除旗標歸零，印出刪除完成訊息，跳出 while 迴圈
        if key == ord('q') or should_quit:
            cv2.destroyAllWindows() # 關閉所有 OpenCV 視窗
            sys.exit() # 強制結束整個程式的執行。

cv2.destroyAllWindows() # 主迴圈結束後，再保險關閉所有視窗