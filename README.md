# K60168A Dongle 與雷達手勢辨識專案使用指南

本目錄包含兩個互相配合的專案：

1. **`K60168A_Dongle/`**：官方提供的 KSOC 工具、韌體設定與 RDI（Range-Doppler Image）產生腳本，用於**資料擷取與雷達視覺化**。
2. **`radar-gesture-recognition-chore-update-20250815/`**：以 K60168A Dongle 為資料來源所建置的 **毫米波手勢辨識模型管線**，涵蓋資料前處理、3D CNN 訓練、評估與即時推論 GUI。

以下說明如何從硬體啟動、擷取資料、離線訓練，到線上推論。

---

## 1. 必備條件

- **硬體**：KaikuTek K60168A Dongle（含官方驅動與授權檔）。
- **作業系統**：Windows 10/11（KSOC 工具與 KKT 模組依賴 .NET 與 WinUSB）。
- **Python**：建議 3.8.9（`radar-gesture-recognition-chore-update-20250815/README.md` 同步建議）。
- **GPU（可選）**：NVIDIA CUDA 11.x 以上可加速 3D CNN 訓練與推論。

---

## 2. `K60168A_Dongle/` 工具組

### 2.1 目錄概觀

| 路徑 | 說明 |
| ---- | ---- |
| `KSOC_Tool/Collect_RawData/` | 以 `Collect_RawData.exe` 擷取 `.h5` 原始資料，`Config/` 內含 `HW_168A_setting.json`、`Mode_Configs.json` 等射頻設定，`Licence/` 放置授權檔。|
| `KSOC_Tool/Collect_RDI/` | 以 `Collect_RDI.exe` 直接輸出 RDI。結構與 RawData 版相同，另附 `KSOC_Libs/`（驅動/動態函式庫）。|
| `TempParam/` | 臨時參數（校正、SIC 設定等）會在工具執行期間放入此處。|
| `RDI_generation/` | Python 範例（`example_code_rdi*.py`）展示如何離線生成 RDI/PHD 圖。`Test_pattern/` 內含示範 `.h5`。|
| `Sample_Code/` | `KKT_Module_Example_20240820.7z`：更多 API 範例，可自行解壓。|

> 其他使用手冊（例如 `ksoc_tool-release-60000-001-v2.0.0 user guideline.pptx`）位於 `KSOC_Tool/Collect_RawData/Documents/`，請依公司授權閱讀。

### 2.2 安裝與授權

1. 以系統管理員身分執行 `Collect_RawData.exe`／`Collect_RDI.exe`，第一次啟動會提示匯入 `Licence/` 內容。
2. 確認 Dongle 連接並於「裝置管理員」顯示正確驅動。
3. 若需更新設定，修改 `Config/*.json` 或 `Reciver_Configs.ini`，再重新啟動工具。

### 2.3 擷取流程（Raw Data/RDI）

1. 依量測情境選擇 `HW_168A_setting.json` 或自訂檔，並在工具 UI 中匯入。
2. 於 `TempParam/` 中放置最新的校正檔（通常由原廠提供），以確保相位補償正確。
3. `Collect_RawData.exe`：開始錄製後輸出 `.h5`，建議依手勢分類建立子資料夾，後續可直接給前處理腳本。
4. `Collect_RDI.exe`：適合量測時立即檢查 Range-Doppler/Angle 圖，輸出路徑可於 UI 內設定。

### 2.4 使用 Python 生成 RDI/PHD

`RDI_generation/example_code_rdi.py` 與 `example_code_rdi_phd.py`（`K60168A_Dongle/RDI_generation/example_code_rdi.py:16-59`、`example_code_rdi_phd.py:16-76`）示範：

```bash
cd K60168A_Dongle/RDI_generation
python -m venv venv
venv\Scripts\activate
pip install numpy matplotlib h5py
python example_code_rdi.py          # 生成並即時顯示 RDI
python example_code_rdi_phd.py      # 額外生成 PHD (角度-多普勒)
```

執行前請將自有 `.h5` 放入 `Test_pattern/` 或更新腳本中的 `h5py.File(...)` 路徑。腳本會：

1. 讀取 `DS1` 併重排為 `[sample, chirp, antenna, frame]`。
2. 針對 RX1 進行相位補償（使用 `RF_CONFIG` 屬性）。
3. 進行 Range FFT / Doppler FFT（必要時再做 Angle FFT），並以 `matplotlib` 繪製熱圖。

---

## 3. `radar-gesture-recognition-chore-update-20250815/` 機器學習專案

### 3.1 目錄重點

- `Config/`, `TempParam/`, `Library/`, `KKT_Module/`：由開酷科技提供的 Python API，需透過 `setup.bat` 以 editable mode 安裝。
- `src/`：資料流程與模型主程式。
- `requirements.txt`：列出 PyTorch、PySide2、scikit-learn 等依賴。

### 3.2 Python 環境建置

```powershell
cd radar-gesture-recognition-chore-update-20250815
py -3.8 -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
.\setup.bat    # 安裝 KKT_Module、Library、UI（三者以 editable mode 匯入）
```

> 若需 GPU 版 PyTorch，請依自身 CUDA 版本另行安裝（`requirements.txt` 未固定 torch 版本）。

### 3.3 資料取得與結構

- 參考 Kaggle 資料集（`README.md` 同步提供連結）。
- 每個手勢一個子資料夾，`.h5` 需至少包含 `DS1`（形狀 `(2,32,32,T)`）與 `LABEL`（長度 `T` 的 0/1）。
- 建議結構：

```
data/
  train/
    background/*.h5
    patpat/*.h5
    wave/*.h5
    come/*.h5
  val/
    ...
```

### 3.4 前處理（`.h5` ➜ `.npz`）

- 主要邏輯在 `src/data_preprocessing.py`，包含以下可調整常數（`radar-gesture-recognition-chore-update-20250815/src/data_preprocessing.py:5-105`）：
  - `DATA_DIR`：指向原始 `.h5` 目錄。
  - `PROCESSED_DATA_DIR` 與 `PROCESSED_DATA_FILE`：輸出位置。
  - `gesture_types` 會依資料夾自動排序並建立 label 映射。
- 運行方式：

```powershell
python .\src\data_preprocessing.py
```

輸出 `.npz` 內含 `features`、`labels`、`ground_truths`，並對每筆序列填補至等長，且根據 label 產生高斯型 soft label。

### 3.5 資料檢視與統計

`src/read.py` 可互動檢視指定 index 的 `labels` 與 `ground_truths` 並輸出統計（內含 `get_max_label_interval` 與 `analyze_label_intervals`，方便檢查片段長度）。將 `npz_file_path` 換成實際路徑後執行：

```powershell
python .\src\read.py
```

### 3.6 模型訓練

`src/training.py` 定義完整 3D CNN 管線（`radar-gesture-recognition-chore-update-20250815/src/training.py:19-200`）：

- 需先設定 `TRAINING_DATA_DIR`、`VAL_DATA_DIR` 指向前處理輸出的 `.npz`。
- 重要超參數：`WINDOW_SIZE`、`STEP_SIZE`、`BATCH_SIZE`、`EPOCHS`、`LEARNING_RATE`、`NUM_CLASSES`。
- 程式會自動建立 `output/models/<timestamp>/` 並儲存最佳/最終權重。
- 執行：

```powershell
python .\src\training.py
```

完成後可依 console log 確認最佳模型檔名（`epoch_xx_valLoss_xxxx.pth`）與最終 `3d_cnn_model.pth`。

### 3.7 混淆矩陣與報告

`src/confusion.py` 使用與訓練同架構的模型進行 clip-level 評估（`radar-gesture-recognition-chore-update-20250815/src/confusion.py:12-178`）：

1. 更新 `TEST_DATA_FILE`、`MODEL_PATH`。
2. 視需要調整 `WINDOW_SIZE`、雙閾值 `HIGH_TH` / `LOW_TH`。
3. 執行後會輸出：
   - 4（真實類）× 6（預測類）混淆矩陣。
   - `classification_report`。
   - `matplotlib + seaborn` 視覺化。

### 3.8 即時推論與 GUI

`src/online_inference_gui.py`（`radar-gesture-recognition-chore-update-20250815/src/online_inference_gui.py:11-275`）整合 KKT 模組與自訂 GUI：

1. 先以 `setup.bat` 安裝 `KKT_Module`、`Library`、`UI`，並確保 Dongle 已連線。
2. 依現場設定更新以下參數：
   - `MODEL_PATH`：訓練輸出的 `.pth`。
   - `SETTING_FILE`：對應 KSOC 腳本的資料夾（需放在 `TempParam/` 或官方預設路徑）。
   - `WINDOW_SIZE`、`ENTER_TH`、`EXIT_TH`、`STREAM_TYPE`（`feature_map` 或 `raw_data`）。
3. `gesture_gui_pyside.py` 提供 `GestureGUI` 元件，可視需求調整配色/手勢名稱。
4. 啟動流程：

```powershell
python .\src\online_inference_gui.py
```

程式會：

- 建立 Qt 事件圈並顯示 GUI。
- 透過 `kgl.ksoclib` 連線、載入設定腳本、切換輸出源。
- 建立 3D CNN，持續接收 `MultiResult4168BReceiver` 的資料，利用雙閾值避免抖動，並透過 GUI 即時顯示各類別機率。

### 3.9 典型研發流程

1. **資料擷取**：用 `KSOC_Tool/Collect_RawData` 錄製手勢，並依手勢分類存放。
2. **RDI/品質檢查**：可用 `Collect_RDI` 或 `RDI_generation` Python 腳本快速檢查。
3. **前處理**：設定 `DATA_DIR` 後執行 `src/data_preprocessing.py` 產出 `.npz`。
4. **訓練/驗證**：跑 `src/training.py`，觀察 loss / accuracy。
5. **評估**：用 `src/confusion.py` 檢查 clip 級結果與多手勢/未完成情況。
6. **即時推論**：整合模型與 GUI，實際連線 Dongle 測試。

---

## 4. 疑難排解與建議

- **路徑與編碼**：Python 腳本預設 Windows 絕對路徑，若在其他系統上執行，請改為相對路徑或 `Pathlib` 以避免錯誤。
- **授權/驅動**：KSOC 工具無法啟動時，先確認 `Licence/` 是否存在與 Dongle SN 一致的授權檔。
- **版本差異**：`requirements.txt` 安裝的 `torch`, `PySide2` 等版本可依硬體調整；若 PySide2 無法與 Qt 版本匹配，可改用 `pip install PySide2==5.15.2`.
- **資料欄位**：若 `.h5` 檔案的 `DS1` 或 `LABEL` 命名不同，請於 `data_preprocessing.py` 的 `load_h5_file` 中調整鍵值。
- **模型輸出順序**：訓練、評估、線上推論須保持一致（`["Background","PatPat","Wave","Come"]`）。若新增類別，請同時調整 `gesture_types`、`CLASS_NAMES`、GUI 顯示與資料夾順序。

---

## 5. 後續工作建議

1. 將 `K60168A_Dongle/KSOC_Tool` 內的 PPT/文件轉為可全文檢索的 Markdown，方便版本控管。
2. 為 `src/data_preprocessing.py`、`src/training.py` 引入設定檔（如 `yaml`），減少修改原始碼的機會。
3. 建立自動化測試（例如以虛擬 `.h5` 產生器跑 `pytest`）以確保前處理與推論在升級依賴時仍可運作。

祝開發順利！

