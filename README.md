# 🖥️ Advanced Page Replacement Algorithm Simulator

這是一個基於 Python 與 Streamlit 開發的互動式作業系統頁面置換演算法模擬器。
本專案旨在視覺化不同演算法在各種記憶體配置下的效能差異，深入分析 **Belady's Anomaly** 現象，並透過多種參照字串生成模式來驗證演算法對「局部性 (Locality)」的適應能力。

## 🔗 線上演示 (Live Demo)

👉 **點擊這裡開啟網頁**： https://ospagereplacementalgorithm-arwaxvdpgywhaxk5gb9pra.streamlit.app/

-----

## ✨ 主要功能 (Features)

  * **五大演算法實作與比較**：
      * **FIFO** (First-In, First-Out)
      * **LFU** (Least Frequently Used)
      * **MFU** (Most Frequently Used)
      * **LRU** (Least Recently Used)
      * **OPT** (Optimal - 理論最佳解)
  * **Belady's Anomaly 自動偵測**：
      * 自動捕捉 FIFO 發生異常的瞬間（Frame 增加但 Page Faults 反增）。
      * 提供詳細日誌與該次異常的視覺化快照。
  * **多樣化參照字串生成模式**：
      * **Uniform**: 完全隨機分佈。
      * **80/20 Rule**: 模擬高度局部性 (Locality)，展示 LRU/LFU 優勢。
      * **Gaussian**: 常態分佈。
      * **Cyclic**: 循環模式，專門展示 MFU 的優勢場景。
  * **深度效能分析**：
      * **區間勝率矩陣**：分析在記憶體資源 25%, 50%, 75%, 100% 時，誰是最佳演算法。
      * **Competitive Ratio**：計算各演算法與 OPT (最佳解) 的差距比例與評級。
  * **互動式歷程回放**：
      * 透過滑桿檢視每一次隨機模擬的詳細折線圖與數據。

-----

## 📊 支援的生成模式說明

本模擬器支援四種 Reference String 生成分佈，用於測試演算法在不同場景下的強健性：

| 模式名稱 | 描述與用途 |
| :--- | :--- |
| **Uniform** | **完全隨機**。所有頁面被存取的機率均等，測試基準線。 |
| **80/20 Rule** | **高度局部性**。20% 的熱門頁面佔據 80% 的存取量。此模式下 LRU 與 LFU 表現通常極佳。 |
| **Gaussian** | **常態分佈**。存取集中在中間區段的頁面 (呈現鐘形曲線)。 |
| **Cyclic (MFU Friendly)** | **階段性工作切換**。模擬工作集 (Working Set) 的平移，舊頁面頻率雖高但不再使用。此模式下 MFU 通常能正確置換，而 LFU 會失效。 |

-----

## 🛠️ 安裝與執行 (Installation)

若您希望在本地端電腦執行此專案，請按照以下步驟操作：

### 1\. 複製專案 (Clone Repository)

```bash
git clone https://github.com/Shen-M0/OS_Page_Replacement_Algorithm.git
cd OS_Page_Replacement_Algorithm
```

### 2\. 安裝依賴套件 (Install Dependencies)

確保您已安裝 Python 3.8 或以上版本。

```bash
pip install -r requirements.txt
```

*建議的 `requirements.txt` 內容：*

```text
streamlit
pandas
matplotlib
numpy
```

### 3\. 啟動 Streamlit (Run App)

```bash
streamlit run app.py
```

啟動後，瀏覽器將自動開啟 `http://localhost:8501`。

-----

## 📂 專案結構 (Project Structure)

```text
.
├── app.py                # 主要應用程式邏輯 (Streamlit, Algorithms, Plotting)
├── requirements.txt      # 專案依賴套件清單
└── README.md             # 專案說明文件
```

-----

## 📝 關於 Tie-Breaking 規則

在 LFU 與 MFU 演算法中，若遇到多個頁面頻率相同的情況，本模擬器採用 **FIFO (First-In, First-Out)** 規則進行決策。意即在頻率相同時，優先置換最早進入記憶體的頁面，以確保模擬結果的穩定性與可重現性。


