# 🖥️ 作業系統頁面置換演算法模擬器 (OS Page Replacement Algorithm Simulator)

這是一個基於 **Python** 與 **Streamlit** 開發的互動式作業系統頁面置換演算法模擬器。
本專案不僅用於視覺化 FIFO, LRU, LFU, MFU 與 OPT 等演算法在不同記憶體配置下的效能差異，更具備 **驗證** 功能，允許使用者回放並驗證每一次的模擬結果，同時自動偵測並捕捉 **Belady's Anomaly** 現象。

## 🔗 線上演示 (Live Demo)

👉 **點擊這裡開啟網頁**： https://ospagereplacementalgorithm-arwaxvdpgywhaxk5gb9pra.streamlit.app/

-----

## ✨ 核心功能 (Key Features)

### 1. 演算法實作與比較
支援五種作業系統經典演算法，並以不同顏色與線條樣式區分：
* **FIFO (First-In, First-Out)**: 🔵 藍色 / 檢測 Belady 異常的基準。
* **LFU (Least Frequently Used)**: 🟢 綠色 / 適合長期穩定的頻率模式。
* **MFU (Most Frequently Used)**: 🔴 紅色 / 適合循環或階段性工作負載。
* **LRU (Least Recently Used)**: 🟠 橘色 / 實務上最接近最佳解的演算法。
* **OPT (Optimal)**: 🟣 紫色虛線 / 理論上的最佳解 (作為效能天花板)。

### 2. 多維度數據分析 (Tab 1 & 2)
* **區間勝率矩陣 (Interval Win Rates)**：分析在記憶體資源不同（如 25%, 50%, 75%, 100%）時，哪個演算法表現最佳。
* **區間平均錯誤矩陣**：計算各演算法在不同頁框數區間內的平均 Page Faults。
* **競爭比分析 (Competitive Ratio)**：計算各演算法與 OPT (最佳解) 的差距比例，量化演算法效率。
* **平均效能曲線**：繪製隨記憶體增加，缺頁率下降的趨勢圖。

### 3. Belady's Anomaly 自動偵測 (Tab 3)
* 系統會自動掃描所有模擬回合。
* 當偵測到 **FIFO** (或其他非堆疊演算法) 發生「頁框增加但缺頁數反而增加」的異常時，會自動記錄。
* 提供**視覺化快照**，標記異常發生的確切位置 (Frame 數與 Fault 變化)。

### 4. 歷程回放與科學驗證 (Tab 4) 🆕
* **歷程回放**：透過滑桿 (Slider) 調閱過去 50~200 次模擬中，任一回合的詳細 Reference String 與執行結果。
* **可重複驗證 (Reproducibility)**：提供 **「重新模擬」** 按鈕。系統會使用該回合相同的 Reference String 再次執行演算法，確保圖表與數據的一致性，證明實驗結果非隨機捏造。

-----

## 📊 參照字串生成模式 (Data Generators)

本模擬器支援四種分佈模式，用於測試演算法在不同場景下的適應性：

| 模式名稱 | 特性描述 | 適用場景 |
| :--- | :--- | :--- |
| **Uniform** | **完全隨機**。所有頁面被存取的機率均等。 | 測試演算法在無規律環境下的基準表現 (Baseline)。 |
| **80/20 Rule** | **高度局部性 (Locality)**。20% 的頁面佔據 80% 的存取量。 | 模擬真實世界的熱門資料存取，LRU 與 LFU 在此模式下表現極佳。 |
| **Gaussian** | **常態分佈**。存取集中在中間區段的頁面 (鐘形曲線)。 | 模擬存取特定記憶體區段的行為。 |
| **Cyclic** | **循環/階段性切換**。模擬工作集 (Working Set) 的平移與循環。 | 專門展示 **MFU** 的優勢場景 (舊頁面頻率高但不再使用)，此時 LFU 容易失效。 |

-----

## 🛠️ 安裝與執行 (Installation)

若您希望在本地端電腦執行此專案，請按照以下步驟操作：

### 1. 複製專案 (Clone Repository)

```bash
git clone https://github.com/Shen-M0/OS_Page_Replacement_Algorithm.git
cd OS_Page_Replacement_Algorithm

```

### 2. 安裝依賴套件 (Install Dependencies)

確保您已安裝 Python 3.8 或以上版本。

```bash
pip install -r requirements.txt

```

*requirements.txt 內容：*

```text
streamlit
pandas
matplotlib
numpy

```

### 3. 啟動 Streamlit (Run App)

```bash
streamlit run app.py

```

啟動後，瀏覽器將自動開啟 `http://localhost:8501`。

---

## 📝 補充說明

* **Tie-Breaking 規則**：在 LFU 與 MFU 演算法中，若遇到多個頁面頻率相同的情況，本模擬器採用 **FIFO (先進先出)** 規則進行決策。
* **OPT 實作**：採用「向前探查 (Look-ahead)」機制，掃描未來的 Reference String 以決定置換對象。

