import streamlit as st
import random
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import math

# ==========================================
# 0. 全域設定
# ==========================================
st.set_page_config(page_title="Page Replacement Sim", layout="wide")

STYLE_CONFIG = {
    'FIFO': {'color': 'blue',  'marker': 'o', 'style': '-'},
    'LFU':  {'color': 'green', 'marker': 's', 'style': '-'},
    'MFU':  {'color': 'red',   'marker': '^', 'style': '-'}
}

# ==========================================
# 1. 核心演算法 (保持不變)
# ==========================================
def run_fifo(ref_string, frame_size):
    memory = deque()
    page_faults = 0
    for page in ref_string:
        if page not in memory:
            page_faults += 1
            if len(memory) < frame_size:
                memory.append(page)
            else:
                memory.popleft() 
                memory.append(page)
    return page_faults

def run_lfu(ref_string, frame_size):
    memory = [] 
    frequency = defaultdict(int)
    page_faults = 0
    for page in ref_string:
        frequency[page] += 1
        if page not in memory:
            page_faults += 1
            if len(memory) < frame_size:
                memory.append(page)
            else:
                min_freq = float('inf')
                victim = -1
                for p in memory:
                    if frequency[p] < min_freq:
                        min_freq = frequency[p]
                        victim = p
                memory.remove(victim)
                memory.append(page)
    return page_faults

def run_mfu(ref_string, frame_size):
    memory = []
    frequency = defaultdict(int)
    page_faults = 0
    for page in ref_string:
        frequency[page] += 1
        if page not in memory:
            page_faults += 1
            if len(memory) < frame_size:
                memory.append(page)
            else:
                max_freq = -1
                victim = -1
                for p in memory:
                    if frequency[p] > max_freq:
                        max_freq = frequency[p]
                        victim = p
                memory.remove(victim)
                memory.append(page)
    return page_faults

# ==========================================
# 2. 輔助函數
# ==========================================
def generate_reference_string(length, num_pages):
    return [random.randint(0, num_pages - 1) for _ in range(length)]

def check_belady_anomaly(algo_func, ref_string, max_frames):
    prev_faults = float('inf')
    anomalies = []
    faults_data = []
    
    for f in range(1, max_frames + 1):
        faults = algo_func(ref_string, f)
        faults_data.append(faults)
        
        if f > 1 and faults > prev_faults:
            msg = f"At {f-1}->{f} Frames (Faults: {prev_faults}->{faults})"
            anomalies.append(msg)
        prev_faults = faults
        
    return len(anomalies) > 0, anomalies, faults_data

# 用於 Streamlit 的繪圖函數 (回傳 figure 物件)
def create_plot(frame_axis, data_dict, title, anomaly_info=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for algo_name, y_values in data_dict.items():
        style = STYLE_CONFIG[algo_name]
        ax.plot(frame_axis, y_values, label=algo_name, 
                 color=style['color'], marker=style['marker'], linestyle=style['style'])

    if anomaly_info:
        for algo, details in anomaly_info.items():
            if algo in data_dict and details:
                try:
                    first_detail = details[0]
                    # 解析字串 "At 3->4 Frames..."
                    frame_change = int(first_detail.split('->')[0].split()[-1])
                    faults_change = int(first_detail.split('Faults: ')[1].split('->')[1].replace(')', ''))
                    
                    ax.annotate(f'{algo} Anomaly!', xy=(frame_change, faults_change), 
                                 xytext=(0, 15), textcoords='offset points', ha='center', 
                                 color=STYLE_CONFIG[algo]['color'], 
                                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                                 fontsize=9, fontweight='bold')
                except: pass

    ax.set_title(title)
    ax.set_xlabel('Number of Frames')
    ax.set_ylabel('Page Faults')
    ax.set_xticks(frame_axis)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    return fig

# ==========================================
# 3. Streamlit 前端介面邏輯
# ==========================================
def main():
    st.title("🖥️ Page Replacement & Belady's Anomaly Simulator")
    st.markdown("比較 FIFO, LFU, MFU 演算法並驗證異常現象")

    # --- 側邊欄：參數設定 ---
    st.sidebar.header("⚙️ 模擬參數設定")
    
    # 修改 app.py 中的這幾行
    NUM_PAGES = st.sidebar.number_input("Page Types", min_value=5, max_value=100, value=60) # 改為 60
    REF_LENGTH = st.sidebar.number_input("Ref String Length", min_value=10, max_value=5000, value=1000) # 改為 1000
    NUM_ITERATIONS = st.sidebar.slider("Iterations", 1, 200, 100) # 改為 100
    MAX_FRAMES = st.sidebar.slider("Max Frames", 3, 50, 30) # 改為 30
    
    run_btn = st.sidebar.button("🚀 開始模擬", type="primary")

    # --- 主程式區塊 ---
    if run_btn:
        with st.spinner(f'正在執行 {NUM_ITERATIONS} 組模擬...'):
            
            # 初始化數據容器
            ALGO_FUNCTIONS = {'FIFO': run_fifo, 'LFU': run_lfu, 'MFU': run_mfu}
            percentages = [25, 50, 75, 100]
            frame_thresholds = {p: max(1, math.ceil(MAX_FRAMES * (p / 100))) for p in percentages}
            
            all_results = {name: defaultdict(list) for name in ALGO_FUNCTIONS}
            anomaly_report = {name: [] for name in ALGO_FUNCTIONS}
            
            stats = {
                'interval_wins': {p: {name: 0 for name in ALGO_FUNCTIONS} for p in percentages},
                'interval_fault_sums': {p: {name: 0 for name in ALGO_FUNCTIONS} for p in percentages}
            }
            
            frames_axis = list(range(1, MAX_FRAMES + 1))
            
            # 隨機挑選一組有異常的來展示 (為了 Demo 效果)
            example_anomaly_run = None
            
            # --- 模擬迴圈 ---
            progress_bar = st.progress(0)
            for i in range(1, NUM_ITERATIONS + 1):
                ref_str = generate_reference_string(REF_LENGTH, NUM_PAGES)
                current_run_data = {}
                current_anomalies = {}
                
                for name, func in ALGO_FUNCTIONS.items():
                    is_anomaly, details, faults = check_belady_anomaly(func, ref_str, MAX_FRAMES)
                    current_run_data[name] = faults
                    
                    # 數據累積
                    for f_idx, val in enumerate(faults):
                        all_results[name][frames_axis[f_idx]].append(val)
                        
                    if is_anomaly:
                        anomaly_report[name].append({'Run': i, 'Details': details})
                        current_anomalies[name] = details
                
                # 若這組有異常，且還沒存過範例，就存下來畫圖用
                if current_anomalies and example_anomaly_run is None:
                    example_anomaly_run = (i, current_run_data, current_anomalies)

                # 區間統計
                for p in percentages:
                    limit = frame_thresholds[p]
                    interval_sums = {name: sum(current_run_data[name][:limit]) for name in ALGO_FUNCTIONS}
                    winner = min(interval_sums, key=interval_sums.get)
                    stats['interval_wins'][p][winner] += 1
                    for name in ALGO_FUNCTIONS:
                        stats['interval_fault_sums'][p][name] += interval_sums[name]
                
                progress_bar.progress(i / NUM_ITERATIONS)
            
            # --- 模擬結束，整理數據 ---
            avg_data = {name: [np.mean(all_results[name][f]) for f in frames_axis] for name in ALGO_FUNCTIONS}

            # --- 顯示結果 (使用 Tabs 分頁) ---
            tab1, tab2, tab3 = st.tabs(["📊 綜合分析矩陣", "📈 趨勢與異常圖表", "📝 詳細異常報告"])
            
            # Tab 1: 矩陣表格
            with tab1:
                st.subheader("1. 區間勝率矩陣 (Interval Win Rates)")
                st.caption(f"定義：在特定 Frames 限制下 (Frame <= X)，該演算法錯誤最少的機率。 Frame Cuts: {frame_thresholds}")
                
                # 製作 DataFrame
                win_data = []
                for name in ALGO_FUNCTIONS:
                    row = {'Algorithm': name}
                    for p in percentages:
                        rate = (stats['interval_wins'][p][name] / NUM_ITERATIONS) * 100
                        # 標記 Best
                        all_wins = [stats['interval_wins'][p][algo] for algo in ALGO_FUNCTIONS]
                        label = f"{rate:.1f}%"
                        if stats['interval_wins'][p][name] == max(all_wins):
                            label += " (Best)"
                        row[f"Top {p}% (F<={frame_thresholds[p]})"] = label
                    win_data.append(row)
                st.dataframe(pd.DataFrame(win_data).set_index('Algorithm'), use_container_width=True)

                st.divider()

                st.subheader("2. 區間平均錯誤矩陣 (Avg Faults per Interval)")
                st.caption("定義：在該區間內，平均花費多少個 Page Faults 完成任務 (越低越好)。")
                
                avg_fault_data = []
                for name in ALGO_FUNCTIONS:
                    row = {'Algorithm': name}
                    for p in percentages:
                        frame_count = frame_thresholds[p]
                        val = stats['interval_fault_sums'][p][name] / (NUM_ITERATIONS * frame_count)
                        
                        # 標記 Best
                        all_vals = [stats['interval_fault_sums'][p][algo] / (NUM_ITERATIONS * frame_count) for algo in ALGO_FUNCTIONS]
                        label = f"{val:.2f}"
                        if val == min(all_vals):
                            label += " (Best)"
                        row[f"Top {p}% (F<={frame_thresholds[p]})"] = label
                    avg_fault_data.append(row)
                st.dataframe(pd.DataFrame(avg_fault_data).set_index('Algorithm'), use_container_width=True)

            # Tab 2: 圖表
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("平均效能曲線 (Average Curve)")
                    st.caption(f"基於 {NUM_ITERATIONS} 次測試的平均結果")
                    fig_avg = create_plot(frames_axis, avg_data, "Average Page Faults vs Frames")
                    st.pyplot(fig_avg)
                
                with col2:
                    st.subheader("單次模擬 (異常捕捉範例)")
                    if example_anomaly_run:
                        run_id, run_data, run_anomalies = example_anomaly_run
                        st.caption(f"Run {run_id}: 偵測到 Belady's Anomaly (詳見箭頭)")
                        fig_single = create_plot(frames_axis, run_data, f"Run {run_id} Performance", run_anomalies)
                        st.pyplot(fig_single)
                    else:
                        st.info("本次隨機模擬未捕捉到 Belady 異常範例，請嘗試增加 Iterations 或 Pages。")

            # Tab 3: 詳細報告
            with tab3:
                st.subheader("Belady's Anomaly 偵測統計")
                cols = st.columns(3)
                for idx, algo in enumerate(ALGO_FUNCTIONS):
                    count = len(anomaly_report[algo])
                    rate = (count / NUM_ITERATIONS) * 100
                    with cols[idx]:
                        st.metric(label=f"{algo} Anomaly Rate", value=f"{rate:.1f}%", delta=f"{count} 次")
                
                st.divider()
                st.markdown("#### 詳細異常日誌")
                for algo, logs in anomaly_report.items():
                    if logs:
                        with st.expander(f"查看 {algo} 的 {len(logs)} 筆異常紀錄"):
                            for item in logs:
                                st.text(f"Run {item['Run']}: {item['Details'][0]}")
                    else:
                        st.text(f"{algo}: 無異常偵測紀錄 (Stable)")

if __name__ == "__main__":
    main()