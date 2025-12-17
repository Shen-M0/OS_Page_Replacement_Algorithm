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
    'FIFO': {'color': 'blue',   'marker': 'o', 'style': '-'},
    'LFU':  {'color': 'green',  'marker': 's', 'style': '-'},
    'MFU':  {'color': 'red',    'marker': '^', 'style': '-'},
    'LRU':  {'color': 'orange', 'marker': 'D', 'style': '-'},
    'OPT':  {'color': 'purple', 'marker': '*', 'style': '--'} 
}

# ==========================================
# 1. 核心演算法
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

def run_lru(ref_string, frame_size):
    memory = []
    page_faults = 0
    for page in ref_string:
        if page in memory:
            memory.remove(page)
            memory.append(page)
        else:
            page_faults += 1
            if len(memory) < frame_size:
                memory.append(page)
            else:
                memory.pop(0) 
                memory.append(page)
    return page_faults

def run_opt(ref_string, frame_size):
    memory = []
    page_faults = 0
    for i, page in enumerate(ref_string):
        if page in memory:
            continue
        page_faults += 1
        if len(memory) < frame_size:
            memory.append(page)
        else:
            furthest_idx = -1
            victim = -1
            for mem_page in memory:
                try:
                    next_use = ref_string[i+1:].index(mem_page)
                except ValueError:
                    next_use = float('inf')
                if next_use > furthest_idx:
                    furthest_idx = next_use
                    victim = mem_page
            memory.remove(victim)
            memory.append(page)
    return page_faults

# ==========================================
# 2. 輔助函數
# ==========================================
def generate_reference_string(length, num_pages, method="Uniform"):
    ref_string = []
    
    if method == "Uniform":
        ref_string = [random.randint(0, num_pages - 1) for _ in range(length)]
        
    elif method == "80/20 Rule":
        cutoff = max(1, int(num_pages * 0.2))
        hot_pages = list(range(0, cutoff))
        cold_pages = list(range(cutoff, num_pages))
        for _ in range(length):
            if random.random() < 0.8:
                ref_string.append(random.choice(hot_pages))
            else:
                ref_string.append(random.choice(cold_pages))
                
    elif method == "Gaussian":
        mean = (num_pages - 1) / 2
        sigma = num_pages / 6 
        for _ in range(length):
            val = int(random.gauss(mean, sigma))
            val = max(0, min(num_pages - 1, val))
            ref_string.append(val)
    
    # [修正] 這裡原本是 "Cyclic (MFU Friendly)"，導致與 selectbox 不符
    elif method == "Cyclic":
        current_page = 0
        window_size = max(2, int(num_pages * 0.1)) 
        
        while len(ref_string) < length:
            subset = []
            for k in range(window_size):
                subset.append((current_page + k) % num_pages)
            
            repeats = random.randint(5, 8)
            for _ in range(repeats):
                for p in subset:
                    ref_string.append(p)
                    if len(ref_string) >= length: break
                if len(ref_string) >= length: break
            
            current_page = (current_page + window_size) % num_pages

    # [防呆] 如果上面都沒對應到，回傳隨機避免報錯
    if not ref_string:
        ref_string = [random.randint(0, num_pages - 1) for _ in range(length)]

    return ref_string

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

def create_plot(frame_axis, data_dict, title, anomaly_info=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ordered_algos = [k for k in data_dict.keys() if k != 'OPT']
    if 'OPT' in data_dict:
        ordered_algos.append('OPT')

    for algo_name in ordered_algos:
        y_values = data_dict[algo_name]
        style = STYLE_CONFIG.get(algo_name, {'color': 'black', 'marker': 'x', 'style': ':'})
        lw = 2 if algo_name == 'OPT' else 1.5
        ax.plot(frame_axis, y_values, label=algo_name, 
                 color=style['color'], marker=style['marker'], linestyle=style['style'], linewidth=lw)

    if anomaly_info:
        for algo, details in anomaly_info.items():
            if algo in data_dict and details:
                try:
                    first_detail = details[0]
                    frame_change = int(first_detail.split('->')[0].split()[-1])
                    faults_change = int(first_detail.split('Faults: ')[1].split('->')[1].replace(')', ''))
                    ax.annotate(f'{algo} Anomaly!', xy=(frame_change, faults_change), 
                                 xytext=(0, 25), textcoords='offset points', ha='center', 
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
    plt.tight_layout()
    return fig

# ==========================================
# 3. Streamlit 前端介面邏輯
# ==========================================
def main():
    st.title("Page and Frame Replacement Algorithms Simulator")
    st.markdown("比較 **FIFO, LFU, MFU, LRU, OPT** 演算法效能與 Belady 異常")

    st.sidebar.header("模擬參數設定")
    
    GEN_METHOD = st.sidebar.selectbox(
        "Reference String Distribution (參照字串生成模式)", 
        ("Uniform", "80/20 Rule", "Gaussian", "Cyclic"),
        help="""
        Uniform: 完全隨機分佈，所有頁面被選中的機率均等。\n
        80/20 Rule: 模擬高度局部性，20% 的頁面佔據 80% 的存取量。\n
        Gaussian: 常態分佈，存取集中在中間區段的頁面。\n
        Cyclic: 模擬階段性工作切換，舊頁面頻率高但不再使用。\n
        """
    )
    
    NUM_PAGES = st.sidebar.number_input("Page Range (頁面種類範圍)", 5, 100, 60)
    REF_LENGTH = st.sidebar.number_input("Ref String Length (參照字串長度)", 10, 5000, 1000)
    NUM_ITERATIONS = st.sidebar.slider("Iterations (測試次數)", 1, 200, 50)
    MAX_FRAMES = st.sidebar.slider("Max Frames (頁框數)", 3, 50, 30)
    
    run_btn = st.sidebar.button("開始模擬", type="primary")

    # 初始化 Session State
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'verification_result' not in st.session_state:
        st.session_state.verification_result = None

    # 如果按下開始按鈕
    if run_btn:
        st.session_state.verification_result = None # 清空舊的驗證圖
        
        st.info(f"目前的Reference String生成模式為：{GEN_METHOD}")
        
        ALGO_FUNCTIONS = {
            'FIFO': run_fifo, 
            'LFU':  run_lfu, 
            'MFU':  run_mfu,
            'LRU':  run_lru,
            'OPT':  run_opt
        }
        
        with st.spinner('計算中...'):
            percentages = [25, 50, 75, 100]
            frame_thresholds = {p: max(1, math.ceil(MAX_FRAMES * (p / 100))) for p in percentages}
            
            all_results = {name: defaultdict(list) for name in ALGO_FUNCTIONS}
            anomaly_report = {name: [] for name in ALGO_FUNCTIONS}
            all_runs_history = [] 

            stats = {
                'interval_wins': {p: {name: 0 for name in ALGO_FUNCTIONS} for p in percentages},
                'interval_fault_sums': {p: {name: 0 for name in ALGO_FUNCTIONS} for p in percentages}
            }
            
            frames_axis = list(range(1, MAX_FRAMES + 1))
            
            for i in range(1, NUM_ITERATIONS + 1):
                ref_str = generate_reference_string(REF_LENGTH, NUM_PAGES, GEN_METHOD)
                current_run_data = {}
                current_anomalies = {}
                
                for name, func in ALGO_FUNCTIONS.items():
                    is_anomaly, details, faults = check_belady_anomaly(func, ref_str, MAX_FRAMES)
                    current_run_data[name] = faults
                    for f_idx, val in enumerate(faults):
                        all_results[name][frames_axis[f_idx]].append(val)
                        
                    if is_anomaly:
                        anomaly_report[name].append({
                            'Run': i, 'Details': details,
                            'FullData': current_run_data, 'AllAnomalies': current_anomalies
                        })
                        current_anomalies[name] = details
                
                all_runs_history.append({
                    'id': i, 
                    'data': current_run_data, 
                    'anomalies': current_anomalies,
                    'ref_str': ref_str 
                })

                for p in percentages:
                    limit = frame_thresholds[p]
                    interval_sums = {name: sum(current_run_data[name][:limit]) for name in ALGO_FUNCTIONS}
                    
                    for name in ALGO_FUNCTIONS:
                        stats['interval_fault_sums'][p][name] += interval_sums[name]
                    
                    practical_sums = {k: v for k, v in interval_sums.items() if k != 'OPT'}
                    winner = min(practical_sums, key=practical_sums.get)
                    stats['interval_wins'][p][winner] += 1
            
            avg_data = {name: [np.mean(all_results[name][f]) for f in frames_axis] for name in ALGO_FUNCTIONS}

            # 儲存結果
            st.session_state.simulation_results = {
                'stats': stats,
                'avg_data': avg_data,
                'anomaly_report': anomaly_report,
                'all_runs_history': all_runs_history,
                'frames_axis': frames_axis,
                'frame_thresholds': frame_thresholds,
                'percentages': percentages,
                'ALGO_FUNCTIONS': ALGO_FUNCTIONS,
                'GEN_METHOD': GEN_METHOD,
                'NUM_ITERATIONS': NUM_ITERATIONS
            }

    # 顯示結果 (如果有數據)
    if st.session_state.simulation_results is not None:
        res = st.session_state.simulation_results
        
        # 檢查參數一致性 (防止切換參數後未按開始就報錯)
        if res['GEN_METHOD'] != GEN_METHOD:
            st.warning("偵測到參數變更！請點擊「開始模擬」以更新結果。目前顯示為舊參數的數據。")

        stats = res['stats']
        avg_data = res['avg_data']
        anomaly_report = res['anomaly_report']
        all_runs_history = res['all_runs_history']
        frames_axis = res['frames_axis']
        frame_thresholds = res['frame_thresholds']
        percentages = res['percentages']
        ALGO_FUNCTIONS = res['ALGO_FUNCTIONS']
        NUM_ITERATIONS = res['NUM_ITERATIONS']

        # --- 顯示結果 ---
        tab1, tab2, tab3, tab4 = st.tabs(["矩陣與分析", "平均趨勢", "異常日誌", "歷程回放與驗證"])
        
        with tab1:
            st.subheader("1. 區間勝率矩陣 (排除 OPT)")
            st.caption(f"定義：在實務演算法 (FIFO, LFU, MFU, LRU) 中，誰是表現最好的")
            
            win_data = []
            for name in ALGO_FUNCTIONS:
                if name == 'OPT': continue 
                
                row = {'Algorithm': name}
                for p in percentages:
                    rate = (stats['interval_wins'][p][name] / NUM_ITERATIONS) * 100
                    all_wins = [stats['interval_wins'][p][algo] for algo in ALGO_FUNCTIONS if algo != 'OPT']
                    label = f"{rate:.1f}%"
                    if stats['interval_wins'][p][name] == max(all_wins): label += " (Best)"
                    row[f"Top {p}% (F<={frame_thresholds[p]})"] = label
                win_data.append(row)
            st.dataframe(pd.DataFrame(win_data).set_index('Algorithm'), use_container_width=True)

            st.divider()

            st.subheader("2. 區間平均錯誤矩陣")
            st.caption("定義：平均發生多少次 Page Faults (越低越好)。(Best) 標記僅比較實務演算法。")
            
            avg_fault_data = []
            for name in ALGO_FUNCTIONS:
                row = {'Algorithm': name}
                for p in percentages:
                    frame_count = frame_thresholds[p]
                    val = stats['interval_fault_sums'][p][name] / (NUM_ITERATIONS * frame_count)
                    
                    practical_vals = [
                        stats['interval_fault_sums'][p][algo] / (NUM_ITERATIONS * frame_count) 
                        for algo in ALGO_FUNCTIONS if algo != 'OPT'
                    ]
                    min_practical_val = min(practical_vals)
                    
                    label = f"{val:.2f}"
                    if name != 'OPT' and val == min_practical_val:
                        label += " (Best)"
                    
                    row[f"Top {p}% (F<={frame_thresholds[p]})"] = label
                avg_fault_data.append(row)
            st.dataframe(pd.DataFrame(avg_fault_data).set_index('Algorithm'), use_container_width=True)

            st.divider()

            st.subheader("3. 與 OPT (最佳解) 的差距比較")
            st.caption("競爭比 (Ratio) = 該演算法錯誤數 / OPT錯誤數。")
            
            ratio_data = []
            for name in ALGO_FUNCTIONS:
                if name == 'OPT': continue
                
                row = {'Algorithm': name}
                total_algo_faults = stats['interval_fault_sums'][100][name]
                total_opt_faults = stats['interval_fault_sums'][100]['OPT']
                
                if total_opt_faults == 0: total_opt_faults = 1
                
                ratio = total_algo_faults / total_opt_faults
                diff_pct = (ratio - 1) * 100
                
                row['Competitive Ratio'] = f"{ratio:.3f}"
                row['Diff from OPT'] = f"+{diff_pct:.1f}%"
                
                
                ratio_data.append(row)
            st.dataframe(pd.DataFrame(ratio_data).set_index('Algorithm'), use_container_width=True)

        with tab2:
            st.subheader(f"平均效能曲線")
            st.caption(f"虛線(Purple) 為 OPT 理論最佳值，其他演算法應盡量貼近此線。")
            fig_avg = create_plot(frames_axis, avg_data, "Average Page Faults vs Frames")
            st.pyplot(fig_avg)

        with tab3:
            st.subheader("Belady's Anomaly 詳細報告")
            cols = st.columns(len(ALGO_FUNCTIONS))
            for idx, algo in enumerate(ALGO_FUNCTIONS):
                count = len(anomaly_report[algo])
                rate = (count / NUM_ITERATIONS) * 100
                with cols[idx]:
                    st.metric(label=algo, value=f"{count}次", delta=f"{rate:.1f}%")
            
            st.divider()
            for algo, logs in anomaly_report.items():
                if logs:
                    with st.expander(f"查看 {algo} 的異常紀錄 ({len(logs)} 筆)"):
                        for item in logs:
                            run_id = item['Run']
                            st.text(f"Run {run_id}: {item['Details'][0]}")
                            fig_anomaly = create_plot(frames_axis, item['FullData'], f"Run {run_id} Snapshot", item['AllAnomalies'])
                            st.pyplot(fig_anomaly)

        with tab4:
            st.subheader("模擬歷程回放與驗證")
            
            # 使用 key 來確保滑桿狀態正確
            selected_run_id = st.slider("選擇 Run ID", 1, NUM_ITERATIONS, 1, key="history_slider")
            
            # 確保索引不越界 (防止參數變更後舊的索引過大)
            if selected_run_id > len(all_runs_history):
                selected_run_id = 1
                
            run_record = all_runs_history[selected_run_id - 1]
            
            with st.expander(f"查看 Run {selected_run_id} 的參照字串 (Reference String)"):
                st.text_area("Reference String Content", str(run_record['ref_str']), height=100)
            
            run_opt_faults = sum(run_record['data']['OPT'])
            st.markdown("#### 該次模擬的 OPT 差距比較：")
            cols = st.columns(len(ALGO_FUNCTIONS)-1)
            idx = 0
            for algo in ALGO_FUNCTIONS:
                if algo == 'OPT': continue
                my_faults = sum(run_record['data'][algo])
                ratio = my_faults / run_opt_faults if run_opt_faults > 0 else 1
                cols[idx].metric(algo, f"{my_faults}", f"x{ratio:.2f} of OPT", delta_color="inverse")
                idx+=1

            st.write("##### 原始模擬結果：")
            fig_replay = create_plot(frames_axis, run_record['data'], f"Run {selected_run_id} Performance (Original)", run_record['anomalies'])
            st.pyplot(fig_replay)

            st.divider()
            
            st.subheader("重新模擬")
            
            # 驗證按鈕
            if st.button(f"重新模擬 Run {selected_run_id} "):
                with st.spinner("正在重新模擬..."):
                    verify_ref_str = run_record['ref_str']
                    verify_data = {}
                    verify_max_frames = frames_axis[-1]
                    
                    for name, func in ALGO_FUNCTIONS.items():
                        _, _, faults = check_belady_anomaly(func, verify_ref_str, verify_max_frames)
                        verify_data[name] = faults
                    
                    # [優化] 將驗證結果存入 Session State，防止圖片消失
                    st.session_state.verification_result = {
                        'id': selected_run_id,
                        'data': verify_data,
                        'frames_axis': frames_axis
                    }

            # [優化] 如果有驗證結果，且 ID 符合，就顯示出來
            if st.session_state.verification_result:
                v_res = st.session_state.verification_result
                if v_res['id'] == selected_run_id:
                    st.success("重新模擬完成！")
                    st.write("##### 重新模擬結果：")
                    fig_verify = create_plot(v_res['frames_axis'], v_res['data'], f"Run {selected_run_id} Verification (Re-run)")
                    st.pyplot(fig_verify)
                else:
                    # 如果滑桿ID變了，清空舊的驗證圖
                    st.session_state.verification_result = None

if __name__ == "__main__":
    main()
