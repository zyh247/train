import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import pandas as pd
import ast
import re
import os
import torch
import json
import numpy as np
from datetime import datetime
import glob
import sys
import tempfile
import time
import visualize as viz
import shutil
import traceback
import uuid
import subprocess

# 根据 visualize_training.py 的位置来寻找 pt.darts 目录并添加到系统路径
# 如果 pt.darts 位于 visualize_training.py 的父目录，这行是正确的
# 如果 pt.darts 位于 visualize_training.py 的同级目录或其他位置，需要相应修改
pt_darts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pt.darts'))
if os.path.exists(pt_darts_path):
    sys.path.append(pt_darts_path)
    print(f"Added {pt_darts_path} to sys.path") # 添加打印用于调试
else:
    print(f"Warning: {pt_darts_path} not found. pt.darts module might not be available.") # 添加打印用于调试


# 尝试导入graphviz，如果不可用则提供替代方案
try:
    from graphviz import Digraph
    HAS_GRAPHVIZ = True
    print("graphviz imported successfully.") # 添加打印用于调试
except ImportError:
    HAS_GRAPHVIZ = False
    st.warning("未安装graphviz包，某些可视化功能将不可用。请使用pip install graphviz安装。")
    print("graphviz import failed.") # 添加打印用于调试

# Check if Graphviz dot executable is available and print its version
try:
    result = subprocess.run(['dot', '-V'], capture_output=True, text=True, check=True)
    print(f"Graphviz dot command is available. Version:\n{result.stdout.strip()}")
except FileNotFoundError:
    error_msg = "Error: 'dot' command not found. Graphviz is not installed or not in PATH. Image generation will fail."
    print(error_msg)
    print(f"Current PATH: {os.environ.get('PATH')}")
    st.error(error_msg)
except subprocess.CalledProcessError as e:
    error_msg = f"Error running 'dot -V': {e}\nStderr:\n{e.stderr.strip()}"
    print(error_msg)
    st.error(error_msg)
except Exception as e:
    error_msg = f"An unexpected error occurred checking dot: {e}"
    print(error_msg)
    st.error(error_msg)

# 标记前k个最大值的位置为1，其余为0
def mark_topk_positions(input_tensor, k):
    """标记每行前k个最大值的位置为1，其余为0"""
    assert input_tensor.dim() == 2, "输入应为二维张量"
    rows, cols = input_tensor.shape
    assert 1 <= k <= cols, f"k必须介于1和列数{cols}之间"

    result = torch.zeros((rows, cols), dtype=torch.long)
    for i in range(rows):
        row = input_tensor[i, :]
        sorted_indices = torch.argsort(row, descending=True)
        topk_indices = sorted_indices[:k]
        result[i, topk_indices] = 1
    return result

class TrainingVisualizer:
    def __init__(self, log_dir):
        """
        初始化可视化器

        参数:
        - log_dir: 训练日志和断点文件所在的目录
        """
        self.log_dir = log_dir
        self.checkpoints_dir = os.path.join(log_dir, 'checkpoints')
        self.log_file = os.path.join(log_dir, 'log.txt')
        self.epochs_data = self._parse_logs()

    def _parse_logs(self):
        """解析训练日志文件"""
        epochs = []
        current_epoch = None

        # 检查日志文件是否存在
        if not os.path.exists(self.log_file):
            st.error(f"日志文件未找到: {self.log_file}")
            print(f"Error: Log file not found at {self.log_file}") # 添加打印用于调试
            return []

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = list(f)
                i = 0
                while i < len(lines):
                    line = lines[i]
                    # 解析epoch信息
                    if "epoch" in line and "lr" in line:
                        if current_epoch:
                            epochs.append(current_epoch)
                        match = re.search(r"epoch (\d+) lr ([\d\.e+-]+)", line)
                        if match:
                            current_epoch = {
                                "epoch": int(match.group(1)),
                                "lr": float(match.group(2)),
                                "train_loss": [],
                                "train_acc": [],
                                "valid_acc": [],
                                "genotype": None,
                                "normal_alpha": None,
                                "reduce_alpha": None
                            }
                        i += 1
                        continue
                    # 解析训练指标
                    elif "train_acc" in line:
                        match = re.search(r"train_acc ([\d\.]+)", line)
                        if match and current_epoch:
                            current_epoch["train_acc"].append(float(match.group(1)))
                        i += 1
                        continue
                    elif "valid_acc" in line:
                        match = re.search(r"valid_acc ([\d\.]+)", line)
                        if match and current_epoch:
                            current_epoch["valid_acc"].append(float(match.group(1)))
                        i += 1
                        continue
                    # 解析架构参数 - normal_alpha
                    elif "alphas normal :" in line:
                        normal_alpha_lines = []
                        i += 1
                        while i < len(lines) and not ("alphas reduce :" in lines[i] or "genotype =" in lines[i]):
                            normal_alpha_lines.append(lines[i].strip())
                            i += 1
                        if normal_alpha_lines and current_epoch:
                            try:
                                all_text = " ".join(normal_alpha_lines).replace('[', ' ').replace(']', ' ')
                                nums = re.findall(r"[-+]?\d*\.\d+|\d+", all_text)
                                arr = np.array([float(x) for x in nums])
                                if arr.size % 7 == 0:
                                    arr = arr.reshape(-1, 7)
                                current_epoch["normal_alpha"] = arr
                            except Exception as e:
                                print(f"解析normal_alpha失败: {e}")
                                # 尝试从checkpoint加载
                                checkpoint_path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{current_epoch["epoch"]}.pt')
                                if os.path.exists(checkpoint_path):
                                    try:
                                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                                        if 'best_individual' in checkpoint:
                                            current_epoch['normal_alpha'] = checkpoint['best_individual'][0].cpu().numpy()
                                    except Exception as e:
                                        print(f"加载checkpoint失败 {checkpoint_path}: {str(e)}")
                        continue
                    # 解析架构参数 - reduce_alpha
                    elif "alphas reduce :" in line:
                        reduce_alpha_lines = []
                        i += 1
                        while i < len(lines) and not ("genotype =" in lines[i] or "alphas normal :" in lines[i]):
                            reduce_alpha_lines.append(lines[i].strip())
                            i += 1
                        if reduce_alpha_lines and current_epoch:
                            try:
                                all_text = " ".join(reduce_alpha_lines).replace('[', ' ').replace(']', ' ')
                                nums = re.findall(r"[-+]?\d*\.\d+|\d+", all_text)
                                arr = np.array([float(x) for x in nums])
                                if arr.size % 7 == 0:
                                    arr = arr.reshape(-1, 7)
                                current_epoch["reduce_alpha"] = arr
                            except Exception as e:
                                print(f"解析reduce_alpha失败: {e}")
                                # 尝试从checkpoint加载
                                checkpoint_path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{current_epoch["epoch"]}.pt')
                                if os.path.exists(checkpoint_path):
                                    try:
                                        checkpoint = torch.load(checkpoint_path, map_location='cpu')
                                        if 'best_individual' in checkpoint:
                                            current_epoch['reduce_alpha'] = checkpoint['best_individual'][1].cpu().numpy()
                                    except Exception as e:
                                        print(f"加载checkpoint失败 {checkpoint_path}: {str(e)}")
                        continue
                    # 解析基因型
                    elif "genotype = " in line:
                        try:
                            genotype_str = line.split("genotype = ")[1].strip()
                            normal_match = re.search(r"normal=\[(.*?)\]", genotype_str)
                            reduce_match = re.search(r"reduce=\[(.*?)\]", genotype_str)
                            if normal_match and reduce_match:
                                normal_list = ast.literal_eval("[" + normal_match.group(1) + "]")
                                reduce_list = ast.literal_eval("[" + reduce_match.group(1) + "]")
                                current_epoch["genotype"] = {"normal": normal_list, "reduce": reduce_list}
                            else:
                                current_epoch["genotype"] = None
                        except Exception as e:
                            print("genotype解析失败：", e)
                            current_epoch["genotype"] = None
                        i += 1
                        continue
                    else:
                        i += 1
                # 添加最后一个epoch
                if current_epoch:
                    epochs.append(current_epoch)
                # 加载每个epoch的架构参数（如果有断点文件）
                # 这里的checkpoint路径也需要基于log_dir
                for epoch in epochs:
                    checkpoint_path = os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch["epoch"]}.pt')
                    if os.path.exists(checkpoint_path):
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location='cpu')
                            if 'best_individual' in checkpoint:
                                epoch['normal_alpha'] = checkpoint['best_individual'][0].cpu().numpy()
                                epoch['reduce_alpha'] = checkpoint['best_individual'][1].cpu().numpy()
                            elif 'alpha_normal' in checkpoint and 'alpha_reduce' in checkpoint:
                                epoch['normal_alpha'] = checkpoint['alpha_normal'].cpu().numpy()
                                epoch['reduce_alpha'] = checkpoint['alpha_reduce'].cpu().numpy()
                            print(f"成功加载检查点 {checkpoint_path}")
                        except Exception as e:
                            print(f"加载检查点失败 {checkpoint_path}: {str(e)}")
                    print(f"Epoch {epoch['epoch']} normal_alpha shape: {epoch['normal_alpha'].shape if epoch['normal_alpha'] is not None else 'None'}")
                return epochs
        except Exception as e:
            st.error(f"解析日志文件时出错: {str(e)}")
            print(f"Error during log file parsing: {str(e)}\n{traceback.format_exc()}") # 添加打印用于调试
            return []

    def visualize_genotype(self, genotype):
        """可视化神经网络架构"""
        if genotype is None:
            return None
        G = Network(height="400px", width="100%", directed=True, notebook=True)
        G.set_options("""
        {
            "nodes": {
                "font": {"size": 20},
                "shape": "dot",
                "size": 30
            },
            "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 1}},
                "smooth": {"enabled": true, "type": "continuous"}
            },
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 200,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
            }
        }
        """)
        # 添加输入节点
        G.add_node("n0", label="Input", color="#97C2FC")
        G.add_node("n1", label="Input_1", color="#97C2FC")
        node_names = ["n0", "n1"]
        # normal cell
        node_idx = 2
        for i, (op, idx) in enumerate(genotype['normal']):
            node_name = f"n{node_idx}"
            G.add_node(node_name, label=op, color="#FFB6C1")
            G.add_edge(node_names[idx], node_name)
            node_names.append(node_name)
            node_idx += 1
        # reduce cell
        reduce_node_names = []
        for i, (op, idx) in enumerate(genotype['reduce']):
            node_name = f"r{i+2}"  # reduce cell节点编号从r2开始
            G.add_node(node_name, label=op, color="#98FB98")
            # reduce cell的输入只能来自输入节点或normal cell节点
            if idx < len(node_names):
                G.add_edge(node_names[idx], node_name)
            else:
                # 防御性处理，避免越界
                G.add_edge("n0", node_name)
            reduce_node_names.append(node_name)
        return G

    def plot_metrics(self, epochs_data, selected_epoch):
        """绘制训练指标图表"""
        df = pd.DataFrame([{
            "epoch": e["epoch"],
            "train_acc": e["train_acc"][-1] if e["train_acc"] else None,
            "valid_acc": e["valid_acc"][-1] if e["valid_acc"] else None,
            "lr": e["lr"]
        } for e in epochs_data[:selected_epoch+1]])

        fig = go.Figure()

        # 添加准确率曲线
        fig.add_trace(go.Scatter(
            x=df.epoch, y=df.train_acc,
            name="训练准确率",
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=df.epoch, y=df.valid_acc,
            name="验证准确率",
            line=dict(color='#ff7f0e')
        ))

        # 添加学习率曲线（使用次坐标轴）
        fig.add_trace(go.Scatter(
            x=df.epoch, y=df.lr,
            name="学习率",
            yaxis="y2",
            line=dict(color='#2ca02c', dash='dash')
        ))

        # 更新布局
        fig.update_layout(
            title="训练过程指标",
            xaxis_title="Epoch",
            yaxis_title="准确率",
            yaxis2=dict(
                title="学习率",
                overlaying="y",
                side="right"
            ),
            hovermode="x unified",
            showlegend=True
        )

        return fig

    def plot_alpha_heatmap(self, alpha_matrix, title, binary_mask=None):
        """绘制alpha参数热力图"""
        if alpha_matrix is None:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(alpha_matrix, cmap='viridis')

        # 添加颜色条
        plt.colorbar(im, ax=ax)

        # 设置标题和标签
        ax.set_title(title)
        ax.set_xlabel("操作索引")
        ax.set_ylabel("边索引")

        # 如果提供了二进制掩码，在热力图上标记topk位置
        if binary_mask is not None:
            rows, cols = binary_mask.shape
            for i in range(rows):
                for j in range(cols):
                    if binary_mask[i, j] == 1:
                        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=2)
                        ax.add_patch(rect)

        return fig

    def plot_genotype_graph(self, genotype, cell_type="normal"):
        if genotype is None or not HAS_GRAPHVIZ:
            st.error("未检测到genotype或Graphviz不可用")
            return None
        try:
            if cell_type == "normal":
                cell_genotype = genotype["normal"]
                caption = "Normal Cell"
            else:
                cell_genotype = genotype["reduce"]
                caption = "Reduce Cell"
            tmp_path = os.path.join(tempfile.gettempdir(), f"nas_{uuid.uuid4().hex}")
            viz.plot(cell_genotype, tmp_path, caption)
            rendered_path = tmp_path + '.png'
            if os.path.exists(rendered_path):
                return rendered_path
            else:
                st.error(f"图片文件未生成: {rendered_path}")
                print(f"Error: Rendered image file not found at {rendered_path}") # 添加打印用于调试
                return None
        except Exception as e:
            st.error(f"架构图生成失败: {str(e)}\n{traceback.format_exc()}")
            print(f"Error generating genotype graph: {str(e)}\n{traceback.format_exc()}") # 添加打印用于调试
            return None

@st.cache_resource
def get_visualizer(selected_exp):
    return TrainingVisualizer(selected_exp)

@st.cache_data
def get_alpha_df(alpha):
    if alpha is not None and hasattr(alpha, 'size') and alpha.size > 0:
        df = pd.DataFrame(alpha)
        df = df.applymap(lambda x: f"{x:.4f}")
        return df
    return None

@st.cache_data
def get_genotype_img(genotype, cell_type, selected_exp):
    # 由于visualizer不可序列化，传入selected_exp，内部重新获取
    visualizer = get_visualizer(selected_exp)
    return visualizer.plot_genotype_graph(genotype, cell_type)

def main():
    st.set_page_config(layout="wide", page_title="NAS训练可视化")
    st.title("神经架构搜索训练过程可视化")
    st.sidebar.title("控制面板")

    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(__file__)

    # 构建 search-* 目录的完整路径模式
    # 根据您GitHub的截图，search-* 目录与 visualize_training.py 在同一目录下
    search_dir_pattern = os.path.join(script_dir, "search-*")

    # 使用新的模式来查找目录
    exp_dirs = glob.glob(search_dir_pattern)

    # ====== 临时添加的代码，用于排查 (保留这段有助于确认修改是否生效) ======
    st.write(f"当前脚本目录: {script_dir}")
    st.write(f"用于查找实验目录的模式: {search_dir_pattern}")
    try:
        # 列出脚本所在目录下的文件和目录
        st.write(f"脚本目录下的文件和目录: {os.listdir(script_dir)}")
    except Exception as e:
        st.write(f"列出脚本目录内容时出错: {e}")
    st.write(f"glob.glob('{search_dir_pattern}') 的结果: {exp_dirs}")
    # ==================================================================

    if not exp_dirs:
        st.error(f"未找到训练实验目录！在 '{script_dir}' 中查找 '{search_dir_pattern}' 时没有找到匹配项。请确保search-*目录存在且名称正确。")
        print(f"Error: No experiment directories found using pattern {search_dir_pattern} in {script_dir}") # 添加打印用于调试
        return

    selected_exp = st.sidebar.selectbox(
        "选择实验目录",
        exp_dirs,
        format_func=lambda x: os.path.basename(x)
    )

    # 由于TrainingVisualizer需要完整的目录路径，这里直接传入找到的目录
    visualizer = get_visualizer(selected_exp)

    if not visualizer.epochs_data:
        st.error("无法加载训练数据！请检查log.txt文件是否存在且内容正确。")
        print(f"Error: Failed to load training data from {selected_exp}") # 添加打印用于调试
        return

    max_epoch = len(visualizer.epochs_data) - 1
    if max_epoch == 0:
        selected_epoch = 0
        st.sidebar.text("当前只有一个Epoch (0)")
    else:
        selected_epoch = st.sidebar.slider(
            "选择Epoch",
            0, max_epoch,
            0,
            help="滑动选择要查看的训练轮次"
        )
    st.sidebar.markdown("### 架构参数设置")
    use_topk = st.sidebar.checkbox("启用TopK掩码", help="启用二进制掩码显示")
    normal_alpha = visualizer.epochs_data[selected_epoch].get("normal_alpha")
    max_k = 7
    if normal_alpha is not None and hasattr(normal_alpha, 'shape'):
        max_k = normal_alpha.shape[1]
    topk_value = st.sidebar.slider(
        "选择TopK值",
        1, max_k, 1,
        help="选择每行保留的前k个最大值，用于生成二进制掩码矩阵",
        disabled=not use_topk
    )
    auto_play = st.sidebar.checkbox("自动播放", help="自动播放训练过程")
    if auto_play:
        time_interval = st.sidebar.slider("播放速度(秒/epoch)", 0.5, 5.0, 1.0)
    st.subheader("架构参数和架构图")
    # 第一排：参数表格
    st.subheader("架构参数")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Normal Alpha")
        normal_alpha = visualizer.epochs_data[selected_epoch].get("normal_alpha")
        if use_topk and normal_alpha is not None:
            normal_tensor = torch.tensor(normal_alpha)
            mask = mark_topk_positions(normal_tensor, topk_value).numpy()
            df = get_alpha_df(mask)
        else:
            df = get_alpha_df(normal_alpha)
        if df is not None:
            st.dataframe(df, use_container_width=True, height=320)
        else:
            st.warning("无有效normal_alpha参数")
    with col2:
        st.markdown("#### Reduce Alpha")
        reduce_alpha = visualizer.epochs_data[selected_epoch].get("reduce_alpha")
        if use_topk and reduce_alpha is not None:
            reduce_tensor = torch.tensor(reduce_alpha)
            mask = mark_topk_positions(reduce_tensor, topk_value).numpy()
            df = get_alpha_df(mask)
        else:
            df = get_alpha_df(reduce_alpha)
        if df is not None:
            st.dataframe(df, use_container_width=True, height=320)
        else:
            st.warning("无有效reduce_alpha参数")

    # 第二排：架构图
    st.subheader("架构图")
    col3, col4 = st.columns(2)
    with col3:
        genotype = visualizer.epochs_data[selected_epoch]["genotype"]
        if genotype and HAS_GRAPHVIZ:
            img_path = get_genotype_img(genotype, "normal", selected_exp)
            if img_path:
                st.image(img_path, use_container_width=True)
    with col4:
        if genotype and HAS_GRAPHVIZ:
            img_path = get_genotype_img(genotype, "reduce", selected_exp)
            if img_path:
                st.image(img_path, use_container_width=True)

    # 第三排：训练指标和epoch详情
    st.subheader("训练指标与当前Epoch详情")
    col5, col6 = st.columns([2, 1])
    with col5:
        fig = visualizer.plot_metrics(visualizer.epochs_data, selected_epoch)
        st.plotly_chart(fig, use_container_width=True)
    with col6:
        current_epoch = visualizer.epochs_data[selected_epoch]
        train_acc = current_epoch['train_acc'][-1] if current_epoch['train_acc'] else None
        valid_acc = current_epoch['valid_acc'][-1] if current_epoch['valid_acc'] else None
        if train_acc is not None:
            if train_acc > 1.1:
                train_acc_str = f"{train_acc:.2f}%"
            else:
                train_acc_str = f"{train_acc*100:.2f}%"
        else:
            train_acc_str = "N/A"
        if valid_acc is not None:
            if valid_acc > 1.1:
                valid_acc_str = f"{valid_acc:.2f}%"
            else:
                valid_acc_str = f"{valid_acc*100:.2f}%"
        else:
            valid_acc_str = "N/A"
        st.markdown(f"""
        - **Epoch**: {current_epoch['epoch']}
        - **学习率**: {current_epoch['lr']:.2e}
        - **训练准确率**: {train_acc_str}
        - **验证准确率**: {valid_acc_str}
        """)
    if auto_play:
        time.sleep(time_interval)
        if selected_epoch < max_epoch:
            st.rerun()

if __name__ == "__main__":
    # 在云端环境中，我们不需要设置Graphviz路径，可以移除或注释掉这行
    # os.environ['PATH'] = r'E:\Graphviz\bin;' + os.environ['PATH']
    main()
