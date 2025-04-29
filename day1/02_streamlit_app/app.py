# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder
import os
import time

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# パフォーマンスモニタリング関数
def log_performance(action, start_time):
    """パフォーマンスメトリクスを記録する"""
    elapsed = time.time() - start_time
    st.session_state.perf_logs.append(f"{action}: {elapsed:.2f}秒")

# セッション状態の初期化
if 'perf_logs' not in st.session_state:
    st.session_state.perf_logs = []

# LLMモデルのロード（キャッシュを利用）
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    start_time = time.time()
    try:
        # GPUメモリの使用状況を確認（CUDAが利用可能な場合）
        if torch.cuda.is_available():
            device = "cuda"
            mem_info = torch.cuda.mem_get_info()
            free_mem = mem_info[0] / (1024 ** 3)  # GBに変換
            total_mem = mem_info[1] / (1024 ** 3)  # GBに変換
            st.info(f"GPU: {torch.cuda.get_device_name(0)} - 空きメモリ: {free_mem:.2f}GB / 合計: {total_mem:.2f}GB")
        else:
            device = "cpu"
            st.info(f"Using device: {device} (GPUが利用できません)")
        
        # モデルロード処理
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        log_performance("モデルロード", start_time)
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。(所要時間: {time.time() - start_time:.2f}秒)")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

# モデルをロード
pipe = load_model()

# --- Streamlit アプリケーション ---
st.title("🤖 Gemma 2 Chatbot with Feedback")
st.write("Gemmaモデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "チャット" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理", "パフォーマンス"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理", "パフォーマンス"].index(st.session_state.page)
    if st.session_state.page in ["チャット", "履歴閲覧", "サンプルデータ管理", "パフォーマンス"] else 0,
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector)
)

# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()
elif st.session_state.page == "パフォーマンス":
    st.header("システムパフォーマンス")
    
    # システム情報の表示
    st.subheader("システム情報")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Python Version", os.sys.version.split()[0])
        st.metric("Torch Version", torch.__version__)
    with col2:
        st.metric("CUDA Available", "Yes" if torch.cuda.is_available() else "No")
        if torch.cuda.is_available():
            st.metric("CUDA Version", torch.version.cuda)
    
    # パフォーマンスログの表示
    st.subheader("パフォーマンスログ")
    if st.session_state.perf_logs:
        for log in st.session_state.perf_logs:
            st.text(log)
    else:
        st.info("パフォーマンスデータはまだ記録されていません。")
    
    if st.button("ログをクリア"):
        st.session_state.perf_logs = []
        st.experimental_rerun()

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")
st.sidebar.text(f"最終更新: {time.strftime('%Y-%m-%d %H:%M')}")
