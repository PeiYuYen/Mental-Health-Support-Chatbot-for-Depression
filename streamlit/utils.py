import streamlit as st
from datetime import datetime
import pandas as pd
import sqlite3
import plotly.express as px
import whisper
import tempfile
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3
import json
import re
from datetime import datetime

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

import sqlite3
from datetime import datetime


# 初始化 LLM
MODEL_PATH = 'lzw1008/Emollama-chat-7b'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
emo_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto',offload_folder="offload")

# 改良版情緒分析函式（分類 + 強度）
def analyze_emotions(text: str) -> dict:
    prompt = f"""Human:
Task: Analyze the emotional tone of the following text. Identify the presence of one or more of the following emotions and assign each detected emotion a score from 0 to 1 indicating its intensity.

Emotions to consider: anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust.

Text: {text}

Output format:
{{"emotion_1": score, "emotion_2": score, ...}}

Assistant:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(emo_model.device)
    generate_ids = emo_model.generate(inputs["input_ids"], max_length=1024)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]

    print("🔎 原始模型回應:\n", response)

    emotion_labels = {
        "anger", "anticipation", "disgust", "fear", "joy", "love",
        "optimism", "pessimism", "sadness", "surprise", "trust"
    }

    try:
        after_assistant = response.split("Assistant:")[-1]

        # 優先解析 JSON 格式
        json_match = re.search(r"\{.*?\}", after_assistant, re.DOTALL)
        if json_match:
            emotions_str = json_match.group()
            emotions = json.loads(emotions_str)
            # 過濾掉非情緒鍵
            filtered = {
                k.lower(): float(v)
                for k, v in emotions.items()
                if k.lower() in emotion_labels
            }
            if filtered:
                return filtered

        # fallback：解析 "0.45 disgust" 格式
        pattern = r"(\d*\.\d+|\d+)\s+([a-zA-Z]+)"
        matches = re.findall(pattern, after_assistant)
        fallback_emotions = {
            emotion.lower(): float(score)
            for score, emotion in matches
            if emotion.lower() in emotion_labels
        }

        return fallback_emotions if fallback_emotions else {}

    except Exception as e:
        print("❌ 解析失敗：", e)
        return {}


def get_connection():
    conn = sqlite3.connect("journal.db")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS journal_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            date TEXT,
            content TEXT,
            emotions TEXT
        )
    ''')
    conn.commit()
    return conn

# 儲存到 SQLite
def store_analysis(user_id: str, text: str, emotions: dict):
    conn = get_connection()
    cursor = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emotions_json = json.dumps(emotions, ensure_ascii=False)
    cursor.execute('''
        INSERT INTO journal_entries (user_id, date, content, emotions)
        VALUES (?, ?, ?, ?)
    ''', (user_id, date_str, text, emotions_json))
    conn.commit()
    conn.close()


# 建議回饋產生器
def generate_suggestion(dominant_emotion: str) -> str:
    suggestions = {
        "sadness": "看起來你最近有些低落，也許可以試著寫下三件感謝的事，或安排與朋友見面。",
        "pessimism": "建議嘗試每日一句正面肯定語，調整思維視角。",
        "joy": "你看起來很開心，記得紀錄並珍惜這些好時刻！",
        "fear": "如果你感到擔憂，可以考慮深呼吸冥想，或與信任的朋友聊聊。",
        "anger": "當生氣時，試試寫下原因並暫時離開情境，給自己一個緩衝時間。",
        "trust": "你展現了信任的情緒，也許可以更多與他人合作與連結。"
    }
    return suggestions.get(dominant_emotion, "建議多關注自己的內在感受，給自己一些溫柔的空間。")

# 新增音訊轉文字功能
@st.cache_resource
def load_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """載入 Whisper 模型（使用 cache 避免重複載入）, 嘗試 GPU, 若 VRAM 不足或 GPU 不可用則使用 CPU"""
    if torch.cuda.is_available():
        try:
            print(f"Attempting to load Whisper model '{model_size}' on GPU...")
            model = whisper.load_model(model_size, device="cuda")
            print(f"Whisper model '{model_size}' successfully loaded on GPU.")
            return model
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "not enough memory" in str(e).lower():
                print(f"CUDA out of memory when loading Whisper model '{model_size}' on GPU. Falling back to CPU.")
            else:
                print(f"RuntimeError encountered when loading Whisper model '{model_size}' on GPU: {e}. Falling back to CPU.")
        except Exception as e: # Catch any other unexpected errors during GPU load
            print(f"An unexpected error occurred when trying to load Whisper model '{model_size}' on GPU: {e}. Falling back to CPU.")
    else:
        print("CUDA not available. Proceeding to load Whisper model on CPU.")

    # Fallback to CPU if CUDA is not available or if GPU loading failed
    print(f"Loading Whisper model '{model_size}' on CPU...")
    model = whisper.load_model(model_size, device="cpu")
    print(f"Whisper model '{model_size}' successfully loaded on CPU.")
    return model

def transcribe_audio(audio_bytes: bytes) -> str:
    """將音訊位元組轉換為文字"""
    try:
        # 創建暫存檔案
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = tmp_audio.name
        
        # 載入模型並轉錄
        model = load_whisper_model("medium")
        result = model.transcribe(tmp_audio_path, language="zh")
        
        # 清理暫存檔案
        os.remove(tmp_audio_path)
        
        return result["text"].strip()
    
    except Exception as e:
        st.error(f"音訊轉錄失敗: {e}")
        return ""


## RAG
# ————— 參數設定 —————
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIALOG_INDEX   = "./rag_dataset/dialog_index.faiss"    # 歷史對話向量索引檔
SUPPORT_INDEX  = "./rag_dataset/support_index.faiss"   # 支持材料向量索引檔
DIALOG_CHUNKS  = "./rag_dataset/dialog_chunks.json"    # 歷史對話片段與對應回覆
SUPPORT_CHUNKS = "./rag_dataset/support_chunks.json"   # 指導／理由／摘要／背景片段

SIM_THRESH     = 0.9   # 相似度門檻
SUPPORT_THRESH = 1.25
TOP_K          = 5      # 最多檢索幾條支持材料

# ————— 載入模型與索引 —————
embedder      = SentenceTransformer(EMBED_MODEL)
dialog_index  = faiss.read_index(DIALOG_INDEX)
support_index = faiss.read_index(SUPPORT_INDEX)

with open(DIALOG_CHUNKS, 'r', encoding='utf-8') as f:
    dialog_chunks = json.load(f)
with open(SUPPORT_CHUNKS, 'r', encoding='utf-8') as f:
    support_chunks = json.load(f)

def retrieve_dialog(query: str):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)     # 就地归一化
    D, I  = dialog_index.search(q_emb, 1)
    sim, idx = float(D[0][0]), int(I[0][0])
    return sim, dialog_chunks[idx]

def retrieve_support_materials(query: str, top_k: int = 1):
    """
    输入：用户 query，返回最相似的 session 的 background、reasoning、summary 及相似度
    """
    # a) 向量化并归一化
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb /= (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10)

    # b) 检索
    D, I = support_index.search(q_emb, top_k)
    results = []
    for sim, idx in zip(D[0], I[0]):
        if sim >= SUPPORT_THRESH:
            entry = support_chunks[idx]
            results.append({
                "session_id": entry["session_id"],
                "background": entry["background"],
                "reasoning": entry["reasoning"],
                "summary": entry["summary"],
                "similarity": float(sim)
            })
    return results

def retrieve_support(query: str):
    """
    檢索支持材料：輸入使用者提問，返回所有相似度>=門檻的支持片段列表
    """
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = support_index.search(q_emb, TOP_K)
    hits = []
    for sim, idx in zip(D[0], I[0]):
        if sim >= SUPPORT_THRESH:
            hits.append((float(sim), support_chunks[idx]))
    # 按相似度由高到低排序
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits  # 列表元素為 (相似度, 片段_dict)

def make_unified_prompt(query: str,
                        dlg_sim: float,
                        dlg_entry: dict,
                        support_hits: list[tuple[float, dict]]):
    """
    統一構造 Prompt：
    - 顯示使用者問題
    - 若歷史對話相似度足夠，高亮最相似的一輪示例
    - 列出所有檢索到的支持材料
    - 最後要求模型以心理諮商師的口吻給出回應
    """
    parts =""
    if dlg_sim >= SIM_THRESH:
        print(dlg_sim)
        parts += "以下是其他個案中最相似的一輪心理諮商談話，請注意，來訪者面臨問題不盡相同，只需學習諮商師的回應方向和角度：\n"
        parts += f"  Question：{dlg_entry['question']}"
        parts += f"  Response：{dlg_entry['response']}"
        return parts
    elif support_hits:
        parts += "以下是與您的問題相關的諮商指導或背景材料，請注意，來訪者面臨問題不盡相同，只需參考建議措施與諮商師的回應方向和角度："
        for sim, hit in support_hits:
            print(sim)
            parts += f" 來訪者問題大綱:{hit['background']}"
            parts += f" 建議措施:{hit['reasoning']}"
            parts += f" 諮商師實際措施:{hit['summary']}"
        return parts
    else:
        return None

def rerank_with_cross_encoder(query: str, hits: list[tuple[float, dict]]) -> list[tuple[float, dict, float]]:
    """
    对 FAISS hits 做二次排序。
    输入：query, hits = [(faiss_sim, entry_dict), …]
    输出：[(faiss_sim, entry_dict, rerank_score), …]，按 rerank_score 降序
    """
    # 准备交叉打分的输入对列表 [(query, text), …]
    pairs = [(query, entry["background"]) for _, entry in hits]
    # 2. 让 cross-encoder 打分
    rerank_scores = reranker.predict(pairs)  # 返回一个 list of floats

    # 3. 合并原始 sim, entry, rerank score
    combined = []
    for (faiss_sim, entry), rerank_score in zip(hits, rerank_scores):
        combined.append((faiss_sim, entry, float(rerank_score)))

    # 4. 按 cross-encoder 分数降序排列
    combined.sort(key=lambda x: x[2], reverse=True)
    return combined[:1]
    
##記憶管理

def init_db():
    conn = sqlite3.connect("journal.db")
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS chat_history (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   TEXT NOT NULL,
        role      TEXT NOT NULL,
        content   TEXT NOT NULL,
        ts        TEXT NOT NULL
      )
    """)
    conn.commit()
    conn.close()

def log_chat_message(user_id, role: str, content: str):
    """
    把一條對話記錄（user／assistant、內容、時間戳）寫入 SQLite。
    """
    conn = sqlite3.connect("journal.db")
    c = conn.cursor()
    c.execute("""
      CREATE TABLE IF NOT EXISTS chat_history (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   TEXT NOT NULL,
        role      TEXT NOT NULL,    -- 'user' 或 'assistant'
        content   TEXT NOT NULL,
        ts        TEXT NOT NULL     -- ISO 時間字串
      )
    """)
    c.execute(
      "INSERT INTO chat_history (user_id, role, content, ts) VALUES (?, ?, ?, ?)",
      (user_id, role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def load_chat_history(user_id, limit: int = 100) -> list[dict]:
    """
    讀取最近 limit 條該使用者的對話，返回 dict list。
    """
    init_db()
    conn = sqlite3.connect("journal.db")
    c = conn.cursor()
    c.execute("""
      SELECT role, content
        FROM chat_history
       WHERE user_id = ?
       ORDER BY ts ASC
       LIMIT ?
    """, (user_id, limit))
    rows = c.fetchall()
    conn.close()
    # 每個 entry 變成 {'role': 'user'/'assistant', 'content': '...'}
    return [{"role": r, "content": t} for r, t in rows]


if __name__ == "__main__":
    # ——— 測試範例 ———
    def dummy_generator(prompt: str) -> str:
        ##自己寫model
        print("=== 給模型的 Prompt ===")
        print(prompt)
        return "（模型根據上述資訊生成的專業回應）"
    
    user_q = "我總覺得壓力很大，不知道該怎麼紓解。"
    resp = mixed_rag_response(user_q, dummy_generator)
    print("諮商師：", resp)