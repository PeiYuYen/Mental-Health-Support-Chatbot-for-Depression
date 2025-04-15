import streamlit as st
from datetime import datetime
import pandas as pd
import sqlite3
import plotly.express as px

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sqlite3
import json
import re
from datetime import datetime

# 初始化 LLM
MODEL_PATH = 'lzw1008/Emollama-chat-7b'
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
emo_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')

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
