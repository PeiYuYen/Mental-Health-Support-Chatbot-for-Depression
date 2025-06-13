import streamlit as st
from streamlit_chat import message
from langchain_ollama import OllamaLLM
import json
import os
import pandas as pd
import plotly.express as px
import re
import numpy as np
import matplotlib.pyplot as plt
from streamlit_chat_widget import chat_input_widget
from streamlit_extras.bottom_container import bottom
from utils import generate_suggestion, analyze_emotions, store_analysis, get_connection, transcribe_audio, retrieve_dialog, retrieve_support, make_unified_prompt, load_chat_history, log_chat_message, rerank_with_cross_encoder

# os.environ["STREAMLIT_WATCHER_IGNORE_FILES"] = "torch"

DB_FILE = 'user_db.json'
if not os.path.exists(DB_FILE):
    with open(DB_FILE, 'w') as file:
        db = {"users": {}}
        json.dump(db, file)
else:
    with open(DB_FILE, 'r') as file:
        db = json.load(file)

def save_db():
    with open(DB_FILE, 'w') as file:
        json.dump(db, file)

# Loading the model of your choice
llm = OllamaLLM(model='llama3.2:3b')

def main():
    st.title("🥰 Mental Health Support Chatbot")

    username = st.session_state.get("username")

    st.sidebar.title(f"👋 Welcome! **{username}**")

    #if 有做user_data
    user_history = []
    # 1. 初始化 user_history 到 session_state
    if "user_history" not in st.session_state:
        st.session_state["user_history"] = []

    # 2. 第一次进来，加载一次历史
    if "history_loaded" not in st.session_state:
        hist = load_chat_history(username)
        for e in hist:
            st.session_state["user_history"].append({
                "role":    e["role"],
                "type":    "text",
                "content": e["content"]
            })
        st.session_state["history_loaded"] = True
    
    # 初始化 session_state
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
    if "last_figure" not in st.session_state:
        st.session_state.last_figure = None
    if "last_suggestion" not in st.session_state:
        st.session_state.last_suggestion = None

    page = st.sidebar.radio("Select operating mode", ["💬 Chat Support Mode", "📆 Diary Analysis Mode"])

    st.sidebar.write(f"Current Page: {page}")

    if page == "💬 Chat Support Mode":
        st.subheader("💬 聊天輔導支持")

        # Inject CSS to control iframe height for chat_input_widget
        st.markdown("""<style>
        iframe[title="streamlit_chat_widget.chat_input_widget"] {
            height: 80px !important; /* You can adjust this value */
        }
        </style>""", unsafe_allow_html=True)

        # Function for handling conversation with history

        
        def conversational_chat(query):
            messages = [
                {"role": "user" if entry["role"] == "user" else "assistant", "content": entry["content"]}
                for entry in st.session_state['history']
            ]
            # messages.append({"role": "user", "content": query})  # 新增使用者輸入
            print(messages)
            response = llm.invoke(messages)  # Pass conversation context
            print(response)
            # **更新對話歷史，但不會重複新增**
            for i in range(len(st.session_state['history']) - 1, -1, -1):
                if st.session_state['history'][i]["content"] == "⏳ ...":
                    st.session_state['history'][i] = {"role": "assistant", "type": "text", "content": response}  # ✅ 更新 AI 回應
                    break

            st.session_state['waiting_for_response'] = None  # 清除等待狀態
            
            return response
        
        def conversational_chat_with_rag(query):
    
            # 1) 先把历史对话打包成 messages​'
            print("user_history")
            print(user_history)
            combined = st.session_state["user_history"] + st.session_state["history"]
            messages = [
                {"role": "user" if e["role"]=="user" else "assistant",
                "content": e["content"]}
                for e in combined
            ]
            # 1) 歷史對話檢索
            dlg_sim, dlg_entry = retrieve_dialog(query)
            
            # 2) 支持材料檢索
            support_hits = retrieve_support(query)
            #reranked = rerank_with_cross_encoder(query, support_hits)
            
            # 3) 統一構造 Prompt
            prompt = make_unified_prompt(query, dlg_sim, dlg_entry, support_hits)
            if prompt != None:
                messages[-2]['content'] = f"{messages[-2]['content']}\n\n{prompt}"
            print(messages)
            response = llm.invoke(messages)
            print(response)

            prompt = f"""
            你是一名助理。請判斷下面這句話是否包含使用者的重要個人經歷，需要被記錄下來以便後續參考。
            請僅回答「是」或「否」，不要多餘文字。

            句子："{query}"
            """
            resp = llm.invoke([{"role":"system","content":prompt}])
            if resp.strip() == "是":
                log_chat_message(username, "user", query)

            # **更新對話歷史，但不會重複新增**
            for i in range(len(st.session_state['history']) - 1, -1, -1):
                if st.session_state['history'][i]["content"] == "⏳ ...":
                    st.session_state['history'][i] = {"role": "assistant", "type": "text", "content": response}  # ✅ 更新 AI 回應
                    break

            st.session_state['waiting_for_response'] = None  # 清除等待狀態
            return response

        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        if 'waiting_for_response' not in st.session_state:
            st.session_state['waiting_for_response'] = None  # 存放等待 AI 回應的訊息 
 


        message("哈囉! 歡迎你來到這裡。你可以在這裡放心地說你想說的任何事，要不要花點時間跟我聊聊呢?", avatar_style="thumbs")

        # **顯示歷史對話**
        chat_container = st.container()
        with chat_container:
            for i, entry in enumerate(st.session_state['history']):
                if entry["role"] == "user" and entry["type"] == "text":
                    message(entry["content"], is_user=True, key=f"user_{i}")
                elif entry["role"] == "assistant" and entry["type"] == "text":
                    message(entry["content"], key=f"bot_{i}", avatar_style="thumbs")

        # **處理等待中的 AI 回應**
        if st.session_state['waiting_for_response']:
            user_input_text = st.session_state['waiting_for_response']
            
            # **找到最後一筆 "⏳ ..." 並更新**
            for i in range(len(st.session_state['history']) - 1, -1, -1):
                if st.session_state['history'][i]["content"] == "⏳ ...":
                    #response = conversational_chat(user_input_text)  # 取得 LLM 回應
                    response = conversational_chat_with_rag(user_input_text)
                    st.session_state['history'][i] = {"role": "assistant", "type": "text", "content": response}  # **直接替換 bot 的回應**
                    #log_chat_message(username, "assistant", response)
                    st.session_state['waiting_for_response'] = None  # 清除等待狀態
                    st.rerun()  # 🔄 重新渲染頁面，讓 AI 回應顯示
                    break

        # **新的音訊/文字輸入框**
        with bottom():
            user_input = chat_input_widget()

        if user_input:
            if st.session_state['waiting_for_response'] is None:  # 只有在沒有等待中的回應時才加入新訊息
                input_text = ""
                
                # 處理文字輸入
                if "text" in user_input:
                    input_text = user_input["text"]
                    st.session_state['history'].append({"role": "user", "type": "text", "content": input_text})
                    #log_chat_message(username, 'user', input_text)
                
                # 處理音訊輸入
                elif "audioFile" in user_input:
                    with st.spinner("🎤 正在轉錄音訊..."):
                        audio_bytes = bytes(user_input["audioFile"])
                        input_text = transcribe_audio(audio_bytes)
                        if input_text:
                            st.session_state['history'].append({"role": "user", "type": "text", "content": f"{input_text}"})
                            log_chat_message(username, 'user', input_text)
                        else:
                            st.error("音訊轉錄失敗，請重新嘗試")
                            return
                
                if input_text:
                    st.session_state['history'].append({"role": "assistant", "type": "text", "content": "⏳ ..."})  # 顯示等待中的訊息
                    st.session_state['waiting_for_response'] = input_text  # 標記等待 AI 回應
                    st.rerun()  # 立即更新畫面，讓使用者輸入先顯示

        # **滾動到底部標記**
        st.markdown("<div id='scroll-bottom'></div>", unsafe_allow_html=True)

        # **使用 JavaScript 自動滾動到底部**
        st.markdown(
            """
            <script>
            var scrollBottom = document.getElementById("scroll-bottom");
            if (scrollBottom) {
                scrollBottom.scrollIntoView({ behavior: "smooth" });
            }
            </script>
            """, unsafe_allow_html=True
        )
    elif page == "📆 Diary Analysis Mode":

        user_id = st.session_state.get("username")
        st.subheader("📝 我的情緒日記分析")
        
        # 文字輸入區域
        diary_text = st.text_area("請輸入你今天的心情與事件...", height=200)
        
        # 音訊輸入區域
        st.write("**或者用語音分享你的心情：**")
        audio_input = st.audio_input("🎤 錄製語音日記")
        
        # 處理音訊輸入
        if audio_input and not diary_text.strip():
            with st.spinner("正在轉錄語音日記..."):
                audio_bytes = audio_input.getvalue()
                transcribed_text = transcribe_audio(audio_bytes)
                if transcribed_text:
                    diary_text = transcribed_text
                    st.success(f"語音轉錄完成: {transcribed_text}")
                else:
                    st.error("語音轉錄失敗，請重新錄製")

        if st.button("分析我的情緒"):
            if diary_text.strip():
                with st.spinner("分析中..."):
                    emotions = analyze_emotions(diary_text)
                    print("日記內容：", diary_text)
                    print("使用者 ID：", user_id)
                    print("情緒分析結果：", emotions)
                    store_analysis(user_id, diary_text, emotions)
                    
                # 顯示分析結果
                st.success("✅ 分析完成！")
                if emotions:
                    st.session_state.last_analysis = emotions
                    df = pd.DataFrame(list(emotions.items()), columns=["情緒", "強度"])
                    df["強度"] = df["強度"].astype(float)
                    # st.dataframe(df.set_index("情緒"))

                    # 圓餅圖視覺化
                    fig = px.pie(df, names="情緒", values="強度", hole=0.3)
                    st.session_state.last_figure = fig
                    # 💡 建議區
                    dominant = max(emotions, key=emotions.get)
                    st.session_state.last_suggestion = generate_suggestion(dominant)

            else:
                st.warning("請輸入日記內容或錄製語音才能分析喔。")
            
        # 顯示之前的分析結果（如果有的話）
        if st.session_state.last_analysis:
            st.subheader("🎯 情緒分析結果")
            st.plotly_chart(st.session_state.last_figure)
            st.subheader("💡 建議回饋")
            st.info(st.session_state.last_suggestion)

        # 歷史查詢選擇
        with st.expander("📈 查看過去的情緒趨勢"):
            conn = get_connection()
            try:
                df = pd.read_sql_query(
                    "SELECT * FROM journal_entries WHERE user_id = ?",
                    conn, params=(user_id,)
                )
            except Exception as e:
                st.error(f"無法讀取歷史資料：{e}")
                df = pd.DataFrame()
            conn.close()

            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df["parsed"] = df["emotions"].apply(eval)  # 將字串轉 dict
                emotion_df = pd.json_normalize(df["parsed"]).fillna(0)
                emotion_df["date"] = df["date"]

                # 安全處理 default 選項
                options = emotion_df.columns.drop("date").tolist()
                default_emotions = [e for e in ["sadness", "joy"] if e in options]
                selected = st.multiselect("選擇要顯示的情緒", options, default=default_emotions)

                if selected:
                    fig = px.line(emotion_df, x="date", y=selected, title="情緒趨勢圖（過去日記）")
                    st.plotly_chart(fig)
            else:
                st.info("尚無歷史記錄")


def login_or_signup():
    st.title("🔑 Login or Create Account")

    # 切換登入或註冊
    if "signup_mode" not in st.session_state:
        st.session_state["signup_mode"] = False

    if st.session_state["signup_mode"]:
        signup_page()  # 顯示註冊頁面
        return

    # 登入頁面
    if st.session_state.get("logged_in", False):
        st.success(f"Welcome back, {st.session_state['username']}!")
        return

    username = st.text_input("Username", value=st.session_state.get("username", ""))
    password = st.text_input("Password", type="password")
    
    col1, col2, col3 = st.columns([1, 3, 1])  # 左中右三欄

    with col3:
        login = st.button("Login", use_container_width=True)

    with col1:
        if st.button("Create Account"):
            st.session_state["signup_mode"] = True  # 切換到註冊頁面
            st.rerun()  # 重新載入頁面

    if login:
        if username in db["users"] and db["users"][username]["password"] == password:
            st.success(f"Welcome, {username}!")
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

def signup_page():
    st.write("### 📝 Create an Account")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col3:
        if st.button("Submit", use_container_width=True):
            if username in db["users"]:
                st.error("Username already exists. Try another one.")
            else:
                # 註冊並存入使用者角色
                db["users"][username] = {"password": password}
                save_db()
                st.success("Account created successfully! Redirecting to login...")

                # 回到登入畫面
                st.session_state["signup_mode"] = False
                st.rerun()

    with col1:
        if st.button("↩️"):
            st.session_state["signup_mode"] = False  # 切換回登入模式
            st.rerun()



if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.session_state.get("logged_in", False):
        main()
    else:
        login_or_signup()