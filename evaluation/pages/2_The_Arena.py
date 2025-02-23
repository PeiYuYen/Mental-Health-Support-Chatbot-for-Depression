# import sys
# import ollama
# import streamlit as st
# import asyncio
# import time
# import base64
# from openai import AsyncOpenAI
# from streamlit_extras.stylable_container import stylable_container
# from streamlit_extras.bottom_container import bottom
# import random
# from utils import style_page, clear_everything, meta_formatting, create_logger
# import uuid

# from functools import partial

# voting_logger = create_logger("voting", "logs/voting.log")
# requests_logger = create_logger("requests", "logs/requests.log")

# title = "🏟️ The Arena"
# st.set_page_config(page_title=title, layout="wide")
# style_page()
# st.title(title)

# if not "models" in st.session_state:
#     st.session_state.models = []

# if not "models" in st.session_state or len(st.session_state.models) < 2:
#     if len(st.session_state.models) == 0:
#         st.write("You haven't selected any models, so the arena won't be much use!")
#     if len(st.session_state.models) == 1:    
#         st.write("You have only selected 1 mode. Go back and select one more!")
#     if st.button("Select models"):
#         st.switch_page("pages/1_Select_Models.py")
#     st.stop()


# if not "messages1" in st.session_state:
#     st.session_state.messages1 = []

# if not "messages2" in st.session_state:
#     st.session_state.messages2 = []

# client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ignore-me")


# if not "selected_models" in st.session_state or len(st.session_state.selected_models) == 0:
#     st.session_state.selected_models = random.sample(st.session_state.models, 2)

# model_1, model_2 = st.session_state.selected_models

# col1, col2 = st.columns(2)

# meta_1 = col1.empty()
# meta_2 = col2.empty()

# meta_1.write(f"## :blue[Model 1]")
# meta_2.write(f"## :red[Model 2]")

# body_1 = col1.empty()
# body_2 = col2.empty()

# with bottom():
#     voting_buttons = st.empty()
#     prompt = st.chat_input("Message Ollama")
#     new_found = st.empty()
#     with new_found.container():
#         if len(st.session_state.messages1) > 0 or len(st.session_state.messages2) > 0:
#             with stylable_container(
#                 key="next_round_button",
#                 css_styles="""
#                     button {
#                         background-color: green;
#                         color: white;
#                         border-radius: 10px;
#                         width: 100%
#                     }
#                     """,
#             ):
#                 new_round = st.button("New Round", key="new_round", on_click=clear_everything)
            

# # Render existing state
# if "vote" in st.session_state:
#     model_1_display= model_1.replace(":", "\\:")
#     model_2_display= model_2.replace(":", "\\:")
#     meta_1.write(partial(meta_formatting, "blue", "Model 1")(model_1_display))
#     meta_2.write(partial(meta_formatting, "red", "Model 2")(model_2_display))

# if len(st.session_state.messages1) > 0 or len(st.session_state.messages2) > 0:
#     with body_1.container():
#         for message in st.session_state.messages1:
#             chat_entry = st.chat_message(name=message['role'])
#             chat_entry.write(message['content'])

#     with body_2.container():
#         for message in st.session_state.messages2:
#             chat_entry = st.chat_message(name=message['role'])
#             chat_entry.write(message['content'])

# async def run_prompt(placeholder, model, message_history):
#     with placeholder.container():
#         for message in message_history:
#             chat_entry = st.chat_message(name=message['role'])
#             chat_entry.write(message['content'])
#         assistant = st.chat_message(name="assistant")

#         with open("images/loading-gif.gif", "rb") as file:
#             contents = file.read()
#             data_url = base64.b64encode(contents).decode("utf-8")

#         assistant.html(f"<img src='data:image/gif;base64,{data_url}' class='spinner' width='25' />")
# # system prompt
#     messages = [
#         {"role": "system", "content": "你是個有同理心的專業心理諮商師，你的任務是要讓來訪者了解到你願意陪伴他渡過難關，並且透過問句引導來訪者說出心裡的想法。請你只能用2~3句話回應，你必須使用繁體中文回答。現在，請參考以下諮商師的對話範例繼續扮演心理諮商師角色回應來訪者。\n範例：\n來訪者：我覺得我沒辦法停止難過，我總是會莫名的想哭。\n諮商師：我感覺到你最近似乎真的很難過，你可以哭出來，不用壓抑自己的情緒，無論如何我都會站在你這邊。你願意說說看，最近有什麼事情讓你感到難過嗎？\n來訪者：我覺得好孤單，而且沒有人願意聆聽我的感受與想法，我覺得很受傷。\n諮商師：別擔心，我很樂意聆聽你的感受與想法，你可以盡情的分享，我會一直在這裡陪伴你，你不避感到有壓力。你可以跟我說說最近有什麼事情讓你感到孤單嗎？\n來訪者：我覺得我沒有辦法控制自己的情緒，我覺得我快要崩潰了。\n諮商師：你不必獨自面對這些情緒，你可以在這裡盡情的宣洩，我很樂意聆聽你的情緒。最近你遇到了哪些事情讓你感到情緒失控嗎？"},
#         *message_history
#     ]

#     request_id = str(uuid.uuid4())
#     requests_logger.info("Request starts", id=request_id, model=model, prompt=message_history[-1]["content"])
#     stream = await client.chat.completions.create(
#         model=model,
#         messages=messages,
#         stream=True
#     )
#     streamed_text = ""
#     async for chunk in stream:
#         chunk_content = chunk.choices[0].delta.content
#         if chunk_content is not None:
#             streamed_text = streamed_text + chunk_content
#             with placeholder.container():
#                 for message in message_history:
#                     chat_entry = st.chat_message(name=message['role'])
#                     chat_entry.write(message['content'])
#                 assistant = st.chat_message(name="assistant")
#                 assistant.write(streamed_text)    
#     requests_logger.info("Request finished", id=request_id, model=model, response=streamed_text)
                
#     message_history.append({"role": "assistant", "content": streamed_text})


# def do_vote(choice):
#     st.session_state.vote = {"choice": choice}
#     voting_logger.info("Vote", model1=model_1, model2=model_2, choice=choice)

#     model_1_display= model_1.replace(":", "\\:")
#     model_2_display= model_2.replace(":", "\\:")

#     if choice == "model1":        
#         vote_choice = f":blue[{model_1_display}]"
#     elif choice == "model2":
#         vote_choice = f":red[{model_2_display}]"
#     else:
#         vote_choice = ":grey[Both the same]"

#     st.toast(f"""##### :blue[{model_1_display}] vs :red[{model_2_display}]    
# ###### Vote cast: {vote_choice}""", icon='🗳️')

# def vote():
#     with voting_buttons.container():
#         with stylable_container(
#             key="voting_button",
#             css_styles="""
#                 button {
#                     background-color: #CCCCCC;
#                     color: black;
#                     border-radius: 10px;
#                     width: 100%;
#                 }

#                 """,
#         ):
#             col1, col2, col3 = st.columns(3)
#             model1 = col1.button("Model 1 👈", key="model1", on_click=do_vote, args=["model1"])
#             model2 = col2.button("Model 2 👉", key="model2", on_click=do_vote, args=["model2"])
#             neither = col3.button("Both the same 🤝", key="same", on_click=do_vote, args=["same"])
#     with new_found.container():
#         with stylable_container(
#             key="next_round_button",
#             css_styles="""
#                 button {
#                     background-color: green;
#                     color: white;
#                     border-radius: 10px;
#                     width: 100%
#                 }
#                 """,
#         ):
#             new_round = st.button("New Round", key="new_round_later", on_click=clear_everything)

# async def main():
#     await asyncio.gather(
#         run_prompt(body_1,  model=model_1, message_history=st.session_state.messages1),
#         run_prompt(body_2,  model=model_2, message_history=st.session_state.messages2)
#     )
#     if "vote" not in st.session_state:
#         vote()

# if prompt:
#     if prompt == "":
#         st.warning("Please enter a prompt")
#     else:        
#         st.session_state.messages1.append({"role": "user", "content": prompt})
#         st.session_state.messages2.append({"role": "user", "content": prompt})
#         asyncio.run(main())



import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import orjson


# -- 載入知識庫 --
with open("/home/pui/NYCU_course/chatbot/chatbot-arena/pages/knowledge_soul_pairedv2.json", "rb") as f:
    knowledge_base = orjson.loads(f.read())

# -- 確認 corpus 和 metadata 的長度一致 --
if len(knowledge_base['corpus']) != len(knowledge_base['metadata']):
    raise ValueError("Corpus 和 Metadata 的長度不一致！")

# -- 載入 FAISS 索引 --
index = faiss.read_index("/home/pui/NYCU_course/chatbot/chatbot-arena/pages/faiss_soul_ivfpq_pairedv2.index")

# -- 初始化同樣的 Embedding 模型 (與建索引時相同) --
retriever_model = SentenceTransformer('shibing624/text2vec-base-chinese')

def retrieve_paired_answer(query, knowledge_base, index, top_k=3):
    """根據 query 到 FAISS 檢索, 回傳最相關的 Q&A 配對列表"""
    # 生成查詢的 embedding
    query_embedding = retriever_model.encode([query], convert_to_tensor=False)
    # 確保 embedding 是 float32 並且是 2D
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    
    # 確認維度一致
    if query_embedding.shape[1] != index.d:
        raise ValueError(f"查詢 embedding 的維度 {query_embedding.shape[1]} 與 FAISS 索引的維度 {index.d} 不一致！")
    
    # 搜索 FAISS 索引
    distances, indices = index.search(query_embedding, top_k)
    
    #print("Distances:", distances)
    #print("Indices:", indices)
    
    results = []
    for idx in indices[0]:
        if idx < len(knowledge_base['corpus']):
            metadata = knowledge_base['metadata'][idx]
            results.append(metadata)
    return results

def rag_pipeline_paired(query, top_k=3):
    """RAG 流程：檢索並生成答案 (基於 Q&A 配對)"""
    # 檢索相關對話配對
    related_pairs = retrieve_paired_answer(query, knowledge_base, index, top_k=top_k)
    
    # 構建上下文
    context = ""
    for pair in related_pairs:
        user_q = pair.get('user', '')
        assistant_a = pair.get('assistant', '')
        context += f"Q: {user_q}\nA: {assistant_a}\n\n"
    
    # 將上下文與使用者問題拼接
    input_text = f"根據以下上下文內容，用中文回答問題。\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return input_text



import sys
import ollama
import streamlit as st
import asyncio
import time
import base64
from openai import AsyncOpenAI
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.bottom_container import bottom
import random
from utils import style_page, clear_everything, meta_formatting, create_logger
import uuid
from functools import partial

voting_logger = create_logger("voting", "logs/voting.log")
requests_logger = create_logger("requests", "logs/requests.log")

title = "🏟️ The Arena"
st.set_page_config(page_title=title, layout="wide")
style_page()
st.title(title)

if not "models" in st.session_state:
    st.session_state.models = []

if not "models" in st.session_state or len(st.session_state.models) < 2:
    if len(st.session_state.models) == 0:
        st.write("You haven't selected any models, so the arena won't be much use!")
    if len(st.session_state.models) == 1:    
        st.write("You have only selected 1 mode. Go back and select one more!")
    if st.button("Select models"):
        st.switch_page("pages/1_Select_Models.py")
    st.stop()


if not "messages1" in st.session_state:
    st.session_state.messages1 = []

if not "messages2" in st.session_state:
    st.session_state.messages2 = []

client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ignore-me")


if not "selected_models" in st.session_state or len(st.session_state.selected_models) == 0:
    st.session_state.selected_models = random.sample(st.session_state.models, 2)

model_1, model_2 = st.session_state.selected_models

col1, col2 = st.columns(2)

meta_1 = col1.empty()
meta_2 = col2.empty()

meta_1.write(f"## :blue[Model 1]")
meta_2.write(f"## :red[Model 2]")

body_1 = col1.empty()
body_2 = col2.empty()

with bottom():
    voting_buttons = st.empty()
    prompt = st.chat_input("Message Ollama")
    new_found = st.empty()
    with new_found.container():
        if len(st.session_state.messages1) > 0 or len(st.session_state.messages2) > 0:
            with stylable_container(
                key="next_round_button",
                css_styles="""
                    button {
                        background-color: green;
                        color: white;
                        border-radius: 10px;
                        width: 100%
                    }
                    """,
            ):
                new_round = st.button("New Round", key="new_round", on_click=clear_everything)
            

# Render existing state
if "vote" in st.session_state:
    model_1_display= model_1.replace(":", "\\:")
    model_2_display= model_2.replace(":", "\\:")
    meta_1.write(partial(meta_formatting, "blue", "Model 1")(model_1_display))
    meta_2.write(partial(meta_formatting, "red", "Model 2")(model_2_display))

if len(st.session_state.messages1) > 0 or len(st.session_state.messages2) > 0:
    with body_1.container():
        for message in st.session_state.messages1:
            chat_entry = st.chat_message(name=message['role'])
            chat_entry.write(message['content'])

    with body_2.container():
        for message in st.session_state.messages2:
            chat_entry = st.chat_message(name=message['role'])
            chat_entry.write(message['content'])

async def run_prompt(placeholder, model, message_history):
    # 1) 先複製一份給「顯示用」
    display_message_history = message_history.copy()
    user_input = message_history[-1]["content"]
    rag_context = rag_pipeline_paired(user_input)
    # 在 system message 上加 RAG context
    system_msg = {
        "role": "system", "content": f"你是個有同理心的專業心理諮商師，你的任務是要讓來訪者了解到你願意陪伴他渡過難關，並且透過問句引導來訪者說出心裡的想法。請你只能用2~3句話回應，你必須使用繁體中文回答。現在，請參考以下諮商師的對話範例繼續扮演心理諮商師角色回應來訪者。\n範例：\n來訪者：我覺得我沒辦法停止難過，我總是會莫名的想哭。\n諮商師：我感覺到你最近似乎真的很難過，你可以哭出來，不用壓抑自己的情緒，無論如何我都會站在你這邊。你願意說說看，最近有什麼事情讓你感到難過嗎？\n來訪者：我覺得好孤單，而且沒有人願意聆聽我的感受與想法，我覺得很受傷。\n諮商師：別擔心，我很樂意聆聽你的感受與想法，你可以盡情的分享，我會一直在這裡陪伴你，你不避感到有壓力。你可以跟我說說最近有什麼事情讓你感到孤單嗎？\n來訪者：我覺得我沒有辦法控制自己的情緒，我覺得我快要崩潰了。\n諮商師：你不必獨自面對這些情緒，你可以在這裡盡情的宣洩，我很樂意聆聽你的情緒。最近你遇到了哪些事情讓你感到情緒失控嗎？\n{rag_context}"
    }
    # model_messages 裝「插入 RAG 後」的內容
    model_messages = [system_msg, *message_history]
    # print(f"model_messages:{model_messages}")
    
    
    
    with placeholder.container():
        #for message in message_history:
        for message in display_message_history:
            chat_entry = st.chat_message(name=message['role'])
            chat_entry.write(message['content'])
        assistant = st.chat_message(name="assistant")

        with open("images/loading-gif.gif", "rb") as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode("utf-8")

        assistant.html(f"<img src='data:image/gif;base64,{data_url}' class='spinner' width='25' />")
    request_id = str(uuid.uuid4())
    ##有改
    requests_logger.info("Request starts", id=request_id, model=model, prompt=display_message_history[-1]["content"])
    stream = await client.chat.completions.create(
        model=model,
        messages=model_messages,
        stream=True
    )
    streamed_text = ""
    async for chunk in stream:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content is not None:
            streamed_text = streamed_text + chunk_content
            with placeholder.container():
                ##有改
                for message in display_message_history:
                    chat_entry = st.chat_message(name=message['role'])
                    chat_entry.write(message['content'])
                assistant = st.chat_message(name="assistant")
                assistant.write(streamed_text)    
    requests_logger.info("Request finished", id=request_id, model=model, response=streamed_text)
                
    #message_history.append({"role": "assistant", "content": streamed_text})
    new_assistant_msg = {"role": "assistant", "content": streamed_text}
    display_message_history.append(new_assistant_msg)
    message_history.append(new_assistant_msg)


def do_vote(choice):
    st.session_state.vote = {"choice": choice}
    voting_logger.info("Vote", model1=model_1, model2=model_2, choice=choice)

    model_1_display= model_1.replace(":", "\\:")
    model_2_display= model_2.replace(":", "\\:")

    if choice == "model1":        
        vote_choice = f":blue[{model_1_display}]"
    elif choice == "model2":
        vote_choice = f":red[{model_2_display}]"
    else:
        vote_choice = ":grey[Both the same]"

    st.toast(f"""##### :blue[{model_1_display}] vs :red[{model_2_display}]    
###### Vote cast: {vote_choice}""", icon='🗳️')

def vote():
    with voting_buttons.container():
        with stylable_container(
            key="voting_button",
            css_styles="""
                button {
                    background-color: #CCCCCC;
                    color: black;
                    border-radius: 10px;
                    width: 100%;
                }

                """,
        ):
            col1, col2, col3 = st.columns(3)
            model1 = col1.button("Model 1 👈", key="model1", on_click=do_vote, args=["model1"])
            model2 = col2.button("Model 2 👉", key="model2", on_click=do_vote, args=["model2"])
            neither = col3.button("Both the same 🤝", key="same", on_click=do_vote, args=["same"])
    with new_found.container():
        with stylable_container(
            key="next_round_button",
            css_styles="""
                button {
                    background-color: green;
                    color: white;
                    border-radius: 10px;
                    width: 100%
                }
                """,
        ):
            new_round = st.button("New Round", key="new_round_later", on_click=clear_everything)

async def main():
    #print("this is in main")
    await asyncio.gather(
        run_prompt(body_1,  model=model_1, message_history=st.session_state.messages1),
        run_prompt(body_2,  model=model_2, message_history=st.session_state.messages2)
    )
    if "vote" not in st.session_state:
        vote()

if prompt:
    if prompt == "":
        st.warning("Please enter a prompt")
    else:
        #print(f"rag_prompt in the arena {rag_prompt}")
        st.session_state.messages1.append({"role": "user", "content": prompt})
        st.session_state.messages2.append({"role": "user", "content": prompt})
        asyncio.run(main())
