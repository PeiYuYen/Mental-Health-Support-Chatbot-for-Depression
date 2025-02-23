import time
import ollama
import json
import streamlit as st

import logging
import structlog
from pathlib import Path


def style_page():
  st.html("""
  <style>
  hr {
      margin: -0.5em 0 0 0;
      background-color: red;
  }
  p.prompt {
      margin: 0;
      font-size: 14px;
  }

  img.spinner {
      margin: 0 0 0 0;
  }

  div.block-container {
    padding-top: 2rem;
  }

  ul[data-testid="stSidebarNavItems"] {
    padding-top: 3.5rem;
  }
  </style>
  """)


def clear_everything():
  st.session_state.messages1 = []
  st.session_state.messages2 = []
  del st.session_state.selected_models
  if "vote" in st.session_state:
      del st.session_state.vote

def meta_formatting(color, prefix, model_name):
    return f"## :{color}[{prefix}: {model_name}]"


def create_logger(name, log_file_path):
    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers if any (important for avoiding duplicates)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create and add a file handler
    log_file = Path(log_file_path)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    
    # Ensure no propagation to the root logger to avoid duplicate logs
    logger.propagate = False

    # Wrap the logger with structlog
    struct_logger = structlog.wrap_logger(
        logger,
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )
    return struct_logger

def all_chat_models():
    # 從 ollama.list() 中提取模型名稱
    all_models = ollama.list()
    if "models" in all_models:
        models = [(model["name"], model["size"]) for model in all_models["models"]]
        # print("Available models:", models)
        return models
    else:
        print("Error: 'models' key not found in the response.")
        models = []
    # data = ollama.list()  # 假設返回類似於 {"models": [...]}
    # print(data)  # 調試：檢查 `ollama.list()` 返回的完整結構
    # return [
    #     (m['name'], m['details']['parameter_size'])
    #     for m in ollama.list()["models"]  
    #     if m["details"]["family"] in ["llama", "gemma"]
    #     and "clip" not in (m["details"]["families"] or [])
    #     and m["details"]["parameter_size"] in ['1B', '3B', '4B', '7B', '8B', '9B']
    # ]
