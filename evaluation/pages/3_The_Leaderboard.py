import streamlit as st
import ollama
import jsonlines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.stylable_container import stylable_container
from trueskill import Rating, rate_1vs1
from utils import all_chat_models, style_page

title = "🏆 The Leaderboard"
st.set_page_config(page_title=title, layout="wide")
style_page()
st.title(title)

# st.write("The models are ranked using Microsoft's TrueSkill algorithm.")

models = all_chat_models()
models_with_ratings = {
  name: {
    "size": f"{size / 1e9:.1f}GB",
    "rating": Rating(), 
    "comparisons": 0,
    "wins": 0  # 新增 wins 欄位
  } 
  for name, size in models
}

with jsonlines.open('logs/voting.log') as reader:
    for row in reader:
      if row["model1"] in models_with_ratings and row["model2"] in models_with_ratings:
        m1 = models_with_ratings[row["model1"]]["rating"]
        m2 = models_with_ratings[row["model2"]]["rating"]

        if row["choice"] == "same":
          m1, m2 = rate_1vs1(m1, m2, drawn=True)
        if row["choice"] == "model1":
          m1, m2 = rate_1vs1(m1, m2)
          models_with_ratings[row["model1"]]["wins"] += 1  # model1 獲勝
        if row["choice"] == "model2":
          m2, m1 = rate_1vs1(m2, m1)
          models_with_ratings[row["model2"]]["wins"] += 1
        
        models_with_ratings[row["model1"]]["rating"] = m1
        models_with_ratings[row["model1"]]["comparisons"] += 1

        models_with_ratings[row["model2"]]["rating"] = m2
        models_with_ratings[row["model2"]]["comparisons"] += 1

df = pd.DataFrame({
    "Name": models_with_ratings.keys(),
    "Size": [v['size'] for k,v in models_with_ratings.items()],
    "Rating": [v['rating'].mu for k,v in models_with_ratings.items()],
    "Certainty": [v['rating'].sigma for k,v in models_with_ratings.items()],
    "Comparisons": [v['comparisons'] for k,v in models_with_ratings.items()],
    "Wins": [v['wins'] for k,v in models_with_ratings.items()]  # 新增 wins 欄位
  })

df["Win Rate"] = df["Wins"] / df["Comparisons"]

st.dataframe(
  df.sort_values(by = ["Rating"], ascending=False), hide_index=True
)
# 根據 Rating 由高到低排序原始 DataFrame
df_sorted = df.sort_values(by=["Rating"], ascending=False)

# 模擬分布數據，根據排序後的 Rating 和 Certainty 生成多組樣本
simulated_data = []
for _, row in df_sorted.iterrows():  # 使用排序後的 df_sorted
    name = row["Name"]
    mean = row["Rating"]
    std_dev = row["Certainty"]
    
    # 根據正態分布生成數據
    samples = np.random.normal(loc=mean, scale=std_dev, size=1000)
    for sample in samples:
        simulated_data.append({"Name": name, "Value": sample})

simulated_df = pd.DataFrame(simulated_data)

# 繪製箱型圖，模型名稱（Name）放在 x 軸，Rating 值放在 y 軸，按 Rating 排序
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=simulated_df,
    x="Name",  # x 軸改為模型名稱
    y="Value",  # y 軸顯示 Rating 值
    order=df_sorted["Name"],  # 按排序後的 Name 順序繪圖
    orient="v",  # 垂直方向
    palette="Set2"
)

# 設定圖表標題和標籤
plt.title("Confidence Intervals on Model Strength", fontsize=14)
plt.xlabel("Model Name")
plt.ylabel("Rating")
# 旋轉 x 軸標籤，避免重疊
plt.xticks(rotation=45, ha="right")  # 旋轉 45 度，並將標籤對齊右側

# 顯示圖表
st.pyplot(plt.gcf())  # 使用 gcf() 確保當前圖形對象正確
plt.clf()  # 清除當前圖形


# 平均勝率長條圖
df_sorted_by_win_rate = df.sort_values(by="Win Rate", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_sorted_by_win_rate,
    x="Name",
    y="Win Rate",
    palette="viridis"
)
plt.title("Average Win Rate of Models")
plt.xlabel("Model Name")
plt.ylabel("Win Rate")
# 旋轉 x 軸標籤，避免重疊
plt.xticks(rotation=45, ha="right")  # 旋轉 45 度，並將標籤對齊右側
plt.ylim(0, 1)
st.pyplot(plt)


# 模型對戰次數圖
from collections import Counter

battle_counts = Counter()
with jsonlines.open('logs/voting.log') as reader:
    for row in reader:
        pair = tuple(sorted([row["model1"], row["model2"]]))  # 確保模型對排序一致
        battle_counts[pair] += 1

# 使用與勝率熱圖一致的模型順序
models = list(models_with_ratings.keys())

# 創建對戰次數矩陣，並按一致的模型順序初始化
battle_matrix = pd.DataFrame(0, index=models, columns=models)

# 填充對戰次數
for (model1, model2), count in battle_counts.items():
    battle_matrix.at[model1, model2] = count
    battle_matrix.at[model2, model1] = count  # 對稱矩陣

# 繪製對戰次數熱圖
plt.figure(figsize=(12, 10))
sns.heatmap(
    battle_matrix,
    annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5,
    xticklabels=models, yticklabels=models
)
plt.title("Battle Count for Each Combination of Models", fontsize=14)
plt.xlabel("Model B")
plt.ylabel("Model A")
# plt.xticks(rotation=45, ha="right")  # 旋轉 45 度，並將標籤對齊右側
st.pyplot(plt.gcf())
plt.clf()  # 清除當前圖形

# 模型勝率熱圖（保持不變）
win_fraction_matrix = pd.DataFrame(
    data=0, index=models, columns=models, dtype=float
)
battle_counts_matrix = pd.DataFrame(
    data=0, index=models, columns=models, dtype=int
)

# 填充對戰矩陣
with jsonlines.open('logs/voting.log') as reader:
    for row in reader:
        if row["choice"] == "model1":
            win_fraction_matrix.at[row["model1"], row["model2"]] += 1
        elif row["choice"] == "model2":
            win_fraction_matrix.at[row["model2"], row["model1"]] += 1
        battle_counts_matrix.at[row["model1"], row["model2"]] += 1
        battle_counts_matrix.at[row["model2"], row["model1"]] += 1

# 計算勝率矩陣
win_fraction_matrix = win_fraction_matrix.div(
    battle_counts_matrix.replace(0, np.nan)  # 避免零除
).fillna(0)

plt.figure(figsize=(12, 10))
sns.heatmap(
    win_fraction_matrix,
    annot=True, fmt=".2f",
    cmap="RdYlGn",
    xticklabels=models, yticklabels=models,
    linewidths=0.5, linecolor='gray'
)
plt.title("Fraction of Model A Wins (Non-tied A vs. B Battles)")
plt.xlabel("Model B")
plt.ylabel("Model A")
st.pyplot(plt)
