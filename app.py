import streamlit as st
import pandas as pd
import re
from bertopic import BERTopic
import plotly.express as px

st.title("Themenanalyse mit BERTopic")

uploaded_file = st.file_uploader("Wähle eine CSV-Datei", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, on_bad_lines='warn')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    st.write("Datenvorschau:")
    st.dataframe(df.head())

    def clean_text(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s\-]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return text

    df['clean_title'] = df['title'].apply(clean_text)

    st.write("Bereinigte Titel-Vorschau:")
    st.dataframe(df[['title', 'clean_title']].head())

    with st.spinner("Trainiere BERTopic-Modell..."):
        topic_model = BERTopic(verbose=True)
        topics, probs = topic_model.fit_transform(df['clean_title'])
        df['topic'] = topics
        topic_info = topic_model.get_topic_info()

    st.success("Modelltraining abgeschlossen!")
    st.write("Themenübersicht:")
    st.dataframe(topic_info.head())

    topic_names = {row['Topic']: row['Name'] for _, row in topic_info.iterrows()}
    df['topic_name'] = df['topic'].map(topic_names)

    st.write("Beispielhafte Zuordnung von Topics:")
    st.dataframe(df[['topic', 'topic_name']].head())

    df_topic_counts = df.groupby([pd.Grouper(key="timestamp", freq="D"), "topic_name"]) \
                        .size().reset_index(name="count")

    def natural_sort_key(s):
        s = str(s).strip()
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    topics_sorted = sorted(df_topic_counts["topic_name"].unique(), key=natural_sort_key)

    fig = px.line(
        df_topic_counts,
        x="timestamp",
        y="count",
        color="topic_name",
        markers=True,
        title="Topics over time",
        labels={"timestamp": "Date", "count": "Count", "topic_name": "Topic"},
        category_orders={'topic_name': topics_sorted}
    )

    st.plotly_chart(fig, use_container_width=True)
