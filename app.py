import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from googleapiclient.discovery import build
import re
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("YT_TOKEN")

youtube = build("youtube", "v3", developerKey=API_KEY)

def truncate_description(desc, max_words=100):
    if pd.isna(desc):
        return ""
    words = str(desc).split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + "..."
    return desc

def extract_youtube_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_youtube_video_data( video_url):
    
    request = youtube.videos().list(
        part="snippet",
        id=extract_youtube_id(video_url)
    )
    response = request.execute()
    
    if "items" in response and len(response["items"]) > 0:
        return {
            "description" : response["items"][0]["snippet"]["description"],
            "title" : response["items"][0]["snippet"]["title"],
            "channelTitle" : response["items"][0]["snippet"]["channelTitle"],
            "tags" : response["items"][0]["snippet"]["tags"]
            }
    else:
        return None

@st.cache_resource
def load_model():
    model_path = 'YakuninVla/youtube_finetuned_gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device

def generate_comment(video_url, sentiment, temperature, model, tokenizer, device):
    video_data = get_youtube_video_data(video_url)
    prompt = f"""
    sentiment: {sentiment}
    video: {video_data['title']}
    channel: {video_data['channelTitle']}
    tags: {", ".join(video_data['tags'])}
    description: {truncate_description(video_data['description'])}
    comment:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()


    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            num_return_sequences=1
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]

def main():
    st.title("🤖 Генератор комментариев для YouTube")
    st.markdown("""
    Эта нейросеть генерирует реалистичные комментарии для видео на основе:
    - Вашей ссылки
    - Выбранной тональности (позитивная либо негативная)
    - Температуры
    """)
    
    try:
        model, tokenizer, device = load_model()
        st.success("Модель успешно загружена!")
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        video_url = st.text_input("🔗 Ссылка на видео", 
                                placeholder="https://www.youtube.com/watch?v=...")
        
    with col2:
        sentiment = st.selectbox("🎭 Тональность комментария", 
                               ["POSITIVE", "NEGATIVE"],
                               index=0)
    
    temperature = st.slider("🌡️ температура", 
                          min_value=0.1, 
                          max_value=1.5, 
                          value=0.7, 
                          step=0.1,
                          help="Чем выше значение, тем более креативным будет комментарий")

    word_count = st.slider("число слов", 
                          min_value=10, 
                          max_value=200, 
                          value=50, 
                          step=10,
                          help="число слов в комментарии")
    
    if st.button("✨ Сгенерировать комментарий"):
        if not video_url:
            st.warning("Пожалуйста, введите ссылку на видео")
            return
        
        with st.spinner("Генерирую комментарий..."):
            try:
                comment = generate_comment(video_url, sentiment, temperature, model, tokenizer, device)
                comment = comment.split(' ')[:word_count]
                comment = " ".join(comment).replace('\n', ' ')

                st.subheader("📝 Результат:")
                st.markdown(f"""
                <div style="
                    background: #f0f2f6;
                    padding: 15px;
                    border-radius: 10px;
                    margin-top: 10px;
                ">
                {comment}
                </div>
                """, unsafe_allow_html=True)
                                
            except Exception as e:
                st.error(f"Ошибка генерации: {str(e)}")

if __name__ == "__main__":
    main()