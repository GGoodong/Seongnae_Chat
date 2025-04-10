import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# 🔑 OpenAI 최신 방식 API Key 등록
client = openai.OpenAI(api_key="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")

# 📄 예시 데이터프레임
example_df = pd.DataFrame({
    "제목": [
        "인원수",
        "지점장",
        "성내지점",
        "운영팀",
        "동부지역본부"
    ],
    "내용": [
        "성내지점은 26명의 인원으로 구성되어있습니다.",
        "지점장님의 성함은 송승영입니다.",
        "성내지점은 대한민국에서 제일 큰 지점으로 24년 하반기 최우수 지점으로 선정 되었습니다.",
        "성내지점 운영팀에는 김태준 과장님, 고영미 팀장님, 채화정 매니저님, 이기순 매니저님이 계십니다.",
        "성내지점은 동부지역본부 소속입니다."
    ]
})
example_df["전체"] = example_df["제목"] + " " + example_df["내용"]

# 🖥️ Streamlit UI
st.set_page_config(page_title="Seongnae Chat", page_icon="🚙")
st.title("Seongnae Chat")

uploaded_file = st.file_uploader("📂 파일 업로드(Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Success")
else:
    df = example_df
    st.warning("⚠️ 파일 미업로드 시, 성내지점 소개 Chatbot의 역할을 합니다.")

# 전체 텍스트 결합: 제목 + 내용
if "제목" in df.columns and "내용" in df.columns:
    df["전체"] = df["제목"].fillna("") + " " + df["내용"].fillna("")
    col_options = ["제목", "내용", "전체"]
else:
    col_options = df.columns.tolist()

col = st.selectbox("📝 카테고리", col_options)
texts = df[col].fillna("").astype(str).tolist()

# 💬 질문 입력
query = st.text_input("💬 질문을 입력하세요:")

if query:
    with st.spinner("🔎 Excel 데이터에서 검색 중..."):
        model = SentenceTransformer("BM-K/KoSimCSE-roberta")
        doc_embeddings = model.encode(texts, convert_to_numpy=True)
        query_embedding = model.encode([query], convert_to_numpy=True)

        # 코사인 유사도 계산
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_indices = similarities.argsort()[-3:][::-1]
        top_docs = [texts[i] for i in top_indices]

    # 📚 GPT 프롬프트 구성
    context = "\n\n".join(top_docs)
    prompt = f"""다음 문서를 참고하여 질문에 답해주세요.\n\n{context}\n\nQ: {query}\nA:"""

    with st.spinner("💡 답변 생성중..."):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = response.choices[0].message.content

    st.subheader("🛞 Answer")
    st.write(answer)
