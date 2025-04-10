import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI  # ✅ 최신 방식

#🔑 API Key
client = OpenAI(api_key="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")

# 📄 예시 데이터 생성
example_data = {
    "제목": ["인원수", "지점장", "성내지점", "운영팀", "동부지역본부"],
    "내용": [
        "성내지점은 26명의 인원으로 구성되어있습니다.",
        "지점장님의 성함은 송승영입니다.",
        "성내지점은 대한민국에서 제일 큰 지점으로 24년 하반기 최우수 지점으로 선정 되었습니다.",
        "성내지점 운영팀에는 김태준 과장님, 고영미 팀장님, 채화정 매니저님, 이기순 매니저님이 계십니다.",
        "성내지점은 동부지역본부 소속입니다."
    ]
}
example_df = pd.DataFrame(example_data)
example_path = "example_docs.xlsx"
example_df.to_excel(example_path, index=False)

# 🖥️ Streamlit UI
st.set_page_config(page_title="Seongnae Chat", page_icon="🚙")
st.title("Seongnae Chat")

uploaded_file = st.file_uploader("📂 파일 업로드(Excel)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ Success")
else:
    df = pd.read_excel(example_path)
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
