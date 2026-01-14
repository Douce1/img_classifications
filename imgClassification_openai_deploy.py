# UI 추가하기
# - 0) streamlit 라이브러리 추가하기
# - 1) model 선택하기 : st.sidebar / st.selectbox
# - 2) prompt 작성하기 : st.text_area
# - 3) 이미지 업로드 하기 : st.file_uploader
# - 4) 업로드한 이미지 보여주기 : st.image
# - 5) 분류 실행하기 : st.button / st.spinner
# - 6) 결과 출력하기 : st.write / st.code


# 1. 라이브러리 가져오고 api key를 환경 변수에서 가져오기
import os
import base64
from io import BytesIO

from PIL import Image
# from dotenv import load_dotenv
from openai import OpenAI
# - 0) streamlit 라이브러리 추가하기
import streamlit as st

# load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# client는 보통 전역 1회 생성 권장
client = OpenAI(api_key=api_key)

# 2. 이미지를 문자열로 인코딩하는 함수 정의하기

# 이미지를 base64 문자열로 인코딩하는 함수
def encode_image(img: Image.Image, max_side: int = 512) -> str:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
# 3. 모델이 이미지 분류 요청 함수 정의하기
# GPT 모델에게 이미지와 프롬프트를 보내 결과를 받아오는 함수
def classify_image(prompt: str, img: Image.Image, model: str = "gpt-4o") -> str:
    b64 = encode_image(img)
    data_uri = f"data:image/jpeg;base64,{b64}"

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_uri},
                ],
            }
        ],
        temperature=0,
    )

    return resp.output_text
# 4. 프롬프트 선언하고 이미지 분류 실행하기
# GPT에게 보낼 프롬프트 정의
prompt = """
영상을 보고 다음 보기 내용이 포함되면 1, 포함되지 않으면 0으로 분류해줘.
보기 = [건축물, 바다, 산]
JSON format으로 키는 'building', 'sea', 'mountain'으로 하고 각각 건축물, 바다, 산에 대응되도록 출력해줘.
자연 이외의 건축물이 조금이라도 존재하면 'building'을 1로, 물이 조금이라도 존재하면 'sea'을 1로, 산이 조금이라도 보이면 'mountain'을 1로 설정해줘.
markdown format은 포함하지 말아줘.
"""




import streamlit as st

st.set_page_config(
    page_title="Image Classification - OpenAI",
    page_icon="🖥",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("이미지 분류기 - OpenAI")
# - 1) model 선택하기 : st.sidebar / st.selectbox

with st.sidebar:
    model = st.selectbox("모델 선택",
                         options = ["gpt-4o", "gpt-4o-mini"],
                         index = 0)
# - 2) prompt 작성하기 : st.text_area
prompt = """
영상을 보고 다음 보기 내용이 포함되면 1, 포함되지 않으면 0으로 분류해줘.
보기 = [건축물, 바다, 산]
JSON format으로 키는 'building', 'sea', 'mountain'으로 하고 각각 건축물, 바다, 산에 대응되도록 출력해줘.
자연 이외의 건축물이 조금이라도 존재하면 'building'을 1로, 물이 조금이라도 존재하면 'sea'을 1로, 산이 조금이라도 보이면 'mountain'을 1로 설정해줘.
markdown format은 포함하지 말아줘.
"""
st.text_area("프롬프트 입력", value = prompt, height = 200)
# - 3) 이미지 업로드 하기 : st.file_uploader
uploaded_file = st.file_uploader ("이미지 업로드", type = ["jpe", "jpeg", "png"])
# - 4) 업로드한 이미지 보여주기 : st.image
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption = "업로드한 이미지", width = "stretch")
# - 5) 분류 실행하기 : st.button / st.spinner
    if st.button("분류 실행"):
        with st.spinner("분류중..."):
            response = classify_image(prompt, img, model = model)
# - 6) 결과 출력하기 : st.write / st.code
        st.subheader("분류 결과 : ")
        st.code(response) # st.write or st.code

