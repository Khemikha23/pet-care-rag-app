import os
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# 1. ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Pet Care Specialist & Health Guide",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Custom CSS สำหรับตกแต่ง UI
st.markdown("""
<style>
    /* สไตล์ส่วน Header ด้านบน */
    .header-container {
        background: linear-gradient(135deg, #FF9A9E 0%, #FECFEF 99%, #FECFEF 100%);
        padding: 24px;
        border-radius: 16px;
        color: #2D3748;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
        color: #2D3748;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #4A5568;
    }
    /* แต่งกล่องคำถามและคำตอบ */
    .stChatMessage {
        border-radius: 12px;
        padding: 8px;
    }
    /* แต่งปุ่ม Quick Suggestion */
    .stButton > button {
        border-radius: 20px;
        border: 1px solid #FF9A9E;
        color: #4A5568;
        background-color: #FFFFFF;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #FF9A9E;
        color: white;
        border-color: #FF9A9E;
    }
</style>
""", unsafe_allow_html=True)

# ส่วนแสดง Header หลัก
st.markdown("""
<div class="header-container">
    <div class="header-title">🐾 Pet Care RAG Specialist</div>
    <div class="header-subtitle">ผู้ช่วยอัจฉริยะตอบคำถามการดูแล สุขภาพ และอาหารสัตว์เลี้ยง อ้างอิงจากคลังเอกสารความรู้</div>
</div>
""", unsafe_allow_html=True)

# 2. ฟังก์ชันโหลดเอกสาร และสร้าง FAISS Vector DB (ทำ Caching)
@st.cache_resource(show_spinner="กำลังโหลดคลังข้อมูลสัตว์เลี้ยงและประมวลผล Vector Database...")
def init_vector_db():
    data_dir = "data"
    if not os.path.exists(data_dir):
        st.error(f"❌ ไม่พบโฟลเดอร์ `{data_dir}` กรุณาสร้างโฟลเดอร์และใส่ไฟล์ .txt ให้เรียบร้อย")
        st.stop()
        
    loader = DirectoryLoader(
        data_dir,
        glob="./*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

try:
    vector_db = init_vector_db()
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการสร้าง Vector Database: {e}")
    st.stop()

# 3. จัดการ Groq API Key
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY")
except Exception:
    groq_api_key = None

if not groq_api_key:
    groq_api_key = os.environ.get("GROQ_API_KEY")

# 4. Sidebar สำหรับตั้งค่าและดูเอกสาร
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=80)
    st.title("⚙️ การตั้งค่าระบบ")
    
    if not groq_api_key:
        st.subheader("🔑 API Key")
        groq_api_key = st.text_input("กรอก Groq API Key:", type="password", help="รับ Key ได้จาก console.groq.com")
        if not groq_api_key:
            st.warning("⚠️ กรุณากรอก Groq API Key เพื่อเริ่มใช้งานระบบ")
            st.stop()

    st.divider()
    st.subheader("📊 เครื่องมือทดสอบ")
    show_test_csv = st.checkbox("แสดงชุดคำถามทดสอบ (test_questions.csv)")
    if show_test_csv:
        if os.path.exists("test_questions.csv"):
            df_test = pd.read_csv("test_questions.csv")
            st.dataframe(df_test, use_container_width=True)
        else:
            st.info("ไม่พบไฟล์ test_questions.csv")
            
    st.divider()
    st.markdown("💡 **เกี่ยวกับระบบ:**\nระบบใช้ FAISS Vector DB ร่วมกับ Groq LLM เพื่อให้คำตอบที่ถูกต้องแม่นยำและไม่ออกนอกบริบท")

client = Groq(api_key=groq_api_key)

# 5. จัดการประวัติการสนทนา (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "สวัสดีครับ! ผมคือผู้ช่วยดูแลสัตว์เลี้ยง มีข้อสงสัยเรื่องอาหาร วัคซีน การทำความสะอาด หรือการปฐมพยาบาลสุนัข แมว แฮมสเตอร์ สอบถามได้เลยครับ 🐶🐱🐹"}
    ]

# สร้าง State สำหรับรับคำถามจากปุ่ม Quick Suggestions
if "user_query_input" not in st.session_state:
    st.session_state.user_query_input = ""

def set_query(query_text):
    st.session_state.user_query_input = query_text

# ปุ่ม Quick Suggestions สำหรับลองกดถาม
st.markdown("##### 💡 ตัวอย่างคำถามยอดฮิต:")
col1, col2, col3 = st.columns(3)

# ปรับข้อความใน args(...) ให้กระชับ สั้น และตรงคีย์เวิร์ดในไฟล์
col1.button(
    "🍫 สุนัขกินช็อกโกแลตได้ไหม?", 
    on_click=set_query, 
    args=("สุนัขกินช็อกโกแลต",)  # <-- ตัดคำว่า ได้ไหม ออก
)

col2.button(
    "🧼 วิธีทำความสะอาดหูสุนัข/แมว", 
    on_click=set_query, 
    args=("ทำความสะอาดหู สุนัข แมว",) # <-- ใช้คีย์เวิร์ดหลักตรงๆ
)

col3.button(
    "💉 ตารางฉีดวัคซีนแมว", 
    on_click=set_query, 
    args=("โปรแกรมฉีดวัคซีนแมว",)
)

# แสดงประวัติแชต
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# รับ Input จากผู้ใช้ (ทั้งจากการพิมพ์เอง หรือจากปุ่ม Quick Suggestion)
chat_input_val = st.chat_input("พิมพ์คำถามเกี่ยวกับสัตว์เลี้ยงที่นี่...")

# ตรวจสอบว่ามีคำถามมาจากช่องพิมพ์ หรือมาจากปุ่ม
active_query = None
if chat_input_val:
    active_query = chat_input_val
elif st.session_state.user_query_input:
    active_query = st.session_state.user_query_input
    st.session_state.user_query_input = "" # ล้างค่าหลังดึงมาใช้แล้ว

# 6. ประมวลผล RAG เมื่อมีคำถามเข้ามา
if active_query:
    # บันทึกคำถามผู้ใช้ลง Session
    st.session_state.messages.append({"role": "user", "content": active_query})
    st.chat_message("user").write(active_query)

    # ค้นหาเอกสารจาก FAISS (Retrieve k=5 chunks)
    retrieved_docs = vector_db.similarity_search(active_query, k=5)
    
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = list(set([os.path.basename(doc.metadata.get("source", "ไม่ระบุไฟล์")) for doc in retrieved_docs]))
    
    # Prompt Engineering
    system_prompt = f"""คุณคือ "ผู้ช่วยดูแลสัตว์เลี้ยงสำหรับมือใหม่" หน้าที่ของคุณคือตอบคำถามเกี่ยวกับสัตว์เลี้ยงโดยใช้ข้อมูลจาก Context ที่กำหนดให้เท่านั้น

กฎสำคัญในการตอบ:
1. ตอบคำถามให้ถูกต้อง ชัดเจน และเข้าใจง่าย โดยอิงจาก Context ที่ให้มาเท่านั้น
2. หากใน Context ไม่มีข้อมูลที่ตรงกับคำถาม หรือไม่สามารถสรุปคำตอบได้ ให้ตอบปฏิเสธด้วยประโยคนี้เท่านั้น:
   "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในคลังเอกสารความรู้"
3. ห้ามใช้ความรู้ภายนอก ห้ามคาดเดา หรือมโนคำตอบเองเด็ดขาด

Context ที่ค้นหาพบจากคลังความรู้:
{context_text}
"""

    # ส่งคำสั่งไปยัง Groq LLM API
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาคลังความรู้และประมวลคำตอบ..."):
            try:
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": active_query}
                    ],
                    temperature=0.1
                )
                
                answer = response.choices[0].message.content
                st.write(answer)
                
                # แสดงแหล่งที่มาหากตอบได้
                if "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในคลังเอกสารความรู้" not in answer:
                    sources_str = ", ".join(sources)
                    st.markdown(f"📌 **แหล่งที่มาข้อมูล:** `{sources_str}`")
                    
                    with st.expander("🔍 ดูเนื้อหาอ้างอิงจากคลังเอกสาร (Context)"):
                        st.caption(context_text)

                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Groq API: {e}")