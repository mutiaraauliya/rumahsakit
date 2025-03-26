import os
import pandas as pd
from dotenv import load_dotenv

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from predictor import DiagnosisPredictor

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

user_avatar = "asset/user.png"
assistant_avatar = "asset/dokter.png"

df = pd.read_csv('cleaned.csv')
df_ases_soap = df['ases_soap']
df_diagnosa_dokter = df['diagnosa_dokter']

model_dir = 'saved_model/indobert'
embeddings_path = 'saved_model/dataset_embeddings.pkl'
predictor = DiagnosisPredictor(model_dir, embeddings_path, df)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Fungsi untuk membuat bubble chat
def chat_message(role, content):
    if role == "user":
        # Render pesan pengguna dengan profil di kanan
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: 10px;">
                <div style="background-color: #dcf8c6; padding: 10px 15px; border-radius: 10px; max-width: 60%; text-align: right; color: black;">
                    {content}
                </div>
                <img src="data:image/jpeg;base64,{get_image_as_base64(user_avatar)}" 
                     alt="User" style="margin-left: 10px; width: 40px; height: 40px; border-radius: 50%;">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Render pesan asisten dengan profil di kiri
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; align-items: center; margin-bottom: 10px;">
                <img src="data:image/jpeg;base64,{get_image_as_base64(assistant_avatar)}" 
                     alt="Assistant" style="margin-right: 10px; width: 40px; height: 40px; border-radius: 50%;">
                <div style="background-color: #f1f0f0; padding: 10px 15px; border-radius: 10px; max-width: 60%; color: black;">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Fungsi untuk mengonversi gambar ke format Base64
def get_image_as_base64(image_path):
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
# create the app
st.set_page_config(
    page_title="Diagnose Generator",
)
st.title("Try to Diagnose Generator")

# Initialize chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.optimized_inputs = {}  # Store optimized inputs for ases_soap and diagnosa_dokter
    st.session_state.predictions = {}  # Store predictions here
    welcome_message = "Hai! Saya akan mencoba menebak diagnosa dari penyakit Anda."
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    
    ases_soap_request = "Silahkan masukkan deskripsi untuk ases_soap."
    st.session_state.messages.append({"role": "assistant", "content": ases_soap_request})

# Render older messages
for message in st.session_state.messages:
    chat_message(message["role"], message["content"])

# User input
prompt = st.chat_input("Enter your message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_message("user", prompt)

    if "ases_soap" not in st.session_state.optimized_inputs:
        # Optimize ases_soap using Gemini AI
        command_ases = f"""                            
        [SYSTEM]
        Anda adalah seorang data engineer professional yang akan membantu prediksi dari sebuah model.
        [/SYSTEM]

        [CONVERTATION]
        'role': 'user', 'content': '{prompt}'
        [/CONVERTATION]

        [CONTEXT]
        {df_ases_soap.values}
        [/CONTEXT]

        [SUMMARY]
        Tolong perbaiki inputan dari user yaitu content agar sesuai dengan CONTEXT di atas. Anda akan mencari nilai dari CONTEXT yang sesuai dengan apa yang diinput oleh user dan berikan nilai yang sesuai berdasarkan CONTEXT.
        [/SUMMARY]
        [END] Mohon tampilkan jawabanmu saja tanpa menulis ulang apa yang diketik di atas [/END]
        """
        
        llm_ases = llm.invoke(command_ases)
        st.session_state.optimized_inputs["ases_soap"] = llm_ases.content

        diagnosa_dokter_request = "Selanjutnya masukkan deskripsi untuk diagnosa dokter."
        st.session_state.messages.append({"role": "assistant", "content": diagnosa_dokter_request})
        chat_message("assistant", diagnosa_dokter_request)
    
    elif "diagnosa_dokter" not in st.session_state.optimized_inputs:
        # Optimize diagnosa_dokter using Gemini AI
        command_diag_dokter = f"""                            
        [SYSTEM]
        Anda adalah seorang data engineer professional yang akan membantu prediksi dari sebuah model.
        [/SYSTEM]

        [CONVERTATION]
        'role': 'user', 'content': '{prompt}'
        [/CONVERTATION]

        [CONTEXT]
        {df_diagnosa_dokter.values}
        [/CONTEXT]

        [SUMMARY]
        Tolong perbaiki inputan dari user yaitu content agar sesuai dengan CONTEXT di atas. Anda akan mencari nilai dari CONTEXT yang sesuai dengan apa yang diinput oleh user dan berikan nilai yang sesuai berdasarkan CONTEXT.
        [/SUMMARY]
        [END] Mohon tampilkan jawabanmu saja tanpa menulis ulang apa yang diketik di atas [/END]
        """
        
        llm_diag_dokter = llm.invoke(command_diag_dokter)
        st.session_state.optimized_inputs["diagnosa_dokter"] = llm_diag_dokter.content

        # Use DiagnosisPredictor to predict diagnosis based on optimized inputs
        pred_primer, pred_sekunder = predictor.predict_diagnosis(
            st.session_state.optimized_inputs["ases_soap"], 
            st.session_state.optimized_inputs["diagnosa_dokter"]
        )

        # Generate a summary response with Gemini AI
        command_summary = f"""                            
        [SYSTEM]
        Anda adalah asisten virtual yang menyusun ringkasan sederhana dari data input pengguna yang telah dioptimalkan dan hasil prediksi diagnosis.
        [/SYSTEM]

        [CONVERTATION]
        Teks input yang sudah dioptimalkan adalah sebagai berikut:
        - ases_soap: "{st.session_state.optimized_inputs["ases_soap"]}"
        - diagnosa_dokter: "{st.session_state.optimized_inputs["diagnosa_dokter"]}"

        Hasil prediksi diagnosis adalah:
        - Diagnosis Primer: "{pred_primer[0]}"
        - Diagnosis Sekunder: "{pred_sekunder[0]}"
        [/CONVERTATION]

        [SUMMARY]
        Susun kesimpulan yang ringkas, mencakup kedua input yang telah dioptimalkan (ases_soap dan diagnosa_dokter), serta hasil prediksi diagnosis primer dan sekunder. Juga, tambahkan wawasan berdasarkan hasil dari ases_soap dan diagnosa_dokter yang telah dioptimalkan.
        [/SUMMARY]
        [END] Mohon tampilkan jawabanmu saja tanpa menulis ulang apa yang diketik di atas [/END]
        """

        llm_summary = llm.invoke(command_summary)
        summary_message = llm_summary.content
        summary_message = summary_message.replace('\n', ' ').strip()

        # Display the summary message
        st.session_state.messages.append({"role": "assistant", "content": summary_message})
        chat_message("assistant", summary_message)
        chat_message("assistant", "Refresh browser untuk mencoba mendiagnosis lagi.")

# Tambahkan CSS untuk mendukung mode gelap
st.markdown(
    """
    <style>
    /* Deteksi mode gelap */
    @media (prefers-color-scheme: dark) {
        div[style*="background-color: #dcf8c6"] {
            background-color: #056162 !important; /* Warna bubble untuk mode gelap */
            color: white !important; /* Ubah teks menjadi putih untuk mode gelap */
        }
        div[style*="background-color: #f1f0f0"] {
            background-color: #333333 !important; /* Warna bubble untuk mode gelap */
            color: white !important; /* Ubah teks menjadi putih untuk mode gelap */
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)