import streamlit as st
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# กำหนดชื่อโมเดล
model_name = "tinkerbell7997/test_model_attention_mask_V2"

# ใช้ st.spinner เพื่อแสดงสถานะระหว่างการโหลดโมเดล
with st.spinner("กำลังโหลดโมเดล..."):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        device_map="auto", 
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=False, 
        trust_remote_code=True
    )

st.success("โหลดโมเดลและ tokenizer สำเร็จ!")

# กำหนดอุปกรณ์ (CPU หรือ GPU) เผื่อใช้กับ input_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Precompile the regex pattern for symbol detection
symbol_pattern = re.compile(r'\b[A-Z]{2,}\b')

def extract_symbols(text):
    return set(symbol_pattern.findall(text))

def generate_text(input_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)

    # Extract SYMBOLs using the precompiled regex pattern
    original_symbols = extract_symbols(input_text)
    num_symbols = len(original_symbols)

    # Adjust the number of return sequences based on symbol count
    num_return_sequences = 5 if num_symbols == 1 else max(num_symbols * 3, 3)
    num_beams = max(5, num_return_sequences)

    # Generate outputs
    outputs = model.generate(input_ids, num_return_sequences=num_return_sequences, do_sample=True, num_beams=num_beams, max_length=50)
    output_texts = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]

    return output_texts

# Streamlit Interface
st.title("Aspect-Based Sentiment Analysis (ABSA) Web Interface")
st.write("กรุณากรอกข้อความข่าวหุ้นเพื่อให้โมเดลทำการวิเคราะห์")

user_input = st.text_area("ข้อความข่าวหุ้น:", "")

if st.button("วิเคราะห์"):
    if user_input:
        with st.spinner("กำลังวิเคราะห์..."):
            results = generate_text(user_input)

        st.success("ผลลัพธ์:")
        for i, result in enumerate(results):
            parts = result.split("<")
            extracted_info = {
                "SYMBOL": None,
                "ASPECT": None,
                "OPINION": None,
                "SENTIMENT": None,
                "SENTIMENT_COLOR": None
            }

            for part in parts:
                if "SYMBOL>" in part:
                    extracted_info["SYMBOL"] = part.replace("SYMBOL> ", "").strip()
                elif "ASPECT>" in part:
                    extracted_info["ASPECT"] = part.replace("ASPECT> ", "").strip()
                elif "OPINION>" in part:
                    extracted_info["OPINION"] = part.replace("OPINION> ", "").strip()
                elif "POS>" in part:
                    extracted_info["SENTIMENT"] = "เชิงบวก (Positive)"
                    extracted_info["SENTIMENT_COLOR"] = "lightgreen"
                elif "NEG>" in part:
                    extracted_info["SENTIMENT"] = "เชิงลบ (Negative)"
                    extracted_info["SENTIMENT_COLOR"] = "tomato"
                elif "NEU>" in part:
                    extracted_info["SENTIMENT"] = "เชิงกลาง (Neutral)"
                    extracted_info["SENTIMENT_COLOR"] = "#ffeb99"

            if all(extracted_info.values()):
                st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; border: 1px solid #ddd; margin-bottom: 10px; display: flex; align-items: center;">
                        <span style="margin-right: 10px;"><b>SYMBOL:</b> <span style="background-color: #cce5ff; padding: 5px; border-radius: 5px;">{extracted_info['SYMBOL']}</span></span>
                        <span style="margin-right: 10px;"><b>ASPECT:</b> <span style="background-color: #e6ccff; padding: 5px; border-radius: 5px;">{extracted_info['ASPECT']}</span></span>
                        <span style="margin-right: 10px;"><b>OPINION:</b> <span style="background-color: #ffccf2; padding: 5px; border-radius: 5px;">{extracted_info['OPINION']}</span></span>
                        <span><b>Sentiment:</b> <span style="background-color: {extracted_info['SENTIMENT_COLOR']}; padding: 5px; border-radius: 5px;">{extracted_info['SENTIMENT']}</span></span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("ไม่พบข้อมูลในผลลัพธ์")
