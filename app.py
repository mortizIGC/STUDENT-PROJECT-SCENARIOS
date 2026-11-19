import io
import json
import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-3.7-flash"

FIELDS = [
    "Document_Type",
    "Document_Number",
    "Date",
    "Issuer",
    "Recipient",
    "Total_Amount",
    "Currency",
    "Summary",
]

PROMPT = """You will receive one or more business documents (invoices, receipts or certificates).
Each document comes after a line "File: <file name>".
Return ONLY a JSON object like {"documents": [...]} with one object per document and these keys:
- "File_Name": the file name exactly as given
- "Document_Type": invoice, receipt, certificate or other
- "Document_Number": the invoice/certificate number
- "Date": the document date in YYYY-MM-DD format
- "Issuer": the company or institution that issued the document
- "Recipient": the customer or person the document is for
- "Total_Amount": the final total as a number, without currency symbols
- "Currency": the currency code, for example USD, EUR, INR, COP
- "Summary": a short summary of the document in 2-3 sentences
If a value is not in the document, use an empty string. Do not invent values."""


def get_api_key():
    # Streamlit Cloud uses st.secrets; on your computer you can also use an environment variable
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY", "")


def read_documents(files):
    # all documents are sent to Gemini in ONE request
    contents = []
    for file in files:
        contents.append(f"File: {file.name}")
        contents.append(types.Part.from_bytes(data=file.getvalue(), mime_type=file.type))
    contents.append(PROMPT)

    # show in the terminal what is sent to the API (file bytes are not printed)
    print("\n========== REQUEST TO GEMINI API ==========")
    print("Model:", MODEL)
    for file in files:
        print(f"File: {file.name} | type: {file.type} | size: {len(file.getvalue()) / 1024:.1f} KB")
    print("Prompt:")
    print(PROMPT)
    print("===========================================\n")

    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            # reading documents does not need much reasoning, so the answer is faster
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    data = json.loads(response.text)

    # save the last API response (it is replaced every time)
    with open("last_response.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Response saved to last_response.json")

    return data["documents"]


def read_with_retry(files, attempts=3):
    for attempt in range(attempts):
        try:
            return read_documents(files)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == attempts - 1:
                raise
            time.sleep(3)   # wait a little before trying again


def to_excel(df):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()


# ---------------- Page ----------------
st.set_page_config(page_title="AI Document Understanding", layout="wide")
st.title("AI Document Understanding System")
st.write("Upload invoices, receipts or certificates. The AI reads them, extracts the key fields "
         "and summarizes them. Check the values before adding them to the records.")

if "pending" not in st.session_state:
    st.session_state.pending = {}   # documents extracted but not verified yet
if "records" not in st.session_state:
    st.session_state.records = []   # verified documents
if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0

if not get_api_key():
    st.error("GEMINI_API_KEY is missing. Add it to .streamlit/secrets.toml (see README).")
    st.stop()

# ---------------- Task 1: upload ----------------
uploaded_files = st.file_uploader("Upload one or more documents", type=["png", "jpg", "jpeg", "pdf"],
                                  accept_multiple_files=True)

# ---------------- Tasks 2-4: read, extract and summarize ----------------
if uploaded_files and st.button("Extract information", type="primary"):
    saved_files = [r["File_Name"] for r in st.session_state.records]
    to_process = [f for f in uploaded_files
                  if f.name not in st.session_state.pending and f.name not in saved_files]

    if not to_process:
        st.info("All uploaded documents have already been processed.")
    else:
        with st.spinner("Reading documents..."):
            try:
                results = read_with_retry(to_process)
                for file, result in zip(to_process, results):
                    name = result.get("File_Name") or file.name
                    if name not in [f.name for f in to_process]:
                        name = file.name
                    row = {"Verified": False, "File_Name": name}
                    row.update({field: str(result.get(field, "")) for field in FIELDS})
                    st.session_state.pending[name] = row
            except Exception as e:
                st.error(f"The documents could not be read: {e}. Click Extract information to try again.")
        st.session_state.editor_version += 1

# ---------------- Task 6: verify ----------------
st.divider()
st.subheader("Extracted documents")

if st.session_state.pending:
    st.caption("AI can make mistakes. Check every value, correct it in the table if needed, "
               "tick Verified and click Add verified to records.")

    pending_df = pd.DataFrame(list(st.session_state.pending.values()))
    edited_df = st.data_editor(
        pending_df,
        disabled=["File_Name"],
        hide_index=True,
        width="stretch",
        column_config={"Verified": st.column_config.CheckboxColumn("Verified")},
        key=f"editor_{st.session_state.editor_version}",
    )

    if st.button("Add verified to records"):
        verified_rows = edited_df[edited_df["Verified"]]
        if verified_rows.empty:
            st.warning("Tick Verified in at least one row first.")
        else:
            for _, row in edited_df.iterrows():
                row = row.to_dict()
                name = row["File_Name"]
                if row["Verified"]:
                    row.pop("Verified")
                    row["Verified_At"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.records.append(row)
                    del st.session_state.pending[name]
                else:
                    st.session_state.pending[name] = row   # keep the user's corrections
            st.session_state.editor_version += 1
            st.rerun()
else:
    st.write("No documents waiting for verification.")

# ---------------- Task 5: export ----------------
st.divider()
st.subheader("Verified records")

if st.session_state.records:
    records_df = pd.DataFrame(st.session_state.records)
    st.dataframe(records_df, hide_index=True, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.download_button("Download CSV", records_df.to_csv(index=False),
                         "extracted_documents.csv", "text/csv")
    col2.download_button("Download Excel", to_excel(records_df), "extracted_documents.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if col3.button("Clear records"):
        st.session_state.records = []
        st.rerun()
else:
    st.write("No verified records yet.")