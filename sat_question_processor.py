import streamlit as st
import os
import json
import time
import re
import requests
import PyPDF2
import pandas as pd
import io

# API configuration
API_URL = "https://api.anthropic.com/v1/messages"

def call_claude_api(prompt, api_key):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 8192,
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['content'][0]['text']
    else:
        st.error(f"API call failed with status code: {response.status_code}")
        st.error(f"Response: {response.text}")
        return None

def extract_text_from_pdf(pdf_file, start_page, end_page):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(start_page, min(end_page, len(reader.pages))):
        text += reader.pages[page_num].extract_text()
    return text

def process_pdf_chunk(chunk_text, api_key):
    prompt = f"""
    Analyze the following text extracted from an SAT question paper and format it into a structured JSON output. Extract the following details for each question:

    - Question text including the complete passage, table (in markdown), etc.
    - Options (A, B, C, D, and sometimes E)
    - Correct answer (the letter of the correct option)
    - Rationale (a rationale of why the correct answer is right which is provided)
    - Skill (if available, otherwise use "Not specified")
    - Domain (if available, otherwise use "Not specified")
    - Difficulty (if available, otherwise use "Not specified")

    Format the extracted information into a JSON structure as follows:

    {{
      "questions": [
        {{
          "question_text": "What is the capital of France?",
          "options": [
            {{"label": "A", "text": "London"}},
            {{"label": "B", "text": "Paris"}},
            {{"label": "C", "text": "Berlin"}},
            {{"label": "D", "text": "Madrid"}}
          ],
          "correct_answer": "B",
          "rationale": "Paris is the capital and largest city of France.",
          "skill": "Geography",
          "domain": "Social Studies",
          "difficulty": "Easy"
        }}
      ]
    }}

    Here's the text to process:

    {chunk_text}

    Important instructions:
    - Your response should contain ONLY the JSON output, nothing else.
    - Process ALL questions in the input text.
    - Ensure the JSON is properly formatted and can be parsed by a JSON parser.
    - Use double quotes for all strings in the JSON.
    - Escape any special characters in the text that might break the JSON structure.
    - If a question doesn't have a clear skill, domain, or difficulty, use "Not specified" as the value.
    - Do not include any commentary or explanations outside the JSON structure.
    """

    response = call_claude_api(prompt, api_key)
    
    if response:
        json_match = re.search(r'(\{|\[).*(\}|\])', response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group().strip()
                parsed_json = json.loads(json_str)
                if "questions" in parsed_json and isinstance(parsed_json["questions"], list):
                    return parsed_json
                else:
                    st.error(f"Invalid JSON structure. Expected 'questions' key with list value.")
                    return None
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON in response: {e}")
                st.error(f"JSON string: {json_str}")
                return None
        else:
            st.error(f"No valid JSON found in the response.")
            st.error(f"Response: {response}")
            return None
    else:
        st.error(f"Failed to process input text.")
        return None

def main():
    st.title("SAT Question Processor")

    api_key = st.text_input("Enter your Claude API key:", type="password")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None and api_key:
        if st.button("Process PDF"):
            reader = PyPDF2.PdfReader(uploaded_file)
            total_pages = len(reader.pages)

            progress_bar = st.progress(0)
            status_text = st.empty()

            all_questions = []

            for start_page in range(0, total_pages, 8):
                end_page = min(start_page + 8, total_pages)
                chunk_text = extract_text_from_pdf(uploaded_file, start_page, end_page)
                
                processed_data = process_pdf_chunk(chunk_text, api_key)
                time.sleep(10)
                
                if processed_data and 'questions' in processed_data:
                    all_questions.extend(processed_data['questions'])
                    status_text.text(f"Processed pages {start_page+1}-{end_page}")
                else:
                    status_text.text(f"No valid data found for pages {start_page+1}-{end_page}")
                
                progress = (end_page / total_pages)
                progress_bar.progress(progress)
                
                time.sleep(2)

            if all_questions:
                df = pd.DataFrame(all_questions)
                df['options'] = df['options'].apply(lambda x: '; '.join([f"{opt['label']}: {opt['text']}" for opt in x]))
                
                csv = df.to_csv(index=False)
                csv_bytes = csv.encode()
                
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="sat_questions.csv",
                    mime="text/csv"
                )
                st.success("Processing complete! You can now download the CSV file.")
            else:
                st.error("No questions were processed successfully.")

if __name__ == "__main__":
    main()