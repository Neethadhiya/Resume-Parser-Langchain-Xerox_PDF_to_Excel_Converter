# 📄 AI Resume Parser Agent

> An intelligent multi-agent system that extracts structured data from hundreds of PDF resumes automatically — powered by GPT-4o-mini, Anthropic Claude, and Zerox OCR.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic_Claude-D97706?style=flat&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

---

## 🚀 What It Does

Upload one or multiple PDF resumes — the system automatically:

- 📥 Extracts text from PDFs using **Zerox OCR**
- 🔍 Detects if a single PDF contains **multiple resumes**
- 🤖 Parses each resume using **GPT-4o-mini** and **Anthropic Claude**
- 🧠 Extracts **17 structured fields** per candidate
- 📊 Outputs a clean, downloadable **CSV file**
- 🖥️ Displays results in an interactive **Streamlit dashboard**

---

## 🎯 Extracted Fields

| Field | Field | Field |
|---|---|---|
| Full Name | Email | Mobile / Phone |
| Nationality | Date of Birth | Gender |
| Work Experience | Total Experience | Skills |
| Education | Passport Info | Home Language |
| Spoken Languages | Driving License | Marital Status |
| Current Address | Home Address | — |

---

## 🏗️ Architecture

```
PDF Upload (Streamlit UI)
        ↓
Zerox OCR → Markdown extraction
        ↓
Multiple Resume Detection (GPT-4o-mini)
        ↓
    ┌───────────────┐
    │ Single Resume │  →  GPT-4o-mini Parser
    │ Multi Resume  │  →  Split → Parse each
    └───────────────┘
        ↓
LangChain Prompt → Structured JSON extraction
        ↓
JSON → CSV conversion with date calculation
        ↓
Streamlit dashboard + CSV download
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| PDF OCR | Zerox |
| LLM (primary) | GPT-4o-mini (OpenAI) |
| LLM (secondary) | Anthropic Claude |
| Orchestration | LangChain |
| Prompt Templates | LangChain ChatPromptTemplate |
| UI | Streamlit |
| Data processing | Pandas, CSV |
| Async processing | Python asyncio |
| Environment | python-dotenv |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Neethadhiya/resume-parser-agent.git
cd resume-parser-agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```txt
streamlit
langchain
langchain-openai
anthropic
pyzerox
pandas
python-dotenv
tqdm
asyncio
```

---

## 🖥️ Usage

1. Open the app in your browser at `http://localhost:8501`
2. Click **"Choose PDF files"** and upload one or multiple resume PDFs
3. Click **"🔍 Process Resumes"**
4. View extracted data in the interactive table
5. Click **"📥 Download CSV"** to save results

---

## 📁 Project Structure

```
resume-parser-agent/
│
├── app.py                  # Main application file
├── .env                    # API keys (not committed)
├── .env.example            # Template for environment variables
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── resumes/                # Uploaded PDF files (auto-created)
└── output_results/         # Generated MD and CSV files (auto-created)
    ├── *.md                # Zerox OCR output
    └── parsed_resumes.csv  # Final structured output
```

---

## ✨ Key Features

- **Multi-resume detection** — automatically identifies if one PDF contains multiple candidates
- **Date normalisation** — converts all date formats to DD/MM/YYYY, handles "till date" → current date
- **Total experience calculation** — auto-calculates years and months of total work experience
- **Batch processing** — process hundreds of resumes in one run with progress tracking
- **Async architecture** — fast parallel processing using Python asyncio
- **Structured output** — clean CSV ready for ATS or HR systems

---

## 🔑 API Keys Required

| Service | Where to get |
|---|---|
| OpenAI API Key | [platform.openai.com](https://platform.openai.com) |
| Anthropic API Key | [console.anthropic.com](https://console.anthropic.com) |

---

## 🙋‍♀️ Built By

**Neetha PA** — AI/ML Engineer  
📧 neetha.pa2023@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/Neethadhiya)

---

## ⭐ If you find this useful, give it a star!
