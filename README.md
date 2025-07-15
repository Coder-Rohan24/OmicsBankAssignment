# 🧠 ICD-10 Diagnosis Mapper (RAG-Powered)

This project is a high-performance, scalable system for **automated ICD-10 code mapping** from free-text patient diagnoses. It uses a **Retrieval-Augmented Generation (RAG)** approach combining a **Chroma vector store**, **HuggingFace embeddings**, and **Gemini Flash LLM**, with **rule-based fallbacks** to ensure reliable coding even in edge cases.

## 📈 Performance Logging

- Logs batch time, fallback count, and cache stats
- Saves fallback justifications to `fallbacks.csv`
- Saves timing stats per 10-diagnosis batch to `timing_log.csv`

---

## 📌 Features

- 🔎 **Vector-based semantic search** over enhanced ICD-10 definitions
- 🤖 **LLM-powered reasoning** using Gemini Flash via prompt engineering
- 💾 **Persistent Chroma vector store** for fast retrieval
- ♻️ **Rule-based fallback logic** for unmatched or low-confidence cases
- 📈 **Batch processing** of large diagnosis datasets
- 📄 **Structured outputs** with code, description, justification, and confidence
- 🧠 Designed to scale for **lakhs of diagnosis entries**

---

## 📐 System Architecture

```
    A[Patient Diagnoses] --> B[Preprocessing(Tokenize list)]
    B --> C[Diagnosis-wise RAG Workflow]
    C --> D[Vector Retrieval(Chroma Vector Store)]
    D --> E[Prompt Creation]
    E --> F[LLM (Gemini Flash)]
    F --> G[Structured ICD-10 Output]

    C --> H[⛽ Rule-based Fallback]
    H --> I[Generic Default Code]
    I --> G

    F -->|Error / Low Confidence| H
```

---

## 🛠️ Technologies Used

| Library/Service             | Purpose                                                    | Alternatives                            |
| --------------------------- | ---------------------------------------------------------- | --------------------------------------- |
| **LangChain**               | Orchestration of retrieval and LLM chains                  | LlamaIndex, Haystack                    |
| **Chroma**                  | Local vector store for semantic retrieval                  | FAISS, Weaviate, Qdrant                 |
| **HuggingFace Embeddings**  | Generates vector embeddings for ICD descriptions           | SentenceTransformers, OpenAI embeddings |
| **Gemini Flash (Google)**   | Fast and lightweight LLM for code suggestion and reasoning | OpenAI GPT, Claude, Cohere, Mistral     |
| **Transformers (Fallback)** | Rule-based code matcher in edge cases                      | spaCy, scikit-learn keyword matching    |
| **Pandas**                  | DataFrame operations and processing                        | Polars, Dask (for bigger data)          |
| **TQDM**                    | Real-time progress bars for long processes                 | Rich                                    |
| **Logging**                 | File-based detailed logs for performance & debugging       | Loguru, standard logging                |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Coder-Rohan24/OmicsBankAssignment.git
cd OmicsBankAssignment
```

### 2. Quick Start with Google Colab

The easiest way to get started is with our Google Colab notebook:

1. Open the included Colab notebook in your browser
2. Run all cells to see the complete implementation
3. The notebook includes sample data and step-by-step execution

### 3. Local Setup

For local development, install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Dataset and Resources

This repository includes:

- **ICD-10 Dataset**: Pre-processed ICD-10 codes dataset (`icd_df.csv`) ready for use
- **Google Colab Notebook**: Interactive notebook with complete implementation
- **Architecture PDF**: Detailed technical documentation of the system architecture

Load the included dataset:

```python
import pandas as pd
icd_df = pd.read_csv("icd_df.csv")
```

### 4. Setup Diagnosis Data

Ensure a column `"Diagnoses_list"` that contains diagnosis strings as Python lists.

Example:

```csv
Diagnoses_list
["hypertension", "type 2 diabetes"]
["fever", "cough"]
```

Load and convert:

```python
diagnosis_df = pd.read_csv("path/to/diagnosis_data.csv")
```

---

## 💡 Usage Example

```python
from icd10_mapper import ImprovedICD10Mapper

# Load the included ICD dataset
icd_df = pd.read_csv("icd_df.csv")

# Instantiate the Mapper
mapper = ImprovedICD10Mapper(icd_df=icd_df, google_api_key="your_gemini_api_key")

# Process diagnosis data
diagnosis_df = pd.read_csv("diagnosis_list.csv")
results_df = mapper.process_diagnosis_dataset(diagnosis_df)

# Save output
results_df.to_csv("mapped_diagnoses.csv", index=False)

# Optional: Save fallback and performance logs
mapper.save_logs()
```

---

## 📊 Output Format

The `results_df` contains:

| Column                | Description                                 |
| --------------------- | ------------------------------------------- |
| `icd_code`            | Predicted ICD-10 code                       |
| `description`         | Corresponding ICD description               |
| `justification`       | Reasoning provided by LLM or fallback logic |
| `confidence`          | High / Medium / Low                         |
| `original_diagnosis`  | The raw input diagnosis                     |
| `retrieved_icd_codes` | Top codes retrieved from vector store       |
| `patient_row`         | Index of original patient record            |

---

## 🧪 Testing

You can test individual diagnoses:

```python
mapper.map_diagnosis_with_rag("acute myocardial infarction")
```

---

## 📂 File Structure

```
OmicsBankAssignment/
├── icd10_mapper.py                 # Main implementation
├── [notebook_name].ipynb           # Google Colab notebook
├── icd_df.csv                     # ICD-10 codes dataset
├── architecture_details.pdf       # Technical architecture documentation
├── diagnosis_list.csv             # Sample diagnosis data
├── processing.log                 # Runtime logs
├── fallbacks.csv                  # Fallback cases log
├── timing_log.csv                 # Performance timing data
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 📋 Repository Contents

- **📔 Google Colab Notebook**: Complete interactive implementation with examples
- **📊 ICD-10 Dataset**: Pre-processed dataset ready for immediate use
- **📄 Architecture PDF**: Detailed technical documentation and system design
- **🐍 Python Implementation**: Core mapping logic and utilities
- **📈 Performance Logs**: Runtime metrics and fallback analysis

## 📚 Documentation

For detailed technical information about the system architecture, algorithms, and implementation details, please refer to the `architecture_details.pdf` included in this repository.

---
## Google Colab Link: https://colab.research.google.com/drive/14XfeK6VtsAnJWqmXZeVs55_xjfRkzE2N?usp=sharing

## 📎 License

MIT License. Free to use for research and education.

---

## 👤 Author

**Rohan**  
GitHub: [@Coder-Rohan24](https://github.com/Coder-Rohan24)

> Passionate about large-scale AI systems, NLP, and production-grade machine learning.

---

## 🌐 Acknowledgements

- LangChain for seamless LLM orchestration
- Google Gemini API
- HuggingFace for MiniLM embeddings
- ChromaDB for fast local vector search

---

## 🙋‍♂️ Questions?

Raise an issue or contact me via GitHub for help or improvements.
