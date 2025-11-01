# UW-Madison Course Analytics RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that enables natural language queries about UW-Madison course grade distributions, average GPAs, and enrollment statistics.

## 🎯 Overview

This project allows students to ask questions like:
- "What's the average GPA for CS 400?"
- "Show me grade distributions for Data Science courses"
- "Which computer science courses have the highest average grades?"

The chatbot uses semantic search to retrieve relevant course data and generates natural language responses.

## 🛠️ Tech Stack

- **LlamaIndex**: Pipeline orchestration and RAG framework
- **FAISS**: Vector similarity search for efficient retrieval
- **HuggingFace**: Embedding models for semantic understanding
- **Python**: Core development language


## 🏗️ Architecture

```
User Query → Embedding Model → FAISS Vector Search → Context Retrieval → LLM Response Generation
```

1. **Data Preprocessing**: Course grade data is cleaned and structured
2. **Embedding Generation**: HuggingFace models convert text to vector embeddings
3. **Vector Storage**: FAISS indexes embeddings for fast similarity search
4. **Query Processing**: User questions are embedded and matched against the index
5. **Response Generation**: Retrieved context is used to generate accurate answers

## 📋 Prerequisites

- Python 3.8+
- pip or conda for package management

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/uw-course-rag-chatbot.git
cd uw-course-rag-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## 💻 Usage

### Running the Chatbot

```bash
python main.py
```

### Example Queries

```python
# Example 1: Average GPA query
>>> "What is the average GPA for CS 540?"
Response: "Based on historical data, CS 540 (Intro to AI) has an average GPA of 3.2..."

# Example 2: Grade distribution
>>> "Show me the grade distribution for STAT 371"
Response: "STAT 371 grade distribution: A: 25%, AB: 30%, B: 25%..."
```

## 📁 Project Structure

```
uw-course-rag-chatbot/
├── data/
│   ├── raw/                 # Raw grade data
│   └── processed/           # Cleaned and structured data
├── src/
│   ├── data_processing.py   # Data cleaning and preparation
│   ├── embeddings.py        # Embedding generation
│   ├── retrieval.py         # FAISS search implementation
│   └── chatbot.py           # Main chatbot logic
├── notebooks/
│   └── exploration.ipynb    # Data exploration and testing
├── tests/
│   └── test_retrieval.py    # Unit tests
├── .env.example             # Environment variable template
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                  # Entry point
```

## 🔧 Configuration

Key configuration options in `config.py`:
- `EMBEDDING_MODEL`: HuggingFace model for embeddings
- `CHUNK_SIZE`: Text chunk size for indexing
- `TOP_K`: Number of results to retrieve
- `TEMPERATURE`: LLM response creativity

## 🚧 Roadmap

- [x] Data preprocessing pipeline
- [x] FAISS vector indexing
- [x] Basic query functionality
- [ ] Support for additional query types (professor ratings, course prerequisites)
- [ ] Historical trend analysis

## 🤝 Contributing

This is a collaborative project. Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📊 Data Source

Course grade data is sourced from [UW-Madison's public grade distribution reports]. All data is anonymized and aggregated.

## 📝 License

MIT License - see LICENSE file for details

## 👥 Authors

- Brandon Wee - [GitHub](https://github.com/yourusername) | [LinkedIn](your-linkedin)
- Collaborator Name

## 🙏 Acknowledgments

- UW-Madison for publicly available grade data
- LlamaIndex, FAISS, and HuggingFace communities

## 📧 Contact

For questions or feedback, reach out at brandondwee@gmail.com

---

**Note**: This project is for educational purposes to help students make informed course decisions.
