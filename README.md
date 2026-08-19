# 🔬 Multi-Agent Research & Report Generation System

An autonomous agentic AI system where **5 specialized agents** collaborate via a LangGraph `StateGraph` to research any topic, analyze sources, and produce a structured, cited PDF report — all with **zero human input** after the initial prompt.

## 🏗️ Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Planner  │───▶│ Search   │───▶│ Analyst  │───▶│ Writer   │───▶│ Critic   │
│ Agent    │    │ Agent    │    │ Agent    │    │ Agent    │    │ Agent    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────┬───┘
                                                     ▲                │
                                                     │   quality < 7  │
                                                     └────────────────┘
                                                       (max 2 loops)
```

| Agent | Role |
|-------|------|
| **Planner** | Decomposes the topic into 4–6 targeted subtopics |
| **Search** | Searches Tavily (web) + ArXiv (papers) per subtopic |
| **Analyst** | Extracts key findings, methodology, relevance score per source |
| **Writer** | Generates a structured markdown report with inline citations |
| **Critic** | Evaluates quality and requests revisions if score < 7/10 |

## 🛠️ Tech Stack

- **Agent Framework:** LangGraph (StateGraph)
- **LLM:** Llama 3.3 70B via Groq API
- **Web Search:** Tavily API
- **Academic Papers:** ArXiv API
- **PDF Generation:** ReportLab
- **UI:** Streamlit

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the example env file and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Run the application

```bash
streamlit run app.py
```

### 4. Enter a research topic and click **Generate** 🚀

## 📁 Project Structure

```
├── .env                    # API keys (not committed)
├── .env.example            # Template
├── requirements.txt        # Dependencies
├── app.py                  # Streamlit UI
├── config.py               # Configuration & LLM factory
├── state.py                # Shared TypedDict state
├── graph.py                # LangGraph pipeline assembly
├── agents/
│   ├── planner.py          # Topic decomposition
│   ├── search.py           # Web + ArXiv search
│   ├── analyst.py          # Source analysis
│   ├── writer.py           # Report writing
│   └── critic.py           # Quality evaluation
├── tools/
│   ├── tavily_search.py    # Tavily wrapper
│   └── arxiv_search.py     # ArXiv wrapper
├── utils/
│   └── pdf_generator.py    # ReportLab PDF generation
└── output/                 # Generated reports
```

## 📊 Output

Each run produces:
- **Markdown report** with inline citations `[1]`, `[2]`, etc.
- **Styled PDF** with cover page, branded headers, and formatted references
- **Quality assessment** from the Critic Agent

## 📝 License

MIT
