# Personal Swiggy Logger 🍽️

Personal Swiggy Logger is an intelligent agent system that analyzes Swiggy order statements using RAG (Retrieval Augmented Generation) techniques with local LLM integration. Built with Ollama and Gemma, it provides detailed insights about your ordering patterns, spending habits, and order history.

## 🌟 Features

- **Natural Language Understanding**: Process queries about your Swiggy orders in plain English
- **Local LLM Integration**: Uses Ollama with Gemma 3B for privacy-focused analysis
- **Advanced RAG Implementation**: Employs FAISS for efficient document retrieval
- **Multi-Stage Processing Pipeline**:
  - Perception: Understanding user intent and entities
  - Memory: Contextual information retention
  - Decision Making: Intelligent action planning
  - Action Execution: Tool-based response generation

## 🛠️ Technical Architecture

The system consists of four main components:

1. **Perception Module**: Processes user input to extract intents and entities
2. **Memory Manager**: Maintains context using FAISS vector store
3. **Decision Engine**: Plans actions based on user queries
4. **Action Executor**: Interfaces with tools and generates responses

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Required Python packages (install via requirements.txt)
- Swiggy order statements in PDF format

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SwiggyRAG.git
cd SwiggyRAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Pull the required Gemma model using Ollama:
```bash
ollama pull gemma3:1b
```

4. Place your Swiggy statement PDFs in the `data/` directory

## 🔧 Configuration

1. Ensure Ollama is running on the default port (11434)
2. The system will automatically create necessary indexes on first run

## 💻 Usage

1. Start the MCP server:
```bash
python src/mcp_server.py
```

2. Run the main agent:
```bash
python src/agent.py
```

3. Start asking questions! Examples:
```
🧑 What do you want to solve today? → What are my most ordered items?
🧑 What do you want to solve today? → What's my average order value?
🧑 What do you want to solve today? → Show me my ordering patterns
🧑 What do you want to solve today? → Give me a summary of all my orders
🧑 What do you want to solve today? → What are my spending trends?
```

## 🏗️ Project Structure

```
SwiggyRAG/
├── src/
│   ├── agent.py          # Main agent orchestration
│   ├── perception.py     # Intent and entity extraction
│   ├── memory.py         # Vector store and context management
│   ├── decision.py       # Action planning
│   ├── action.py         # Tool execution
│   └── mcp_server.py     # Tool server implementation
├── data/                 # Directory for Swiggy PDFs
└── faiss_index/         # Auto-generated vector indexes
```

## 🔍 How It Works

1. **Document Processing**:
   - PDFs are processed and chunked into manageable segments
   - Text chunks are embedded using the nomic-embed-text model
   - Embeddings are stored in a FAISS index for efficient retrieval

2. **Query Processing**:
   - User queries are analyzed for intent and entities
   - Relevant context is retrieved from the vector store
   - The decision engine plans appropriate actions
   - Tools execute the plan and generate responses

3. **Memory Management**:
   - Maintains session context for more coherent interactions
   - Stores tool outputs and facts for future reference
   - Uses semantic search for relevant information retrieval

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Ollama](https://ollama.ai/)
- Uses Google's Gemma model
- FAISS by Facebook Research
- Inspired by the RAG (Retrieval Augmented Generation) architecture

## ⚠️ Disclaimer

This is an unofficial tool and is not affiliated with, maintained, authorized, endorsed, or sponsored by Swiggy.
