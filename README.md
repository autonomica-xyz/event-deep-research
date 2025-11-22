[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# Deep Research Agent

**Generalized AI-powered research agent** that can perform deep research on any topic with pluggable research types. Originally built for biographical research, now supports **people research, company research, market analysis, topic exploration, and custom research types**.

<img src="media/event-deep-research.webp" alt="Deep Research Agent" width="600"/>

## Table of Contents

- [Deep Research Agent](#deep-research-agent)
  - [Table of Contents](#table-of-contents)
  - [🌟 What's New](#-whats-new)
  - [Features](#features)
  - [Available Research Types](#available-research-types)
  - [Quick Start Examples](#quick-start-examples)
    - [Biography Research](#biography-research)
    - [Company Research](#company-research)
    - [Market Research](#market-research)
    - [Topic Research](#topic-research)
  - [🚀 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Via LangGraph Studio (Recommended)](#via-langgraph-studio-recommended)
    - [Via Python Script](#via-python-script)
  - [Creating Custom Research Types](#creating-custom-research-types)
  - [Configuration](#configuration)
  - [Architecture](#architecture)
    - [Core Components](#core-components)
    - [Research Type System](#research-type-system)
  - [Roadmap / Future Work](#roadmap--future-work)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

---

## 🌟 What's New

**v2.1 - Open Source Search Providers**

- 🔍 **Pluggable Search Providers**: Use Tavily, Brave, DuckDuckGo, or SearXNG
- 🆓 **100% Free Option**: DuckDuckGo search requires NO API key
- 🏠 **Self-Hosted Option**: SearXNG for complete control
- 💰 **Free Tiers**: Brave Search offers 2,000 queries/month free
- 🔧 **Auto-Detection**: Automatically selects best available provider

**v2.0 - Generalized Research System**

- ✨ **Pluggable Research Types**: Easily add new research domains without modifying core code
- 🏢 **Company Research**: Extract structured company profiles (leadership, products, financials)
- 📊 **Market Research**: Analyze markets and industries (trends, players, outlook)
- 📚 **Topic Research**: General-purpose research for any subject
- 🎯 **Custom Research Types**: Create your own with minimal code (see [examples/add_custom_research_type.py](examples/add_custom_research_type.py))
- 🔧 **Backward Compatible**: Existing biography research continues to work

## Available Search Providers

| Provider | Cost | API Key | Setup | Best For |
|----------|------|---------|-------|----------|
| **DuckDuckGo** | Free ∞ | ❌ No | None | Development, personal projects, privacy |
| **Brave** | Free (2K/mo) | ✅ Yes | Sign up | Production, high quality results |
| **SearXNG** | Free ∞ | ❌ No | Self-host | Full control, unlimited queries |
| **Tavily** | Paid | ✅ Yes | Subscribe | Original, high quality (optional) |

**Quick Setup:**
- **DuckDuckGo**: Works out of the box, no configuration needed!
- **Brave**: Get API key at https://brave.com/search/api/ (free tier: 2,000/month)
- **SearXNG**: `docker run -d -p 8888:8080 searxng/searxng` then set `SEARXNG_URL=http://localhost:8888`

See [examples/search_provider_comparison.py](examples/search_provider_comparison.py) to test all providers.

## Features

- **Multiple Research Types**: Biography, Company, Market, Topic, and Custom
- **Breadth-First Parallel Research**: People research executes 7 domain queries in parallel for comprehensive coverage
- **Supervisor Agent**: Coordinates workflow with multiple tools (Research, Think, Finish)
- **Smart Merging**: Deduplicate and combine information from multiple sources
- **Multi-Model Support**: OpenAI, Anthropic, Google, or Local models (Ollama)
- **Extensible Architecture**: Add new research types in minutes
- **LangGraph Studio**: Visual debugging and real-time monitoring

## Available Research Types

| Research Type | Description | Output Schema |
|--------------|-------------|---------------|
| **Biography** | Historical figures, people | Chronological timeline of life events |
| **People** | Individuals, public figures (breadth-first parallel research) | Comprehensive profile with categorized facts across 7 domains |
| **Company** | Businesses, organizations | Company profile with facts by category |
| **Market** | Industries, markets | Market insights and analysis |
| **Topic** | General knowledge | Structured sections with key points |
| **Custom** | *Your domain* | *Define your own* |

## Quick Start Examples

### Biography Research

Research historical figures and extract structured life events:

```python
from src.graph import graph

result = await graph.ainvoke({
    "research_subject": "Albert Einstein",
    "research_type": "biography"
})

# Output: Chronological timeline with events
print(result["structured_output"])
# [ChronologyEvent(name="Birth in Ulm", date={year: 1879}, ...), ...]
```

**Example Output:**
```json
{
  "structured_output": [
    {
      "id": "birth_1879",
      "name": "Birth in Ulm",
      "description": "Albert Einstein was born in Ulm, Germany",
      "date": {"year": 1879, "note": "March 14"},
      "location": "Ulm, German Empire"
    },
    {
      "id": "nobel_1921",
      "name": "Nobel Prize in Physics",
      "description": "Awarded Nobel Prize for photoelectric effect discovery",
      "date": {"year": 1921, "note": ""},
      "location": "Stockholm, Sweden"
    }
  ]
}
```

### People Research (Breadth-First Parallel Strategy)

Research individuals using **parallel breadth-first search** across 7 domains:

```python
result = await graph.ainvoke({
    "research_subject": "Elon Musk",
    "research_type": "people"
})

# Output: PeopleProfile with categorized facts from 7 parallel research domains
print(result["structured_output"])
# PeopleProfile(person_name="Elon Musk", summary="...", facts=[...])
```

**Research Domains (executed in parallel):**
1. Professional & Social Media (LinkedIn, career)
2. Technical Contributions (GitHub, open source)
3. Cryptocurrency/Blockchain (Bitcoin, crypto projects)
4. Publications & Media (articles, interviews)
5. Business & Legal (companies, startups)
6. Academic & Education (degrees, research)
7. Community & Speaking (conferences, talks)

**Example Output (from parallel breadth-first search):**
```json
{
  "structured_output": {
    "person_name": "Elon Musk",
    "summary": "Entrepreneur and business magnate known for founding and leading multiple technology companies including Tesla, SpaceX, and Neuralink.",
    "facts": [
      {
        "category": "demographics",
        "title": "Birth and Early Life",
        "content": "Born June 28, 1971 in Pretoria, South Africa. Moved to Canada at age 17...",
        "source_date": "2024"
      },
      {
        "category": "professional",
        "title": "CEO of Tesla and SpaceX",
        "content": "Currently serves as CEO of Tesla Inc. and SpaceX, leading innovation in electric vehicles and space exploration...",
        "source_date": "2024"
      },
      {
        "category": "technical_contributions",
        "title": "Open Source Contributions",
        "content": "Released Tesla's electric vehicle patents to open source in 2014...",
        "source_date": "2014"
      },
      {
        "category": "crypto_blockchain",
        "title": "Cryptocurrency Advocacy",
        "content": "Active supporter of Dogecoin and Bitcoin, Tesla briefly accepted BTC payments...",
        "source_date": "2021"
      },
      {
        "category": "business_ventures",
        "title": "Multiple Company Founder",
        "content": "Founded/co-founded PayPal, SpaceX, Tesla, Neuralink, The Boring Company, and X (Twitter)...",
        "source_date": "2024"
      }
    ]
  }
}
```

### Company Research

Research companies and extract structured profiles:

```python
result = await graph.ainvoke({
    "research_subject": "OpenAI",
    "research_type": "company"
})

# Output: CompanyProfile with categorized facts
print(result["structured_output"])
# CompanyProfile(company_name="OpenAI", facts=[...])
```

### Market Research

Analyze markets and industries:

```python
result = await graph.ainvoke({
    "research_subject": "AI Chip Market",
    "research_type": "market"
})

# Output: MarketReport with insights
print(result["structured_output"])
# MarketReport(market_name="AI Chip Market", insights=[...])
```

### Topic Research

Research any general topic:

```python
result = await graph.ainvoke({
    "research_subject": "Quantum Computing",
    "research_type": "topic"
})

# Output: TopicReport with structured sections
print(result["structured_output"])
# TopicReport(topic_name="Quantum Computing", sections=[...])
```

## 🚀 Installation

### Prerequisites

- **Python 3.12+**
- **uv** (Python package manager)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/autonomica-xyz/event-deep-research.git
cd event-deep-research

# 2. Create virtual environment and install dependencies
uv venv && source .venv/bin/activate
uv sync

# 3. Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# Required:
# - OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY
#   (Choose your preferred LLM provider)
#
# Optional (for web scraping):
# - FIRECRAWL_API_KEY (if not set, will use alternative scraping methods)
#
# Optional (search providers - DuckDuckGo works out of box!):
# - BRAVE_API_KEY (free tier: 2,000 queries/month)
# - SEARXNG_URL (self-hosted instance URL)
# - TAVILY_API_KEY (original provider, now optional)

# 4. Start the development server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev --allow-blocking
# Open http://localhost:2024 to access LangGraph Studio
```

## Usage

### Via LangGraph Studio (Recommended)

1. Start the development server:
   ```bash
   uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.12 langgraph dev --allow-blocking
   ```

2. Open http://localhost:2024

3. Select the `supervisor` graph

4. Input your research query:
   ```json
   {
     "research_subject": "Tesla Inc",
     "research_type": "company"
   }
   ```

5. Watch the agent work in real-time!

### Via Python Script

See the [examples/](examples/) directory for complete examples:

**Research Types:**
- `biography_research.py` - Research historical figures
- `people_research.py` - Research individuals and public figures
- `company_research.py` - Research companies
- `market_research.py` - Analyze markets
- `topic_research.py` - Research any topic
- `add_custom_research_type.py` - Create your own research type

**Search Providers:**
- `using_duckduckgo_search.py` - Use DuckDuckGo (100% free, no API key)
- `using_brave_search.py` - Use Brave Search API (free tier)
- `using_searxng.py` - Use SearXNG (self-hosted)
- `search_provider_comparison.py` - Compare all providers

```bash
# Run a biography research example
python examples/biography_research.py

# Run with DuckDuckGo (no API key needed!)
python examples/using_duckduckgo_search.py

# Compare all available search providers
python examples/search_provider_comparison.py
```

## Creating Custom Research Types

Adding a new research type takes just a few steps:

1. **Define your data models** (input/output schemas)
2. **Implement ResearchType interface** (prompts, structuring logic)
3. **Register your type** with the registry

**Complete example:** [examples/add_custom_research_type.py](examples/add_custom_research_type.py)

```python
from src.research_types.base import ResearchType
from src.research_types.registry import ResearchTypeRegistry

class MyResearchType(ResearchType):
    @property
    def name(self) -> str:
        return "my_type"

    def get_supervisor_prompt(self) -> str:
        # Return your supervisor prompt template
        pass

    def get_output_schema(self) -> Type[BaseModel]:
        # Return your Pydantic output model
        pass

    # Implement other abstract methods...

# Register it
ResearchTypeRegistry.register(MyResearchType())

# Use it
result = await graph.ainvoke({
    "research_subject": "My Subject",
    "research_type": "my_type"
})
```

## Configuration

Located in `src/configuration.py`:

```python
class Configuration(BaseModel):
    # Research type to use (biography, company, market, topic, etc.)
    research_type: str = "biography"

    # Search provider (auto-detects if None)
    search_provider: str | None = None  # tavily, brave, duckduckgo, searxng

    # Primary LLM model for all tasks
    llm_model: str = "google_genai:gemini-2.5-flash"

    # Optional model overrides for specific tasks
    structured_llm_model: str | None = None  # For JSON output
    tools_llm_model: str | None = None       # For tool calling
    chunk_llm_model: str | None = None       # For chunk processing

    # Token limits
    structured_llm_max_tokens: int = 4096
    tools_llm_max_tokens: int = 4096

    # Retry policies
    max_structured_output_retries: int = 3
    max_tools_output_retries: int = 3

    # Processing constraints
    default_chunk_size: int = 800
    max_content_length: int = 100000
    max_tool_iterations: int = 5
    max_chunks: int = 20
```

**Search Provider Selection:**
- If `search_provider` is None, auto-detects based on available API keys
- Priority: Tavily → Brave → SearXNG → DuckDuckGo
- DuckDuckGo always works as fallback (no API key needed)

**Supported Models:**
- OpenAI: `openai:gpt-4-turbo`, `openai:gpt-3.5-turbo`
- Anthropic: `anthropic:claude-3-5-sonnet-20241022`
- Google: `google_genai:gemini-2.5-flash`, `google_genai:gemini-2.5-pro`
- Ollama: `ollama:mistral-nemo`, `ollama:llama3.1`

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│              SUPERVISOR AGENT (Generic)                      │
│   - Loads appropriate ResearchType                          │
│   - Uses research-type-specific prompts                     │
│   - Coordinates all sub-agents                              │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
   ┌────▼──────────────────┐  ┌──▼────────────────────┐
   │ RESEARCH AGENT        │  │ STRUCTURE OUTPUT      │
   │ - Web search (Tavily) │  │ - Research-type aware │
   │ - URL crawling        │  │ - Custom formatting   │
   │ - Data merging        │  │ - JSON generation     │
   └────┬──────────────────┘  └───────────────────────┘
        │
   ┌────┴──────────────────────────────────────┐
   │                                            │
┌──▼──────────────────┐        ┌───────────────▼──┐
│ URL CRAWLER         │        │ MERGE AGENT       │
│ - Firecrawl scrape  │        │ - Chunk text      │
│ - Content extract   │        │ - Filter relevant │
└─────────────────────┘        │ - Deduplicate     │
                               └───────────────────┘
```

### Research Type System

The system uses a **pluggable architecture** where each research type is self-contained:

```
src/research_types/
├── base.py                 # ResearchType abstract class
├── registry.py             # Central registry
├── biography.py            # Biographical research
├── company.py              # Company research
├── market.py               # Market research
├── topic.py                # General topic research
└── __init__.py             # Auto-registration
```

Each research type defines:
- **Supervisor prompts** - How to guide the research
- **Data schemas** - What structure to accumulate data in
- **Gap analysis prompts** - How to identify missing information
- **Output formatting** - How to structure final results

**Key Benefits:**
- ✅ Add new research types without touching core code
- ✅ Each type is isolated and testable
- ✅ Easy to maintain and extend
- ✅ Backward compatible

<img src="media/kronologs-graph.webp" alt="Agent Graph" />

## Roadmap / Future Work

- [x] **Open Source Search**: ✅ Implemented! (DuckDuckGo, Brave, SearXNG)
- [ ] **Open Source Scraping**: Replace Firecrawl with Playwright/Trafilatura
- [ ] **Research Type Library**: More built-in types (product, scientific paper, legislation)
- [ ] **Multimedia Support**: Add images and videos to research output
- [ ] **Performance**: Improve merge agent speed
- [ ] **Observability**: Enhanced LangSmith/Langfuse integration
- [ ] **Custom Search Providers**: Easy interface for adding custom search backends

## Contributing

We welcome contributions! This is a great project to learn:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

**Ideas for contributions:**
- New research types (products, scientific papers, recipes, etc.)
- Open-source tool replacements (search, scraping)
- Performance optimizations
- Documentation improvements
- Test coverage

See the [open issues](https://github.com/autonomica-xyz/event-deep-research/issues) for a full list of proposed features and known issues.

## License

Distributed under the MIT License. See `LICENSE.txt` for details.

## Acknowledgments

- **[LangChain](https://github.com/langchain-ai/langchain)** - Foundational LLM framework
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Multi-agent orchestration
- **[Open Deep Research](https://github.com/langchain-ai/open_deep_research)** - Research methodology inspiration
- **[Brave Search](https://brave.com/search/api/)** - Privacy-focused search API
- **[DuckDuckGo](https://duckduckgo.com/)** - Privacy-focused search
- **[SearXNG](https://github.com/searxng/searxng)** - Self-hosted meta-search engine
- **[Tavily](https://tavily.ai/)** - Original web search provider (optional)
- **[Firecrawl](https://www.firecrawl.com/)** - Web scraping

[contributors-shield]: https://img.shields.io/github/contributors/autonomica-xyz/event-deep-research.svg?style=for-the-badge
[contributors-url]: https://github.com/autonomica-xyz/event-deep-research/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/autonomica-xyz/event-deep-research.svg?style=for-the-badge
[forks-url]: https://github.com/autonomica-xyz/event-deep-research/network/members
[stars-shield]: https://img.shields.io/github/stars/autonomica-xyz/event-deep-research.svg?style=for-the-badge
[stars-url]: https://github.com/autonomica-xyz/event-deep-research/stargazers
[issues-shield]: https://img.shields.io/github/issues/autonomica-xyz/event-deep-research.svg?style=for-the-badge
[issues-url]: https://github.com/autonomica-xyz/event-deep-research/issues
[license-shield]: https://img.shields.io/github/license/autonomica-xyz/event-deep-research.svg?style=for-the-badge
[license-url]: https://github.com/autonomica-xyz/event-deep-research/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/bernat-sampera-195152107/
