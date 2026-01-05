# Build a Multi-Agent System With LangGraph

The future of AI isn’t about building a smarter chatbot; it’s about building a team.

Today, we will build that team using LangGraph. We will create a Multi-Agent System where one AI agent acts as a Researcher (browsing the web), and another acts as a Writer (synthesising that info). They will pass work to each other like colleagues in a newsroom.

If you are a student or a developer looking to break into the Agentic Workflow space, this is the perfect starting point. Let’s dive in.

## What is LangGraph?

Before we write code, let’s understand what we are building. Imagine a relay race. Runner A has the baton (data). They run their lap (task) and then pass the baton to Runner B. Runner B cannot start until they receive the baton.

**LangGraph** allows us to code this relay race.

- Nodes: These are the agents or functions (The Runners).
- Edges: These are the rules of who goes next (The Track).
- State: This is the shared memory (The Baton).

Instead of one giant prompt, we break the logic into small, reliable steps.

## The Setup

To keep this accessible and free, we are going to use Ollama to run a local LLM (Llama 3). This means you don’t need an OpenAI API key to follow along, though you will need a decent internet connection for the search tool.

**Prerequisites:**

- **Python Installed:** Make sure you have Python 3.9+.
- **Ollama Installed:** Download it from ollama.com.
- **Pull the Model:** Open your terminal and run: ollama pull llama3.

Next, create a virtual environment (optional but recommended) and install the necessary libraries:

```bash
pip install langgraph langchain langchain-community langchain-ollama duckduckgo-search
```

## Building a Multi-Agent System With LangGraph: Getting Started

We will build this in three parts: **The State**, **The Agents**, and **The Graph**.

### Step 1: Defining the Shared State

Think of the AgentState as a shared clipboard that hangs on the office wall. Every agent can read from it and write to it. This ensures that when the Researcher finds something, the Writer can actually see it:

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# ----- Shared State -----
class AgentState(TypedDict):
    topic: str
    research_data: List[str]  # A list of findings
    blog_post: str            # The final output
```

Without a structured state, agents are just shouting into the void. This TypedDict ensures type safety and clarity.

### Step 2: The Researcher Agent

Our first employee is the Researcher. Their job is simple: take a topic, search DuckDuckGo, and paste the results onto the clipboard (State):

```python
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

def researcher_node(state: AgentState):
    """
    Researcher agent that searches DuckDuckGo for the latest news and key facts about a given topic.
    """
    topic = state["topic"]
    print(f"Researcher is looking up: {topic}...")
    
    search = DuckDuckGoSearchRun()
    
    try:
        # You can tweak this query as you like
        results = search.run(f"key facts and latest news about {topic}")
    except Exception as e:
        results = f"Could not find data: {e}"
        
    print("Research complete.")
    
    # Only return the keys you want to update
    return {"research_data": state.get("research_data", []) + [results]}
```

Notice we aren’t using an LLM here yet! We are just using a deterministic tool (Search). This saves cost and reduces hallucinations. We are grounding the workflow in real data first.

### Step 3: The Writer Agent

Now, the Writer steps in. This agent uses Llama 3 (via Ollama). It reads the research_data found by the previous agent and drafts the content:

```python
# ChatOllama from langchain-ollama package
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
​
def writer_node(state: AgentState):
    print("Writer is drafting the post...")
    
    topic = state["topic"]
    data = state["research_data"][-1] if state["research_data"] else ""
    
    llm = ChatOllama(model="llama3", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a tech blog writer. 
Write a short, engaging blog post about "{topic}" 
based ONLY on the following research data:
​
{data}
​
Return just the blog post content."""
    )
    
    chain = prompt | llm
    response = chain.invoke({"topic": topic, "data": data})
    
    print("Writing complete.")
    return {"blog_post": response.content}
```

The temperature=0.7 gives the model a bit of creativity. If you wanted a strict report, you might lower this to 0.1.

### Step 4: Wiring the Graph

This is the key part. We define the workflow. It is a linear path: Start -> Researcher -> Writer -> End:

```python
# ----- Build the LangGraph -----
workflow = StateGraph(AgentState)
​
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)
​
# Flow: Start -> Researcher -> Writer -> END
workflow.set_entry_point("Researcher")
workflow.add_edge("Researcher", "Writer")
workflow.add_edge("Writer", END)
​
app = workflow.compile()
```

### Step 5: Running the System

Now, let’s fire it up. We trigger the app.invoke method with our initial input:

```python
if __name__ == "__main__":
    print("Starting the Multi-Agent System...\n")
    
    inputs: AgentState = {
        "topic": "The future of AI Agents",
        "research_data": [],
        "blog_post": "",
    }
    
    result = app.invoke(inputs)
    
    print("\n---------------- FINAL OUTPUT ----------------\n")
    print(result["blog_post"])
```

When you run this script, you will see the logs in your terminal:

1. The **researcher** will print that it is looking up “The future of AI Agents”.
2. A pause will occur while it fetches data from DuckDuckGo.
3. The writer will print that it is a draft.
4. A pause will occur while Llama 3 generates the text.
5. Final Output: A concise blog post appears, generated based on the actual search results.

Here’s the output you will see in the end:

```code
(env) (base) amankharwal@Amans-MacBook-Pro aiagent % python langagent.py     
Starting the Multi-Agent System...

Researcher is looking up: The future of AI Agents...
Research complete.
Writer is drafting the post...
Writing complete.

---------------- FINAL OUTPUT ----------------

**The Future of AI Agents: Revolutionizing Business and Commerce**

As we step into 2025, the world of business and commerce is poised to undergo a significant transformation with the rise of agentic AI. Gone are the days of automation and prediction; AI agents are now capable of taking real actions, from running marketing campaigns to managing supply chains.
...
```

## Closing Thoughts

So, this is how to build a Multi-Agent System with LangGraph. Building Multi-Agent systems can feel intimidating. It requires us to stop thinking like users, prompting a box, and start thinking like managers directing a team. Now, you are no longer limited by what one neural network can hold in its context window. You are orchestrating a system that can browse, think, critique, and refine.
