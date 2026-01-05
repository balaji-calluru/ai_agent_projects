# Build a Real-Time AI Assistant Using RAG + LangChain

If you’ve ever wanted to build your own real-time AI assistant that can answer questions from your documents, websites, PDFs, notes, or knowledge bases, this is your perfect starting point. Today, I’ll teach you how to build a powerful, real-time AI assistant using RAG and LangChain, all with 100% free, open-source tools you can run on your own machine.

Our goal is to build an assistant that can answer questions by searching the web right now. We won’t be using any paid APIs from OpenAI or Google. This is all you.

Here’s the toolkit we will be using for this task:

1. Ollama: A fantastic tool that lets you download and run powerful open-source LLMs (like Meta’s Llama 3 or Mistral’s Mistral) right on your computer.
2. LangChain: The core framework we’ll use to build our application’s chain of logic.
3. DuckDuckGo Search: A free Python library that lets us perform web searches without needing an API key.
Let’s get started!

## Step 1: Set Up Your Environment

First, you need to install Ollama. Go to ollama.com and download the app for your OS (Mac, Windows, or Linux).

Once installed, open your terminal and pull a model. Let’s use Llama 3 8B, a powerful and fast model:

```bash
ollama pull llama3:8b
```

Next, let’s install the required Python libraries:

```bash
pip install langchain langchain_community langchain_ollama duckduckgo-search
```

## Step 2: Assemble Your Components in Python

Create a new Python file (assistant.py). Let’s import our tools and set up the main components:

```python
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
```

Now, let’s initialise the two main parts:

1. The LLM: Llama 3, running via Ollama.
2. The Search Tool: DuckDuckGo.

```python
# We specify the model we pulled in Step 1
llm = Ollama(model="llama3:8b")
​
# This tool will run a web search
search = DuckDuckGoSearchRun()
```

## Step 3: Define the Chain with LCEL

This is the key part of our real-time AI Assistant. We need to tell LangChain how to route the information. We’ll use the LangChain Expression Language (LCEL), which looks like Python pipes.

First, we create a prompt template. This is the briefing we give to our LLM. Notice how we have placeholders for {context} (the search results) and {question}:

```python
# This is the prompt template, our instruction manual for the LLM
prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant. You must answer the user's question 
    based *only* on the following search results. If the search results 
    are empty or do not contain the answer, say 'I could not find 
    any information on that.'
​
    Search Results:
    {context}
​
    Question:
    {question}
    """
)
```

Now, we build the chain itself. Read the comments in the code to see how the data flows:

```python
# This is our RAG chain
chain = (
    RunnablePassthrough.assign(
        # "context" is a new key we add to the dictionary.
        # Its value is the *output* of running the 'search' tool
        # with the original 'question' as input.
        context=lambda x: search.run(x["question"])
    )
    | prompt  # The dictionary (now with 'context' and 'question') is "piped" into the prompt
    | llm     # The formatted prompt is "piped" into the LLM
)
```

That RunnablePassthrough.assign is the key. It takes the original input (a dictionary with a question key), retains it, and assigns a new key called ‘context’ by running the search tool.

## Step 4: Run Your Real-Time Assistant!

That’s it. The chain is built. Now, all we have to do is invoke it:

```python
print("🤖 Hello! I'm a real-time AI assistant. What's new?")
while True:
    try:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("🤖 Goodbye!")
            break
        
        print("🤖 Thinking...")
        
        # This one line runs the whole RAG process
        response = chain.invoke({"question": user_query})
        
        print(f"🤖: {response}")
​
    except Exception as e:
        print(f"An error occurred: {e}")
```

Run your Python file: python assistant.py.

Here’s what I asked it and what it replied:

```code
🤖 Hello! I'm a real-time AI assistant. What's new?
You: hi, can you help me find some ideas for writing a new article on any trending news based on the democracy in India
🤖 Thinking...
🤖: I'd be happy to help!

Based on the search results, I found a mention of "trend analysis" which suggests that Google Trends and social media hashtags can be used to stay updated on industry trends. This could potentially provide some ideas for writing an article about democracy in India.

One idea could be to analyze the trending news related to democracy in India using Google Trends or social media hashtags, such as #DemocracyInIndia or #IndianPolitics. You could also explore topics that are currently being discussed online, such as reform ideas or leadership issues.

Another idea could be to write an article about how Hong Kong's lack of protests for democracy (mentioned in the search results) contrasts with India's experiences and challenges. This could provide a thought-provoking perspective on the differences between these two regions.

I hope these ideas help spark some inspiration for your article!
You: exit
🤖 Goodbye!
```

You are now talking to an AI that can learn about the world in real-time.

Quick Note: What about my own documents? What we just built is Real-Time RAG. The classic RAG is very similar, but instead of using a web search as the retriever, you use a vector database (like the free, local ChromaDB). Here’s a guided example.

Final Words
RAG isn’t just a technical trick to get around knowledge cut-offs. It’s a profound step toward grounded AI. It connects the abstract, statistical intelligence of an LLM to the concrete, verifiable facts of the real world.

By building this, you’ve done more than pipe some data. You’ve given your AI a sense of now, a library and the curiosity to use it. You’ve built an assistant that doesn’t just know things; it’s ready to learn things.Build a Real-Time AI Assistant Using RAG + LangChain

