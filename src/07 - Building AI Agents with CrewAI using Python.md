# Building AI Agents with CrewAI using Python

CrewAI is an agent orchestration framework that allows you to coordinate multiple AI agents (each with a role, goal, memory, and tool access) to solve complex tasks collaboratively. If you want to learn how to use CrewAI to build AI Agents, this article is for you. In this article, I’ll take you through a guide to building AI Agents with CrewAI using Python.

Building AI Agents with CrewAI using Python
To explain building AI Agents with CrewAI, I’ll create a small market strategy crew, made of:

A Market Research Analyst Agent
A Content Strategist Agent
We will design the agents in a way to work together on a multi-step task, like:

Analyzing top competitors
Drafting a content marketing plan
To get started, make sure you have all the essential libraries and frameworks installed in your Python environment. Just execute the command mentioned below in your terminal or command prompt to get things ready:

pip install crewai llama-cpp-python transformers accelerate langchain
Step 1: Importing a Free Local LLM
Before we dive into building AI agents with CrewAI, we need a local language model that can process natural language prompts and generate meaningful responses.

Instead of relying on paid APIs like OpenAI, we’ll use Mistral-7B Instruct, a powerful open-source model available in the GGUF format, which runs efficiently on consumer hardware using the llama-cpp-python library.

In this step, we’ll load the quantized Mistral model into memory and wrap it in a simple Python function so that we can use it later as the core LLM behind our agents. Download this model from here:

1
from llama_cpp import Llama
2
​
3
llm = Llama(
4
    model_path="./models/mistral-7b-instruct.Q4_K_M.gguf",
5
    n_ctx=2048,
6
    n_threads=4,
7
    n_gpu_layers=0
8
)
9
​
10
def local_llm(prompt: str) -> str:
11
    output = llm(prompt, max_tokens=500, stop=["</s>"])
12
    return output["choices"][0]["text"].strip()
This setup ensures our AI agents operate entirely offline, cost-free, and fully under our control.

Step 2: Define Custom LLM Wrapper for CrewAI
Now that we’ve defined our local_llm() function to run prompts through a local Mistral model, we need to make it compatible with CrewAI’s agent system. CrewAI expects any language model to expose a .complete(prompt) method.

Since our local_llm() function doesn’t have this interface, we’ll wrap it inside a simple Python class that conforms to CrewAI’s expected structure:

1
class LocalLLMWrapper:
2
    def __init__(self, engine):
3
        self.engine = engine
4
​
5
    def complete(self, prompt: str) -> str:
6
        return self.engine(prompt)
7
​
8
llm_wrapper = LocalLLMWrapper(local_llm)
This wrapper will act as a bridge between our custom local model and the CrewAI framework, allowing us to use the model seamlessly when defining agents and tasks.

Step 3: Define Agents with Roles, Goals, and Backstories
With our local language model now running and wrapped to match CrewAI’s expected interface, we’re ready to define the core components of any agentic system: the agents themselves.

In CrewAI, each agent represents a unique persona with a specific role, goal, and backstory. This helps guide how the agent interprets tasks and generates responses.

In this step, we’ll create two agents: a Market Research Analyst and a Content Strategist, each powered by our local LLM:

1
from crewai import Agent
2
​
3
researcher = Agent(
4
    role='Market Research Analyst',
5
    goal='Analyze competitors and summarize their marketing strategies',
6
    backstory='An expert in market intelligence and competitive analysis.',
7
    llm=llm_wrapper,
8
    allow_delegation=False
9
)
10
​
11
writer = Agent(
12
    role='Content Strategist',
13
    goal='Use research to create a compelling marketing strategy document',
14
    backstory='A seasoned content strategist with a flair for storytelling.',
15
    llm=llm_wrapper
16
)
These agents will collaborate later on a shared task, and defining them properly now lays the foundation for effective multi-agent coordination.

Step 4: Define Tasks for Each Agent
Now that we’ve defined our agents and assigned each one a clear role and purpose, the next step is to define the tasks they need to perform.

In CrewAI, tasks are the actionable units of work that agents are responsible for. Each task includes a description of what needs to be done, the agent responsible, and the expected output. You can also define dependencies between tasks, allowing one agent to build upon the output of another. This is especially useful for multi-step workflows.

In the following code, we’ll define two tasks: one for analyzing competitors, and another for creating a content strategy based on that analysis:

1
from crewai import Task
2
​
3
task1 = Task(
4
    description="List top 3 competitors and their marketing strategies based on current trends.",
5
    agent=researcher,
6
    expected_output="A summary of 3 competitors with key marketing strategies."
7
)
8
​
9
task2 = Task(
10
    description="Create a content marketing strategy based on the competitor summary.",
11
    agent=writer,
12
    expected_output="A structured document with our content strategy inspired by competitors.",
13
    depends_on=[task1]
14
)
This structure ensures a clear flow of information and coordination between agents.

Step 5: Create and Run a Crew
With our agents and tasks defined, the final step is to orchestrate the entire workflow by bringing everything together into a Crew. A Crew in CrewAI is essentially the manager who coordinates how agents interact and execute their assigned tasks.

By passing in our list of agents and tasks, we will define a collaborative environment where each agent knows what to do and when:

1
from crewai import Crew
2
​
3
crew = Crew(
4
    agents=[researcher, writer],
5
    tasks=[task1, task2],
6
    verbose=True  # see what each agent does
7
)
8
​
9
result = crew.kickoff()
10
print(result)
Setting verbose=True allows us to observe how each agent reasons through its task in real-time, making debugging and optimization easier. The kickoff() method then launches the multi-agent workflow and returns the final output after all tasks have been completed.

Here’s the output you will see in the end:

Running task for agent: Market Research Analyst
Task: List top 3 competitors and summarize their marketing strategies.

Agent output:
1. Company A uses SEO-driven content and publishes long-form blog posts weekly.
2. Company B focuses on social media marketing, especially Instagram and TikTok influencers.
3. Company C emphasizes thought leadership with monthly webinars and whitepapers.

Running task for agent: Content Strategist
Task: Write a content marketing plan based on the research.

Agent output:
Based on the competitor research, we propose the following strategy:
- Develop a blog content calendar focused on high-volume keywords.
- Partner with micro-influencers in the marketing domain.
- Launch a quarterly webinar series to drive brand authority and engagement.

Crew execution completed.
Final output:
Our proposed content strategy synthesizes the best practices from competitors and aligns with our brand goals. It includes SEO blogging, influencer collaborations, and quarterly webinars to boost reach and authority.
Final Words
And that’s how you can build AI Agents using CrewAI entirely offline with Python and free open-source models. By combining thoughtful agent design, clear task definitions, and local LLMs, you now have a scalable, cost-free foundation for building intelligent, collaborative systems. I hope you liked this article on a guide to building AI Agents with CrewAI using Python.
