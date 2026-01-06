# Building an AI Agent using OpenAI API

## Introduction

With the rise of large language models like GPT-4o, building intelligent AI agents has never been more accessible. Using the OpenAI API, we can create agents that understand context, reason through information, and respond naturally to user queries. In this guide, we'll focus on building an AI Agent using the OpenAI API to intelligently process structured data and deliver accurate, conversational answers.

## Getting Started

Here, we will be building an AI Agent that will:

- Understand the data context
- Accept user queries in natural language
- Use OpenAI GPT models to analyze and respond intelligently

## Prerequisites

To get started with this task, make sure you sign up for the OpenAI API. Here are the steps you can follow:

1. Go to OpenAI's API platform and sign up or log in
2. Navigate to the API Keys section
3. Click on Create New Secret Key and copy it
4. Use it safely inside your Python code

Once you get your API key, you can use it like this:

```python
from openai import OpenAI

client = OpenAI(api_key='YOUR_OPENAI_API_KEY')
```

## Importing the Data

Now, we will import a dataset based on loan applications. We will use this data to build an AI Agent that can answer questions based on loan approvals. You can download the dataset from here.

```python
import pandas as pd

df = pd.read_csv('loan_prediction.csv')
```

## Creating Data Summary

Now, we will create a function to summarize the data:

```python
def create_data_summary(df):
    summary = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
    summary += "Columns:\n"
    for col in df.columns:
        summary += f"- {col} (type: {df[col].dtype})\n"
    return summary
```

This summary will help the agent understand the structure without actually loading the entire data into the prompt (which would exceed token limits).

## Building the AI Agent Function

Now, let's define the AI Agent that will handle user queries based on the data summary:

```python
def ai_agent(user_query, df):
    data_context = create_data_summary(df)

    prompt = f"""
You are a data expert AI agent.

You have been provided with this dataset summary:
{data_context}

Now, based on the user's question:
'{user_query}'

Think step-by-step. Assume you can access and analyze the dataset like a Data Scientist would using Pandas.

Give a clear, final answer.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    return answer
```

Here, we defined a function `ai_agent` that takes a user query and the dataset, summarizes the dataset structure, and creates a prompt combining both the context and the question. This prompt is then sent to OpenAI's GPT-4o model using the `client.chat.completions.create()` method, and the model's step-by-step, natural-language response is returned to the user.

## Creating an Interactive Loop

Now, we will create an interactive loop where users can ask questions to the AI Agent:

```python
print("Welcome to Loan Review AI Agent!")
print("You can ask anything about the loan applicants data.")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nYour question: ")
    if user_input.lower() == "exit":
        break
    response = ai_agent(user_input, df)
    print("\nAI Agent Response:")
    print(response)
```

In this part, we have created a simple interactive loop that continuously prompts the user to ask questions. When the user inputs a query, it is passed to the `ai_agent` function, which processes it and returns a natural language answer based on the dataset. If the user types "exit," the loop breaks and the program ends.

## Testing the Agent

Now, let's ask some queries to test the agent:

**Example Query 1:** What is the average loan amount applied for by all applicants?

- Step 1: Analyze the 'LoanAmount' column in the dataset.
- Step 2: Calculate the mean value.
- **Result:** The average loan amount applied for by all applicants is ₹146.41 thousand.

**Example Query 2:** Who has the highest applicant income?

- Step 1: Find the maximum value in the 'ApplicantIncome' column.
- **Result:** The highest applicant income is ₹81,000.

**Example Query 3:** How many applicants are self-employed?

- Step 1: Filter the dataset where 'Self_Employed' is 'Yes'.
- **Result:** There are 82 self-employed applicants.

## Summary

So, this is how we can build an AI Agent using the OpenAI API. Using the OpenAI API, we can create agents that understand context, reason through information, and respond naturally to user queries. I hope you liked this guide on building an AI Agent using the OpenAI API.
