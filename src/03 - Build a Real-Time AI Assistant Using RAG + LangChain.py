#!/usr/bin/env python3
"""
Real-Time AI Assistant Using RAG + LangChain

A powerful AI assistant that can answer questions by searching the web in real-time.
Uses Ollama (local LLM) and DuckDuckGo Search for completely free, open-source operation.
"""

import sys
import argparse
from typing import Dict, Any

try:
    from langchain_community.llms import Ollama
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
except ImportError as e:
    print(f"Error: Missing required package. Please install dependencies:")
    print(f"  pip install langchain langchain-community langchain-ollama duckduckgo-search")
    print(f"\nOriginal error: {e}")
    sys.exit(1)


class RealTimeAssistant:
    """A real-time AI assistant powered by RAG and LangChain."""
    
    def __init__(self, model: str = "llama3:latest", base_url: str = None):
        """
        Initialize the AI assistant.
        
        Args:
            model: The Ollama model to use (default: "llama3:latest")
            base_url: Optional custom base URL for Ollama (default: None, uses localhost)
        """
        self.model_name = model
        print(f"🤖 Initializing AI Assistant with model: {model}")
        
        try:
            # Initialize the LLM
            if base_url:
                self.llm = Ollama(model=model, base_url=base_url)
            else:
                self.llm = Ollama(model=model)
            
            # Initialize the search tool
            print("🔍 Initializing search tool...")
            self.search = DuckDuckGoSearchRun()
            
            # Create the prompt template
            self.prompt = ChatPromptTemplate.from_template(
                """You are a helpful AI assistant. You must answer the user's question 
                based *only* on the following search results. If the search results 
                are empty or do not contain the answer, say 'I could not find 
                any information on that.'

                Search Results:
                {context}

                Question:
                {question}
                """
            )
            
            # Build the RAG chain
            print("🔗 Building RAG chain...")
            self.chain = (
                RunnablePassthrough.assign(
                    # "context" is a new key we add to the dictionary.
                    # Its value is the *output* of running the 'search' tool
                    # with the original 'question' as input.
                    context=lambda x: self.search.run(x["question"])
                )
                | self.prompt  # The dictionary (now with 'context' and 'question') is "piped" into the prompt
                | self.llm     # The formatted prompt is "piped" into the LLM
            )
            
            print("✅ Assistant ready!\n")
            
        except Exception as e:
            print(f"❌ Error initializing assistant: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure Ollama is installed and running")
            print(f"2. Verify the model '{model}' is available: ollama pull {model}")
            print("3. Check your internet connection for search functionality")
            raise
    
    def ask(self, question: str) -> str:
        """
        Ask a question to the assistant.
        
        Args:
            question: The user's question
            
        Returns:
            The assistant's response
        """
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            return f"❌ An error occurred while processing your question: {e}"
    
    def run_interactive(self):
        """Run the assistant in interactive mode."""
        print("🤖 Hello! I'm a real-time AI assistant. What's new?")
        print("💡 Type 'exit' or 'quit' to end the conversation.\n")
        
        while True:
            try:
                user_query = input("You: ").strip()
                
                if not user_query:
                    continue
                
                if user_query.lower() in ["exit", "quit", "q"]:
                    print("🤖 Goodbye!")
                    break
                
                print("🤖 Thinking...")
                response = self.ask(user_query)
                print(f"🤖: {response}\n")
                
            except KeyboardInterrupt:
                print("\n🤖 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {e}\n")


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Real-Time AI Assistant Using RAG + LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in interactive mode (default)
  python assistant.py
  
  # Use a different model (you have mistral:7b available)
  python assistant.py --model mistral:7b
  
  # Ask a single question
  python assistant.py --query "What is the weather today?"
  
  # Use custom Ollama URL
  python assistant.py --base-url http://localhost:11434
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama3:latest",
        help="Ollama model to use (default: llama3:latest)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for Ollama (default: localhost)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Ask a single question and exit (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    try:
        assistant = RealTimeAssistant(model=args.model, base_url=args.base_url)
        
        if args.query:
            # Single query mode
            print(f"🤖 Question: {args.query}\n")
            print("🤖 Thinking...")
            response = assistant.ask(args.query)
            print(f"🤖: {response}")
        else:
            # Interactive mode
            assistant.run_interactive()
            
    except Exception as e:
        print(f"❌ Failed to start assistant: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

