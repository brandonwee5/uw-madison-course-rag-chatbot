"""
UW-Madison Course Analytics RAG Chatbot
Main entry point for the application
"""

from src.chatbot import run_chatbot

def main():
    print("=" * 50)
    print("UW-Madison Course Analytics Chatbot")
    print("=" * 50)
    print("\nAsk questions about course grades and GPAs!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    run_chatbot()

if __name__ == "__main__":
    main()
