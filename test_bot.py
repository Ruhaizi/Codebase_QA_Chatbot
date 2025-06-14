from qa_chain import ask_question

question = "How does the authentication flow work in this codebase?"
answer = ask_question(question)
print(answer)
