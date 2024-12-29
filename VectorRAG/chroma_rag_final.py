import sys
import importlib
import os
import dotenv

# Force the use of pysqlite3 instead of the standard sqlite3
#importlib.import_module('pysqlite3')
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now, import other modules
import sqlite3

print("SQLite version being used:", sqlite3.sqlite_version)

import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
import argparse

class TextRetrieval:
    def __init__(self, openai_api_key, vector_type, query_language="English"):
        self.openai_api_key = openai_api_key
        self.vector_type = vector_type
        self.query_language = query_language
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)
        self.vectorstore = self.load_vectors(vector_type)

    def load_vectors(self, vector_type):
        if vector_type == "zh_vectors":
            persist_directory = 'chroma_chinese'
        else:
            persist_directory = 'chroma_english'
        return Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)

    def find_similar_chunks(self, query):
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 9})
        print("Looking for similar articles...")
        results = retriever.invoke(query)
        return results, [result.page_content for result in results]

    def create_prompt(self, query):
        complete_chunks, chunks = self.find_similar_chunks(query)
        print(f"Found {len(chunks)} articles")
        prompt_template = PromptTemplate.from_template("""
Query: {query}

Texts: 
1. {chunk_1}

2. {chunk_2}

3. {chunk_3}""")
        prompt = prompt_template.format(query=query, chunk_1=chunks[0], chunk_2=chunks[1], chunk_3=chunks[2])
        return prompt, [doc.metadata['filename'] for doc in complete_chunks]

    def answer_question(self, query, prompt=None, filenames=None):
        if prompt is None:
            prompt, filenames = self.create_prompt(query)


        if self.query_language == "German":
            system_message = "Du bist ein hilfsbereiter KI-Assistent, der basierend auf Zeitungsausschnitten Nutzerfragen beantwortet. In jedem Prompt wird dir eine Frage gestellt und eine Liste mit Textabschnitten zur Verfügung gestellt. Bitte beantworte die Frage ausschließlich auf Grundlage der Texte. Falls die Texte nur teilweise relevant für die Fragestellung sind, versuche bitte, die Frage teilweise oder allgemein zu beantworten. In dem Fall, dass die Texte keinerlei hilfreiche Informationen beinhalten, antworte bitte mit 'nicht zutreffend'."
            if self.vector_type == 'zh_vectors':
                system_message += "Beantworte die Frage bitte auf Deutsch, auch wenn die Texte auf Chinesisch geschrieben sind."
        else:
            system_message = "You are a helpful assistant that answers user questions based on given texts taken from newspaper articles. You will be given a query and a list of chunks of text. Please answer the query based only on the contents of the texts. If texts are only marginally related to the question, try to give a partial or general answer. However, if the texts do not help you answer the question at all, please answer with 'none'."
            if self.vector_type == 'zh_vectors':
                system_message += "Please answer the question in English even though the texts are in Chinese."

        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0
        )
        return prompt, response, response.choices[-1].message.content, filenames


def main():
    parser = argparse.ArgumentParser(description="Process a query and specify vector type.")
    parser.add_argument("--zh_vectors", action="store_true", help="Use Chinese vectors")
    parser.add_argument("--en_vectors", action="store_true", help="Use English vectors")
    parser.add_argument("--language", type=str, default="English", help="Specify the query language (default: English)")
    args = parser.parse_args()


    load_dotenv()
    
    if args.zh_vectors and args.en_vectors:
        raise ValueError("Please specify only one of --zh_vectors or --en_vectors.")
    elif not args.zh_vectors and not args.en_vectors:
        raise ValueError("Please specify one of --zh_vectors or --en_vectors.")

    vector_type = "zh_vectors" if args.zh_vectors else "en_vectors"
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the TextRetrieval class
    text_retrieval = TextRetrieval(openai_api_key, vector_type, query_language=args.language)

    query = input(f"Please enter your query in {args.language}: ")
    prompt, model_response, response_text, filenames = text_retrieval.answer_question(query)

    # Output response and filenames
    print(response_text + '\n' + str(filenames))


if __name__ == "__main__":
    main()
