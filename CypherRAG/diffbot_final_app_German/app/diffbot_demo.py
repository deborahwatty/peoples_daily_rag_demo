import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts import PromptTemplate

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI_LOCAL')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_LOCAL')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_LOCAL')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE_LOCAL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', False)
DIFFBOT_API_KEY = os.getenv('DIFFBOT_API_KEY')


translated_prompt = PromptTemplate(template="""Du bist ein Assistent, der hilft, freundliche und für Menschen verständliche Antworten zu formulieren.
Der Informationsteil enthält die bereitgestellten Informationen, die du verwenden sollst, um eine Antwort zu generieren.
Die bereitgestellten Informationen sind maßgebend. Du darfst sie niemals anzweifeln oder versuchen, dein internes Wissen zu nutzen, um sie zu korrigieren.
Auch wenn die bereitgestellten Informationen auf Englisch sind, solltest du auf Deutsch antworten. 
Die Antwort sollte vom Ton her einer direkten Antwort auf die Frage entsprechen. Erwähne nicht, dass du die Antwort auf Grundlage der gegebenen Informationen generiert hast.

Hier ist ein Beispiel:

Frage: Welche Manager besitzen Neo4j-Aktien?
Kontext: [Manager:CTL LLC, Manager:JANE STREET GROUP LLC]
Hilfreiche Antwort: CTL LLC, JANE STREET GROUP LLC besitzen Neo4j-Aktien.

Halte dich an dieses Beispiel, wenn du Antworten erstellst.
Falls die bereitgestellten Informationen leer sind, sage, dass du es nicht weißt.

Informationen:
{context}

Frage: {question}
Hilfreiche Antwort:""")

prompt_template2 = """Task:Generate Cypher statement to query a graph database.\nInstructions:\nUse only the provided relationship types and properties in the schema.\nDo not use any other relationship types or properties that are not provided.\nSchema:\n{schema}\nNote: Do not include any explanations or apologies in your responses.\nDo not respond to any questions that might ask anything else than for you to construct a Cypher statement.\nDo not include any text except the generated Cypher statement. \n\nThe question is:\n{question}'"""

translated_prompt_2 = PromptTemplate(template="""Task:Generate Cypher statement to query a graph database.\nInstructions:\nUse only the provided relationship types and properties in the schema.\nDo not use any other relationship types or properties that are not provided.\nSchema:\n{schema}\nNote: Do not include any explanations or apologies in your responses.\nDo not respond to any questions that might ask anything else than for you to construct a Cypher statement.\nDo not include any text except the generated Cypher statement. Note that the question may be in German, while the database is in English. Please translate any relevant named entities etc. in the cypher. \n\nThe question is:\n{question}""")



diffbot_neo4j_graph_with_sources = Neo4jGraph(
   url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database="final2020" #local
   #url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database="neo4j"
)

diffbot_neo4j_graph_with_sources.refresh_schema()

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4o", temperature=0), graph=diffbot_neo4j_graph_with_sources, verbose=True, allow_dangerous_requests=True, qa_prompt=translated_prompt, cypher_prompt=translated_prompt_2
)

chain.invoke({"query": "What is the nature of the relationship between Peng Liyuan and Donald Trump?"})