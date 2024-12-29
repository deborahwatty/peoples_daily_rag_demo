import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import GraphCypherQAChain

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI_LOCAL')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_LOCAL')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_LOCAL')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE_LOCAL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', False)


diffbot_neo4j_graph_with_sources = Neo4jGraph(
   url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database="final2020" #local
   #url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database="neo4j"
)

diffbot_neo4j_graph_with_sources.refresh_schema()

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4o", temperature=0), graph=diffbot_neo4j_graph_with_sources, verbose=True, allow_dangerous_requests=True
)

chain.invoke({"query": "What is the nature of the relationship between Peng Liyuan and Donald Trump?"})