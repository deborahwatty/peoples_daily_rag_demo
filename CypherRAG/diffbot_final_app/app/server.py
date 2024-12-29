from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os
from langserve import CustomUserType
import uvicorn

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI_LOCAL')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_LOCAL')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_LOCAL')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE_LOCAL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

#prompt = ChatPromptTemplate.from_template("Tell me a story about {topic}")
#model = ChatOpenAI()

diffbot_neo4j_graph_with_sources = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database="final2020"
    #url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database="neo4j"
)

diffbot_neo4j_graph_with_sources.refresh_schema()


class Question(CustomUserType):
    question: str


chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-4o", temperature=0), graph=diffbot_neo4j_graph_with_sources, verbose=True, allow_dangerous_requests=True
).with_types(input_type=Question)

# Edit this to add the chain you want to add
#add_routes(app,
#           prompt | model,
#           path = '/story'
#           )

add_routes(app,
           chain,
           path = '/diffbotdemo'
           )

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8080)
