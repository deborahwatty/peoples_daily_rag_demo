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
from langchain.prompts import PromptTemplate

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI_LOCAL')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME_LOCAL')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD_LOCAL')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE_LOCALL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DIFFBOT_API_KEY = os.getenv('DIFFBOT_API_KEY')

app = FastAPI()


prompt_template = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question. Do not mention that you based the result on the given information.
Here is an example:

Question: Which managers own Neo4j stocks?
Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers.
If the provided information is empty, say that you don't know the answer.
Information:
{context}

Question: {question}
Helpful Answer:"""


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
    ChatOpenAI(model="gpt-4o", temperature=0), graph=diffbot_neo4j_graph_with_sources, verbose=True, allow_dangerous_requests=True, qa_prompt=translated_prompt, cypher_prompt=translated_prompt_2
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
