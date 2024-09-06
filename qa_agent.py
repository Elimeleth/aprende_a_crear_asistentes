from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from retriever import qdrant_retrieval
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from messages_reccomend import retrieve_message
import datetime

from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responde unicamente con el contexto proveido\n\nContexto:\n{context}"),
    ("human", """
Fijate de las respuestas previas que fueron categorizadas como respuestas positivas:
{templates}

Usa esa informacion para generar respuestas acordes respondiendo solo la informacion solicitada.
     """),
    ("human", """Manten presente la fecha actual para dar mejores respuestas sobre horarios y temas particulares
fecha actual: {date}""".format(date=datetime.datetime.now())),
    ("human", "user input: {input}")
])
llm = ChatGroq(
    model="llama3-70b-8192"
    )

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def chat(input: str):
    templates = retrieve_message(input)
    docs = qdrant_retrieval(input)
    context = "\n".join(list(map(lambda doc: doc.page_content, docs)))
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({ "input": input, "templates": templates, "context": context })