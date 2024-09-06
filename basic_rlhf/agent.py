from mirascope.core import groq, prompt_template
from groq import Groq
from groq.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, ConfigDict
from typing import List

from retriever import qdrant_retrieval
from messages_reccomend import retrieve_message
import os
from dotenv import load_dotenv
load_dotenv()

llm = Groq(
    api_key=os.environ["GROQ_API_KEY"],
)

class QdrantRetrieval(BaseModel):
    """
    Busca informacion relacionada a la query del cliente.

    Args:
      query: La query que el cliente ha realizado.

    Returns:
      Un string con la informacion relacionada a la query.
    """
    query: str= Field(..., description="La intencion de busqueda del cliente", examples=["Precio del brunch", "Daypass pizza", "horarios"])

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "precio del brunch"},
                {"query": "daypass pizza"},
                {"query": "horarios"},
            ]
        }
    )

    def call(self) -> str:
      return qdrant_retrieval(self.query)

class Assistant(BaseModel):
    history: List[ChatCompletionMessageParam] = []

    @groq.call(model="llama3-groq-70b-8192-tool-use-preview", stream=True, client=llm, tools=[QdrantRetrieval])
    @prompt_template(
        """
         SYSTEM:
        Eres Marina, encargado del restaurante Rancho Santa Marina.
        Tu tarea es responder preguntas sobre el restaurante.
        Usa la tool `qdrant_retrieval` para obtener la informacion necesaria sobre la pregunta del cliente.
        Responde unicamente con el contexto proveido por `qdrant_retrieval` tool.
        
        Fijate de las respuestas previas que fueron categorizadas como respuestas positivas:
        {templates}

        MESSAGES:
        {self.history}

        USER:
        {question}
        """
    )
    def _step(self, question: str, templates: str): ...

    def run(self, question: str) -> str:
        
        templates = retrieve_message(question)
        stream = self._step(question, templates)
        result, tools_and_outputs = "", []
        for chunk, tool in stream:
            if tool:
                tools_and_outputs.append((tool, tool.call()))
            else:
                result += chunk.content
        if stream.user_message_param:
            self.history.append(stream.user_message_param)
        self.history.append(stream.message_param)
        if tools_and_outputs:
            self.history += stream.tool_message_params(tools_and_outputs)
            return self.run("")
        return result