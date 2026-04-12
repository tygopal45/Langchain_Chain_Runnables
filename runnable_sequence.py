from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Write a joke about \n{topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke\n{response}',
    input_variables=['response']
)

parser = StrOutputParser()
chain = RunnableSequence(prompt1, model1, parser, prompt2, model1, parser)

print(chain.invoke({'topic': 'AI'}))

