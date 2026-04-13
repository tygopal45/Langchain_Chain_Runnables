from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a short tweet about a topic in english.\n{topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a short linkedIn post about a topic in english.\n{topic}",
    input_variables=["topic"]
)

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model1, parser),
    'linkedIn' : RunnableSequence(prompt2, model1, parser)
})

# we get a dictionary with 2 keys 'tweet' and 'linkedIn' with the respective outputs

result = parallel_chain.invoke({'topic': 'AI'})
print(result['tweet'])
print(result['linkedIn'])
