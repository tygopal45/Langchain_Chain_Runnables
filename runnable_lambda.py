from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda, RunnablePassthrough

from dotenv import load_dotenv

load_dotenv()

def word_counter(txt):
    return len(txt.split())

# runnable_word_counter = RunnableLambda(word_counter)
# print(runnable_word_counter.invoke("Hello world! This is a test."))


llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    input_variables=["question"],
    template="Tell me a joke about {question}."
)

joke_gen_chain = RunnableSequence(prompt1, model1, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_counter)
    # "word_count": RunnableLambda(lambda x: len(x.split()))
})

result = RunnableSequence(joke_gen_chain, parallel_chain)

result = result.invoke("programming")

print(result['joke'])
print(result['word_count'])

