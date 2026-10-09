from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
facts="""Our company is called BlueShoe.
We were founded in 2020.
We sell eco-friendly shoes made from recycled plastic."""
knowledge = StringKnowledgeSource(content=facts)

print("knowledge source ready!") 

llm = LLM(model="gpt-4.1-mini")
print("OpenAi brain is ready!")

#from crewai import Agent
assistant = Agent(
			role="Company Helper",
			goal="Answer using only company knowledge",
			backstory="You know all about BlueShoe.",
			llm=llm,
			)

task = Task(
			description="What does our company sell, and when were we founded?",
			expected_output="A short answer from the company facts.",
			agent=assistant,
			)

print("Agent and task ready!") 

crew = Crew(agents=[assistant],
             tasks=[task],
             knowledge_sources=[knowledge],)
result = crew.kickoff()
print(result)
print("End RAG crew!") 