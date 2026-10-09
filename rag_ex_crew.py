from dotenv import load_dotenv
load_dotenv()

from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

facts = """Our company is called BlueShoe.
We were founded in 2019.
We sell eco-friendly shoes made from recycled plastic."""

knowledge = StringKnowledgeSource(content=facts)
print("Knowledge source ready!")

from crewai import Agent, Task, Crew, LLM

llm = LLM(model="gpt-4o-mini")

agent = Agent(
    role="Company Helper",
    goal="Answer using only company knowledge",
    backstory="You know all about BlueShoe.",
    llm=llm,
)

task = Task(
    description="What does our company sell, and when were we founded?",
    expected_output="A short answer from the company facts.",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    knowledge_sources=[knowledge],   # <-- attach your knowledge
)

# In a Jupyter notebook, use kickoff_async() with "await".
result = crew.kickoff_async()
print(result)