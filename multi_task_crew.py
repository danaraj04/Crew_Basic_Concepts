from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, LLM
llm = LLM(model="gpt-4.1-mini")

writer = Agent(
		role="Content Writer",
		goal="Create simple, clear content",
		backstory="you write friendly content for beginners.",
		llm=llm
	)


task1 = Task(
        description="List 3 fun facts about the moon.",
		expected_output="3 short fun facts.",
		agent=writer,
		)

task2 = Task(
        description="Use the facts above to write one short tweet.",
		expected_output="One short tweet.",
		agent=writer,
		)

crew =Crew(
      agents=[writer],
	  tasks=[task1,task2],
	  )
result = crew.kickoff()
print(result)