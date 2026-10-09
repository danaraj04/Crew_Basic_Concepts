from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, LLM
llm = LLM(model="gpt-4.1-mini")


reporter = Agent(
		role="Reporter",
		goal="Find key facts about a topic",
		backstory="you gather clear, true facts.",
		llm=llm
	)

editor = Agent(
		role="Editor",
		goal="Turn facts into a short, neat article",
		backstory="you write polished, simple articles.",
		llm=llm
	)
	
find_facts = Task(
        description="Find 3 key facts about electric cars.",
		expected_output="3 short facts.",
		agent=reporter,
		)

write_article = Task(
        description="Use the facts to write a 4-sentence article.",
		expected_output="A short 4-sentence article.",
		agent=editor,
		)

crew =Crew(
      agents=[reporter, editor],
	  tasks=[find_facts,write_article],
	  )

result = crew.kickoff()
print(result)