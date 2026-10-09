from dotenv import load_dotenv

load_dotenv()
#print("Setup done! CrewAI can now use your OpenAI Key")

from crewai import Agent, Task, Crew, LLM

llm = LLM(model="gpt-4.1-mini")
#print("OpenAi brain is ready!")

from crewai_tools import  SerperDevTool
search_tool = SerperDevTool()

#from crewai import Agent
researcher = Agent(
			role="Researcher",
			goal="Find current information from the web",
			backstory ="You search the internet for fresh facts.",
			tools=[search_tool],
			llm=llm,
		)
print("Researcher with web search is ready!")
task = Task(
			description="Search the web andtell me one recent news headline about AI.",
			expected_output="One recent headline with a short note.",
			agent=researcher,
			)
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
print(result)
