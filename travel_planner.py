
#import os
import sys
from dotenv import load_dotenv

load_dotenv()
#print("Setup done! CrewAI can now use your OpenAI Key")

from crewai import Agent, Task, Crew
from crewai import LLM
llm = LLM(model="gpt-4.1-mini")

city = input("Which city? ").strip()
if not city:
    sys.exit("Please enter a city name.")
 
planner = Agent(
    role="Travel Planner",
    goal=(
        "Suggest the three places in a city that a first-time visitor "
        "should not miss, each with one practical reason to go."
    ),
    backstory=(
        "You have spent fifteen years as a local guide, walking visitors "
        "through cities around the world. You write for first-time travellers "
        "who have only a day or two and no patience for long guidebooks. "
        "You pick real, well-known places, you never invent one, and you "
        "always add a short, useful tip such as the best time to go."
    ),
    llm=llm,
    verbose=True,
)
 
task = Task(
    description=(
        "Suggest exactly 3 places to visit in {city}. "
        "Choose places that really exist in {city} and that differ from each "
        "other, for example one historic site, one landmark and one museum or "
        "market. For each place give its name and a short note on what makes "
        "it worth visiting, with a practical tip where it helps."
    ),
    expected_output=(
        "A numbered list of exactly 3 lines and nothing else: no heading, "
        "no introduction, no closing remark, no bold or other markdown. "
        "Each line follows this shape: "
        "'<number>. <Place name> - <one short phrase of at most 12 words>'. "
        "Example: '1. Amber Fort - hilltop fort with mirrored halls, go early'."
    ),
    agent=planner,
)
 
crew = Crew(agents=[planner], tasks=[task])
 
result = crew.kickoff(inputs={"city": city})
print(result)