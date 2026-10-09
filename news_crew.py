import os
import sys
from datetime import date

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Read both keys from environment variables and stop early if one is missing
for key in ("OPENAI_API_KEY", "SERPER_API_KEY"):
    if not os.environ.get(key):
        sys.exit(f"Missing environment variable: {key}")

MODEL = os.environ.get("MODEL", "gpt-4o-mini")
today = date.today().strftime("%d %B %Y")

search = SerperDevTool()  # picks up SERPER_API_KEY from the environment

researcher = Agent(
    role="Tech News Researcher",
    goal=f"Find the 3 most important tech news stories published on {today}",
    backstory=(
        "You are a sharp technology reporter who scans the web every morning. "
        "You only trust reputable outlets, you never invent facts, and you "
        "always check that a story is from today before you include it."
    ),
    tools=[search],
    llm=MODEL,
    verbose=True,
)

writer = Agent(
    role="Tech Newsletter Writer",
    goal="Turn the researcher's 3 stories into a crisp 5-line newsletter",
    backstory=(
        "You are a newsletter editor known for short, punchy writing. "
        "You work only from the research you are given. You never add "
        "stories or facts that are not in it."
    ),
    llm=MODEL,
    verbose=True,
)

find = Task(
    description=(
        f"Today is {today}. Search the web for today's top tech news and "
        "pick the 3 most significant stories. Prefer stories published "
        "today or in the last 24 hours, from reputable sources."
    ),
    expected_output=(
        "A numbered list of exactly 3 stories. Each item has three parts: "
        "Headline, a 1-2 sentence Summary, and the Source name with its URL. "
        "No introduction and no closing remarks."
    ),
    agent=researcher,
)

write = Task(
    description=(
        "Using only the 3 stories from the researcher, write a tech "
        "newsletter. Do not add any story or fact that is not in the research."
    ),
    expected_output=(
        "Exactly 5 lines of plain text, with no blank lines, no markdown "
        "and nothing before or after:\n"
        "Line 1: the title 'Tech in 5 lines'\n"
        "Line 2: '1. ' followed by the first story in one line\n"
        "Line 3: '2. ' followed by the second story in one line\n"
        "Line 4: '3. ' followed by the third story in one line\n"
        "Line 5: 'Why it matters: ' followed by one line tying the stories together"
    ),
    agent=writer,
    context=[find],  # hands the Researcher's output to the Writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[find, write],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n" + str(result))