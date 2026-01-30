from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
)


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


response = client.responses.parse(
    model="openai/gpt-oss-20b",
    input="Alexandre e Paolla vão ao cinema na segunda feira",
    instructions="Extraia informações do evento",
    text_format=CalendarEvent,
    temperature=0,
    top_p=1,
)

print(response.output_parsed)
print(response.output_parsed.participants)
