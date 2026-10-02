import operator
import os

import requests
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated, TypedDict


load_dotenv()

AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


class TravelRequest(BaseModel):
    origin: str | None = Field(
        default=None,
        description="Origin city mentioned by the user.",
    )
    origin_iata: str | None = Field(
        default=None,
        description="IATA code for the origin airport",
    )
    destination: str = Field(
        description="Travel destination mentioned by the user.",
    )
    destination_iata: str | None = Field(
        default=None,
        description="IATA code for the destination airport",
    )
    departure_date: str | None = Field(
        default=None,
        description="Departure date in YYYY-MM-DD format, if provided.",
    )
    duration_days: int | None = Field(
        default=None,
        description="Trip duration in days, if provided.",
    )


class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    travel_request: TravelRequest
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


def query_parser_node(state: TravelState):
    print("Parsing travel request...")

    structured_llm = llm.with_structured_output(TravelRequest)

    travel_request = structured_llm.invoke(
        [
            SystemMessage(
                content=(
                    "Extract structured travel information from the user's query. "
                    "Do not invent missing information. "
                    "If city names are provided, attempt to find their corresponding IATA airport codes. "
                    "Return null for optional fields that are not provided."
                )
            ),
            HumanMessage(content=state["user_query"]),
        ]
    )

    print(f"Parsed Travel Request: {travel_request}")

    return {
        "travel_request": travel_request,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def flight_agent(state: TravelState):
    print("Searching flights...")

    travel_request = state["travel_request"]

    if not travel_request.origin:
        return {
            "flight_results": (
                "Flight search skipped because the origin was not provided."
            )
        }

    params = {
        "access_key": AVIATIONSTACK_API_KEY,
        "dep_iata": travel_request.origin_iata or travel_request.origin,
        "arr_iata": travel_request.destination_iata or travel_request.destination,
        "limit": 5,
    }

    response = requests.get(
        "http://api.aviationstack.com/v1/flights",
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    flight_data = response.json().get("data", [])

    if not flight_data:
        return {
            "flight_results": "No flight information was found."
        }

    flight_results = []

    for flight in flight_data:
        airline = flight.get("airline", {}).get("name", "Unknown airline")
        flight_number = flight.get("flight", {}).get("iata", "N/A")
        departure = flight.get("departure", {}).get("airport", "Unknown")
        arrival = flight.get("arrival", {}).get("airport", "Unknown")

        flight_results.append(
            f"{airline} {flight_number}: {departure} to {arrival}"
        )

    return {
        "flight_results": "\n".join(flight_results)
    }


def hotel_agent(state: TravelState):
    print("Searching hotels...")

    travel_request = state["travel_request"]

    search_query = (
        f"Best hotels in {travel_request.destination}"
    )

    if travel_request.departure_date:
        search_query += (
            f" for travel around {travel_request.departure_date}"
        )

    search_results = tavily_client.search(
        query=search_query,
        max_results=5,
    )

    hotel_results = []

    for result in search_results.get("results", []):
        title = result.get("title", "Hotel information")
        content = result.get("content", "")

        hotel_results.append(
            f"{title}: {content}"
        )

    return {
        "hotel_results": "\n".join(hotel_results)
    }


def itinerary_agent(state: TravelState):
    print("Creating itinerary...")

    travel_request = state["travel_request"]

    prompt = f"""
Create a practical travel itinerary based on the available information.

User Query:
{state["user_query"]}

Destination:
{travel_request.destination}

Trip Duration:
{travel_request.duration_days or "Not specified"}

Flight Results:
{state["flight_results"]}

Hotel Results:
{state["hotel_results"]}

Do not invent flight or hotel availability.
Clearly distinguish suggestions from retrieved information.
"""

    response = llm.invoke(prompt)

    return {
        "itinerary": response.content,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def final_agent(state: TravelState):
    print("Generating final travel response...")

    prompt = f"""
Generate a clear and concise final travel plan.

User Query:
{state["user_query"]}

Flights:
{state["flight_results"]}

Hotels:
{state["hotel_results"]}

Itinerary:
{state["itinerary"]}

Organize the response using clear sections.
Do not claim that flights or hotels are booked.
"""

    response = llm.invoke(prompt)

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def build_graph():
    builder = StateGraph(TravelState)

    builder.add_node("query_parser", query_parser_node)
    builder.add_node("flight_agent", flight_agent)
    builder.add_node("hotel_agent", hotel_agent)
    builder.add_node("itinerary_agent", itinerary_agent)
    builder.add_node("final_agent", final_agent)

    builder.add_edge(START, "query_parser")

    builder.add_edge("query_parser", "flight_agent")
    builder.add_edge("query_parser", "hotel_agent")

    builder.add_edge("flight_agent", "itinerary_agent")
    builder.add_edge("hotel_agent", "itinerary_agent")

    builder.add_edge("itinerary_agent", "final_agent")
    builder.add_edge("final_agent", END)

    return builder.compile()


def main():
    graph = build_graph()

    user_query = (
        # "Plan a 5-day trip from Ahmedabad to Dubai."
        # "Plan a 5-day trip from Ahmedabad to Dubai from August 4"
        "Plan a 5-day trip from Ahmedabad to Mumbai from November 11"
    )

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "llm_calls": 0,
        }
    )

    print("\nTravel Plan\n")
    print(result["messages"][-1].content)

    print(f"\nLLM Calls: {result['llm_calls']}")


if __name__ == "__main__":
    main()