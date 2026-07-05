# LangGraph Travel Planner

An AI-powered travel planning workflow built with **LangGraph** that demonstrates multi-step LLM orchestration, structured data extraction, parallel external data retrieval, shared graph state, and AI-generated itinerary planning.

The application accepts a natural-language travel request, extracts structured travel information, retrieves flight and hotel information, and generates a consolidated travel itinerary.

## Features

* Natural-language travel request processing
* Structured travel data extraction using LLM output models
* Origin and destination IATA code identification
* Parallel flight and hotel information retrieval
* Flight data integration using Aviationstack
* Hotel discovery using Tavily Search
* AI-generated travel itinerary
* Final response aggregation using an LLM
* Shared workflow state using LangGraph
* LLM invocation tracking

## Workflow Architecture

The workflow follows this topology:

```text
START
  │
  ▼
Query Parser
  ├──────────────▶ Flight Agent ──────┐
  │                                   │
  └──────────────▶ Hotel Agent ───────┤
                                      ▼
                               Itinerary Agent
                                      │
                                      ▼
                                  Final Agent
                                      │
                                      ▼
                                     END
```

The **Query Parser** extracts structured travel details, including `origin_iata` and `destination_iata`.

The workflow then branches into two independent retrieval nodes:

* **Flight Agent** retrieves flight information using the Aviationstack API.
* **Hotel Agent** discovers hotel information using Tavily Search.

Both branches update the shared graph state and converge at the **Itinerary Agent**.

The **Itinerary Agent** uses the combined flight and hotel context to generate a travel itinerary before the **Final Agent** prepares the consolidated response.

The **Flight Agent** and **Hotel Agent** execute as independent workflow branches after the travel request is parsed.

LangGraph merges their state updates before the **Itinerary Agent** generates the travel plan.

## Workflow

### 1. Query Parser

The Query Parser processes the user's natural-language travel request and extracts structured travel information.

Example input:

```text
Plan a 5-day trip from Ahmedabad to Mumbai from July 11.
```

Example structured output:

```json
{
  "origin": "Ahmedabad",
  "origin_iata": "AMD",
  "destination": "Dubai",
  "destination_iata": "BOM",
  "departure_date": "2026-07-11",
  "duration_days": 5
}
```

Optional information is not invented when it is missing from the user request.

### 2. Flight Agent

The Flight Agent uses the extracted `origin_iata` and `destination_iata` codes to retrieve flight information from the Aviationstack API.

The agent retrieves available flight data and converts the API response into a simplified format for downstream workflow nodes.

### 3. Hotel Agent

The Hotel Agent uses Tavily Search to discover hotel information for the requested destination.

Search context can include available travel details such as the destination and departure date.

### 4. Itinerary Agent

The Itinerary Agent combines:

* Original user request
* Structured travel information
* Flight results
* Hotel results

The LLM generates a practical travel itinerary while distinguishing retrieved information from AI-generated recommendations.

### 5. Final Agent

The Final Agent aggregates the flight information, hotel information, and generated itinerary into a clear final travel response.

The response is formatted into readable travel sections and does not claim that flights or hotels have been booked.

## Graph State

The workflow uses a shared LangGraph state.

```python
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    travel_request: TravelRequest
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
```

Each graph node reads the required state fields and returns only its state updates.

## Tech Stack

* Python
* LangGraph
* LangChain Core
* Groq
* Pydantic
* Aviationstack API
* Tavily Search
* Requests

## 🔑 API Keys

This project uses Groq for LLM inference, Aviationstack for flight information, and Tavily for hotel search.

### Groq API Key

1. Visit [GroqCloud Console](https://console.groq.com/).
2. Create an account or sign in.
3. Open the **API Keys** section.
4. Create a new API key.
5. Add the key to your `.env` file:

```dotenv id="f64jwc"
GROQ_API_KEY=your_groq_api_key
```

The application uses the Groq API through `ChatGroq` for structured travel request extraction, itinerary generation, and final response generation.

### Aviationstack API Key

1. Visit [Aviationstack](https://aviationstack.com/).
2. Create an account or sign in.
3. Choose an available API plan.
4. Open your account dashboard and copy your API access key.
5. Add the key to your `.env` file:

```dotenv id="4k5lcs"
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
```

> **Note:** API capabilities and request limits may vary depending on the selected Aviationstack plan.

### Tavily API Key

1. Visit [Tavily](https://www.tavily.com/).
2. Create an account or sign in.
3. Open the dashboard.
4. Generate or copy your API key.
5. Add the key to your `.env` file:

```dotenv id="4tznsv"
TAVILY_API_KEY=your_tavily_api_key
```

The Tavily Python client uses the API key to perform web searches for hotel information.


## Project Structure

```text
langgraph-travel-planner/
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/buntynara/langgraph-travel-planner.git
cd langgraph-travel-planner
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Update `.env` with your API keys:

```dotenv
GROQ_API_KEY=your_groq_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Run the Application

```bash
python main.py
```

The application processes the configured travel query and prints the generated travel plan.

## Key Concepts Demonstrated

This project demonstrates:

* LangGraph graph-based LLM orchestration
* Shared state management across workflow nodes
* Structured LLM output using Pydantic models
* Natural-language to structured-data transformation
* Parallel workflow branches
* External REST API integration
* AI-oriented search integration
* Multi-source context aggregation
* LLM-based itinerary generation
* Separation of retrieval, reasoning, and response-generation stages

## Architecture Evolution

This project intentionally uses direct API and SDK integrations for external travel capabilities.

A future implementation will explore exposing external capabilities through **Model Context Protocol (MCP) servers**, allowing the orchestration layer to interact with travel tools through standardized MCP interfaces.

This provides a clear architecture evolution from direct service integrations to MCP-based tool and agent communication.

## Disclaimer

This project is intended for learning and architecture demonstration purposes.

Flight and hotel information depends on external API and search results. Generated itineraries are AI-assisted recommendations and should be verified before making travel decisions.
