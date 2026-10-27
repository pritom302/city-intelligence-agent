# 🌆 City Intelligence

A Streamlit app where an AI agent answers questions about the **current weather** and **latest local news** for any city. Built with LangChain, Mistral AI (`codestral-latest`), OpenWeatherMap and Tavily.

## Features

- Chat interface with an AI agent that decides which tool to use
- Weather cards that change colour and icon with the conditions
- News results shown as clickable cards with a snippet and source
- Optional approval step: confirm or block each tool call before it runs
- Sidebar showing whether your API keys were found

## Tech stack

- [Streamlit](https://streamlit.io) for the UI
- [LangChain](https://python.langchain.com) and LangGraph for the agent
- [Mistral AI](https://mistral.ai) (`codestral-latest`) as the language model
- [OpenWeatherMap](https://openweathermap.org/api) for weather data
- [Tavily](https://tavily.com) for news search

## Project structure

```
.
├── app.py            # Streamlit UI and agent wiring
├── city_tools.py     # get_weather and get_news tools
├── requirements.txt
├── .env.example      # template for your API keys
└── .gitignore
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/city-intelligence-agent.git
   cd city-intelligence-agent
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux
   pip install -r requirements.txt
   ```

3. **Add your API keys**

   Copy `.env.example` to `.env` and fill in your keys:

   ```
   MISTRAL_API_KEY=your_key_here
   OPENWEATHER_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

## Usage

Try questions like:

- "What's the weather in Habiganj, Bangladesh?"
- "What's the latest news in Sylhet?"
- "Give me the weather and latest news for Dhaka."

Turn off **Ask before running tools** in the sidebar if you don't want to approve each lookup.

## Notes

- The free Mistral plan has a low rate limit. If you see a rate limit message, wait about a minute and try again.
- `codestral-latest` is tuned for code, so it sometimes answers without calling a tool. Mentioning the city and the word "weather" or "news" in your question helps.
- Never commit your `.env` file. It is listed in `.gitignore`.

## License

Add a license of your choice (for example MIT).
