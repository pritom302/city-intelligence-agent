from dotenv import load_dotenv
load_dotenv()

from main import get_weather

result = get_weather.invoke({"city": "Dhaka"})
print(result)