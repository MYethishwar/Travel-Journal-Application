
from agents import run_travel_agents

print("=" * 60)
print("  TRAVEL MULTI-AGENT SYSTEM — CLI TEST")
print("=" * 60)

# You can change this query to anything
query = "Plan a 10 days trip to Aurangabad, Maharashtra, India. Include historical places and popular places."

print(f"\nQuery: {query}\n")
print("-" * 60)

result = run_travel_agents(query)

print("\n" + "=" * 60)
print("DESTINATION DETECTED:", result["destination"])
print("=" * 60)

print("\n📍 RESEARCH:")
print(result["research"])

print("\n🗓️  ITINERARY:")
print(result["itinerary"])

print("\n💰 BUDGET:")
print(result["budget"])

print("\n✨ FINAL SUMMARY:")
print(result["summary"])

print("\n" + "=" * 60)
print("✅ Done! Check LangSmith dashboard for full trace.")
print("   https://smith.langchain.com")
print("=" * 60)