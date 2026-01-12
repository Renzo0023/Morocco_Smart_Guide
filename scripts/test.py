from app.data.loader import load_places

places = load_places()

cities = {}
for p in places:
    c = p.get("city")
    cities[c] = cities.get(c, 0) + 1

print("\nVilles dans load_places():")
for k,v in sorted(cities.items(), key=lambda x: -x[1]):
    print(k, ":", v)

print("\nTotal places:", len(places))
