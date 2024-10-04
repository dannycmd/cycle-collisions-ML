import pandas as pd

min_lat = 51.1
max_lat = 51.2
min_lon = 0
max_lon = 0.3

df = pd.read_csv("dft_traffic_counts_raw_counts.csv", header=0)
test = df[['latitude', 'longitude']].drop_duplicates()
test['a'] = ""
test = test[(test["latitude"] >= min_lat) & (test["latitude"] <= max_lat) & (test["longitude"] >= min_lon) & (test["longitude"] <= max_lon)]
test.to_csv("traffic_count_locations.csv", index=False)