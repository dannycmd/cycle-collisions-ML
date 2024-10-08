import pandas as pd

df = pd.read_csv("stats19CycleCollisions2022.csv", header=0)
df = df.rename(columns={
    'longitude.x': 'longitude',
    'latitude.x': 'latitude',
    'date.x': 'date'
})

# determining number of vehicles involved in each collision - assume this will be 2 the majority of the time
accident_summary = df.groupby("accident_index").agg({"number_of_vehicles": "max",
                                                     "vehicle_reference": pd.Series.nunique,
                                                     "casualty_reference": pd.Series.nunique})

# cases where number_of_vehicles not equal to vehicle count
test1 = accident_summary[accident_summary['number_of_vehicles'] != accident_summary['vehicle_reference']]
# -> no cases

# cases where there is more than 1 casualty
test2 = accident_summary[accident_summary['casualty_reference'] > 1]
# -> 222 cases
# examples include:
#  - 1 bike and no other vehicle, 2 casualties 
#  - bike and motorbike, 2 casualties
#  - bike, motorbike and car, 2 casualties from bike and motorbike
#  - 2 bikes, 2 casualties
#  - 2 bikes and van, 2 casualties
# First, for collisions where there is more than 1 bike casualty, I will create a separate collision for each bike casualty
# Then for collisions where there is more than 1 vehicle, I will take only one vehicle based on the following hierarchy:
#    1. HGV
#    2. Agricultural vehicle
#    3. Bus
#    4. Van
#    5. Car
#    6. Motorbike

# creating vehicle subtype hierarchy
df['vehicle_subtype'] = [
    "4. Van" if vehicle_type[0] in ["VAN / GOODS 3.5 TONNES MGW OR UNDER", "MINIBUS (8 - 16 PASSENGER SEATS)"]
    else "1. HGV" if "GOODS" in vehicle_type[0]
    else "2. Agricultural vehicle" if vehicle_type[0] == "AGRICULTURAL VEHICLE"
    else "3. Bus" if vehicle_type[0] == "BUS OR COACH (17 OR MORE PASS SEATS)"
    else "5. Car" if "CAR" in vehicle_type[0]
    else "6. Motorbike" if "MOTORCYCLE" in vehicle_type[0]
    else "7. Bike" if vehicle_type[0] == "PEDAL CYCLE"
    else "8. Unknown"
    for vehicle_type in zip(df['vehicle_type'].str.upper())
]

# creating a separate collision for each bike casualty
casualty_vars = [
    'longitude', 
    'latitude', 
    'date',
    'day_of_week',
    'time', 
    'first_road_class',
    'road_type', 
    'speed_limit',
    'junction_detail', 
    'junction_control', 
    'second_road_class',
    'pedestrian_crossing_human_control', 
    'pedestrian_crossing_physical_facilities', 
    'light_conditions',
    'weather_conditions', 
    'road_surface_conditions',
    'special_conditions_at_site', 
    'carriageway_hazards',
    'urban_or_rural_area',
    'RH2M', 
    'T2M', 
    'PRECTOTCORR',
    'WS2M',
    'sex_of_casualty', 
    'age_of_casualty', 
    'age_band_of_casualty',
    'casualty_severity',
    'casualty_home_area_type', 
    'casualty_imd_decile', 
    'lsoa_of_casualty'
]
vehicle_vars = [
    'vehicle_type', 
    'vehicle_subtype',
    'towing_and_articulation',
    'vehicle_manoeuvre',
    'vehicle_location_restricted_lane', 
    'junction_location',
    'skidding_and_overturning', 
    'hit_object_in_carriageway',
    'vehicle_leaving_carriageway', 
    'hit_object_off_carriageway',
    'first_point_of_impact', 
    'vehicle_left_hand_drive',
    'engine_capacity_cc', 
    'propulsion_code',
    'age_of_vehicle'
]
driver_vars = [
    'journey_purpose_of_driver', 
    'sex_of_driver', 
    'age_of_driver',
    'age_band_of_driver',
    'driver_imd_decile',
    'driver_home_area_type', 
    'lsoa_of_driver'
]
unique_casualties = df[(df['vehicle_type'] == 'Pedal cycle') & ~pd.isnull(df['casualty_reference'])][['accident_index', 'casualty_reference'] + casualty_vars].drop_duplicates().reset_index(drop=True)
unique_casualties['accident_index_2'] = unique_casualties.index

df_expanded = pd.merge(unique_casualties, df[['accident_index', 'casualty_reference', 'number_of_vehicles'] + vehicle_vars + driver_vars], how="inner", on="accident_index", suffixes=["", "_y"])
df_expanded = df_expanded[((df_expanded['casualty_reference'] != df_expanded['casualty_reference_y']) & (df_expanded['number_of_vehicles'] > 1)) | (df_expanded['number_of_vehicles'] == 1)]\
    .drop(columns=['casualty_reference', 'casualty_reference_y', 'accident_index', 'number_of_vehicles'])

# keep 1 vehicle per collision, based on hierarchy
df_expanded = df_expanded.sort_values(['accident_index_2', 'vehicle_subtype'])
df_expanded = df_expanded.groupby('accident_index_2').nth(0)

# write to csv
df_expanded.to_csv("transformed_data.csv", index=False)