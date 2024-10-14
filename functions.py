import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def transform_raw_data(path_to_csv):
    df = pd.read_csv(path_to_csv, header=0)
    df = df.rename(columns={
        'longitude.x': 'longitude',
        'latitude.x': 'latitude',
        'date.x': 'date'
    })

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

    # creating date and time features
    df_expanded['time_period'] = (df_expanded.time.str.slice(start=0, stop=2).astype('int') // 4)
    df_expanded['time_period'] = (df_expanded['time_period'] * 4).astype('str') + ':00 - ' + ((df_expanded['time_period'] + 1) * 4).astype('str') + ':00'

    df_expanded['date'] = pd.to_datetime(df_expanded.date)
    df_expanded['month'] = df_expanded['date'].dt.month
    df_expanded['season'] = [
        'spring' if month in [3, 4, 5]
        else 'summer' if month in [6, 7, 8]
        else 'autumn' if month in [9, 10, 11]
        else 'winter'
        for month
        in df_expanded['month']
    ]

    df_expanded = df_expanded.drop(columns=['date', 'month'])

    return df_expanded

def clean_df(df):
    # standardise missing values
    missing_values = ['Data missing or out of range', 'unknown (self reported)', np.nan, 'Unknown', 'Not known', 'Undefined']
    df = df.replace({i: "Missing" for i in missing_values})

    drop_columns = [
        'accident_index_2',
        'lsoa_of_casualty', 'lsoa_of_driver', 
        'age_band_of_casualty', 'age_band_of_driver',
        'special_conditions_at_site', 'carriageway_hazards', 'skidding_and_overturning', 'hit_object_in_carriageway', 'hit_object_off_carriageway', 'journey_purpose_of_driver', 'vehicle_leaving_carriageway',
        'age_of_vehicle'
    ]
    drop_columns = [col for col in drop_columns if col in df.columns]
    if len(drop_columns) > 0:
        df = df.drop(columns=drop_columns)

    # categorise speed_limit
    if 'speed_limit' in df.columns:
        df['speed_limit'] = df['speed_limit'].astype('object')

    # convert numeric to continuous
    for col in ['engine_capacity_cc', 'age_of_casualty', 'age_of_driver']:
        if col in df.columns:
            df[col] = df[col].replace('Missing', np.nan).astype('float')

    # does not make sense for propulsion_code to have a value for cyclists
    if 'propulsion_code' in df.columns:
        df["propulsion_code"] = np.where(df["vehicle_type"] == "Pedal cycle", "Undefined", df["propulsion_code"])

    return df

def impute_fit_df(df):
    # calculate category distributions of each categorical variable
    df = df.replace("Missing", np.nan).reset_index(drop=True)

    categorical_freqs = {}
    for col in df.select_dtypes(include='object').columns:
        freq_dict = df[col].value_counts(normalize=True).to_dict()
        categorical_freqs[col] = freq_dict

    # imputing continuous variables
    # creating new vehicle_type column because some values of vehicle_type have no values for engine_capacity_cc -> using other types of vehicle that have the most similar engine size
    if "vehicle_type" in df.columns:
        df["vehicle_type_2"] = [
            "Goods over 3.5t. and under 7.5t" if vehicle_type in ["Agricultural vehicle", "Goods vehicle - unknown weight"]
            else "Motorcycle 50cc and under" if vehicle_type == "Electric motorcycle"
            else "Motorcycle 125cc and under" if vehicle_type == "Motorcycle - unknown cc"
            else "Car" if vehicle_type == "Unknown vehicle type (self rep only)"
            else vehicle_type
            for vehicle_type in df["vehicle_type"]
        ]

    vars_to_groupby = {
        "engine_capacity_cc": "vehicle_type_2",
        "age_of_casualty": "towing_and_articulation",
        "age_of_driver": "vehicle_type"
    }
    continuous_medians_grouped = {}
    continuous_medians = {}

    for col in vars_to_groupby:
        if col in df.columns:
            impute_value = df[~df[col].isna()][col].median()
            continuous_medians[col] = impute_value

            if vars_to_groupby[col] in df.columns:
                lookup_df = df.groupby(vars_to_groupby[col])[col].median()
                continuous_medians_grouped[col] = lookup_df
            else:
                continuous_medians_grouped[col] = None

    # fit scaler
    continuous_vars = [col for col in list(df.select_dtypes(exclude='object').columns) if col not in ['longitude', 'latitude']]
    if len(continuous_vars) > 0:
        scaler = MinMaxScaler()
        scaler.fit(df[continuous_vars])
    else:
        scaler = None

    return categorical_freqs, vars_to_groupby, continuous_medians_grouped, continuous_medians, scaler

def impute_transform_df(df, categorical_freqs, vars_to_groupby, continuous_medians_grouped,
                        continuous_medians, scaler, select_features, encoded_cols_dict=None,
                        one_hot_encode=True):
    missing = pd.DataFrame([i for i in zip(df.columns, df.dtypes, df.nunique(), 100 * ((df == "Missing") | (df.isnull())).mean()) if i[3] > 0], columns=['column', 'dtype', 'nunique', 'missing %']).sort_values("missing %")

    # categorical missing values imputed while keeping the category distributions of each variable the same
    categorical_missing = list(missing[missing['dtype'] == 'object']['column'])
    df = df.replace("Missing", np.nan).reset_index(drop=True)

    for col in categorical_missing:
        freq_dict = categorical_freqs[col]
        df[col] = df[col].fillna(pd.Series(np.random.choice(list(freq_dict.keys()), p=list(freq_dict.values()), size=len(df))))

    # imputing continuous variables
    # creating new vehicle_type column because some values of vehicle_type have no values for engine_capacity_cc -> using other types of vehicle that have the most similar engine size
    if "vehicle_type" in df.columns:
        df["vehicle_type_2"] = [
            "Goods over 3.5t. and under 7.5t" if vehicle_type in ["Agricultural vehicle", "Goods vehicle - unknown weight"]
            else "Motorcycle 50cc and under" if vehicle_type == "Electric motorcycle"
            else "Motorcycle 125cc and under" if vehicle_type == "Motorcycle - unknown cc"
            else "Car" if vehicle_type == "Unknown vehicle type (self rep only)"
            else vehicle_type
            for vehicle_type in df["vehicle_type"]
        ]

    # first imputation should capture most nulls
    # second imputation should capture any remaining nulls (remaining because there were no values in the lookup group)
    def impute_continuous_vars(df, impute_lookup, grouped=False):
        for var in impute_lookup:
            if var in df.columns:
                if grouped and vars_to_groupby[var] in df.columns:
                    lookup_df = impute_lookup[var]
                    df = pd.merge(df, lookup_df, how="left", on=vars_to_groupby[var], suffixes=["", "_y"])
                    df[var] = np.where(df[var].isna(), df[var + '_y'], df[var])
                    df = df.drop(columns=[var + '_y'])
                else:
                    impute_value = impute_lookup[var]
                    df[var] = df[var].replace(np.nan, impute_value)

        return df

    df_1 = impute_continuous_vars(df, continuous_medians_grouped, grouped=True)
    df_2 = impute_continuous_vars(df_1, continuous_medians)
    df = df_2

    # "Pedal cycle" has no values for engine_capacity_cc for obvious reasons
    # assume average cyclist can push 100W ≈ 0.13 horsepower -> horsepower of standard car ~200 -> cyclist horsepower 0.2/200=0.00065 of a car -> set engine_capacity_cc of "Pedal cycle" to 0.00125 that of a car
    if 'engine_capacity_cc' in df.columns:
        try:
            engine_capacity_car = continuous_medians_grouped["engine_capacity_cc"].loc["Car"]
        except:
            engine_capacity_car = continuous_medians["engine_capacity_cc"]
        df["engine_capacity_cc"] = np.where(df["vehicle_type"] == "Pedal cycle", 0.00065 * engine_capacity_car, df["engine_capacity_cc"])

    if "vehicle_type_2" in df.columns:
        df = df.drop(columns=["vehicle_type_2"])

    # check there are no more missing values
    assert sum(((df.isna()) | (df == "Missing")).any()) == 0

    # select features
    df = df[select_features]

    # apply transformations
    continuous_vars = [col for col in list(df.select_dtypes(exclude='object').columns) if col not in ['longitude', 'latitude']]
    if scaler != None:
        df[continuous_vars] = scaler.transform(df[continuous_vars])

    if one_hot_encode:
        categorical_vars = [col for col in list(df.select_dtypes(include='object').columns) if col not in ['date', 'time', 'casualty_severity']]
        df = pd.get_dummies(df, columns=categorical_vars, drop_first=False)

    # add empty columns for one-hot encoded columns that are missing
    # sklearn requires that train and test data has the same columns
    if encoded_cols_dict != None and one_hot_encode:
        encoded_cols = []
        for col in select_features:
            if col in encoded_cols_dict:
                encoded_cols.extend(encoded_cols_dict[col])

        missing_encoded_cols = [col for col in encoded_cols if col not in df.columns]
        df[missing_encoded_cols] = np.zeros((len(df), len(missing_encoded_cols)))
        df[missing_encoded_cols] = df[missing_encoded_cols].astype('bool')

    # sklearn requires that train and test data has same column order
    cols = sorted(list(df.columns))
    df = df[cols]

    return df