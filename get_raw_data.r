# install.packages("stats19")
# install.packages("nasapower")
library(stats19)
library(dplyr)
library(tidyr)
library(stringr)
library(purrr)
library(nasapower)

years <- 2019:2022

# get_stats19 function downloads a csv of the requested data and cleans it
accidents <- get_stats19(year = years[1], type = "accidents", ask = FALSE, format = FALSE)
vehicles <- get_stats19(year = years[1], type = "vehicle", ask = FALSE, format = FALSE)
casualties <- get_stats19(year = years[1], type = "casualty", ask = FALSE, format = FALSE)

for (year in years[-1]) {
  accidents_append <- get_stats19(year = year, type = "accidents", ask = FALSE, format = FALSE)
  accidents <- bind_rows(accidents, accidents_append)

  vehicles_append <- get_stats19(year = year, type = "vehicle", ask = FALSE, format = FALSE)
  vehicles <- bind_rows(vehicles, vehicles_append)

  casualties_append <- get_stats19(year = year, type = "casualty", ask = FALSE, format = FALSE)
  casualties <- bind_rows(casualties, casualties_append)
}

# overwriting format_stats19 function
# bug in the get_stats19 function with format = TRUE for certain values of year and type
# another bug in the format_stats19 function which causes an empty lookup table, so had to fix that too
format_stats19 = function(x, type) {
  # Rename columns
  old_names = names(x)
  new_names = format_column_names(old_names)
  names(x) = new_names

  # create lookup table
  # lkp = stats19::stats19_variables[stats19::stats19_variables$table == tolower(type),]

  # vkeep is boolean vector
  vkeep = new_names %in% stats19::stats19_schema$variable
  # which() function produces a list of indexes where vkeep is True
  vars_to_change = which(vkeep)

  for (i in vars_to_change) {
    # lkp_name = unique(lkp$variable[lkp$variable %in% new_names[i]])
    # lookup = stats19::stats19_schema[
    #   stats19::stats19_schema$variable %in% lkp_name,
    #   c("code", "label")
    # ]
    lookup = stats19::stats19_schema[
      stats19::stats19_schema$variable %in% new_names[i],
      c("code", "label")
    ]
    # print(lookup)
    original_class = class(x[[i]])
    # Use lookup to replace codes with labels, but keep original values for non-matches
    matched_labels = lookup$label[match(x[[i]], lookup$code)]
    # print(matched_labels[1:10])
    x[[i]] = ifelse(is.na(matched_labels), x[[i]], matched_labels)
    x[[i]] = methods::as(x[[i]], original_class)
  }

  date_in_names = "date" %in% names(x)
  if(date_in_names) {
    date_char = x$date
    x$date = as.Date(date_char, format = "%d/%m/%Y")
  }
  if(date_in_names && "time" %in% names(x)) {
    # Add formated datetime column, tell people about this new feature
    message("date and time columns present, creating formatted datetime column")

    x$datetime = as.POSIXct(paste(date_char, x$time), tz = 'Europe/London', format = "%d/%m/%Y %H:%M")
  }
  cregex = "easting|northing|latitude|longitude"
  names_coordinates = names(x)[grepl(pattern = cregex, x = names(x), ignore.case = TRUE)]
  # convert them to numeric if not already:
  for(i in names_coordinates) {
    if(!is.numeric(x[[i]])) {
      x[[i]] = as.numeric(x[[i]])
    }
  }
  x
}

accidents <- format_stats19(accidents, type = "Accident")
vehicles <- format_stats19(vehicles, type = "Vehicle")
casualties <- format_stats19(casualties, type = "Casualty")

# every collision has a unique accident_index - get the vehicles associated with each collision
accidents_vehicles <- inner_join(x = accidents,
                                 y = vehicles,
                                 by = c("accident_index"),
                                 suffix = c(".accidents", ".vehicles"),
                                 keep = FALSE,
                                 relationship = "one-to-many")

# get the casualties associated with each vehicle
accidents_vehicles_casualties <- left_join(x = accidents_vehicles,
                                           y = casualties,
                                           by = c("accident_index", "vehicle_reference"),
                                           suffix = c("", ".casualties"),
                                           keep = FALSE)

# get accident_index of every collision involving a cyclist
involving_cyclist <- accidents_vehicles_casualties %>%
  filter(vehicle_type == "Pedal cycle") %>%
  distinct(accident_index, longitude, latitude, date)

### join additional weather data using 'nasapower' API

# create empty tibble to append weather data to
weather <- tibble(
  accident_index = character(),
  RH2M = numeric(),
  T2M = numeric(),
  PRECTOTCORR = numeric(),
  WS2M = numeric()
)

# loop through all collisions, querying the API for the weather at the time and location of each collision
n <- nrow(involving_cyclist)

for (i in seq_len(n)) {
  tbl <- involving_cyclist[i, ]
  idx <- tbl[["accident_index"]]
  lat <- tbl[["latitude"]]
  lon <- tbl[["longitude"]]
  date <- tbl[["date"]]

  # API query returns a tibble containing specified weather variables:
  # RH2M = Relative Humidity at 2 Meters (%)
  # T2M = Temperature at 2 Meters (C)
  # PRECTOTCORR = Precipitation Corrected (mm/day)
  daily_ag <- get_power(
    community = "ag",
    lonlat = c(lon, lat),
    pars = c("RH2M", "T2M", "PRECTOTCORR", "WS2M"),
    dates = date,
    temporal_api = "daily"
  )

  daily_ag["accident_index"] <- idx
  # removing unnecessary columns
  daily_ag <- daily_ag %>% select(-c(YEAR, MM, DD, DOY, LAT, LON, YYYYMMDD))
  # appending weather data to master tibble
  weather <- bind_rows(weather, daily_ag)

  print(paste0("completion: ", round(100 * i / n, 2), "%"))
}

involving_cyclist <- inner_join(x = involving_cyclist,
                                y = weather,
                                by = c("accident_index"),
                                keep = FALSE,
                                relationship = "one-to-one")

# only keep collisions involving cyclists
extract <- inner_join(x = accidents_vehicles_casualties,
                      y = involving_cyclist,
                      by = c("accident_index"),
                      keep = FALSE)

write.csv(extract,
          "C:/Users/danjr/Documents/Projects/cycle collisions/stats19CycleCollisions.csv",
          row.names = FALSE)