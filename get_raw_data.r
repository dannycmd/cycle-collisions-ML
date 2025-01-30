# install.packages("stats19")
# install.packages("nasapower")
library(stats19)
library(dplyr)
library("nasapower")

year_filter <- 2022
# defining a rectangular geographical area around London
min_lat <- 51
max_lat <- 52
min_lon <- -1
max_lon <- 0.4

# get_stats19 function downloads a csv of the requested data and cleans it
accidents_2022 <- get_stats19(year = year_filter, type = "Accidents", ask = FALSE)
vehicles_2022 <- get_stats19(year = year_filter, type = "vehicle", ask = FALSE)
casualties_2022 <- get_stats19(year = year_filter, type = "casualty", ask = FALSE)

# every collision has a unique accident_index - get the vehicles associated with each collision
accidents_vehicles <- inner_join(x = accidents_2022,
                                 y = vehicles_2022,
                                 by = c("accident_index"),
                                 suffix = c(".accidents", ".vehicles"),
                                 keep = FALSE,
                                 relationship = "one-to-many")

# get the casualties associated with each vehicle
accidents_vehicles_casualties <- left_join(x = accidents_vehicles,
                                           y = casualties_2022,
                                           by = c("accident_index", "vehicle_reference"),
                                           suffix = c("", ".casualties"),
                                           keep = FALSE)

# get accident_index of every collision involving a cyclist
involving_cyclist <- accidents_vehicles_casualties %>%
  # filter(vehicle_type == "Pedal cycle" & latitude >= min_lat & latitude <= max_lat & longitude >= min_lon & longitude <= max_lon) %>%
  filter(vehicle_type == "Pedal cycle") %>%
  distinct(accident_index, longitude, latitude, date)

### join additional weather data using 'nasapower' API

# create empty tibble to append weather data to
# weather <- tibble(
#   accident_index = character(),
#   RH2M = numeric(),
#   T2M = numeric(),
#   PRECTOTCORR = numeric(),
#   WS2M = numeric()
# )

# # loop through all collisions, querying the API for the weather at the time and location of each collision
# n <- nrow(involving_cyclist)

# for (i in seq_len(n)) {
#   tbl <- involving_cyclist[i, ]
#   idx <- tbl[["accident_index"]]
#   lat <- tbl[["latitude"]]
#   lon <- tbl[["longitude"]]
#   date <- tbl[["date"]]

#   # API query returns a tibble containing specified weather variables:
#   # RH2M = Relative Humidity at 2 Meters (%)
#   # T2M = Temperature at 2 Meters (C)
#   # PRECTOTCORR = Precipitation Corrected (mm/day)
#   daily_ag <- get_power(
#     community = "ag",
#     lonlat = c(lon, lat),
#     pars = c("RH2M", "T2M", "PRECTOTCORR", "WS2M"),
#     dates = date,
#     temporal_api = "daily"
#   )

#   daily_ag["accident_index"] <- idx
#   # removing unnecessary columns
#   daily_ag <- daily_ag %>% select(-c(YEAR, MM, DD, DOY, LAT, LON, YYYYMMDD))
#   # appending weather data to master tibble
#   weather <- bind_rows(weather, daily_ag)

#   print(paste0("completion: ", round(100 * i / n, 2), "%"))
# }

# involving_cyclist <- inner_join(x = involving_cyclist,
#                                 y = weather,
#                                 by = c("accident_index"),
#                                 keep = FALSE,
#                                 relationship = "one-to-one")

# only keep collisions involving cyclists
extract <- inner_join(x = accidents_vehicles_casualties,
                      y = involving_cyclist,
                      by = c("accident_index"),
                      keep = FALSE)

write.csv(extract,
          "C:/Users/danjr/Documents/Projects/cycle collisions/stats19CycleCollisions2022.csv",
          row.names = FALSE)