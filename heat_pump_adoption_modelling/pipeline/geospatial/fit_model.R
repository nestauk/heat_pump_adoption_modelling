### EST project

# Libraries
library(tidyverse)    # for data wrangling
library(data.table)   # to handle large datasets
library(naniar)       # to explore missingness patterns
library(rgdal)        # to deal with spatial objects
library(sf)
library(raster)
library(spatstat)     # dummify function
library(caramellar)   # to build adjacency matrix
library(INLA)         # for model fitting
library(inlabru)      # utilities for post-processing model outputs
library(brinla)
library(sf)
library(ggplot2)

# settings for config
relevant_cols <- c(
  'POSTCODE', 'FINAL_PROPERTY_TYPE', 'FINAL_PROP_TENURE', 'FINAL_PROPERTY_AGE',
  'FINAL_HAB_ROOMS', 'FINAL_FLOOR_AREA', 'FINAL_WALL_INS', 'FINAL_LOFT_INS',
  'FINAL_GLAZ_TYPE', 'FINAL_FLOOR_INS', 'FINAL_LOW_ENERGY_LIGHTING',
  'FINAL_WIND_FLAG', 'FINAL_HEATING_SYSTEM'
)
postcode_areas <- c('DA')

# utility functions
reformat_postcode <- function(postcode) {
  paste(
    substr(postcode, 1, 4) %>% str_trim,
    substr(postcode, 5, 7)
  )
}

impute_from_distribution <- function(x) { # simple imputation according to empirical distribution function

  nas <- sum(is.na(x))
  x <- as.data.frame(x)

  if (nas>0) {

    which_na <- is.na(x)

    # obtain the distribution function over non-missing records
    tmp_x <- x[!which_na]
    distribution <- table(tmp_x)/sum(table(tmp_x))

    final_x <- x
    final_x[which_na] <- sample(names(distribution),nas,
                                      replace=TRUE,
                                      prob=distribution)

    return(final_x)

  } else return(x)

}


#### IMPORT DATA

# map shapefiles
map_polygons <- readOGR('./inputs/Sectors.shp')
regions_map <- map_polygons[str_extract(map_polygons@data$name, "^[A-Z]+") %in% postcode_areas,]

# postcode centroids
# from https://osdatahub.os.uk/downloads/open/CodePointOpen?_ga=2.21799097.324920968.1632920662-135460240.1632920662
files <- list.files(path="inputs/centroids", pattern="*.csv", full.names = T)
# ideally only read in the files with a particular sector
# but this is quick enough for now
# should we just be reading in the live ones?

centroids <- files %>%
  map_df(fread) %>%
  tibble %>%
  rename(postcode=V1, easting=V3, northing=V4) %>%
  dplyr::select(postcode, easting, northing) %>%
  mutate(postcode_area = str_extract(postcode, "^[A-Z]+")) %>%
  filter(postcode_area %in% postcode_areas) %>%
  mutate(postcode = sapply(postcode, reformat_postcode))

# estimated building populations
households <- read_csv('inputs/postcode_household_estimates.csv') %>%
  rename(postcode=Postcode, n_households=Occupied_Households) %>%
  dplyr::select(postcode, n_households) %>%
  mutate(postcode_area = str_extract(postcode, "^[A-Z]+")) %>%
  filter(postcode_area %in% postcode_areas) %>%
  mutate(postcode = sapply(postcode, reformat_postcode))

# epc data
full_epc <- read_csv('inputs/EPC_Records__cleansed_and_deduplicated.csv')

filtered_epc <- full_epc %>%
  dplyr::select(all_of(relevant_cols)) %>%
  mutate(postcode_area = str_extract(POSTCODE, "^[A-Z]+")) %>%
  filter(postcode_area %in% postcode_areas) %>%
  dplyr::select(!postcode_area)

names(filtered_epc) <- tolower(names(filtered_epc))


# adding variables and filling in gaps

enhanced_epc <- sapply(filtered_epc, impute_from_distribution) %>%
  as_tibble %>%
  `colnames<-`(names(filtered_epc)) %>%
  mutate(heat_pump = (final_heating_system == "Heat pump")) %>%
  mutate_if(sapply(., is.character), as.factor) %>%
  mutate(final_low_energy_lighting=ordered(final_low_energy_lighting),
         final_loft_ins=ordered(final_loft_ins,
                                levels=c('No Loft',
                                         '0-50mm',
                                         '51-100mm',
                                         '101-150mm',
                                         '151-200mm',
                                         '201mm+')),
         final_hab_rooms=ordered(final_hab_rooms),
         final_glaz_type=ordered(final_glaz_type),
         final_property_age=ordered(final_property_age,
                                    levels=c('Pre_1900',
                                             '1900_1929',
                                             '1930_1949',
                                             '1950_1966',
                                             '1967_1982',
                                             '1983_1995',
                                             'Post_1996'))) %>%
  group_by(postcode) %>%
  mutate(n_heat_pumps=sum(heat_pump=='TRUE')) %>% # define outcome as number of installations per postcode
  ungroup

household_data <- left_join(enhanced_epc, centroids, by='postcode') %>%
  # add postcode centroids
  left_join(., households %>% dplyr::select(-one_of('postcode_area')),
            by='postcode') %>%
  filter(n_households>0, !is.na(easting))
# for now, filter out those with zero/NA population
# and no coordinates


#### AGGREGATION

# create postcode-level variables from individual level ones

tmp_dummify <- data.frame(
  dummify(household_data$final_property_type),
  dummify(household_data$final_prop_tenure)
)

names(tmp_dummify) <- paste0('is_', tolower(gsub('\\.','_',names(tmp_dummify))))

household_data <- cbind(household_data, tmp_dummify) %>% tibble

rm(tmp_dummify)

# bear in mind that these are impacted by the choice of relevant columns
# and should ideally account for different choices
postcode_data <- household_data %>%
  group_by(postcode) %>%
  mutate(median_property_age = median(as.numeric(final_property_age)),
         median_hab_rooms = median(as.numeric(final_hab_rooms)),
         median_loft_ins = median(as.numeric(final_loft_ins)),
         p_dt_glaz = mean(final_glaz_type=='Double/Triple'),
         median_low_energy_lighting = median(as.numeric(final_low_energy_lighting)),
         median_floor_area = median(final_floor_area),
         n_block_of_flats = sum(is_block_of_flats),
         p_detached_house = mean(is_detached_house),
         n_end_terraced_house = sum(is_end_terraced_house),
         n_mid_terraced_house = sum(is_mid_terraced_house),
         n_semi_detached_house = sum(is_semi_detached_house),
         p_owner_occupied = mean(is_owner_occupied),
         n_privately_rented = sum(is_privately_rented),
         n_social = sum(is_social),
         p_wall_ins = mean(final_wall_ins=='Insulated'),
         epc_household_count = n()) %>%
  slice(1) %>%
  ungroup %>%
  dplyr::select(-starts_with(c("final_", "is_")), -heat_pump) %>%
  mutate(better_household_estimate = pmax(n_households, epc_household_count))

postcode_data$postcode_n <- 1:nrow(postcode_data)

# create adjacency matrix via Voronoi tessellation
adjacency_graph <- caramellar::voronoi_adjacency(
  postcode_data %>% dplyr::select(postcode_n, easting, northing),
  postcode_n ~ easting + northing
  )$Adjacencies %>% inla.read.graph
# really this should use the full postcode list as some postcodes do not appear in EPC


#### MODEL FITTING

# set inla controls
control <- list(predictor=list(compute=TRUE,link=1),
                results=list(return.marginals.random=TRUE,
                             return.marginals.predictor=TRUE),
                compute=list(hyperpar=TRUE, return.marginals=TRUE,
                             dic=TRUE, mlik=TRUE, cpo=TRUE, po=TRUE,
                             waic=TRUE, graph=TRUE, gdensity=TRUE))

# model specification
model <- n_heat_pumps ~ 1 +
  f(inla.group(median_floor_area, n=100), model="rw1") +
  f(median_property_age, model="rw1") +
  f(median_hab_rooms, model="rw1") +
  f(postcode_n, model = 'besag', graph=adjacency_graph) +
  f(postcode, model='iid')

# simple model
cheap_approximation <- inla(model,
                            family='binomial',
                            Ntrials=better_household_estimate,
                            data=postcode_data,
                            control.inla=list(diagonal=100,
                                              strategy="gaussian",
                                              int.strategy="eb"),
                            control.compute=control$compute,
                            control.predictor=control$predictor,
                            control.results=control$results,
                            verbose=TRUE)

# use the command
#
#model_fit <- cheap_approximation
#
# and ignore the following model fit block to be able to use the rest of the
# code without having to wait for the more accurate approximation to be obtained

# fit the model using the cheap approximation estimates as starting values
# note: this step takes much longer
model_fit <- inla(model,
                  family='binomial',
                  data=postcode_data,
                  Ntrials=better_household_estimate,
                  control.inla=list(diagonal=10),
                  control.fixed = list(prec.intercept = 0.1),
                  control.compute=control$compute,
                  control.predictor=control$predictor,
                  control.results=control$results,
                  control.mode=list(result=cheap_approximation,
                                    restart=TRUE),
                  verbose=TRUE)


#### POST-PROCESSING AND PLOTTING

# add the fitted values to the dataset and make postcode sectors into factors
data_fitted <- postcode_data %>%
  mutate(fitted=model_fit$summary.fitted.values$mean,
         observed=n_heat_pumps/better_household_estimate)

# obtain average fitted probabilities by postcode sector
average_fitted <- data_fitted %>%
  mutate(postcode_sector =
           factor(substr(postcode, 1, nchar(as.character(postcode))-2))) %>%
  group_by(postcode_sector) %>%
  summarise(avg_fitted=mean(fitted),
            avg_observed=mean(observed))

# fitted vs observed
plot(data_fitted$observed, data_fitted$fitted,
     pch=16, cex=.5, col='#00000075')
plot(average_fitted$avg_observed, average_fitted$avg_fitted,
     pch=16, cex=.5, col='#00000075', asp=1)

# model diagnostics
model_plot = function(variable) {
  ggplot(data_fitted, aes_string(x=variable, y='fitted')) +
      geom_point(alpha=0.3) +
      geom_smooth(method='loess',colour='#0000FF75') +
      labs(x = variable, y = "Estimated probability", colour = "Median property age") +
      theme_bw()
}

model_plot('better_household_estimate')
model_plot('median_floor_area')
model_plot('median_property_age')
model_plot('median_hab_rooms')
model_plot('p_detached_house')


# plots for EST
# plot(dataset$median_floor_area, dataset$fitted, col=ifelse(dataset$median_property_age<=2, '#0000FF25','#FF000025'), pch=16, cex=.75)
#
# ggplot(dataset, aes(x=median_floor_area, y=fitted, colour=ifelse(median_property_age<=2, 'Pre-1930','Post-1930'))) +
#   geom_point(alpha=0.3) +
#   labs(x = "Median floor area", y = "Estimated probability", colour = "Median property age") +
#   theme_bw()
#
#
# ggplot(dataset, aes(x=median_hab_rooms, y=fitted)) +
#   geom_point(alpha=0.3) +
#   geom_smooth(method='loess',colour='#0000FF75',width=4)
#
#
# estBetaParams <- function(mu, var) {
#   alpha <- ((1 - mu) / var - 1 / mu) * mu ^ 2
#   beta <- alpha * (1 / mu - 1)
#   return(params = list(alpha = alpha, beta = beta))
# }
#
# data_fitted = data_fitted %>% mutate(hp_prop = n_heat_pumps/better_household_estimate)
# beta_params = estBetaParams(mean(data_fitted$hp_prop), var(data_fitted$hp_prop))
# data_fitted = data_fitted %>% mutate(
#   posterior = (n_heat_pumps + beta_params$alpha) / (better_household_estimate + beta_params$alpha + beta_params$beta)
# ) %>%
#   arrange(desc(posterior)) %>%
#   dplyr::select(postcode, n_heat_pumps, better_household_estimate, posterior, fitted)


# extract the exponentiated spatial residuals
# and add them to the dataset
exp_residuals <- numeric(length(model_fit$marginals.random$postcode_n))

for (i in 1:length(exp_residuals)) {
  tmp <- model_fit$marginals.random$postcode_n[[i]]
  exp_residuals[i] <- inla.emarginal(exp,tmp)
}

rm(tmp,i) # clean up

# make the residuals into a tibble that also contains the postcodes
# and postcode sectors
exp_residuals <- data.frame(postcode_n=1:length(exp_residuals),
                            postcode=unique(dataset$postcode),
                            exp_residuals=exp_residuals) %>%
  mutate(postcode=as.character(postcode)) %>%
  left_join(.,dataset %>%
              dplyr::select(postcode,postcode_sector),
            by=c('postcode')) %>%
  group_by(postcode) %>%
  slice(1)

# add the exponentiated residuals to the dataset
dataset <- left_join(dataset,
                     exp_residuals %>%
                       dplyr::select(postcode_n,exp_residuals),
                     by='postcode_n')

# set up the shapefile and data for plotting
geo_df <- dataset
coordinates(geo_df) <- ~easting+northing

map_utm <- st_transform(st_as_sf(regions_map), "+init=epsg:27700")

plot(map_utm$geometry)
points(geo_df,pch=16,cex=.5,
       col=ifelse(geo_df$n_heat_pumps>0,'#FF000075','#00000005'))

# # obtain the quantiles of the exponentiated residuals distribution
# # residuals are constant for each postcode, extract only one line each
# dataset_singletons <- dataset %>%
#   group_by(postcode_n) %>%
#   slice(1) %>%
#   ungroup

exp_residuals_quantiles <- quantile(dataset$exp_residuals,
                                    p=seq(0,1,by=.025))

#################
# produce plots #
#################

# come back and  fix this
exp_residuals %>%
  mutate(temp = substr(postcode_sector, 1, 3)) %>%
  filter(temp=='NP7') %>%
  group_by(postcode_sector) %>%
  mutate(q1=quantile(exp_residuals,.25),
         q2=quantile(exp_residuals,.5),
         q3=quantile(exp_residuals,.75),
         w=1.5*(q3-q1)) %>%
  ungroup %>%
  ggplot()+geom_histogram(aes(exp_residuals),binwidth = .000001)+
  geom_vline(xintercept = 1, lty=2)+
  ylim(-3,18)+
  xlim(.9999,1.0001)+
  geom_hline(yintercept=0,col='grey')+
  geom_segment(aes(y=-2,yend=-2,x=q1-w,xend=q1))+
  geom_segment(aes(y=-2,yend=-2,x=q1,xend=q3),lwd=2)+
  geom_segment(aes(y=-2,yend=-2,x=q3,xend=q3+w))+
  geom_point(aes(y=-2,x=q2),lwd=2,pch=16,col='white')+
  geom_point(aes(y=-2,x=q2),lwd=2,pch=1)+
  theme_bw()+facet_wrap(~postcode_sector)


`# change quantiles as needed
lower_quantile <- 2   # corresponding to 2.5% in the quantiles vector
upper_quantile <- 40  # corresponding to 97.5% in the quantiles vector

# create the proportions meeting the quantile requirements
prop_q_by_sector <- dataset %>%
  group_by(postcode_sector) %>%
  summarise(postcode_sector=first(postcode_sector),
            prop_l_q2.5=mean(exp_residuals<exp_residuals_quantiles[lower_quantile]),
            prop_g_q97.5=mean(exp_residuals>exp_residuals_quantiles[upper_quantile]))

map_final <- map_utm %>%
  left_join(prop_q_by_sector,by=c('name'='postcode_sector'))

## deal with NAs: to be reviewed to make sure what these are ### DOUBLE-CHECK
map_final$prop_l_q2.5 <- replace_na(map_final$prop_l_q2.5,
                                    mean(map_final$prop_l_q2.5,na.rm=TRUE))
map_final$prop_g_q97.5 <- replace_na(map_final$prop_g_q97.5,
                                     mean(map_final$prop_g_q97.5,na.rm=TRUE))

# define a scaling factor for the grey scale: the max observed proportion
# is slightly less than 0.2, rescaling aids visualisation
scale_factor <- .3

plot(map_final['prop_l_q2.5'],
     main='higher-than-average adopters',
     #border='black',
     col=grey(1-map_final$prop_l_q2.5))

plot(map_final['prop_g_q97.5'],
     main='lower-than-average adopters',
     #border='black',
     col=grey(1-map_final$prop_g_q97.5))


fitted_map = map_utm %>% left_join(average_fitted, by=c('name'='postcode_sector'))
fitted_map$avg_fitted = replace_na(fitted_map$avg_fitted,
                                   mean(fitted_map$avg_fitted,na.rm=TRUE))
plot(fitted_map['avg_fitted'],
     main='fitted probabilities',
     col=rev(leaflet::colorNumeric(palette = "YlOrRd", ## double-check colour palette
                                   domain = fitted_map$avg_fitted)(fitted_map$avg_fitted)))

range=range(fitted_map$avg_fitted)
plot(fitted_map['avg_fitted'],
     main='Average estimated probabilities by postcode sector',
     col=rgb(red=1, green=0, blue=0, alpha=(fitted_map$avg_fitted - range[1])/(range[2] - range[1])))

### implement
###   i) estimation of exceedance probabilities # DONE
###  ii) fix postcode for lower-than/over- plots # DONE
### iii) create maps with exceedance probabilities # DONE


## exceedance probabilities
exc_p_threshold <- 0.1

dataset$exceedance_p <- sapply(model_fit$marginals.fitted.values,
                               function(x) {
                                 1 - inla.pmarginal(q = exc_p_threshold, marginal = x)
                               }
)

map_final <- left_join(map_final, dataset %>%
                         group_by(postcode_sector) %>%
                         summarise(avg_exc_p=mean(exceedance_p)) %>%
                         dplyr::select(postcode_sector,avg_exc_p),
                       by=c('name'='postcode_sector'))

## deal with NAs: to be reviewed to make sure what these are ### DOUBLE-CHECK
map_final$avg_exc_p <- replace_na(map_final$avg_exc_p,
                                  mean(map_final$avg_exc_p,na.rm=TRUE))
# pthreshrange = range(map_final$avg_exc_p)
pthreshrange = c(0, 0.4)
# plot(map_final['avg_exc_p'],
#      main=paste('Average exceedance probabilities - threshold = ', exc_p_threshold),
# #     border='aquamarine4',
#      col=rev(leaflet::colorNumeric(palette = "YlOrRd", ## double-check colour palette
#                                domain = map_final$avg_exc_p)(map_final$avg_exc_p)))

plot(map_final['avg_exc_p'],
     main=paste('Average exceedance probabilities - threshold =', exc_p_threshold),
     #     border='aquamarine4',
     col=rgb(red=0.1, green=0.3, blue=0.1, alpha=(map_final$avg_exc_p + ifelse(map_final$avg_exc_p > 0.2, 0.3, 0))
     )
