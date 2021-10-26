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

# working dir

setwd("D:/Dropbox/Nesta/Projects/EST")

# x <- fread('Data/EPC Records - cleansed and deduplicated .csv')
# y <- fread('Data/EPC_GB_preprocessed_and_deduplicated.csv')
# heat_pumps <- x %>%
#   group_by(POSTCODE) %>%
#   summarise(sum(FINAL_HEATING_SYSTEM=='Heat pump',na.rm=TRUE))
#
#
# names(x)
# length(unique(x$POSTCODE))
#
# x %>% group_by(COUNTY) %>% summarise(n()) %>% as.data.frame
#
# x %>% filter(is.na(COUNTY))

# read in data

map_polygons <- readOGR('Data/Distribution/Sectors.shp')

### where
where  <- 'NP'
where2 <- 'CF'
###

map_np <- map_polygons[substr(map_polygons@data$name,1,nchar(where))%in%c(where,where2),] # polygons

wales_postcodes <- read.csv('Data/open_postcode_geo_wales.csv',
                            header=FALSE,stringsAsFactors = TRUE)

# from https://osdatahub.os.uk/downloads/open/CodePointOpen?_ga=2.21799097.324920968.1632920662-135460240.1632920662
# np_centroids <- read.csv('Data/np.csv',header=FALSE,stringsAsFactors = TRUE) %>% tibble %>%
#   filter(substr(V1,1,nchar(where))==where) %>%
#   rename(postcode=V1,easting=V3,northing=V4) %>%
#   mutate(postcode=paste(
#     substr(postcode,1,nchar(as.character(postcode))-3),
#     substr(postcode,nchar(as.character(postcode))-2,nchar(as.character(postcode))))
#     ) %>%
#   mutate(postcode=gsub('\\s{2}|\\s{3}',' ',postcode,perl=TRUE)) %>%
#   dplyr::select(postcode,easting,northing)

np_centroids <- rbind(
  read.csv('Data/np.csv',header=FALSE,stringsAsFactors = TRUE),
  read.csv('Data/cf.csv',header=FALSE,stringsAsFactors = TRUE)
  ) %>% tibble %>%
  filter(substr(V1,1,nchar(where))%in%c(where,where2)) %>%
  rename(postcode=V1,easting=V3,northing=V4) %>%
  mutate(postcode=paste(
    substr(postcode,1,nchar(as.character(postcode))-3),
    substr(postcode,nchar(as.character(postcode))-2,nchar(as.character(postcode))))
  ) %>%
  mutate(postcode=gsub('\\s{2}|\\s{3}',' ',postcode,perl=TRUE)) %>%
  dplyr::select(postcode,easting,northing)


# read in estimated building population
# np_population <- read_csv('Data/Postcode_Estimates_Table_1.csv') %>%
#   mutate(postcode=paste(
#     substr(Postcode,1,nchar(Postcode)-3),
#     substr(Postcode,nchar(Postcode)-2,nchar(Postcode)))
#     ) %>%
#   mutate(postcode=gsub('\\s{2}|\\s{3}',' ',postcode,perl=TRUE)) %>%
#   filter(substr(postcode,1,nchar(where))==where) %>%
#   dplyr::select(postcode,n_households=Occupied_Households)

np_population <- read_csv('Data/Postcode_Estimates_Table_1.csv') %>%
  mutate(postcode=paste(
    substr(Postcode,1,nchar(Postcode)-3),
    substr(Postcode,nchar(Postcode)-2,nchar(Postcode)))
  ) %>%
  mutate(postcode=gsub('\\s{2}|\\s{3}',' ',postcode,perl=TRUE)) %>%
  filter(substr(postcode,1,nchar(where))%in%c(where,where2)) %>%
  dplyr::select(postcode,n_households=Occupied_Households)

names(np_population) <- tolower(names(np_population))

#x_full <- read.csv('Data/np_subset.csv',header=TRUE,stringsAsFactors = TRUE) %>% tibble
x_full <- rbind(read.csv('Data/np_subset.csv',header=TRUE,stringsAsFactors = TRUE),
                read.csv('Data/cf_subset.csv',header=TRUE,stringsAsFactors = TRUE)) %>% tibble

x <- x_full %>%
  mutate(HEAT_PUMP=FINAL_HEATING_SYSTEM=='Heat pump') %>%
  dplyr::select(POSTCODE,brn=BUILDING_REFERENCE_NUMBER,
                LODGEMENT_DATE, FINAL_PROPERTY_TYPE, FINAL_PROP_TENURE,
                FINAL_PROPERTY_AGE, FINAL_HAB_ROOMS, FINAL_FLOOR_AREA,
                FINAL_WALL_TYPE, FINAL_WALL_INS, FINAL_RIR,
                FINAL_LOFT_INS, FINAL_ROOF_TYPE, FINAL_MAIN_FUEL,
                FINAL_SEC_SYSTEM, FINAL_SEC_FUEL_TYPE, FINAL_GLAZ_TYPE,
                FINAL_ENERGY_CONSUMPTION, FINAL_EPC_BAND, FINAL_EPC_SCORE,
                FINAL_CO2_EMISSIONS, FINAL_FUEL_BILL, FINAL_METER_TYPE,
                FINAL_FLOOR_TYPE, FINAL_FLOOR_INS, FINAL_HEAT_CONTROL,
                FINAL_LOW_ENERGY_LIGHTING, FINAL_FIREPLACES, FINAL_WIND_FLAG,
                FINAL_PV_FLAG, FINAL_SOLAR_THERMAL_FLAG, FINAL_MAIN_FUEL_NEW,
                HEAT_PUMP) %>%
  filter(!is.na(HEAT_PUMP))

names(x) <- tolower(names(x))

x[x=='']<- NA

# miss_summary_res <- miss_summary(x)
#
# vis_miss(x,sort_miss=FALSE,warn_large_data = FALSE)
#
# sets <- 10
# nint <- 30
# p_miss <- gg_miss_upset(x,nsets=sets,nintersects=nint)
# p_miss

# pre-processing

x2 <- x %>%
  dplyr::select(-one_of("final_solar_thermal_flag",
                        "final_pv_flag"))

gg_miss_upset(x2,nsets=sets,nintersects=nint)
vis_miss(x2,sort_miss=FALSE,warn_large_data = FALSE)

miss_summary_res <- miss_summary(x2)
miss_summary_res

###### process data on missingness
#
# x3 <- x2[complete.cases(x2),] # too aggressive a strategy

# for now, single-impute values

###########################################
### TO BE REVIEWED BEFORE FULL ANALYSIS ###
### DOUBLE-CHECK MISSINGNESS BY DESIGN  ###
###########################################

# is.na(final_property_type) + is.na(final_prop_tenure) seem to account for the largest
# row-wise missingness

impute_from_distribution <- function(x) { # simple imputation according to empirical distribution function

  nas <- sum(is.na(x))
  x <- as.data.frame(x)

  if (nas>0) {

      which_na <- is.na(x)

      # obtain the distribution function over non-missing records
      tmp_x <- x[!which_na]
      distribution <- table(tmp_x)/sum(table(tmp_x))

      final_x <- x

      final_x[which_na] <- names(sample(distribution,nas,
                                        replace=TRUE,
                                        prob=distribution))

      #cat(class(final_x))

      return(final_x)

    } else return(as.data.frame(x))

  }


x3 <- apply(x2,2,impute_from_distribution) %>%
  as.data.frame %>% tibble %>%
  `colnames<-`(names(x2)) %>%
  mutate_if(sapply(., is.character), as.factor) %>%
  mutate_if(names(.)%in%c('final_fireplaces','final_fuel_bill','final_co2_emissions',
                'final_epc_score','final_energy_consumption','final_floor_area'),
         as.numeric) %>%
  mutate(final_low_energy_lighting=ordered(final_low_energy_lighting),
         final_epc_band=ordered(final_epc_band),
         final_loft_ins=ordered(final_loft_ins,
                                levels=c('No Loft','0-50mm','51-100mm',
                                         '101-150mm','151-200mm','201mm+')),
         final_hab_rooms=ordered(final_hab_rooms),
         final_glaz_type=ordered(final_glaz_type),
         final_property_age=ordered(final_property_age,
                                    levels=c('Pre_1900','1900_1929', '1930_1949','1950_1966',
                                             '1967_1982', '1983_1995', 'Post_1996')),
         postcode_n=as.numeric(postcode)) %>%
  arrange(postcode_n)

#
######

# create the outcome variable and merge centroids and (estimated) population

x4 <- x3 %>%
  group_by(postcode_n) %>%
  mutate(n_heat_pumps=sum(heat_pump=='TRUE')) %>% # define outcome as number of installations per postcode
  ungroup

x4$final_hab_rooms[x4$final_hab_rooms=='07-Aug'] <- '7-8' # quick fix for excel dates problem

x5 <- x4 %>% left_join(.,np_centroids,by='postcode') %>%   # add postcode centroids
  left_join(.,np_population %>%
              dplyr::select(postcode,n_households),
            by='postcode') %>%
  filter(n_households>0,!is.na(easting))          #### for now, filter out those with zero/NA population ####
                                                  #### and no coordinates ####
###################
### AGGREGATION ###
###################

# create postcode-level variables from individual level ones
#
# decisions to be made re how to aggregate categorical/ordinal household characteristics
#
#   i) go with some summary measure, such as a mode/median
#  ii) pick a category and compute proportion with/below/above the characteristic by postcode
# iii) split using dummy variables, then sum by postcode to obtain counts
#  iv) consider using functional data as covariates
#   v) use dimensionality reduction techniques
#
# some thoughts
#
#   i) light-touch, but we would lose a lot of information
#  ii) slightly better than i), but still losing information + arbitrary choice of categories
# iii) would seem the better solution, the number of covariates would increase, but not the
#      number of parameters (still k-1 for a k categories variable?). However, we risk losing
#      the ordinal nature of some of the variables - maybe use ii) for those? <- think.
#  iv) quite complex, but maybe not unfeasible, and would ideally allow to retain
#      the most information - not sure how this would affect estimation, whereas it would
#      certainly make interpretation more complex (nuanced?)
#   v) we first would need a good chat with domain experts to understand which bits of information
#      can be "mixed together", and which would be better to leave as standalone variables
#
# For now, let's just go with iii), mindful that we are somewhat losing the ordinal nature of
# the following:
#
# "final_property_age"          <- take middle value of classes then numerify
# "final_hab_rooms"             <- numerify, with 8+==8, 0-2==1.5 [we might recover the actual numbers]
# "final_loft_ins"              <- is this temporally antecedent heat pump installations?
# "final_glaz_type"             <- dichotomous, Single/Partial vs Total
# "final_epc_band"              <- when was this measured? pre/post installation?
# "final_low_energy_lighting"   <- pick 75% as threshold and compute proportions [ask Chris]

tmp_dummify <- data.frame(dummify(x5$final_property_type),dummify(x5$final_prop_tenure),
                          dummify(x5$final_wall_type),dummify(x5$final_roof_type),
                          dummify(x5$final_main_fuel),dummify(x5$final_sec_fuel_type),
                          dummify(x5$final_floor_type),dummify(x5$final_heat_control))

x6 <- cbind(x5,tmp_dummify) %>% tibble
names(x6)[47] <- 'wall_park_home'
names(x6) <- tolower(gsub('\\.','_',names(x6)))

x7 <- x6 %>%
  group_by(postcode_n) %>%
  mutate(median_property_age=median(as.numeric(final_property_age)),
         median_hab_rooms=median(as.numeric(final_hab_rooms)),
         median_loft_ins=median(as.numeric(final_loft_ins)),
         p_dt_glaz=mean(final_glaz_type=='Double/Triple'),
         median_epc_band=median(as.numeric(final_epc_band)),
         median_epc_score=median(final_epc_score),
         median_low_energy_lighting=median(as.numeric(final_low_energy_lighting)),
         median_fireplaces=median(final_fireplaces),
         median_energy_consumption=median(final_energy_consumption),
         median_epc_score=median(final_epc_score),
         median_co2_emissions=median(final_co2_emissions),
         median_fuel_bill=median(final_fuel_bill),
         median_floor_area=median(final_floor_area),
         n_block_of_flats=sum(block_of_flats),
         n_detached_house=sum(detached_house),
         n_end_terraced_house=sum(end_terraced_house),
         n_mid_terraced_house=sum(mid_terraced_house),
         n_park_home=sum(park_home),
         n_semi_detached_house=sum(semi_detached_house),
         n_owner_occupied=sum(owner_occupied),
         n_privately_rented=sum(privately_rented),
         n_social=sum(social),
         n_cavity_construction=sum(cavity_construction),
         n_wall_park_home=sum(wall_park_home),
         n_solid_brick_or_stone=sum(solid_brick_or_stone),
         n_system_built=sum(system_built),
         n_timber_frame=sum(timber_frame),
         n_dwelling_above=sum(dwelling_above),
         n_flat=sum(flat),
         n_pitched=sum(pitched),
         n_thatched=sum(thatched),
         n_biomass_solid=sum(biomass_solid),
         n_communal=sum(communal),
         n_electricity=sum(electricity),
         n_lpg=sum(lpg),
         n_mains_gas=sum(mains_gas),
         n_no_heating_system_present=sum(no_heating_system_present),
         n_oil=sum(oil),
         n_biomass_solid_1=sum(biomass_solid_1),
         n_electricity_1=sum(electricity_1),
         n_lpg_1=sum(lpg_1),
         n_mains_gas_1=sum(mains_gas_1),
         n_no_secondary_heating_system=sum(no_secondary_heating_system),
         n_oil_1=sum(oil_1),
         n_solid=sum(solid),
         n_suspended=sum(suspended),
         n_unheated_space_other_premise_below=sum(unheated_space_other_premise_below),
         n_no_heating_control=sum(no_heating_control),
         n_programmer_and_thermostat=sum(programmer_and_thermostat),
         n_programmer_only=sum(programmer_only),
         n_thermostat_only=sum(thermostat_only),
         p_sec_system=mean(final_sec_system=='Yes'),
         p_dual_meter_type=mean(final_meter_type=='Dual'),
         p_rir=mean(final_rir=='RIR'),
         p_wall_insulateds=mean(final_wall_ins=='Insulated')) %>%
  slice(1) %>%
  ungroup

x7$postcode_n <- 1:nrow(x7)
x7 <- x7 %>% arrange(postcode_n)

# create adjacency matrix via Voronoi tessellation

adjacency_graph <- voronoi_adjacency(x7 %>%
                                       dplyr::select(postcode_n,
                                                     easting,northing),
                                     postcode_n~easting+northing)$Adjacencies %>% inla.read.graph


#################
# fit the model #
#################

# set inla controls
control <- list(predictor=list(compute=TRUE,link=1),
                results=list(return.marginals.random=TRUE,
                             return.marginals.predictor=TRUE),
                compute=list(hyperpar=TRUE, return.marginals=TRUE,
                             dic=TRUE, mlik=TRUE, cpo=TRUE, po=TRUE,
                             waic=TRUE, graph=TRUE, gdensity=TRUE))

# model specification
model <- n_heat_pumps ~ 1 + n_households +  median_floor_area + median_property_age +
  f(median_epc_score, model = "rw1", constr = FALSE) +
  f(postcode_n, model = 'besag',graph=adjacency_graph) +     # structured spatial component (iCAR)
  f(postcode, model='iid')                                   # unstructured spatial component (iid)

# f(n_households, model = "rw1", constr = FALSE) +
#   f(median_epc_score, model = "rw1", constr = FALSE) +
#   f(median_floor_area, model = "rw1", constr = FALSE) +
#   f(median_property_age, model = "rw1", constr = FALSE) +
#   f(postcode_n, model = 'besag',graph=adjacency_graph) +     # structured spatial component (iCAR)
#   f(postcode, model='iid')                                   # unstructured spatial component (iid)
# f(median_fireplaces, model = "rw1", constr = FALSE) +
# f(median_loft_ins, model = "rw1", constr = FALSE) +
# f(median_hab_rooms, model = "rw1", constr = FALSE) +
#
#
# final_property_type + final_prop_tenure + final_floor_area +
# final_wall_type + final_wall_ins + final_rir + final_roof_type + final_main_fuel + final_sec_system +
# final_sec_fuel_type + final_glaz_type + final_energy_consumption + final_epc_score + final_co2_emissions +
# final_fuel_bill + final_meter_type + final_floor_type + final_floor_ins + final_heat_control +
# final_low_energy_lighting + final_wind_flag + final_main_fuel_new +


cheap_approximation <- inla(model,
                            family='zeroinflatedpoisson0',
                            data=x7,
                            control.inla=list(diagonal=100,
                                              strategy="gaussian",
                                              int.strategy="eb"),
                            control.compute=control$compute,
                            control.predictor=control$predictor,
                            control.results=control$results,
                            verbose=TRUE)

# use the command
#
# model_fit <- cheap_approximation
#
# and ignore the following model fit block to be able to use the rest of the
# code without having to wait for the more accurate approximation to be obtained

# fit the model using the cheap approximation estimates as starting values
# note: this step takes much longer
model_fit <- inla(model,
                  family='zeroinflatedpoisson0',
                  data=x7,
                  control.inla=list(diagonal=10),
                  control.fixed = list(prec.intercept = 0.1),
                  control.compute=control$compute,
                  control.predictor=control$predictor,
                  control.results=control$results,
                  control.mode=list(result=cheap_approximation,
                                    restart=TRUE),
                  verbose=TRUE)

###################################
# posterior distributions summary #
###################################

# table of fixed effects posterior estimates
model_fit$summary.fixed

# table of random effects posterior estimates
model_fit$summary.random        # expressed in terms of precision
bri.hyperpar.summary(model_fit) # expressed in terms of standard deviation

#######################################
# post-processing of the model output #
#######################################

## plot predictors against fitted (example)

par(mfrow=c(2,2))
# epc score
plot(x7$median_epc_score,model_fit$summary.fitted.values$mean,
     pch=16,cex=.5,col='#00000050')
lines(lowess(model_fit$summary.fitted.values$mean~x7$median_epc_score),
  col='#0000FF75',lwd=4)
# number of households
plot(x7$n_households,model_fit$summary.fitted.values$mean,
     pch=16,cex=.5,col='#00000050')
lines(lowess(model_fit$summary.fitted.values$mean~x7$n_households),
      col='#0000FF75',lwd=4)
# floor area
plot(x7$median_floor_area,model_fit$summary.fitted.values$mean,
     pch=16,cex=.5,col='#00000050')
lines(lowess(model_fit$summary.fitted.values$mean~x7$median_floor_area),
      col='#0000FF75',lwd=4)
# property age
plot(x7$median_property_age,model_fit$summary.fitted.values$mean,
     pch=16,cex=.5,col='#00000050')
lines(lowess(model_fit$summary.fitted.values$mean~x7$median_property_age),
      col='#0000FF75',lwd=4)
par(mfrow=c(1,1))


## add the fitted values to the dataset and make postcode sectors into factors
dataset <- x7 %>%
  mutate(fitted=model_fit$summary.fitted.values$mean,
         postcode_sector=factor(substr(postcode,1,nchar(as.character(postcode))-2)))

# obtain predictions by postcode sector
average_fitted <- dataset %>%
  group_by(postcode_sector) %>%
  summarise(avg_fitted=mean(fitted),
            avg_observed=mean(n_heat_pumps))

# extract the exponentiated spatial residuals
# and add them to the dataset
exp_residuals <- numeric(length(model_fit$marginals.random$postcode_n))

for (i in 1:length(exp_residuals)) {

  tmp <- model_fit$marginals.random$postcode_n[[i]]

  exp_residuals[i] <- inla.emarginal(exp,tmp)

}

rm(tmp,i) # clean up

# make the residuals  into a tibble that also contains the postcodes
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

map_np_utm <- st_transform(st_as_sf(map_np), "+init=epsg:27700")

plot(map_np_utm$geometry)
points(geo_df,pch=16,cex=.5,
       col=ifelse(geo_df$n_heat_pumps>0,'#FF000075','#00000025'))

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

exp_residuals %>%
  group_by(postcode_sector) %>%
  mutate(q1=quantile(exp_residuals,.25),
         q2=quantile(exp_residuals,.5),
         q3=quantile(exp_residuals,.75),
         w=1.5*(q3-q1)) %>%
  ungroup %>%
  ggplot()+geom_histogram(aes(exp_residuals),binwidth = .000001)+
  geom_vline(xintercept = 1, lty=2)+
  ylim(-3,18)+
  #xlim(.9999,1.0001)+
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

map_np_final <- map_np_utm %>%
  left_join(prop_q_by_sector,by=c('name'='postcode_sector'))

## deal with NAs: to be reviewed to make sure what these are ### DOUBLE-CHECK
map_np_final$prop_l_q2.5 <- replace_na(map_np_final$prop_l_q2.5,
                                       mean(map_np_final$prop_l_q2.5,na.rm=TRUE))
map_np_final$prop_g_q97.5 <- replace_na(map_np_final$prop_g_q97.5,
                                       mean(map_np_final$prop_g_q97.5,na.rm=TRUE))

# define a scaling factor for the grey scale: the max observed proportion
# is slightly less than 0.2, rescaling aids visualisation
scale_factor <- .3

plot(map_np_final['prop_l_q2.5'],
     main='higher-than-average adopters',
     #border='black',
     col=grey(1-map_np_final$prop_l_q2.5))

plot(map_np_final['prop_g_q97.5'],
     main='lower-than-average adopters',
     #border='black',
     col=grey(1-map_np_final$prop_g_q97.5))

### implement
###   i) estimation of exceedance probabilities # DONE
###  ii) fix postcode for lower-than/over- plots # DONE
### iii) create maps with exceedance probabilities # DONE


## exceedance probabilities
exc_p_threshold <- 1.5

dataset$exceedance_p <- sapply(model_fit$marginals.fitted.values,
              function(x) {
                1 - inla.pmarginal(q = exc_p_threshold, marginal = x)
                }
              )

map_np_final <- left_join(map_np_final, dataset %>%
                            group_by(postcode_sector) %>%
                            summarise(avg_exc_p=mean(exceedance_p)) %>%
                            dplyr::select(postcode_sector,avg_exc_p),
                          by=c('name'='postcode_sector'))

## deal with NAs: to be reviewed to make sure what these are ### DOUBLE-CHECK
map_np_final$avg_exc_p <- replace_na(map_np_final$avg_exc_p,
                                       mean(map_np_final$avg_exc_p,na.rm=TRUE))
plot(map_np_final['avg_exc_p'],
     main=paste('exceedance probabilities - threshold=',exc_p_threshold),
#     border='aquamarine4',
     col=rev(leaflet::colorNumeric(palette = "YlOrRd", ## double-check colour palette
                               domain = map_np_final$avg_exc_p)(map_np_final$avg_exc_p)))
