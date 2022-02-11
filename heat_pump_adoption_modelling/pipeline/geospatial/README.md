# Geostatistical model

This approach uses a geostatistical framework to model heat pump uptake on a postcode level. After processing the household-level EPC data and aggregating certain features by postcode, we use [INLA](https://www.r-inla.org/) to model the distribution of heat pump counts. The number of heat pumps in postcode i is modelled as a Binomial(n<sub>i</sub>, p<sub>i</sub>) random variable where n<sub>i</sub> is the number of properties in the postcode and p<sub>i</sub> is the fitted probability, determined by a combination of the EPC-derived features and a spatial process encoding the relationship between adjacent postcodes. Model diagnostic plots are then produced to investigate the relationship between the features and the fitted values, including choropleth maps to analyse spatial patterns.

## Data sources
- [Postcode district shapefiles](https://longair.net/blog/2021/08/23/open-data-gb-postcode-unit-boundaries)
- [Postcode centroids](https://osdatahub.os.uk/downloads/open/CodePointOpen?_ga=2.21799097.324920968.1632920662-135460240.1632920662)
- [Postcode household count estimates](https://www.nomisweb.co.uk/census/2011/postcode_headcounts_and_household_estimates) 
- EPC: originally from https://epc.opendatacommunities.org/, processed version obtained from Energy Saving Trust


## License

Contains OS data © Crown copyright and database right 2020

Contains Royal Mail data © Royal Mail copyright and database right 2020

Source: Office for National Statistics licensed under the Open Government Licence v.3.0
