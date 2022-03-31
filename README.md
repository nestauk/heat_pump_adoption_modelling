# Heat Pump Adoption Modelling

## Contents

[Overview](#overview)

[Modelling methods](#methods)

[Setup instructions](#setup)

<a name="overview"></a>
## Overview

This repository holds the code used for Nesta’s collaboration with the Energy Saving Trust about **modelling the uptake of heat pumps in the UK**. The aim of this project is to
* understand the profile of households who have already installed a heat pump,
* predict which households or areas could form the next wave of heat pump adopters, and
* make some progress towards predicting the shape and rate of heat pump growth in the UK over the coming years.

More background information about the project can be found [here](https://www.nesta.org.uk/project/speeding-heat-pump-adoption/).

The primary data source is the Energy Performance Certificate (EPC) register. More details about this dataset can be found [here](https://epc.opendatacommunities.org/) (for England and Wales) and [here](https://statistics.gov.scot/resource?uri=http%3A%2F%2Fstatistics.gov.scot%2Fdata%2Fdomestic-energy-performance-certificates) (for Scotland).

Two different modelling approaches are used: a **supervised machine learning** based model and a **geostatistical Bayesian** model. Further technical details can be found below.

This repository will not be maintained beyond March 2022 (other than occasional refactoring/tidying) due to the discontinuation of the project.


<a name="methods"></a>
## Modelling methods

### Supervised model

This approach uses supervised machine learning to predict heat pump adoption based on the characteristics of current heat pump adopters and their properties. One model learns what factors are most informative for predicting heat pump uptake from historical data about individual properties. An alternative model takes the slightly broader approach of predicting the growth in heat pump installations at a postcode level instead of individual households, indicating which areas are more likely to adopt heat pumps in the future.

### Geostatistical model

This approach uses a geostatistical framework to model heat pump uptake on a postcode level. After processing the household-level EPC data and aggregating certain features by postcode, [INLA](https://www.r-inla.org/) is used to model the distribution of heat pump counts. The number of heat pumps in postcode i is modelled as a Binomial(n<sub>i</sub>, p<sub>i</sub>) random variable where n<sub>i</sub> is the number of properties in the postcode and p<sub>i</sub> is the fitted probability, determined by a combination of the EPC-derived features and a spatial process encoding the relationship between adjacent postcodes. Model diagnostic plots are then produced to investigate the relationship between the features and the fitted values, including choropleth maps to analyse spatial patterns.

**Data sources:**
- [Postcode district shapefiles](https://longair.net/blog/2021/08/23/open-data-gb-postcode-unit-boundaries)
- [Postcode centroids](https://osdatahub.os.uk/downloads/open/CodePointOpen?_ga=2.21799097.324920968.1632920662-135460240.1632920662)
- [Postcode household count estimates](https://www.nomisweb.co.uk/census/2011/postcode_headcounts_and_household_estimates) 
- EPC: originally from https://epc.opendatacommunities.org/, processed version obtained from Energy Saving Trust


<a name="setup"></a>
## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `git-crypt` and `conda`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS

The input data can be downloaded from Nesta's S3 **asf-core-data** bucket.


#### Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
