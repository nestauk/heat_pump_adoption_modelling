# Heat Pump Adoption Modelling

### Geospatial Model

##### Premises

- Outcome: number/rate of heat pump adoptions in a prescribed area over a fixed period of time (say, a year)
- Geography: postcode level data.
- Time window:
- Explanatory variables: household characteristics (building type, total floor area, ...), energy-related (gas availability, energy rating, ...), socio-demographic (postcode IMD, …)
- Study design: official register data.

##### Model

- Model and link function: a Poisson regression model with log link will allow to model counts/rates (given a proper offset)
- Structured spatial component: intrinsic Conditional Auto Regressive (iCAR) at the postcode-level
- Variable transformation: non-linear relationships with the outcome will be allowed by entering the covariates via splines, when the interpretation of attached coefficients will not be of primary interest.

##### Outcomes

- Outcome predictions at the desired geographical level will be available and will possess uncertainty measures around them.
- Analysis of model’s residuals will allow to identify areas that, after adjustment for the explanatory variables, still exhibit behaviours that are extreme with respect to the overall average, as estimated by the model.
- Exceedance probabilities, i.e. an estimate of the probability of exceeding a given threshold of likelihood to adopt a heat pump solution.
- Visualisation: all of the above can be represented on maps, which can help improve readability of the findings and aid dissemination to a wider, non-technical audience.

### Supervised Model

##### Premises

- Target: expected % increase in heat pump adoptions in a prescribed area at time (year) t
  -Geography: postcode level data
- Features: household characteristics (building type, total floor area, ...), energy-related (gas availability, energy rating, ...), socio-demographic (postcode IMD, …). Some features will be taken as a snapshot at t-1, while others will be taken for time period t-n to t-1 (where n is a parameter to be tuned)

##### Model

- Model: linear regression as a baseline. Investigate more specific models such as XGBoost regressor.
- Pipeline: Split data into target year (2019?) and feature years. Use variables to build additional features and use to predict % of heat pumps in a postcode (or % increase) for the target year. Feature engineering might include snapshots of variables and growth rates.
- Prediction: Use model to predict for 2020\*

In subsequent iterations we might explore VAR models or variations on LSTM neural networks.

\*we know that 2020 was a highly disrupted year, but we can still make and inspect predictions

##### Outcomes

- Predictions will be available at the post code level and for the year specified.
- An analysis of the model’s residuals will be used to identify postcodes that adopted significantly more or less heat pumps than expected. In addition they will be used to explore the distribution of error dependent on splits among the features.
- Investigation of specific model predictions to past growth can be used to further understand the patterns of heat pump adoption.

### Nearest Neighbour Search

##### Premises

- Assumption: That post code areas that might adopt heat pumps, but have not already, might be identified through sharing similar characteristics with those that have.
- Geography: household/postcode level
- Outcome: Similarity of non-heat pump households/postcodes to those with heat pumps

##### Model

- Pipeline:
  - Dimensionality reduction of household data
  - Split into households with and without heat pumps
  - For each household with a heat pump, find the K nearest neighbours from households without heat pumps
- Geography: postcode level
- Outcomes: Similarity of non-heat pump postcodes to households with heat pumps

##### Outputs

- Triangulation with other models to validate and evaluate our approaches by comparing the nearest neighbours detected with the postcodes/households predicted as adopters.
- Household level predictions where the other approaches yield only postcode level analyses, this method will allow investigation of specific households.
- Ability to test and explore assumptions about the groups who install heat pumps through combination with further qualitative research
- Geographic analysis of which groups are more or less prevalent across regions of the UK

## Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `git-crypt` and `conda`
  - Have a Nesta AWS account configured with `awscli`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure pre-commit
  - Configure metaflow to use AWS

#### Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
