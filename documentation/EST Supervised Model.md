# Supervised Model to Predict Heat Pump Adoption

## Goals

The goal of this branch of the project is to build a supervised model for predicting heat pump adoption in domestic housing.

A static model predicts whether a single household will likely get a heat pump in the future, while a temporal model will predict the growth in heat pump installations for a given area (postcode) over a specific time.

We hope to gain insights into heat adoption on a temporal level and aim to identify factors that inhibit or accelerate heat pump adoption.

## Data

Data used for this model

- Preprocessed EPC data (Version Nesta 2021)
- Index of Multiple Deprivation: IMD Rank, Income Score, Employment Score
- MCS data for exact HP installation date

To Do:

- [ ] Add few more description features, e.g. wall cavity type
- [ ] EPC/MCS: fine-tune merging parameter
- [ ] Include more socio-demographic data (e.g. Acorn)

## Data Processing

### Feature Inspection and Selection

We have 67 features from the EPC dataset. Some of them are redudant as they are either constructed from another feature (e.g. Energy Rating Category from Current Energy Rating) or are highly correlated (e.g. Energy Consumption Crrent and CO2 Emissions per Floor Area).

The correlation matrix below shows the correlation between a reduced set of features. For the full matrix, check folder `/outputs/figures/correlation_matrix_complete.png`. The matrix allows us to identify a number of highly correlated features that bring additional information to the feature space, for example, lodgement date (highly correlated with inspection date).

<img src="./img/correlation.png" width="75%">

This chart shows an example for highly correlated features.

<img src="./img/Highly correlated features.jpg" width="75%">

While interesting to analyse the correlations between features, we only discard few features using this method with very high correlations: LODGEMENT_DATE, INSPECTION_DATE (as string), CURR_ENERGY_RATING_NUM, ENERGY_RATING_CAT, UNIQUE_ADDRESS, MAINHEAT_DESCRIPTION, MAINHEAT_SYSTEM.

### Integration of MCS Installation Dates

Using address matching, we find matches for 51150 out of 60690 domestic MCS entries. For EPC entries with MCS matches we update the heat pump status and installation date if necessary. We process the different cases according to this flowchart.

<img src="./img/mcs_epc.png" width="85%">

### Target Variables

For the static model that predicts the current heat pump status of a property, the target variable is HP_INSTALLED. The features HP_SYSTEM, HP_TYPE and HP_INSTALLED are derived from MAINHEAT_DESCRIPTION (describing the current heating system) and strongly corrlate with them and thus all of them need to be removed from the training data X.

For predicting the future heat pump status, we need properties with at least two EPC entries, e.g. one before the heat pump installation and one after. The target variable is HP_INSTALLED of the latest EPC entry.
For this model, removing the feature related to MAINHEAT_DESCRIPTION do not necessarily need to be removed, although we discard some of them due to redundancy.

---

For the temporal model, possible target varaibles are the percentage of properties with a heat pump in a given area (HP coverage) or the growth between time _t_ and time _t+1_. As we accumulate the HP installations, the growth is always positive.

**Issue**

- The EPC Registry only represents 50% of the properties in GB.
- We currently have no official data for how many properties there are per postcode

So how do we normalise the number of properties with heat pumps per postcode in order to gain the HP coverage?

##### Option a)

We normalise by the number of properties with EPC entry up to time _t_ or _t+1_, respectively.
However, if many properties are added to EPC between _t_ and _t+1_, this can cause negative growth.

_Example:_
Postcode XYZ contains 5 properties at time t (e.g. year 2012) and 20 properties at time t+1 (e.g. 2013).
At time t: 3 out of 5 properties have HP (60% coverage)
At time t+1 : 10 out of 20 properties have HP (50% coverage)
Growth: -10%

##### Option b)

We normalise by the number of properties with EPC entry at time _t+1_. This guarantees positive growth and better comparabilty.

_Example_
Number of properties at time t+1: 20
At time t: 3 out of 20 properties have HP (15% coverage)
At time t+1 : 10 out of 20 properties have HP (50% coverage)
Growth: +35%

However, when predicting for several time windows, the results are not necessarily comporable because the number of properties at time t+1 varies with t. Growth may be "watered down" in later years as more and more properties with EPC adre added.
Also, it ignores the actual real-world number of properties per postcode.

#### Option c)

We normalise by the number of total EPC entries for that postcode, not just entries up to t+1. This allows for continious comparisons across years.

_Example_
Number of properties (overall EPC): 50
At time t: 3 out of 50 properties have HP (6% coverage)
At time t+1 : 10 out of 50 properties have HP (20% coverage)
Growth: +14%

However, when comparing smaller time windows the growth is less visible.

##### Option d)

We normalise by the real-world number of properties. However, we discard this option for the time beeing for the following reasons:

- We currently on have an approximation for the number of properties per postcode
- The representation of properties in the EPC registry may vary per postcode, with some postcodes being completely covered by EPC and others not at all. This could give a skewed view on the adoption and growth.
- The number of real-world properties comes from a different "dimension", not connected to the EPC registry.

### Feature Encoding

We have two types of features: numerical and categorical features. A numerical feature consists of numbers, for example the TOTAL*FLOOR_AREA. A categorical feature consists of differente categories, for example \_owner-occupied* or _social rental_ for TENURE or _very poor_ to _very good_ for WINDOW_ENERGY_EFFICIENCY.

Numeric features do not require any special encoding.

#### Ordinal Encoding

Categorical features can be divided two groups: those with a natural ordering and those without. For instance, the WINDOW ENERGY EFFICIENCY categories _very poor_, _poor_, _average_, _good_ and _very good_ have a natural relationship so the categories can be ranked or ordered.

We apply ordinal encoding to those features using manually created rankings. The different categories are given integer values in ascending order, starting with 1.

The follow features are ordinal encoded:
`MAINHEAT_ENERGY_EFF", "CURRENT_ENERGY_RATING", "POTENTIAL_ENERGY_RATING", "FLOOR_ENERGY_EFF", "WINDOWS_ENERGY_EFF", "HOT_WATER_ENERGY_EFF", "LIGHTING_ENERGY_EFF", "GLAZED_TYPE", "MAINHEATC_ENERGY_EFF", "WALLS_ENERGY_EFF", "ROOF_ENERGY_EFF", "MAINS_GAS_FLAG", "CONSTRUCTION_AGE_BAND_ORIGINAL", "CONSTRUCTION_AGE_BAND", "N_ENTRIES", "N_ENTRIES_BUILD_ID", "ENERGY_RATING_CAT"`

#### One-Hot Encoding

The remaining categorical features are one-hot encoded.

For features with a large number of categories, e.g. GLAZED TYPE, we first reduce the number of categories by merging them. For example, `double glazing, unknown install date`, `double, unknown data` and `double glazing` are all mapped to _double glazing_.

### Feature Aggregation

For the temporal model, we need to aggregate the features on postcode level.
Short version:

- For numerical features, we take the median
- For categorical ones, we get the % of properties with that category

_More detailed description follows_

### Preprocessing

Since most machine learning models cannot handle NaN values, we impute the filling in the mean of the repsective feature's values.

Since we have a large feature space (= large number of features), we perform dimensionality reduction using PCA (Principal Component Analysis). We keep the number of principal components sum to an explained variance ratio of 90%.

In case of the static model, this is reduces the number of features from 87 to 24.

<img src="./img/PCA.png" width="70%">

Finally, we standardise our data using a Min-Max scaler.

To Do:

- [ ] More sophistical way for data imputing

### Training Data

For the static model that predicts the current heat pump status, we can include...

_To be completed_

## Models and Performance

### Static HP Model for predicting future HP installation

Linear Support Vector Classifier on balanced set:

```
Number of samples: 10572
Number of features: 87

Accuracy train: 87.0%
Accuracy test:   86.0%

10-fold Cross Validation
---------
Accuracy: 0.87
F1 Score: 0.87
Recall: 0.85
Precision: 0.89
```

<img src="./img/Validationset.png" width="40%">

Linear Support Vector Classifier on imbalanced set (90% non-HP):

```
Accuracy train: 93.0%
Accuracy test:   92.0%

10-fold Cross Validation
---------

Accuracy: 0.93
F1 Score: 0.57
Recall: 0.48
Precision: 0.7
```

<img src="./img/FutureHPStatusCoefficientContributionsPCA.png" width="90%">

Most relevant features for predicting future HP installation:

- Income Score (neg)
- Wind turbine count
- Entry Year
- total floor area
- Number of entries per building
- Current C2 Emissions
- Bungalows
- Mains Gas Flag (neg)
- Social rental
- Main Heating Controls (neg)

### Temporal HP Model

**HP Coverage**

| Model                           | SME mean | Standard Deviation | Accuracy on 5% steps |
| ------------------------------- | -------- | ------------------ | -------------------- |
| SVM Regressor train             | 0.064    | 0.010              | 0.24                 |
| SVM Regressor test              |          |                    | 0.22                 |
| Linear Regression train         | 0.041    | 0.010              | 0.86                 |
| Linear Regression test          |          |                    | 0.82                 |
| Decision Tree Regressor train\* | 0.049    | 0.010              | 1.0                  |
| Decision Tree Regressor test\*  |          |                    | 0.86                 |
| Random Forest Regressor train   | 0.043    | 0.006              | 0.94                 |
| Random Forest Regressor test    |          |                    | 0.89                 |

- not parameter-screened yet

<img src="./img/HP_random_forest.png" width="60%">

<img src="./img/HP_random_forest_valid.png" width="60%">

<img src="./img/HP Coverage at t+1 using Random Forest Regressor on Training Set.png" width="60%">

<img src="./img/HP Coverage at t+1 using Random Forest Regressor on Validation Set.png" width="60%">

**Growth**

| Model                           | SME mean | Standard Deviation | Accuracy on 5% steps |
| ------------------------------- | -------- | ------------------ | -------------------- |
| Linear Regression train         | 0.020    | 0.002              | 0.96                 |
| Linear Regression test          |          |                    | 0.96                 |
| Decision Tree Regressor train\* | 0.049    | 0.010              | 1.00                 |
| Decision Tree Regressor test\*  |          |                    | 0.93                 |
| Random Forest Regressor train   | 0.012    | 0.002              | 0.98                 |
| Random Forest Regressor test    |          |                    | 0.95                 |

<img src="./img//Growth using Random Forest Regressor on Training Set.png" width="60%">

<img src="./img//Growth using Random Forest Regressor on Validation Set.png" width="60%">

## To Do

- [ ] X: ground truth, y : error, relatiive error
- [ ] Map with predictions, ground truths and errors
- [ ] Pick interesting postcodes
- [ ] Predict until 2025
- [ ] Plot decision tree

## Some notes to integrate somewhere (will be deleted later):

- How to deal with NaN? Set to 0, -1, 999? Drop?
- Optimise balance vs. representativness
- Try different ratios
- variance of samples in negative samples reflects variance of population
- for numeric; mean or median, more rigourosly NAn values

- also try with imbalanced set, 99%
- predict change in following years
- time window: larger windows to predict next year(s)
- do predictors change over time?

- What about prediction probabiltities for linear model? What do households on decision boundary look like?

- Feature ablation: if not sure about feature, remove and look at influence

NEWER NOTES

Done:

- Include MCS
- Make sure to drop duplicates

Inspection:
Pull out some interesting postcodes and see what went wrong/right
What are we predicting right or wrong? Color code tenure type in error map or predictions
X: ground truth, y : error, relatiive error
Predict up to 2025
Different windows: time t+1 = 1-2-3 years
What information is missing?
Broader limitations of approach?

Example Selection

---

sample = new.loc[(new['predict'] == False) & (new['ground truth'] == True)]

ID = 2952861554346758

- higher IMD
- higher IMD Decile
- 80 square metere
- energy efficiency between 50-60
- C D rank
- around 10kg emissions
- off gas
- 1950-1966"
- 2009-2013
- mechnical natural
- owner-occupied
- marketet sale
- detached
- house
- boiler and radiator

proba; 0.899918

- semi-detached house with gas boiler and radiator
- owner occupied
- - 1950-1966"
- C rank
- 91 square metre
- Decile 6
- 5 rooms
- relatively high heating costs

False
False

- Low IMD, Flat, Terraced, social, pre 1950-1966", C rated, green deal recent update
  or: Middle IMD, semi-detached House, private (rental), new, C/B rated
  especially easy in cases where it reached its max

---

Model Name: Random Forest Regressor

---

---

## Training Set

Category Accuracy with 5% steps : 0.97
Scores: [0.04735578 0.04733392 0.04765571]
Mean: 0.04744847331770922
Standard deviation: 0.00014681238577870959

---

## Validation Set

Category Accuracy with 5% steps : 0.94
Scores: [0.04734595 0.04743171 0.04775259]
Mean: 0.04751008271080479
Standard deviation: 0.00017501594035653913

GROWTH

---

Model Name: Random Forest Regressor

---

---

## Training Set

Category Accuracy with 5% steps : 0.99

Scores: [0.03115394 0.03148233 0.03118248]
Mean: 0.031272919897221946
Standard deviation: 0.00014853571549387916

## Validation Set

Category Accuracy with 5% steps : 0.98

2.36% growth for 2025
