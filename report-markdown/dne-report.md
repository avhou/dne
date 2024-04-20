---
title: "IM1102-232433M - Deep Neural Engingeering assignment 2"
subtitle: "Modifying the attention mechanism of transformers for time series forecasting"
author: "Arne Lescrauwaet - Joachim Verschelde - Alexander Van Hecke"
institute: "OU"
date: \today
geometry: margin=2.5cm
numbersections: false
link-citations: true
link-bibliography: true
papersize: a4
---


# Introduction

This report details the steps taken by Arne Lescrauwaet (852617312), Joachim Verschelde (852594432) and Alexander Van Hecke (852631385) for the second assignment of the 2023 Deep Neural Engineering course organised by the Open University [@dne].

For this assignment we look at different attention mechanisms in transformers [@transformer] for use with time series data.  The attention mechanism enables a transformer model to selectively focus on relevant parts of the input data.  The goal is to be able to capture long range dependencies and relationships between items of the input data.  This is particularly important for time series data containing recurring patterns, e.g. hourly traffic counts on busy highways and hourly power consumption of nations.  We expect these types of data to contain clear and recurring patterns (i.e. traffic will typically be lower during weekends) and we want an attention mechanism to capture these patterns.  In addition to capturing recurring patterns, we would also like to be able to capture the ``local context'' of a pattern to predict new values.  That is, when encountering an event that is similar to a past event, we want to take the outcome of that past event into account in our prediction.

Different kinds of attention mechanisms exist.  Convolutional self-attention is introduced in [@paper], which aims to capture the local context of input events, but does this using a symmetric convolution, thereby taking both input data leading to a particular event and the outcome of that event into account.  A dual-stage attention mechanism is used for a Recurrent Neural Network (RNN) architecture in [@dualstage], using an input attention mechanism in an encoder step, and a temporal attention mechanism in a decoder step.

Even though transformers were originally designed in the field of natural language processing (NLP), a lot of work has been done to use transformers with time series data.  An overview of different ways to adapt transformers to time series data is given in [@timeseries1].  The time2vec encoding mechanism is introduced in [@timeseries2].  The authors of this paper use transformer models to predict stock prices, and claim these models can be used both for short and long term predictions.  The effectiveness of applying transformers to time series data is tested in [@timeseries3].

The original transformer architecture introduces a quadratic time and space complexity.  Much work has been done to improve on this.  The LogSparse transformer is introduced in [@paper], which reduces the memory cost to $O(L {(\log {}L)}^2)$.  The informer model [@informer] even achieves $O(L {\log {}L)}$ memory complexity.  In this report we will focus on attention mechanisms in the context of time series forecasting, ignoring space and time complexity of the transformer algorithm.

# Goal

In this paper, we focus on using transformers for time series forecasting. We aim to compare different attention mechanism and determine which mechanism best captures the outcome of past events.  We formulate a first research question : 

> **RQ 1 : When comparing regular self-attention, convoluted self-attention, right-padded convoluted self-attention and fourier transform based self-attention, which mechanism best predicts future values using mean square error (MSE) as metric?**

The Elia dataset used is fully described in [the dataset description section](#sec:dataset).  It not only contains time series data, but also day+1 and day+7 predictions of the same data.  We formulate a second research question : 

> **RQ 2 : Is the MSE of a transformer model better than the Elia prediction model?**

Firstly, this report will look at the characteristics of the dataset used and discuss pre-processing steps.  Then, we will consider several attention mechanisms,  discuss design and implementation details and finally evaluate the performance of these attention mechanisms on the dataset.

# Data analysis

## Dataset description {#sec:dataset}

We use data from Elia [@elia], which operates the electricity transmission network in Belgium.  In particular, we use the solar power forecast datasets.  These contain time series of actual measured power in megawatt (MW), and also  day+1 and day+7 predictions of solar power output in MW.  Data is available in monthly datasets for the period of February 2013 to February 2024.  Measurements and predictions are recorded every quarter of an hour.  The measured value is always the amount of power equivalent to the running average measured for that particular quarter-hour.  The layout of the dataset is fully described here [@dataset].  We recap the most important points in Table \ref{table:features}.

| feature          | description                           | range                            |
|:-----------------|:--------------------------------------|:---------------------------------|
| DateTime         | Date and time per quarter hour        | [00:00 - 24:00] in quarter hours |
| Measurement      | Measured solar power production in MW | [0.0 - 6000.0]                   |
| Day+1 prediction | D+1 solar power forecast in MW        | [0.0 - 6000.0]                   |
| Day+7 prediction | D+7 solar power forecast in MW        | [0.0 - 6000.0]                   |

Table:  Features captured per quarter-hour in @dataset \label{table:features}

## Data general properties

Data is not normally distributed but highly regular and contains obvious day - night recurring patterns.  Since we are using solar power production data, data typically shows no values in the early morning, building towards a peak around noon, and then slowly reducing values towards the evening.  This is illustrated in Figure @{fig:recurrent-pattern-september}.

There are obvious differences in solar power generation between summer months and winter months, but the general pattern remains the same, as illustrated in Figure @{fig:recurrent-pattern-january}.

## Data pre-processing

The Elia data [@dataset] is very fine grained and contains $24*4=96$ measurements per day, resulting in $30*24*4=2880$ measurements for a 30 day month.  In order to be able to limit memory and computational resources, we have added the possibility to aggregate these dataset.  Possible choices are **(i)** no aggregation, **(ii)** hourly aggregation, **(iii)** aggregation every 4 hours (starting from 00:00, resulting in 6 values per day), and finally **(iv)** aggregation per day.  Aggregation is done by averaging the values in the selected timeframe.

Elia provides a lot of historical data, going from February of 2013 to February of 2024.  All data were taken into account, in order to maximize the possibility of finding interesting patterns in the data.  Input length $L$ has to be chosen carefully in basic transformer architectures because of the quadratic complexity in $L$.  Taking too few measurements into acount, it will be difficult to spot similar events in the past.  Taking too many measurements into account, it will be prohibitely expensive in terms of memory and computational resources to train and evaluate the model.  The model implemented allowed for easy selection of input length $L$.  This is related to the level of aggregations in terms of how many hours or days this represents, i.e. when using hourly aggregating and taking 24 input measurements, we are looking at the data of exactly one day.

### Outlier analysis {#sec:outlier}

A visual outlier analysis yielded no abnormal or obiously wrong values.  This makes sense, as the data contains actually measured solar power.  Therefore, no values were discarded.

# Methodology and Implementation

## Research methodology

We started by examining the dataset [@dataset]. Outlier analysis yielded no results, and we performed a number of standard checks on the quality of the data and decided not to exclude any data from the dataset.  

Given a basic transformer architecture, we implemented a number of attention mechanisms to investigate influence on prediction MSE.  We first evaluated different models by varying some hyperparameters and by varying data aggregation (see Table \ref{table:hyperparameters}).  Given our limited computational resources, we chose a fixed set of hyperparameter and aggregation values for the rest of the experiments.  

Data was split in a training part (63 %), a validation part (10 %), and a test part (27 %).  To accomplish this split, all data was sorted chronologically.  All data up to but not including 2020 served as training data, the data of 2020 up to but not including 2021 served as validation data, and data of 2021 and later served as test data.  Models were first trained on the training dataset and then validated on the validation dataset for a maximum of 100 epochs.  However, to limit computation, we kept track of the minimum average validation error across all epochs.  An early stop was forced if the average validation error of the current epoch exceeded the minimum average validation error 5 times, as this would indicate the validation error was no longer decreasing.  Each model was then tested on the testing set and all losses (training losses, validation losses and test losses) were kept for later analysis.   In all cases, MSE was used as the loss metric.  


## Design elaboration

We decided to implement and evaluate the following attention mechanisms (Table \ref{table:attention-mechanisms}) : 

- regular self-attention (AM-1).  This is the mechanism described in the original transformer paper [@transformer].
- convoluted self-attention as described in [@paper] (AM-2).  This mechanism generalizes the regular self-attention mechanism and uses a 1D convolution to transform the Query (Q) and Key (K) values before using them in the transformer architecture.
- right padded convoluted self-attention (AM-3).  This is a variation of the mechanism described in [@paper].  Whereas [@paper] uses a symmetric convolution, here we use a convolution that focuses on the right hand side to transform Q and K values before using them in the transformer architecture.  Padding to the right is done to prevent looking at future values.
- fourier transform based self-attention (AM-4).  This uses the fourier transform to decompose the input embedding in a vector of frequency values.  These vectors are used as a measure of similarity between keys and values to determine where to direct attention.


| attention mechanism                  | abbreviation |
|:-------------------------------------|:-------------|
| regular self-attention               | AM-1         |
| convoluted self-attention            | AM-2         |
| asymmetric convoluted self-attention | AM-3         |
| fourier transform self-attention     | AM-4         |

Table:  Attention mechanisms \label{table:attention-mechanisms}

Transformer models have several tunable hyperparameters.  We first experimented with variations of number of layers, number of heads, levels of forward expansion, convolution kernel sizes and aggregation levels of input data for the base transformer model, as detailed in Table \ref{table:hyperparameters}.  This yielded very small differences in average test loss (see Table \ref{table:avg-test-losses-base-transformer}).  Given our limited computational resources, we decided to fix the number of layers to 2, the number of heads to 4, the forward expansion to 256 and the aggregation to ``1 day''.  For the models using a convolution (AM-2 and AM-3), we experimented with kernel sizes of [3, 6, 9].  In all scenarios, MSE was used as the measure to optimize for.

| attention mechanism | hyperparameters                                                                           |
|:--------------------|:------------------------------------------------------------------------------------------|
| AM-1                | layers [2, 4, 6], heads [4, 8], forward expansion [256, 512], aggregation [1day, 4 hours] |
| AM-2                | layers [2], heads [4], forward expansion [256], aggregation [1day], kernel size [3, 6, 9] |
| AM-3                | layers [2], heads [4], forward expansion [256], aggregation [1day], kernel size [3, 6, 9]                                                                                      |
| AM-4                | TODO                                                                                      |

Table:  Hyperparameters \label{table:hyperparameters}

Feature embedding was done using a combination of both positional encoding and a more specific temporal encoding, taking into account hour of the day, day of the week, day of the month and month of the year of the data.  The temporal encoding was added to the input vector and served as an additional clue for the transformer model to link similar events.

TODO TODO 
In each iteration, we run the grid search using the training data, and predict against the test data (y~predicted_test~).  All predictions are stored for later analysis.  

This entire design is repeated for a number of different scenarios.  We detail these in Table \ref{table:scenarios}.

- scenario summer-season : In this scenario, we aim to investigate recurring events in the most sunny season of one year.  We concatenate all data of June, July, and August of 2023.
- scenario winter-season : In this scenario, we aim to investigate recurring events in the least sunny season of one year.  We concatenate all data of December, January and February of 2023.
- scenario summer-month : In this scenario, we aim to investigate recurring events in the most sunny month of all years (period 2014-2023).  We concatenate all data of August for years 2014-2023.
- scenario winter-month : In this scenario, we aim to investigate recurring events in the least sunny month of all years (period 2014-2023).  We concatenate all data of February for years 2014-2023.


| scenario      | description                                              |
|:--------------|:---------------------------------------------------------|
| summer-season | concatenation of June, July, and August of 2023          |
| winter-season | concatenation of December, January, and February of 2023 |
| summer-month  | concatenation of August data of period 2014-2023         |
| winter-month  | concatenation of February data of period 2014-2023       |

Table: learning scenarios \label{table:scenarios}

## Implementation

All code and data is available in a github repository [@github].  All deep learning models were implemented using the pytorch python package.  We recap the most important files here : 

- `building_blocks.py` contains all pytorch modules and models, and other support code to execute different scenarios.
- `datasets.py` contains the code to load and aggregate the Elia data into one pytorch dataloader.
- `figures.ipynb` is a jupyter notebook that contains code to generate figures.
- `stats.ipynb` is a jupyter notebook that contains code to check statistical validity.
- `scenario-base-transformer.ipynb` is a jupyter notebook focusing on the execution of scenario using a vanilla transformer
- TODO andere notebooks

# Evaluation and Results

## Evaluation 

TODO beschrijven wat we exact willen meten en hoe dit te meten (loss) (accuracy?)

To evaluate whether ... TODO ... self-attention ... , we formulate the following H~0~ hypothesis : 

> **H~0~ : A self-attention mechanism using XYZ is not better at predicting ... than regular self-attention .**

If the p-value is below $\alpha$ = 0.05, we can reject H~0~ and accept the alternative hypothesis, that there is indeed a difference between the TODO.  

-> vergelijken met base line voorspellingen elia?
-> regressieanalyse van de residuals.

## Results

### Scenario 1 : summer-season

This scenario uses data for June, July and August of 2023 for forecasting.  Results are summarized in Table \ref{table:ttest-original}.

| mechanism | metric mean | AM-1 mean | t-test value |  p-value | H~0~ rejected |
|-----------|------------:|----------:|-------------:|---------:|:-------------:|
| MA-2      |      0.9580 |    0.9864 |    -1.77E+01 | 2.29E-32 | yes           |
| MA-3      |      0.9858 |    0.9864 |    -1.84E+01 | 1.17E-33 | yes           |
| MA-4      |      0.9782 |    0.9864 |    -2.58E+01 | 9.62E-46 | yes           |


Table: one sample t-test to determine whether TODO \label{table:ttest-original}

TODO resultaten beschrijven

### Scenario 2 : winter-season

TODO idem hierboven

### Scenario 3 : summer-month

TODO idem hierboven

### Scenario 4 : winter-month

TODO idem hierboven

# Conclusions and Discussion

In this study, we have used xyz dataset and pre-processed thus and thus.

We evaluated x self-attention mechanisms, x, y and z in x different scenarios.  Results were : 

- result 1
- result 2

TODO some discussion

TODO future work

# References

::: {#refs}
:::

# Appendix A : Data general properties

![Typical recurrent patterns, here for September 2023](figures/recurrent-pattern-september2023.png){#fig:recurrent-pattern-september}

![Typical recurrent patterns, here for January 2023](figures/recurrent-pattern-january2023.png){#fig:recurrent-pattern-january}


# Appendix B : Average test losses (base transformer) 

| layers | heads | forward expansion | average test loss |
|-------:|------:|------------------:|------------------:|
| 2      |     4 |               256 |              TODO |
| 2      |     8 |               256 |              TODO |
| 4      |     4 |               256 |              TODO |
| 4      |     8 |               256 |              TODO |
| 6      |     4 |               256 |              TODO |
| 6      |     8 |               256 |              TODO |
| 2      |     4 |               512 |              TODO |
| 2      |     8 |               512 |              TODO |
| 4      |     4 |               512 |              TODO |
| 4      |     8 |               512 |              TODO |
| 6      |     4 |               512 |              TODO |
| 6      |     8 |               512 |              TODO |


Table: Average test losses for base transformer hyperparameter variations \label{table:avg-test-losses-base-transformer}
