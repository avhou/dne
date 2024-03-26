---
title: "IM1102-232433M - Deep Neural Engingeering assignment 2"
subtitle: "Modifying the attention mechanism of transformers for time series forecasting"
author: "Arne Lescrauwaet (852617312) - Joachim Verschelde (TODO) - Alexander Van Hecke (852631385)"
institute: "OU"
date: \today
geometry: margin=2.5cm
numbersections: false
link-citations: true
link-bibliography: true
papersize: a4
---


# Introduction

This report details the steps taken by Arne Lescrauwaet (852617312), Joachim Verschelde (TODO) and Alexander Van Hecke (852631385) for the second assignment of the 2023 Deep Neural Engineering course organised by the Open University [@dne].

For this assignment we look at different attention mechanisms in transformers [@transformer] for use with time series data.  The attention mechanism enables a transformer model to selectively focus on relevant parts of the input data.  The goal is to be able to capture long range dependencies and relationships between items of the input data.  This is particularly important for time series data containing recurring patterns, e.g. hourly traffic counts on busy highways and hourly power consumption of nations.  We expect these types of data to contain clear and recurring patterns (i.e. traffic will typically be lower during weekends) and we want an attention mechanism to capture these patterns.  In addition to capturing recurring patterns, we would also like to be able to capture the ``local context'' of a pattern to predict new values.  That is, when encountering an event that is similar to a past event, we want to take the outcome of that past event into account in our prediction.

Different kinds of attention mechanisms exist.  Convolutional self-attention is introduced in [@paper], which aims to capture the local context of input events, but does this using a symmetric convolution, thereby taking both input data leading to a particular event and the outcome of that event into account.  A dual-stage attention mechanism is used for a Recurrent Neural Network (RNN) architecture in [@dualstage], using an input attention mechanism in an encoder step, and a temporal attention mechanism in a decoder step.

Even though transformers were originally designed in the field of natural language processing (NLP), a lot of work has been done to use transformers with time series data.  An overview of different ways to adapt transformers to time series data is given in [@timeseries1].  The time2vec encoding mechanism is introduced in [@timeseries2].  The authors of this paper use transformer models to predict stock prices, and claim these models can be used both for short and long term predictions.  The effectiveness of applying transformers to time series data is tested in [@timeseries3].

TODO hier korte literature review
TODO nog meer zoeken naar verschillende attention mechanismen.


# Goal

In this paper, we focus on using transformers for time series forecasting.  We aim to compare different attention mechanism and determine which mechanism best captures the outcome of past events.

> **RQ : When comparing regular self-attention, convoluted self-attention, xyz self-attention, xyz2 self-attention, which mechanism best predicts future values?**

TODO nagaan wat we juist willen onderzoeken?  accuracy?  best outcomes capteren (zonder naar resultaat te kijken zegt dit niet veel)?

Firstly, this report will look at the characteristics of the dataset used and discuss pre-processing steps.  Then, we will consider several attention mechanisms,  discuss design and implementation details and finally evaluate the performance of these attention mechanisms on the dataset.

TODO nagaan of we idd performance willen meten.

# Data analysis

## Dataset description {#sec:dataset}

We use data from Elia, which operates the electricity transmission network in Belgium.  In particular, we use the solar power forecast datasets.  These contain time series of actual measured power in megawatt (MW), and also 1 day ahead and 7 day ahead predicted solar power output in MW.  Data is available for a period of 12 years (February 2012 until now) in monthly datasets.  Measurements and predictions are recorded every quarter of an hour.  The measured value is always the amount of power equivalent to the running average measured for that particular quarter-hour.  The layout of the dataset is fully described here [@dataset].  We recap the most important points here.

TODO iets zeggen over welke maanden we selecteren?


| feature                 | description                           | range                            |
|:------------------------|:--------------------------------------|:---------------------------------|
| DateTime                | Date and time per quarter hour        | [00:00 - 24:00] in quarter hours |
| Measurement             | Measured solar power production in MW | [0.0 - 5000.0]                   |
| Prediction 1 day ahead  | D+1 forecast in MW                    | [0.0 - 5000.0]                   |
| Prediction 7 days ahead | D+7 forecast in MW                    | [0.0 - 5000.0]                   |

Table:  Features captured per quarter-hour in @dataset \label{table:features}

TODO outliers

The features are organized as a time series of quarter-hour values.

TODO welke feature of features gaan we bekijken?  Indien solar pv, kijken we ook naar de forecasts van elia zelf?


## Data general properties

TODO spreken over regularity.  zon, dag / nacht, ...  correlation is hier allicht geen concern als we maar 1 feature zouden in beschouwing nemen

## Data pre-processing

TODO omschrijven van de data pre-processing stappen die we gaan nemen.  bv selectie van data (maanden), aggregeren tot uren / dagen / weken?  concatenatie van waarden per maand?  per zomer?  op welke lengte opslitsen?   nachtelijke uren eruit halen?

### PCA analysis

TODO NVT?

### Duplicates analysis

TODO NVT?

### Outlier analysis {#sec:outlier}

TODO NVT?


# Methodology and Implementation

## Research methodology

We started by examining the dataset provided [@dataset].  

Beschrijven van de verschillende stappen.

All analysis was done using the pytorch python package.

## Design elaboration

We decided to evaluate the following attention mechanisms : 

- regular self-attention
- convoluted self-attention as described in [@paper]
- XYZ
- XYZ

TODO experiment beschrijven naar dataset of datasets, parameters (bv kernel size) en hyperparameters (indien van toepassing)

| algo                      | parameter   | range |
|:--------------------------|:------------|------:|
| regular self-attention    | ?           |     ? |
| convoluted self-attention | kernel size |  2-10 |
| XYZ                       | ?           |     ? |

Table: parameters used for the attention mechanisms \label{table:parameters}

TODO indien meerdere scenarios (bv meerdere datasets, of een scenario zomer/winter/...)  hier opsommen van scenarios

This entire design is repeated for a number of different scenarios.  We detail these in Table \ref{table:scenarios}.

| scenario | description              |
|:---------|:-------------------------|
| summer   | TODO july - september    |
| winter   | TODO december - february |

Table: learning scenarios \label{table:scenarios}

## Implementation

All code is available in a github repository [@github].

TODO eventueel speciale vermeldingen rond implementatie

# Evaluation and Results

## Evaluation 

TODO beschrijven wat we exact willen meten en hoe dit te meten (loss) (accuracy?)

To evaluate whether ... TODO ... self-attention ... , we formulate the following H~0~ hypothesis : 

> **H~0~ : A self-attention mechanism using XYZ is not better at predicting ... than regular self-attention .**

If the p-value is below $\alpha$ = 0.05, we can reject H~0~ and accept the alternative hypothesis, that there is indeed a difference between the TODO.  

## Results

### Scenario 1 : Regular self-attention

This is the baseline scenario.  Results are summarized in Table \ref{table:ttest-original}.

| metric   | metric mean | population mean | t-test value |  p-value | H~0~ rejected |
|----------|------------:|----------------:|-------------:|---------:|:-------------:|
| accuracy |      0.9580 |          0.9864 |    -1.77E+01 | 2.29E-32 | yes           |
| accuracy |      0.9858 |          0.9864 |    -1.84E+01 | 1.17E-33 | yes           |
| accuracy |      0.9782 |          0.9864 |    -2.58E+01 | 9.62E-46 | yes           |


Table: one sample t-test to determine whether TODO \label{table:ttest-original}

TODO resultaten beschrijven

### Scenario 2 : Convoluted self-attention

TODO idem hierboven

### Recap

TODO hier summary van scenarios

| scenario         | attention mechanism    | TODO              |
|:-----------------|:-----------------------|:------------------|
| summer           | regular self-attention | ?                 |
| summer           | convoluted self-attention | ?                 |
| winter           | regular self-attention | ?                 |
| winter           | convoluted self-attention | ?                 |

Table: Features importance summary \label{table:features-importance}


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

