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

The original transformer architecture introduces a quadratic time and space complexity.  Much work has been done to improve on this.  The LogSparse transformer is introduced in [@paper], which reduces the memory cost to $O(L {(\log {}L)}^2)$.  The informer model [@informer] even achieves $O(L {\log {}L)}$ memory complexity.  In this report we will focus on attention mechanisms in the context of time series forecasting, ignoring space and time complexity of the transformer algorithm.

# Goal

In this paper, we focus on using transformers for time series forecasting. We aim to compare different attention mechanism and determine which mechanism best captures the outcome of past events.  We formulate a first research question : 

> **RQ 1 : When comparing regular self-attention, convoluted self-attention, TODO, TODO, which mechanism best predicts future values using accuracy as metric?**

The Elia dataset used is fully described in [the dataset description section](#sec:dataset).  It not only contains time series data, but also day+1 and day+7 predictions of the same data.  We formulate a second research question : 

> **RQ 2 : Is the accuracy of a transformer model as good as the Elia prediction model?**

Firstly, this report will look at the characteristics of the dataset used and discuss pre-processing steps.  Then, we will consider several attention mechanisms,  discuss design and implementation details and finally evaluate the performance of these attention mechanisms on the dataset.

# Data analysis

## Dataset description {#sec:dataset}

We use data from Elia [@elia], which operates the electricity transmission network in Belgium.  In particular, we use the solar power forecast datasets.  These contain time series of actual measured power in megawatt (MW), and also  day+1 and day+7 predictions of solar power output in MW.  Data is available for a period of 12 years (February 2012 until now) in monthly datasets.  Measurements and predictions are recorded every quarter of an hour.  The measured value is always the amount of power equivalent to the running average measured for that particular quarter-hour.  The layout of the dataset is fully described here [@dataset].  We recap the most important points here.

TODO iets zeggen over welke maanden we selecteren?

-> verschillende scenario's : 
- scenario 1 : zelfde maand (augustus) over alle jaren heen
- scenario 2 : 3 maanden van een seizoen (zomer) voor 1 jaar
- scenario 3 : zelfde maand (december) over alle jaren heen
- scenario 4 : 3 maanden van een seizoen (winter) voor 1 jaar

te varieren met input embedding size 5 - 10 - 20 dagen?


| feature          | description                           | range                            |
|:-----------------|:--------------------------------------|:---------------------------------|
| DateTime         | Date and time per quarter hour        | [00:00 - 24:00] in quarter hours |
| Measurement      | Measured solar power production in MW | [0.0 - 6000.0]                   |
| Day+1 prediction | D+1 solar power forecast in MW        | [0.0 - 6000.0]                   |
| Day+7 prediction | D+7 solar power forecast in MW        | [0.0 - 6000.0]                   |

Table:  Features captured per quarter-hour in @dataset \label{table:features}

TODO extra features / embedding
-> feature + positional encoding + one-hot encoding van dag of maand of week "temporal encoding"

## Data general properties

Data is highly regular and contains obvious day - night recurring patterns.  Since we are using solar power production data, data typically shows no values in the early morning, building towards a peak around noon, and then slowly reducing values towards the evening.  This is illustrated in Figure @{fig:recurrent-pattern}.


![Typical recurrent patterns, here for September 2023](figures/recurrent-pattern.png){#fig:recurrent-pattern}

TODO spreken over regularity.  zon, dag / nacht, ...  
 -> checken normaal verdeeld

## Data pre-processing

TODO omschrijven van de data pre-processing stappen die we gaan nemen.  bv selectie van data (maanden), aggregeren tot uren / dagen / weken?  concatenatie van waarden per maand?  per zomer?  op welke lengte opslitsen?   nachtelijke uren eruit halen?

-> aggregaties te voorzien :

- geen 
- hourly
- om de 4 uur 
- om de dag

### Outlier analysis {#sec:outlier}

-> check doen op foute waarden, uitschieters
-> we gaan normaal gezien geen waarden verwijderen 


# Methodology and Implementation

## Research methodology

We started by examining the dataset provided [@dataset].  

Beschrijven van de verschillende stappen.

All analysis was done using the pytorch python package.

## Design elaboration

We decided to evaluate the following attention mechanisms : 

- regular self-attention
- convoluted self-attention as described in [@paper]
- asymetric convoluted self-attention 
- XYZ

-> kijken naar RCNN?
-> eigenvectoren?

TODO experiment beschrijven naar dataset of datasets, parameters (bv kernel size) en hyperparameters (indien van toepassing)

hyperparameters : 
-> kernel size 
-> attention head (softmax tussen key en query)
-> gridsearch, random search of half random search (scikit learn)?

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

-> vergelijken met base line voorspellingen elia?
-> regressieanalyse van de residuals.

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

