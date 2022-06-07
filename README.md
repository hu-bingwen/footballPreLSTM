Bingwen Hu, Dau Cheng 

# Football Match Probability Prediction

<img src="https://activeforlife.com/img/large-webp/2018/07/soccer-ball-2121x1414.jpg" alt="drawing" width="400"/> 



## Context and Datasets

The dataset and original idea of this project is retrieved from a soccer prediction competition on kaggle presented by Octosport and Sportmonks. The competition is about predicting the probabilities of more than 150,000 match outcomes from 2019 to 2021 using the recent sequence of 10 matches of the teams.

And our goal is to apply machine learning models on the given dataset and try to compare the result with the bookmakers odds. 



## Approach

We plan to use Python as our main programming language with Pandas, numpy and matplotlib package for data wrangling and visualization. Scikit-Learn、 Tensorflow(LSTM model) are leveraged for data training, prediction and validation. Concretely, the scikit-Learn is utilized to scale data, implement cross validation and also compute log-loss to validate the prediction. Tensorflow is a platform to implement our machine learning models, we eventually chose a Long short-term memory model, the LSTM model to train the datasets.



## Datasets

https://www.kaggle.com/competitions/football-match-probability-prediction/data



## Model

Before we apply the LSTM model, the messy raw datasets should be cleaned and organized in the way of contributing to the training fitting and prediction. Also, the feature engineering is the most important part of Machine Learning. Finding decisive features can influence the result more than tuning the coefficient in machine learning models. 

The objectives for feature engineering part are:
*   Drop N/A matches
*   Convert "date" columns to datetime format
*   Create more features:
    *   days difference
    *   Goals difference
    *   Combine result
    *   Rating manipulation
    *   Combine Coach/League



**Models Selection**: 
Below are some models that we tested in this project. The simple LSTM is working fine, so we built our final model based on it. An extensive and complex NN model that includes more than 3 BILSTM layers worked terribly and took a lot of time. The convolution layer was added while testing but did not include in our final structure since the result is not that different. Also, we do not find a reasonable physical meaning for adding a convolution layer to our model based on the data. 

We also tested the GRU model since we learned that it is a lighter and faster version of LSTM, so hoping it could save our time. It turns out that it does help, but the real game-changer is using GPU to compute. 

Some example of our testing model structure: 
*   Single BILSTM layer + denselayer
*   3+ BILSTM layer + denselayer
*   2 BIGRU layer + denselayer
*   BILSTM layer without dropout/Flatten



## Result

The validiation log-loss of our model is around 1.01 and the accuracy is around 50%. Considered the bookmakers' model given by the competition holder is around 0.99, we think this is a reasonable result but still have a chance to improve. 

## Reference
[1] Football Match Probability Prediction | Kaggle (www.kaggle.com/competitions/football-match-probability-prediction/overview) 
[2] Emmanuel, T., Maupong, T., Mpoeleng, D. et al (2021). A survey on missing data in machine learning. J Big Data 8, 140 
[3] D. Prasetio and D. Harlili (2016). Predicting football match results with logistic regression International Conference On Advanced Informatics: Concepts, Theory And Application,  1-5 
[4] Rahul Baboota, Harleen Kaur (2019). Predictive analysis and modelling football results using machine learning approach for English Premier League, International Journal of Forecasting, 35(2), 741-755 
[5] K. Huang and W. Chang (2010). A neural network method for prediction of 2006 World Cup Football Game The 2010 International Joint Conference on Neural Networks, 1-8. 
[6] lfredo, Y.F., & Isa, S.M. (2019). Football Match Prediction with Tree Based Model Classification. International Journal of Intelligent Systems and Applications. 
[7] Qiyun Zhang, Xuyun Zhang et al. (2021). Sports match prediction model for training and exercise using attention-based LSTM network, Digital Communications and Networks
[8] https://en.wikipedia.org/wiki/Long_short-term_memory 
[9] Mohammed Alhamid. LSTM and Bidirectional LSTM for Regression 
[10] http://dprogrammer.org/rnn-lstm-gru
