# Neural_Network_Charity_Analysis

## Overview of Analysis
  - This analysis set out to use machine learning/ neural Network algorithms to successfully predict whether certain companies would become successful/ or benefit from being funded by a company referred to as AlphabetSoup. Our goal was to take a data set, pre-process it then build and fit a neural network algorithm to this data set and try to predict the outcome of being successful/ not being successful (target) based on the features (feature variables) within the data set. I created an initial neural network model that was able to successfully predict a successful outcome at about 72.4% accuracy. After this, I then attempted 3 more times, using the same pre-processing techniques to achieve an accuracy level greater than 75%. For the 3 additional attempts I shifted the hyperparameters of my neural network model to try to achieve this goal, but was unable to get higher than 72.7% accuracy. 

## Results
### Data Preprocessing
  - What variable(s) are considered the target(s) for your model?
    - The target variable in this data set was the "IS_SUCCESSFUL" column or referred to the "y" variable in our model. This could also be considered the dependent variable. This column essentially was the determining    column, portraying a successful company as a 1, and the adverse to be a 0.

![Screenshot 2023-04-09 083231](https://user-images.githubusercontent.com/117245167/230772695-040f4d61-34ba-48e2-8b01-31e379fb22f9.png)

---
  - What variable(s) are considered to be the features for your model?
    - The Feature variables were all the other variables in the data set, other than that "IS_SUCCSESFUL" column. These would be considered the independent variables that help determine the value of the dependent variable ("IS_SUCCESSFUL"). These are the variables that we would utilize in our model to try to predict the value of the target variable. They are referred to as "X" in the model. There were 43 Feature variables, which were created from HotEncoding and Binning all categorical variables (non-numerical).
   
   ```
Index(['STATUS', 'ASK_AMT', 'APPLICATION_TYPE_Other', 'APPLICATION_TYPE_T10',
       'APPLICATION_TYPE_T19', 'APPLICATION_TYPE_T3', 'APPLICATION_TYPE_T4',
       'APPLICATION_TYPE_T5', 'APPLICATION_TYPE_T6', 'APPLICATION_TYPE_T7',
       'APPLICATION_TYPE_T8', 'AFFILIATION_CompanySponsored',
       'AFFILIATION_Family/Parent', 'AFFILIATION_Independent',
       'AFFILIATION_National', 'AFFILIATION_Other', 'AFFILIATION_Regional',
       'CLASSIFICATION_C1000', 'CLASSIFICATION_C1200', 'CLASSIFICATION_C2000',
       'CLASSIFICATION_C2100', 'CLASSIFICATION_C3000', 'CLASSIFICATION_Other',
       'USE_CASE_CommunityServ', 'USE_CASE_Heathcare', 'USE_CASE_Other',
       'USE_CASE_Preservation', 'USE_CASE_ProductDev',
       'ORGANIZATION_Association', 'ORGANIZATION_Co-operative',
       'ORGANIZATION_Corporation', 'ORGANIZATION_Trust', 'INCOME_AMT_0',
       'INCOME_AMT_1-9999', 'INCOME_AMT_10000-24999',
       'INCOME_AMT_100000-499999', 'INCOME_AMT_10M-50M', 'INCOME_AMT_1M-5M',
       'INCOME_AMT_25000-99999', 'INCOME_AMT_50M+', 'INCOME_AMT_5M-10M',
       'SPECIAL_CONSIDERATIONS_N', 'SPECIAL_CONSIDERATIONS_Y'],
      dtype='object')
   ```
   ---
  - What variable(s) are neither targets nor features, and should be removed from the input data?
    - I dropped two columns, which can be seen below. I dropped "EIN" and "NAME" from the data set as these are identifying features and really do not have any impact on the determination of a successful outcome/ not successful.
    
    ![Screenshot 2023-04-09 085941](https://user-images.githubusercontent.com/117245167/230774073-ce86e318-1dd0-4482-9070-00d46ef4bf6a.png)
--- 
### Compiling, Training, and Evaluating the Model
  - How many neurons, layers, and activation functions did you select for your neural network model, and why?
    1. I Attempted to achieve a model accuracy of 75% on four occasions. My first attempt was the initial model created (see below) as a part of the second deliverable of the assignment, tis model utilized two hidden layers and an output layer. with the two hidden layers having 80 and 30 nodes respectively. I utilized the "ReLU" activation function for the hidden layers, with "Sigmoid" as the activation function for the output layer. I used 100 epochs for this model.
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 80)                3520      
                                                                 
 dense_1 (Dense)             (None, 30)                2430      
                                                                 
 dense_2 (Dense)             (None, 1)                 31        
                                                                 
=================================================================
Total params: 5,981
Trainable params: 5,981
Non-trainable params: 0
_________________________________________________________________
```
   2. My second attempt / model utilized three hidden layers and an output layer. The three hidden layers having 80, 30, and 10 nodes respectively. I utilized the "ReLU" activation function for the hidden layers, with "Sigmoid" as the activation function for the output layer. I used 100 epochs for this model.
 ```
 Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 80)                3520      
                                                                 
 dense_1 (Dense)             (None, 30)                2430      
                                                                 
 dense_2 (Dense)             (None, 10)                310       
                                                                 
 dense_3 (Dense)             (None, 1)                 11        
                                                                 
=================================================================
Total params: 6,271
Trainable params: 6,271
Non-trainable params: 0
_________________________________________________________________
 ```
   3. The third attempt utilized two hidden layers and an output layer. The two hidden layers having 90 and 65 nodes respectively. I utilized the "ReLU" activation function for the hidden layers, with "Sigmoid" as the activation function for the output layer. I used 150 epochs for this model.
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_4 (Dense)             (None, 90)                3960      
                                                                 
 dense_5 (Dense)             (None, 65)                5915      
                                                                 
 dense_6 (Dense)             (None, 1)                 66        
                                                                 
=================================================================
Total params: 9,941
Trainable params: 9,941
Non-trainable params: 0
_________________________________________________________________
```
   
   4. Lastly, my fourth attempt utilized three hidden layers and an out put layer. With the three hidden layers having 200, 100, and 15 nodes respectively. I utilized the "ReLU" activation function for the hidden layers, with "Sigmoid" as the activation function for the output layer. I used 100 epochs for this model.
```
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_14 (Dense)            (None, 200)               8800      
                                                                 
 dense_15 (Dense)            (None, 100)               20100     
                                                                 
 dense_16 (Dense)            (None, 15)                1515      
                                                                 
 dense_17 (Dense)            (None, 1)                 16        
                                                                 
=================================================================
Total params: 30,431
Trainable params: 30,431
Non-trainable params: 0
_________________________________________________________________
```
---

  - Were you able to achieve the target model performance?
    - I was not able to achieve the target model performance of 75% accuracy. I was not able to achieve an accuracy score greater than roughly 72.7%.
    
---   
 
  - What steps did you take to try and increase model performance?
    - As you can see in the above images, I tried a couple of different approaches. Mainly I focused on the neural network model, and not the data by changing the hyperparameters of the model. I tried changing the number of hidden layers and nodes in each layer. I increased to 3 total layers at one point (which I understand to be a general rule of thumb) and generally increased the number of total nodes in the model as I went through the attempts. I also increased/ decreased the number of epochs as well. IN addition, I also tried to change the activation functions in a couple of the models. Instead of focusing on ReLU in the hidden layers I changed the final hidden layer to Sigmoid to match the output layer, but this did not make any huge impact on the accuracy result.
  
## Summary
  - My neural network models did not achieve the overall accuracy threshold we set out to achieve, of 75%. We were able to basically recreate the same accuracy score across all four of the models, which hovered around 72%. even though, from what I understand I changed some significant aspects of the model throughout the different attempts. This leads me to believe that I should have adjusted the data set in some way by either reducing columns or adding some back. Another direction could have been to either increase or decrease the number of bins when carrying out the preprocessing portion of the challenge. It is my understanding he model should be able to learn to overcome some of those shortcomings though. As for a different method or model that we could have implemented to tackle this specific challenge, I think a balanced Random Forrest classifier either in place of the neural network model or in addition to the model. I understand that this is also a very powerful model and can be used in place of a neural network model in some instances or even replace it.
