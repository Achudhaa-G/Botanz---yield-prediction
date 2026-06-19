# Botanz---yield-prediction
A web application that uses Random forrest algorithm to predict the yield amount of the particular crop.

HOW THE ALGORITHM WORKS 

1 LAND PRICE PREDICTION:

 considering the features like 
	location 
	land size 
	zonal structure 
	proximity of amenities 
	Market trends
 ONTO THE METHOD 
 
Data collection - the features listed above 
|
Data preprocessing - explained in implementation 
|
Random forest training 
|
Model prediction - after training the RF makes prediction
	by aggregating the prediction of all trees 
	final prediction is avg of all trees 
| 
Model evaluation - calculating MAE and MSE 
	along with R-squared 
	R squared tells us the variance in price 
|
feature importance - provides insight to stakeholders 

and the flow of work concludes there 
now onto the depths of second hand understanding 

Hypothetically 
take that we have a dataset with land and attribute with 
prices 

	RF will learn from these features and make 	
	predictions for new data based on the pattern 
	it has identified on the hypo-dataset 

THAT CONCLUDES THE EXPLANATION FOR LAND PRICE

2 CROP YIELD FORECASTING 

the steps are the same as the Land price 
without any major changes 

the only change would be the features we consider 

features we look out for :
 
soil moisture 
temperature
rainfall in area
pest incidence 
crop type 
soil type 
fertilizer use
historical yield data



Implementation.

Implementing Random forest algorithm 

the Idea is to use the algorithm for:
 land price prediction and crop yield forcasting

a step by step process of implementing the algorithm 

explaination:

1 setting up the env 
	pandas for data manipulation , numpy for Num-ops
	sklearn for ML algorithms , Matplotlib for project

2 loading the data set 
	loading the dataset using pandas. 
	the data set should contain features such as:
		soil moisture , temperature 
		distance to amenities 
		land size and a target variable 

3 preprocessing the data: 
	cleaning and preparation are crucial afterall 
	
	Handling missing values 
	encode categorical variables
	feature scaling 

4 Spliting the data into training and testing dataset 
	split the data into two sets with ratio of 
	80:20 as training : test datasets respectively 

5 Training the model 
	import RandomForestReggressor 
	Hyperparameter tuning : we can optimize the 
	performance of the model by tuning parameters 
	these can be done with techniques such as 
		Grid search 
		Randomized search 

6 Evaluate the model 
	once the model is trained we evaluate it using
	testset and find MAE mean absolute error 
	MSE mean squared error 
	R-squared 

7 Visualize the results 

8 Making predictions on new dataset
	after training and testing , we can officially 
	use it on new unseen datas.

 
