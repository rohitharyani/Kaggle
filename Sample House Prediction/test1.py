import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Path of the file to read
iowa_file_path = "E://Job Search//Learning//Machine Learning//Kaggle//Sample House Prediction//train.csv"

home_data = pd.read_csv(iowa_file_path)

#Create target object 
y = home_data.SalePrice

#List of features 
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

#Split in to validation and training
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state =1)

#Specify model
iowa_model = DecisionTreeRegressor(random_state=1)

#Fit model
iowa_model.fit(train_X,train_y)

#Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print(f"Validation MAE when not specifiying max_leaf_nodes: {val_mae}")

#using best value for max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
	model.fit(train_X, train_y)
	pred_val = model.predict(val_X)
	mae = mean_absolute_error(val_y, pred_val)
	return mae

candidate_max_leaf_nodes = [5,25,50,100,250,500]
scores = {i: get_mae(i,train_X,val_X,train_y,val_y) for i in candidate_max_leaf_nodes}
best_tree_size = min(scores,key=scores.get)

iowa_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state =1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)
print(f"Validation MAE for Best Value of Leaf Nodes: : {val_mae}")

#Define the model for Random Forest
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X,train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print(f"Validation MAE for Random Forest Model: {rf_val_mae}")

#Random Forest on Full Data
rf_model_full_data = RandomForestRegressor(random_state=1)
rf_model_full_data.fit(X,y)

#Apply above model on test data

test_data_path = "E://Job Search//Learning//Machine Learning//Kaggle//Sample House Prediction//test.csv"
test_data = pd.read_csv(test_data_path)

test_X = test_data[features]
test_preds = rf_model_full_data.predict(test_X)

output = pd.DataFrame({"Id": test_data.Id, "SalePrice": test_preds})
print(output)
output.to_csv("test1_output.csv", index=False)