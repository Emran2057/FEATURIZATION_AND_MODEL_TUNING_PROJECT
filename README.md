<h1>Problem Statement provided by Great Learning</h1>

<h2>Data Description:</h2>
<p>
  The actual concrete compressive strength (MPa) for a given mixture under a specific age (days) was determined from laboratory. Data is in raw form (not scaled). 
 <br>
  The data has 8 quantitative input variables, and 1 quantitative output variable, and 1030 instances (observations).
</p>

<h2>Domain:</h2>
<p>Cement Manufacturing</p>

<h2>Context:</h2>
Concrete is the most important material in civil engineering. The concrete compressive strength is a highly nonlinear function of age and ingredients. These ingredients include cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate.

<h2>Attribute Information:</h2>

* Cement : measured in kg in a m3 mixture
* Blast : measured in kg in a m3 mixture
* Fly ash : measured in kg in a m3 mixture
* Water : measured in kg in a m3 mixture
* Superplasticizer : measured in kg in a m3 mixture
* Coarse Aggregate : measured in kg in a m3 mixture
* Fine Aggregate : measured in kg in a m3 mixture
* Age : day (1~365)
* Concrete compressive strength measured in MPa

<h2>Learning Outcomes:</h2>

* Exploratory Data Analysis
* Building ML models for regression
* Hyper parameter tuning

<h2>Objective:</h2>
Modeling of strength of high performance concrete using Machine Learning

<h2>Steps and tasks:</h2>

1. Deliverable -1 (Exploratory data quality report reflecting the following)

a. Univariate analysis

i. Univariate analysis – data types and description of the independent attributes which should include (name, meaning, range of values observed, central values (mean and median), standard deviation and quartiles, analysis of the body of distributions / tails, missing values, outliers)

b. Multivariate analysis
i. Bi-variate analysis between the predictor variables and between the predictor variables and target column. Comment on your findings in terms of their relationship and degree of relation if any. Presence of leverage points. Visualize the analysis using boxplots and pair plots, histograms or density curves. Select the most appropriate attributes

c. Pick one strategy to address the presence outliers and missing values and perform necessary imputation

2. Deliverable -2 (Feature Engineering techniques)

a. Identify opportunities (if any) to create a composite feature, drop a feature etc.

b. Decide on complexity of the model, should it be simple linear
model in terms of parameters or would a quadratic or higher
degree help

c. Explore for gaussians. If data is likely to be a mix of gaussians, explore individual clusters and present your findings in terms of the independent attributes and their suitability to predict strength

3. Deliverable -3 (create the model )

a. Obtain feature importance for the individual features and present your findings

4. Deliverable -4 (Tuning the model)

a. Algorithms that you think will be suitable for this project

b. Techniques employed to squeeze that extra performance out of the model without making it overfit or underfit

c. Model performance range at 95% confidence level

<h2>Results/ Accuracy obtained in this project:</h2>

We can conclude that RMSE is low for test data so it show that it is generalized and R squared is 0.89 so we can say that 89% is the accuracy of our hyperparameter tune Random Forest Regressor model compared to other models such as Linear regression, Decision Tree Regressor and SVR.

<h2>Note:</h2>

* There is one csv file for data and other two is jupyter nootebook and python file.
* You can run this code one of this file by downloading or copy it in any plateform which support python.
* You can not see the boxplot in github because it does not support plotly library but you can see it by pasting the link of my GitHub notebook into http://nbviewer.jupyter.org/.*