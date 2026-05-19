#Took some help or assistance from AI,GOOGLE AND NOTES(BRIEFLY DESCRIBED IN THE CODE AT WHICH PLACE IT WAS USED)
#We were not knowing which library to make a linear regression plot so we searched on google
#This is the website https://www.statsmodels.org/stable/regression.html
#We didnt know which library to import for checking correlation for that too we searched on google
#This is the website https://www.statsmodels.org/stable/generated/statsmodels.graphics.tsaplots.plot_acf.html

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

#The next step was loading the dataset file
df = pd.read_csv("DataBreaches_Cleaned.csv")

#We were confused on which model to built so we first tried it using Time series
#We wanted to groupby the dataset by year and we wanted total of the records because it will help us to see the change of severity over time
#This time series plotting and graph we took assistance from google and ai too because we were not having full knowledge about what is is actually
#This is the site which we took reference from https://www.tigerdata.com/blog/how-to-work-with-time-series-in-python
yearly_records = df.groupby("Year")["Records"].sum()
plt.close('all')
plt.figure(figsize=(12, 6))
plt.plot(yearly_records.index, yearly_records.values, marker="o", linewidth=2)
plt.title("Time Series Plot: Total Records Lost by Year")
plt.xlabel("Year")
plt.ylabel("Total Records Lost")
plt.grid(True)
plt.tight_layout()
plt.show()

#We took help from our professor in class and from google sites what correlation actually is and from that we made sure that Linear regression
#is perfect for our model analysis
#https://www.geeksforgeeks.org/python/how-to-calculate-autocorrelation-in-python/
plt.close('all')
plt.figure(figsize=(10, 5))
plot_acf(yearly_records)
plt.title("Autocorrelation Graph of Total Records Lost Over Time")
plt.tight_layout()
plt.show()

#In linear regression model we took reference from our notes that we need to decide one dependant and independat variable
#in our case year is independant variable and Records Lost Over Years is dependant variable
X = df["Year"]
y = df["Records"]  # FIXED COLUMN NAME

#We added this by taking a look at one of website from google that if we dont add an intercept our trend line cannot be adjusted or arranged properly
#https://medium.com/%40deepml1818/python-statistical-modeling-linear-and-logistic-regression-with-statsmodels-124b3da9c30c
X = sm.add_constant(X)

#This was also mentioned in the site which i provided above https://medium.com/%40deepml1818/python-statistical-modeling-linear-and-logistic-regression-with-statsmodels-124b3da9c30c
#We came to know after studying that this is used for analyzing the relationship between years and records
model = sm.OLS(y, X).fit()

# MODEL SUMMARY TABLE
print("\n SIMPLE LINEAR REGRESSION (YEAR ONLY) - MODEL SUMMARY TABLE ")
print(model.summary())

# REQUIRED METRICS ONLY
#We took some help for this code generation from ai because we were not aware of model.rsquared etc to be used)
print("\n SIMPLE MODEL EVALUATION METRICS ")
print("R²:", model.rsquared)
print("p-value for Year:", model.pvalues["Year"])
print("Coefficient (Slope for Year):", model.params["Year"])


# ADDED SECOND PREDICTOR (ORGANIZATION TYPE) FOR MULTIPLE REGRESSION MODEL
# Based on our Week 12 EDA and summary, we observed that some industries like web,
# online marketing, government and financial service companies have  higher
# total and average records lost. So as per the feedback and discussion we have and by looking some options on google too and choosing the best one we selected another predictor
# "Organization type" and included it in the model to check and see if it can make some noticeable and clean changes in our analysis.
# We took help of google and Chatgpt to understand how to convert categorical variables to
# numeric codes using pandas so they can be used in the regression model.

#This line helps in coding  the organization type into numeric codes so that we can use it in the regression model
df["Org_encoded"] = df["Organization type"].astype("category").cat.codes

#Now we build a multiple regression model using both Year and Org_encoded as predictors
#We searched on google that which other model we can use if we have the following points and they suggested this model fits best then to dive deep into what multi regression is we took the help of this website https://www.geeksforgeeks.org/machine-learning/ml-multiple-linear-regression-using-python/
#The following findings were made through Week 12's EDA:
#Industries with the largest breaches are predominantly Web, Online Marketing, Technology, Government, and Financial Services.
#Breaches are unequally distributed among industries in terms of severity.
#Most exposed records belong to just a handful of categories.
#These data indicate that there is a correlation between the type of industry and the effects of a breach, requiring us to create a statistical model to account for this variable when testing against other influences in our analytical process.
#As such, we developed a multiple regression model that includes the following variables:
#Records = β0 + β1(Year) + β2(Organization Type)
X_multi = df[["Year", "Org_encoded"]]
X_multi = sm.add_constant(X_multi)

#We used the same OLS method from statsmodels as above
#What is Ols method was searched by us via youtube the link is https://youtu.be/Z6oXPk6UDdo?si=f7U90KQBDJjUHWDi
multi_model = sm.OLS(y, X_multi).fit()

print("\n MULTIPLE LINEAR REGRESSION (YEAR + ORGANIZATION TYPE) - MODEL SUMMARY TABLE ")
print(multi_model.summary())

print("\n MULTIPLE MODEL EVALUATION METRICS ")
print("R²:", multi_model.rsquared)
print("p-value for Year:", multi_model.pvalues["Year"])
print("p-value for Org_encoded:", multi_model.pvalues["Org_encoded"])
print("Coefficient (Slope for Year):", multi_model.params["Year"])
print("Coefficient (Slope for Org_encoded):", multi_model.params["Org_encoded"])


#We again took some help from AI and google to understand how to read the summary output and R² values.

#This is the linear regression graph we took assistance from this site https://www.geeksforgeeks.org/machine-learning/linear-regression-python-implementation/
plt.close('all')
plt.figure(figsize=(15, 6))
#This helps in plotting individual breach events
sns.scatterplot(x=df["Year"], y=df["Records"], alpha=0.6)
#This line we took help from ai as we were not were of it after taking assistance we came to know that this is used
#because it adds a linear regression trend line in our graph
plt.plot(df["Year"], model.predict(sm.add_constant(df["Year"])), color="Black", linewidth=2)
plt.title("Linear Regression Trend Line (Year vs Records Lost)")
plt.xlabel("Year")
plt.ylabel("Records Lost Over Years")
#we found out from the google site mentioned above that grid and tight layout are very usefull for clean readabiltiy
#of the graph true for the background clarity and tight layout helps in overlapping the text
plt.grid(True)
plt.tight_layout()
plt.show()

# BAR CHART which best communicate our findings
#We took the code from our previous week assignment where we produced code by taking help from our notes and google for some part
top_methods = (
    df.groupby("Method")["Records"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
#This part of code we have already done in our previous week
plt.close('all')
plt.figure(figsize=(15, 6))
sns.barplot(x=top_methods.index, y=top_methods.values)
plt.title("Top 5 Breach Methods by Total Records Lost")
plt.xlabel("Breach Method")
plt.ylabel("Total Records Lost")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# HEATMAP which best communicate our findings
#The heatmap is a concise medium that communicate our results - showing via color just the right sets of organization types and breach methods that exhibit the most records lost, it also gives some graphical insight into the variability and outliers impact
#We came to know about heatmap and selected it because we already used in in our previous week assignment
##It prepares the data in a grid format that the heatmap can visualize.This usually helps in viewing the heatmap clearly took help from gpt and google
pivot = df.pivot_table(
    index="Organization type",
    columns="Method",
    values="Records",
    aggfunc="sum",
    fill_value=0
)

#This line finds the top 10 organisation type and breach methods with the highest total records lost across all methods
#fOR THIS SECTION OF CODE WE TOOK SOME ASSISTANCE FROM GOOGLE AS WE WERE UNAWARE OF HOW TO MAKE OUR HEATMAP LOOK CLEAN
top_orgs = pivot.sum(axis=1).sort_values(ascending=False).head(10).index
top_methods_heatmap = pivot.sum(axis=0).sort_values(ascending=False).head(5).index
#we used this because it select the top most organisation and breach methods and help us in making our heatmap look clean
pivot_final = pivot.loc[top_orgs, top_methods_heatmap]

#This is same as we used for other charts  and we took some help from our notes too.
plt.close('all')
plt.figure(figsize=(15, 6))
sns.heatmap(pivot_final, cmap="Blues", annot=True, fmt=".0f", linewidths=0.5)
plt.title("Records Lost by Organization Type & Breach Methods")
plt.tight_layout()
plt.show()
