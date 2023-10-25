import joblib
import pandas as pd

df =pd.read_csv(r'C:\Users\sharr\OneDrive\Desktop\small project\a.csv')

df.columns = df.columns.str.replace(' ', '_')


temp=df[df['Loan_Status']=='Fully Paid'][:22639]
temp1=df[df['Loan_Status']=='Charged Off'][:22639]
x={'Loan_Status':[],
    'Current_Loan_Amount':[],
    'Term':[],
    'Credit_Score':[],
    'Annual_Income':[],
    'Years_in_current_job':[],
    'Home_Ownership':[],
    'Purpose':[],
    'Monthly_Debt':[],
    'Years_of_Credit_History':[],
    'Months_since_last_delinquent':[],
    'Number_of_Open_Accounts':[],
    'Number_of_Credit_Problems':[],
    'Current_Credit_Balance':[],
    'Maximum_Open_Credit':[],
    'Bankruptcies':[],
    'Tax_Liens':[]}
data = pd.DataFrame(x)
data = pd.concat([temp1, temp], ignore_index=True)
m=data['Credit_Score'].median()
data['Credit_Score'].fillna(m, inplace=True)
m1=data['Annual_Income'].mean()
data['Annual_Income'].fillna(m1, inplace=True)
data['Years_in_current_job'].fillna('5 years', inplace=True)
data['Months_since_last_delinquent'].fillna(method='ffill', inplace=True)
data['Bankruptcies'].fillna(method='bfill', inplace=True)
data['Tax_Liens'].fillna(method='ffill', inplace=True)



data.dropna(inplace=True)
print(data.isnull().sum())


unique_values = data['Loan_Status'].unique()
print(unique_values)
mapping1 = {
    'Fully Paid': '1',
    'Charged Off': '0'}
data['Loan_Status'] = data['Loan_Status'].map(mapping1)
unique_values = data['Term'].unique()
print(unique_values)
mapping2 = {
    'Short Term': '1',
    'Long Term': '0'}
data['Term'] = data['Term'].map(mapping2)
unique_values = data['Years_in_current_job'].unique()
print(unique_values)
mapping3 = {
    '8 years': 8,
    '< 1 year': 0,
    '2 years': 2,
    '3 years': 3,
    '10+ years':10,
    '4 years':4,
    '6 years':6,
    '7 years':7,
    '5 years':5,
    '1 year':1,
    '9 years':9}
data['Years_in_current_job'] = data['Years_in_current_job'].map(mapping3)
unique_values = data['Home_Ownership'].unique()
print(unique_values)
mapping4 = {
    'Own Home': 1,
    'Home Mortgage': 0,
    'Rent':2,
    'HaveMortgage':3}
data['Home_Ownership'] = data['Home_Ownership'].map(mapping4)
unique_values = data['Purpose'].unique()
print(unique_values)
mapping5 = {
    'Debt Consolidation': 1,
    'Buy House': 2,
    'other': 3,
    'Take a Trip': 4,
    'Home Improvements':5,
    'Other':3,
    'Buy a Car':6,
    'Medical Bills':7,
    'wedding':8,
    'Business Loan':9,
    'small_business':10,
    'major_purchase':11,
    'vacation':4,
    'Educational Expenses':12,
    'moving':13,
    'renewable_energy':14}
data['Purpose'] = data['Purpose'].map(mapping5)
data.drop('Loan_ID',axis=1,inplace=True)
data.drop('Customer_ID',axis=1,inplace=True)
df.drop('Customer_ID', axis=1,inplace=True)
df.drop('Loan_ID', axis=1,inplace=True)
import seaborn as sns
import matplotlib.pyplot as plt
m = df['Credit_Score'].median()
df['Credit_Score'].fillna(m, inplace=True)
m1 = df['Annual_Income'].mean()
df['Annual_Income'].fillna(m1, inplace=True)
df['Years_in_current_job'].fillna('5 years', inplace=True)
df['Months_since_last_delinquent'].fillna(method='ffill', inplace=True)
df['Bankruptcies'].fillna(method='bfill', inplace=True)
df['Tax_Liens'].fillna(method='ffill', inplace=True)

df.dropna(inplace=True)
print(df.isnull().sum())

unique_values = df['Loan_Status'].unique()
print(unique_values)
mapping1 = {
    'Fully Paid': '1',
    'Charged Off': '0'}
df['Loan_Status'] = df['Loan_Status'].map(mapping1)
unique_values = df['Term'].unique()
print(unique_values)
mapping2 = {
    'Short Term': '1',
    'Long Term': '0'}
df['Term'] = df['Term'].map(mapping2)
unique_values = df['Years_in_current_job'].unique()
print(unique_values)
mapping3 = {
    '8 years': 8,
    '< 1 year': 0,
    '2 years': 2,
    '3 years': 3,
    '10+ years': 10,
    '4 years': 4,
    '6 years': 6,
    '7 years': 7,
    '5 years': 5,
    '1 year': 1,
    '9 years': 9}
df['Years_in_current_job'] = df['Years_in_current_job'].map(mapping3)
unique_values = df['Home_Ownership'].unique()
print(unique_values)
mapping4 = {
    'Own Home': 1,
    'Home Mortgage': 0,
    'Rent': 2,
    'HaveMortgage': 3}
df['Home_Ownership'] = df['Home_Ownership'].map(mapping4)
unique_values = df['Purpose'].unique()
print(unique_values)
mapping5 = {
    'Debt Consolidation': 1,
    'Buy House': 2,
    'other': 3,
    'Take a Trip': 4,
    'Home Improvements': 5,
    'Other': 3,
    'Buy a Car': 6,
    'Medical Bills': 7,
    'wedding': 8,
    'Business Loan': 9,
    'small_business': 10,
    'major_purchase': 11,
    'vacation': 4,
    'Educational Expenses': 12,
    'moving': 13,
    'renewable_energy': 14}
df['Purpose'] = df['Purpose'].map(mapping5)

# Set the style for Seaborn plots
sns.set(style="whitegrid")

# Correlation heatmap
cor = df.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(cor, annot=True, cmap="coolwarm")

# Loan Status distribution pie chart
plt.figure(figsize=(6, 6))
df['Loan_Status'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Loan Status Distribution')


# Pie chart for Years in Current Job distribution
plt.figure(figsize=(6, 6))
df['Years_in_current_job'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Years in Current Job Distribution')

# Bar plot showing the relationship between Years in Current Job, Loan Status, and count
plt.figure(figsize=(10, 6))
sns.barplot(x='Loan_Status', y='total', hue='Years_in_current_job', data=df[['Years_in_current_job', 'Loan_Status']].value_counts().reset_index(name='total'))
plt.title('Years in Current Job vs. Loan Status')

# Pie chart for Bankruptcies distribution
plt.figure(figsize=(6, 6))
df['Bankruptcies'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Bankruptcies Distribution')

# Bar plot showing the sum of Annual Income by Loan Status
plt.figure(figsize=(8, 6))
sns.barplot(x='Loan_Status', y='Annual_Income', data=df, ci=None, estimator='sum')
plt.title('Annual Income vs. Loan Status (Sum)')

# Bar plot showing the mean Annual Income by Loan Status
plt.figure(figsize=(8, 6))
sns.barplot(x='Loan_Status', y='Annual_Income', data=df, ci=None, estimator='mean')
plt.title('Annual Income vs. Loan Status (Mean)')

# Histogram of Monthly Debt by Loan Status
plt.figure(figsize=(10, 6))
sns.histplot(df, x='Monthly_Debt', hue='Loan_Status', element='step', common_norm=False)
plt.title('Monthly Debt Distribution by Loan Status')

# Box plot for Years of Credit History by Loan Status
plt.figure(figsize=(8, 6))
sns.boxplot(x='Loan_Status', y='Years_of_Credit_History', data=df)
plt.title('Years of Credit History vs. Loan Status')

# Scatter plot of Credit Score vs. Monthly Debt with Loan Status coloring
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Credit_Score', y='Monthly_Debt', data=df, hue='Loan_Status')
plt.title('Credit Score vs. Monthly Debt with Loan Status')

plt.show()

from statistics import mode

# Find the mode of the 'Credit_Score' column
credit_score_mode = mode(data['Credit_Score'])
# Replace 4-digit and 2-digit values with the mode
data['Credit_Score'] = data['Credit_Score'].apply(lambda x: credit_score_mode if (len(str(x)) == 4 or len(str(x)) == 2) else x)




y = data['Loan_Status']
x = data.drop('Loan_Status', axis=1)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=45)

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier()
rf.fit(x_train,y_train)
predictions = rf.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")


joblib.dump(rf,'loan_prediction_model.joblib')