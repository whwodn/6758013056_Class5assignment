import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'data/class5/class5data.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Drop the rows where 'Occupation' is 'Free Lancer'
df = df[df['Occupation'] != 'Free Lancer']

# Fix typos in the 'Gender' column regardless of capitalization
df['Gender'] = df['Gender'].replace(
    to_replace=[r'(?i)fe male', r'(?i)female'], value='Female', regex=True)

# Standardize 'MaritalStatus' by replacing 'Unmarried' with 'Single' regardless of capitalization
df['MaritalStatus'] = df['MaritalStatus'].replace(
    to_replace=[r'(?i)Unmarried'], value='Single', regex=True)

# Now we will proceed to label encode the categorical columns
label_encoder = LabelEncoder()

# 1. Gender: 'Male' = 0, 'Female' = 1
# df['Gender'] = label_encoder.fit_transform(df['Gender'])

# 2. Occupation: Label encode the occupation types
#df['Occupation'] = label_encoder.fit_transform(df['Occupation'])

# 3. MaritalStatus: Label encode marital status, where 'Single' -> 0, 'Married' -> 1, 'Divorced' -> 2
# df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])

# 4. ProductPitched: Label encode the type of product pitched
# this is tiered
#df['ProductPitched'] = label_encoder.fit_transform(df['ProductPitched'])

# 5. Designation: Label encode designation types
# also tiered
#df['Designation'] = label_encoder.fit_transform(df['Designation'])

# 6. TypeofContact: Label encode contact type
# df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])

# 7. CityTier: No need to modify, already numeric

# After transforming all categorical variables, save the transformed dataset
df.to_csv('data/class5/C5aclean.csv', index=False)

# Displaying the first few rows of the transformed dataset
print(df.head())