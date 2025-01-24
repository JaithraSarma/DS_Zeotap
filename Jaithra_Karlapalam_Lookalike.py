import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

merged_data = pd.merge(transactions, products[['ProductID', 'Category']], on='ProductID', how='inner')
merged_data = pd.merge(merged_data, customers, on='CustomerID', how='inner')

customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
customers['Tenure'] = (pd.to_datetime('today') - customers['SignupDate']).dt.days

encoder = OneHotEncoder(sparse_output=False)
region_encoded = encoder.fit_transform(customers[['Region']])
region_encoded_df = pd.DataFrame(region_encoded, columns=encoder.categories_[0], index=customers['CustomerID'])
customers_profile = pd.concat([customers[['CustomerID', 'Tenure']], region_encoded_df], axis=1)

transaction_agg = merged_data.groupby(['CustomerID', 'Category']).agg(
    total_spent=('TotalValue', 'sum'),
    transaction_count=('TransactionID', 'count')
).reset_index()

transaction_pivot = transaction_agg.pivot_table(
    index='CustomerID',
    columns='Category',
    values=['total_spent', 'transaction_count'],
    aggfunc='sum',
    fill_value=0
)

transaction_pivot.columns = [f"{col[1]}_{col[0]}" for col in transaction_pivot.columns]
customer_data = pd.merge(customers_profile, transaction_pivot, left_on='CustomerID', right_index=True)

customer_data = customer_data.fillna(0)

scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data.drop(columns='CustomerID'))

similarity_matrix = cosine_similarity(customer_data_scaled)
similarity_df = pd.DataFrame(similarity_matrix, index=customer_data['CustomerID'], columns=customer_data['CustomerID'])

top_20_customers = customers['CustomerID'][:20]
lookalikes = []

for customer in top_20_customers:
    similarity_scores = similarity_df[customer]
    similarity_scores = similarity_scores.drop(customer)
    top_similar = similarity_scores.nlargest(3)
    for sim_customer, score in zip(top_similar.index, top_similar.values):
        lookalikes.append([customer, sim_customer, score])

lookalike_df = pd.DataFrame(lookalikes, columns=["CustomerID", "LookalikeCustomerID", "SimilarityScore"])

lookalike_df.to_csv('Jaithra_Karlapalam_Lookalike.csv', index=False)

print("Lookalike.csv has been generated successfully!")
