import pandas as pd

# Step 1: Load `news.csv` and filter rows where 'platform' == '微博'
news_df = pd.read_csv('news.csv')
A = news_df[news_df['platform'] == '微博']

# Step 2: Load `social_context.csv` and filter rows where 'news_id' is in A
social_context_df = pd.read_csv('social_context.csv')
B = social_context_df[social_context_df['news_id'].isin(A['news_id'])]
B = B[['news_id', 'content']]
# Step 4: Save the result
B.to_csv('clean-social_context.csv', index=False)
