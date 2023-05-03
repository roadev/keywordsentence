import pandas as pd

input_file = 'Resume.csv'
df = pd.read_csv(input_file)

filtered_df = df[df['Category'] == 'INFORMATION-TECHNOLOGY']

output_file = 'filtered_resumes.csv'
filtered_df.to_csv(output_file, index=False)
