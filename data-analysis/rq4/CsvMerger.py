import pandas as pd

def generate_input():
    training_df = pd.read_csv('../../data-collection/data/training.csv')
    static_df = pd.read_csv('../../data-collection/data/static.csv')
    summary_df = pd.read_csv('../../data-collection/data/summary.csv')
    combined_df = pd.merge(training_df, static_df, on='RunID', how = 'left')    
    final_df = pd.merge(combined_df, summary_df, on = 'RunID' , how = 'left')
    final_df.to_csv('input.csv', index = False)