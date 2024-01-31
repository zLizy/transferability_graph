import pandas as pd

def fill_null_value(df,columns,value='mean'):
    for col in columns:
        if value == 'mean':
            df[col].fillna((df[col].mean()), inplace=True)
        else:
            df[col].fillna((value), inplace=True)
    return df

df_records = pd.read_csv('../../doc/records.csv',index_col=0)
df_performace = df_records[['model','finetuned_dataset','test_accuracy']]
df_pivot = df_performace.pivot(index='model',columns='finetuned_dataset',values='test_accuracy')
df_pivot = df_pivot.drop(columns=['food101'])

df_pivot = fill_null_value(df_pivot,columns=['smallnorb_label_azimuth','smallnorb_label_elevation'])

df_pivot.to_csv('results/model_list_full.csv')
df_pivot = df_pivot.dropna()
df_pivot.to_csv('results/model_list.csv')