import seaborn as sns 
import pandas as pd
import matplotlib.pylab as plt


title = 'performance_with_model_features'
file = f'../doc/{title}.csv'
df = pd.read_csv(file,index_col=0)

dataset_list = [
    'kitti','cifar100', 'dmlab', 'diabetic_retinopathy_detection', 'svhn_cropped', 
    'patch_camelyon'#, 'eurosat', 'oxford_iiit_pet', 'oxford_flowers102', 'resisc45',
    # 'dtd', 'sun397', 'caltech101','clevr'
]
df['AUC'] = df['test_AUC']

# g = sns.catplot(
#         df,
#         hue='finetune_ratio',
#         y='AUC',
#         # col='contain_data_similarity',
#         # row='test_dataset',
#         kind="box",
#         x='accuracy_thres'
#     )
# g.figure.savefig(f"../plot/AUC_{title}.png")
# g.figure.savefig(f"../plot/AUC_{title}.pdf")


df = pd.read_csv('../doc/rank.csv')
# df['diff'] = df['avg_gt_accu'] - df['avg_pred_accu']
df['diff'] = df['max_gt_accu'] - df['max_pred_accu']
# df = df[df['topK'] == 5]
df.loc[df['intersection'].notna(),'bool_intersection'] = True
df.loc[df['intersection'].isna(),'bool_intersection'] = False
for i, dataset in enumerate(dataset_list):
    df_sub = df[df['test_dataset']==dataset]
    try:
        g = sns.catplot(
            data=df_sub, x="AUC", y="diff",
            col="bool_intersection", hue="contain_model_feature", 
            row='topK'
            # style="contain_model_feature",
            # kind="scatter"
            # kind="bar",
        )
        [plt.setp(ax.get_xticklabels(), rotation=270) for ax in g.axes.flat]
        g.fig.suptitle(dataset)
        g.figure.savefig(f"../plot/catplot_{i}.png")
        # g.figure.savefig("../plot/catplot.pdf")
    except Exception as e:
        print(e)
        continue
    
## dataset
# df = df[df['finetune_ratio']>0.5]
# sns.set(rc={'figure.figsize':(14.7,8.27)})
# for i, dataset in enumerate(dataset_list):
#     df_sub = df[df['test_dataset']==dataset]
#     try:
#         g = sns.catplot(
#                 df_sub,
#                 x='accuracy_thres',
#                 y='AUC',
#                 # col='contain_data_similarity',
#                 # row='test_dataset',
#                 # hue='finetune_ratio',
#                 kind="box",
#                 # hue='test_dataset'
#             ).set(title=dataset)
#         g.figure.savefig(f"../plot/catplot_{i}.png")
#         # g.figure.savefig("../plot/catplot.pdf")
#     except Exception as e:
#         print(e)
#         continue