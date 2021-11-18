import pandas as pd
import torch

def get_layers(model: torch.nn.Module):
    # get layers from model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_layers(child))
            except TypeError:
                flatt_children.append(get_layers(child))
    return flatt_children
    
def my_loss(out_image, out_demo, data, census_data, factor=5):
    reconstruct_loss = torch.mean((out_image - data)**2)
    regression_loss = torch.mean((out_demo - census_data)**2)
    # print(reconstruct_loss, regression_loss)
    return reconstruct_loss + regression_loss * factor

def load_demo(data_dir):
    trp_bhv = pd.read_csv(data_dir+"origin_trip_behavior.csv")
    trp_bhv['census_tract'] = [s1+"_"+s2+"_"+s3 for (s1, s2, s3) in zip(trp_bhv['state_fips_1'].astype(str), trp_bhv['county_fips_1'].astype(str), trp_bhv['tract_fips_1'].astype(str))]
    trp_bhv = trp_bhv.sort_values(by='census_tract')
    # trp_bhv = trp_bhv[trp_bhv['census_tract'].isin(unique_ct)]
    trp_bhv['mode_share'] = trp_bhv['wtperfin_mode'] / trp_bhv['wtperfin_all']

    demo_df = pd.read_csv(data_dir+"census_demo_df.csv")
    demo_df['census_tract'] = '17_'+demo_df['COUNTYA'].astype(str)+'_'+demo_df['TRACTA'].astype(str)

    trp_gen = trp_bhv.groupby(by=['census_tract'], as_index=False).first()[['census_tract','wtperfin_all']]
    trp_gen.columns = ['census_tract','trp_gen']
    trp_gen['trp_gen'] = trp_gen['trp_gen']/100000
    trp_auto = trp_bhv[trp_bhv['mode']==2][['census_tract','mode_share']]
    trp_auto.columns = ['census_tract','auto_share']

    demo_df = demo_df.merge(trp_gen, on='census_tract')
    demo_df = demo_df.merge(trp_auto, on='census_tract')

    for d in ['tot_population','pct25_34yrs','pct35_50yrs','pctover65yrs',
             'pctwhite_alone','pct_nonwhite','pctblack_alone',
             'pct_col_grad','pctPTcommute','avg_tt_to_work','inc_per_capita']:
        demo_df[d] = demo_df[d]/demo_df[d].max()

    demo_np = demo_df[['trp_gen','auto_share','tot_population','pct25_34yrs','pct35_50yrs','pctover65yrs',
             'pctwhite_alone','pct_nonwhite','pctblack_alone',
             'pct_col_grad','pctPTcommute','avg_tt_to_work','inc_per_capita']].to_numpy()
    demo_cs = demo_df['census_tract'].tolist()
    
    return demo_cs, demo_np