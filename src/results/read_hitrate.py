import torch
outputs = torch.load("../../databases/results/posebusters_all_df.pt", weights_only=False)
df = outputs["bust_df"]["Vina"][0]
df = df.loc[~df.rmsd.isna()]
hitcount = df.loc[df.rmsd < 2].shape[0]
hitrate = (hitcount/df.shape[0])*100
print(f"Hitrate: {hitrate:.3f}%")