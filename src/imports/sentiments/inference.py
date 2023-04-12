# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv
import pandas as pd
import datetime
from .train import TweetDataset, tokenizer


test_project = pd.read_csv('../tweets.csv',  on_bad_lines='skip',delimiter=",",
                           header=0, encoding='utf-8')
test_project = test_project.sample(1000)
test_project = test_project[['date','content']]
test_project.rename(columns={'content':'text'}, inplace=True)
test_project['date'] = pd.to_datetime(test_project['date'])
print(len(test_project))

three_days = datetime.date(2023, 2, 8)
one_week = datetime.date(2023, 2, 12)
test_3d = test_project[test_project['date'].dt.date<=three_days]
test_1w = test_project[test_project['date'].dt.date<=one_week][test_project['date'].dt.date>three_days]
test_after_1w = test_project[test_project['date'].dt.date>one_week]
print(len(test_3d), len(test_1w), len(test_after_1w))


##### Prediction

def get_text_predictions(model, loader):
    device = torch.device('cuda')

    model = model.to(device)

    results_predictions =[]
    with torch.no_grad():
        model.eval()
        for data_input, _ in tqdm(loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)

            output = model(input_ids, attention_mask)

            output = (output>0.5).int()
            results_predictions.append(output)

    return torch.cat(results_predictions).cpu().detach().numpy()


model = torch.load('best_model.pt')

time_list = [test_3d, test_1w, test_after_1w]
with open('./sentiment_results.csv', 'w') as f:
    f.write('time,total_tweets,negative_tweets,ratio\n')
    for i in range(len(time_list)):
        if i == 0:
            t = '0 to 3 days'
        elif i == 1:
            t = '4 to 7 days'
        else:
            t = '8 to 16 days'
        test_dataloader = DataLoader(TweetDataset(time_list[i], tokenizer),
                                     batch_size=8,
                                     shuffle=False,
                                     num_workers=0)

        r = get_text_predictions(model, test_dataloader)
        total = len(r)
        negative_count = r.sum()
        ratio = negative_count / total
        f.write(t + f',{total}' + f',{negative_count}' + f',{ratio}\n')