import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime as dt
import json
import re
import os
from transformers import pipeline
import pandas as pd
import random

url_pattern = re.compile(r'https?://\S+')
emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F]"  
            "|[\U0001F300-\U0001F5FF]" 
            "|[\U0001F680-\U0001F6FF]"  
            "|[\U0001F1E0-\U0001F1FF]" 
            "|[\U00002700-\U000027BF]" 
            "|[\U0001F900-\U0001F9FF]" 
            "|[\U0001FA70-\U0001FAFF]" 
            "|[\U00002600-\U000026FF]" 
            "|[\U00002B50-\U00002B55]" 
        )

def tweets_embedding(each_user_tweets_path, community=2, dpo_epoch=None, output_path=None, device=0):
        if output_path is None:
            if dpo_epoch is None:
                output_path = f"./data/processed_data/community_{comm}/human_data/tweets_tensor.pt"
            else:
                output_path = f"./data/processed_data/community_{community}/tweets_dpo_{dpo_epoch}_tensor.pt"
        else:
            output_path = output_path
        if not os.path.exists(output_path):
            feature_extract=pipeline('feature-extraction',model='roberta-base', device=device, tokenizer='roberta-base', truncation=True,padding='max_length', add_special_tokens = True, max_length=512)
            feature_extract.tokenizer.model_max_length = 512
            each_user_tweets=json.load(open(each_user_tweets_path,'r'))
            print('Running feature2 embedding')
            if True:
                url_pattern = re.compile(r'https?://\S+')
                tweets_list=[]
                for i in tqdm(range(len(each_user_tweets))):
                    if len(each_user_tweets[str(i)])==0:
                        total_each_person_tweets=torch.zeros(768)
                    else:
                        clean_texts = []
                        for j in range(len(each_user_tweets[str(i)])):
                            if j == 20:
                                break
                            each_tweet=each_user_tweets[str(i)][j]
                            clean_text = re.sub(url_pattern, '', each_tweet)
                            clean_text = re.sub(r'\d+\.$', '', clean_text).strip()
                            clean_texts.append(clean_text)
                        if len(clean_texts) == 0:
                            total_word_tensor=torch.zeros(768)
                        else:
                            try:
                                all_tweet_tensor=torch.tensor(feature_extract(clean_texts, padding=True, max_length=512, truncation=True, batch_size=len(clean_texts)))
                            except:
                                continue
                            total_word_tensor = torch.mean(all_tweet_tensor, dim=2)
                            total_each_person_tweets = torch.mean(total_word_tensor, dim=0)
                            
                    tweets_list.append(total_each_person_tweets.squeeze())
                    # noise = torch.randn_like(total_each_person_tweets) * 0.1
                    # noisy_tensor = total_each_person_tweets + noise
                    # tweets_list.append(noisy_tensor.squeeze())
                if dpo_epoch is not None:
                    split_index = int((dpo_epoch+1) * len(tweets_list)/6)
                    shuffled_part = tweets_list[:split_index]
                    random.shuffle(shuffled_part)
                    tweets_list = shuffled_part + tweets_list[split_index:]
                tweet_tensor=torch.stack(tweets_list)
                torch.save(tweet_tensor,output_path)
        else:
            pass

    # Des_embbeding()
def extract_tweets_1(community, epoch=None, output_path=None):
    if output_path is None:
        if epoch is not None:
            output_path = f'./data/processed_data/community_{community}/id_tweet_dpo_{epoch}.json'
        else:
            output_path = f'./data/processed_data/community_{community}/id_tweet_llama.json'
    else:
        output_path = output_path
    if not os.path.exists(output_path):
        print("extracting each_user's tweets")
        id_tweet={i:[] for i in range(len(user_idx))}
        human_tweets = json.load(open(f"./data/raw_data/community_{community}/human_tweet.json",'r'))
        if epoch is not None: 
            bot_tweets=json.load(open(f"./data/raw_data/community_{community}/dpo_{epoch}_bot_tweet.json",'r'))
        else:
            bot_tweets=json.load(open(f"./data/raw_data/community_{community}/llama_bot_tweet.json",'r'))
        for each_human in human_tweets:
            uid='u'+str(each_human['author_id'])
            text=each_human['text']
            try:
                index=uid_index[uid]
                id_tweet[index].append(text)
            except KeyError:
                continue

        for each in bot_tweets:
            segments = re.split(r'(?<=\d\.)\s', each['text'].strip())
            segments = [re.sub(r"\d+\.$", "", segment) for segment in segments if segment][1:]
            # uid='u'+str(each['author_id'])
            uid=each['author_id']
            try:
                for text in segments:
                    
                    index=uid_index[uid]
                    id_tweet[index].append(text)
            except KeyError:
                continue
        json.dump(id_tweet,open(output_path,'w'))

if __name__ == "__main__":
    for comm in [10, 11]:
    #     output_dir = f"./data/processed_data/community_2/"
    #     each_user_tweets_path=output_dir + f'id_tweet_dpo_{comm}.json'
    #     tweets_embedding(each_user_tweets_path, community=comm, output_path = f"./data/processed_data/community_2/tweets_dpo_{comm}_tensor.pt")
        # continue
        path=f'./data/raw_data/community_{comm}/'

        user=pd.read_json(path+'human_user_info.json')
        edge=pd.read_csv(path+'edge.csv')
        user_idx=user['id']
        uid_index={uid:index for index,uid in enumerate(user_idx.values)}
        user_index_to_uid = list(user.id)
        uid_to_user_index = {x : i for i, x in enumerate(user_index_to_uid)}
        tweet = pd.read_json(path + 'tweet.json')
        tweet_idx=tweet['id']
        tid_index={tid:index for index,tid in enumerate(tweet_idx.values)}
        tweet_index_to_tid = list(tweet.id)
        tid_to_tweet_index = {x : i for i, x in enumerate(tweet_index_to_tid)}
        feature_extract=pipeline('feature-extraction',model='roberta-base', device=1, tokenizer='roberta-base', truncation=True,padding='max_length', add_special_tokens = True, max_length=512)
        print('extracting labels and splits')
        split=pd.read_csv(path + "split.csv")
        label=pd.read_csv(path + "label.csv")
        uid_label={uid:label for uid, label in zip(label['id'].values,label['label'].values)}
        uid_split={uid:split for uid, split in zip(split['id'].values,split['split'].values)}
        label_new=[]
        train_idx=[]
        test_idx=[]
        val_idx=[]
        for i,uid in enumerate(tqdm(user_idx.values)):
            single_label=uid_label[uid]
            single_split=uid_split[uid]
            if single_label =='human':
                label_new.append(0)
            else:
                label_new.append(1)
            if single_split=='train':
                train_idx.append(i)
            elif single_split=='test':
                test_idx.append(i)
            else:
                val_idx.append(i)

        labels=torch.LongTensor(label_new)
        train_mask = torch.LongTensor(train_idx)
        valid_mask = torch.LongTensor(val_idx)
        test_mask = torch.LongTensor(test_idx)
        output_dir = f"./data/processed_data/community_{comm}/human_data/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(train_mask, output_dir+"train_idx.pt")
        torch.save(valid_mask,output_dir+"val_idx.pt")
        torch.save(test_mask,output_dir+"test_idx.pt")
        torch.save(labels,output_dir+"label.pt")

        print('extracting edge_index&edge_type')
        edge_index=[]
        edge_type=[]
        for i in tqdm(range(len(edge))):
            sid=edge['source_id'][i]
            tid=edge['target_id'][i]
            if sid in uid_index and tid in uid_index:
                if edge['relation'][i]=='followers':
                    try:
                        edge_index.append([uid_index[sid],uid_index[tid]])
                        edge_type.append(0)
                    except KeyError:
                        continue
                elif edge['relation'][i]=='following':
                    try:
                        edge_index.append([uid_index[sid],uid_index[tid]])
                        edge_type.append(1)
                    except KeyError:
                        continue
            # elif edge['relation'][i] == 'post':
            #     try:
            #         edge_index.append([uid_index[sid],tid_index[tid]])
            #         edge_type.append(2)
            #     except KeyError:
            #         continue
            # elif edge['relation'][i] == 'like':
            #     try:
            #         edge_index.append([uid_index[sid],tid_index[tid]])
            #         edge_type.append(3)
            #     except KeyError:
            #         continue
            # elif edge['relation'][i] == 'mentioned':
            #     try:
            #         edge_index.append([tid_index[sid],uid_index[tid]])
            #         edge_type.append(4)
            #     except KeyError:
            #         continue
            # elif edge['relation'][i] == 'retweeted':
            #     try:
            #         edge_index.append([tid_index[sid],tid_index[tid]])
            #         edge_type.append(5)
            #     except KeyError:
            #         continue
            # elif edge['relation'][i] == 'quoted':
            #     try:
            #         edge_index.append([tid_index[sid],tid_index[tid]])
            #         edge_type.append(6)
            #     except KeyError:
            #         continue
            # elif edge['relation'][i] == 'replied':
            #     try:
            #         edge_index.append([tid_index[sid],tid_index[tid]])
            #         edge_type.append(7)
            #     except KeyError:
            #         continue

        torch.save(torch.LongTensor(edge_index).t(),output_dir+"edge_index.pt")
        torch.save(torch.LongTensor(edge_type),output_dir+"edge_type.pt")

        print('extracting num_properties')
        following_count=[]
        for i,each in enumerate(user['public_metrics']):
            if i==len(user):
                break
            if each is not None and isinstance(each,dict):
                if each['following_count'] is not None:
                    following_count.append(each['following_count'])
                else:
                    following_count.append(0)
            else:
                following_count.append(0)
                
        statues=[]
        for i,each in enumerate(user['public_metrics']):
            if i==len(user):
                break
            if each is not None and isinstance(each,dict):
                if each['tweet_count'] is not None:
                    statues.append(each['tweet_count'])
                else:
                    statues.append(0)
            else:
                statues.append(0)

        followers_count=[]
        for each in user['public_metrics']:
            if each is not None and each['followers_count'] is not None:
                followers_count.append(int(each['followers_count']))
            else:
                followers_count.append(0)
                
        num_username=[]
        for each in user['username']:
            if each is not None:
                num_username.append(len(each))
            else:
                num_username.append(int(0))
                
        created_at=user['created_at']
        created_at=pd.to_datetime(created_at,unit='s')

        followers_count=pd.DataFrame(followers_count)
        followers_count=(followers_count-followers_count.mean())/followers_count.std()
        followers_count=torch.tensor(np.array(followers_count),dtype=torch.float32)

        date0=dt.strptime('Tue Sep 5 00:00:00 +0000 2020 ','%a %b %d %X %z %Y ')
        active_days=[]
        for each in created_at:
            active_days.append((date0-each).days)
            
        active_days=pd.DataFrame(active_days)
        active_days=active_days.fillna(int(1)).astype(np.float32)

        screen_name_length=[]
        for each in user['name']:
            if each is not None:
                screen_name_length.append(len(each))
            else:
                screen_name_length.append(int(0))

        followers_count=(followers_count-followers_count.mean())/followers_count.std()
        followers_count=torch.tensor(np.array(followers_count),dtype=torch.float32)

        active_days=pd.DataFrame(active_days)
        active_days.fillna(int(0))
        active_days=active_days.fillna(int(0)).astype(np.float32)

        active_days=(active_days-active_days.mean())/active_days.std()
        active_days=torch.tensor(np.array(active_days),dtype=torch.float32)

        screen_name_length=pd.DataFrame(screen_name_length)
        screen_name_length=(screen_name_length-screen_name_length.mean())/screen_name_length.std()
        screen_name_length=torch.tensor(np.array(screen_name_length),dtype=torch.float32)

        following_count=pd.DataFrame(following_count)
        following_count=(following_count-following_count.mean())/following_count.std()
        following_count=torch.tensor(np.array(following_count),dtype=torch.float32)

        statues=pd.DataFrame(statues)
        statues=(statues-statues.mean())/statues.std()
        statues=torch.tensor(np.array(statues),dtype=torch.float32)

        num_properties_tensor=torch.cat([followers_count,active_days,screen_name_length,following_count,statues],dim=1)

        num_properties_tensor=torch.cat([followers_count,active_days,screen_name_length,following_count,statues],dim=1)

        pd.DataFrame(num_properties_tensor.detach().numpy()).isna().value_counts()
        print('extracting cat_properties')
        protected=user['protected']
        verified=user['verified']

        protected_list=[]
        for each in protected:
            if each == True:
                protected_list.append(1)
            else:
                protected_list.append(0)
                
        verified_list=[]
        for each in verified:
            if each == True:
                verified_list.append(1)
            else:
                verified_list.append(0)
                
        default_profile_image=[]
        for each in user['profile_image_url']:
            if each is not None:
                if each=='https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png':
                    default_profile_image.append(int(1))
                elif each=='':
                    default_profile_image.append(int(1))
                else:
                    default_profile_image.append(int(0))
            else:
                default_profile_image.append(int(1))

        protected_tensor=torch.tensor(protected_list,dtype=torch.float)
        verified_tensor=torch.tensor(verified_list,dtype=torch.float)
        default_profile_image_tensor=torch.tensor(default_profile_image,dtype=torch.float)

        cat_properties_tensor=torch.cat([protected_tensor.reshape([protected_tensor.shape[0],1]),
                                        verified_tensor.reshape([verified_tensor.shape[0],1]),
                                        default_profile_image_tensor.reshape([default_profile_image_tensor.shape[0],1])],dim=1)

        torch.save(num_properties_tensor,output_dir+"num_properties_tensor.pt")

        torch.save(cat_properties_tensor,output_dir+"cat_properties_tensor.pt")

        print("extracting each_user's tweets")
        id_tweet={i:[] for i in range(len(user_idx))}
        # for i in range(9):
        #     name='tweet_'+str(i)+'.json'
        human_tweets = json.load(open(f"./data/raw_data/community_{comm}/human_tweet.json",'r'))
        bot_tweets=json.load(open(f"./data/raw_data/community_{comm}/bot_tweet.json",'r'))
        for each in human_tweets + bot_tweets:
            uid='u'+str(each['author_id'])
            text=each['text']
            text = re.sub(url_pattern, '', text)
            text = re.sub(r'RT @\w+: |@\w+', '', text)
            text = emoji_pattern.sub('', text)
            text = re.sub(r'[ \t]+', ' ', text)
            try:
                index=uid_index[uid]
                id_tweet[index].append(text)
            except KeyError:
                continue
        # for each in bot_tweets:
        #     segments = re.split(r'(?<=\d\.)\s', each['text'].strip())
        #     segments = [segment for segment in segments if segment][1:]
        #     # uid='u'+str(each['author_id'])
        #     uid=each['author_id']
        #     try:
        #         for text in segments:
                    
        #             index=uid_index[uid]
        #             id_tweet[index].append(text)
        #     except KeyError:
        #         continue
        json.dump(id_tweet,open(output_dir + 'id_tweet.json','w'), ensure_ascii=False, indent=4)

        

        # user=pd.read_json(path + 'user_info.json')

        user_text=list(user['description'])
        each_user_tweets_path=output_dir + 'id_tweet.json'

        
        print('Running feature1 embedding')
        path=output_dir + "des_tensor.pt"
        if not os.path.exists(path):
            des_vec=[]
            for k,each in enumerate(tqdm(user_text)):
                if each is None:
                    des_vec.append(torch.zeros(768))
                else:
                    feature=torch.Tensor(feature_extract(each))
                    for (i,tensor) in enumerate(feature[0]):
                        if i==0:
                            feature_tensor=tensor
                        else:
                            feature_tensor+=tensor
                    feature_tensor/=feature.shape[1]
                    des_vec.append(feature_tensor)
                    
            des_tensor=torch.stack(des_vec,0)
            torch.save(des_tensor,path)
        else:
            des_tensor=torch.load(path)
        print('Finished')

        tweets_embedding(each_user_tweets_path, community=comm, output_path = f"./data/processed_data/community_{comm}/human_data/tweets_tensor.pt")
