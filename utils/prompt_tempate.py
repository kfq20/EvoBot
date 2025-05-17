import csv

def find_user_neighbors(file_path, user_id, n):
    neighbors = set()  # Using a set to avoid duplicate neighbors
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Check if the relationship is either following or followers
                if row['relation'] in ['following', 'followers']:
                    # Add to neighbors if user_id is either source or target
                    if row['source_id'] == user_id:
                        neighbors.add(row['target_id'])         
                    elif row['target_id'] == user_id:
                        neighbors.add(row['source_id'])
                    
                    # Stop if we have reached the desired number of neighbors
                    if len(neighbors) >= n:
                        break
        return list(neighbors)[:n]  # Return exactly n neighbors
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def llama_prompt(user_info, neighbor_infos):
    user_prompt = f"You are using the social media Twitter. Here is the discription about you: {user_info['description']}.\n"
    neighbor_info_prompts = {}
    for i in range(len(neighbor_infos)):
        neighbor_info_prompt = neighbor_infos[i]['description']
        neighbor_info_prompts[f"Neighbor {i+1}"] = neighbor_info_prompt
    user_prompt += f"Additionally, you also know information about several of your neighbors in the social network (i.e., users with whom you have a following or followed-by relationship): {neighbor_info_prompts}\n"
    user_prompt += "Now, based on the above information, please generate several tweets. The topics are unrestricted, but they should fully showcase your personal characteristics and integrate into the online community."
    total_prompt = f"[INST]{user_prompt}[/INST]"
    
    return total_prompt