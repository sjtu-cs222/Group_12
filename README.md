# Group_12
DATA POISONING ATTACK AGAINST  GRAPH EMBEDDING BASED LINK PREDICTION

# Dependencies
  * python == 3.6.7
  * tensorflow ==1.3.0
  * networkx == 1.11
  * Gpu
  
# Dataset:
  
  facebook:This dataset consists of 'circles' (or 'friends lists') from Facebook. Facebook data was collected from survey participants using Facebook app. The dataset includes node features (profiles), circles, and ego networks. 
  
  It's available in /code/facebook_graph.rar
  

# Code:
  
  line.py  and model.py : Computing the embedding
  
  xxx_attack.py : Different ways to attack the social network 
  
  
# Training:
 
 1.random attack:
 
    python random_attack.py
 
 2.target attack:
 
    python target_attack.py

 3.available attack:

    python available_attack.py
