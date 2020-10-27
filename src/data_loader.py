import collections
import os
import numpy as np
import re

def load_data(args):
    train_data, eval_data, test_data, user_history_dict = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    ripple_set = get_ripple_set(args, kg, user_history_dict)
    #print("train_data",train_data)
    #print("eval_data",eval_data)
    #print("test_data",test_data)
    #print("n_entity",n_entity)
    #print("n_relation",n_relation)
   # print("ripple_set",ripple_set)

    #fp=open("D:/a.txt","w")
    #fp.write(str(ripple_set))
    #fp.close()
    #a=input()
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = 'D:/data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)

    # n_user = len(set(rating_np[:, 0]))
    # n_item = len(set(rating_np[:, 1]))
    return dataset_split(rating_np)


def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data, user_history_dict


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_file = 'D:/data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))

    kg = construct_kg(kg_np)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg





def  find_similar():
    fp=open("D:/data/book/ratings_final.txt","rb")
    r=fp.readlines()
    fp.close()


    a=np.zeros((17860,20000))


    #print(a)

    sum=[]


    z=[]

    for i in r:
        index=re.findall("\d+",str(i),re.S)
        #b.append(int(index[1]))
        z.append(int(index[0])) 
        # print(int(index[0]),int(index[1]))
        a[int(index[0])][int(index[1])]=int(index[2])

    #print(a)


    def cosine_distance(a, b):
          if a.shape != b.shape:
              raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
          if a.ndim==1:
              a_norm = np.linalg.norm(a)
              b_norm = np.linalg.norm(b)
          elif a.ndim==2:
              a_norm = np.linalg.norm(a, axis=1, keepdims=True)
              b_norm = np.linalg.norm(b, axis=1, keepdims=True)
          else:
             raise RuntimeError("array dimensions {} not right".format(a.ndim))
          similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
          dist = 1. - similiarity
          return dist

    #print(cosine_distance(a,a))
    a=cosine_distance(a,a)
   # print(a.shape)
    similarity_dict = {}
    p=0
    for i in a:
        #print(i)
        b=i.copy()
       # print(len(i))
        i.sort()
        #print(np.where(b==i[2]))
        #print(np.where(b==i[2])[0][0])
        similarity_dict[p]=np.where(b==i[2])[0][0]
        p=p+1
   
    return similarity_dict


def get_ripple_set(args, kg, user_history_dict):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(args.n_hop-2):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            if h == 1:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < args.n_memory
                indices = np.random.choice(len(memories_h), size=args.n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))
    similar_dict=find_similar()
    print(ripple_set)
    #print( similar_dict)
    #print(ripple_set[0])
    #print(type(ripple_set[0])) 
    #print(similar_dict.keys())

   # print(ripple_set["36"])
    for user in list(ripple_set.keys()):
        #print(type(user))
        print("user",user)
        
        id=similar_dict[user]
        print("id:",id)
       
        print(ripple_set[user])
        if id not in list(ripple_set.keys()) :
           ripple_set[user].append((ripple_set[user][0]))
           ripple_set[user].append((ripple_set[user][1]))
        else:
          ripple_set[user].append((ripple_set[id][0]))
          ripple_set[user].append((ripple_set[id][1]))
       
    for i in  range(10):
         print(len(list(ripple_set[i])))
    return ripple_set
