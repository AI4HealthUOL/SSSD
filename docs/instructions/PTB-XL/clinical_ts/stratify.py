__all__ = ['split_stratified','stratified_subsets']
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

##############################################################################
#STRATIFIED SUBSETS
##############################################################################
        
def split_stratified(df_train,subset_ratios,filename="subset.txt",col_subset="subset",col_index=None,col_label="label",col_group=None,label_multi_hot=False,random_seed=0):
    '''splits df_train via stratified split (result is written into column col_subset)'''
    assert(isinstance(subset_ratios,list) or isinstance(subset_ratios,np.ndarray))#subset_ratios has to be a list

    if(os.path.exists(filename)): #load predefined splits
        print("Loading predefined splits from disk...")
        df_split = pd.read_csv(filename,header=None)
        if(len(df_split.columns)==1):#just one column
            selected_ids = list(df_split[0])
            #print(len(selected_ids),"files selected")
            if(col_index is None):#if no index specified use filename
                df_train[col_subset] = df_train.path.apply(lambda x: x.name in selected_ids)
            else:
                df_train[col_subset] = df_train[col_index].apply(lambda x: x in selected_ids)
        else:#two columns
            df_split.columns = ["name",col_subset]
            df_split = df_split.set_index("name")
            if(col_subset in df_train):
                df_train.drop([col_subset],axis=1,inplace=True)
            if(col_index is None):
                df_train["idx_tmp"]=df_train.path.apply(lambda x:x.name)
            df_train = df_train.join(df_split,how="left",on="idx_tmp" if col_index is None else col_index)
            if(col_index is None):
                df_train.drop(["idx_tmp"],axis=1,inplace=True)
    else: #generate from scratch
        print("Generating stratified splits from scratch...")
        df_train[col_subset]=stratified_subsets(df_train,col_label,subset_ratios,col_group,label_multi_hot=label_multi_hot,random_seed=random_seed)
        #write file to disk
        if(len(subset_ratios)==2):#just two subsets
            with open(filename, "w") as outfile:
                if(col_index is None):
                    selected_ids = list(df_train[df_train[col_subset]==1]["path"].apply(lambda x:x.name))
                else:
                    selected_ids = list(df_train[df_train[col_subset]==1][col_index])
                outfile.write("\n".join(str(item) for item in selected_ids))
        else:#more than two subsets
            df_output = df_train[["path" if col_index is None else col_index,col_subset]].copy()
            if(col_index is None):
                df_output["name"] = df_train.path.apply(lambda x:x.name)
                df_output = df_output[["name",col_subset]]
            else:
                df_output = df_output[[col_index,col_subset]]
            df_output.to_csv(filename,index=False,header=False)
            
    print("Subset assignments after split_stratified:\n",df_train[col_subset].value_counts())
    return df_train

def stratified_subsets(df,col_label,subset_ratios,col_group=None,label_multi_hot=False,random_seed=0):
    '''partitions df into n subsets (with fractions specified by subset_ratios) keeping the label distribution 
    in col_label intact also respecting patient/group assignments in col_group
    returns a list of len(df) designating the subset for each row
    '''

    #trivial cases
    if(len(subset_ratios)==2 and subset_ratios[1]==1.0):
        return np.ones(len(df))
    elif(len(subset_ratios)==2 and subset_ratios[1]==0.0):
        return np.zeros(len(df))

    multi_label = isinstance(df.iloc[0][col_label],list) or isinstance(df.iloc[0][col_label],np.ndarray)
    #find unique classes
    if(not(multi_label)):
        classes = np.unique(df[col_label])
    else:
        if(label_multi_hot):
            classes = range(len(df.iloc[0][col_label]))
            df[col_label+"_tmp"] = df[col_label].apply(lambda x: np.where(x)[0])
            col_label = col_label+"_tmp"
        else:
            classes = np.unique([item for sublist in list(df[col_label]) for item in sublist])
    
    if(col_group is not None):#col_group set- i.e. aggregate according to patients etc.
        df_group = df.groupby(col_group)[col_label].apply(lambda x: list(x)).to_frame()
        if(multi_label):#turn into flat list
            df_group[col_label]=df_group[col_label].apply(lambda x: [item for sublist in x for item in sublist])
        df_group["samples_per_group"] = df_group[col_label].apply(lambda x:len(x))
        group_ids = stratify(list(df_group[col_label]),classes,subset_ratios,list(df_group["samples_per_group"]),random_seed=random_seed)
        group_ids_lst = np.zeros(len(df_group),dtype=np.int8)
        ktoi = {k:i for i,k in enumerate(df_group.index.values)}
        for i in range(len(group_ids)-1):
            group_ids_lst[group_ids[i+1]] = i+1

        return list(df[col_group].apply(lambda x: group_ids_lst[ktoi[x]]))
    else:
        ids_subsets = stratify(list(df[col_label]) if multi_label else list(df[col_label].apply(lambda x:[x])),classes,subset_ratios)
        ids_lst =  np.zeros(len(df),dtype=np.int8)
        for i in range(len(ids_subsets)-1):
            ids_lst[ids_subsets[i+1]]=i+1
        return ids_lst

#######################################################################################
#low-level routines
#######################################################################################
def stratify_batched(data, classes, ratios, samples_per_group, random_seed=0, verbose=True, batch_size=20000):
    '''calls stratify in batches and collects results afterwards (use only for really large datasets)'''
    num_data = len(data)
    num_batches = num_data // batch_size
    rest = num_data % batch_size
    rest_div = rest// num_batches
    rest_final = rest-(num_batches-1)*rest_div
    
    start_idx=[]
    end_idx=[]
    for i in range(num_batches):
        if(i==0):
            start_idx.append(0)
        else:
            start_idx.append(end_idx[-1])
        end_idx.append(start_idx[-1]+batch_size+rest_final if i==num_batches-1 else start_idx[-1]+batch_size+ rest_div)
    
    res_final=None    
    for s,e in tqdm(list(zip(start_idx,end_idx))):
        
        res= stratify(data[s:e], classes, ratios, samples_per_group=samples_per_group[s:e] if samples_per_group is not None else None, random_seed=random_seed, verbose=verbose)
        if(res_final is None):
            res_final = res
        else:
            for i in range(len(res)):
                res_final[i]= np.concatenate([res_final[i],np.array(res[i])+s])
    return res_final
    
def stratify(data, classes, ratios, samples_per_group=None,random_seed=0,verbose=True):
    """Stratifying procedure. Modified from https://vict0rs.ch/2018/05/24/sample-multilabel-dataset/ (based on Sechidis 2011)

    data is a list of lists: a list of labels, for each sample (possibly containing duplicates not multi-hot encoded).
    
    classes is the list of classes each label can take

    ratios is a list, summing to 1, of how the dataset should be split

    samples_per_group: list with number of samples per patient/group

    """
    np.random.seed(random_seed) # fix the random seed

    # data is now always a list of lists; len(data) is the number of patients; data[i] is the list of all labels for patient i (possibly multiple identical entries)

    if(samples_per_group is None):
        samples_per_group = np.ones(len(data))
        
    #size is the number of ecgs
    size = np.sum(samples_per_group)

    # Organize data per label: for each label l, per_label_data[l] contains the list of patients
    # in data which have this label (potentially multiple identical entries)
    per_label_data = {c: [] for c in classes}
    for i, d in enumerate(data):
        for l in d:
            per_label_data[l].append(i)

    # In order not to compute lengths each time, they are tracked here.
    subset_sizes = [r * size for r in ratios] #list of subset_sizes in terms of ecgs
    per_label_subset_sizes = { c: [r * len(per_label_data[c]) for r in ratios] for c in classes } #dictionary with label: list of subset sizes in terms of patients

    # For each subset we want, the set of sample-ids which should end up in it
    stratified_data_ids = [set() for _ in range(len(ratios))] #initialize empty

    # For each sample in the data set
    #print("Starting fold distribution...")
    size_prev=size+1 #just for output
    
    #while size>0:
    for _ in tqdm(list(range(len(classes)))):
        if(size==0):
            break
        #print("counter",counter,"size",size,"non-empty labels",int(np.sum([1 for l, label_data in per_label_data.items() if len(label_data)>0])),"classes",len(classes))
        #counter+=1
        #if(int(size_prev/1000) > int(size/1000) or verbose):
        #    print("Remaining entries to distribute:",int(size),"non-empty labels:", int(np.sum([1 for l, label_data in per_label_data.items() if len(label_data)>0])))
        size_prev=size
        # Compute |Di| 
        lengths = {
            l: len(label_data)
            for l, label_data in per_label_data.items()
        } #dictionary label: number of ecgs with this label that have not been assigned to a fold yet
        try:
            # Find label of smallest |Di|
            label = min({k: v for k, v in lengths.items() if v > 0}, key=lengths.get)
        except ValueError:
            # If the dictionary in `min` is empty we get a Value Error. 
            # This can happen if there are unlabeled samples.
            # In this case, `size` would be > 0 but only samples without label would remain.
            # "No label" could be a class in itself: it's up to you to formaxxxt your data accordingly.
            break
        # For each patient with label `label` get patient and corresponding counts
        unique_samples, unique_counts = np.unique(per_label_data[label],return_counts=True)
        idxs_sorted = np.argsort(unique_counts, kind='stable')[::-1]
        unique_samples = unique_samples[idxs_sorted] # this is a list of all patient ids with this label sort by size descending
        unique_counts =  unique_counts[idxs_sorted] # these are the corresponding counts
        
        # loop through all patient ids with this label
        for current_id, current_count in tqdm(list(zip(unique_samples,unique_counts)),leave=False):
            
            subset_sizes_for_label = per_label_subset_sizes[label] #current subset sizes for the chosen label

            # Find argmax clj i.e. subset in greatest need of the current label
            largest_subsets = np.argwhere(subset_sizes_for_label == np.amax(subset_sizes_for_label)).flatten()
            
            # if there is a single best choice: assign it
            if len(largest_subsets) == 1:
                subset = largest_subsets[0]
            # If there is more than one such subset, find the one in greatest need of any label
            else:
                largest_subsets2 = np.argwhere(np.array(subset_sizes)[largest_subsets] == np.amax(np.array(subset_sizes)[largest_subsets])).flatten()
                subset = largest_subsets[np.random.choice(largest_subsets2)]

            # Store the sample's id in the selected subset
            stratified_data_ids[subset].add(current_id)

            # There is current_count fewer samples to distribute
            size -= samples_per_group[current_id]
            # The selected subset needs current_count fewer samples
            subset_sizes[subset] -= samples_per_group[current_id]

            # In the selected subset, there is one more example for each label
            # the current sample has
            for l in data[current_id]:
                per_label_subset_sizes[l][subset] -= 1
               
            # Remove the sample from the dataset, meaning from all per_label dataset created
            for x in per_label_data.keys():
                per_label_data[x] = [y for y in per_label_data[x] if y!=current_id]
              
    # Create the stratified dataset as a list of subsets, each containing the orginal labels
    stratified_data_ids = [sorted(strat) for strat in stratified_data_ids]
    #stratified_data = [
    #    [data[i] for i in strat] for strat in stratified_data_ids
    #]

    # Return both the stratified indexes, to be used to sample the `features` associated with your labels
    # And the stratified labels dataset

    #return stratified_data_ids, stratified_data
    return stratified_data_ids
