# -*- coding: utf-8 -*-
"""
HNSW + KNN Querying: Base method
"""

import datetime
import workflow_neo4j_cluster_V2 as wflo
import os
import pickle
import numpy as np

import neo4j
import pandas as pd
import time

# Database Credentials
httpUri = "http://localhost:7474"
boltUri = "bolt://localhost:7687"
httpsUri = "https://username:password@localhost:7473"
userName = "username"
password = "password"

# TODO
driver = neo4j.GraphDatabase.driver(boltUri, auth=(userName, password))
# driver = []

relFileLink_85_1 = "D:\\AKS_DATA\\cosine_sim_pt85_1.pkl"

source = "D:\\AKS_DATA\\Cosine_Similarity"

# Getting 5-KNN Results by:
# 1. Running HNSW to get top 5 L1 Clusters
# 2. Running KNN on these candidate clusters to get BEST 5-NN Matches
# Results are in: "D:\AKS_DATA\retrieval_results\Query_2_RANDOMIZED"

import hnswlib
def hnsw(queryParts, dataDf, hnswFilePath):
    params = {'metric': 'cosine', # possible options are l2, cosine or ip
              "ef_constr": 100,
              "m_constr": 50, # TODO 50
              "ef_recall": 10,
              "query_k_val": 5}
    
    
    # LOADING THE TRAINING PARTS.................
    # Set dimension of input data vector
    dim = dataDf["new_signature"][0].shape[0]
    # Size of training Data
    num_elements = dataDf.shape[0]
    # TODO Stacking the data
    data = np.vstack(dataDf["new_signature"])
    
    
    
    # # TRAINING FROM SCRATCH.................................
    # # Declaring index
    # p = hnswlib.Index(space=params["metric"], dim=dim)  
    
    # p.init_index(max_elements = num_elements, 
    #               ef_construction = params["ef_constr"], 
    #               M = params["m_constr"])
    # # Adding the data to the index
    # p.add_items(data)
    # # Controlling the recall by setting ef:
    # # higher ef leads to better accuracy, but slower search
    # p.set_ef(params["ef_recall"])
    
    # # Set number of threads used during batch search/construction
    # # By default using all available cores
    # p.set_num_threads(4) 
    # print("Training Done.....")
    
    
    
        
    # LOADING THE PRE-TRAINED .bin FILE..........
    # Re-init-ing, loading the index
    p = hnswlib.Index(space='cosine', dim=dim)  # the space can be changed - keeps the data, alters the distance function.
    print("\nLoading index from 'HNSW_trained_model_25k.bin'\n")
    # Increase the total capacity (max_elements), so that it will handle the new data
    p.load_index(hnswFilePath)
    
    
        
    
    # RUNNING THE KNN ALGORITHM..................
    print("Running {}-NN on {} elements".format(params["query_k_val"], 
                                                len(data)))
    
    # Query the elements for which nearest neighbors are required:
    labels, distances = p.knn_query(queryParts, k=params["query_k_val"])
    
    return labels, distances


def knn(queryParts, # Queried Part Vectors
        queryUuids, # Queried Part UUIDs
        lvl1List, # Set of UUID Results for each Query (List of List)  
        knnDataDf, 
        kVal = 5):
    
    allSearchUuids = []
    for i, lst in enumerate(lvl1List):
        for j, ele in enumerate(lst):
            allSearchUuids.append(ele)
    
    timeSum = 0
    knnSearchSizes = []
    finalList = []
    for i in range(len(queryParts)):
        # Get this query part's vector
        queryVector = queryParts[i].reshape(1, -1)
        # Get this query part's "Training" data matches
        clusterSearchIds = lvl1List[i]
        
        # Get the Cluster IDs of all the mat
        # Get all the parts in the 5 Top Clusters
        res1 = list(  driver.session().run("""
                                      MATCH (p:part) WHERE p.uuid IN $idList
                                      RETURN p.uuid AS uuid, 
                                              p.l1_cluster as clusterId
                                      """, idList =  clusterSearchIds) 
                                       
                   )
        res2 = list(  driver.session().run("""
                                      MATCH (p:part) 
                                      WHERE p.l1_cluster IN $clusList
                                      RETURN p.uuid AS uuid, 
                                              p.l1_cluster as clusterId
                                      """, 
                              clusList = [a["clusterId"] 
                                          for a in res1])
                   )
        
        # Get the Vectors of these UUIDs from knnDataDf
        finalDf = knnDataDf[knnDataDf["uuid"].isin([b["uuid"] 
                                        for b in res2])]
        finalUuids = finalDf["uuid"].to_list()
        vectors = np.vstack(finalDf["new_signature"])
        knnSearchSizes.append(vectors.shape[0])
        
        t = time.time()
        # Get 5 NNs
        from sklearn.neighbors import KDTree, NearestNeighbors
        # tree = KDTree(vectors, leaf_size=4)
        # dist, ind = tree.query(queryVector, k=kVal)
        
        neigh = NearestNeighbors(n_neighbors=kVal)
        neigh.fit(vectors)
        dist, ind = neigh.kneighbors(queryVector)
        
        # Get the corresponding UUIDs
        resultUuids = [finalUuids[a] for a in ind[0]]
        finalList.append(resultUuids)
        timeSum += time.time() - t
        
    return finalList, timeSum, knnSearchSizes


def plot_graph(trnImgPath,
               imgShowList, # The Training part UUIDs to display
               queryFullPaths, # The Query part UUIDs to display
               currentClass, # The class being queried
               start # Start at 0 or 10
               ):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from mpl_toolkits.axes_grid1 import ImageGrid
    
       
    print("Reading the images...")
    fig = plt.figure(figsize=(50,50))
    
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols = (5 + 2, 
                                10
                                ),  
                 axes_pad=0.3,  # pad between axes in inch.
                 )
    # nrows_ncols=(len(imgShowList[0]) + 2, 
    #               len(imgShowList[start : start+10])
    #             ),
    
    
    appender = []
    ctr = 1
    # Loop through 5 results of each part i.e Row
    for j in range(len(imgShowList[0])):
        # print(i)
        # Loop through the 20 parts i.e, Column
        for i in range(len(imgShowList[start : start+10])):
            img = cv2.imread(os.path.join(trnImgPath, 
                             imgShowList[start : start+10][i][j] 
                                         + ".jpg"))
            appender.append(img)
            # fig.add_subplot(len(imgShowList[0:20]),
            #                 len(imgShowList[0]),
            #                 ctr)
            # plt.imshow(img)
            # ctr += 1
    

    # Get blank images to fill one full row
    blank_image = np.zeros((10,600,3), np.uint8)
    blankImgs = []
    for i in range(0, len(imgShowList[start : start+10])): 
        blankImgs.append(blank_image)
    
    
    # Get the query parts images, one full row
    queryImgs = []
    for q in queryFullPaths:
        img = mpimg.imread(str.lower(q + ".png"))
        queryImgs.append(img)
    # Final List to Display!!!!!!!!!!!!
    finalList = queryImgs[start : start+10] + blankImgs + appender
    for ax, im in zip(grid, finalList):
        ax.imshow(im)
        ax.set_axis_off()
    # plt.show()
    fig.savefig("D:\\AKS_DATA\\retrieval_results\\" +
                currentClass + str(int(start/10)) + ".png")
    plt.close(fig)
        
    
    

import pandas as pd
# HNSW Trained Model File.................
hnswFilePath = "D:\\AKS_DATA\\HNSW_trained_model_25k.bin"

# Loading the Training Data DF......STEP 1
trnDataPkl = "D:\\AKS_DATA\\HNSW_training_data_25k.pkl"
trnDataDf = pd.read_pickle(trnDataPkl)
trnUuids = trnDataDf["uuid"].to_list()

# Loading the KNN DF 110k Parts.......STEP 2
knnDataPkl = "D:\\AKS_DATA\\Cosine_Dist\\all_signature.pkl"
knnDataDf = pd.read_pickle(knnDataPkl)
knnDataDf["new_signature"] = [a[0][0] for a in 
                              knnDataDf["signature"] ]
knnUuids = knnDataDf["uuid"].to_list()

# LOADING THE QUERY PARTS..................
cadReposSign = "D:\\AKS_DATA\\10_CLASS_V2cad_repos_signatures.pkl"
cadReposSign = pd.read_pickle(cadReposSign)
cadReposSign["new_signature"] = [a[0] for a in 
                                 cadReposSign["intermediate_pred"]]
cadReposSign = cadReposSign.rename(columns = {"test_directory": "uuid"})
allClasses = cadReposSign["directory"].unique()
classList = list(allClasses)

# Create a list for appending dictionaries for time analysis
dataList = []

for i in range(0, len(classList)):

    # ------------------Choosing the query class-----------------
    currentClass = classList[i]
    print("------------------------> " + currentClass)
    numQuery = 20
    queryImgPath = "G:\\My Drive\\CAD Repository\\" + str(currentClass) + "\\CNN_Imgs\\12views_Icosa"
    if currentClass == "O-Rings": 
        queryImgPath = "G:\\My Drive\\CAD Repository\\" + "O_Rings" + "\\CNN_Imgs\\12views_Icosa"
    # ------------------Choosing the query class-----------------
    
    dataDict = {"class": currentClass}
    t = time.time()
    """ HNSW: STEP 1 """
    queryDf = cadReposSign[cadReposSign["directory"] == currentClass]
    if queryDf.shape[0] < numQuery:
        numQuery = queryDf.shape[0]
        
    # ------------> SELECT "numQuery" RANDOM PARTS FROM QueryDf
    searchDf = queryDf.sample(n = numQuery)
    queryParts = np.vstack(searchDf["new_signature"].iloc[0:numQuery])
    queryUuids = searchDf["uuid"].iloc[0:numQuery]
    
    dataDict["num_queries"] = len(queryUuids)
    dataDict["query_uuids"] = queryUuids.to_list()
    
    # GET HNSW NEAREST NEIGHBORS
    labels, distances = hnsw(queryParts, trnDataDf, hnswFilePath)
    # print("1.---> HNSW Done. Took {}. Starting KNN..".format(
    #                                 datetime.datetime.now() - t))
    dataDict["hnsw_time"] = time.time() - t
    
    # print("Getting image paths..")
    # imgShowList = []
    # for i in range(labels.shape[0]):
    #     imgIdList = [trnUuids[a] for a in labels[i, :]]
    #     imgShowList.append(imgIdList)
    # print("Done. ")
    
    # # Plotting the images
    # trnImgPath = "D:\\AKS_DATA\\JPEG_Images\\All_Images"
    # plot_graph(trnImgPath, imgShowList, queryFullPaths, 0)
    # plot_graph(trnImgPath, imgShowList, queryFullPaths, 10)
    
    t = datetime.datetime.now()
    """5-KNN for the Clusters Found in STEP 1"""
    # Plotting the images
    trnImgPath = "D:\\AKS_DATA\\JPEG_Images\\All_Images"
    print("Getting image paths..")
    lvl1List = []
    for i in range(labels.shape[0]):
        imgIdList = [trnUuids[a] for a in labels[i, :]]
        lvl1List.append(imgIdList) # A List of-List of results.
    print("Done. ")
    
    finalShowList, timeSum , knnSearchSizes = knn(queryParts, 
                                                  queryUuids, lvl1List, knnDataDf)
    # print("2.---> KNN Done. Took {}. Plotting..".format(
    #                                 datetime.datetime.now() - t))
    dataDict["knn_time"] = timeSum
    dataDict["knn_search_sizes"] = knnSearchSizes
    
    a = time.time()
    queryFullPaths = [os.path.join(queryImgPath, a, a + str(".0.0")) 
                      for a in queryUuids]
    
    if numQuery == 20:
        plot_graph(trnImgPath, finalShowList, queryFullPaths, currentClass, 0)
        plot_graph(trnImgPath, finalShowList, queryFullPaths, currentClass, 10)
    else:
        plot_graph(trnImgPath, finalShowList, queryFullPaths, currentClass, 0)
    
    dataDict["plot_time"] = time.time() - a
    dataList.append(dataDict)

dataDf = pd.DataFrame(dataList)
dataDf.to_csv(r"D:\AKS_DATA\retrieval_results\time_data.csv")
