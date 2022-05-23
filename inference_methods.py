# -*- coding: utf-8 -*-
"""

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


# Finding Jaccard Similarity between assemblies based on the clusters that
# their parts belong to, at level = label_propagation_cluster_9.
# Also, check whether ward_cluster_80 and ward_cluster_93 are possible.
def assembly_jaccard():
    # Query all the assembly UUIDs
    assemQuery = driver.session().run(
        "MATCH (a:assembly) RETURN a.uuid AS uuid"
        )
    assemUuids = [a["uuid"] for a in assemQuery]

    """-------------------THIS USED A PATH-QUERYING APPROACH TO GETTING PARTS OF AN
    ASSEMBLY. WAS TOO SLOW FOR LARGE ASSEMBLIES, SO MOVED TO MONGODB
    QUERIES BELOW------------------------"""
    # time = datetime.datetime.now()
    # assemClustersDict = {}
    # # Query all parts belonging to each assembly
    # for ct, assembly in enumerate(assemUuids[2:]):
    #     assemUuid = assembly
    #     assemParts = """
    #     MATCH (a:assembly) WHERE a.uuid = $assemUuid
    #     CALL apoc.path.expand(a, "subset", "/part", 1, 10)
    #     YIELD path
    #     WITH path, nodes(path) AS pathNodes
    #     RETURN pathNodes,
    #             length(path) AS hops,
    #             [node IN pathNodes | labels(node)]
    #             ORDER BY hops ASC
    #     """
    #     assemPartRes = driver.session().run(assemParts,
    #                                           assemUuid = assemUuid)
    #     assemPartResList = list(assemPartRes)
    #     startNodeIds = [a[0][0]["uuid"] for a in assemPartResList]
    #     endNodeIds = [a[0][-1]["uuid"] for a in assemPartResList]
    #     endNodeClusters = [a[0][-1]["label_propagation_cluster_9"]
    #                        for a in assemPartResList]

    #     startNodeLabels = [a[2][0] for a in assemPartResList]
    #     endNodeLabels = [a[2][-1] for a in assemPartResList]
    #     assemClustersDict[assemUuid] = endNodeClusters
    #     if ct%1 == 0:
    #         print("Queried {} Parts from Assembly {} of {}".format(
    #                                             len(endNodeLabels),
    #                                             ct,
    #                                             len(assemUuids)
    #                                                             )
    #              )

    #         print("Time taken so far: {}".format(
    #             datetime.datetime.now() - time))
    """------------------------END OF PATH QUERYING APPROACH----------------"""



    """------------START OF MONGODB APPROACH, WITH NEO4J FOR LABEL IDs------------"""
    # MongoDB Credentials
    import pymongo
    client = pymongo.MongoClient("10.76.152.37:27017")
    cadMetaColl = "SRC3:CAD_Repository"
    cadSignColl = "CAD_Repos_Properties"
    partMetaColls = ["SRC1:Parts_0", "SRC1:Parts_1", "SRC1:Parts_2",
                     "SRC1:Parts_3", "SRC1:Parts_4", "SRC1:Parts_5",
                     "SRC1:Parts_6", "SRC1:Parts_7", "SRC1:Parts_8",
                     "SRC1:Parts_9", "SRC1:Parts_10"]
    partSignColl = "Part_Sub_Properties"
    assemMetaColl = "SRC1:Assemblies_0"

    db = client["FabWave_Data"]

    def get_part_docs_assem_id(db, collNames, assemId):
        docList = []
        for collName in collNames:
            partCollec = db[collName]
            mongoDocList = [z for z in partCollec.find({})]

            partDocs = [ doc for doc in mongoDocList if assemId in doc["UUID_assembly"] ]
            docList = docList + partDocs

        return docList


    time = datetime.datetime.now()
    assemClustersDict = {}
    assemPartsDict = {}
    totalPartsCounter = 0

    getAllClusIdQuery = """
                MATCH (p:part)
                RETURN p.label_propagation_cluster_9 AS clusterId,
                        p.uuid AS uuid
                """
    allPartClusterIds = driver.session().run(getAllClusIdQuery)
    allPartClusterIds = list(allPartClusterIds)
    uuidList = [a["uuid"] for a in allPartClusterIds]
    clusterIdList = [a["clusterId"] for a in allPartClusterIds]
    allPartClusterDf = pd.DataFrame(list(zip(uuidList, clusterIdList)),
                                    columns =['uuid', 'clusterId'])
    print("I have all the cluster IDs for {} parts. Time taken: {}".format(
                                        allPartClusterDf.shape[0],
                                        datetime.datetime.now() - time))

    time = datetime.datetime.now()
    # Loop through each assembly to get its parts, and its cluster IDs.
    for ct, assembly in enumerate(assemUuids):
        assemUuid = assembly
        # Get list of PARTS for this assembly
        docList = get_part_docs_assem_id(db, partMetaColls, assemUuid)
        assemPartResList = docList
        startNodeIds = [assemUuid for i in range(len(assemPartResList))]
        endNodeIds = [a["UUID_part"] for a in assemPartResList]

        # If this assembly has parts, query their cluster IDs from Neo4J
        if endNodeIds != []:

            thisAssemDf = pd.DataFrame(endNodeIds, columns = ["uuid"])
            # Get cluster IDs for this assembly's parts
            mergeDf = pd.merge(allPartClusterDf, thisAssemDf,
                               on = "uuid")
            assemPartResList = mergeDf["clusterId"].tolist()
            endNodeClusters = assemPartResList
            # Saving each assembly's parts and their resp. Cluster IDs.
            assemPartsDict[assemUuid] = endNodeIds
            assemClustersDict[assemUuid] = endNodeClusters

        # If there are no parts related to this assembly, just put an
        # empty list into the dictionary.
        else:
            assemPartResList = []
            endNodeClusters = []
            # Saving each assembly's parts and their resp. Cluster IDs.
            assemPartsDict[assemUuid] = endNodeIds
            assemClustersDict[assemUuid] = endNodeClusters

        # Count the total number of parts processed so far.
        totalPartsCounter += len(endNodeIds)
        if ct%10 == 0:
            print("Queried {} Parts from {} Assemblies out of {}".format(
                                                    totalPartsCounter,
                                                    ct,
                                                    len(assemUuids)
                                                                    )
                      )
            print("Time taken so far: {}".format(
            datetime.datetime.now() - time))
    """-------------------END OF MONGODB Approach--------------------"""

    direc = "G:\\My Drive\\Work\\Neo4j Stuff\\Export_and_Community_Detection"
    with open(os.path.join(direc, "assemClustersDict.pkl"), 'wb') as handle:
        pickle.dump(assemClustersDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(direc, "assemPartsDict.pkl"), 'wb') as handle:
        pickle.dump(assemPartsDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Jaccard Similarity uploading to Neo4J using APOC.
# Relations in the range [0.25, 0.5] are considered and uploaded.
# This is the top 50% of the range of Jaccard Similarity.
# There are 11,017 relations uploaded.
def assembly_jaccard_upload():
    direc = "G:\\My Drive\\Work\\Neo4j Stuff\\Export_and_Community_Detection"
    with open(os.path.join(direc, "assemClustersDict.pkl"), 'rb') as handle:
        assemClustersDict = pickle.load(handle)

    from collections import Counter
    # Compute Generalized Jaccard Similarity between Assemblies
    # in the f2m of "Multisets" through "counter" objects
    jaccardSims = np.zeros( ( len(assemClustersDict.keys()),
                              len(assemClustersDict.keys())
                            )
                          )
    for i, key1 in enumerate(assemClustersDict.keys()):
        if i%100 == 0:
            print("""Computing Jaccard distances for assembly:
                  {} vs {}""".format(i, len(assemClustersDict.keys()))
                 )
        for j, key2 in enumerate(assemClustersDict.keys()):
            if j > i:
                break
            ctr1 = Counter(assemClustersDict[key1])
            ctr2 = Counter(assemClustersDict[key2])
            intersection = ctr1 & ctr2
            union = ctr1 | ctr2

            # This is BAG INTERSECTION of the two counters
            numerator = sum(intersection.values())
            # This is the BAG SUM of the two counters
            # i.e, summing up all the values of BOTH COUNTERS
            # instead of a Union.
            denominator = sum(ctr1.values()) + sum(ctr2.values())
            # Jaccard Similarity for Multisets
            # Range: [0, 1/2]
            try:
                jaccardSims[i][j] = numerator / denominator
            except:
                pass


    import seaborn as sns
    import matplotlib.pylab as plt
    def heatmap2d(arr: np.ndarray):
        plt.imshow(arr, cmap='viridis')
        plt.colorbar()
        plt.show()

    # lowerTriangle = np.tril(jaccardSims)
    # heatmap2d(lowerTriangle)
    # plt.hist(lowerTriangle, bins = [0,0.05,0.1,0.15,0.2,0.25,
    #                               0.3,0.35,0.4,0.45,0.5])


    # Get indices where the Jaccard Distance is in top half of
    # the Jaccard Range
    # THEN, update these values for the assemblies
    indices = np.where((0.25 <= jaccardSims) &
                       (jaccardSims <= 0.5))
    rowIndex, colIndex = indices
    assemblyList = list(assemClustersDict.keys())
    ctr = 0
    dictList = []
    for j in range(0, rowIndex.shape[0]):#range(rowIndex.shape[0]):
        if rowIndex[j] != colIndex[j]:
            assemId1 = assemblyList[rowIndex[j]]
            assemId2 = assemblyList[colIndex[j]]
            jaccardSimVal = jaccardSims[rowIndex[j]][colIndex[j]]

            dicter = {"assemId1": assemId1,
                      "assemId2": assemId2,
                      "jaccardSimVal": jaccardSimVal}
            # dicter = [assemId1, assemId2, jaccardSimVal]
            dictList.append(dicter)

            ctr += 1
            if ctr%1000 == 0:
                print("Updated rels: {} of potential {}".format(ctr,
                                                    rowIndex.shape[0]))


    jaccardUpdateQuery = """
    CALL apoc.periodic.iterate('UNWIND $dictList AS dicter RETURN dicter',

    'MATCH (a1:assembly), (a2:assembly) ' +
    'WHERE a1.uuid = dicter["assemId1"] AND a2.uuid = dicter["assemId2"] ' +
    'CREATE (a1)-[r:similar_to {jaccard_sim: dicter["jaccardSimVal"]}]->(a2) ',

                    {batchSize:100,
                    batchMode:"BATCH",
                    parallel:false,
                    params:{dictList:$dictList}
                    }
    )
    """
    jaccardRes = driver.session().run(jaccardUpdateQuery,
                                    dictList = dictList)


# Association Rule Mining
class RuleMining():
    def __init__(self):
        pass
    
    def get_rules(finalList,
              min_support = 0.5, 
              min_threshold = 0.7):
        # Get the Cluster IDs to actually generate the rules from
        requiredClusters = driver.session().run("""
        UNWIND $final_uuids as uid
        MATCH (p:part) WHERE p.uuid = uid
        RETURN p.l1_cluster as cluster
        """, final_uuids = finalList)
        requiredClusters = list(requiredClusters)
        finalClusters = [str(a["cluster"]) for a in requiredClusters]
        # print(finalClusters)
        
        
        # allLabels = driver.session().run("""
        # call apoc.meta.data() yield label, property
        # with ['part', 'material_category'] as labels, property, label where label in labels
        # return property, label""")
        
        # allLabels = list(allLabels)
        # allLabels = [a["property"] for a in allLabels]
        requiredValues = [
        'uuid', 'l1_cluster', 'l1_center', 'l1_center_score', 
         
        'quantity', 'precision', 'part_description', 
        'part_category', 'model_type', 'material_category', 
        'feature_descriptions', 'main_process', 'sub_process', 
        'features', 'component_type']
        
        labeledParts = driver.session().run("""
        MATCH (p:part) WHERE EXISTS(p.quantity) AND EXISTS(p.precision)
        AND EXISTS(p.part_description)
        RETURN ID(p), p
        """)
        labeledParts = list(labeledParts)
        
        print("Neo4J Queries Done.")
        
        finalList = []
        for part in labeledParts:
            dicter = {}
            dicter["id"] = part[0]
            for key in requiredValues:
                if key == "features" or key == "feature_descriptions":
                    dicter[key + "0"] = part[1][key][0]
                    dicter[key + "1"] = part[1][key][1]
                    dicter[key + "2"] = part[1][key][2]
                else:
                    dicter[key] = part[1][key]
                
            finalList.append(dicter)
        
        labeledDf = pd.DataFrame(finalList)
        labeledDf = labeledDf.replace(r'^\s*$', "NA", regex=True)
                
        # Get only the columns that you want
        trainDf = labeledDf[[
            'l1_cluster', 'precision',
            'part_category', 'material_category', 
            'main_process',  
            
            'features0', 'features1', 'features2']]
        # Analysis of which clusters have labeled parts
        uniqueClus = trainDf["l1_cluster"].value_counts()
        uniqueDf = pd.DataFrame(uniqueClus)
        uniqueDf["cluster"] = uniqueDf.index
        uniqueDf.reset_index(level=0, inplace=True)
        
        # Renaming column names to simple index numbers
        trainDf.set_axis([a for a in range(0, trainDf.shape[1])], 
                         axis = 1, inplace = True)
        trainDf[0] = trainDf[0].apply(str)
            
        print("Getting the rules...")
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import fpmax, fpgrowth
        
        trainDf = trainDf[trainDf[0].isin(finalClusters)]
        
        if trainDf.shape[0] != 0:
            trainList = trainDf.values.tolist()
            dataset = []
            for lst in trainList:
                dataset.append([a for a in lst if a != "NA"])
            
            te = TransactionEncoder()
            te_ary = te.fit(dataset).transform(dataset)
            df = pd.DataFrame(te_ary, columns=te.columns_)
            
            frequent_itemsets = fpgrowth(df, min_support=min_support, 
                                         use_colnames=True)
            from mlxtend.frequent_patterns import association_rules
            
            rules = association_rules(frequent_itemsets, 
                                      metric="confidence", 
                                      min_threshold=min_threshold)
            
            # Convert Frozen Set to tuple
            cols = ["antecedents", "consequents"]
            rules[cols] = rules[cols].applymap(lambda x: tuple(x))
            
            
        else:
            rules = []
            
        return uniqueDf, rules, list(set(finalClusters))
    
    
    
    import hnswlib
    def hnsw(queryParts, dataDf, hnswFilePath):
        params = {'metric': 'cosine', # possible options are l2, cosine or ip
                  "ef_constr": 100,
                  "m_constr": 50, # TODO 50
                  "ef_recall": 10,
                  "query_k_val": 10}
        
        
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
            from sklearn.neighbors import KDTree
            tree = KDTree(vectors, leaf_size=4)
            dist, ind = tree.query(queryVector, k=kVal)
            
            # Get the corresponding UUIDs
            resultUuids = [finalUuids[a] for a in ind[0]]
            finalList.append(resultUuids)
            timeSum += time.time() - t
            
        return finalList, timeSum, knnSearchSizes
    
    
    def plot_graph(trnImgPath,
                   imgShowList # The Training part UUIDs to display
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
                     nrows_ncols = (5, 
                                    10
                                    ),  
                     axes_pad=0.3,  # pad between axes in inch.
                     )   
        
        appender = []
        ctr = 1
        for i in range(len(imgShowList)):
            img = cv2.imread(os.path.join(trnImgPath, 
                             imgShowList[i] + ".jpg"))
            appender.append(img)    
        
        # Final List to Display!!!!!!!!!!!!
        finalList = appender
        for ax, im in zip(grid, finalList):
            ax.imshow(im)
            ax.set_axis_off()
            
        plt.show()
        fig.savefig("C:\\Users\\abharad3\\Desktop\\" + "query.png")
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
    
    # for i in range(0, len(classList)):
    for i in range(27, 28):
        # ------------------Choosing the query class-----------------
        # currentClass = classList[i]
        currentClass = "Machining"
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
        for j in range(labels.shape[0]):
            imgIdList = [trnUuids[a] for a in labels[j, :]]
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
        
        finalList = []
        for lst in finalShowList:
            for element in lst:
                finalList.append(element)
        print("Starting the Rule Mining: ")
        uniqueDf, rules, finalClusters = get_rules(finalList)
        
    
    
    totalData = uniqueDf[uniqueDf["cluster"].isin(
                            [int(b) for b in finalClusters])]
    searcher = 'Machining'
    finalRules = rules[rules['antecedents'].apply(
                        lambda x: True if searcher
                                   in x else False)]
    
    # # Plot checking the clusters
    # runner = list(driver.session().run("""
    # MATCH (p:part) WHERE p.l1_cluster = $searchId
    # RETURN p.uuid AS uuid
    # """, searchId = 2))
    # shownImgs = [a["uuid"] for a in runner]
    # if len(shownImgs) > 50:
    #     import random
    #     shownImgs = random.sample(shownImgs, 50)
    # plot_graph(trnImgPath, shownImgs)
    


# 'Same Context' Part Recommendation
class ContextualReco():
    def __init__(self):
        pass
    
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
            knnDataDf):
        
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
            from sklearn.neighbors import KDTree
            tree = KDTree(vectors, leaf_size=4)
            dist, ind = tree.query(queryVector, k=5)
            
            # Get the corresponding UUIDs
            resultUuids = [finalUuids[a] for a in ind[0]]
            finalList.append(resultUuids)
            timeSum += time.time() - t
            
        return finalList, timeSum, knnSearchSizes
    
    
    def plot_graph(trnImgPath,
                       imgShowList # The Training part UUIDs to display
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
                     nrows_ncols = (5, 
                                    10
                                    ),  
                     axes_pad=0.3,  # pad between axes in inch.
                     )   
        
        appender = []
        ctr = 1
        for i in range(len(imgShowList)):
            img = cv2.imread(os.path.join(trnImgPath, 
                             imgShowList[i] + ".jpg"))
            appender.append(img)    
        
        # Final List to Display!!!!!!!!!!!!
        finalList = appender
        for ax, im in zip(grid, finalList):
            ax.imshow(im)
            ax.set_axis_off()
            
        plt.show()
        fig.savefig("C:\\Users\\abharad3\\Desktop\\" + "query.png")
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
    
    for i in range(0, 1):#len(classList)):
    
        # ------------------Choosing the query class-----------------
        currentClass = "Unthreaded_Flanges"
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
        
        # if numQuery == 20:
        #     plot_graph(trnImgPath, finalShowList, queryFullPaths, currentClass, 0)
        #     plot_graph(trnImgPath, finalShowList, queryFullPaths, currentClass, 10)
        # else:
        #     plot_graph(trnImgPath, finalShowList, queryFullPaths, currentClass, 0)
        
        # dataDict["plot_time"] = time.time() - a
        # dataList.append(dataDict)
    
    
    # dataDf = pd.DataFrame(dataList)
    # dataDf.to_csv(r"D:\AKS_DATA\retrieval_results\time_data.csv")
    
    
    
    '''--------> Finding associated parts from same sub-assemblies '''
    
    # Get the matching Sub-Assems for all the resulting parts
    matchParts = [uuid for group in finalShowList for uuid in group]
    query = driver.session().run("""
    UNWIND $mp AS pid
    MATCH (p:part)-[s:subset_of]->(as) WHERE p.uuid = pid
    RETURN as
    """, mp = [matchParts[0], matchParts[5], matchParts[10],
                matchParts[15], matchParts[20]])
    query = list(query)
    query = [a['as'] for a in query]
    saIds = [b['uuid'] for b in query]
    
    # Find all the parts under this subassembly
    aQuery = driver.session().run("""
    UNWIND $saIds as sa
    MATCH (p:part)-[r:subset_of]->(s) WHERE s.uuid = sa
    RETURN p.uuid AS uuid, p.l1_cluster as l1c
    """, saIds = saIds)
    aQuery = list(aQuery)
    pIds = [b["uuid"] for b in aQuery]
    l1cIds = [b["l1c"] for b in aQuery]
    thisDf = pd.DataFrame(list(zip(pIds, l1cIds)), columns = ['uuid', 'l1_cluster'])
    
    # Sort the clusters by their counts
    from collections import Counter
    import operator
    fc = Counter(l1cIds)
    sorted_x = sorted(fc.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    
    finalList = []
    # Get one part from each of the top clusters to display
    for ele in sorted_x:
        dfa = thisDf[thisDf['l1_cluster'] == ele[0]]
        finalList.append(dfa['uuid'].sample(n=1, random_state = 0).iloc[0])
    
    plot_graph(trnImgPath, finalList[0:50])
    
    
    
    # # Plot checking the clusters
    # runner = list(driver.session().run("""
    # MATCH (p:part) WHERE p.l1_cluster = $searchId
    # RETURN p.uuid AS uuid
    # """, searchId = 1))
    # shownImgs = [a["uuid"] for a in runner]
    # if len(shownImgs) > 50:
    #     import random
    #     shownImgs = random.sample(shownImgs, 50)
    # plot_graph(trnImgPath, shownImgs)
    print(0)
