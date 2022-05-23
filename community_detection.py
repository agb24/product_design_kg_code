# -*- coding: utf-8 -*-
"""
Community Detection
"""
import datetime
import workflow_neo4j_cluster_V2 as wflo
import os
import pickle
import numpy as np

import neo4j
import pandas as pd


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

def comm_detec():
    lLimit = 0.85
    uLimit = 1.0
    saveFolder = "D:\\AKS_DATA\\algoresults"
    
    t = datetime.datetime.now()
    # Get all parts and connections
    searchStr = "EXISTS(p.uuid)"
    deffer = wflo.create_nxgraph_for_clustering(lLimit, uLimit, searchStr, 1)
    graph = deffer.G
    print("Graph took: {}".format(datetime.datetime.now() - t))
    
    
    # """ Removing all edges below 0.875 """
    # lLimit = 0.875
    # uLimit = 1.0
    
    # threshold = lLimit
    # # filter out all edges above threshold and grab id's
    # low_edges = list(filter(lambda e: e[2] < threshold, (e for e in
    #                                                      graph.edges.data('weight'))))
    # low_ids = list(e[:2] for e in low_edges)
    # # remove filtered edges from graph G
    # graph.remove_edges_from(low_ids)
    
    
    algorithms = ["Louvain", "Leiden", "Label Propagation",
                      "Walktrap", "Infomap", "Surprise", "Spectral"]
    
    # Get Cluster IDs for Clusters from the Above Algorithms
    for algoName in algorithms:
        t = datetime.datetime.now()
        print("Running: {}".format(algoName))
        nodeCommDict, commNodeDict = wflo.clustering(graph, algoName)
        with open(os.path.join(saveFolder, "{}_{}_{}.pkl".format(lLimit, uLimit,
                                                        algoName)), "wb") as file:
            pickle.dump([nodeCommDict, commNodeDict], file)
    
        print("{} took: {}".format(algoName, str(datetime.datetime.now() - t)))

def metrics():
    lLimit = 0.85
    uLimit = 1.0
    saveFolder = "D:\\AKS_DATA\\algoresults"

    t = datetime.datetime.now()
    # Get all parts and connections
    searchStr = "EXISTS(p.uuid)"
    deffer = wflo.create_nxgraph_for_clustering(lLimit, uLimit, searchStr, 1)
    graph = deffer.G
    print("Graph took: {}".format(datetime.datetime.now() - t))

    # Get signatures
    with open("D:\\AKS_DATA\\Cosine_Dist\\all_signature.pkl", "rb") as file:
        signatures = pickle.load(file)
    signatures["new_signature"] = [a[0][0] for a in signatures["signature"]]

    # Making signatures into 1-D Vector
    signArr = np.vstack(signatures["new_signature"])
    signArr3d = np.array([signArr])
    shape = signArr3d.shape
    signArr3d = np.reshape(signArr3d, (shape[1], shape[0], shape[2]))

    # Get the Neo4J Node IDs
    lvlNodeQuery = '''
                    MATCH (a:part)
                    RETURN a.uuid as uuid, a
                   '''
    neoRelObj = wflo.CreateNeoRel(source, lLimit, uLimit, lvlNodeQuery)
    with driver.session() as session:
        dictIds = session.write_transaction(neoRelObj.get_node_ids)


    # Start reading the result files, and get the Metrics!
    t = datetime.datetime.now()

    resultsDir = "D:\\AKS_DATA\\algoresults"
    fileList = os.listdir(resultsDir)
    fileList = [a for a in fileList if ".pkl" in a]

    evalDictList = []
    for pklFile in fileList:
        pklPath = os.path.join(resultsDir, pklFile)
        with open(pklPath, "rb") as file:
            nodeCommDict, commNodeDict = pickle.load(file)

        # Get DF of signatures, community, IDs and UUIDs
        partIdDf = pd.DataFrame(dictIds.items(), columns = ["uuid", "id"])
        nodeCommDf = pd.DataFrame(nodeCommDict.items(), columns = ["id", "comm"])
        oldVectorDf = pd.merge(partIdDf, nodeCommDf, how = "inner", on=["id"] )
        vectorDf = pd.merge(oldVectorDf, signatures, how = "inner", on=["uuid"] )

        # Find the lower and upper limits from filename
        limits = pklFile.rpartition("_")[0]
        """ Removing all edges below Lower Limit """
        lLimit = float(limits.replace(limits[-4:], ""))
        uLimit = float(limits[-3:])

        print(str(pklFile.rpartition("_")[2][:-4]) + ": " + str(lLimit)
              + " to " + str(uLimit))

        # Filter out lower edges if the LL > 0.85
        if lLimit != 0.85:
            threshold = lLimit
            # filter out all edges above threshold and grab id's
            low_edges = list(filter(lambda e: e[2] < threshold, (e for e in
                                                                  graph.edges.data('weight'))))
            low_ids = list(e[:2] for e in low_edges)
            # remove filtered edges from graph G
            graph.remove_edges_from(low_ids)

        # Finally, get the metrics and store them in a list
        evalDict = wflo.network_eval_metrics(graph, commNodeDict, nodeCommDict, vectorDf)
        evalDict["algorithm"] = pklFile.rpartition("_")[2][:-4]
        evalDict["lower_limit"] = lLimit
        evalDict["upper_limit"] = uLimit
        evalDictList.append(evalDict)

    print("Metrics took: {}".format(datetime.datetime.now() - t))

    # Saving metrics
    with open("D:\\AKS_DATA\\algoresults\\evaluation_results.pkl", "rb") as file:
        checker = pickle.load(file)

    finalList = []
    for d in checker:
        newDict = {}
        newDict = d
        newDict["modul_final"]= d["modularity"][2]
        newDict["modul_dens_final"] = d["modularity_density"][2]
        finalList.append(newDict)

    evalDf = pd.DataFrame(finalList)
    # evalDf.to_pickle("D:\\AKS_DATA\\algoresults\\evaldf.pkl")
    
# Plot metrics and find significance comparison
def plot_metrics():
    with open("D:\\AKS_DATA\\algoresults\\evaldf.pkl", "rb") as file:
        evalDf = pickle.load(file)
    cols = evalDf.columns
    print(cols)

    evalDf = evalDf.drop(["modularity", "modularity_density",
                          "silhouette_samples"], axis = 1)

    import matplotlib.pyplot as plt
    algorithms = ["Louvain", "Leiden", "Label Propagation",
                  "Walktrap", "Infomap", "Surprise", "Spectral"]
    lineWidth = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    color = ["r", "m", "c", "k", "y", "g", "b"]
    for i, algo in enumerate(algorithms):
        thisDf = evalDf[evalDf["algorithm"] == algo]
        plt.plot(thisDf["lower_limit"], thisDf["silhouette_score"], color[i],
                 linestyle = "-",
                 marker = "o",
                 label = algo,
                 lw = lineWidth[i])
    plt.legend()
    plt.xlabel("Cosine Dist: Lower Threshold")
    plt.ylabel("Silhouette Score: Cosine Dist")
    plt.show()


    for i, algo in enumerate(algorithms):
        thisDf = evalDf[evalDf["algorithm"] == algo]
        plt.plot(thisDf["lower_limit"], thisDf["modul_dens_final"], color[i],
                 linestyle = "-",
                 marker = "^",
                 label = algo,
                 lw = lineWidth[i])
    plt.legend()
    plt.xlabel("Cosine Dist: Lower Threshold")
    plt.ylabel("Modularity Density")
    plt.show()

    # POST-HOC TESTS
    # ROWS ARE BLOCKS (i.e the different measures used), and
    # COLUMNS ARE GROUPS (i.e, the primary factor, the ALGORITHM)
    finalEvalDf = evalDf[evalDf["lower_limit"] == 0.9]
    finalEvalDf["mean_internal_dens"] = 0
    for k in range(0, finalEvalDf.shape[0]):
        zz = finalEvalDf['internal_edge_density'].iloc[k]
        finalEvalDf["mean_internal_dens"].iloc[k] = sum(zz)/len(zz)

    a1 = finalEvalDf["silhouette_score"].to_numpy()
    a2 = finalEvalDf["mean_internal_dens"].to_numpy()
    a3 = finalEvalDf["modul_dens_final"].to_numpy()
    colNames = finalEvalDf["algorithm"]

    postHocArr = np.vstack((a1, a2, a3))
    import sklearn
    normalArray = sklearn.preprocessing.normalize(postHocArr, norm="l1")

    # Friedman Chi Square
    import scipy.stats as ss
    fried = ss.friedmanchisquare(*normalArray.T)
    # FriedmanchisquareResult(statistic=14.857142857142847,
    # pvalue=0.021397402189979795)
    # By Friedman chi-square test, SIGNIFICANT AT 0.05.

    import scikit_posthocs as sp
    nemFried = sp.posthoc_nemenyi_friedman(normalArray)
    nemFried.columns = colNames.to_list()
    nemFried["algorithm"] = colNames.to_list()
    # The two best performing algos are WalkTrap and Label Propagation.
    # However, considering Silhouette Score, Mean Internal Edge Density, and
    # Modularity Density measures, there is no significant difference
    # between these two clusterings (when using posthoc Friedman (and Nemenyi-Friedman)
    # analysis.) There is significant diff between other cases,
    # but we choose the one with the best Modularity Density and move forward,
    # since it shows high correlation to NMI index. See paper in references
    # by Tanmoy Chakraborty
    # CHOOSE LABEL PROPAGATION AT LOWER LIMIT 0.9all, SINCE IT HAS HIGHER MD VALUE.


# Update clusters in Neo4J using Label Propagation dict
# Get cluster centers & update them
# Create next set of connections
def update_clusters():
    initLLevel = 0.85
    initULevel = 1.0
    algoName = "label_propagation"

    # Create Graph for Clustering
    searchStr = "EXISTS(p.uuid)"
    deffer = wflo.create_nxgraph_for_clustering(initLLevel, initULevel, searchStr, 1)
    graph = deffer.G

    """ Removing all edges below 0.9 """
    lLimit = 0.9
    uLimit = 1.0

    threshold = lLimit
    # filter out all edges above threshold and grab id's
    low_edges = list(filter(lambda e: e[2] < threshold, (e for e in
                                                          graph.edges.data('weight')))
                      )
    low_ids = list(e[:2] for e in low_edges)
    # remove filtered edges from graph G
    graph.remove_edges_from(low_ids)

    # Get Cluster IDs for Clusters for Label Propagation
    print("Updating cluster IDs in Neo4J: ")
    with open("D:\\AKS_DATA\\algoresults\\" + str(lLimit) + "_1.0_Label Propagation.pkl", "rb") as file:
        nodeClusterDict, clusterNodeDict = pickle.load(file)
    listOfDicts = []
    for i, key in enumerate(nodeClusterDict.keys()):
        dicter = {}
        dicter["node_id"] = key
        dicter["cluster_id"] = nodeClusterDict[key][0]
        listOfDicts.append(dicter)

    # Update the CLUSTER IDs in the Neo4J Graph USING nodeClusterDict
    # The values of nodeClusterDict is a LIST
    # TODO: CAREFUL, you multiply lLimit by 10 since neo4j cant use '.'
    arg = {"algo": algoName, "level": str(int(lLimit*10))}
    clusterUpdateStr = """
                        CALL apoc.periodic.iterate("UNWIND $rList AS x RETURN x",

                        "MATCH (a:part) WHERE ID(a) = x.node_id " +
                        "SET a.{algo}_cluster_{level} = x.cluster_id ",

                        """.format(**arg)
    clusterUpdateStr += """
                        {batchSize:$batchSize,
                        batchMode:"BATCH",
                        parallel:false,
                        params:{rList:$relnList}})
                        """
    retval = driver.session().run(clusterUpdateStr, batchSize = 5000, relnList = listOfDicts)


    # Get Cluster Centers
    # Find Centers and Update the cluster center property
    print("Getting cluster centers & updating:")
    # TODO: CAREFUL, you multiply lLimit by 10 since neo4j cant use '.'
    centerObj = wflo.CenterDetec(algoName, str(int(lLimit*10)))
    with driver.session() as session:
        try:
            centerObj.drop_graph(session)
        except:
            pass
    centerObj.run()

    # Create next set of connections
    print("Creating next set of connections: ")
    # TODO: CAREFUL, you multiply lLimit by 10 since neo4j cant use '.'
    argx = {"algo": algoName, "level": str(int(lLimit*10))}
    lvlNodeQuery = '''
                    MATCH (a:part) WHERE a.{algo}_center = {level}
                    RETURN a.uuid as uuid, a
                    '''.format(**argx)
    # DEFINING NEW LIMITS FOR CREATING RELATIONS!!!!
    newLLimit = 0.6
    newULimit = 0.9
    neoCreateObject = wflo.CreateNeoRel(source, newLLimit, newULimit, lvlNodeQuery)
    neoCreateObject.run()


if __name__ =="__main__":
    comm_detec()
    metrics()
    plot_metrics()
    update_clusters()
