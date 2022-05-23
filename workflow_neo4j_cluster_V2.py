# -*- coding: utf-8 -*-
"""

"""
import neo4j
import networkx as nx
import cdlib
from cdlib import algorithms as cdalgo
from cdlib import evaluation as cdeval
import math
import pickle
import time
import datetime
import os
import pandas as pd
import numpy as np
import infomap
import numpy as np
from scipy.sparse import find, csr_matrix
from sklearn.metrics import silhouette_samples, silhouette_score

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

class NxGraph():
    def __init__(self, ll, ul, query1, query2, whichTime):
        self.ll = ll
        self.ul = ul
        self.query1 = query1
        self.query2 = query2
        self.t = time.time()
        self.finalRelList = []
        self.whichTime = whichTime

    def create_nodes(self):
        print("Getting nodes from Neo4J and adding to NX Graph")
        self.G = nx.Graph()
        # Getting nodes from Neo4J
        results1 = driver.session().run(self.query1)
        nodes = list(results1.graph()._nodes.values())
        self.nodeIds = [a.id for a in nodes]
        # Creating nodes
        for node in nodes:
            self.G.add_node(node.id, labels=[x for x in node._labels], properties=node._properties)

    def create_rels_scratch(self):
        print("Getting relations from Neo4J and adding to NX Graph")
        nodeIds = self.nodeIds
        query2 = self.query2

        # If running it for the first time,
        # Consider all nodes when querying
        if self.whichTime == 1:
            finalRelList = []
            limit = math.floor(len(nodeIds)/1000)
            # Getting relations from Neo4J: Looping through 1000 nodes at a time, and
            # getting all the relations for each of these nodes, for given cosine_sim limits.
            for i in range(0, limit-1):
                results2 = driver.session().run(query2,
                                                nodeIdList = nodeIds[i*1000:(i+1)*1000])
                rels = list(results2.graph()._relationships.values())
                finalRelList = finalRelList + rels
                if (i%10 == 0):
                    print(str(i*1000) + " of " + str(len(nodeIds)))
                    print("Rels so far: {}".format(len(finalRelList)))
            # Looping thro final nodes not included in above loop
            finalRes = driver.session().run(query2,
                                            nodeIdList = nodeIds[(limit)*1000:])
            relsFinal = list(finalRes.graph()._relationships.values())
            finalRelList = finalRelList + relsFinal

        # IF NOT RUNNING IT FOR THE FIRST TIME,
        # then consider two different lists of nodes.
        # First list is the subset of nodes in that particular loop
        # Second list is all the nodes considered in this query
        elif self.whichTime != 1:
            finalRelList = []
            limit = math.floor(len(nodeIds)/1000)
            # Getting relations from Neo4J: Looping through 1000 nodes at a time, and
            # getting all the relations for each of these nodes, for given cosine_sim limits.
            for i in range(0, limit-1):
                results2 = driver.session().run(query2,
                                                nodeIdList1 = nodeIds[i*1000:(i+1)*1000],
                                                )
                # nodeIdList2 = nodeIds)

                rels = list(results2.graph()._relationships.values())
                finalRelList = finalRelList + rels
                if (i%1 == 0):
                    print(str(i*1000) + " of " + str(len(nodeIds)))
                    print("Rels so far: {}".format(len(finalRelList)))
            # Looping thro final nodes not included in above loop
            finalRes = driver.session().run(query2,
                                            nodeIdList1 = nodeIds[(limit)*1000:],
                                            )
            # nodeIdList2 = nodeIds)

            relsFinal = list(finalRes.graph()._relationships.values())
            finalRelList = finalRelList + relsFinal

        self.finalRelList = finalRelList


        # Creating relationships
        for u, rel in enumerate(finalRelList):
            self.G.add_edge(rel.start_node.id, rel.end_node.id,
                            key=rel.id, type=rel.type,
                            properties=rel._properties,
                            weight = rel._properties["cosine_sim"])
            if (u%1000000 == 0):
                print("Gone through rel: " + str(u) + " of " + str(len(finalRelList)))
                print(nx.info(self.G))

        print("Done. Time taken for relations: ", str(self.t - time.time()))

    # For 0.85 <= Cosine_Sim <= 1.0, I have already queried relations previously and saved.
    # Just reuse this for the given limits, to save time.
    def create_rels_by_import(self):
        print("Getting relations from Pickle File and adding to NX Graph")
        if((self.ll == 0.85) and (self.ul == 1.0)):
            with open(relFileLink_85_1, "rb") as relProp_file:
                finalRelList = pickle.load(relProp_file)

        for u, rel in enumerate(finalRelList):
            self.G.add_edge(rel["start_node_id"], rel["end_node_id"], key=rel["rel_key"],
                       rtype=rel["rel_type"],
                       properties=rel["prop"],
                       weight = rel["prop"]["cosine_sim"])

            if (u%1000000 == 0):
                print("Gone through rel: " + str(u) + " of " + str(len(finalRelList)))
                print(nx.info(self.G))

class CreateNeoRel():
    def __init__(self, src, lLimit, uLimit, lvlNodeQuery):#src = "C:\\Users\\abharad3\\OneDrive - North Carolina State University\\Cosine_Dist"
        self.source = src
        self.lLimit = lLimit
        self.uLimit = uLimit
        self.lvlNodeQuery = lvlNodeQuery

    def read_df(self, src, fileName):
        path = os.path.join(src, fileName)
        dFrame = pd.read_hdf(path)
        #dFrame["uuid_rows"][1]
        return dFrame

    def read_cosine_dist(self, index, src):

        # Reading the cosine signatures
        files = [a for a in os.listdir(src) if "cosine_similarity_" in a]
        fileName = files[index-1]
        t = time.time()
        print("Reading the Cosine DF:\n")
        cosineDf = self.read_df(src, fileName)
        #cosineDf["uuid_rows"][1]
        print("Done " + str(time.time() - t))
        return cosineDf

    # USE NUMPY FOR SPEED!!!!!!!!!!!
    def upload_relns_numpy(self, nodeDict, cosineDf, lLimit, uLimit, fileIndex):

        uuidRow = cosineDf.loc[:, "uuid_rows"].tolist()
        uuidCol = list(cosineDf.columns)
        uuidCol.remove("uuid_rows")

        # Convert DF to Numpy array, excluding the "uuid_rows"
        npCosArr = cosineDf.loc[:, cosineDf.columns != "uuid_rows" ].to_numpy()

        # Querying just the rows within the cosine limits reqd. This gives
        # TRUE/FALSE boolean matrix for the specified conditions.
        # For ul==1.0, I include both the Upper and Lower Limits.
        if (uLimit == 1.0):
            npCosBool = (npCosArr>=lLimit) & (npCosArr<=uLimit)
        # For all others, [lLimit,uLimit), include Lower and exclude Upper Lt.
        else:
            npCosBool = (npCosArr>=lLimit) & (npCosArr<uLimit)

        # Getting the TUPLE of LISTS indices of where "True" values
        # are present in the boolean matrix
        npCosTrue = np.where(npCosBool)

        # Getting list of (X,Y) tuples of the above two lists in "npCosTrue" tuple
        npCosTupList = list(zip(npCosTrue[0], npCosTrue[1]))
        # Getting unique combinations of tuples (X,Y) where (X,Y) == (Y,X)
        uniqueTupList = [tuple(x) for x in set(map(frozenset, npCosTupList))]
        # Swapping around the (X,Y) elements in below list, if the index of the ROW > 10,000
        # (which is undefined, since each particular CosineDf has only 10,000 rows)
        modUniqueTupList = []
        # Recording the number of tuples of size 2. Some of these will be of size 1.
        # SELF CONNECTIONS COULD EXIST HERE, BUT THEY ARE REMOVED IN THE LINE:
        # --------------------------> if(selRowNames[i] != selColNames[i]):


        # CHECKING WHETHER THERE ARE ANY ELEMENTS TO ACTUALLY LOOP THROUGH
        if len(uniqueTupList) > 0:
            nonTupleCounter = 0
            for i in uniqueTupList:
                if len(i) == 2:
                    if ( (i[0] >= len(uuidRow)) ):
                        modUniqueTupList.append((i[1], i[0]))
                    else:
                        modUniqueTupList.append(i)
                else:
                    nonTupleCounter += 1

            modUniqueTupList1 = [a for a in modUniqueTupList
                                 if ( a[0]+( (fileIndex - 1) * 10000) != a[1]) ]

            # Putting these into a LIST of TWO LISTS, one for X and one for Y
            finalLists = [ [t[0] for t in modUniqueTupList1],
                           [t[1] for t in modUniqueTupList1] ]

            if ( len(finalLists[0]) == len(finalLists[1]) ):
                print("Final Lists OK.")
            else:
                print("Problem!!! ")
                return

            print( "Original no of Relations, with cosine size filters: " + str(len(npCosTrue[0])) )
            print( "Length after removing most duplicate elements: "
                  + str(len(finalLists[0])) )

            # Get the combination of Column and Row UUID names:
            # These lists are a Row in list one, Column in list two, combo, and
            # each list has same number of elements.
            print(len(uuidRow), max(finalLists[0]))
            selRowNames = [uuidRow[i] for i in finalLists[0]]
            selColNames = [uuidCol[i] for i in finalLists[1]]

            # Get the actual cosine values corresponding to the row and column UUIDs
            selValues = npCosArr[[finalLists[0]], [finalLists[1]]]

            relnList = []
            numSelfConn = 0
            # Looping through each of these values and creating relations
            t = time.time()
            for i in range(0, len(selRowNames)):

                # Ensuring no SELF-Connections
                if(selRowNames[i] != selColNames[i]):
                    rowNode = nodeDict[selRowNames[i]]
                    colNode = nodeDict[selColNames[i]]
                    cosineVal = selValues[0][i]

                    singleRelnDict = {
                                        "type": "similar_to",
                                        "cosine_sim": float(round(cosineVal, 2)),
                                        "node1_uuid": selRowNames[i],
                                        "node2_uuid": selColNames[i],

                    }
                                        # "node1_node": rowNode,
                                        # "node2_node": colNode

                    relnList.append(singleRelnDict)

                    if (i == len(selRowNames) - 1):
                        print("Relation-> " + str(i) + " of potential " + str(len(selRowNames))
                              + ". Time elapsed: " + str(time.time() - t))

                else:
                    numSelfConn = numSelfConn + 1

            return(relnList, len(npCosTrue[0]), len(finalLists[0]), len(relnList))

        # IF NO ELEMENTS TO CREATE RELATIONS FOR, RETURN EMPTY
        else:
            return([], len(npCosTrue[0]), 0, 0)

    def create_mult_reln_apoc(self, tx, relnList, dictIds, batchSize):

        lister = []
        upld = tx.run('''
                          CALL apoc.periodic.iterate("UNWIND $rList AS x RETURN x",

                          "MATCH (a:part), (b:part) WHERE ID(a)=x.node1_nid " +
                              "AND ID(b)=x.node2_nid " +
                          "MERGE (a)-[r:similar_to {cosine_sim:x.cosine_sim}]->(b) ",

                          {batchSize:$batchSize,
                              batchMode:"BATCH",
                              parallel:false,
                              params:{rList:$relnList}})

                      ''',
                      relnList = relnList, batchSize = batchSize
                      )
        # YIELD batches, total, errorMessages
        #                   RETURN batches, total, errorMessages
        for rec in upld:
            lister.append(rec)

        return lister

    def get_node_ids(self, tx):
        lister = []
        result = tx.run('''
                           MATCH (a:part) RETURN ID(a) AS ids, a.uuid AS uuids
                       ''')
        for i in result:
            lister.append(i)

        keys = [a[1] for a in lister]
        values = [a[0] for a in lister]
        diction = dict(zip(keys, values))
        return diction

    def get_nodes_reqd(self, tx, lvlNodeQuery):
        #THIS IS FOR: louvain_center_pgrk1
        nodeDict = {}
        result = tx.run(lvlNodeQuery)
        for i in result:
            nodeDict[i[0]] = i[1]
        return nodeDict

    def run_checker(self):
        lLimit = self.lLimit
        uLimit = self.uLimit
        lvlNodeQuery = self.lvlNodeQuery
        src = self.source

        finalRel = []
        for k in range(1,13):
            print("\n\n\n STARTING TO READ DATAFRAME NO. " + str(k))
            cosineDf = self.read_cosine_dist(k, src)
            with driver.session() as session:
                nodes = session.write_transaction(self.get_nodes_reqd,
                                                      lvlNodeQuery)
            print("Number of nodes searched for: {}".format(
                      len(list( nodes.keys() ))) )
            # Restricting the cosineDf columns and rows to only those nodes required:
            subFinalDf = cosineDf[cosineDf['uuid_rows'].isin(list( nodes.keys() ))]
            finalDf = subFinalDf[subFinalDf.columns.intersection(list( nodes.keys() )
                                                                            + ["uuid_rows"]
                                                                )
                                 ]
            # CHECK IF THE DF HAS ANY ROWS
            if finalDf.shape[0] > 0:

                relations, origNum, removDuplNum, finalNum = self.upload_relns_numpy(
                                                                                    nodes,
                                                                                    finalDf,
                                                                                    lLimit,
                                                                                    uLimit,
                                                                                    k
                                                                                )
            finalRel = finalRel + relations

        return finalRel

    def run(self):
        lLimit = self.lLimit
        uLimit = self.uLimit
        lvlNodeQuery = self.lvlNodeQuery
        src = self.source

        relations = []
        for k in range(1,13):
            print("\n\n\n STARTING TO READ DATAFRAME NO. " + str(k))
            cosineDf = self.read_cosine_dist(k, src)
            # Returning list of required nodes
            with driver.session() as session:
                nodes = session.write_transaction(self.get_nodes_reqd,
                                                  lvlNodeQuery)

            print("Number of nodes searched for: {}".format(
                  len(list( nodes.keys() ))) )

            # Restricting the cosineDf columns and rows to only those nodes required:
            subFinalDf = cosineDf[cosineDf['uuid_rows'].isin(list( nodes.keys() ))]
            finalDf = subFinalDf[subFinalDf.columns.intersection(list( nodes.keys() )
                                                                 + ["uuid_rows"]
                                                                )
                                ]


            # CHECK IF THE DF HAS ANY ROWS
            if finalDf.shape[0] > 0:

                relns, origNum, removDuplNum, finalNum = self.upload_relns_numpy(
                                                                                    nodes,
                                                                                    finalDf,
                                                                                    lLimit,
                                                                                    uLimit,
                                                                                    k
                                                                                )
                relations = relations + relns
            else:
                print("Skipping this DF entirely; it does not have any rows we need.")

        """ GETTING THE NODE IDS (NEO4J IDS) USING THE FabWave UUIDs"""
        with driver.session() as session:
            dictIds = session.write_transaction(self.get_node_ids)
        for i in range(len(relations)):
            uid1 = relations[i]["node1_uuid"]
            uid2 = relations[i]["node2_uuid"]
            relations[i]["node1_nid"] = dictIds[uid1]
            relations[i]["node2_nid"] = dictIds[uid2]

        """REMOVING All Repeated Relationships"""
        # Put all relations in a DF
        self.relDf = pd.DataFrame(relations)
        relations = []
        # Seperate ID columns to find unique combinations of IDs
        df1 = self.relDf[["node1_nid", "node2_nid"]]
        # This line gets all unique combinations of relationships,
        # where (X.id, Y.id) == (Y.id, X.id)
        # by sorting each row by its column value,
        # and then removing all duplicates, keeping only the first ones
        newdf1 = df1[pd.DataFrame(np.sort(df1.values),
                            columns=df1.columns,
                            index=df1.index).duplicated(keep="first")]
        # This MERGES the original relation DF, and the sorted unique DF,
        # to get the intersection of these two DFs.
        # Gives a DF with unique relations ONLY.
        finalRelDf= pd.merge(newdf1, self.relDf,
                             how = "inner", on=["node1_nid", "node2_nid"] )
        print("---------------------------------------------------")
        print("THE FINAL FINAL UNIQUE DATAFRAME'S SHAPE IS->> {}".format(
                                                        finalRelDf.shape[0])
                                                                        )
        print("---------------------------------------------------")

        # THIS IS HOW YOU Loop thro all the rows of the Unique Relations DF
        # for row in finalRelDf.itertuples(index = True, name = "Pandas"):

        # Converting unique DF to a list of dictionaries ->
        finalRelations = finalRelDf.to_dict("records")
        print("Final relations list length: {}".format(len(finalRelations)))
        print("---------------------------------------------------")


        """ FINALLY UPLOADING RELATIONS """

        relCtr = 0
        tt = time.time()
        batchSize = 10000
        print("Starting to upload Relations at..." + str(tt))

        # 2 LEVELS OF BATCHING:
        # BATCHING RELATIONS IN GROUPS OF 500,000; THEN, BATCHING THEM
        # 10,000 AT A TIME, IN ORDER TO PREVENT JAVA HEAP ERROR
        if len(finalRelations) > 500000:
            ranges = []
            for j in range(0, math.floor( len(finalRelations) / 500000 )):
                ranges.append( (j*500000, (j+1)*500000) )
            for rang_e in ranges:
                relnList = finalRelations[rang_e[0]: rang_e[1]]
                with driver.session() as session:
                    upload = session.write_transaction(self.create_mult_reln_apoc,
                                                            relnList,
                                                            dictIds,
                                                            batchSize
                                                            )
                print("Time Elapsed for " + str(rang_e) + " relations: "
                      + str(time.time() - tt) )
            # OVERFLOW FROM PREVIOUS LOOP
            if(ranges[-1][1] < len(finalRelations)):
                newRelnList = finalRelations[ranges[-1][1]:]
                with driver.session() as session:
                    upload = session.write_transaction(self.create_mult_reln_apoc,
                                                        newRelnList,
                                                        dictIds,
                                                        batchSize
                                                        )
                    print("Time Elapsed for " + str(ranges[-1][1]) + ":"
                          + str(len(finalRelations))
                          + " relations: "
                          + str(time.time() - tt) )

        # NO NEED FOR 2 LEVELS OF BATCHING, SINCE THERE ARE LESS THAN 500,000 RELATIONS.
        # ALL OF THEM UPLOADED IN BATCHES OF 10,000.
        elif 0 < len(finalRelations) <= 500000:
            relnList = finalRelations
            with driver.session() as session:
                upload = session.write_transaction(self.create_mult_reln_apoc,
                                                    relnList,
                                                    dictIds,
                                                    batchSize
                                                    )

        # print("Original, RemovDupl, FINALNUM, Number of Diag. Conn")
        # print(origNum, removDuplNum, finalNum, origNum - finalNum)

        else:
            print("No relations to upload")


class CenterDetec():
    def __init__(self, algo, level):
        self.algo = algo
        self.level = level

    # Getting list of all louvain_cluster IDs
    def get_comm_ids(self, tx, algo, level):
        lister = []
        arg = {"algo": algo, "level": level}
        query = '''
                    MATCH (p:part) WITH DISTINCT p.{algo}_cluster_{level} AS community
                    RETURN community
                '''.format(**arg)
        upld = tx.run(query)
        for rec in upld:
            lister.append(rec)

        return lister

    # Getting all nodes from each community with cluster IDs
    def get_nodes_from_comm(self, tx, algo, level, clustID):
        lister = []
        check = []
        arg = {"algo": algo, "level": level}
        query = """
                    MATCH (p:part) WHERE p.{algo}_cluster_{level} = $cID RETURN ID(p) AS id
                """.format(**arg)
        checker = tx.run(query, cID = clustID)
        for c in checker:
            check.append(c)

        if len(check) > 1:

            query1 = """
                    CALL gds.graph.create.cypher('tekkGraph',
                      'MATCH (p:part) WHERE p.{algo}_cluster_{level} = $cID RETURN ID(p) AS id, labels(p) as labels',
                      'MATCH (a:part)-[r:similar_to]-(b:part) WHERE a.{algo}_cluster_{level} = $cID AND b.{algo}_cluster_{level} = $cID RETURN ID(a) AS source, ID(b) AS target, r.cosine_sim as cosine_sim',

                    """.format(**arg)

            query1 += """
                      {
                          parameters:{cID: $clusterID}
                      }

                      )
                    YIELD graphName, nodeCount, relationshipCount, createMillis
                    """
            upld = tx.run(query1, clusterID = clustID)
            for rec in upld:
                lister.append(rec)

            return check, lister

        else:
            return check, check

    def drop_graph(self, tx):
        lister = []
        gg = tx.run("""  CALL gds.graph.drop("tekkGraph") YIELD graphName
                    """)
        for i in gg:
            lister.append(i)
        return lister

    # Getting PageRank of all nodes of each cluster, sorted by the PageRank score
    def page_rank(self, tx):
        lister = []
        pagerank = tx.run("""
                          CALL gds.pageRank.stream(
                                  'tekkGraph',
                                  {relationshipWeightProperty: "cosine_sim"}
                                  )
                          YIELD nodeId, score
                          RETURN nodeId, score
                          ORDER BY score DESC
                          """)
        for r in pagerank:
            lister.append(r)

        return lister

    # If cluster has only one part
    def update_clus_center_single(self, tx, algo, level, cluster_id):
        lister = []
        arg = {"algo": algo, "level": level}
        query = '''
                    MATCH (p:part) WHERE p.{algo}_cluster_{level} = $clustID
                    SET p.{algo}_center = {level}
                    RETURN p.uuid
                '''.format(**arg)
        upld = tx.run(query, clustID = cluster_id)
        for rec in upld:
            lister.append(rec)

        return lister

    # Updating the cluster centers markers
    def update_clus_center_multiple(self, tx, algo, level, cluster_id, node_list):
        lister = []
        arg = {"algo": algo, "level": level}
        query = '''
                    MATCH (p:part) WHERE ID(p) = $partID
                    AND p.{algo}_cluster_{level} = $clustID
                    SET p.{algo}_center = {level}
                    SET p.{algo}_center_{level}_score = $score1
                    RETURN p.uuid

                '''.format(**arg)
        upld = tx.run(query, partID = node_list[0]["nodeId"], clustID = cluster_id,
                          score1 = node_list[0]["score"])
        for rec in upld:
            lister.append(rec)

        return lister

    def run(self):
        recorder = {}
        with driver.session() as session:
            # Get all cluster IDs
            ress = self.get_comm_ids(session,
                                    self.algo,
                                    self.level)
            # ALL Community IDs for the given level
            results = [a["community"] for a in ress]

            # Looping through each community ID
            for i, result in enumerate(results):
                # Creating subgraph with the nodes in this cluster
                checker, graph_return = self.get_nodes_from_comm(session,
                                                                  self.algo,
                                                                  self.level,
                                                                  result)

                # If Cluster has more than one node
                if len(checker) > 1:
                    pageRankRes = self.page_rank(session)
                    # Recording PageRank values of this cluster
                    recorder[result]= pageRankRes
                    # Deleting graph if it has been created
                    del_graph = self.drop_graph(session)

                # If Cluster has single node
                else:
                    recorder[result] = result

                if (i%500 == 0):
                    print("Getting centres of cluster number :" + str(i))

            # Looping through the Recorder Dict using "results" list of Cluster IDs
            for j, res in enumerate(results):

                cluster_id = res
                node_list = recorder[cluster_id]

                # If cluster has only one part
                if node_list == cluster_id:
                    retval = self.update_clus_center_single(session,
                                                       self.algo,
                                                       self.level,
                                                       cluster_id)
                # If cluster has multiple parts
                else:
                    retval = self.update_clus_center_multiple(session,
                                                       self.algo,
                                                       self.level,
                                                       cluster_id,
                                                       node_list)

                if (j%500 == 0):
                    print("UPDATING centres of cluster number :" + str(j))

# Create NetworkX Graph before clustering it.
def create_nxgraph_for_clustering(ll, ul, searchStr, whichTime):
    #arg1 = {"part_prop": "EXISTS p.uuid"}
    #arg1 = {"part_prop": "p.louvain_center_pgrk1 = 'T'"}
    arg1 = {"part_prop": searchStr}
    query1 = """
    MATCH (p:part) WHERE {part_prop} RETURN p
    """.format(**arg1)

    if (ul == 1.0):
        arg2 = {"cosine_sim_ll": str(ll) + "<=",
                "cosine_sim_ul": "<=" + str(ul),
                "part_prop": searchStr,
                "mod_part_prop": searchStr[2:]
                }
    # The two values below are added for the cases where UL != 1.0,
    # which implies that this is the second iteration of creating connections.
    # "mod_part_prop": searchStr[2:],
    # "center_value": int(ul*10)}
    else:
        arg2 = {"cosine_sim_ll": str(ll) + "<=",
                "cosine_sim_ul": "<" + str(ul),
                "part_prop": searchStr,
                "mod_part_prop": searchStr[2:],
                "center_value": int(ul*10)}
    print(arg2)

    if whichTime == 1:
        query2 = """
        MATCH (p:part)-[r:similar_to]->(q:part) WHERE ID(p) IN $nodeIdList
        AND {cosine_sim_ll}r.cosine_sim{cosine_sim_ul}
        RETURN p,q,r
        """.format(**arg2)
        # MATCH (p:part) WHERE ID(p) IN $nodeIdList
        # MATCH (q:part) WHERE ID(q) IN $nodeIdList
    elif whichTime != 1:
        query2 = """
        MATCH (p:part)-[r:similar_to]->(q:part) WHERE ID(p) IN $nodeIdList1
        AND q.{mod_part_prop} = {center_value}
        AND {cosine_sim_ll}r.cosine_sim{cosine_sim_ul}
        RETURN p,q,r
        """.format(**arg2)

    deffer = NxGraph(ll, ul, query1, query2, whichTime)
    deffer.create_nodes()
    if((ll == 0.85 and ul == 1.0)):
        deffer.create_rels_by_import()
    else:
        deffer.create_rels_scratch()

    return deffer


# If 'nodeClus' is a NodeClustering object, assign dictYN = 'N'.
# If 'nodeClus' is dict of {Node: Comm} as is the default in Infomap,
# then assign dictYN = 'Y'.
def community_maps(nodeClus, dictYN):
    if dictYN == "N" or dictYN == "n":
        nodeCommId = nodeClus.to_node_community_map()
    else:
        nodeCommId = nodeClus
    community_to_node_dict = {}
    for j, node in enumerate(list(nodeCommId.keys())):
        if dictYN == "N" or dictYN == "n":
            community = list(nodeCommId.values())[j][0]
        else:
            community = list(nodeCommId.values())[j]
        if community in community_to_node_dict.keys():
            community_to_node_dict[community] += [node]
        else:
            community_to_node_dict[community] = [node]
    commNodeId = community_to_node_dict
    return nodeCommId, commNodeId

# Cluster the NetworkX sub-graph, with appropriate algorithm: SINGLE LEVEL, NO HIERARCHY.
# AND, update the CLUSTER IDs in the Neo4J Graph
def clustering(G, algo):

    # algorithms = ["Louvain", "Leiden", "Label Propagation", "Pycombo"
    #               "Walktrap", "Infomap", "Surprise", "Spectral"]

    # Louvain: WEIGHTED
    if ((algo == "Louvain") or (algo == "louvain")):
        nodeClus = cdalgo.louvain(G, weight = "cosine_sim")
    # Leiden: WEIGHTED
    if ((algo == "Leiden") or (algo == "leiden")):
        nodeClus = cdalgo.leiden(G, weights = "weight")
    # Label Propagation: UN-WEIGHTED
    elif ((algo == "Label Propagation") or (algo == "label_propagation")):
        nodeClus = cdalgo.label_propagation(G)
    # Combo algorithm: WEIGHTED
    elif ((algo == "Combo") or (algo == "Pycombo") or (algo=="combo")):
        nodeClus = cdalgo.pycombo(G, weight = "weight")
    # Walktrap: UN-WEIGHTED
    elif ((algo == "Walktrap") or (algo == "walktrap")):
        nodeClus = cdalgo.walktrap(G)
    # Infomap: WEIGHTED
    elif ((algo == "Infomap") or (algo == "infomap")):
        im = infomap.Infomap()  #"--two-level"
        for n in G.nodes():
            im.addNode(n)
        for e in G.edges(data=True):
            im.addLink(e[0], e[1], e[2]["properties"]["cosine_sim"])

        im.run()
        levels = im.numLevels()

        modulesList = []
        mappersList = []
        for j in range(1, levels+1):
            modules = im.get_modules(depth_level = j)
            modulesList.append(modules)
            mappers = community_maps(modules, "y")
            mappersList.append(mappers)

        # Creating the CDLib NodeClustering object, from Infomap package Output
        nodeClus = cdlib.NodeClustering(list(mappersList[0][1].values()), G)

    # Asymptotical Surprise Communities: WEIGHTED
    elif ((algo == "Surprise") or (algo == "surprise")):
        nodeClus = cdalgo.surprise_communities(G, weights = "weight")
    # Spectral Clustering: Applies KMeans onto Fielder's vector
    # UNWEIGHTED
    elif ((algo == "Spectral") or (algo == "spectral")):
        nodeClus = cdalgo.surprise_communities(G, weights = "weight")



    # Getting the actual NODE -> COMMUNITY Matches--->
    nodeCommDict, commNodeDict = community_maps(nodeClus, "n")

    return nodeCommDict, commNodeDict

def network_eval_metrics(G,
                         commNodeDict,
                         nodeCommDict,
                         vectorDf):

    if G != []:
        # nodeClusObj: cdlib.classes.node_clustering.NodeClustering,
        nodeClusObj = cdlib.classes.node_clustering.NodeClustering(
                                            list(commNodeDict.values()),
                                            G)

        evalDict = {}

        # Newman-Girvan Modularity
        evalDict["modularity"] = cdeval.newman_girvan_modularity(G, nodeClusObj)
        # Modularity Density
        evalDict["modularity_density"] = cdeval.modularity_density(G, nodeClusObj)

        # Conductance
        evalDict["conductance"] = cdeval.conductance(G, nodeClusObj,
                                                     summary = False)
        # Internal Edge Density
        evalDict["internal_edge_density"] = cdeval.internal_edge_density(
                                                     G,
                                                     nodeClusObj,
                                                     summary = False)


    try:
        # Silhouette Score: Overall (USING COSINE DIST)
        silScore = silhouette_score(np.vstack(vectorDf["new_signature"]),
                                    [a[0] for a in vectorDf["comm"].tolist()],
                                    metric = "cosine")
        evalDict["silhouette_score"] = silScore
        # Silhouette Score: Sample wise (USING COSINE DIST)
        sampleSilScores = silhouette_samples(np.vstack(vectorDf["new_signature"]),
                                    [a[0] for a in vectorDf["comm"].tolist()],
                                    metric = "cosine")
    except:
        # Silhouette Score: Overall (USING COSINE DIST)
        silScore = silhouette_score(np.vstack(vectorDf["new_signature"]),
                                    [a for a in vectorDf["comm"].tolist()],
                                    metric = "cosine")
        evalDict["silhouette_score"] = silScore
        # Silhouette Score: Sample wise (USING COSINE DIST)
        sampleSilScores = silhouette_samples(np.vstack(vectorDf["new_signature"]),
                                    [a for a in vectorDf["comm"].tolist()],
                                    metric = "cosine")
    evalDict["silhouette_samples"] = sampleSilScores


    return evalDict


class HierClust():
    def __init__(self, adjacency, signatures, completeAdj):
        self.adjacency = adjacency
        self.signatures = signatures
        self.completeAdj = completeAdj

    def paris(self):
        from sknetwork.hierarchy import Paris
        from sknetwork.hierarchy import cut_straight

        adjacency = self.adjacency

        # Each Node can have "uniform" weight or "degree" weight
        # Reorder=True gives Dendrogram order based on
        # non-decreasing order of height
        t = datetime.datetime.now()
        paris = Paris(weights = "degree", reorder = True)
        dendrogram = paris.fit_transform(adjacency)
        print("Paris took: {}".format(datetime.datetime.now()-t))

        # Checking for infinite values
        infBool = np.isfinite(dendrogram[:, 2])
        xx = np.where(np.invert(infBool))
        # THE 2 PROBLEMATIC VALUES: graph.edges[51409][11157] and graph[51408][5224]
        dendrogram[xx[0][0], 2] = 50000
        dendrogram[xx[0][1], 2] = 50000
        # Getting the unique values of the nodes, BOTH old AND newly created
        vArr = np.vstack((dendrogram[0,:], dendrogram[1,:]))
        vUniq = np.unique(vArr)

        # Getting the hierarchies. This is above the 75th percentile of height values.
        t = datetime.datetime.now()
        cutDict = {}
        checker = 100
        thresh = 0
        prevLen = 500000
        while checker != 0:
            cuts = cut_straight(dendrogram, threshold = thresh)
            if len(list(set(cuts))) != prevLen:
                cutDict[thresh] = [cuts, len(list(set(cuts)))]
                thresh += 0.05
                prevLen = len(list(set(cuts)))
                print(prevLen)
            else:
                checker = 0
        print("The hierarchy cuts took: {}".format(datetime.datetime.now()-t))

        self.parisDend = dendrogram
        self.parisCutDict = cutDict

    def ward(self):
        from sknetwork.hierarchy import Ward
        adjacency = self.adjacency
        ward = Ward()
        dendrogram = ward.fit_transform(adjacency)
        self.wardDend = dendrogram

    def louvain(self):
        from sknetwork.hierarchy import LouvainHierarchy
        adjacency = self.adjacency
        louvain = LouvainHierarchy()
        dendrogram = louvain.fit_transform(adjacency)
        self.louvainDend = dendrogram

    def hdbscan(self):
        import hdbscan
        signatures = self.signatures

        clusterer = hdbscan.HDBSCAN(min_cluster_size = 15,
                                    metric = "euclidean",
                                    algorithm = "best",
                                    cluster_selection_method="eom",
                                    prediction_data=True)
        print("HDBSCAN: Fitting the clusterer -->")
        clusterFit = clusterer.fit(signatures)


        print("Getting membership vectors -->")
        softClusters = hdbscan.all_points_membership_vectors(clusterer)
        print("Getting prediction data -->")
        clusterFit.generate_prediction_data()
        print("Getting single linkage tree -->")
        singleLinkage = clusterer.single_linkage_tree_

        self.hdbscanClus = clusterFit
        self.hdbscanSoftClus = softClusters
        self.hdbscanDend = singleLinkage

    def evaluate(adjacency, dendrogram):
        from sknetwork.hierarchy import tree_sampling_divergence, dasgupta_score
        # Tree-Sampling Divergence
        tsDivergencescore = tree_sampling_divergence(adjacency, dendrogram)
        # Dasgupta Score = (1 - Normalized Dasgupta's Cost)
        dasguptaScore = dasgupta_score(adjacency, dendrogram)
        # TODO: Do the Silhouette Coefficient

        return tsDivergencescore, dasguptaScore







# Create Neo4J Conns
def step_one():
    # Creating relations from 0.85 to 0.9
    # All relations from 0.85 to 1.0 have now been included in Neo4J.
    lLimit = 0.85
    uLimit = 0.9

    lvlNodeQuery = '''
                    MATCH (a:part)
                    RETURN a.uuid as uuid, a
                   '''
    neoCreateObj = CreateNeoRel(source, lLimit, uLimit, lvlNodeQuery)
    neoCreateObj.run()

# Save Neo4J Conns as PKL file for easy re-construction of NXgraph
def step_two():
    lLimit = 0.85
    uLimit = 1.0

    # Get all parts and connections
    searchStr = "EXISTS(p.uuid)"
    deffer = create_nxgraph_for_clustering(lLimit, uLimit, searchStr, 1)
    graph = deffer.G

    aaa = deffer.finalRelList

    lister = []
    for b in aaa:
        finalDict = {}
        relDict = b.__dict__
        startNodeDict = relDict["_start_node"].__dict__
        endNodeDict = relDict["_end_node"].__dict__
        finalDict = {'start_node_id': startNodeDict["_id"],
                     'end_node_id': endNodeDict["_id"],
                     'rel_key': relDict["_id"],
                     'rel_type': 'similar_to',
                     'prop': relDict["_properties"]
                    }
        lister.append(finalDict)

    with open("D:\\AKS_DATA\\cosine_sim_pt85_1.pkl", "wb") as filer:
        pickle.dump(lister, filer)

    folder = "D:\\AKS_DATA\\Cosine_Similarity"
    files = os.listdir(folder)

    listHist = []
    for i in range(0, len(files)):
        print(files[i])
        csdf = pd.read_hdf(os.path.join(folder, files[i]))
        csdf.drop(["uuid_rows"], axis = 1, inplace = True)
        csdf.to_numpy()
        hist = np.histogram(csdf, bins = [a/10 for a in range(-10, +11, 1)])
        listHist.append(hist)

    histo = np.sum([hist[0] for hist in listHist], axis = 0)
    import matplotlib.pyplot as plt

    font = {"size": 12}
    plt.rc("font", **font)
    plt.hist([a/100 for a in range(-95, 105, 10)], bins=listHist[0][1],
             weights=histo)
    plt.show()