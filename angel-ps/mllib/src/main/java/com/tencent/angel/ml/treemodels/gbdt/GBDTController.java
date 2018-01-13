package com.tencent.angel.ml.treemodels.gbdt;

import com.tencent.angel.ml.math.vector.DenseFloatVector;
import com.tencent.angel.ml.metric.GlobalMetrics;
import com.tencent.angel.ml.model.PSModel;
import com.tencent.angel.ml.objective.Loss;
import com.tencent.angel.ml.objective.ObjFunc;
import com.tencent.angel.ml.objective.RegLossObj;
import com.tencent.angel.ml.objective.SoftmaxMultiClassObj;
import com.tencent.angel.ml.treemodels.param.GBDTParam;
import com.tencent.angel.ml.treemodels.storage.DataStore;
import com.tencent.angel.ml.treemodels.tree.regression.GradPair;
import com.tencent.angel.ml.treemodels.tree.regression.RegTNode;
import com.tencent.angel.ml.treemodels.tree.regression.RegTNodeStat;
import com.tencent.angel.ml.treemodels.tree.regression.RegTree;
import com.tencent.angel.worker.task.TaskContext;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public abstract class GBDTController<TrainDataStore extends DataStore, TestDataStore extends DataStore> {
    protected static final Log LOG = LogFactory.getLog(GBDTController.class);

    protected TaskContext taskContext;
    protected GBDTParam param;
    protected GBDTModel model;

    protected TrainDataStore trainDataStore;
    protected TestDataStore validDataStore;

    protected RegTree[] forest;  // all regression trees
    protected int currentTree;  // current tree id
    protected GBDTPhase phase;  // current phase

    protected ObjFunc objFunc;  // objective function
    protected GradPair[] gradPairs;  // gradient pairs of training instances

    protected int[] fset;  // subsample features
    protected Set<Integer> readyNodes;  // nodes ready to process
    protected Set<Integer> activeNodes;  // nodes in process

    // map tree node to instance, each item is an instance id
    // be cautious we need each span of `nodeToIns` in ascending order
    protected int[] nodeToIns;
    protected int[] nodePosStart;
    protected int[] nodePosEnd;

    protected ExecutorService threadPool;

    public GBDTController(TaskContext taskContext, GBDTParam param, GBDTModel model,
                          TrainDataStore trainDataStore, TestDataStore validDataStore) {
        this.taskContext = taskContext;
        this.param = param;
        this.model = model;
        this.trainDataStore = trainDataStore;
        this.validDataStore = validDataStore;
    }

    public void init() {
        forest = new RegTree[param.numTree];
        currentTree = 0;
        phase = GBDTPhase.NEW_TREE;
        //objFunc = new RegLossObj(new Loss.BinaryLogisticLoss());
        gradPairs = new GradPair[trainDataStore.getNumInstances()];
        readyNodes = new TreeSet<>();
        activeNodes = new TreeSet<>();
        nodeToIns = new int[trainDataStore.getNumInstances()];
        Arrays.setAll(nodeToIns, i -> i);
        nodePosStart = new int[param.maxNodeNum];
        nodePosEnd = new int[param.maxNodeNum];
        nodePosEnd[0] = trainDataStore.getNumInstances() - 1;
        threadPool = Executors.newFixedThreadPool(param.numThread);
    }

    protected void clockAllMatrix(Set<String> needFlushMatrices, boolean wait) throws Exception {
        long startTime = System.currentTimeMillis();

        List<Future> clockFutures = new ArrayList<Future>();
        for (Map.Entry<String, PSModel> entry : model.getPSModels().entrySet()) {
            if (needFlushMatrices.contains(entry.getKey())) {
                clockFutures.add(entry.getValue().clock(true));
            } else {
                clockFutures.add(entry.getValue().clock(false));
            }
        }

        if (wait) {
            int size = clockFutures.size();
            for (int i = 0; i < size; i++) {
                clockFutures.get(i).get();
            }
        }

        LOG.info(String.format("clock and flush matrices %s cost %d ms",
                needFlushMatrices, System.currentTimeMillis() - startTime));
    }

    public abstract void createNewTree() throws Exception;

    protected abstract void sampleFeature();

    protected Loss.BinaryLogisticLoss objective = new Loss.BinaryLogisticLoss();
    protected void calGradPairs() throws Exception {
        LOG.info("------Calc grad pairs------");
        long startTime = System.currentTimeMillis();
        int numInstances = trainDataStore.getNumInstances();
        float[] labels = trainDataStore.getLabels();
        float[] preds = trainDataStore.getPreds();
        float[] weights = trainDataStore.getWeights();
        //gradPairs = objFunc.calGrad(labels, preds, weights, gradPairs);
        if (param.numClass == 2) {
            float sumGrad = 0.0f;
            float sumHess = 0.0f;
            for (int insId = 0; insId < numInstances; insId++) {
                float prob = objective.transPred(preds[insId]);
                float grad = objective.firOrderGrad(prob, labels[insId]) * weights[insId];
                float hess = objective.secOrderGrad(prob, labels[insId]) * weights[insId];
                gradPairs[insId] = new GradPair(grad, hess);
                sumGrad += grad;
                sumHess += hess;
            }
            RegTNodeStat rootStat = new RegTNodeStat(sumGrad, sumHess);
            forest[currentTree].getRoot().setNodeStat(rootStat);
        } else {
            float[] sumGrad = new float[param.numClass];
            float[] sumHess = new float[param.numClass];
            // sum up
            for (int i = 0; i < param.numClass; i++) {
                RegTNodeStat rootStat = new RegTNodeStat(sumGrad[i], sumHess[i]);
                forest[currentTree].getRoot().setNodeStat(i, rootStat);
            }
        }
        if (taskContext.getTaskIndex() == 0) {
            updateNodeStat(0);
        }
        model.getPSModel(GBDTModel.NODE_GRAD_MAT()).clock(true).get();
        LOG.info(String.format("Calc grad pair cost %d ms", System.currentTimeMillis() - startTime));
    }

    public void chooseActive() {
        LOG.info("------Choose active nodes------");
        LOG.info("Ready nodes: " + readyNodes.toString());
        for (Integer nid : readyNodes) {
            if (2 * nid + 1 >= param.maxNodeNum) {
                setNodeToLeaf(nid);
            } else {
                if (nodePosEnd[nid] - nodePosStart[nid] < 1000) {
                    setNodeToLeaf(nid);
                } else {
                    activeNodes.add(nid);
                }
            }
        }
        readyNodes.clear();
        if (activeNodes.size() > 0) {
            LOG.info("Ready nodes: " + activeNodes.toString());
            this.phase = GBDTPhase.RUN_ACTIVE;
        } else {
            LOG.info("No active nodes");
            this.phase = GBDTPhase.FINISH_TREE;
        }
    }

    public void runActiveNodes() {
        LOG.info("------Run active node------");
        long startTime = System.currentTimeMillis();

    }

    public abstract void findSplit();

    public abstract void afterSplit();

    public void finishCurrentTree(GlobalMetrics globalMetrics) {

    }

    protected void setNodeToLeaf(int nid) {
        activeNodes.remove(nid);
        forest[currentTree].getNode(nid).chgToLeaf();
        forest[currentTree].getNode(nid).calcWeight(param);
    }

    protected void updateNodeStat(int nid) {
        DenseFloatVector vec;
        RegTNode node = forest[currentTree].getNode(nid);
        if (param.numClass == 2) {
            vec = new DenseFloatVector(param.maxNodeNum * 2);
            RegTNodeStat nodeStat = node.getNodeStat();
            vec.set(nid, nodeStat.getSumGrad());
            vec.set(nid + param.maxNodeNum, nodeStat.getSumHess());
        } else {
            vec = new DenseFloatVector(param.numClass * param.maxNodeNum * 2);
            for (int i = 0; i < param.numClass; i++) {
                RegTNodeStat nodeStat = node.getNodeStat(i);
                vec.set(nid * param.numClass + i, nodeStat.getSumGrad());
                vec.set((nid + param.maxNodeNum) * param.numClass + i, nodeStat.getSumHess());
            }
        }
        PSModel nodeStatsModel = model.getPSModel(GBDTModel.NODE_GRAD_MAT());
        nodeStatsModel.increment(currentTree, vec);
    }
}
