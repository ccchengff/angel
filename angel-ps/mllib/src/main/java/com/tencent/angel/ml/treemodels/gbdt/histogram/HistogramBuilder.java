package com.tencent.angel.ml.treemodels.gbdt.histogram;

import com.tencent.angel.exception.AngelException;
import com.tencent.angel.ml.math.vector.DenseFloatVector;
import com.tencent.angel.ml.treemodels.gbdt.GBDTController;
import com.tencent.angel.ml.treemodels.gbdt.GBDTModel;
import com.tencent.angel.ml.treemodels.gbdt.ParallelMode;
import com.tencent.angel.ml.treemodels.gbdt.dp.DPGBDTController;
import com.tencent.angel.ml.treemodels.gbdt.fp.FPGBDTController;
import com.tencent.angel.ml.treemodels.param.GBDTParam;
import com.tencent.angel.ml.treemodels.storage.DPDataStore;
import com.tencent.angel.ml.treemodels.storage.FPDataStore;
import com.tencent.angel.ml.treemodels.tree.regression.RegTNodeStat;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class HistogramBuilder {
    private static final Log LOG = LogFactory.getLog(HistogramBuilder.class);

    private final GBDTParam param;
    private final GBDTModel model;
    private final GBDTController controller;

    private BuilderThread[] threads;
    private ExecutorService threadPool;

    public HistogramBuilder(final GBDTParam param, final GBDTModel model,
                            final GBDTController controller) {
        this.param = param;
        this.model = model;
        this.controller = controller;
        this.threads = new BuilderThread[param.numThread];
        for (int i = 0; i < param.numThread; i++) {
            this.threads[i] = new BuilderThread(i);
        }
        this.threadPool = Executors.newFixedThreadPool(param.numThread);
    }

    public Histogram buildHistogram(int nid) throws Exception {
        LOG.info(String.format("Build histogram of node[%d], parallelism[%d]",
                nid, param.numThread));
        long startTime = System.currentTimeMillis();
        Histogram res = null;
        // 1. check whether can build from subtraction
        boolean canSubtract = false;
        if (nid != 0) {
            int siblingNid = (nid & 1) == 0 ? nid - 1 : nid + 1;
            Histogram siblingHist = controller.getHistogram(siblingNid);
            if (siblingHist != null) {
                canSubtract = true;
                int parentNid = (nid - 1) >> 1;
                Histogram parentHist = controller.getHistogram(parentNid);
                res = parentHist.subtract(siblingHist);
            }
        }
        // 2. build from grad pair in parallel
        if (!canSubtract) {
            res = new Histogram(controller.getFset().length,
                    param.numSplit, param.numClass);
            List<Future<Void>> futures = new ArrayList<>(threads.length);
            for (BuilderThread thread : threads) {
                thread.nid = nid;
                thread.histogram = res;
                futures.add(threadPool.submit(thread));
            }
            for (Future<Void> future : futures) {
                future.get();
            }
        }
        LOG.info(String.format("Build histogram of node[%d] cost %d ms",
                nid, System.currentTimeMillis() - startTime));
        return res;
    }

    private class BuilderThread implements Callable<Void> {
        private final int threadId;
        private int nid;
        private Histogram histogram;
        private Histogram selfHist;  // for data-parallel mode

        private void dpBuild(DPGBDTController controller) {
            DPDataStore trainDataStore = controller.getTrainDataStore();
            int[] fset = controller.getFset();
            int[] nodeToIns = controller.getNodeToIns();
            int nodeStart = controller.getNodePosStart(nid);
            int nodeEnd = controller.getNodePosEnd(nid);
            int insPerThread = (nodeEnd - nodeStart + 1) / param.numThread;
            int from = threadId * insPerThread;
            int to = threadId + 1 == param.numThread
                    ? nodeEnd + 1 : from + insPerThread;
            float[] insGrad = controller.getInsGrad();
            float[] insHess = controller.getInsHess();
            RegTNodeStat nodeStat = controller.getLastTree().getNode(nid).getNodeStat();
            if (param.numClass == 2) {
                dpBinaryClassBuild(trainDataStore, fset, insGrad, insHess,
                        nodeStat, nodeToIns, from, to);
            } else {
                throw new AngelException("Multi-class not implemented");
            }
            synchronized (this) {
                histogram.plusBy(selfHist);
            }
        }

        private void dpBinaryClassBuild(DPDataStore trainDataStore, int[] fset,
                                        float[] insGrad, float[] insHess, RegTNodeStat nodeStat,
                                        int[] nodeToIns, int from, int to) {
            // 1. allocate histogram
            selfHist = new Histogram(fset.length, param.numSplit, param.numClass);
            selfHist.alloc();
            // 2. for each instance, loop non-zero features,
            // add to histogram, and record the gradients taken
            float gradTaken = 0, hessTaken = 0;
            for (int i = from; i < to; i++) {
                // 2.1. get instance
                int insId = nodeToIns[i];
                int[] indices = trainDataStore.getInsIndices(insId);
                int[] bins = trainDataStore.getInsBins(insId);
                int nnz = indices.length;
                // 2.2. loop non-zero instances
                for (int j = 0; j < nnz; j++) {
                    int fid = indices[j];
                    // 2.3. add to histogram
                    DenseFloatVector hist;
                    if (fset.length == param.numFeature) {
                        hist = histogram.getHistogram(fid);
                    } else {
                        int index = Arrays.binarySearch(fset, fid);
                        if (index < 0) {
                            continue;
                        }
                        hist = histogram.getHistogram(index);
                    }
                    int binId = bins[j];
                    int gradId = binId;
                    int hessId = gradId + param.numSplit;
                    hist.set(gradId, hist.get(gradId) + insGrad[insId]);
                    hist.set(hessId, hist.get(hessId) + insHess[insId]);
                }
                // 2.3. record gradients taken
                gradTaken += insGrad[insId];
                hessTaken += insHess[insId];
            }
            // 3. subtract gradients taken from zero bucket
            for (int i = 0; i < fset.length; i++) {
                DenseFloatVector hist = histogram.getHistogram(i);
                int zeroId = trainDataStore.getZeroBin(fset[i]);
                int gradId = zeroId;
                int hessId = gradId + param.numSplit;
                hist.set(gradId, hist.get(gradId) - gradTaken);
                hist.set(hessId, hist.get(hessId) - hessTaken);
            }
        }

        private void fpBuid(FPGBDTController controller) {
            FPDataStore trainDataStore = controller.getTrainDataStore();
            int[] fset = controller.getFset();
            int featPerThread = fset.length / param.numThread;
            int from = threadId * featPerThread;
            int to = threadId + 1 == param.numThread
                    ? fset.length : from + featPerThread;
            float[] insGrad = controller.getInsGrad();
            float[] insHess = controller.getInsHess();
            RegTNodeStat nodeStat = controller.getLastTree().getNode(nid).getNodeStat();
            int[] insToNode = controller.getInsToNode();
            int nodeStart = controller.getNodePosStart(nid);
            int nodeEnd = controller.getNodePosEnd(nid);
            if (param.numClass == 2) {
                fpBinaryClassBuild(trainDataStore, fset, insGrad, insHess, nodeStat,
                        insToNode, nodeStart, nodeEnd, from, to);
            } else {
                throw new AngelException("Multi-class not implemented");
            }
        }

        private void fpBinaryClassBuild(FPDataStore trainDataStore, int[] fset,
                                        float[] insGrad, float[] insHess, RegTNodeStat nodeStat,
                                        int[] insToNode, int nodeStart, int nodeEnd,
                                        int from, int to) {
            float sumGrad = nodeStat.getSumGrad();
            float sumHess = nodeStat.getSumHess();
            for (int i = from; i < to; i++) {
                // 1. get feature row
                int fid = fset[i];
                int[] indices = trainDataStore.getFeatIndices(fid);
                int[] bins = trainDataStore.getFeatBins(fid);
                int nnz = indices.length;
                // 2. allocate histogram
                DenseFloatVector hist = new DenseFloatVector(param.numSplit * 2);
                // 3. loop non-zero instances, add to histogram, and record the gradients taken
                float gradTaken = 0, hessTaken = 0;
                for (int j = 0; j < nnz; j++) {
                    int insId = indices[j];
                    int insPos = insToNode[insId];
                    if (nodeStart <= insPos && insPos <= nodeEnd) {
                        int binId = bins[j];
                        int gradId = binId;
                        int hessId = gradId + param.numSplit;
                        hist.set(gradId, hist.get(gradId) + insGrad[insId]);
                        hist.set(hessId, hist.get(hessId) + insHess[insId]);
                        gradTaken += insGrad[insId];
                        hessTaken += insHess[insId];
                    }
                }
                // 4. add remaining grad and hess to zero bin
                int zeroId = trainDataStore.getZeroBin(fid);
                int gradId = zeroId;
                int hessId = gradId + param.numSplit;
                hist.set(gradId, sumGrad - gradTaken);
                hist.set(hessId, sumHess - hessTaken);
                // 5. put to result
                histogram.set(i, hist);
            }
        }

        private BuilderThread(int threadId) {
            this.threadId = threadId;
        }

        @Override
        public Void call() throws Exception {
            switch (controller.getParallelMode()) {
                case ParallelMode.DATA_PARALLEL:
                    dpBuild((DPGBDTController) controller);
                    break;
                case ParallelMode.FEATURE_PARALLEL:
                    fpBuid((FPGBDTController) controller);
                    break;
                default:
                    throw new AngelException(
                            "Unrecognizable parallel mode: "
                                    + controller.getParallelMode());
            }
            return null;
        }
    }

}
