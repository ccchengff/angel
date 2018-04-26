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
            if (param.numThread > 1) {
              List<Future<Void>> futures = new ArrayList<>(threads.length);
              for (BuilderThread thread : threads) {
                thread.nid = nid;
                thread.histogram = res;
                futures.add(threadPool.submit(thread));
              }
              for (Future<Void> future : futures) {
                future.get();
              }
            } else {
              threads[0].nid = nid;
              threads[0].histogram = res;
              threads[0].call();
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
            if (param.numClass == 2) {
                RegTNodeStat nodeStat = controller.getLastTree().getNode(nid).getNodeStat();
                dpBinaryClassBuild(trainDataStore, fset, insGrad, insHess,
                        nodeStat, nodeToIns, from, to);
            } else {
                RegTNodeStat[] nodeStats = controller.getLastTree().getNode(nid).getNodeStats();
                dpMultiClassBuild(trainDataStore, fset, insGrad, insHess,
                        nodeStats, nodeToIns, from, to);
            }
            synchronized (BuilderThread.class) {
                if (histogram.getHistogram(0) == null) {
                    histogram.alloc();
                }
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
            // add to histogram, and record the sum of grad & hess
            float sumGrad = 0, sumHess = 0;
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
                        hist = selfHist.getHistogram(fid);
                    } else {
                        int index = Arrays.binarySearch(fset, fid);
                        if (index < 0) {
                            continue;
                        }
                        hist = selfHist.getHistogram(index);
                    }
                    int binId = bins[j];
                    int gradId = binId;
                    int hessId = gradId + param.numSplit;
                    hist.set(gradId, hist.get(gradId) + insGrad[insId]);
                    hist.set(hessId, hist.get(hessId) + insHess[insId]);
                    // 2.4. add the reverse to zero bin
                    int zeroId = trainDataStore.getZeroBin(fid);
                    int gradZeroId = zeroId;
                    int hessZeroId = gradZeroId + param.numSplit;
                    hist.set(gradZeroId, hist.get(gradZeroId) - insGrad[insId]);
                    hist.set(hessZeroId, hist.get(hessZeroId) - insHess[insId]);
                }
                // 2.5. sum up
                // sumGrad & sumHess in nodeStat is calculated
                // from all instances, not instances on current worker
                // so we need to sum up grad & hess in this loop
                sumGrad += insGrad[insId];
                sumHess += insHess[insId];
            }
            // 3. add to zero bin
            for (int i = 0; i < fset.length; i++) {
                DenseFloatVector hist = selfHist.getHistogram(i);
                int zeroId = trainDataStore.getZeroBin(fset[i]);
                int gradZeroId = zeroId;
                int hessZeroId = gradZeroId + param.numSplit;
                hist.set(gradZeroId, hist.get(gradZeroId) + sumGrad);
                hist.set(hessZeroId, hist.get(hessZeroId) + sumHess);
            }
        }

        private void dpMultiClassBuild(DPDataStore trainDataStore, int[] fset,
                                       float[] insGrad, float[] insHess, RegTNodeStat[] nodeStats,
                                       int[] nodeToIns, int from, int to) {
            // 1. allocate histogram
            selfHist = new Histogram(fset.length, param.numSplit, param.numClass);
            selfHist.alloc();
            // 2. for each instance, loop non-zero features,
            // add to histogram, and record the gradients taken
            float[] sumGrad = new float[param.numClass];
            float[] sumHess = new float[param.numClass];
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
                        hist = selfHist.getHistogram(fid);
                    } else {
                        int index = Arrays.binarySearch(fset, fid);
                        if (index < 0) {
                            continue;
                        }
                        hist = selfHist.getHistogram(index);
                    }
                    int binId = bins[j];
                    int zeroId = trainDataStore.getZeroBin(fid);
                    //int gradId = binId;
                    //int hessId = gradId + param.numSplit;
                    for (int k = 0; k < param.numClass; k++) {
                        int gradId = k * param.numSplit * 2 + binId;
                        int hessId = gradId + param.numSplit;
                        float grad = insGrad[insId + param.numClass + k];
                        float hess = insHess[insId + param.numClass + k];
                        hist.set(gradId, hist.get(gradId) + grad);
                        hist.set(hessId, hist.get(hessId) + hess);
                        // 2.4. add the reverse to zero bin
                        int gradZeroId = k * param.numSplit * 2 + zeroId;
                        int hessZeroId = gradZeroId + param.numSplit;
                        hist.set(gradZeroId, hist.get(gradZeroId) - grad);
                        hist.set(hessZeroId, hist.get(hessZeroId) - hess);
                    }
                }
                // 2.5. sum up
                // sumGrad & sumHess in nodeStat is calculated
                // from all instances, not instances on current worker
                // so we need to sum up grad & hess in this loop
                for (int k = 0; k < param.numClass; k++) {
                    sumGrad[k] += insGrad[insId + param.numClass + k];
                    sumHess[k] += insHess[insId + param.numClass + k];
                }
            }
            // 3. add to zero bin
            for (int i = 0; i < fset.length; i++) {
                DenseFloatVector hist = selfHist.getHistogram(i);
                int zeroId = trainDataStore.getZeroBin(fset[i]);
                for (int k = 0; k < param.numClass; k++) {
                    int gradZeroId = k * param.numSplit * 2 + zeroId;
                    int hessZeroId = gradZeroId + param.numSplit;
                    hist.set(gradZeroId, hist.get(gradZeroId) + sumGrad[k]);
                    hist.set(hessZeroId, hist.get(hessZeroId) + sumHess[k]);
                }
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
            int[] insToNode = controller.getInsToNode();
            int nodeStart = controller.getNodePosStart(nid);
            int nodeEnd = controller.getNodePosEnd(nid);
            if (param.numClass == 2) {
                RegTNodeStat nodeStat = controller.getLastTree().getNode(nid).getNodeStat();
                fpBinaryClassBuild(trainDataStore, fset, insGrad, insHess, nodeStat,
                        insToNode, nodeStart, nodeEnd, from, to);
            } else {
                RegTNodeStat[] nodeStats = controller.getLastTree().getNode(nid).getNodeStats();
                fpMultiClassBuild(trainDataStore, fset, insGrad, insHess, nodeStats,
                        insToNode, nodeStart, nodeEnd, from, to);
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
                histogram.alloc(i);
                DenseFloatVector hist = histogram.getHistogram(i);
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
                int gradZeroId = zeroId;
                int hessZeroId = gradZeroId + param.numSplit;
                hist.set(gradZeroId, sumGrad - gradTaken);
                hist.set(hessZeroId, sumHess - hessTaken);
            }
        }

        private void fpMultiClassBuild(FPDataStore trainDataStore, int[] fset,
                                        float[] insGrad, float[] insHess, RegTNodeStat[] nodeStats,
                                        int[] insToNode, int nodeStart, int nodeEnd,
                                        int from, int to) {
            float sumGrad[] = new float[param.numClass];
            float sumHess[] = new float[param.numClass];
            for (int k = 0; k < param.numClass; k++) {
                sumGrad[k] = nodeStats[k].getSumGrad();
                sumHess[k] = nodeStats[k].getSumHess();
            }
            for (int i = from; i < to; i++) {
                // 1. get feature row
                int fid = fset[i];
                int[] indices = trainDataStore.getFeatIndices(fid);
                int[] bins = trainDataStore.getFeatBins(fid);
                int nnz = indices.length;
                // 2. allocate histogram
                histogram.alloc(i);
                DenseFloatVector hist = histogram.getHistogram(i);
                //DenseFloatVector hist = new DenseFloatVector(param.numClass * param.numSplit * 2);
                // 3. loop non-zero instances, add to histogram, and record the gradients taken
                float[] gradTaken = new float[param.numClass];
                float[] hessTaken = new float[param.numClass];
                for (int j = 0; j < nnz; j++) {
                    int insId = indices[j];
                    int insPos = insToNode[insId];
                    if (nodeStart <= insPos && insPos <= nodeEnd) {
                        int binId = bins[j];
                        for (int k = 0; k < param.numClass; k++) {
                            int gradId = k * param.numSplit * 2 + binId;
                            int hessId = gradId + param.numSplit;
                            float grad = insGrad[insId + param.numClass + k];
                            float hess = insHess[insId + param.numClass + k];
                            hist.set(gradId, hist.get(gradId) + grad);
                            hist.set(hessId, hist.get(hessId) + hess);
                            gradTaken[k] += grad;
                            hessTaken[k] += hess;
                        }
                    }
                }
                // 4. add remaining grad and hess to zero bin
                int zeroId = trainDataStore.getZeroBin(fid);
                for (int k = 0; k < param.numClass; k++) {
                    int gradZeroId = k * param.numSplit * 2 + zeroId;
                    int hessZeroId = gradZeroId + param.numSplit;
                    hist.set(gradZeroId, sumGrad[k] - gradTaken[k]);
                    hist.set(hessZeroId, sumHess[k] - hessTaken[k]);
                }
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
