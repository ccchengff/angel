package com.tencent.angel.ml.treemodels.gbdt.histogram;

import com.tencent.angel.ml.math.vector.DenseFloatVector;
import com.tencent.angel.ml.treemodels.gbdt.GBDTController;
import com.tencent.angel.ml.treemodels.param.GBDTParam;
import com.tencent.angel.ml.treemodels.storage.DataStore;
import com.tencent.angel.ml.treemodels.tree.basic.SplitEntry;
import com.tencent.angel.ml.treemodels.tree.regression.GradPair;
import com.tencent.angel.ml.treemodels.tree.regression.RegTNode;
import com.tencent.angel.ml.treemodels.tree.regression.RegTNodeStat;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class SplitFinder {
    private static final Log LOG = LogFactory.getLog(SplitFinder.class);

    private final GBDTParam param;
    private final GBDTController controller;
    private final DataStore trainDataStore;

    public SplitFinder(GBDTParam param, GBDTController controller, DataStore trainDataStore) {
        this.param = param;
        this.controller = controller;
        this.trainDataStore = trainDataStore;
    }


    public SplitEntry findBestSplit(Histogram histogram, RegTNode node,
                                int[] fset) {
        LOG.info(String.format("------To find the best split of node[%d]------",
                node.getNid()));
        SplitEntry splitEntry = new SplitEntry();
        node.calcGain(param);
        for (int i = 0; i < fset.length; i++) {
            int fid = fset[i];
            SplitEntry curSplit = findBestSplitOfOneFeature(fid,
                    histogram.getHistogram(i), node);
            splitEntry.update(curSplit);
        }
        LOG.info(String.format("Local best split of node[%d]: fid[%d], " +
                "fvalue[%f], loss gain[%f]", node.getNid(), splitEntry.getFid(),
                splitEntry.getFvalue(), splitEntry.getLossChg()));
        return splitEntry;
    }

    private SplitEntry findBestSplitOfOneFeature(int fid, DenseFloatVector hist,
                                                 RegTNode node) {
        SplitEntry splitEntry = new SplitEntry();
        // 1. set feature id
        splitEntry.setFid(fid);
        // 2. create the best left grad stats and right grad stats
        GradPair bestLeftStat = new GradPair();
        GradPair bestRightStat = new GradPair();
        // 3. calculate gain of node, create empty grad stats
        GradPair leftStat = new GradPair();
        GradPair rightStat = new GradPair();
        // 4. loop over histogram and find the best
        int numHist = param.numClass == 2 ? 1 : param.numClass;
        for (int k = 0; k < numHist; k++) {
            int offset = k * param.numSplit * 2;
            // 4.1. reset grad stats
            leftStat.update(0, 0);
            rightStat.update(0, 0);
            // 4.2. get node stat of current class
            RegTNodeStat nodeStat = node.getNodeStat(k);
            float nodeGain = nodeStat.getGain();
            float sumGrad = nodeStat.getSumGrad();
            float sumHess = nodeStat.getSumHess();
            // 4.3. loop over split positions, find the best split of current feature
            for (int splitPos = offset; splitPos < offset + param.numSplit - 1; splitPos++) {
                // 4.3.1. get grad and hess
                float grad = hist.get(splitPos);
                float hess = hist.get(splitPos + param.numSplit);
                leftStat.add(grad, hess);
                // 4.3.2. check whether we can split
                if (leftStat.getHess() >= param.minChildWeight) {
                    // right = root - left
                    rightStat.update(sumGrad - leftStat.getGrad(),
                            sumHess - leftStat.getHess());
                    if (rightStat.getHess() >= param.minChildWeight) {
                        // 4.3.3. calculate gain after current split
                        float lossChg = leftStat.calcGain(param) +
                                rightStat.calcGain(param) - nodeGain;
                        // 4.3.4. check whether we should update the split
                        int splitId = splitPos - offset + 1;
                        if (splitEntry.update(lossChg, fid, trainDataStore.getSplit(fid, splitId))) {
                            bestLeftStat.update(leftStat.getGrad(), leftStat.getHess());
                            bestRightStat.update(rightStat.getGrad(), rightStat.getHess());
                        }
                    }
                }
            }
        }
        // 5. set best left and right grad stats
        splitEntry.setLeftGradPair(bestLeftStat);
        splitEntry.setRightGradPair(bestRightStat);
        return splitEntry;
    }

    private SplitEntry findBestSplitOfDiscreteFeature(int fid, DenseFloatVector hist,
                                                      RegTNode node) {
        return null;
    }
}
