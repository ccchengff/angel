package com.tencent.angel.ml.treemodels.tree.regression;

import com.tencent.angel.ml.treemodels.param.GBDTParam;
import com.tencent.angel.ml.treemodels.tree.basic.SplitEntry;
import com.tencent.angel.ml.treemodels.tree.basic.TNode;


public class RegTNode extends TNode<RegTNodeStat> {
    private SplitEntry splitEntry;  // split entry of current node

    public RegTNode(int nid, TNode parent, int numClass) {
        this(nid, parent, null, null, numClass);
    }

    public RegTNode(int nid, TNode parent, TNode left, TNode right, int numClass) {
        super(nid, parent, left, right);
        this.nodeStats = new RegTNodeStat[numClass > 2 ? 1 : numClass];
    }

    public float[] calcWeight(GBDTParam param) {
        float[] nodeWeights = new float[nodeStats.length];
        for (int i = 0; i < nodeStats.length; i++) {
            nodeWeights[i] = nodeStats[i].calcWeight(param);
        }
        return nodeWeights;
    }

    public SplitEntry getSplitEntry() {
        return this.splitEntry;
    }

    public void setSplitEntry(SplitEntry splitEntry) {
        this.splitEntry = splitEntry;
    }
}
