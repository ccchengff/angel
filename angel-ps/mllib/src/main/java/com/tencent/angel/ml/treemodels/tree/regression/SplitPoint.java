package com.tencent.angel.ml.treemodels.tree.regression;

import com.tencent.angel.ml.treemodels.tree.basic.SplitEntry;

public class SplitPoint extends SplitEntry {
    private int fid;  // feature index used to split
    private float fvalue;  // feature value used to split
    private float lossChg;  // loss change after split this node
    private GradPair leftGradPair;  // grad pair of left child
    private GradPair rightGradPair; // grad pair of right child

    public SplitPoint(int fid, float fvalue, float lossChg) {
        this.fid = fid;
        this.fvalue = fvalue;
        this.lossChg = lossChg;
    }

    public SplitPoint() {
        this(-1, 0.0f, 0.0f);
    }

    @Override
    public boolean isEmpty() {
        return fid == -1;
    }

    @Override
    public int flowTo(float value) {
        return value <= fvalue ? 0 : 1;
    }

    @Override
    public int defaultTo() {
        return fvalue <= 0 ? 0 : 1;
    }

    @Override
    public boolean needReplace(float newLossChg, int splitFeature) {
        if (this.fid <= splitFeature) {
            return newLossChg > this.lossChg;
        } else {
            return !(this.lossChg > newLossChg);
        }
    }

    /*@Override
    public SplitType splitType() {
        return SplitType.SPLIT_POINT;
    }*/

    public boolean update(SplitPoint e) {
        if (this.needReplace(e.lossChg, e.fid)) {
            this.lossChg = e.lossChg;
            this.fid = e.fid;
            this.fvalue = e.fvalue;
            this.leftGradPair = e.leftGradPair;
            this.rightGradPair = e.rightGradPair;
            return true;
        } else {
            return false;
        }
    }

    public boolean update(float newLossChg, int splitFeature, float newSplitValue) {
        if (this.needReplace(newLossChg, splitFeature)) {
            this.lossChg = newLossChg;
            this.fid = splitFeature;
            this.fvalue = newSplitValue;
            return true;
        } else {
            return false;
        }
    }

    @Override
    public int getFid() {
        return fid;
    }

    public float getFvalue() {
        return fvalue;
    }

    public float getLossChg() {
        return lossChg;
    }

    public GradPair getLeftGradPair() {
        return leftGradPair;
    }

    public GradPair getRightGradPair() {
        return rightGradPair;
    }

    public void setFid(int fid) {
        this.fid = fid;
    }

    public void setFvalue(float fvalue) {
        this.fvalue = fvalue;
    }

    public void setLossChg(float lossChg) {
        this.lossChg = lossChg;
    }

    public void setLeftGradPair(GradPair leftGradPair) {
        this.leftGradPair = leftGradPair;
    }

    public void setRightGradPair(GradPair rightGradPair) {
        this.rightGradPair = rightGradPair;
    }

    @Override
    public String toString() {
        return String.format("fid[%d], fvalue[%f], lossChg[%f], "
                        + "leftGradPair[%f, %f], rightGradPair[%f, %f]",
                fid, fvalue, lossChg,
                leftGradPair.getGrad(), leftGradPair.getHess(),
                rightGradPair.getGrad(), rightGradPair.getHess());
    }
}
