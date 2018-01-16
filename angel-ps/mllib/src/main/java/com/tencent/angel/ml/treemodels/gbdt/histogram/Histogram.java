package com.tencent.angel.ml.treemodels.gbdt.histogram;

import com.tencent.angel.ml.math.vector.DenseFloatVector;

/**
 * Histogram for GBDT nodes
 */
public class Histogram {
    private int numFeature;
    private int numSplit;
    private int numClass;

    private DenseFloatVector[] histograms;

    public Histogram(int numFeature, int numSplit, int numClass) {
        this.numFeature = numFeature;
        this.numSplit = numSplit;
        this.numClass = numClass;
        histograms = new DenseFloatVector[numFeature];
    }

    public DenseFloatVector getHistogram(int index) {
        return histograms[index];
    }

    public void set(int index, DenseFloatVector hist) {
        this.histograms[index] = hist;
    }

    public Histogram subtract(Histogram other) {
        Histogram res = new Histogram(numFeature, numSplit, numClass);
        for (int i = 0; i < histograms.length; i++) {
            int size = histograms[i].getDimension();
            float[] resValue = new float[size];
            float[] aValue = this.histograms[i].getValues();
            float[] bValue = other.histograms[i].getValues();
            for (int j = 0; j < size; j++) {
                resValue[j] = aValue[j] - bValue[j];
            }
            res.histograms[i] = new DenseFloatVector(size, resValue);
        }
        return res;
    }
}
