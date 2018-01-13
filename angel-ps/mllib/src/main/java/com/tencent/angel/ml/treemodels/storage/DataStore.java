package com.tencent.angel.ml.treemodels.storage;

import com.tencent.angel.ml.FPGBDT.psf.QSketchesGetResult;
import com.tencent.angel.ml.FPGBDT.psf.QSketchesGetFunc;
import com.tencent.angel.ml.FPGBDT.psf.QSketchesMergeFunc;
import com.tencent.angel.ml.feature.LabeledData;
import com.tencent.angel.ml.model.PSModel;
import com.tencent.angel.ml.treemodels.gbdt.GBDTModel;
import com.tencent.angel.ml.treemodels.param.TreeParam;
import com.tencent.angel.ml.FPGBDT.algo.QuantileSketch.HeapQuantileSketch;
import com.tencent.angel.worker.storage.DataBlock;
import com.tencent.angel.worker.task.TaskContext;

import java.util.Arrays;

public abstract class DataStore {
    TaskContext taskContext;
    // param
    protected TreeParam param;
    protected int numInstances;
    protected int numFeatures;
    // info of instances
    protected float[] labels;
    protected float[] preds;
    protected float[] weights;
    // statistic of instances and features
    protected int[] workerNumIns;  // instance num of each worker
    protected int[] nnzLocal;  // local nnz of each feature
    protected int[] nnzGlobal;  // global nnz of each feature
    // candidate splits
    protected float[][] splits;
    protected int[] zeroBins;

    public DataStore(TaskContext taskContext, TreeParam param) {
        this.taskContext = taskContext;
        this.param = param;
        numFeatures = param.numFeature;
    }

    public abstract void init(DataBlock<LabeledData> dataStorage, GBDTModel model) throws Exception;

    public int getNumInstances() {
        return numInstances;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public float getLabel(int index) {
        return labels[index];
    }

    public float[] getLabels() {
        return labels;
    }

    public float[] getPreds(int index) {
        return preds;
    }

    public float[] getPreds() {
        return preds;
    }

    public float getWeight(int index) {
        return weights[index];
    }

    public float[] getWeights() {
        return weights;
    }

    public float[][] getSplits() {
        return splits;
    }

    public int[] getZeroBins() {
        return zeroBins;
    }

    public void setPred(int index, float pred) {
        this.preds[index] = pred;
    }

    public void setPreds(float[] preds) {
        this.preds = preds;
    }

    public void setWeight(int index, float weight) {
        this.weights[index] = weight;
    }

    public void setWeights(float[] weights) {
        this.weights = weights;
    }

    public void setSplits(float[][] splits) {
        this.splits = splits;
    }

    public void setZeroBins(int[] zeroBins) {
        this.zeroBins = zeroBins;
    }

    public static int findZeroBin(float[] arr) {
        int size = arr.length;
        int zeroIdx;
        if (arr[0] > 0.0f) {
            zeroIdx = 0;
        } else if (arr[size - 1] < 0.0f) {
            zeroIdx = size - 1;
        } else {
            zeroIdx = 0;
            while (zeroIdx < size - 1 && arr[zeroIdx + 1] < 0.0f) {
                zeroIdx++;
            }
        }
        return zeroIdx;
    }

    public static int indexOf(float x, float[] arr, int zeroIdx) {
        int size = arr.length;
        int left = zeroIdx;
        int right = zeroIdx;
        if (x < 0.0f) {
            left = 0;
        } else {
            right = size - 1;
        }
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (arr[mid] <= x) {
                if (mid + 1 == size || arr[mid + 1] > x) {
                    return mid;
                } else {
                    left = mid + 1;
                }
            } else {
                right = mid - 1;
            }
        }
        return zeroIdx;
    }

    public int indexOf(float x, int fid) {
        return indexOf(x, splits[fid], zeroBins[fid]);
    }

    /**
     * Push local quantile sketches to PS, merge on PS and pull quantiles
     *
     * In order not to allocate a large matrix on PS,
     * we push a batch of features in one time
     *
     * @param sketches
     * @param estimateNs
     * @return
     */
    protected float[][] mergeSketchAndPullQuantiles(HeapQuantileSketch[] sketches, int[] estimateNs,
                                                    GBDTModel model) throws Exception {
        PSModel sketchModel = model.getPSModel(GBDTModel.SKETCH_MAT());
        int matrixId = sketchModel.getMatrixId();

        int numFeature = param.numFeature;
        int numSplit = param.numSplit;
        int numWorker = param.numWorker;

        float[][] quantiles = new float[numFeature][numSplit];

        int fid = 0;
        int batchSize = 1024;
        int[] rowIndexes = new int[batchSize];
        HeapQuantileSketch[] batchSketches = new HeapQuantileSketch[batchSize];
        long[] batchEstimateNs = new long[batchSize];
        Arrays.setAll(rowIndexes, i -> i);
        while (fid < numFeature) {
            if (fid + batchSize > numFeature) {
                batchSize = numFeature - fid;
                Arrays.setAll(rowIndexes, i -> i);
                batchSketches = new HeapQuantileSketch[batchSize];
                batchEstimateNs = new long[batchSize];
            }
            // 1. set up a batch
            for (int i = 0; i < batchSize; i++) {
                batchSketches[i] = sketches[fid + i];
                batchEstimateNs[i] = estimateNs[fid + i];
            }
            // 2. push to PS and merge on PS
            sketchModel.update(new QSketchesMergeFunc(matrixId, true, rowIndexes,
                    numWorker, numSplit, batchSketches, batchEstimateNs)).get();
            model.sync();
            // 3. pull quantiles from PS
            QSketchesGetResult getResult = (QSketchesGetResult) sketchModel.get(
                    new QSketchesGetFunc(matrixId, rowIndexes, numWorker, numSplit));
            for (int i = 0; i < batchSize; i++) {
                int trueFid = fid + i;
                quantiles[trueFid] = getResult.getQuantiles(i);
            }
            model.sync();
            fid += batchSize;
        }
        return quantiles;
    }

}
