package com.tencent.angel.ml.treemodels.storage;

import com.tencent.angel.exception.AngelException;
import com.tencent.angel.ml.treemodels.sketch.psf.QSketchesGetResult;
import com.tencent.angel.ml.treemodels.sketch.psf.QSketchesGetFunc;
import com.tencent.angel.ml.treemodels.sketch.psf.QSketchesMergeFunc;
import com.tencent.angel.ml.feature.LabeledData;
import com.tencent.angel.ml.model.PSModel;
import com.tencent.angel.ml.treemodels.gbdt.GBDTModel;
import com.tencent.angel.ml.treemodels.param.TreeParam;
import com.tencent.angel.ml.treemodels.sketch.HeapQuantileSketch;
import com.tencent.angel.worker.storage.DataBlock;
import com.tencent.angel.worker.task.TaskContext;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;

public abstract class DataStore {
    private static final Log LOG = LogFactory.getLog(DataStore.class);

    protected final TaskContext taskContext;
    // param
    protected final TreeParam param;
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

    public abstract float get(int rowId, int colId, float defaultValue);

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

    public float getPred(int index) {
        return preds[index];
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

    public float getSplit(int fid, int splitId) {
        return splits[fid][splitId];
    }

    public float[] getSplits(int fid) {
        return splits[fid];
    }

    public float[][] getSplits() {
        return splits;
    }

    public int getZeroBin(int fid) {
        return zeroBins[fid];
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

    /**
     * Find the last split point that is not larger than 0, and return its index
     *
     * @param arr
     * @return
     */
    public static int findZeroBin(float[] arr) {
        int size = arr.length;
        int zeroIdx;
        if (arr[0] > 0.0f) {
            zeroIdx = 0;
        } else if (arr[size - 1] <= 0.0f) {
            zeroIdx = size - 1;
        } else {
            zeroIdx = 0;
            while (zeroIdx < size - 1 && arr[zeroIdx + 1] <= 0.0f) {
                zeroIdx++;
            }
        }
        return zeroIdx;
    }

    /**
     * Find the last split point that is not larger than x, and return its index
     *
     * @param x
     * @param arr
     * @param zeroIdx
     * @return
     */
    public static int indexOf(float x, float[] arr, int zeroIdx) {
        int size = arr.length;
        int left = zeroIdx;
        int right = zeroIdx;
        if (x < 0.0f) {
            left = 0;
        } else {
            right = size - 1;
        }
        while (right - left > 1) {
            int mid = left + ((right - left) >> 1);
            if (arr[mid] <= x) {
                if (mid + 1 == size || arr[mid + 1] > x) {
                    return mid;
                } else {
                    left = mid + 1;
                }
            } else {
                right = mid;
            }
        }
        return left;
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
                                                    final GBDTModel model) throws Exception {
        int batchSize = 1024;
        PSModel sketchModel = model.getPSModel(GBDTModel.SKETCH_MAT());
        int matrixId = sketchModel.getMatrixId();

        int numFeature = param.numFeature;
        int numSplit = param.numSplit;
        int numWorker = param.numWorker;

        float[][] quantiles = new float[numFeature][numSplit];

        int fid = 0;
        int[] rowIndexes = new int[batchSize];
        Arrays.setAll(rowIndexes, i -> i);
        HeapQuantileSketch[] batchSketches = new HeapQuantileSketch[batchSize];
        long[] batchEstimateNs = new long[batchSize];
        while (fid < numFeature) {
            if (fid + batchSize > numFeature) {
                batchSize = numFeature - fid;
                rowIndexes = new int[batchSize];
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

    /**
     * Ensure labels are in {0, 1, ..., #class - 1}
     *
     * @param numClass
     */
    public void ensureLabel(int numClass) {
        if (numClass == 2) {
            for (int i = 0; i < labels.length; i++) {
                if (labels[i] != 1) {
                    labels[i] = 0;
                }
            }
        } else {
            Set<Integer> appearSet = new TreeSet<>();
            for (float label : labels) {
                appearSet.add((int) label);
            }
            Integer[] appear = new Integer[appearSet.size()];
            appearSet.toArray(appear);
            Arrays.sort(appear);
            // check #label
            if (appear.length > numClass) {
                throw new AngelException(String.format(
                        "There are %d classes, more than %d",
                        appear.length, numClass));
            } else if (appear.length < numClass) {
                LOG.warn(String.format("There are %d classes, less than %d. " +
                        "We suggest to check the data or hyper-parameters",
                        appear.length, numClass));
            }
            // check span
            int minLabel = appear[0], maxLabel = appear[appear.length - 1];
            if (maxLabel - minLabel + 1 > numClass) {
                throw new AngelException(String.format(
                        "The span of labels is [%d, %d], which is discontiguous",
                        minLabel, maxLabel));
            } else if (minLabel == 1 || maxLabel == numClass) {
                LOG.warn(String.format("Adjusting labels from [%d, %d] to [%d, %d]",
                        1, numClass, 0, numClass - 1));
                for (int i = 0; i < labels.length; i++) {
                    labels[i] -= 1;
                }
            }
        }
    }

}
