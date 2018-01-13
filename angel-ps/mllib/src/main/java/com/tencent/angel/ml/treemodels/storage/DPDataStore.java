package com.tencent.angel.ml.treemodels.storage;

import com.tencent.angel.ml.feature.LabeledData;
import com.tencent.angel.ml.math.vector.DenseIntVector;
import com.tencent.angel.ml.math.vector.SparseDoubleSortedVector;
import com.tencent.angel.ml.model.PSModel;
import com.tencent.angel.ml.treemodels.gbdt.GBDTModel;
import com.tencent.angel.ml.FPGBDT.algo.QuantileSketch.HeapQuantileSketch;
import com.tencent.angel.ml.treemodels.param.TreeParam;
import com.tencent.angel.worker.storage.DataBlock;
import com.tencent.angel.worker.task.TaskContext;
import it.unimi.dsi.fastutil.floats.FloatArrayList;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Data parallel data store
 */
public class DPDataStore extends DataStore {
    private static final Log LOG = LogFactory.getLog(DPDataStore.class);

    private int[][] insIndices;
    private int[][] insBins;

    public DPDataStore(TaskContext taskContext, TreeParam param) {
        super(taskContext, param);
    }

    @Override
    public void init(DataBlock<LabeledData> dataStorage, GBDTModel model) throws Exception {
        long initStart = System.currentTimeMillis();

        LOG.info("Create data parallel data meta, numFeature=" + numFeatures);
        // 1. read data
        List<SparseDoubleSortedVector> instances;
        if (splits == null) {
            instances = readDataAndCreateSketch(dataStorage, model);
        } else {
            instances = readData(dataStorage);
        }
        // 2. turn feature values into bin indexes
        findBins(instances);

        LOG.info(String.format("Create data-parallel data meta cost %d ms, numInstance=%d",
                (System.currentTimeMillis() - initStart), numInstances));
    }

    private List<SparseDoubleSortedVector> readData(DataBlock<LabeledData> dataStorage) throws IOException {
        // 1. read data
        List<SparseDoubleSortedVector> instances = new ArrayList<>(dataStorage.size());
        FloatArrayList labelsList = new FloatArrayList(dataStorage.size());
        dataStorage.resetReadIndex();
        LabeledData data = dataStorage.read();
        while (data != null) {
            SparseDoubleSortedVector x = (SparseDoubleSortedVector) data.getX();
            float y = (float) data.getY();
            instances.add(x);
            labelsList.add(y);
            data = dataStorage.read();
        }
        numInstances = instances.size();
        // 2. set info for each instance
        labels = new float[numInstances];
        preds = new float[numInstances];
        weights = new float[numInstances];
        for (int insId = 0; insId < numInstances; insId++) {
            labels[insId] = labelsList.getFloat(insId);
        }
        Arrays.fill(weights, 1.0f);

        return instances;
    }

    private List<SparseDoubleSortedVector> readDataAndCreateSketch(
            DataBlock<LabeledData> dataStorage, GBDTModel model) throws Exception {
        long readStart = System.currentTimeMillis();
        // 1. read data
        List<SparseDoubleSortedVector> instances = new ArrayList<>(dataStorage.size());
        FloatArrayList labelsList = new FloatArrayList(dataStorage.size());
        int[] nnzLocal = new int[numFeatures];
        dataStorage.resetReadIndex();
        LabeledData data = dataStorage.read();
        while (data != null) {
            SparseDoubleSortedVector x = (SparseDoubleSortedVector) data.getX();
            float y = (float) data.getY();
            instances.add(x);
            labelsList.add(y);
            int[] featIndices = x.getIndices();
            for (int fid : featIndices) {
                nnzLocal[fid]++;
            }
            data = dataStorage.read();
        }
        numInstances = instances.size();
        // 2. push local instance num, sum up feature nnz
        PSModel workerInsModel = model.getPSModel(GBDTModel.INSTANCE_NUM_MAT());
        PSModel nnzModel = model.getPSModel(GBDTModel.NNZ_NUM_MAT());
        if (taskContext.getTaskIndex() == 0) {
            workerInsModel.zero();
            nnzModel.zero();
        }
        model.sync();

        DenseIntVector workerInsVec = new DenseIntVector(param.numWorker);
        workerInsVec.set(taskContext.getTaskIndex(), numInstances);
        workerInsModel.increment(0, workerInsVec);

        DenseIntVector nnzVec = new DenseIntVector(numFeatures, nnzLocal);
        nnzModel.increment(0, nnzVec);

        workerInsModel.clock(true).get();
        nnzModel.clock(true).get();

        workerNumIns = ((DenseIntVector) workerInsModel.getRow(0)).getValues();
        nnzGlobal = ((DenseIntVector) nnzModel.getRow(0)).getValues();
        // 3. set info for each instance
        labels = new float[numInstances];
        preds = new float[numInstances];
        weights = new float[numInstances];
        for (int insId = 0; insId < numInstances; insId++) {
            labels[insId] = labelsList.getFloat(insId);
        }
        Arrays.fill(weights, 1.0f);
        // 4. create sketches
        createSketch(instances, nnzLocal, nnzGlobal, model);

        LOG.info(String.format("Read data and create sketch cost %d ms",
                System.currentTimeMillis() - readStart));
        return instances;
    }

    private void createSketch(List<SparseDoubleSortedVector> instances,
                              int[] nnzLocal, int[] nnzGlobal,
                              GBDTModel model) throws Exception {
        long createStart = System.currentTimeMillis();
        // 1. create local quantile sketches
        HeapQuantileSketch[] sketches = new HeapQuantileSketch[numFeatures];
        for (int fid = 0; fid < numFeatures; fid++) {
            sketches[fid] = new HeapQuantileSketch((long) nnzLocal[fid]);
        }
        for (int insId = 0; insId < numInstances; insId++) {
            int[] indices = instances.get(insId).getIndices();
            double[] values = instances.get(insId).getValues();
            for (int i = 0; i < indices.length; i++) {
                int fid = indices[i];
                float fvalue = (float) values[i];
                sketches[fid].update(fvalue);
            }
        }
        // 2. push to PS and merge on PS
        splits = mergeSketchAndPullQuantiles(sketches, nnzGlobal, model);
        // 3. set zero bin indexes
        zeroBins = new int[numFeatures];
        Arrays.setAll(zeroBins, i -> findZeroBin(splits[i]));

        LOG.info(String.format("Create sketch cost %d ms",
                System.currentTimeMillis() - createStart));
    }

    private void findBins(List<SparseDoubleSortedVector> instances) {
        insIndices = new int[numInstances][];
        insBins = new int[numInstances][];
        for (int insId = 0; insId < numInstances; insId++) {
            SparseDoubleSortedVector x = instances.get(insId);
            insIndices[insId] = x.getIndices();
            double[] values = x.getValues();
            insBins[insId] = new int[insIndices[insId].length];
            for (int i = 0; i < insIndices[insId].length; i++) {
                int fid = insIndices[insId][i];
                float fvalue = (float) values[i];
                insBins[insId][i] = indexOf(fvalue, fid);
            }
        }
    }

    public int[] getInsIndices(int insId) {
        return insIndices[insId];
    }

    public int[][] getInsIndices() {
        return insIndices;
    }

    public int[] getInsBins(int insId) {
        return insBins[insId];
    }

    public int[][] getInsBins() {
        return insBins;
    }
}
