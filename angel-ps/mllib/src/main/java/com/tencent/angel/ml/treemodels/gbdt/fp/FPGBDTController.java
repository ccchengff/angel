package com.tencent.angel.ml.treemodels.gbdt.fp;

import com.tencent.angel.ml.treemodels.gbdt.GBDTController;
import com.tencent.angel.ml.treemodels.gbdt.GBDTModel;
import com.tencent.angel.ml.treemodels.gbdt.GBDTPhase;
import com.tencent.angel.ml.treemodels.param.GBDTParam;
import com.tencent.angel.ml.treemodels.storage.DPDataStore;
import com.tencent.angel.ml.treemodels.storage.DataStore;
import com.tencent.angel.ml.treemodels.storage.FPDataStore;
import com.tencent.angel.ml.treemodels.tree.regression.RegTree;
import com.tencent.angel.worker.task.TaskContext;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.Arrays;
import java.util.Random;

public class FPGBDTController extends GBDTController<FPDataStore, DPDataStore> {
    private static final Log LOG = LogFactory.getLog(FPGBDTController.class);

    // map instance to tree node, each item is tree node on which it locates
    private int[] insToNode;

    public FPGBDTController(TaskContext taskContext, GBDTParam param, GBDTModel model,
                            FPDataStore trainDataStore, DPDataStore validDataStore) {
        super(taskContext, param, model, trainDataStore, validDataStore);
    }

    @Override
    public void init() {
        super.init();
        this.insToNode = new int[trainDataStore.getNumInstances()];
    }

    @Override
    public void createNewTree() throws Exception {
        LOG.info("------Create new tree------");
        long startTime = System.currentTimeMillis();
        // 1. init new tree
        this.forest[currentTree] = new RegTree();
        // 2. sample features
        sampleFeature();
        // 3. calc grad pairs
        calGradPairs();
        // 4. reset instance position, set the root node's span
        Arrays.setAll(this.nodeToIns, i -> i);
        this.nodePosStart[0] = 0;
        this.nodePosEnd[0] = trainDataStore.getNumInstances() - 1;
        Arrays.fill(this.insToNode, 0);
        // 5. set root as ready
        this.readyNodes.add(0);
        // 6. set phase
        this.phase = GBDTPhase.RUN_ACTIVE;
        LOG.info(String.format("Create new tree cost %d ms",
                System.currentTimeMillis() - startTime));
    }

    @Override
    protected void sampleFeature() {
        LOG.info("------Sample feature------");
        long startTime = System.currentTimeMillis();
        if (param.featSampleRatio < 1) {
            IntArrayList featList = new IntArrayList();
            int featLo = trainDataStore.getFeatLo();
            int featHi = trainDataStore.getFeatHi();
            Random random = new Random();
            do {
                for (int fid = featLo; fid < featHi; fid++) {
                    float prob = param.featSampleRatio;
                    if (random.nextFloat() <= param.featSampleRatio) {
                        featList.add(fid);
                    }
                }
            } while (featList.size() == 0);
            fset = featList.toIntArray();
        } else if (fset == null) {
            int featLo = trainDataStore.getFeatLo();
            int featHi = trainDataStore.getFeatHi();
            fset = new int[featHi - featLo];
            Arrays.setAll(fset, i -> i + featLo);
        }
        LOG.info(String.format("Sample feature cost %d ms, sample ratio %f, return %d features",
                System.currentTimeMillis() - startTime, param.featSampleRatio, fset.length));
    }

    @Override
    public void findSplit() {

    }

    @Override
    public void afterSplit() {

    }
}

