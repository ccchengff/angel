package com.tencent.angel.ml.treemodels.gbdt.dp;

import com.tencent.angel.ml.treemodels.gbdt.GBDTController;
import com.tencent.angel.ml.treemodels.gbdt.GBDTModel;
import com.tencent.angel.ml.treemodels.param.GBDTParam;
import com.tencent.angel.ml.treemodels.storage.DPDataStore;
import com.tencent.angel.worker.task.TaskContext;

public class DPGBDTController extends GBDTController<DPDataStore, DPDataStore> {
    private int[] nodeStats;  // used for multi-batch histogram building

    public DPGBDTController(TaskContext taskContext, GBDTParam param, GBDTModel model,
                            DPDataStore dpDataStore, DPDataStore validDataStore) {
        super(taskContext, param, model, dpDataStore, validDataStore);
    }

    @Override
    public void init() {
        super.init();
        this.nodeStats = new int[param.maxNodeNum];
    }

    @Override
    public void createNewTree() throws Exception {

    }

    @Override
    protected void sampleFeature() {

    }

    @Override
    public void findSplit() {

    }

    @Override
    public void afterSplit() {

    }
}
