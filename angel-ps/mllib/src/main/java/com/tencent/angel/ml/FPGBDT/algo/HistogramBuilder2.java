package com.tencent.angel.ml.FPGBDT.algo;

import com.tencent.angel.ml.FPGBDT.algo.FPRegTreeDataStore.TrainDataStore;
import com.tencent.angel.ml.GBDT.algo.RegTree.GradPair;
import com.tencent.angel.ml.GBDT.algo.RegTree.RegTNodeStat;
import com.tencent.angel.ml.math.vector.DenseFloatVector;
import com.tencent.angel.ml.param.FPGBDTParam;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.Map;
import java.util.Set;

/**
 * Created by ccchengff on 2017/12/1.
 */
public class HistogramBuilder2 implements Runnable {
  private static final Log LOG = LogFactory.getLog(HistogramBuilder2.class);

  private FPGBDTParam param;
  private TrainDataStore trainDataStore;
  private int[] sampleFeats;
  private int[] insToNode;
  private GradPair[] gradPairs;

  private Set<Integer> activeNodeSet;
  private Map<Integer, RegTNodeStat> nodeStatMap;
  private Map<Integer, DenseFloatVector> histogramMap;

  private int builderId;
  private boolean finished;

  public HistogramBuilder2(FPGBDTParam param, TrainDataStore trainDataStore,
                           int[] sampleFeats, int[] insToNode, GradPair[] gradPairs,
                           Set<Integer> activeNodeSet, Map<Integer, RegTNodeStat> nodeStatMap,
                           Map<Integer, DenseFloatVector> histogramMap, int builderId) {
    this.param = param;
    this.trainDataStore = trainDataStore;
    this.sampleFeats = sampleFeats;
    this.insToNode = insToNode;
    this.gradPairs = gradPairs;
    this.activeNodeSet = activeNodeSet;
    this.nodeStatMap = nodeStatMap;
    this.histogramMap = histogramMap;
    this.builderId = builderId;
    this.finished = false;
  }

  @Override
  public void run() {
    // 1. get responsible feature range
    int numThread = param.numThread;
    int numSampleFeats = sampleFeats.length;
    int fStart = builderId * (numSampleFeats / numThread);
    int fEnd = builderId + 1 == numThread ? numSampleFeats
            : fStart + (numSampleFeats / numThread);
    // 2. build histograms
    for (int i = fStart; i < fEnd; i++) {
      // 2.1. get info of current feature
      int fid = sampleFeats[i];
      int[] indices = trainDataStore.getFeatIndices(fid);
      int[] bins = trainDataStore.getFeatBins(fid);
      int nnz = indices.length;
      int gradOffset = i * param.numSplit * 2;
      int hessOffset = gradOffset + param.numSplit;
      float gradTaken = 0.0f;
      float hessTaken = 0.0f;
      // 2.2. loop non-zero instances, add to histogram, and record the gradients taken
      for (int j = 0; j < nnz; j++) {
        int insIdx = indices[j];
        int nid = insToNode[insIdx];
        if (activeNodeSet.contains(nid)) {
          DenseFloatVector hist = histogramMap.get(nid);
          int binIdx = bins[j];
          int gradIdx = gradOffset + binIdx;
          int hessIdx = hessOffset + binIdx;
          GradPair gradPair = gradPairs[insIdx];
          hist.set(gradIdx, hist.get(gradIdx) + gradPair.getGrad());
          hist.set(hessIdx, hist.get(hessIdx) + gradPair.getHess());
          gradTaken += gradPair.getGrad();
          hessTaken += gradPair.getHess();
        }
      }
      // 2.3. add remaining grad and hess to zero bin
      int zeroIdx = trainDataStore.getZeroBin(fid);
      int gradIdx = gradOffset + zeroIdx;
      int hessIdx = hessOffset + zeroIdx;
      for (Map.Entry<Integer, DenseFloatVector> entry: histogramMap.entrySet()) {
        int nid = entry.getKey();
        RegTNodeStat nodeStat = nodeStatMap.get(nid);
        DenseFloatVector hist = entry.getValue();
        hist.set(gradIdx, nodeStat.sumGrad - gradTaken);
        hist.set(hessIdx, nodeStat.sumHess - hessTaken);
      }
    }
    finished = true;
  }

  public boolean isFinished() {
    return finished;
  }
}
