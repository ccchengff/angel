package com.tencent.angel.ml.FPGBDT.psf;

import com.tencent.angel.PartitionKey;
import com.tencent.angel.exception.AngelException;
import com.tencent.angel.ml.FPGBDT.algo.QuantileSketch.HeapQuantileSketch;
import com.tencent.angel.ml.matrix.psf.update.enhance.PartitionUpdateParam;
import com.tencent.angel.ml.matrix.psf.update.enhance.UpdateFunc;
import com.tencent.angel.ml.matrix.psf.update.enhance.UpdateParam;
import com.tencent.angel.ps.impl.PSContext;
import com.tencent.angel.ps.impl.matrix.ServerDenseFloatRow;
import com.tencent.angel.ps.impl.matrix.ServerPartition;
import com.tencent.angel.ps.impl.matrix.ServerRow;
import com.tencent.angel.psagent.PSAgentContext;
import io.netty.buffer.ByteBuf;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by ccchengff on 2017/11/30.
 */
public class QSketchMergeFunc extends UpdateFunc {
  private static final Log LOG = LogFactory.getLog(QSketchMergeFunc.class);

  public QSketchMergeFunc(QSketchMergeParam param) {
    super(param);
  }

  public QSketchMergeFunc() {
    super(null);
  }

  public static class QSketchMergeParam extends UpdateParam {
    protected int rowId;
    protected int numWorker;
    protected int numQuantile;
    protected HeapQuantileSketch qs;
    protected long estimateN;

    public QSketchMergeParam(int matrixId, boolean updateClock, int rowId, int numWorker,
                             int numQuantile, HeapQuantileSketch qs, long estimateN) {
      super(matrixId, updateClock);
      this.rowId = rowId;
      this.numWorker = numWorker;
      this.numQuantile = numQuantile;
      this.qs = qs;
      this.estimateN = estimateN;
    }

    /**
     * Split list.
     *
     * @return the list
     */
    @Override
    public List<PartitionUpdateParam> split() {
      List<PartitionKey> partList =
              PSAgentContext.get().getMatrixPartitionRouter().getPartitionKeyList(matrixId);
      int size = partList.size();
      List<PartitionUpdateParam> partParamList = new ArrayList<>(size);
      for (int i = 0; i < size; i++) {
        PartitionKey partKey = partList.get(i);
        if (partKey.getStartRow() <= rowId && partKey.getEndRow() > rowId) {
          partParamList.add(new QSketchPartitionMergeParam(matrixId, partKey, updateClock,
                  rowId, numWorker, numQuantile, qs, estimateN));
        }
      }
      return partParamList;
    }

  }

  public static class QSketchPartitionMergeParam extends PartitionUpdateParam {
    protected int rowId;
    protected int numWorker;
    protected int numQuantile;
    protected HeapQuantileSketch qs;
    protected long estimateN;

    public QSketchPartitionMergeParam(int matrixId, PartitionKey partKey, boolean updateClock,
                                      int rowId, int numWorker, int numQuantile,
                                      HeapQuantileSketch qs, long estimateN) {
      super(matrixId, partKey, updateClock);
      this.rowId = rowId;
      this.numWorker = numWorker;
      this.numQuantile = numQuantile;
      this.qs = qs;
      this.estimateN = estimateN;
    }

    public QSketchPartitionMergeParam() {
      super();
      this.rowId = -1;
      this.numWorker = -1;
      this.numQuantile = -1;
      this.qs = null;
      this.estimateN = -1;
    }

    @Override
    public void serialize(ByteBuf buf) {
      super.serialize(buf);
      buf.writeInt(rowId);
      buf.writeInt(numWorker);
      buf.writeInt(numQuantile);
      qs.serialize(buf);
      buf.writeLong(estimateN);
    }

    @Override
    public void deserialize(ByteBuf buf) {
      super.deserialize(buf);
      this.rowId = buf.readInt();
      this.numWorker = buf.readInt();
      this.numQuantile = buf.readInt();
      this.qs = new HeapQuantileSketch(buf);
      this.estimateN = buf.readLong();
    }

    @Override
    public int bufferLen() {
      return super.bufferLen() + 20 + qs.bufferLen();
    }
  }

  /**
   * Partition update.
   *
   * @param partParam the partition parameter
   */
  @Override
  public void partitionUpdate(PartitionUpdateParam partParam) {
    ServerPartition part =
            PSContext.get().getMatrixPartitionManager()
                    .getPartition(partParam.getMatrixId(), partParam.getPartKey().getPartitionId());

    if (part != null) {
      QSketchPartitionMergeParam param = (QSketchPartitionMergeParam)partParam;
      int startRow = part.getPartitionKey().getStartRow();
      int endRow = part.getPartitionKey().getEndRow();
      if (startRow <= param.rowId && endRow > param.rowId) {
        ServerRow row = part.getRow(param.rowId);
        if (row == null) {
          throw new AngelException("Get null row: " + param.rowId);
        }
        switch (row.getRowType()) {
          case T_FLOAT_DENSE:
            qsketchMerge((ServerDenseFloatRow) row, param);
            break;

          default:
            break;
        }
      }
    }
  }

  private void qsketchMerge(ServerDenseFloatRow row, QSketchPartitionMergeParam partParam) {
    try {
      row.getLock().writeLock().lock();
      // read sketch from row data, merge sketch
      HeapQuantileSketch qs1 = null;
      HeapQuantileSketch qs2 = partParam.qs;
      byte[] data = row.getDataArray();
      ByteBuffer buf = ByteBuffer.wrap(data);
      buf.mark();
      int numMerged = buf.getInt();
      if (numMerged == 0)
        qs1 = new HeapQuantileSketch(qs2.getK(), partParam.estimateN);
      else
        qs1 = new HeapQuantileSketch(buf);
      LOG.debug(String.format("Row[%d] before merge[%d]: k[%d, %d], n[%d, %d], estimateN[%d, %d]",
              row.getRowId(), numMerged, qs1.getK(), qs2.getK(), qs1.getN(), qs2.getN(),
              qs1.getEstimateN(), qs2.getEstimateN()));
      qs1.merge(qs2);
      numMerged++;
      LOG.debug(String.format("Row[%d] after merge[%d]: k[%d, %d], n[%d, %d], estimateN[%d, %d]",
              row.getRowId(), numMerged, qs1.getK(), qs2.getK(), qs1.getN(), qs2.getN(),
              qs1.getEstimateN(), qs2.getEstimateN()));
      if (numMerged < partParam.numWorker) {
        // write sketch back to row data
        buf.reset();
        buf.putInt(numMerged);
        qs1.serialize(buf);
      }
      else {
        // get quantiles and write to row data
        float[] quantiles = qs1.getQuantiles(partParam.numQuantile);
        //LOG.info("Row[" + row.getRowId() + "] quantiles: " + Arrays.toString(quantiles));
        FloatBuffer writeBuf = row.getData();
        writeBuf.put(Float.intBitsToFloat(numMerged));
        writeBuf.put(quantiles);
      }
    } finally {
      row.getLock().writeLock().unlock();
    }
  }
}
