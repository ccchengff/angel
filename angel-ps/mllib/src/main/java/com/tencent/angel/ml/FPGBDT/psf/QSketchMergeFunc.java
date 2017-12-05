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

import java.util.ArrayList;
import java.util.List;

/**
 * Created by ccchengff on 2017/11/30.
 */
public class QSketchUpdateFunc extends UpdateFunc {
  public static final Log LOG = LogFactory.getLog(QSketchUpdateFunc.class);

  public QSketchUpdateFunc(QSketchUpdateParam param) {
    super(param);
  }

  public QSketchUpdateFunc() {
    super(null);
  }

  public static class QSketchUpdateParam extends UpdateParam {
    protected int rowId;
    protected HeapQuantileSketch qs;
    protected long estimateN;

    public QSketchUpdateParam(int matrixId, boolean updateClock, int rowId,
                              HeapQuantileSketch qs, long estimateN) {
      super(matrixId, updateClock);
      this.rowId = rowId;
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
          partParamList.add(new QSketchPartitionUpdateParam(
                  matrixId, partKey, updateClock, rowId, qs, estimateN));
        }
      }
      return partParamList;
    }

  }

  public static class QSketchPartitionUpdateParam extends PartitionUpdateParam {
    protected int rowId;
    protected HeapQuantileSketch qs;
    protected long estimateN;

    public QSketchPartitionUpdateParam(int matrixId, PartitionKey partKey, boolean updateClock,
                                       int rowId, HeapQuantileSketch qs, long estimateN) {
      super(matrixId, partKey, updateClock);
      this.rowId = rowId;
      this.qs = qs;
      this.estimateN = estimateN;
    }

    public QSketchPartitionUpdateParam() {
      super();
      this.rowId = -1;
      this.qs = null;
      this.estimateN = -1;
    }

    @Override
    public void serialize(ByteBuf buf) {
      super.serialize(buf);
      buf.writeInt(rowId);
      qs.serialize(buf);
      buf.writeLong(estimateN);
    }

    @Override
    public void deserialize(ByteBuf buf) {
      super.deserialize(buf);
      rowId = buf.readInt();
      this.qs = new HeapQuantileSketch(buf);
      estimateN = buf.readLong();
    }

    @Override
    public int bufferLen() {
      return super.bufferLen() + 12 + qs.bufferLen();
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
      QSketchPartitionUpdateParam param = (QSketchPartitionUpdateParam)partParam;
      int startRow = part.getPartitionKey().getStartRow();
      int endRow = part.getPartitionKey().getEndRow();
      for (int i = startRow; i < endRow; i++) {
        if (i == param.rowId) {
          ServerRow row = part.getRow(i);
          if (row == null) {
            throw new AngelException("Get null row: " + i);
          }
          if (row.getRowType() == T_INT_FLOAT) {
            qsketchUpdate((ServerDenseFloatRow) row, param);
          }
        }
      }
    }
  }

  private void qsketchUpdate(ServerDenseFloatRow row, QSketchPartitionUpdateParam partParam) {
    try {
      row.getLock().writeLock().lock();

    } finally {
      row.getLock().writeLock().unlock();
    }
  }
}
