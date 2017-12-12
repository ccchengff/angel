package com.tencent.angel.ml.FPGBDT.psf;

import com.tencent.angel.PartitionKey;
import com.tencent.angel.ml.matrix.psf.get.base.*;
import com.tencent.angel.ps.impl.PSContext;
import com.tencent.angel.ps.impl.matrix.ServerDenseFloatRow;
import com.tencent.angel.psagent.PSAgentContext;
import com.tencent.angel.psagent.matrix.ResponseType;
import io.netty.buffer.ByteBuf;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by ccchengff on 2017/12/1.
 */
public class QSketchGetFunc extends GetFunc {

  public QSketchGetFunc(QSketchGetParam param) {
    super(param);
  }

  public QSketchGetFunc(int matrixId, int rowId, int numWorker, int numQuantiles) {
    this(new QSketchGetParam(matrixId, rowId, numWorker, numQuantiles));
  }

  public QSketchGetFunc() {
    super(null);
  }

  public static class QSketchGetParam extends GetParam {
    protected int rowId;
    protected int numWorker;
    protected int numQuantiles;

    public QSketchGetParam(int matrixId, int rowId, int numWorker, int numSplit) {
      super(matrixId);
      this.rowId = rowId;
      this.numWorker = numWorker;
      this.numQuantiles = numSplit;
    }

    @Override
    public List<PartitionGetParam> split() {
      List<PartitionKey> partList =
              PSAgentContext.get().getMatrixPartitionRouter()
                      .getPartitionKeyList(matrixId);
      int size = partList.size();

      List<PartitionGetParam> partParams = new ArrayList<>();
      for (int i = 0; i < size; i++) {
        PartitionKey partKey = partList.get(i);
        if (partKey.getStartRow() <= rowId && partKey.getEndRow() > rowId) {
          partParams.add(new QSketchPartitionGetParam(matrixId, partKey,
                  rowId, numWorker, numQuantiles));
        }
      }
      return partParams;
    }
  }

  public static class QSketchPartitionGetParam extends PartitionGetParam {
    protected int rowId;
    protected int numWorker;
    protected int numQuantiles;

    public QSketchPartitionGetParam(int matrixId, PartitionKey partKey, int rowId,
                                    int numWorker, int numSplit) {
      super(matrixId, partKey);
      this.rowId = rowId;
      this.numWorker = numWorker;
      this.numQuantiles = numSplit;
    }

    public QSketchPartitionGetParam() {
      super(-1, null);
      this.rowId = -1;
      this.numWorker = -1;
      this.numQuantiles = -1;
    }

    @Override
    public void serialize(ByteBuf buf) {
      super.serialize(buf);
      buf.writeInt(rowId);
      buf.writeInt(numWorker);
      buf.writeInt(numQuantiles);
    }

    @Override
    public void deserialize(ByteBuf buf) {
      super.deserialize(buf);
      this.rowId = buf.readInt();
      this.numWorker = buf.readInt();
      this.numQuantiles = buf.readInt();
    }

    @Override
    public int bufferLen() {
      return super.bufferLen() + 12;
    }
  }

  /**
   * Partition get. This function is called on PS.
   *
   * @param partParam the partition parameter
   * @return the partition result
   */
  @Override
  public PartitionGetResult partitionGet(PartitionGetParam partParam) {
    QSketchPartitionGetParam param = (QSketchPartitionGetParam) partParam;
    ServerDenseFloatRow row = (ServerDenseFloatRow) PSContext.get().getMatrixPartitionManager()
            .getRow(param.getMatrixId(), param.rowId, param.getPartKey().getPartitionId());

    FloatBuffer buf = row.getData();
    int numMerged = Float.floatToIntBits(buf.get(0));
    if (numMerged == param.numWorker) {
      float[] quantiles = new float[param.numQuantiles];
      for (int i = 0; i < param.numQuantiles; i++)
        quantiles[i] = buf.get(i + 1);
      return new QSketchGetResult.QSketchPartitionGetResult(
              true, param.numQuantiles, quantiles);
    }
    else {
      return new QSketchGetResult.QSketchPartitionGetResult(false, -1, null);
    }
  }

  /**
   * Merge the partition get results. This function is called on PSAgent.
   *
   * @param partResults the partition results
   * @return the merged result
   */
  @Override
  public GetResult merge(List<PartitionGetResult> partResults) {
    assert partResults.size() == 1;
    QSketchGetResult.QSketchPartitionGetResult partResult =
            (QSketchGetResult.QSketchPartitionGetResult) partResults.get(0);
    if (partResult.isSuccess()) {
      return new QSketchGetResult(ResponseType.SUCCESS, true, partResult.getQuantiles());
    }
    else {
      return new QSketchGetResult(ResponseType.SUCCESS, false, null);
    }
  }
}
