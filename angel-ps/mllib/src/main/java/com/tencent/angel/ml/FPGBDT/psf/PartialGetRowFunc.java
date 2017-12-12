package com.tencent.angel.ml.FPGBDT.psf;

import com.tencent.angel.PartitionKey;
import com.tencent.angel.ml.matrix.psf.get.base.GetParam;
import com.tencent.angel.ml.matrix.psf.get.base.GetResult;
import com.tencent.angel.ml.matrix.psf.get.base.PartitionGetParam;
import com.tencent.angel.ml.matrix.psf.get.base.PartitionGetResult;
import com.tencent.angel.ml.matrix.psf.get.single.GetRowFunc;
import com.tencent.angel.ps.impl.PSContext;
import com.tencent.angel.ps.impl.matrix.ServerDenseFloatRow;
import com.tencent.angel.psagent.PSAgentContext;
import com.tencent.angel.psagent.matrix.ResponseType;
import io.netty.buffer.ByteBuf;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by ccchengff on 2017/11/30.
 */
public class PartialGetRowFunc extends GetRowFunc {

  public PartialGetRowFunc(PartialGetRowParam param) {
    super(param);
  }

  public PartialGetRowFunc(int matrixId, int rowId, int offset, int length) {
    super(new PartialGetRowParam(matrixId, rowId, offset, length));
  }

  public PartialGetRowFunc() {}

  public static class PartialGetRowParam extends GetParam {
    protected int rowId;
    protected int offset;
    protected int length;

    public PartialGetRowParam(int matrixId, int rowId, int offset, int length) {
      super(matrixId);
      this.rowId = rowId;
      this.offset = offset;
      this.length = length;
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
          int left = Math.max((int)partKey.getStartCol(), offset);
          int right = Math.min((int)partKey.getEndCol(), offset + length);
          if (left < right) {
            partParams.add(new PartialPartitionGetParam(
                    matrixId, partKey, rowId, left, right - left));
          }
        }
      }
      return partParams;
    }
  }

  public static class PartialPartitionGetParam extends PartitionGetParam {
    private int rowId;
    private int offset;
    private int length;

    public PartialPartitionGetParam(int matrixId, PartitionKey partKey,
                                    int rowId, int offset, int length) {
      super(matrixId, partKey);
      this.rowId = rowId;
      this.offset = offset;
      this.length = length;
    }

    public PartialPartitionGetParam() {
      super(-1, null);
      this.rowId = -1;
      this.offset = -1;
      this.length = -1;
    }

    @Override
    public void serialize(ByteBuf buf) {
      super.serialize(buf);
      buf.writeInt(rowId);
      buf.writeInt(offset);
      buf.writeInt(length);
    }

    @Override
    public void deserialize(ByteBuf buf) {
      super.deserialize(buf);
      this.rowId = buf.readInt();
      this.offset = buf.readInt();
      this.length = buf.readInt();
    }

    @Override
    public int bufferLen() {
      return super.bufferLen() + 12;
    }
  }

  @Override
  public PartitionGetResult partitionGet(PartitionGetParam partParam) {
    PartialPartitionGetParam param = (PartialPartitionGetParam) partParam;
    ServerDenseFloatRow row = (ServerDenseFloatRow) PSContext.get().getMatrixPartitionManager()
            .getRow(param.getMatrixId(), param.rowId, param.getPartKey().getPartitionId());
    float[] data = new float[param.length];
    FloatBuffer buf = row.getData();
    int offset = param.offset;
    int limit = param.offset + param.length;
    for (int i = offset; i < limit; i++) {
      data[i - offset] = buf.get(i);
    }
    return new PartialGetRowResult.PartitalPartitionGetResult(data, param.offset, param.length);
  }

  @Override
  public GetResult merge(List<PartitionGetResult> partResults) {
    int size = partResults.size();
    if (size == 1) {
      float[] data = ((PartialGetRowResult.PartitalPartitionGetResult)
              partResults.get(0)).getData();
      return new PartialGetRowResult(ResponseType.SUCCESS, data);
    }
    else {
      PartialGetRowResult.PartitalPartitionGetResult[] results =
              new PartialGetRowResult.PartitalPartitionGetResult[size];
      for (int i = 0; i < size; i++) {
        results[i] = (PartialGetRowResult.PartitalPartitionGetResult) partResults.get(i);
      }
      Arrays.sort(results);
      for (int i = 1; i < size; i++) {
        results[0].append(results[i]);
      }
      return new PartialGetRowResult(ResponseType.SUCCESS, results[0].getData());
    }
  }
}
