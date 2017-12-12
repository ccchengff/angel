package com.tencent.angel.ml.FPGBDT.psf;

import com.tencent.angel.ml.matrix.psf.get.base.GetResult;
import com.tencent.angel.ml.matrix.psf.get.base.PartitionGetResult;
import com.tencent.angel.psagent.matrix.ResponseType;
import io.netty.buffer.ByteBuf;

/**
 * Created by ccchengff on 2017/12/1.
 */
public class QSketchGetResult extends GetResult {
  private boolean success;
  private float[] quantiles;

  public QSketchGetResult(ResponseType type, boolean success, float[] quantiles) {
    super(type);
    this.success = success;
    this.quantiles = quantiles;
  }

  public boolean isSuccess() {
    return success;
  }

  public float[] getQuantiles() {
    return quantiles;
  }

  public static class QSketchPartitionGetResult extends PartitionGetResult {
    private boolean success;
    private int numQuantiles;
    private float[] quantiles;

    public QSketchPartitionGetResult(boolean success, int numQuantiles, float[] quantiles) {
      this.success = success;
      this.numQuantiles = numQuantiles;
      this.quantiles = quantiles;
    }

    public QSketchPartitionGetResult() {
      this.success = false;
      this.numQuantiles = -1;
      this.quantiles = null;
    }

    public boolean isSuccess() {
      return success;
    }

    public int getNumQuantiles() {
      return numQuantiles;
    }

    public float[] getQuantiles() {
      return quantiles;
    }

    /**
     * Serialize object to the Netty ByteBuf.
     *
     * @param buf the Netty ByteBuf
     */
    @Override
    public void serialize(ByteBuf buf) {
      buf.writeBoolean(success);
      if (success) {
        buf.writeInt(numQuantiles);
        for (float q : quantiles)
          buf.writeFloat(q);
      }
    }

    /**
     * Deserialize object from the Netty ByteBuf.
     *
     * @param buf the Netty ByteBuf
     */
    @Override
    public void deserialize(ByteBuf buf) {
      this.success = buf.readBoolean();
      if (success) {
        this.numQuantiles = buf.readInt();
        this.quantiles = new float[this.numQuantiles];
        for (int i = 0; i < this.numQuantiles; i++)
          quantiles[i] = buf.readFloat();
      }
    }

    /**
     * Estimate serialized data size of the object, it used to ByteBuf allocation.
     *
     * @return int serialized data size of the object
     */
    @Override
    public int bufferLen() {
      if (success)
        return 8 + this.numQuantiles * 4;
      else
        return 4 + this.numQuantiles * 4;
    }
  }
}
