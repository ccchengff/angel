package com.tencent.angel.ml.FPGBDT.psf;

import com.tencent.angel.PartitionKey;
import com.tencent.angel.ml.FPGBDT.algo.RangeBitSet;
import com.tencent.angel.ml.matrix.psf.update.enhance.PartitionUpdateParam;
import com.tencent.angel.ml.matrix.psf.update.enhance.UpdateFunc;
import com.tencent.angel.ml.matrix.psf.update.enhance.UpdateParam;
import com.tencent.angel.ps.impl.PSContext;
import com.tencent.angel.ps.impl.matrix.ServerDenseIntRow;
import com.tencent.angel.ps.impl.matrix.ServerPartition;
import com.tencent.angel.ps.impl.matrix.ServerRow;
import com.tencent.angel.psagent.PSAgentContext;
import io.netty.buffer.ByteBuf;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by ccchengff on 2017/11/23.
 */

/**
 * Modify bitset (presented by an array of Integers) on PS
 * The given byte array must be aligned to 8
 */
public class BitsUpdate extends UpdateFunc {
  public static final Log LOG = LogFactory.getLog(BitsUpdate.class);

  public BitsUpdate(BitsUpdateParam param) {
    super(param);
  }

  public BitsUpdate() {
    super(null);
  }

  public static class BitsUpdateParam extends UpdateParam {
    protected final RangeBitSet bitset;
    /*protected final byte[] bits;
    protected final long from;
    protected final long to;

    public BitsUpdateParam(int matrixId, boolean updateClock,
                           byte[] bits, long from, long to) {
      super(matrixId, updateClock);
      this.bits = bits;
      this.from = from;
      this.to = to;
    }*/

    public BitsUpdateParam(int matrixId, boolean updateClock,
                           RangeBitSet bitset) {
      super(matrixId, updateClock);
      this.bitset = bitset;
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
      List<PartitionUpdateParam> partParamList = new ArrayList<>();
      for (int i = 0; i < size; i++) {
        PartitionKey partKey = partList.get(i);
        int left = (int)partKey.getStartCol() * 32;
        int right = (int)partKey.getEndCol() * 32 - 1;
        RangeBitSet subset = this.bitset.overlap(left, right);
        if (subset == null) continue;
        LOG.info(String.format("Return overlap [%d, %d]",
                subset.getRangeFrom(), subset.getRangeTo()));
        BitsPartitionUpdateParam partParam = new BitsPartitionUpdateParam(
                matrixId, partKey, updateClock, subset, left * 8);
        partParamList.add(partParam);
      }
      return partParamList;
    }
  }

  public static class BitsPartitionUpdateParam extends PartitionUpdateParam {
    /*protected final byte[] bits;
    protected final long from;
    protected final long to;

    public BitsPartitionUpdateParam(int matrixId, PartitionKey partKey, boolean updateClock,
                                    byte[] bits, long from, long to) {
      super(matrixId, partKey, updateClock);
      this.bits = bits;
      this.from = from;
      this.to = to;
    }*/

    protected RangeBitSet bitset;
    protected int offset;

    public BitsPartitionUpdateParam(int matrixId, PartitionKey partKey, boolean updateClock,
                                    RangeBitSet bitset, int offset) {
      super(matrixId, partKey, updateClock);
      this.bitset = bitset;
      this.offset = offset;
    }

    public BitsPartitionUpdateParam() {
      super();
      bitset = null;
      offset = -1;
    }

    @Override
    public void serialize(ByteBuf buf) {
      LOG.info("Serialization");
      super.serialize(buf);
      bitset.serialize(buf);
      buf.writeInt(offset);
    }

    @Override
    public void deserialize(ByteBuf buf) {
      LOG.info("Deserialization");
      super.deserialize(buf);
      if (buf.isReadable()) {
        bitset = new RangeBitSet();
        bitset.deserialize(buf);
        offset = buf.readInt();
      }
      else {
        bitset = null;
        offset = -1;
      }
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
      int startRow = part.getPartitionKey().getStartRow();
      int endRow = part.getPartitionKey().getEndRow();
      LOG.info(String.format("Rows: [%d-%d)", startRow, endRow));
      for (int i = startRow; i < endRow; i++) {
        ServerRow row = part.getRow(i);
        if (row == null) {
          continue;
        }
        bitsUpdate(row, (BitsPartitionUpdateParam)partParam);
      }
    }
  }

  private void bitsUpdate(ServerRow row, BitsPartitionUpdateParam partParam) {
    LOG.info("Row type: " + row.getRowType());
    switch (row.getRowType()) {
      case T_INT_DENSE:
        bitsUpdate((ServerDenseIntRow)row, partParam);
        break;
      default:
        break;
    }
  }

  private void bitsUpdate(ServerDenseIntRow row, BitsPartitionUpdateParam partParam) {
    LOG.info(String.format("Row columns: [%d-%d)", row.getStartCol(), row.getEndCol()));
    if (partParam.bitset == null) return;
    try {
      row.getLock().writeLock().lock();
      byte[] data = row.getDataArray();
      int from = partParam.bitset.getRangeFrom() - partParam.offset;
      int to = partParam.bitset.getRangeTo() - partParam.offset;
      LOG.info(String.format("[%d-%d] ==> [%d-%d]", partParam.bitset.getRangeFrom(),
              partParam.bitset.getRangeTo(), from, to));
      byte[] bits = partParam.bitset.toByteArray();
      int first = from >> 3;
      int last = to >> 3;
      // first byte
      byte firstByte = 0;
      int t = from & 0b111;
      for (int i = 0; i < t; i++)
        firstByte |= data[first] & (1 << i);
      for (int i = t; i < 8; i++)
        firstByte |= bits[0] & (1 << i);
      data[first] = firstByte;
      // last byte
      byte lastByte = 0;
      t = to & 0b111;
      for (int i = 0; i <= t; i++)
        lastByte |= bits[last - first] & (1 << i);
      for (int i = t + 1; i < 8; i++)
        lastByte |= data[last] & (1 << i);
      data[last] = lastByte;
      // other bytes
      first++;
      last--;
      System.arraycopy(bits, 1, data, first, last - first + 1);
      print(bits, 0, to - from);
      print(data, from, to);
      compare(data, bits, from, to);
    } finally {
      row.getLock().writeLock().unlock();
    }
  }

  static void print(byte[] arr, int from, int to) {
    String str = "";
    for (int i = from; i <= to; i++) {
      int x = i / 8, y = i % 8;
      if (((arr[x] >> y) & 0x1) == 1)
        str += "1 ";
      else
        str += "0 ";
    }
    LOG.info(str);
  }

  static void compare(byte[] arr1, byte[] arr2, int from, int to) {
    for (int i = from; i <= to; i++) {
      int x1 = i / 8, y1 = i % 8;
      int x2 = (i - from) / 8, y2 = (i - from) % 8;
      if (((arr1[x1] >> y1) & 0x1) != ((arr2[x2] >> y2) & 0x1)) {
        LOG.info("Not match!");
        break;
      }
    }
  }
}



