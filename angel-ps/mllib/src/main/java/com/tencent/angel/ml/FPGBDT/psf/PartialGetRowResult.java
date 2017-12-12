package com.tencent.angel.ml.FPGBDT.psf;

import com.tencent.angel.ml.matrix.psf.get.base.GetResult;
import com.tencent.angel.ml.matrix.psf.get.base.PartitionGetResult;
import com.tencent.angel.psagent.matrix.ResponseType;
import io.netty.buffer.ByteBuf;

import java.util.Arrays;

/**
 * Created by ccchengff on 2017/11/30.
 */
public class PartialGetRowResult extends GetResult {
  private float[] data;

  public PartialGetRowResult(ResponseType type, float[] data) {
    super(type);
    this.data = data;
  }

  public float[] getData() {
    return data;
  }

  public static class PartitalPartitionGetResult extends PartitionGetResult
          implements Comparable {
    private float[] data;
    private int offset;
    private int length;

    public PartitalPartitionGetResult(float[] data, int offset, int length) {
      assert data.length == length;
      this.data = data;
      this.offset = offset;
      this.length = length;
    }

    public PartitalPartitionGetResult() {
      this.data = null;
      this.offset = -1;
      this.length = -1;
    }

    public void append(PartitalPartitionGetResult o) {
      assert offset + length == o.offset;
      data = Arrays.copyOf(data, length + o.length);
      System.arraycopy(o.data, 0, data, length, o.length);
      length += o.length;
    }

    public float[] getData() {
      return data;
    }

    /**
     * Serialize object to the Netty ByteBuf.
     *
     * @param buf the Netty ByteBuf
     */
    @Override
    public void serialize(ByteBuf buf) {
      buf.writeInt(offset);
      buf.writeInt(length);
      for (float v: data) {
        buf.writeFloat(v);
      }
    }

    /**
     * Deserialize object from the Netty ByteBuf.
     *
     * @param buf the Netty ByteBuf
     */
    @Override
    public void deserialize(ByteBuf buf) {
      this.offset = buf.readInt();
      this.length = buf.readInt();
      this.data = new float[this.length];
      for (int i = 0; i < this.length; i++) {
        data[i] = buf.readFloat();
      }

    }

    /**
     * Estimate serialized data size of the object, it used to ByteBuf allocation.
     *
     * @return int serialized data size of the object
     */
    @Override
    public int bufferLen() {
      return 8 + this.data.length * 4;
    }

    /**
     * Compares this object with the specified object for order.  Returns a
     * negative integer, zero, or a positive integer as this object is less
     * than, equal to, or greater than the specified object.
     * <p>
     * <p>The implementor must ensure <tt>sgn(x.compareTo(y)) ==
     * -sgn(y.compareTo(x))</tt> for all <tt>x</tt> and <tt>y</tt>.  (This
     * implies that <tt>x.compareTo(y)</tt> must throw an exception iff
     * <tt>y.compareTo(x)</tt> throws an exception.)
     * <p>
     * <p>The implementor must also ensure that the relation is transitive:
     * <tt>(x.compareTo(y)&gt;0 &amp;&amp; y.compareTo(z)&gt;0)</tt> implies
     * <tt>x.compareTo(z)&gt;0</tt>.
     * <p>
     * <p>Finally, the implementor must ensure that <tt>x.compareTo(y)==0</tt>
     * implies that <tt>sgn(x.compareTo(z)) == sgn(y.compareTo(z))</tt>, for
     * all <tt>z</tt>.
     * <p>
     * <p>It is strongly recommended, but <i>not</i> strictly required that
     * <tt>(x.compareTo(y)==0) == (x.equals(y))</tt>.  Generally speaking, any
     * class that implements the <tt>Comparable</tt> interface and violates
     * this condition should clearly indicate this fact.  The recommended
     * language is "Note: this class has a natural ordering that is
     * inconsistent with equals."
     * <p>
     * <p>In the foregoing description, the notation
     * <tt>sgn(</tt><i>expression</i><tt>)</tt> designates the mathematical
     * <i>signum</i> function, which is defined to return one of <tt>-1</tt>,
     * <tt>0</tt>, or <tt>1</tt> according to whether the value of
     * <i>expression</i> is negative, zero or positive.
     *
     * @param o the object to be compared.
     * @return a negative integer, zero, or a positive integer as this object
     * is less than, equal to, or greater than the specified object.
     * @throws NullPointerException if the specified object is null
     * @throws ClassCastException   if the specified object's type prevents it
     *                              from being compared to this object.
     */
    @Override
    public int compareTo(Object o) {
      return this.offset - ((PartitalPartitionGetResult) o).offset;
    }
  }
}
