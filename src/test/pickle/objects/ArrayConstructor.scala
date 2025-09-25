package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import net.razorvine.pickle.PickleException
import net.razorvine.pickle.PickleUtils
import java.util
import scala.jdk.CollectionConverters.*
/**
 * Creates arrays of objects. Returns a primitive type array such as int[] if
 * the objects are ints, etc. Returns an ArrayList<Object> if it needs to
 * contain arbitrary objects (such as lists).
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
class ArrayConstructor extends IObjectConstructor {
  @throws[PickleException]
  override def construct(args: Array[AnyRef]): AnyRef = {
    // args for array constructor: [ String typecode, ArrayList<Object> values ]
    // or: [ constructor_class, typecode, machinecode_type, byte[] ]
    if (args.length == 4) {
      val constructor = args(0).asInstanceOf[ArrayConstructor]
      val typecode = args(1).asInstanceOf[String].charAt(0)
      val machinecodeType = args(2).asInstanceOf[Integer]
      val data = args(3).asInstanceOf[Array[Byte]]
      return constructor.construct(typecode, machinecodeType, data)
    }
    if (args.length != 2) throw new PickleException("invalid pickle data for array; expected 2 args, got " + args.length)
    val typecode = args(0).asInstanceOf[String]
    if (args(1).isInstanceOf[String]) {
      // python 2.6 encodes the array as a string sequence rather than a list
      // unpickling this is not supported at this time
      throw new PickleException("unsupported Python 2.6 array pickle format")
    }
    @SuppressWarnings(Array("unchecked")) 
    val values = args(1).asInstanceOf[util.ArrayList[AnyRef]]
    typecode.charAt(0) match {
      case 'c' => // character 1 -> char[]
      case 'u' => // Unicode character 2 -> char[]
        val result = new Array[Char](values.size)
        var i = 0
        
        for (c <- values.asScala) {
          result({
            i += 1; i - 1
          }) = c.asInstanceOf[String].charAt(0)
        }
        result
      case 'b' => // signed integer 1 -> byte[]
        val result = new Array[Byte](values.size)
        var i = 0
        
        for (c <- values) {
          result({
            i += 1; i - 1
          }) = c.asInstanceOf[Number].byteValue
        }
        result
      case 'B' => // unsigned integer 1 -> short[]
      case 'h' => // signed integer 2 -> short[]
        val result = new Array[Short](values.size)
        var i = 0
        
        for (c <- values) {
          result({
            i += 1; i - 1
          }) = c.asInstanceOf[Number].shortValue
        }
        result
      case 'H' => // unsigned integer 2 -> int[]
      case 'i' => // signed integer 2 -> int[]
      case 'l' => // signed integer 4 -> int[]
        val result = new Array[Int](values.size)
        var i = 0
       
        for (c <- values) {
          result({
            i += 1; i - 1
          }) = c.asInstanceOf[Number].intValue
        }
        result
      case 'I' => // unsigned integer 4 -> long[]
      case 'L' => // unsigned integer 4 -> long[]
        val result = new Array[Long](values.size)
        var i = 0
        
        for (c <- values) {
          result({
            i += 1; i - 1
          }) = c.asInstanceOf[Number].longValue
        }
        result
      case 'f' => // floating point 4 -> float[]
        val result = new Array[Float](values.size)
        var i = 0
       
        for (c <- values) {
          result({
            i += 1; i - 1
          }) = c.asInstanceOf[Number].floatValue
        }
        result
      case 'd' => // floating point 8 -> double[]
        val result = new Array[Double](values.size)
        var i = 0
        
        for (c <- values) {
          result({
            i += 1; i - 1
          }) = c.asInstanceOf[Number].doubleValue
        }
        result
      case _ =>
        throw new PickleException("invalid array typecode: " + typecode)
    }
  }

  /**
   * Create an object based on machine code type
   */
  @throws[PickleException]
  def construct(typecode: Char, machinecode: Int, data: Array[Byte]): AnyRef = {
    // Machine format codes.
    // Search for "enum machine_format_code" in Modules/arraymodule.c to get
    // the authoritative values.
    // UNKNOWN_FORMAT = -1
    // UNSIGNED_INT8 = 0
    // SIGNED_INT8 = 1
    // UNSIGNED_INT16_LE = 2
    // UNSIGNED_INT16_BE = 3
    // SIGNED_INT16_LE = 4
    // SIGNED_INT16_BE = 5
    // UNSIGNED_INT32_LE = 6
    // UNSIGNED_INT32_BE = 7
    // SIGNED_INT32_LE = 8
    // SIGNED_INT32_BE = 9
    // UNSIGNED_INT64_LE = 10
    // UNSIGNED_INT64_BE = 11
    // SIGNED_INT64_LE = 12
    // SIGNED_INT64_BE = 13
    // IEEE_754_FLOAT_LE = 14
    // IEEE_754_FLOAT_BE = 15
    // IEEE_754_DOUBLE_LE = 16
    // IEEE_754_DOUBLE_BE = 17
    // UTF16_LE = 18
    // UTF16_BE = 19
    // UTF32_LE = 20
    // UTF32_BE = 21
    if (machinecode < 0) throw new PickleException("unknown machine type format")
    typecode match {
      case 'c' => // character 1 -> char[]
      case 'u' => // Unicode character 2 -> char[]
        if (machinecode != 18 && machinecode != 19 && machinecode != 20 && machinecode != 21) throw new PickleException("for c/u type must be 18/19/20/21")
        if (machinecode == 18 || machinecode == 19) {
          // utf-16 , 2 bytes
          if (data.length % 2 != 0) throw new PickleException("data size alignment error")
          constructCharArrayUTF16(machinecode, data)
        }
        else {
          // utf-32, 4 bytes
          if (data.length % 4 != 0) throw new PickleException("data size alignment error")
          constructCharArrayUTF32(machinecode, data)
        }
      case 'b' => // signed integer 1 -> byte[]
        if (machinecode != 1) throw new PickleException("for b type must be 1")
        data
      case 'B' => // unsigned integer 1 -> short[]
        if (machinecode != 0) throw new PickleException("for B type must be 0")
        constructShortArrayFromUByte(data)
      case 'h' => // signed integer 2 -> short[]
        if (machinecode != 4 && machinecode != 5) throw new PickleException("for h type must be 4/5")
        if (data.length % 2 != 0) throw new PickleException("data size alignment error")
        constructShortArraySigned(machinecode, data)
      case 'H' => // unsigned integer 2 -> int[]
        if (machinecode != 2 && machinecode != 3) throw new PickleException("for H type must be 2/3")
        if (data.length % 2 != 0) throw new PickleException("data size alignment error")
        constructIntArrayFromUShort(machinecode, data)
      case 'i' => // signed integer 4 -> int[]
        if (machinecode != 8 && machinecode != 9) throw new PickleException("for i type must be 8/9")
        if (data.length % 4 != 0) throw new PickleException("data size alignment error")
        constructIntArrayFromInt32(machinecode, data)
      case 'l' => // signed integer 4/8 -> int[]
        if (machinecode != 8 && machinecode != 9 && machinecode != 12 && machinecode != 13) throw new PickleException("for l type must be 8/9/12/13")
        if ((machinecode == 8 || machinecode == 9) && (data.length % 4 != 0)) throw new PickleException("data size alignment error")
        if ((machinecode == 12 || machinecode == 13) && (data.length % 8 != 0)) throw new PickleException("data size alignment error")
        if (machinecode == 8 || machinecode == 9) {
          //32 bits
          constructIntArrayFromInt32(machinecode, data)
        }
        else {
          //64 bits
          constructLongArrayFromInt64(machinecode, data)
        }
      case 'I' => // unsigned integer 4 -> long[]
        if (machinecode != 6 && machinecode != 7) throw new PickleException("for I type must be 6/7")
        if (data.length % 4 != 0) throw new PickleException("data size alignment error")
        constructLongArrayFromUInt32(machinecode, data)
      case 'L' => // unsigned integer 4/8 -> long[]
        if (machinecode != 6 && machinecode != 7 && machinecode != 10 && machinecode != 11) throw new PickleException("for L type must be 6/7/10/11")
        if ((machinecode == 6 || machinecode == 7) && (data.length % 4 != 0)) throw new PickleException("data size alignment error")
        if ((machinecode == 10 || machinecode == 11) && (data.length % 8 != 0)) throw new PickleException("data size alignment error")
        if (machinecode == 6 || machinecode == 7) {
          // 32 bits
          constructLongArrayFromUInt32(machinecode, data)
        }
        else {
          // 64 bits
          constructLongArrayFromUInt64(machinecode, data)
        }
      case 'f' => // floating point 4 -> float[]
        if (machinecode != 14 && machinecode != 15) throw new PickleException("for f type must be 14/15")
        if (data.length % 4 != 0) throw new PickleException("data size alignment error")
        constructFloatArray(machinecode, data)
      case 'd' => // floating point 8 -> double[]
        if (machinecode != 16 && machinecode != 17) throw new PickleException("for d type must be 16/17")
        if (data.length % 8 != 0) throw new PickleException("data size alignment error")
        constructDoubleArray(machinecode, data)
      case _ =>
        throw new PickleException("invalid array typecode: " + typecode)
    }
  }

  protected def constructIntArrayFromInt32(machinecode: Int, data: Array[Byte]): Array[Int] = {
    val result = new Array[Int](data.length / 4)
    val bigendian = new Array[Byte](4)
    for (i <- 0 until data.length / 4) {
      if (machinecode == 8) result(i) = PickleUtils.bytes_to_integer(data, i * 4, 4)
      else {
        // big endian, swap
        bigendian(0) = data(3 + i * 4)
        bigendian(1) = data(2 + i * 4)
        bigendian(2) = data(1 + i * 4)
        bigendian(3) = data(0 + i * 4)
        result(i) = PickleUtils.bytes_to_integer(bigendian)
      }
    }
    result
  }

  protected def constructLongArrayFromUInt32(machinecode: Int, data: Array[Byte]): Array[Long] = {
    val result = new Array[Long](data.length / 4)
    val bigendian = new Array[Byte](4)
    for (i <- 0 until data.length / 4) {
      if (machinecode == 6) result(i) = PickleUtils.bytes_to_uint(data, i * 4)
      else {
        // big endian, swap
        bigendian(0) = data(3 + i * 4)
        bigendian(1) = data(2 + i * 4)
        bigendian(2) = data(1 + i * 4)
        bigendian(3) = data(0 + i * 4)
        result(i) = PickleUtils.bytes_to_uint(bigendian, 0)
      }
    }
    result
  }

  protected def constructLongArrayFromUInt64(machinecode: Int, data: Array[Byte]): Array[Long] = {
    // java doesn't have a ulong (unsigned int 64-bits) datatype
    throw new PickleException("unsupported datatype: 64-bits unsigned long")
  }

  protected def constructLongArrayFromInt64(machinecode: Int, data: Array[Byte]): Array[Long] = {
    val result = new Array[Long](data.length / 8)
    val bigendian = new Array[Byte](8)
    for (i <- 0 until data.length / 8) {
      if (machinecode == 12) {
        // little endian can go
        result(i) = PickleUtils.bytes_to_long(data, i * 8)
      }
      else {
        // 13=big endian, swap
        bigendian(0) = data(7 + i * 8)
        bigendian(1) = data(6 + i * 8)
        bigendian(2) = data(5 + i * 8)
        bigendian(3) = data(4 + i * 8)
        bigendian(4) = data(3 + i * 8)
        bigendian(5) = data(2 + i * 8)
        bigendian(6) = data(1 + i * 8)
        bigendian(7) = data(0 + i * 8)
        result(i) = PickleUtils.bytes_to_long(bigendian, 0)
      }
    }
    result
  }

  protected def constructDoubleArray(machinecode: Int, data: Array[Byte]): Array[Double] = {
    val result = new Array[Double](data.length / 8)
    val bigendian = new Array[Byte](8)
    for (i <- 0 until data.length / 8) {
      if (machinecode == 17) result(i) = PickleUtils.bytes_to_double(data, i * 8)
      else {
        // 16=little endian, flip the bytes
        bigendian(0) = data(7 + i * 8)
        bigendian(1) = data(6 + i * 8)
        bigendian(2) = data(5 + i * 8)
        bigendian(3) = data(4 + i * 8)
        bigendian(4) = data(3 + i * 8)
        bigendian(5) = data(2 + i * 8)
        bigendian(6) = data(1 + i * 8)
        bigendian(7) = data(0 + i * 8)
        result(i) = PickleUtils.bytes_to_double(bigendian, 0)
      }
    }
    result
  }

  protected def constructFloatArray(machinecode: Int, data: Array[Byte]): Array[Float] = {
    val result = new Array[Float](data.length / 4)
    val bigendian = new Array[Byte](4)
    for (i <- 0 until data.length / 4) {
      if (machinecode == 15) result(i) = PickleUtils.bytes_to_float(data, i * 4)
      else {
        // 14=little endian, flip the bytes
        bigendian(0) = data(3 + i * 4)
        bigendian(1) = data(2 + i * 4)
        bigendian(2) = data(1 + i * 4)
        bigendian(3) = data(0 + i * 4)
        result(i) = PickleUtils.bytes_to_float(bigendian, 0)
      }
    }
    result
  }

  protected def constructIntArrayFromUShort(machinecode: Int, data: Array[Byte]): Array[Int] = {
    val result = new Array[Int](data.length / 2)
    for (i <- 0 until data.length / 2) {
      val b1 = data(0 + i * 2) & 0xff
      val b2 = data(1 + i * 2) & 0xff
      if (machinecode == 2) result(i) = (b2 << 8) | b1
      else {
        // big endian
        result(i) = (b1 << 8) | b2
      }
    }
    result
  }

  protected def constructShortArraySigned(machinecode: Int, data: Array[Byte]): Array[Short] = {
    val result = new Array[Short](data.length / 2)
    for (i <- 0 until data.length / 2) {
      val b1 = data(0 + i * 2)
      val b2 = data(1 + i * 2)
      if (machinecode == 4) result(i) = ((b2 << 8) | (b1 & 0xff)).toShort
      else {
        // big endian
        result(i) = ((b1 << 8) | (b2 & 0xff)).toShort
      }
    }
    result
  }

  protected def constructShortArrayFromUByte(data: Array[Byte]): Array[Short] = {
    val result = new Array[Short](data.length)
    for (i <- 0 until data.length) {
      result(i) = (data(i) & 0xff).toShort
    }
    result
  }

  protected def constructCharArrayUTF32(machinecode: Int, data: Array[Byte]): Array[Char] = {
    val result = new Array[Char](data.length / 4)
    val bigendian = new Array[Byte](4)
    for (index <- 0 until data.length / 4) {
      if (machinecode == 20) {
        val codepoint = PickleUtils.bytes_to_integer(data, index * 4, 4)
        val cc = Character.toChars(codepoint)
        if (cc.length > 1) throw new PickleException("cannot process UTF-32 character codepoint " + codepoint)
        result(index) = cc(0)
      }
      else {
        // big endian, swap
        bigendian(0) = data(3 + index * 4)
        bigendian(1) = data(2 + index * 4)
        bigendian(2) = data(1 + index * 4)
        bigendian(3) = data(index * 4)
        val codepoint = PickleUtils.bytes_to_integer(bigendian)
        val cc = Character.toChars(codepoint)
        if (cc.length > 1) throw new PickleException("cannot process UTF-32 character codepoint " + codepoint)
        result(index) = cc(0)
      }
    }
    result
  }

  protected def constructCharArrayUTF16(machinecode: Int, data: Array[Byte]): Array[Char] = {
    val result = new Array[Char](data.length / 2)
    val bigendian = new Array[Byte](2)
    for (index <- 0 until data.length / 2) {
      if (machinecode == 18) result(index) = PickleUtils.bytes_to_integer(data, index * 2, 2).toChar
      else {
        // big endian, swap
        bigendian(0) = data(1 + index * 2)
        bigendian(1) = data(0 + index * 2)
        result(index) = PickleUtils.bytes_to_integer(bigendian).toChar
      }
    }
    result
  }
}