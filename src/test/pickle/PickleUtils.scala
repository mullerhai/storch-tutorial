package net.razorvine.pickle

import java.io.IOException
import java.io.InputStream
import java.io.UnsupportedEncodingException
import java.math.BigInteger
import scala.jdk.CollectionConverters.*
/**
 * Utility stuff for dealing with pickle data streams.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
object PickleUtils {
  /**
   * read a line of text, excluding the terminating LF char
   */
  @throws[IOException]
  def readline(input: InputStream): String = readline(input, false)

  /**
   * read a line of text, possibly including the terminating LF char
   */
  @throws[IOException]
  def readline(input: InputStream, includeLF: Boolean): String = {
    val sb = new StringBuilder
    while (true) {
      val c = input.read
      if (c == -1) {
        if (sb.length == 0) throw new IOException("premature end of file")
        break //todo: break is not supported
      }
      if (c != '\n' || includeLF) sb.append(c.toChar)
      if (c == '\n') break //todo: break is not supported
    }
    sb.toString
  }

  /**
   * read a single unsigned byte
   */
  @throws[IOException]
  def readbyte(input: InputStream): Short = {
    val b = input.read
    b.toShort
  }

  /**
   * read a number of signed bytes
   */
  @throws[IOException]
  def readbytes(input: InputStream, n: Int): Array[Byte] = {
    val buffer = new Array[Byte](n)
    readbytes_into(input, buffer, 0, n)
    buffer
  }

  /**
   * read a number of signed bytes
   */
  @throws[IOException]
  def readbytes(input: InputStream, n: Long): Array[Byte] = {
    if (n > Integer.MAX_VALUE) throw new PickleException("pickle too large, can't read more than maxint")
    readbytes(input, n.toInt)
  }

  /**
   * read a number of signed bytes into the specified location in an existing byte array
   */
  @throws[IOException]
  def readbytes_into(input: InputStream, buffer: Array[Byte], offset: Int, length: Int): Unit = {
    while (length > 0) {
      val read = input.read(buffer, offset, length)
      if (read == -1) throw new IOException("expected more bytes in input stream")
      offset += read
      length -= read
    }
  }

  /**
   * Convert a couple of bytes into the corresponding integer number.
   * Can deal with 2-bytes unsigned int and 4-bytes signed int.
   */
  def bytes_to_integer(bytes: Array[Byte]): Int = bytes_to_integer(bytes, 0, bytes.length)

  def bytes_to_integer(bytes: Array[Byte], offset: Int, size: Int): Int = {
    // this operates on little-endian bytes
    if (size == 2) {
      // 2-bytes unsigned int
      var i = bytes(1 + offset) & 0xff
      i <<= 8
      i |= bytes(0 + offset) & 0xff
      i
    }
    else if (size == 4) {
      // 4-bytes signed int
      var i = bytes(3 + offset)
      i <<= 8
      i |= bytes(2 + offset) & 0xff
      i <<= 8
      i |= bytes(1 + offset) & 0xff
      i <<= 8
      i |= bytes(0 + offset) & 0xff
      i
    }
    else throw new PickleException("invalid amount of bytes to convert to int: " + size)
  }

  /**
   * Convert 8 little endian bytes into a long
   */
  def bytes_to_long(bytes: Array[Byte], offset: Int): Long = {
    if (bytes.length - offset < 8) throw new PickleException("too few bytes to convert to long")
    var i = bytes(7 + offset) & 0xff
    i <<= 8
    i |= bytes(6 + offset) & 0xff
    i <<= 8
    i |= bytes(5 + offset) & 0xff
    i <<= 8
    i |= bytes(4 + offset) & 0xff
    i <<= 8
    i |= bytes(3 + offset) & 0xff
    i <<= 8
    i |= bytes(2 + offset) & 0xff
    i <<= 8
    i |= bytes(1 + offset) & 0xff
    i <<= 8
    i |= bytes(offset) & 0xff
    i
  }

  /**
   * Convert 4 little endian bytes into an unsigned int (as a long)
   */
  def bytes_to_uint(bytes: Array[Byte], offset: Int): Long = {
    if (bytes.length - offset < 4) throw new PickleException("too few bytes to convert to long")
    var i = bytes(3 + offset) & 0xff
    i <<= 8
    i |= bytes(2 + offset) & 0xff
    i <<= 8
    i |= bytes(1 + offset) & 0xff
    i <<= 8
    i |= bytes(0 + offset) & 0xff
    i
  }

  /**
   * Convert a signed integer to its 4-byte representation. (little endian)
   */
  def integer_to_bytes(i: Int): Array[Byte] = {
    val b = new Array[Byte](4)
    b(0) = (i & 0xff).toByte
    i >>= 8
    b(1) = (i & 0xff).toByte
    i >>= 8
    b(2) = (i & 0xff).toByte
    i >>= 8
    b(3) = (i & 0xff).toByte
    b
  }

  /**
   * Convert a double to its 8-byte representation (big endian).
   */
  def double_to_bytes(d: Double): Array[Byte] = {
    var bits = java.lang.Double.doubleToRawLongBits(d)
    val b = new Array[Byte](8)
    b(7) = (bits & 0xff).toByte
    bits >>= 8
    b(6) = (bits & 0xff).toByte
    bits >>= 8
    b(5) = (bits & 0xff).toByte
    bits >>= 8
    b(4) = (bits & 0xff).toByte
    bits >>= 8
    b(3) = (bits & 0xff).toByte
    bits >>= 8
    b(2) = (bits & 0xff).toByte
    bits >>= 8
    b(1) = (bits & 0xff).toByte
    bits >>= 8
    b(0) = (bits & 0xff).toByte
    b
  }

  /**
   * Convert a big endian 8-byte to a double.
   */
  def bytes_to_double(bytes: Array[Byte], offset: Int): Double = try {
    var result = bytes(0 + offset) & 0xff
    result <<= 8
    result |= bytes(1 + offset) & 0xff
    result <<= 8
    result |= bytes(2 + offset) & 0xff
    result <<= 8
    result |= bytes(3 + offset) & 0xff
    result <<= 8
    result |= bytes(4 + offset) & 0xff
    result <<= 8
    result |= bytes(5 + offset) & 0xff
    result <<= 8
    result |= bytes(6 + offset) & 0xff
    result <<= 8
    result |= bytes(7 + offset) & 0xff
    java.lang.Double.longBitsToDouble(result)
  } catch {
    case x: IndexOutOfBoundsException =>
      throw new PickleException("decoding double: too few bytes")
  }

  /**
   * Convert a big endian 4-byte to a float.
   */
  def bytes_to_float(bytes: Array[Byte], offset: Int): Float = try {
    var result = bytes(0 + offset) & 0xff
    result <<= 8
    result |= bytes(1 + offset) & 0xff
    result <<= 8
    result |= bytes(2 + offset) & 0xff
    result <<= 8
    result |= bytes(3 + offset) & 0xff
    java.lang.Float.intBitsToFloat(result)
  } catch {
    case x: IndexOutOfBoundsException =>
      throw new PickleException("decoding float: too few bytes")
  }

  /**
   * read an arbitrary 'long' number. Returns an int/long/BigInteger as appropriate to hold the number.
   */
  def decode_long(data: Array[Byte]): Number = {
    if (data.length == 0) return 0L
    // first reverse the byte array because pickle stores it little-endian
    val data2 = new Array[Byte](data.length)
    for (i <- 0 until data.length) {
      data2(data.length - i - 1) = data(i)
    }
    val bigint = new BigInteger(data2)
    optimizeBigint(bigint)
  }

  /**
   * encode an arbitrary long number into a byte array (little endian).
   */
  def encode_long(big: BigInteger): Array[Byte] = {
    val data = big.toByteArray
    // reverse the byte array because pickle uses little endian
    val data2 = new Array[Byte](data.length)
    for (i <- 0 until data.length) {
      data2(data.length - i - 1) = data(i)
    }
    data2
  }

  /**
   * Optimize a biginteger, if possible return a long primitive datatype.
   */
  def optimizeBigint(bigint: BigInteger): Number = {
    // final BigInteger MAXINT=BigInteger.valueOf(Integer.MAX_VALUE);
    // final BigInteger MININT=BigInteger.valueOf(Integer.MIN_VALUE);
    val MAXLONG = BigInteger.valueOf(Long.MaxValue)
    val MINLONG = BigInteger.valueOf(Long.MinValue)
    bigint.signum match {
      case 0 =>
        return 0L
      case 1 => // >0

        // if(bigint.compareTo(MAXINT)<=0) return bigint.intValue();
        if (bigint.compareTo(MAXLONG) <= 0) return bigint.longValue
      case -(1) => // <0

        // if(bigint.compareTo(MININT)>=0) return bigint.intValue();
        if (bigint.compareTo(MINLONG) >= 0) return bigint.longValue
    }
    bigint
  }

  /**
   * Construct a String from the given bytes where these are directly
   * converted to the corresponding chars, without using a given character
   * encoding
   */
  def rawStringFromBytes(data: Array[Byte]): String = {
    val str = new StringBuilder(data.length)
    for (b <- data) {
      str.append((b & 0xff).toChar)
    }
    str.toString
  }

  /**
   * Convert a string to a byte array, no encoding is used. String must only contain characters less than 256.
   */
  @throws[IOException]
  def str2bytes(str: String): Array[Byte] = {
    val b = new Array[Byte](str.length)
    for (i <- 0 until str.length) {
      val c = str.charAt(i)
      if (c > 255) throw new UnsupportedEncodingException("string contained a char > 255, cannot convert to bytes")
      b(i) = c.toByte
    }
    b
  }

  /**
   * Decode a string with possible escaped char sequences in it (\x??).
   */
  def decode_escaped(str: String): String = {
    if (str.indexOf('\\') == -1) return str
    val sb = new lang.StringBuilder(str.length)
    var i = 0
    while (i < str.length) {
      val c = str.charAt(i)
      if (c == '\\') {
        // possible escape sequence
        var c2 = str.charAt({
          i += 1; i
        })
        c2 match {
          case '\\' =>
            // double-escaped '\\'--> '\'
            sb.append(c)
          case 'x' =>
            // hex escaped "\x??"
            val h1 = str.charAt({
              i += 1; i
            })
            val h2 = str.charAt({
              i += 1; i
            })
            c2 = Integer.parseInt("" + h1 + h2, 16).toChar
            sb.append(c2)
          case 'n' =>
            sb.append('\n')
          case 'r' =>
            sb.append('\r')
          case 't' =>
            sb.append('\t')
          case '\'' =>
            sb.append('\'') // sometimes occurs in protocol level 0 strings
          case _ =>
            if (str.length > 80) str = str.substring(0, 80)
            throw new PickleException("invalid escape sequence char '" + c2 + "' in string \"" + str + " [...]\" (possibly truncated)")
        }
      }
      else sb.append(str.charAt(i))
      i += 1
    }
    sb.toString
  }

  /**
   * Decode a string with possible escaped unicode in it (code point 20ac)
   */
  def decode_unicode_escaped(str: String): String = {
    if (str.indexOf('\\') == -1) return str
    val sb = new StringBuilder(str.length)
    var i = 0
    while (i < str.length) {
      val c = str.charAt(i)
      if (c == '\\') {
        // possible escape sequence
        var c2 = str.charAt({
          i += 1; i
        })
        c2 match {
          case '\\' =>
            // double-escaped '\\'--> '\'
            sb.append(c)
          case 'u' =>

            // hex escaped unicode "\u20ac"
            val h1 = str.charAt({
              i += 1; i
            })
            val h2 = str.charAt({
              i += 1; i
            })
            val h3 = str.charAt({
              i += 1; i
            })
            val h4 = str.charAt({
              i += 1; i
            })
            c2 = Integer.parseInt("" + h1 + h2 + h3 + h4, 16).toChar
            sb.append(c2)
          case 'U' =>

            // hex escaped unicode "\U0001f455"
            val h1 = str.charAt({
              i += 1; i
            })
            val h2 = str.charAt({
              i += 1; i
            })
            val h3 = str.charAt({
              i += 1; i
            })
            val h4 = str.charAt({
              i += 1; i
            })
            val h5 = str.charAt({
              i += 1; i
            })
            val h6 = str.charAt({
              i += 1; i
            })
            val h7 = str.charAt({
              i += 1; i
            })
            val h8 = str.charAt({
              i += 1; i
            })
            val encoded = "" + h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8
            val s = new String(Character.toChars(Integer.parseInt(encoded, 16)))
            sb.append(s)
          case 'n' =>
            sb.append('\n')
          case 'r' =>
            sb.append('\r')
          case 't' =>
            sb.append('\t')
          case _ =>
            if (str.length > 80) str = str.substring(0, 80)
            throw new PickleException("invalid escape sequence char '" + c2 + "' in string \"" + str + " [...]\" (possibly truncated)")
        }
      }
      else sb.append(str.charAt(i))
      i += 1
    }
    sb.toString
  }
}