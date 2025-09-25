package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import net.razorvine.pickle.PickleException
import java.io.UnsupportedEncodingException
import java.util
import scala.jdk.CollectionConverters.*
/**
 * Creates byte arrays (byte[]).
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
class ByteArrayConstructor extends IObjectConstructor {
  @throws[PickleException]
  override def construct(args: Array[AnyRef]): AnyRef = {
    // args for bytearray constructor: [ String string, String encoding ]
    // args for bytearray constructor (from python3 bytes): [ ArrayList<Number> ] or just [byte[]] (when it uses BINBYTES opcode)
    // or, zero arguments: empty bytearray.
    if (args.length > 2) throw new PickleException("invalid pickle data for bytearray; expected 0, 1 or 2 args, got " + args.length)
    if (args.length == 0) return new Array[Byte](0)
    if (args.length == 1) {
      if (args(0).isInstanceOf[Array[Byte]]) return args(0)
      @SuppressWarnings(Array("unchecked")) val values = args(0).asInstanceOf[util.ArrayList[Number]]
      val data = new Array[Byte](values.size)
      for (i <- 0 until data.length) {
        data(i) = values.get(i).byteValue
      }
      data
    }
    else {
      val data = args(0).asInstanceOf[String]
      var encoding = args(1).asInstanceOf[String]
      if (encoding.startsWith("latin-")) encoding = "ISO-8859-" + encoding.substring(6)
      try data.getBytes(encoding)
      catch {
        case e: UnsupportedEncodingException =>
          throw new PickleException("error creating bytearray: " + e)
      }
    }
  }
}