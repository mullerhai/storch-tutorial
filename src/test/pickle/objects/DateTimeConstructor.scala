package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import net.razorvine.pickle.PickleException
import java.util.Calendar
import java.util.GregorianCalendar
import java.util.TimeZone
import scala.jdk.CollectionConverters.*
/**
 * This constructor can create various datetime related objects.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
object DateTimeConstructor {
  val DATETIME = 1
  val DATE = 2
  val TIME = 3
  val TIMEDELTA = 4
}

class DateTimeConstructor(private val pythontype: Int) extends IObjectConstructor {
  override def construct(args: Array[AnyRef]): AnyRef = {
    if (this.pythontype == DateTimeConstructor.DATE) return createDate(args)
    if (this.pythontype == DateTimeConstructor.TIME) return createTime(args)
    if (this.pythontype == DateTimeConstructor.DATETIME) return createDateTime(args)
    if (this.pythontype == DateTimeConstructor.TIMEDELTA) return createTimedelta(args)
    throw new PickleException("invalid object type")
  }

  private def createTimedelta(args: Array[AnyRef]) = {
    // python datetime.timedelta -> dt.TimeDelta
    // args is a tuple of 3 ints: days,seconds,microseconds
    if (args.length != 3) throw new PickleException("invalid pickle data for timedelta; expected 3 args, got " + args.length)
    val days = args(0).asInstanceOf[Number].intValue
    val seconds = args(1).asInstanceOf[Number].intValue
    val micro = args(2).asInstanceOf[Number].intValue
    new TimeDelta(days, seconds, micro)
  }

  private def createDateTime(args: Array[AnyRef]): Calendar = {
    // python datetime.time --> GregorianCalendar
    // args is 10 bytes: yhi, ylo, month, day, hour, minute, second, ms1, ms2, ms3
    // (can be String or byte[])
    // alternate constructor is with 7/8 integer arguments: year, month, day, hour, minute, second, microseconds, [timezone]
    if (args.length == 7 || args.length == 8) {
      val year = args(0).asInstanceOf[Integer]
      val month = args(1).asInstanceOf[Integer] - 1 // in java month starts with 0...
      val day = args(2).asInstanceOf[Integer]
      val hour = args(3).asInstanceOf[Integer]
      val minute = args(4).asInstanceOf[Integer]
      val second = args(5).asInstanceOf[Integer]
      val microsec = args(6).asInstanceOf[Integer]
      var tz: TimeZone = null
      if (args.length == 8) tz = args(7).asInstanceOf[TimeZone]
      val cal = new GregorianCalendar(year, month, day, hour, minute, second)
      cal.set(Calendar.MILLISECOND, microsec / 1000)
      if (tz != null) cal.setTimeZone(tz)
      return cal
    }
    if (args.length != 1 && args.length != 2) throw new PickleException("invalid pickle data for datetime; expected 1, 2, 7 or 8 args, got " + args.length)
    var yhi = 0
    var ylo = 0
    var month = 0
    var day = 0
    var hour = 0
    var minute = 0
    var second = 0
    var microsec = 0
    if (args(0).isInstanceOf[String]) {
      val params = args(0).asInstanceOf[String]
      if (params.length != 10) throw new PickleException("invalid pickle data for datetime; expected arg of length 10, got length " + params.length)
      yhi = params.charAt(0)
      ylo = params.charAt(1)
      month = params.charAt(2) - 1 // blargh: months start at 0 in Java
      day = params.charAt(3)
      hour = params.charAt(4)
      minute = params.charAt(5)
      second = params.charAt(6)
      val ms1 = params.charAt(7)
      val ms2 = params.charAt(8)
      val ms3 = params.charAt(9)
      microsec = ((ms1 << 8) | ms2) << 8 | ms3
    }
    else {
      val params = args(0).asInstanceOf[Array[Byte]]
      if (params.length != 10) throw new PickleException("invalid pickle data for datetime; expected arg of length 10, got length " + params.length)
      yhi = params(0) & 0xff
      ylo = params(1) & 0xff
      month = (params(2) & 0xff) - 1 // blargh: months start at 0 in java
      day = params(3) & 0xff
      hour = params(4) & 0xff
      minute = params(5) & 0xff
      second = params(6) & 0xff
      val ms1 = params(7) & 0xff
      val ms2 = params(8) & 0xff
      val ms3 = params(9) & 0xff
      microsec = ((ms1 << 8) | ms2) << 8 | ms3
    }
    val cal = new GregorianCalendar(yhi * 256 + ylo, month, day, hour, minute, second)
    cal.set(Calendar.MILLISECOND, microsec / 1000)
    if (args.length == 2) {
      // Timezone passed as the second constructor arg in pickle protocal 0
      if (args(1).isInstanceOf[TimeZone]) cal.setTimeZone(args(1).asInstanceOf[TimeZone])
      else if (args(1).isInstanceOf[Tzinfo]) cal.setTimeZone(args(1).asInstanceOf[Tzinfo].getTimeZone)
      else throw new PickleException("invalid pickle data for datetime; expected arg 2 to be a Tzinfo or TimeZone")
    }
    cal
  }

  private def createTime(args: Array[AnyRef]): Time = {
    // python datetime.time --> Time object
    // args is 6 bytes: hour, minute, second, ms1,ms2,ms3  (String or byte[])
    // alternate constructor passes 4 integers args: hour, minute, second, microsecond)
    if (args.length == 4) {
      val hour = args(0).asInstanceOf[Integer]
      val minute = args(1).asInstanceOf[Integer]
      val second = args(2).asInstanceOf[Integer]
      val microsec = args(3).asInstanceOf[Integer]
      return new Time(hour, minute, second, microsec)
    }
    if (args.length != 1) throw new PickleException("invalid pickle data for time; expected 1 or 4 args, got " + args.length)
    var hour = 0
    var minute = 0
    var second = 0
    var microsec = 0
    if (args(0).isInstanceOf[String]) {
      val params = args(0).asInstanceOf[String]
      if (params.length != 6) throw new PickleException("invalid pickle data for time; expected arg of length 6, got length " + params.length)
      hour = params.charAt(0)
      minute = params.charAt(1)
      second = params.charAt(2)
      val ms1 = params.charAt(3)
      val ms2 = params.charAt(4)
      val ms3 = params.charAt(5)
      microsec = ((ms1 << 8) | ms2) << 8 | ms3
    }
    else {
      val params = args(0).asInstanceOf[Array[Byte]]
      if (params.length != 6) throw new PickleException("invalid pickle data for datetime; expected arg of length 6, got length " + params.length)
      hour = params(0) & 0xff
      minute = params(1) & 0xff
      second = params(2) & 0xff
      val ms1 = params(3) & 0xff
      val ms2 = params(4) & 0xff
      val ms3 = params(5) & 0xff
      microsec = ((ms1 << 8) | ms2) << 8 | ms3
    }
    new Time(hour, minute, second, microsec)
  }

  private def createDate(args: Array[AnyRef]): Calendar = {
    // python datetime.date --> GregorianCalendar
    // args is a string of 4 bytes yhi, ylo, month, day (String or byte[])
    // alternatively, args is 3 values year, month, day
    if (args.length == 3) {
      val year = args(0).asInstanceOf[Integer]
      val month = args(1).asInstanceOf[Integer] - 1 // blargh: months start at 0 in Java
      val day = args(2).asInstanceOf[Integer]
      return new GregorianCalendar(year, month, day)
    }
    if (args.length != 1) throw new PickleException("invalid pickle data for date; expected 1 arg, got " + args.length)
    var yhi = 0
    var ylo = 0
    var month = 0
    var day = 0
    if (args(0).isInstanceOf[String]) {
      val params = args(0).asInstanceOf[String]
      if (params.length != 4) throw new PickleException("invalid pickle data for date; expected arg of length 4, got length " + params.length)
      yhi = params.charAt(0)
      ylo = params.charAt(1)
      month = params.charAt(2) - 1 // blargh: months start at 0 in Java
      day = params.charAt(3)
    }
    else {
      val params = args(0).asInstanceOf[Array[Byte]]
      if (params.length != 4) throw new PickleException("invalid pickle data for date; expected arg of length 4, got length " + params.length)
      yhi = params(0) & 0xff
      ylo = params(1) & 0xff
      month = (params(2) & 0xff) - 1 // blargh: months start at 0 in java
      day = params(3) & 0xff
    }
    new GregorianCalendar(yhi * 256 + ylo, month, day)
  }
}