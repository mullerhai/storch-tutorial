package net.razorvine.pickle.objects

import java.io.Serializable
import java.text.NumberFormat
import java.util.Locale

/**
 * Helper class to mimic the datetime.timedelta Python object (holds a days/seconds/microsec time difference).
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
@SerialVersionUID(7655189815303876847L)
class TimeDelta(val days: Int, val seconds: Int, val microseconds: Int) extends Serializable {

  final var total_seconds = .0
  this.total_seconds = days * 86400 + seconds + microseconds / 1000000.0
  
  override def toString: String = {
    val nf = NumberFormat.getInstance(Locale.UK)
    nf.setGroupingUsed(false)
    nf.setMaximumFractionDigits(6)
    val floatsecs = nf.format(total_seconds)
    String.format("Timedelta: %d days, %d seconds, %d microseconds (total: %s seconds)", days, seconds, microseconds, floatsecs)
  }

  override def hashCode: Int = {
    val prime = 31
    var result = 1
    result = prime * result + days
    result = prime * result + microseconds
    result = prime * result + seconds
    val temp = java.lang.Double.doubleToLongBits(total_seconds)
    result = prime * result + (temp ^ (temp >>> 32)).toInt
    result
  }

  override def equals(obj: AnyRef): Boolean = {
    if (this eq obj) return true
    if (obj == null) return false
    if (!obj.isInstanceOf[TimeDelta]) return false
    val other = obj.asInstanceOf[TimeDelta]
    days == other.days && seconds == other.seconds && microseconds == other.microseconds && total_seconds == other.total_seconds
  }
}