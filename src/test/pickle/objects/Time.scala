package net.razorvine.pickle.objects

import java.io.Serializable
import java.util.Calendar

/**
 * Helper class to mimic the datetime.time Python object (holds a hours/minute/seconds/microsecond time).
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
@SerialVersionUID(2325820650757621315L)
class Time extends Serializable {
  final var hours = 0
  final var minutes = 0
  final var seconds = 0
  final var microseconds = 0

  def this(h: Int, m: Int, s: Int, microsecs: Int) ={
    this()
    hours = h
    minutes = m
    seconds = s
    microseconds = microsecs
  }

  def this(milliseconds: Long)

  override def toString: String = String.format("Time: %d hours, %d minutes, %d seconds, %d microseconds", hours, minutes, seconds, microseconds)

  override def hashCode: Int = {
    val prime = 31
    var result = 1
    result = prime * result + hours
    result = prime * result + microseconds
    result = prime * result + minutes
    result = prime * result + seconds
    result
  }

  override def equals(obj: AnyRef): Boolean = {
    if (this eq obj) return true
    if (obj == null) return false
    if (!obj.isInstanceOf[Time]) return false
    val other = obj.asInstanceOf[Time]
    hours == other.hours && minutes == other.minutes && seconds == other.seconds && microseconds == other.microseconds
  }
}