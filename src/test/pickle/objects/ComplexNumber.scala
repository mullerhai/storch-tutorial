package net.razorvine.pickle.objects

import java.io.Serializable
import scala.jdk.CollectionConverters.*
/**
 * An immutable Complex Number class.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
@SerialVersionUID(4668080260997226513L)
object ComplexNumber {
  def add(c1: ComplexNumber, c2: ComplexNumber) = new ComplexNumber(c1.r + c2.r, c1.i + c2.i)

  def subtract(c1: ComplexNumber, c2: ComplexNumber) = new ComplexNumber(c1.r - c2.r, c1.i - c2.i)

  def multiply(c1: ComplexNumber, c2: ComplexNumber) = new ComplexNumber(c1.r * c2.r - c1.i * c2.i, c1.r * c2.i + c1.i * c2.r)

  def divide(c1: ComplexNumber, c2: ComplexNumber) = new ComplexNumber((c1.r * c2.r + c1.i * c2.i) / (c2.r * c2.r + c2.i * c2.i), (c1.i * c2.r - c1.r * c2.i) / (c2.r * c2.r + c2.i * c2.i))
}

@SerialVersionUID(4668080260997226513L)
class ComplexNumber extends Serializable {
  final private var r = .0 // real
  final private var i = .0 // imaginary

  def this(rr: Double, ii: Double) ={
    this()
    r = rr
    i = ii
  }

  def this(rr: Double, ii: Double) = {
    this()
    r = rr
    i = ii
  }

  override def toString: String = {
    val sb = new StringBuilder().append(r)
    if (i >= 0) sb.append('+')
    sb.append(i).append('i').toString
  }

  def getReal: Double = r

  def getImaginary: Double = i

  def magnitude: Double = Math.sqrt(r * r + i * i)

  def add(other: ComplexNumber): ComplexNumber = ComplexNumber.add(this, other)

  def subtract(other: ComplexNumber): ComplexNumber = ComplexNumber.subtract(this, other)

  def multiply(other: ComplexNumber): ComplexNumber = ComplexNumber.multiply(this, other)

  override def equals(o: AnyRef): Boolean = {
    if (!o.isInstanceOf[ComplexNumber]) return false
    val other = o.asInstanceOf[ComplexNumber]
    r == other.r && i == other.i
  }

  override def hashCode: Int = java.lang.Double.valueOf(r).hashCode ^ java.lang.Double.valueOf(i).hashCode
}