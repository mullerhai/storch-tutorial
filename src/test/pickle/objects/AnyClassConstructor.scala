package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import net.razorvine.pickle.PickleException
import java.lang.reflect.Constructor
import java.math.BigDecimal
import scala.jdk.CollectionConverters.*
/**
 * This object constructor uses reflection to create instances of any given class.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
class AnyClassConstructor(private val typez: Class[_]) extends IObjectConstructor {
  override def construct(args: Array[AnyRef]): AnyRef = try {
    val paramtypes = new Array[Class[_]](args.length)
    for (i <- 0 until args.length) {
      paramtypes(i) = args(i).getClass
    }
    val cons = typez.getConstructor(paramtypes)
    // special case BigDecimal("NaN") which is not supported in Java, return this as Double.NaN
    if ((typez eq classOf[BigDecimal]) && args.length == 1) {
      val nan = args(0).asInstanceOf[String]
      if (nan.equalsIgnoreCase("nan")) return Double.NaN
    }
    cons.newInstance(args)
  } catch {
    case x: Exception =>
      throw new PickleException("problem construction object: " + x)
  }
}