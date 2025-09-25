package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import net.razorvine.pickle.PickleException
import java.lang.reflect.Method
import scala.jdk.CollectionConverters.*
/**
 * This constructor is called by the helper methods that pickle protocol 0
 * uses from the python copy_reg module to reconstruct c objects.
 */
class Reconstructor extends IObjectConstructor {
  override def construct(args: Array[AnyRef]): AnyRef = {
    if (args.length != 3) throw new PickleException("invalid pickle data; expecting 3 args to copy_reg reconstructor but recieved " + args.length)
    val reconstructor = args(0)
    try {
      val reconstruct = reconstructor.getClass.getMethod("reconstruct", classOf[AnyRef], classOf[AnyRef])
      reconstruct.invoke(reconstructor, args(1), args(2))
    } catch {
      case e: Exception =>
        throw new PickleException("failed to reconstruct()", e)
    }
  }
}