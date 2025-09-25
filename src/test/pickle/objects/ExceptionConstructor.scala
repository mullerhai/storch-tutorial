package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import net.razorvine.pickle.PickleException
import java.lang.reflect.Constructor
import java.lang.reflect.Field
import scala.jdk.CollectionConverters.*
/**
 * This creates Python Exception instances.
 * It keeps track of the original Python exception type name as well.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
class ExceptionConstructor(private val `type`: Class[_], module: String, name: String) extends IObjectConstructor {

  final private var pythonExceptionType: String = null
  if (module != null) pythonExceptionType = module + "." + name
  else pythonExceptionType = name
  
  override def construct(args: Array[AnyRef]): AnyRef = try {
    if (pythonExceptionType != null) {
      // put the python exception type somewhere in the message
      if (args == null || args.length == 0) args = Array[String]("[" + pythonExceptionType + "]")
      else {
        val msg = "[" + pythonExceptionType + "] " + args(0)
        args = Array[String](msg)
      }
    }
    val paramtypes = new Array[Class[_]](args.length)
    for (i <- 0 until args.length) {
      paramtypes(i) = args(i).getClass
    }
    val cons = `type`.getConstructor(paramtypes)
    val ex = cons.newInstance(args)
    try {
      val prop = ex.getClass.getField("pythonExceptionType")
      prop.set(ex, pythonExceptionType)
    } catch {
      case x: NoSuchFieldException =>


      // meh.
    }
    ex
  } catch {
    case x: Exception =>
      throw new PickleException("problem construction object: " + x)
  }
}