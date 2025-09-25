package net.razorvine.pickle

import java.util

/**
 * Exception thrown that represents a certain Python exception.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
@SerialVersionUID(4884843316742683086L)
class PythonException extends RuntimeException {
  var _pyroTraceback: String = null
  var pythonExceptionType: String = null

  def this(message: String, cause: Throwable)= {
    this()
    super (message, cause)
  }

  def this(message: String)= {
    this()
    super (message)
  }

  def this(cause: Throwable)= {
    this()
    super (cause)
  }

  // special constructor for UnicodeDecodeError
  def this(encoding: String, data: Array[Byte], i1: Integer, i2: Integer, message: String)= {
    this()
    super ("UnicodeDecodeError: " + encoding + ": " + message)
  }

  /**
   * called by the unpickler to restore state
   */
  def __setstate__(args: util.HashMap[String, AnyRef]): Unit = {
    val tb = args.get("_pyroTraceback")
    // if the traceback is a list of strings, create one string from it
    if (tb.isInstanceOf[util.List[_]]) {
      val sb = new StringBuilder
      import scala.jdk.CollectionConverters._
//      import scala.collection.JavaConversions._
      for (line <- tb.asInstanceOf[util.List[_]]) {
        sb.append(line)
      }
      _pyroTraceback = sb.toString
    }
    else _pyroTraceback = tb.asInstanceOf[String]
  }
}