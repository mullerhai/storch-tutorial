package net.razorvine.pickle.objects

import java.util
import scala.jdk.CollectionConverters.*
/**
 * A dictionary containing just the fields of the class.
 */
@SerialVersionUID(6157715596627049511L)
class ClassDict(modulename: String, classname: String) extends util.HashMap[String, AnyRef] {
  final private var classname: String = null
  if (modulename == null)  this.classname = classname
  else this.classname = modulename + "." + classname
  this.put("__class__", this.classname)
  

  /**
   * for the unpickler to restore state
   */
  def __setstate__(values: util.HashMap[String, AnyRef]): Unit = {
    this.clear()
    this.put("__class__", this.classname)
    this.putAll(values)
  }

  /**
   * retrieve the (python) class name of the object that was pickled.
   */
  def getClassName: String = this.classname
}