package net.razorvine.pickle

import java.io.Serializable
import java.util
import java.util.Collections

/**
 * Helper type that represents the unpickler working stack.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
@SerialVersionUID(5032718425413805422L)
class UnpickleStack extends Serializable {
  stack = new util.ArrayList[AnyRef]
  MARKER = new AnyRef // any new unique object
  final private var stack: util.ArrayList[AnyRef] = null
  final protected var MARKER: AnyRef = null

  def add(o: AnyRef): Unit = {
    this.stack.add(o)
  }

  def add_mark(): Unit = {
    this.stack.add(this.MARKER)
  }

  def pop: AnyRef = {
    val size = this.stack.size
    val result = this.stack.get(size - 1)
    this.stack.remove(size - 1)
    result
  }

  def pop_all_since_marker: util.List[AnyRef] = {
    val result = new util.ArrayList[AnyRef]
    var o = pop
    while (o ne this.MARKER) {
      result.add(o)
      o = pop
    }
    result.trimToSize()
    Collections.reverse(result)
    result
  }

  def peek: AnyRef = this.stack.get(this.stack.size - 1)

  def trim(): Unit = {
    this.stack.trimToSize()
  }

  def size: Int = this.stack.size

  def clear(): Unit = {
    this.stack.clear()
    this.stack.trimToSize()
  }
}