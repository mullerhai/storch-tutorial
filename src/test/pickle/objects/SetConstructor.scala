package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import java.util
import scala.jdk.CollectionConverters.*
/**
 * This object constructor creates sets.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
class SetConstructor extends IObjectConstructor {
  override def construct(args: Array[AnyRef]): AnyRef = {
    // create a HashSet, args=arraylist of stuff to put in it
    @SuppressWarnings(Array("unchecked")) val data = args(0).asInstanceOf[util.ArrayList[AnyRef]]
    new util.HashSet[AnyRef](data)
  }
}