package net.razorvine.pickle

import java.io.IOException
import java.io.OutputStream

/**
 * Interface for Object Picklers used by the pickler, to pickle custom classes. 
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
trait IObjectPickler {
  /**
   * Pickle an object.
   */
  @throws[PickleException]
  @throws[IOException]
  def pickle(o: AnyRef, out: OutputStream, currentPickler: Pickler): Unit
}