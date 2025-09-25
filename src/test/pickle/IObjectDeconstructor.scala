package net.razorvine.pickle

/**
 * Interface for Object Deconstructors that are used by the pickler
 * to create instances of non-primitive or custom classes.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
trait IObjectDeconstructor {
  /**
   * Get the module of the class being pickled
   */
  def getModule: String

  /**
   * Get the name of the class being pickled
   */
  def getName: String

  /**
   * Deconstructs the arugment of an object. The given args will be used as parameters for the constructor during unpickling.
   */
  @throws[PickleException]
  def deconstruct(obj: AnyRef): Array[AnyRef]
}