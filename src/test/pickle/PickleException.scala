package net.razorvine.pickle

/**
 * Exception thrown when something goes wrong with pickling or unpickling.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
@SerialVersionUID(-5870448664938735316L)
class PickleException extends RuntimeException {
  def this(message: String, cause: Throwable) ={
    this()
    super (message, cause)
  }

  def this(message: String) ={
    this()
    super (message)
  }
}