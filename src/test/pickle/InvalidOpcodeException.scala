package net.razorvine.pickle

/**
 * Exception thrown when the unpickler encounters an invalid opcode.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
@SerialVersionUID(-7691944009311968713L)
class InvalidOpcodeException extends PickleException {
  def this(message: String, cause: Throwable)= {
    this()
    super (message, cause)
  }

  def this(message: String) ={
    this()
    super (message)
  }
}