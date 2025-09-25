package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import net.razorvine.pickle.PickleException

/**
 * This object constructor creates ClassDicts (for unsupported classes).
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
class ClassDictConstructor(private[objects] val module: String, private[objects] val name: String) extends IObjectConstructor {
  override def construct(args: Array[AnyRef]): AnyRef = {
    if (args.length > 0) throw new PickleException("expected zero arguments for construction of ClassDict (for " + module + "." + name + "). This happens when an unsupported/unregistered class is being unpickled that requires construction arguments. Fix it by registering a custom IObjectConstructor for this class.")
    new ClassDict(module, name)
  }
}