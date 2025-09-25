package net.razorvine.pickle

import net.razorvine.pickle.objects._
import java.io.ByteArrayInputStream
import java.io.IOException
import java.io.InputStream
import java.lang.reflect.Method
import java.math.BigDecimal
import java.math.BigInteger
import java.nio.charset.StandardCharsets
import java.util
import scala.jdk.CollectionConverters.*
/**
 * Unpickles an object graph from a pickle data inputstream. Supports all pickle protocol versions.
 * Maps the python objects on the corresponding java equivalents or similar types.
 * This class is NOT threadsafe! (Don't use the same pickler from different threads)
 *
 * See the README.txt for a table of the type mappings.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
object Unpickler {
  /**
   * Used as return value for {@link Unpickler# dispatch} in the general case (because the object graph is built on the stack)
   */
  protected val NO_RETURN_VALUE = new AnyRef
  /**
   * Registry of object constructors that are used to create the appropriate Java objects for the given Python module.typename references.
   */
  protected var objectConstructors: util.Map[String, IObjectConstructor] = null

  /**
   * Register additional object constructors for custom classes.
   */
  def registerConstructor(module: String, classname: String, constructor: IObjectConstructor): Unit = {
    objectConstructors.put(module + "." + classname, constructor)
  }

  try objectConstructors = new util.HashMap[String, IObjectConstructor]
  objectConstructors.put("__builtin__.complex", new AnyClassConstructor(classOf[ComplexNumber]))
  objectConstructors.put("builtins.complex", new AnyClassConstructor(classOf[ComplexNumber]))
  objectConstructors.put("array.array", new ArrayConstructor)
  objectConstructors.put("array._array_reconstructor", new ArrayConstructor)
  objectConstructors.put("__builtin__.bytearray", new ByteArrayConstructor)
  objectConstructors.put("builtins.bytearray", new ByteArrayConstructor)
  objectConstructors.put("__builtin__.bytes", new ByteArrayConstructor)
  objectConstructors.put("__builtin__.set", new SetConstructor)
  objectConstructors.put("builtins.set", new SetConstructor)
  objectConstructors.put("datetime.datetime", new DateTimeConstructor(DateTimeConstructor.DATETIME))
  objectConstructors.put("datetime.time", new DateTimeConstructor(DateTimeConstructor.TIME))
  objectConstructors.put("datetime.date", new DateTimeConstructor(DateTimeConstructor.DATE))
  objectConstructors.put("datetime.timedelta", new DateTimeConstructor(DateTimeConstructor.TIMEDELTA))
  objectConstructors.put("pytz._UTC", new TimeZoneConstructor(TimeZoneConstructor.UTC))
  objectConstructors.put("pytz._p", new TimeZoneConstructor(TimeZoneConstructor.PYTZ))
  objectConstructors.put("pytz.timezone", new TimeZoneConstructor(TimeZoneConstructor.PYTZ))
  objectConstructors.put("dateutil.tz.tzutc", new TimeZoneConstructor(TimeZoneConstructor.DATEUTIL_TZUTC))
  objectConstructors.put("dateutil.tz.tzfile", new TimeZoneConstructor(TimeZoneConstructor.DATEUTIL_TZFILE))
  objectConstructors.put("dateutil.zoneinfo.gettz", new TimeZoneConstructor(TimeZoneConstructor.DATEUTIL_GETTZ))
  objectConstructors.put("datetime.tzinfo", new TimeZoneConstructor(TimeZoneConstructor.TZINFO))
  objectConstructors.put("decimal.Decimal", new AnyClassConstructor(classOf[BigDecimal]))
  objectConstructors.put("copy_reg._reconstructor", new Reconstructor)
  objectConstructors.put("operator.attrgetter", new OperatorAttrGetterForCalendarTz)
  objectConstructors.put("_codecs.encode", new ByteArrayConstructor) // we're lucky, the bytearray constructor is also able to mimic codecs.encode()
}

class Unpickler {
  memo = new util.HashMap[Integer, AnyRef]
  /**
   * The highest Python Pickle protocol version supported by this library.
   */
  final protected val HIGHEST_PROTOCOL = 5
  /**
   * Internal cache of memoized objects.
   */
  protected var memo: util.Map[Integer, AnyRef] = null
  /**
   * The stack that is used for building the resulting object graph.
   */
  protected var stack: UnpickleStack = null
  /**
   * The stream where the pickle data is read from.
   */
  protected var input: InputStream = null

  /**
   * Read a pickled object representation from the given input stream.
   *
   * @return the reconstituted object hierarchy specified in the file.
   */
  @throws[PickleException]
  @throws[IOException]
  def load(stream: InputStream): AnyRef = {
    stack = new UnpickleStack
    input = stream
    while (true) {
      val key = PickleUtils.readbyte(input)
      if (key == -1) throw new IOException("premature end of file")
      val value = dispatch(key)
      if (value ne Unpickler.NO_RETURN_VALUE) return value
    }
  }

  /**
   * Read a pickled object representation from the given pickle data bytes.
   *
   * @return the reconstituted object hierarchy specified in the file.
   */
  @throws[PickleException]
  @throws[IOException]
  def loads(pickledata: Array[Byte]): AnyRef = load(new ByteArrayInputStream(pickledata))

  /**
   * Close the unpickler and frees the resources such as the unpickle stack and memo table.
   */
  def close(): Unit = {
    if (stack != null) stack.clear()
    if (memo != null) memo.clear()
    if (input != null) try input.close()
    catch {
      case ignored: IOException =>
    }
  }

  /**
   * Buffer support for protocol 5 out of band data
   * If you want to unpickle such pickles, you'll have to subclass the unpickler
   * and override this method to return the buffer data you want.
   */
  @throws[PickleException]
  @throws[IOException]
  protected def next_buffer: AnyRef = throw new PickleException("pickle stream refers to out-of-band data but no user-overridden next_buffer() method is used\n")

  /**
   * Process a single pickle stream opcode.
   */
  @throws[PickleException]
  @throws[IOException]
  protected def dispatch(key: Short): AnyRef = {
    key match {
      case Opcodes.MARK =>
        load_mark()
      case Opcodes.STOP =>
        val value = stack.pop
        stack.clear()
        memo.clear()
        return value // final result value
      case Opcodes.POP =>
        load_pop()
      case Opcodes.POP_MARK =>
        load_pop_mark()
      case Opcodes.DUP =>
        load_dup()
      case Opcodes.FLOAT =>
        load_float()
      case Opcodes.INT =>
        load_int()
      case Opcodes.BININT =>
        load_binint()
      case Opcodes.BININT1 =>
        load_binint1()
      case Opcodes.LONG =>
        load_long()
      case Opcodes.BININT2 =>
        load_binint2()
      case Opcodes.NONE =>
        load_none()
      case Opcodes.PERSID =>
        load_persid()
      case Opcodes.BINPERSID =>
        load_binpersid()
      case Opcodes.REDUCE =>
        load_reduce()
      case Opcodes.STRING =>
        load_string()
      case Opcodes.BINSTRING =>
        load_binstring()
      case Opcodes.SHORT_BINSTRING =>
        load_short_binstring()
      case Opcodes.UNICODE =>
        load_unicode()
      case Opcodes.BINUNICODE =>
        load_binunicode()
      case Opcodes.APPEND =>
        load_append()
      case Opcodes.BUILD =>
        load_build()
      case Opcodes.GLOBAL =>
        load_global()
      case Opcodes.DICT =>
        load_dict()
      case Opcodes.EMPTY_DICT =>
        load_empty_dictionary()
      case Opcodes.APPENDS =>
        load_appends()
      case Opcodes.GET =>
        load_get()
      case Opcodes.BINGET =>
        load_binget()
      case Opcodes.INST =>
        load_inst()
      case Opcodes.LONG_BINGET =>
        load_long_binget()
      case Opcodes.LIST =>
        load_list()
      case Opcodes.EMPTY_LIST =>
        load_empty_list()
      case Opcodes.OBJ =>
        load_obj()
      case Opcodes.PUT =>
        load_put()
      case Opcodes.BINPUT =>
        load_binput()
      case Opcodes.LONG_BINPUT =>
        load_long_binput()
      case Opcodes.SETITEM =>
        load_setitem()
      case Opcodes.TUPLE =>
        load_tuple()
      case Opcodes.EMPTY_TUPLE =>
        load_empty_tuple()
      case Opcodes.SETITEMS =>
        load_setitems()
      case Opcodes.BINFLOAT =>
        load_binfloat()

      // protocol 2
      case Opcodes.PROTO =>
        load_proto()
      case Opcodes.NEWOBJ =>
        load_newobj()
      case Opcodes.EXT1 =>
      case Opcodes.EXT2 =>
      case Opcodes.EXT4 =>
        throw new PickleException("Unimplemented opcode EXT1/EXT2/EXT4 encountered. Don't use extension codes when pickling via copyreg.add_extension() to avoid this error.")
      case Opcodes.TUPLE1 =>
        load_tuple1()
      case Opcodes.TUPLE2 =>
        load_tuple2()
      case Opcodes.TUPLE3 =>
        load_tuple3()
      case Opcodes.NEWTRUE =>
        load_true()
      case Opcodes.NEWFALSE =>
        load_false()
      case Opcodes.LONG1 =>
        load_long1()
      case Opcodes.LONG4 =>
        load_long4()

      // Protocol 3 (Python 3.x)
      case Opcodes.BINBYTES =>
        load_binbytes()
      case Opcodes.SHORT_BINBYTES =>
        load_short_binbytes()

      // Protocol 4 (Python 3.4-3.7)
      case Opcodes.BINUNICODE8 =>
        load_binunicode8()
      case Opcodes.SHORT_BINUNICODE =>
        load_short_binunicode()
      case Opcodes.BINBYTES8 =>
        load_binbytes8()
      case Opcodes.EMPTY_SET =>
        load_empty_set()
      case Opcodes.ADDITEMS =>
        load_additems()
      case Opcodes.FROZENSET =>
        load_frozenset()
      case Opcodes.MEMOIZE =>
        load_memoize()
      case Opcodes.FRAME =>
        load_frame()
      case Opcodes.NEWOBJ_EX =>
        load_newobj_ex()
      case Opcodes.STACK_GLOBAL =>
        load_stack_global()

      // protocol 5 (python 3.8+)
      case Opcodes.BYTEARRAY8 =>
        load_bytearray8()
      case Opcodes.READONLY_BUFFER =>
        load_readonly_buffer()
      case Opcodes.NEXT_BUFFER =>
        load_next_buffer()
      case _ =>
        throw new InvalidOpcodeException("invalid pickle opcode: " + key)
    }
    Unpickler.NO_RETURN_VALUE
  }

  private[pickle] def load_readonly_buffer(): Unit = {

    // this opcode is ignored, we don't distinguish between readonly and read/write buffers
  }

  @throws[PickleException]
  @throws[IOException]
  private[pickle] def load_next_buffer(): Unit = {
    stack.add(next_buffer)
  }

  @throws[IOException]
  private[pickle] def load_bytearray8(): Unit = {
    // this is the same as load_binbytes8 because we make no distinction
    // here between the bytes and bytearray python types
    val len = PickleUtils.bytes_to_long(PickleUtils.readbytes(input, 8), 0)
    stack.add(PickleUtils.readbytes(input, len))
  }

  private[pickle] def load_build(): Unit = {
    val args = stack.pop
    val target = stack.peek
    try {
      val setStateMethod = target.getClass.getMethod("__setstate__", args.getClass)
      setStateMethod.invoke(target, args)
    } catch {
      case e: Exception =>
        throw new PickleException("failed to __setstate__()", e)
    }
  }

  @throws[IOException]
  private[pickle] def load_proto(): Unit = {
    val proto = PickleUtils.readbyte(input)
    if (proto < 0 || proto > HIGHEST_PROTOCOL) throw new PickleException("unsupported pickle protocol: " + proto)
  }

  private[pickle] def load_none(): Unit = {
    stack.add(null)
  }

  private[pickle] def load_false(): Unit = {
    stack.add(false)
  }

  private[pickle] def load_true(): Unit = {
    stack.add(true)
  }

  @throws[IOException]
  private[pickle] def load_int(): Unit = {
    val data = PickleUtils.readline(input, true)
    var `val`: AnyRef = null
    if (data == Opcodes.FALSE.substring(1)) `val` = false
    else if (data == Opcodes.TRUE.substring(1)) `val` = true
    else {
      val number = data.substring(0, data.length - 1)
      try `val` = Integer.parseInt(number, 10)
      catch {
        case x: NumberFormatException =>

          // hmm, integer didn't work.. is it perhaps an int from a 64-bit python? so try long:
          `val` = Long.parseLong(number, 10)
      }
    }
    stack.add(`val`)
  }

  @throws[IOException]
  private[pickle] def load_binint(): Unit = {
    val integer = PickleUtils.bytes_to_integer(PickleUtils.readbytes(input, 4))
    stack.add(integer)
  }

  @throws[IOException]
  private[pickle] def load_binint1(): Unit = {
    stack.add(PickleUtils.readbyte(input).toInt)
  }

  @throws[IOException]
  private[pickle] def load_binint2(): Unit = {
    val integer = PickleUtils.bytes_to_integer(PickleUtils.readbytes(input, 2))
    stack.add(integer)
  }

  @throws[IOException]
  private[pickle] def load_long(): Unit = {
    var `val` = PickleUtils.readline(input)
    if (`val` != null && `val`.endsWith("L")) `val` = `val`.substring(0, `val`.length - 1)
    val bi = new BigInteger(`val`)
    stack.add(PickleUtils.optimizeBigint(bi))
  }

  @throws[IOException]
  private[pickle] def load_long1(): Unit = {
    val n = PickleUtils.readbyte(input)
    val data = PickleUtils.readbytes(input, n)
    stack.add(PickleUtils.decode_long(data))
  }

  @throws[IOException]
  private[pickle] def load_long4(): Unit = {
    val n = PickleUtils.bytes_to_integer(PickleUtils.readbytes(input, 4))
    val data = PickleUtils.readbytes(input, n)
    stack.add(PickleUtils.decode_long(data))
  }

  @throws[IOException]
  private[pickle] def load_float(): Unit = {
    val `val` = PickleUtils.readline(input, true)
    stack.add(`val`.toDouble)
  }

  @throws[IOException]
  private[pickle] def load_binfloat(): Unit = {
    val `val` = PickleUtils.bytes_to_double(PickleUtils.readbytes(input, 8), 0)
    stack.add(`val`)
  }

  @throws[IOException]
  private[pickle] def load_string(): Unit = {
    var rep = PickleUtils.readline(input)
    var quotesOk = false
    for (q <- Array[String]("\"", "'")) {
      if (rep.startsWith(q)) {
        if (!rep.endsWith(q)) throw new PickleException("insecure string pickle")
        rep = rep.substring(1, rep.length - 1) // strip quotes
        quotesOk = true
        break //todo: break is not supported
      }
    } // double or single quote
    if (!quotesOk) throw new PickleException("insecure string pickle")
    stack.add(PickleUtils.decode_escaped(rep))
  }

  @throws[IOException]
  private[pickle] def load_binstring(): Unit = {
    val len = PickleUtils.bytes_to_integer(PickleUtils.readbytes(input, 4))
    val data = PickleUtils.readbytes(input, len)
    stack.add(PickleUtils.rawStringFromBytes(data))
  }

  @throws[IOException]
  private[pickle] def load_binbytes(): Unit = {
    val len = PickleUtils.bytes_to_integer(PickleUtils.readbytes(input, 4))
    stack.add(PickleUtils.readbytes(input, len))
  }

  @throws[IOException]
  private[pickle] def load_binbytes8(): Unit = {
    val len = PickleUtils.bytes_to_long(PickleUtils.readbytes(input, 8), 0)
    stack.add(PickleUtils.readbytes(input, len))
  }

  @throws[IOException]
  private[pickle] def load_unicode(): Unit = {
    val str = PickleUtils.decode_unicode_escaped(PickleUtils.readline(input))
    stack.add(str)
  }

  @throws[IOException]
  private[pickle] def load_binunicode(): Unit = {
    val len = PickleUtils.bytes_to_integer(PickleUtils.readbytes(input, 4))
    val data = PickleUtils.readbytes(input, len)
    stack.add(new String(data, StandardCharsets.UTF_8))
  }

  @throws[IOException]
  private[pickle] def load_binunicode8(): Unit = {
    val len = PickleUtils.bytes_to_long(PickleUtils.readbytes(input, 8), 0)
    val data = PickleUtils.readbytes(input, len)
    stack.add(new String(data, StandardCharsets.UTF_8))
  }

  @throws[IOException]
  private[pickle] def load_short_binunicode(): Unit = {
    val len = PickleUtils.readbyte(input)
    val data = PickleUtils.readbytes(input, len)
    stack.add(new String(data, StandardCharsets.UTF_8))
  }

  @throws[IOException]
  private[pickle] def load_short_binstring(): Unit = {
    val len = PickleUtils.readbyte(input)
    val data = PickleUtils.readbytes(input, len)
    stack.add(PickleUtils.rawStringFromBytes(data))
  }

  @throws[IOException]
  private[pickle] def load_short_binbytes(): Unit = {
    val len = PickleUtils.readbyte(input)
    stack.add(PickleUtils.readbytes(input, len))
  }

  private[pickle] def load_tuple(): Unit = {
    val top = stack.pop_all_since_marker
    stack.add(top.toArray)
  }

  private[pickle] def load_empty_tuple(): Unit = {
    stack.add(new Array[AnyRef](0))
  }

  private[pickle] def load_tuple1(): Unit = {
    stack.add(Array[AnyRef](stack.pop))
  }

  private[pickle] def load_tuple2(): Unit = {
    val o2 = stack.pop
    val o1 = stack.pop
    stack.add(Array[AnyRef](o1, o2))
  }

  private[pickle] def load_tuple3(): Unit = {
    val o3 = stack.pop
    val o2 = stack.pop
    val o1 = stack.pop
    stack.add(Array[AnyRef](o1, o2, o3))
  }

  private[pickle] def load_empty_list(): Unit = {
    stack.add(new util.ArrayList[AnyRef](0))
  }

  private[pickle] def load_empty_dictionary(): Unit = {
    stack.add(new util.HashMap[AnyRef, AnyRef](0))
  }

  private[pickle] def load_empty_set(): Unit = {
    stack.add(new util.HashSet[AnyRef])
  }

  private[pickle] def load_list(): Unit = {
    val top = stack.pop_all_since_marker
    stack.add(top) // simply add the top items as a list to the stack again
  }

  private[pickle] def load_dict(): Unit = {
    val top = stack.pop_all_since_marker
    val map = new util.HashMap[AnyRef, AnyRef](top.size)
    var i = 0
    while (i < top.size) {
      val key = top.get(i)
      val value = top.get(i + 1)
      map.put(key, value)
      i += 2
    }
    stack.add(map)
  }

  private[pickle] def load_frozenset(): Unit = {
    val top = stack.pop_all_since_marker
    val set = new util.HashSet[AnyRef](top)
    stack.add(set)
  }

  private[pickle] def load_additems(): Unit = {
    val top = stack.pop_all_since_marker
    @SuppressWarnings(Array("unchecked")) val set = stack.pop.asInstanceOf[util.HashSet[AnyRef]]
    set.addAll(top)
    stack.add(set)
  }

  @throws[IOException]
  private[pickle] def load_global(): Unit = {
    val module = PickleUtils.readline(input)
    val name = PickleUtils.readline(input)
    load_global_sub(module, name)
  }

  private[pickle] def load_stack_global(): Unit = {
    val name = stack.pop.asInstanceOf[String]
    val module = stack.pop.asInstanceOf[String]
    load_global_sub(module, name)
  }

  private[pickle] def load_global_sub(module: String, name: String): Unit = {
    var constructor = Unpickler.objectConstructors.get(module + "." + name)
    if (constructor == null) {
      // check if it is an exception
      if (module == "exceptions") {
        // python 2.x
        constructor = new ExceptionConstructor(classOf[PythonException], module, name)
      }
      else if (module == "builtins" || module == "__builtin__") if (name.endsWith("Error") || name.endsWith("Warning") || name.endsWith("Exception") || name == "GeneratorExit" || name == "KeyboardInterrupt" || name == "StopIteration" || name == "SystemExit") {
        // it's a python 3.x exception
        constructor = new ExceptionConstructor(classOf[PythonException], module, name)
      }
      else {
        // return a dictionary with the class's properties
        constructor = new ClassDictConstructor(module, name)
      }
      else {
        // return a dictionary with the class's properties
        constructor = new ClassDictConstructor(module, name)
      }
    }
    stack.add(constructor)
  }

  private[pickle] def load_pop(): Unit = {
    stack.pop
  }

  private[pickle] def load_pop_mark(): Unit = {
    var o: AnyRef = null
    do o = stack.pop while (o ne stack.MARKER)
    stack.trim()
  }

  private[pickle] def load_dup(): Unit = {
    stack.add(stack.peek)
  }

  @throws[IOException]
  private[pickle] def load_get(): Unit = {
    val i = Integer.parseInt(PickleUtils.readline(input), 10)
    if (!memo.containsKey(i)) throw new PickleException("invalid memo key")
    stack.add(memo.get(i))
  }

  @throws[IOException]
  private[pickle] def load_binget(): Unit = {
    val i = PickleUtils.readbyte(input)
    if (!memo.containsKey(i)) throw new PickleException("invalid memo key")
    stack.add(memo.get(i))
  }

  @throws[IOException]
  private[pickle] def load_long_binget(): Unit = {
    val i = PickleUtils.bytes_to_integer(PickleUtils.readbytes(input, 4))
    if (!memo.containsKey(i)) throw new PickleException("invalid memo key")
    stack.add(memo.get(i))
  }

  @throws[IOException]
  private[pickle] def load_put(): Unit = {
    val i = Integer.parseInt(PickleUtils.readline(input), 10)
    memo.put(i, stack.peek)
  }

  @throws[IOException]
  private[pickle] def load_binput(): Unit = {
    val i = PickleUtils.readbyte(input)
    memo.put(i, stack.peek)
  }

  @throws[IOException]
  private[pickle] def load_long_binput(): Unit = {
    val i = PickleUtils.bytes_to_integer(PickleUtils.readbytes(input, 4))
    memo.put(i, stack.peek)
  }

  private[pickle] def load_memoize(): Unit = {
    memo.put(memo.size, stack.peek)
  }

  private[pickle] def load_append(): Unit = {
    val value = stack.pop
    @SuppressWarnings(Array("unchecked")) val list = stack.peek.asInstanceOf[util.ArrayList[AnyRef]]
    list.add(value)
  }

  private[pickle] def load_appends(): Unit = {
    val top = stack.pop_all_since_marker
    @SuppressWarnings(Array("unchecked")) val list = stack.peek.asInstanceOf[util.ArrayList[AnyRef]]
    list.addAll(top)
    list.trimToSize()
  }

  private[pickle] def load_setitem(): Unit = {
    val value = stack.pop
    val key = stack.pop
    @SuppressWarnings(Array("unchecked")) val dict = stack.peek.asInstanceOf[util.Map[AnyRef, AnyRef]]
    dict.put(key, value)
  }

  private[pickle] def load_setitems(): Unit = {
    val newitems = new util.HashMap[AnyRef, AnyRef]
    var value = stack.pop
    while (value ne stack.MARKER) {
      val key = stack.pop
      newitems.put(key, value)
      value = stack.pop
    }
    @SuppressWarnings(Array("unchecked")) val dict = stack.peek.asInstanceOf[util.Map[AnyRef, AnyRef]]
    dict.putAll(newitems)
  }

  private[pickle] def load_mark(): Unit = {
    stack.add_mark()
  }

  private[pickle] def load_reduce(): Unit = {
    val args = stack.pop.asInstanceOf[Array[AnyRef]]
    val constructor = stack.pop.asInstanceOf[IObjectConstructor]
    stack.add(constructor.construct(args))
  }

  private[pickle] def load_newobj(): Unit = {
    load_reduce() // for Java we just do the same as class(*args) instead of class.__new__(class,*args)
  }

  private[pickle] def load_newobj_ex(): Unit = {
    val kwargs = stack.pop.asInstanceOf[util.HashMap[_, _]]
    val args = stack.pop.asInstanceOf[Array[AnyRef]]
    val constructor = stack.pop.asInstanceOf[IObjectConstructor]
    if (kwargs.isEmpty) stack.add(constructor.construct(args))
    else throw new PickleException("newobj_ex with keyword arguments not supported")
  }

  @throws[IOException]
  private[pickle] def load_frame(): Unit = {
    // for now we simply skip the frame opcode and its length
    PickleUtils.readbytes(input, 8)
  }

  @throws[IOException]
  private[pickle] def load_persid(): Unit = {
    // the persistent id is taken from the argument
    val pid = PickleUtils.readline(input)
    stack.add(persistentLoad(pid))
  }

  @throws[IOException]
  private[pickle] def load_binpersid(): Unit = {
    // the persistent id is taken from the stack
    val pid = stack.pop
    stack.add(persistentLoad(pid))
  }

  @throws[IOException]
  private[pickle] def load_obj(): Unit = {
    var args = stack.pop_all_since_marker
    val constructor = args.get(0).asInstanceOf[IObjectConstructor]
    args = args.subList(1, args.size)
    val `object` = constructor.construct(args.toArray)
    stack.add(`object`)
  }

  @throws[IOException]
  private[pickle] def load_inst(): Unit = {
    val module = PickleUtils.readline(input)
    val classname = PickleUtils.readline(input)
    val args = stack.pop_all_since_marker
    var constructor = Unpickler.objectConstructors.get(module + "." + classname)
    if (constructor == null) {
      constructor = new ClassDictConstructor(module, classname)
      args.clear() // classdict doesn't have constructor args... so we may lose info here, hmm.
    }
    val `object` = constructor.construct(args.toArray)
    stack.add(`object`)
  }

  /**
   * Hook for the persistent id feature where an id is replaced externally by the appropriate object.
   *
   * @param pid the persistent id from the pickle
   * @return the actual object that belongs to that id. The default implementation throws a PickleException,
   *         telling you that you should implement this function yourself in a subclass of the Unpickler.
   */
  protected def persistentLoad(pid: AnyRef): AnyRef = throw new PickleException("A load persistent id instruction was encountered, but no persistentLoad function was specified. (implement it in custom Unpickler subclass)")
}