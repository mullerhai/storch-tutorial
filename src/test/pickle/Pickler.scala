package net.razorvine.pickle

import net.razorvine.pickle.objects.Time
import net.razorvine.pickle.objects.TimeDelta

import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.OutputStream
import java.lang.reflect.InvocationTargetException
import java.lang.reflect.Method
import java.lang.reflect.Modifier
import java.math.BigDecimal
import java.math.BigInteger
import java.nio.charset.StandardCharsets
import java.util
import java.util.{Calendar, Date, TimeZone}
import java.util.Map.Entry
import scala.jdk.CollectionConverters.*

/**
 * Pickle an object graph into a Python-compatible pickle stream. For
 * simplicity, the only supported pickle protocol at this time is protocol 2.
 * This class is NOT threadsafe! (Don't use the same pickler from different threads)
 *
 * See the README.txt for a table with the type mappings.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
object Pickler {
  /**
   * The highest Python pickle protocol supported by this Pickler.
   */
  var HIGHEST_PROTOCOL = 2

  /**
   * A memoized object.
   */
  protected class Memo(val obj: AnyRef, val index: Int) {
  }

  /**
   * Limit on the recursion depth to avoid stack overflows.
   */
  protected val MAX_RECURSE_DEPTH = 1000
  /**
   * Registry of picklers for custom classes, to be able to not just pickle simple built in datatypes.
   * You can add to this via {@link Pickler# registerCustomPickler}
   */
  protected val customPicklers = new util.HashMap[Class[_], IObjectPickler]
  /**
   * Registry of deconstructors for custom classes, to be able to pickle custom classes and also reconstruct.
   * them using {@link Unpickler# registerConstructor}. can add to this via {@link Pickler# registerCustomDeconstructor}
   */
  protected var customDeconstructors = new util.HashMap[Class[_], IObjectDeconstructor]

  /**
   * Register additional object picklers for custom classes.
   * If you register an interface or abstract base class, it means the pickler is used for
   * the whole inheritance tree of all classes ultimately implementing that interface or abstract base class.
   * If you register a normal concrete class, the pickler is only used for objects of exactly that particular class.
   */
  def registerCustomPickler(clazz: Class[_], pickler: IObjectPickler): Unit = {
    customPicklers.put(clazz, pickler)
  }

  /**
   * Register custom object deconstructor for custom classes.
   * An alternative for writing your own pickler, you can create a deconstructor which will have a 
   * name & module, and the deconstructor will convert an object to a list of objects which will then
   * be used as the arguments for reconstructing when unpickling.
   */
  def registerCustomDeconstructor(clazz: Class[_], deconstructor: IObjectDeconstructor): Unit = {
    customDeconstructors.put(clazz, deconstructor)
  }
}

class Pickler(useMemo: Boolean, valueCompare: Boolean){

/**
 * Create a Pickler. Also specify if it is to compare objects by value.
 * If you compare objects by value, the object graph might be altered,
 * as different instances with the same value will be unified.
 * (The default for valueCompare when creating a pickler is true,
 * so if this is problematic for you, you can turn it off here)
 */ 

  /**
   * Current recursion level.
   */
  protected var recurse = 0 // recursion level
  /**
   * Output where the pickle data is written to.
   */
  protected var out: OutputStream = null
  /**
   * The Python pickle protocol version of the pickles created by this library.
   */
  final protected val PROTOCOL = 2
  /**
   * Use memoization or not. This saves pickle size, but can only create pickles of objects that are hashable.
   */
  protected var useMemo = true
  /**
   * When memoizing, compare objects by value. This saves pickle size, but can slow down pickling.
   * Also, it should only be used if the object graph is immutable. Unused if useMemo is false.
   */
  protected var valueCompare = true
  /**
   * The memoization cache.
   */
  protected var memo: util.HashMap[Integer, Pickler.Memo] = null // maps object's identity hash to (object, memo index)
  this.useMemo = useMemo
  this.valueCompare = valueCompare
  /**
   * Create a Pickler. Specify if it is to use a memo table or not.
   * The memo table is NOT reused across different calls.
   * If you use a memo table, you can only pickle objects that are hashable.
   */
  def this(useMemo: Boolean) ={
    this(useMemo, true)
  }

  /**
   * Create a Pickler.
   */
  def this ={
    this(true)
  }

  /**
   * Close the pickler stream, discard any internal buffers.
   */
  @throws[IOException]
  def close(): Unit = {
    memo = null
    out.flush()
    out.close()
  }

  /**
   * Pickle a given object graph, returning the result as a byte array.
   */
  @throws[PickleException]
  @throws[IOException]
  def dumps(o: AnyRef): Array[Byte] = {
    val bo = new ByteArrayOutputStream
    dump(o, bo)
    bo.flush()
    bo.toByteArray
  }

  /**
   * Pickle a given object graph, writing the result to the output stream.
   */
  @throws[IOException]
  @throws[PickleException]
  def dump(o: AnyRef, stream: OutputStream): Unit = {
    out = stream
    recurse = 0
    if (useMemo) memo = new util.HashMap[Integer, Pickler.Memo]
    out.write(Opcodes.PROTO)
    out.write(PROTOCOL)
    save(o)
    memo = null // get rid of the memo table
    out.write(Opcodes.STOP)
    out.flush()
    if (recurse != 0) throw new PickleException("recursive structure error, please report this problem") // sanity check
  }

  /**
   * Pickle a single object and write its pickle representation to the output stream.
   * Normally this is used internally by the pickler, but you can also utilize it from
   * within custom picklers. This is handy if as part of the custom pickler, you need
   * to write a couple of normal objects such as strings or ints, that are already
   * supported by the pickler.
   * This method can be called recursively to output sub-objects.
   */
  @throws[PickleException]
  @throws[IOException]
  def save(o: AnyRef): Unit = {
    recurse += 1
    if (recurse > Pickler.MAX_RECURSE_DEPTH) throw new StackOverflowError("recursion too deep in Pickler.save (>" + Pickler.MAX_RECURSE_DEPTH + ")")
    // null type?
    if (o == null) {
      out.write(Opcodes.NONE)
      recurse -= 1
      return
    }
    // check the memo table, otherwise simply dispatch
    val t = o.getClass
    if (lookupMemo(t, o) || dispatch(t, o)) {
      recurse -= 1
      return
    }
    throw new PickleException("couldn't pickle object of type " + t)
  }

  /**
   * Write the object to the memo table and output a memo write opcode
   * Only works for hashable objects
   */
  @throws[IOException]
  protected def writeMemo(obj: AnyRef): Unit = {
    if (!this.useMemo) return
    val hash = if (valueCompare) obj.hashCode
    else System.identityHashCode(obj)
    if (!memo.containsKey(hash)) {
      val memo_index = memo.size
      memo.put(hash, new Pickler.Memo(obj, memo_index))
      if (memo_index <= 0xFF) {
        out.write(Opcodes.BINPUT)
        out.write(memo_index.toByte)
      }
      else {
        out.write(Opcodes.LONG_BINPUT)
        val index_bytes = PickleUtils.integer_to_bytes(memo_index)
        out.write(index_bytes, 0, 4)
      }
    }
  }

  /**
   * Check the memo table and output a memo lookup if the object is found
   */
  @throws[IOException]
  private def lookupMemo(objectType: Class[_], obj: AnyRef): Boolean = {
    if (!this.useMemo) return false
    if (!objectType.isPrimitive) {
      val hash = if (valueCompare) obj.hashCode
      else System.identityHashCode(obj)
      if (memo.containsKey(hash) && (if (valueCompare) memo.get(hash).obj == obj
      else memo.get(hash).obj eq obj)) { // same object or value
        val memo_index = memo.get(hash).index
        if (memo_index <= 0xff) {
          out.write(Opcodes.BINGET)
          out.write(memo_index.toByte)
        }
        else {
          out.write(Opcodes.LONG_BINGET)
          val index_bytes = PickleUtils.integer_to_bytes(memo_index)
          out.write(index_bytes, 0, 4)
        }
        return true
      }
    }
    false
  }

  /**
   * Process a single object to be pickled.
   */
  @throws[IOException]
  private def dispatch(t: Class[_], o: AnyRef): Boolean = {
    // is it a primitive array?
    val componentType = t.getComponentType
    if (componentType != null) {
      if (componentType.isPrimitive) put_arrayOfPrimitives(componentType, o)
      else put_arrayOfObjects(o.asInstanceOf[Array[AnyRef]])
      return true
    }
    // first the primitive types
    if (o.isInstanceOf[Boolean] || t == java.lang.Boolean.TYPE) {
      put_bool(o.asInstanceOf[Boolean])
      return true
    }
    if (o.isInstanceOf[Byte] || t == java.lang.Byte.TYPE) {
      put_long(o.asInstanceOf[Byte].longValue)
      return true
    }
    if (o.isInstanceOf[Short] || t == java.lang.Short.TYPE) {
      put_long(o.asInstanceOf[Short].longValue)
      return true
    }
    if (o.isInstanceOf[Integer] || t == java.lang.Integer.TYPE) {
      put_long(o.asInstanceOf[Integer].longValue)
      return true
    }
    if (o.isInstanceOf[Long] || t == java.lang.Long.TYPE) {
      put_long(o.asInstanceOf[Long])
      return true
    }
    if (o.isInstanceOf[Float] || t == java.lang.Float.TYPE) {
      put_float(o.asInstanceOf[Float].doubleValue)
      return true
    }
    if (o.isInstanceOf[Double] || t == java.lang.Double.TYPE) {
      put_float(o.asInstanceOf[Double])
      return true
    }
    if (o.isInstanceOf[Character] || t == Character.TYPE) {
      put_string("" + o)
      return true
    }
    // check registry
    val custompickler = getCustomPickler(t)
    if (custompickler != null) {
      custompickler.pickle(o, this.out, this)
      writeMemo(o)
      return true
    }
    val customDeconstructor = getCustomDeconstructor(t)
    if (customDeconstructor != null) {
      put_global(customDeconstructor, o)
      return true
    }
    // Check for persistentId
    val persistentId = persistentId(o)
    if (persistentId != null) {
      if (persistentId.isInstanceOf[String] && !(persistentId.asInstanceOf[String]).contains("\n")) {
        out.write(Opcodes.PERSID)
        out.write(persistentId.asInstanceOf[String].getBytes)
        out.write("\n".getBytes)
      }
      else {
        save(persistentId)
        out.write(Opcodes.BINPERSID)
      }
      return true
    }
    // more complex types
    if (o.isInstanceOf[String]) {
      put_string(o.asInstanceOf[String])
      return true
    }
    if (o.isInstanceOf[BigInteger]) {
      put_bigint(o.asInstanceOf[BigInteger])
      return true
    }
    if (o.isInstanceOf[BigDecimal]) {
      put_decimal(o.asInstanceOf[BigDecimal])
      return true
    }
    if (o.isInstanceOf[Time]) {
      val sqltime = o.asInstanceOf[Time]
      val time = new Time(sqltime.getTime)
      put_time(time)
      return true
    }
    if (o.isInstanceOf[Date]) {
      put_sqldate(o.asInstanceOf[Date])
      return true
    }
    if (o.isInstanceOf[Calendar]) {
      put_calendar(o.asInstanceOf[Calendar])
      return true
    }
    if (o.isInstanceOf[Time]) {
      put_time(o.asInstanceOf[Time])
      return true
    }
    if (o.isInstanceOf[TimeDelta]) {
      put_timedelta(o.asInstanceOf[TimeDelta])
      return true
    }
    if (o.isInstanceOf[Date]) {
      // a java Date contains a date+time so map this on Calendar
      // which will be pickled as a datetime.
      val date = o.asInstanceOf[Date]
      val cal = GregorianCalendar.getInstance
      cal.setTime(date)
      put_calendar(cal)
      return true
    }
    if (o.isInstanceOf[TimeZone]) {
      put_timezone(o.asInstanceOf[TimeZone])
      return true
    }
    if (o.isInstanceOf[(Enum[E]) forSome {type E <: Enum[E]}]) {
      put_string(o.toString)
      return true
    }
    if (o.isInstanceOf[util.Set[_]]) {
      put_set(o.asInstanceOf[util.Set[_]])
      return true
    }
    if (o.isInstanceOf[util.Map[_, _]]) {
      put_map(o.asInstanceOf[util.Map[_, _]])
      return true
    }
    if (o.isInstanceOf[util.List[_]]) {
      put_collection(o.asInstanceOf[util.List[_]])
      return true
    }
    if (o.isInstanceOf[util.Collection[_]]) {
      put_collection(o.asInstanceOf[util.Collection[_]])
      return true
    }
    // javabean
    if (o.isInstanceOf[Serializable]) {
      put_javabean(o)
      return true
    }
    false
  }

  /**
   * Get the custom pickler fot the given class, to be able to pickle not just built in collection types.
   * A custom pickler is matched on the interface or abstract base class that the object implements or inherits from.
   *
   * @param t the class of the object to be pickled
   * @return null (if no custom pickler found) or a pickler registered for this class (via {@link Pickler# registerCustomPickler})
   */
  protected def getCustomPickler(t: Class[_]): IObjectPickler = {
    val pickler = Pickler.customPicklers.get(t)
    if (pickler != null) return pickler // exact match
    // check if there's a custom pickler registered for an interface or abstract base class
    // that this object implements or inherits from.
   
    for (x <- Pickler.customPicklers.entrySet) {
      if (x.getKey.isAssignableFrom(t)) return x.getValue
    }
    null
  }

  /**
   * Get the custom deconstructor fot the given class, to be able to pickle and unpickle custom classes
   * A custom deconstructor is matched on the interface or abstract base class that the object implements or inherits from.
   *
   * @param t the class of the object to be pickled
   * @return null (if no custom deconstructor found) or a deconstructor registered for this class (via {@link Pickler# registerCustomDeconstructor})
   */
  protected def getCustomDeconstructor(t: Class[_]): IObjectDeconstructor = Pickler.customDeconstructors.get(t)

  @throws[IOException]
  private[pickle] def put_collection(list: util.Collection[_]): Unit = {
    out.write(Opcodes.EMPTY_LIST)
    writeMemo(list)
    out.write(Opcodes.MARK)
  
    for (o <- list) {
      save(o)
    }
    out.write(Opcodes.APPENDS)
  }

  @throws[IOException]
  private[pickle] def put_map(o: util.Map[_, _]): Unit = {
    out.write(Opcodes.EMPTY_DICT)
    writeMemo(o)
    out.write(Opcodes.MARK)
    
    for (k <- o.keySet) {
      save(k)
      save(o.get(k))
    }
    out.write(Opcodes.SETITEMS)
  }

  @throws[IOException]
  private[pickle] def put_set(o: util.Set[_]): Unit = {
    out.write(Opcodes.GLOBAL)
    out.write("__builtin__\nset\n".getBytes)
    out.write(Opcodes.EMPTY_LIST)
    out.write(Opcodes.MARK)
    for (x <- o) {
      save(x)
    }
    out.write(Opcodes.APPENDS)
    out.write(Opcodes.TUPLE1)
    out.write(Opcodes.REDUCE)
    writeMemo(o) // sets cannot contain self-reference (they are not hashable) so it's fine to put this at the end
  }

  @throws[IOException]
  private[pickle] def put_calendar(cal: Calendar): Unit = {
    if (cal.getTimeZone != null) {
      // Use special pickle to get the timezone encoded in such a way
      // that when unpickling this, the correct time offset is used.
      // For pytz timezones this means that it is required to not simply
      // pass a pytz object as tzinfo argument to the datetime constructor,
      // you have to call tz.localize(datetime) instead.
      // This is rather tricky to do in pickle because normally you cannot
      // simply call arbitrary functions or methods. However when we throw
      // opcode.attrgetter in the mix we can get access to the localize function
      // of a pytz object and create the datetime from there.
      out.write(Opcodes.GLOBAL)
      out.write("operator\nattrgetter\n".getBytes)
      put_string("localize")
      out.write(Opcodes.TUPLE1)
      out.write(Opcodes.REDUCE)
      put_timezone(cal.getTimeZone)
      out.write(Opcodes.TUPLE1)
      out.write(Opcodes.REDUCE)
      put_calendar_without_timezone(cal, false)
      out.write(Opcodes.TUPLE1)
      out.write(Opcodes.REDUCE)
      writeMemo(cal)
      return
    }
    // Use the regular (non-pytz) calendar pickler.
    put_calendar_without_timezone(cal, true)
  }

  @throws[IOException]
  private[pickle] def put_calendar_without_timezone(cal: Calendar, writememo: Boolean): Unit = {
    // Note that we can't use the 2-arg representation of a datetime here.
    // Python 3 uses the SHORT_BINBYTES opcode to encode the first argument,
    // but python 2 uses SHORT_BINSTRING instead. This causes encoding problems
    // if you want to send the pickle to either a Python 2 or 3 receiver.
    // So instead, we use the full 7 (or 8 with timezone) constructor representation.
    out.write(Opcodes.GLOBAL)
    out.write("datetime\ndatetime\n".getBytes)
    out.write(Opcodes.MARK)
    save(cal.get(Calendar.YEAR))
    save(cal.get(Calendar.MONTH) + 1) // months start at 0 in java
    save(cal.get(Calendar.DAY_OF_MONTH))
    save(cal.get(Calendar.HOUR_OF_DAY))
    save(cal.get(Calendar.MINUTE))
    save(cal.get(Calendar.SECOND))
    save(cal.get(Calendar.MILLISECOND) * 1000)
    // this method by design does NOT pickle the tzinfo argument:
    //		if(cal.getTimeZone()!=null)
    //			save(cal.getTimeZone());
    out.write(Opcodes.TUPLE)
    out.write(Opcodes.REDUCE)
    if (writememo) writeMemo(cal)
  }

  @throws[IOException]
  private[pickle] def put_timedelta(delta: TimeDelta): Unit = {
    out.write(Opcodes.GLOBAL)
    out.write("datetime\ntimedelta\n".getBytes)
    save(delta.days)
    save(delta.seconds)
    save(delta.microseconds)
    out.write(Opcodes.TUPLE3)
    out.write(Opcodes.REDUCE)
    writeMemo(delta)
  }

  @throws[IOException]
  private[pickle] def put_time(time: Time): Unit = {
    out.write(Opcodes.GLOBAL)
    out.write("datetime\ntime\n".getBytes)
    out.write(Opcodes.MARK)
    save(time.hours)
    save(time.minutes)
    save(time.seconds)
    save(time.microseconds)
    out.write(Opcodes.TUPLE)
    out.write(Opcodes.REDUCE)
    writeMemo(time)
  }

  @throws[IOException]
  private[pickle] def put_sqldate(date: Date): Unit = {
    out.write(Opcodes.GLOBAL)
    out.write("datetime\ndate\n".getBytes)
    // python itself uses the constructor with a single timestamp byte string of len 4,
    // we take the easy way out and just provide 3 ints (year/month/day)
    val cal = Calendar.getInstance
    cal.setTime(date)
    save(cal.get(Calendar.YEAR))
    save(cal.get(Calendar.MONTH) + 1) // months start at 0 in java
    save(cal.get(Calendar.DAY_OF_MONTH))
    out.write(Opcodes.TUPLE3)
    out.write(Opcodes.REDUCE)
    writeMemo(date)
  }

  @throws[IOException]
  private[pickle] def put_timezone(timeZone: TimeZone): Unit = {
    out.write(Opcodes.GLOBAL)
    if (timeZone.getID == "UTC") {
      out.write("pytz\n_UTC\n".getBytes)
      out.write(Opcodes.MARK)
    }
    else {
      // Don't write out the shorthand pytz._p for pickle,
      // because it needs the internal timezone offset amounts.
      // It is not possible to supply the correct amounts for that
      // because it would require the context of a specific date/time.
      // So we just use the pytz.timezone("..") constructor from the ID string.
      out.write("pytz\ntimezone\n".getBytes)
      out.write(Opcodes.MARK)
      save(timeZone.getID)
    }
    out.write(Opcodes.TUPLE)
    out.write(Opcodes.REDUCE)
    writeMemo(timeZone)
  }

  @throws[IOException]
  private[pickle] def put_arrayOfObjects(array: Array[AnyRef]): Unit = {
    // 0 objects->EMPTYTUPLE
    // 1 object->TUPLE1
    // 2 objects->TUPLE2
    // 3 objects->TUPLE3
    // 4 or more->MARK+items+TUPLE
    if (array.length == 0) out.write(Opcodes.EMPTY_TUPLE)
    else if (array.length == 1) {
      if (array(0) eq array) throw new PickleException("recursive array not supported, use list")
      save(array(0))
      out.write(Opcodes.TUPLE1)
    }
    else if (array.length == 2) {
      if ((array(0) eq array) || (array(1) eq array)) throw new PickleException("recursive array not supported, use list")
      save(array(0))
      save(array(1))
      out.write(Opcodes.TUPLE2)
    }
    else if (array.length == 3) {
      if ((array(0) eq array) || (array(1) eq array) || (array(2) eq array)) throw new PickleException("recursive array not supported, use list")
      save(array(0))
      save(array(1))
      save(array(2))
      out.write(Opcodes.TUPLE3)
    }
    else {
      out.write(Opcodes.MARK)
      for (o <- array) {
        if (o eq array) throw new PickleException("recursive array not supported, use list")
        save(o)
      }
      out.write(Opcodes.TUPLE)
    }
    writeMemo(array) // tuples cannot contain self-references so it is fine to put this at the end
  }

  @throws[IOException]
  private[pickle] def put_arrayOfPrimitives(t: Class[_], array: AnyRef): Unit = {
    if (t == Boolean.TYPE) {
      // a bool[] isn't written as an array but rather as a tuple
      val source = array.asInstanceOf[Array[Boolean]]
      val boolarray = new Array[Boolean](source.length)
      for (i <- 0 until source.length) {
        boolarray(i) = source(i)
      }
      put_arrayOfObjects(boolarray)
      return
    }
    if (t == Character.TYPE) {
      // a char[] isn't written as an array but rather as a unicode string
      val s = new String(array.asInstanceOf[Array[Char]])
      put_string(s)
      return
    }
    if (t == Byte.TYPE) {
      // a byte[] isn't written as an array but rather as a bytearray object
      out.write(Opcodes.GLOBAL)
      out.write("__builtin__\nbytearray\n".getBytes)
      val str = PickleUtils.rawStringFromBytes(array.asInstanceOf[Array[Byte]])
      put_string(str)
      put_string("latin-1") // this is what Python writes in the pickle
      out.write(Opcodes.TUPLE2)
      out.write(Opcodes.REDUCE)
      writeMemo(array)
      return
    }
    out.write(Opcodes.GLOBAL)
    out.write("array\narray\n".getBytes)
    out.write(Opcodes.SHORT_BINSTRING) // array typecode follows
    out.write(1) // typecode is 1 char
    if (t == Short.TYPE) {
      out.write('h') // signed short
      out.write(Opcodes.EMPTY_LIST)
      out.write(Opcodes.MARK)
      for (s <- array.asInstanceOf[Array[Short]]) {
        save(s)
      }
    }
    else if (t == Integer.TYPE) {
      out.write('i') // signed int
      out.write(Opcodes.EMPTY_LIST)
      out.write(Opcodes.MARK)
      for (i <- array.asInstanceOf[Array[Int]]) {
        save(i)
      }
    }
    else if (t == Long.TYPE) {
      out.write('l') // signed long
      out.write(Opcodes.EMPTY_LIST)
      out.write(Opcodes.MARK)
      for (v <- array.asInstanceOf[Array[Long]]) {
        save(v)
      }
    }
    else if (t == java.lang.Float.TYPE) {
      out.write('f') // float
      out.write(Opcodes.EMPTY_LIST)
      out.write(Opcodes.MARK)
      for (f <- array.asInstanceOf[Array[Float]]) {
        save(f)
      }
    }
    else if (t == java.lang.Double.TYPE) {
      out.write('d') // double
      out.write(Opcodes.EMPTY_LIST)
      out.write(Opcodes.MARK)
      for (d <- array.asInstanceOf[Array[Double]]) {
        save(d)
      }
    }
    out.write(Opcodes.APPENDS)
    out.write(Opcodes.TUPLE2)
    out.write(Opcodes.REDUCE)
    writeMemo(array) // array of primitives can by definition never be recursive, so okay to put this at the end
  }

  @throws[IOException]
  private[pickle] def put_global(deconstructor: IObjectDeconstructor, obj: AnyRef): Unit = {
    out.write(Opcodes.GLOBAL)
    out.write((deconstructor.getModule + "\n" + deconstructor.getName + "\n").getBytes)
    val values = deconstructor.deconstruct(obj)
    if (values.length > 0) {
      save(values)
      out.write(Opcodes.REDUCE)
    }
    writeMemo(obj)
  }

  @throws[IOException]
  private[pickle] def put_decimal(d: BigDecimal): Unit = {
    //"cdecimal\nDecimal\nU\n12345.6789\u0085R."
    out.write(Opcodes.GLOBAL)
    out.write("decimal\nDecimal\n".getBytes)
    put_string(d.toEngineeringString)
    out.write(Opcodes.TUPLE1)
    out.write(Opcodes.REDUCE)
    writeMemo(d)
  }

  @throws[IOException]
  private[pickle] def put_bigint(i: BigInteger): Unit = {
    val b = PickleUtils.encode_long(i)
    if (b.length <= 0xff) {
      out.write(Opcodes.LONG1)
      out.write(b.length)
      out.write(b)
    }
    else {
      out.write(Opcodes.LONG4)
      out.write(PickleUtils.integer_to_bytes(b.length))
      out.write(b)
    }
    writeMemo(i)
  }

  @throws[IOException]
  private[pickle] def put_string(string: String): Unit = {
    val encoded = string.getBytes(StandardCharsets.UTF_8)
    out.write(Opcodes.BINUNICODE)
    out.write(PickleUtils.integer_to_bytes(encoded.length))
    out.write(encoded)
    writeMemo(string)
  }

  @throws[IOException]
  private[pickle] def put_float(d: Double): Unit = {
    out.write(Opcodes.BINFLOAT)
    out.write(PickleUtils.double_to_bytes(d))
  }

  @throws[IOException]
  private[pickle] def put_long(v: Long): Unit = {
    // choose optimal representation
    // first check 1 and 2-byte unsigned ints:
    if (v >= 0) {
      if (v <= 0xff) {
        out.write(Opcodes.BININT1)
        out.write(v.toInt)
        return
      }
      if (v <= 0xffff) {
        out.write(Opcodes.BININT2)
        out.write(v.toInt & 0xff)
        out.write(v.toInt >> 8)
        return
      }
    }
    // 4-byte signed int?
    val high_bits = v >> 31 // shift sign extends
    if (high_bits == 0 || high_bits == -1) {
      // All high bits are copies of bit 2**31, so the value fits in a 4-byte signed int.
      out.write(Opcodes.BININT)
      out.write(PickleUtils.integer_to_bytes(v.toInt))
      return
    }
    // int too big, use bigint to store it as LONG1
    put_bigint(BigInteger.valueOf(v))
  }

  @throws[IOException]
  private[pickle] def put_bool(b: Boolean): Unit = {
    if (b) out.write(Opcodes.NEWTRUE)
    else out.write(Opcodes.NEWFALSE)
  }

  @throws[PickleException]
  @throws[IOException]
  private[pickle] def put_javabean(o: AnyRef): Unit = {
    val map = new util.HashMap[String, AnyRef]
    try {
      // note: don't use the java.bean api, because that is not available on Android.
      for (m <- o.getClass.getMethods) {
        val modifiers = m.getModifiers
        if ((modifiers & Modifier.PUBLIC) != 0 && (modifiers & Modifier.STATIC) == 0) {
          val methodname = m.getName
          var prefixlen = 0
          if (methodname == "getClass") continue //todo: continue is not supported
          if (methodname.startsWith("get")) prefixlen = 3
          else if (methodname.startsWith("is")) prefixlen = 2
          else continue //todo: continue is not supported
          val value = m.invoke(o)
          var name = methodname.substring(prefixlen)
          if (name.length == 1) name = name.toLowerCase
          else if (!Character.isUpperCase(name.charAt(1))) name = Character.toLowerCase(name.charAt(0)) + name.substring(1)
          map.put(name, value)
        }
      }
      map.put("__class__", o.getClass.getName)
      save(map)
    } catch {
      case e: IllegalArgumentException =>
        throw new PickleException("couldn't introspect javabean: " + e)
      case e: IllegalAccessException =>
        throw new PickleException("couldn't introspect javabean: " + e)
      case e: InvocationTargetException =>
        throw new PickleException("couldn't introspect javabean: " + e)
    }
  }

  /**
   * Hook for the persistent id feature where an object is replaced externally by an id
   *
   * @param obj the object to replace with an id
   * @return the id object that belongs to this object, or null if this object isn't replaced by an id.
   */
  protected def persistentId(obj: AnyRef): AnyRef = null
}