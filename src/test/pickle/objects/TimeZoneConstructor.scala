package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import net.razorvine.pickle.PickleException
import java.util.TimeZone
import scala.jdk.CollectionConverters.*
object TimeZoneConstructor {
  val UTC = 1
  val PYTZ = 2
  val DATEUTIL_TZUTC = 3
  val DATEUTIL_TZFILE = 4
  val DATEUTIL_GETTZ = 5
  val TZINFO = 6
}

class TimeZoneConstructor(private val pythontype: Int) extends IObjectConstructor {
  @throws[PickleException]
  override def construct(args: Array[AnyRef]): AnyRef = {
    if (this.pythontype == TimeZoneConstructor.UTC) return createUTC
    if (this.pythontype == TimeZoneConstructor.PYTZ) return createZoneFromPytz(args)
    if (this.pythontype == TimeZoneConstructor.DATEUTIL_TZUTC) return createInfoFromDateutilTzutc(args)
    if (this.pythontype == TimeZoneConstructor.DATEUTIL_TZFILE) return createInfoFromDateutilTzfile(args)
    if (this.pythontype == TimeZoneConstructor.DATEUTIL_GETTZ) return createInfoFromDateutilGettz(args)
    if (this.pythontype == TimeZoneConstructor.TZINFO) return createInfo(args)
    throw new PickleException("invalid object type")
  }

  def reconstruct(baseConstructor: AnyRef, state: AnyRef): AnyRef = {
    if (!state.isInstanceOf[Tzinfo]) throw new PickleException("invalid pickle data for tzinfo reconstruction; expected emtpy tzinfo state class")
    if (!baseConstructor.isInstanceOf[TimeZoneConstructor]) throw new PickleException("invalid pickle data for tzinfo reconstruction; expected a TimeZoneConstructor from a known tzinfo subclass")
    // The subclass (this) is reconstructing the state given the base class and state. If it is known that the
    // subclass is always UTC, ie dateutil.tz.tzutc, then we can just return the timezone we know matches that.
    if (this.pythontype == TimeZoneConstructor.DATEUTIL_TZUTC) TimeZone.getTimeZone("UTC")
    else throw new PickleException("unsupported pickle data for tzinfo reconstruction; support for tzinfo subclasses other than tztuc has not been implemented")
  }

  private def createInfo(args: Array[AnyRef]) = {
    // args is empty, datetime.tzinfo objects are unpickled via setstate, so return an object which is ready to have it's state set
    new Tzinfo
  }

  private def createInfoFromDateutilTzutc(args: Array[AnyRef]) = {
    // In the case of the dateutil.tz.tzutc constructor, which is a python subclass of the datetime.tzinfo class, there is no state
    // to set, because the zone is implied by the constructor. Pass the timezone indicated by the constructor here
    new Tzinfo(TimeZone.getTimeZone("UTC"))
  }

  private def createInfoFromDateutilTzfile(args: Array[AnyRef]) = {
    if (args.length != 1) throw new PickleException("invalid pickle data for dateutil tzfile timezone; expected 1 args, got " + args.length)
    // In the case of the dateutil.tz.tzfile constructor, which is a python subclass of the datetime.tzinfo class, we're passed a
    // fully qualified path to a zoneinfo file. Extract the actual identifier as the part after the "zoneinfo" folder in the
    // absolute path.
    var identifier = args(0).asInstanceOf[String]
    val index = identifier.indexOf("zoneinfo")
    if (index != -1) identifier = identifier.substring(index + 8 + 1)
    else throw new PickleException("couldn't parse timezone identifier from zoneinfo path" + identifier)
    new Tzinfo(TimeZone.getTimeZone(identifier))
  }

  private def createInfoFromDateutilGettz(args: Array[AnyRef]) = {
    if (args.length != 1) throw new PickleException("invalid pickle data for dateutil gettz call; expected 1 args, got " + args.length)
    // In the case of the dateutil.tz.gettz function call, we're passed one string identifier of the the timezone.
    val identifier = args(0).asInstanceOf[String]
    new Tzinfo(TimeZone.getTimeZone(identifier))
  }

  private def createZoneFromPytz(args: Array[AnyRef]) = {
    if (args.length != 4 && args.length != 1) throw new PickleException("invalid pickle data for pytz timezone; expected 1 or 4 args, got " + args.length)
    // args can be a tuple of 4 values: string timezone identifier, int seconds from utc offset, int seconds for DST, string python timezone name
    // if args came from a pytz.DstTzInfo object
    // Or, args is a tuple of 1 value: string timezone identifier
    // if args came from a pytz.StaticTzInfo object
    // In both cases we can ask the system for a timezone with that identifier and it should find one with that identifier if python did.
    if (!args(0).isInstanceOf[String]) throw new PickleException("invalid pickle data for pytz timezone; expected string argument as first tuple member")
    TimeZone.getTimeZone(args(0).asInstanceOf[String])
  }

  private def createUTC = TimeZone.getTimeZone("UTC")
}