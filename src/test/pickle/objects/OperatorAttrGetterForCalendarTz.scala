package net.razorvine.pickle.objects

import net.razorvine.pickle.IObjectConstructor
import net.razorvine.pickle.PickleException
import java.util.Calendar
import java.util.TimeZone

/**
 * This object constructor is a minimalistic placeholder for operator.itemgetter,
 * it can only be used in the case of unpickling the special pickle created for
 * localizing datetimes with pytz timezones.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
object OperatorAttrGetterForCalendarTz {
  private[objects] class AttrGetterForTz extends IObjectConstructor {
    override def construct(args: Array[AnyRef]): AnyRef = {
      if (args.length != 1 || !args(0).isInstanceOf[TimeZone]) throw new PickleException("expected exactly one TimeZone argument for construction of CalendarLocalizer")
      val tz = args(0).asInstanceOf[TimeZone]
      new OperatorAttrGetterForCalendarTz.CalendarLocalizer(tz)
    }
  }

  private[objects] class CalendarLocalizer(private[objects] val tz: TimeZone) extends IObjectConstructor {
    override def construct(args: Array[AnyRef]): AnyRef = {
      if (args.length != 1 || !args(0).isInstanceOf[Calendar]) throw new PickleException("expected exactly one Calendar argument for construction of Calendar with timezone")
      val cal = args(0).asInstanceOf[Calendar]
      cal.setTimeZone(tz)
      cal
    }
  }
}

class OperatorAttrGetterForCalendarTz extends IObjectConstructor {
  override def construct(args: Array[AnyRef]): AnyRef = {
    if (args.length != 1) throw new PickleException("expected exactly one string argument for construction of AttrGetter")
    if ("localize" == args(0)) new OperatorAttrGetterForCalendarTz.AttrGetterForTz
    else throw new PickleException("expected 'localize' string argument for construction of AttrGetter")
  }
}