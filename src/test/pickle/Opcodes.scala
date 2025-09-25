package net.razorvine.pickle

/**
 * Pickle opcodes. Taken from Python's stdlib pickle.py.
 *
 * @author Irmen de Jong (irmen@razorvine.net)
 */
object Opcodes { // protocol 0 and 1
  // Pickle opcodes. See pickletools.py for extensive docs. The listing
  // here is in kind-of alphabetical order of 1-character pickle code.
  // pickletools groups them by purpose.
  // short datatype because they are UNSIGNED bytes 0..255.
  val MARK = '(' // push special markobject on stack
  val STOP = '.' // every pickle ends with STOP
  val POP = '0' // discard topmost stack item
  val POP_MARK = '1' // discard stack top through topmost markobject
  val DUP = '2' // duplicate top stack item
  val FLOAT = 'F' // push float object; decimal string argument
  val INT = 'I' // push integer or bool; decimal string argument
  val BININT = 'J' // push four-byte signed int (little endian)
  val BININT1 = 'K' // push 1-byte unsigned int
  val LONG = 'L' // push long; decimal string argument
  val BININT2 = 'M' // push 2-byte unsigned int
  val NONE = 'N' // push None
  val PERSID = 'P' // push persistent object; id is taken from string arg
  val BINPERSID = 'Q' // push persistent object; id is taken from stack
  val REDUCE = 'R' // apply callable to argtuple, both on stack
  val STRING = 'S' // push string; NL-terminated string argument
  val BINSTRING = 'T' // push string; counted binary string argument
  val SHORT_BINSTRING = 'U' //push string; counted binary string < 256 bytes
  val UNICODE = 'V' // push Unicode string; raw-unicode-escaped'd argument
  val BINUNICODE = 'X' //push Unicode string; counted UTF-8 string argument
  val APPEND = 'a' // append stack top to list below it
  val BUILD = 'b' // call __setstate__ or __dict__.update()
  val GLOBAL = 'c' // push self.find_class(modname, name); 2 string args
  val DICT = 'd' // build a dict from stack items
  val EMPTY_DICT = '}' // push empty dict
  val APPENDS = 'e' // extend list on stack by topmost stack slice
  val GET = 'g' // push item from memo on stack; index is string arg
  val BINGET = 'h' // push item from memo on stack; index is 1-byte arg
  val INST = 'i' // build & push class instance
  val LONG_BINGET = 'j' // push item from memo on stack; index is 4-byte arg
  val LIST = 'l' // build list from topmost stack items
  val EMPTY_LIST = ']' // push empty list
  val OBJ = 'o' // build & push class instance
  val PUT = 'p' // store stack top in memo; index is string arg
  val BINPUT = 'q' //store stack top in memo; index is 1-byte arg
  val LONG_BINPUT = 'r' // store stack top in memo; index is 4-byte arg
  val SETITEM = 's' // add key+value pair to dict
  val TUPLE = 't' // build tuple from topmost stack items
  val EMPTY_TUPLE = ')' // push empty tuple
  val SETITEMS = 'u' // modify dict by adding topmost key+value pairs
  val BINFLOAT = 'G' // push float; arg is 8-byte float encoding
  val TRUE = "I01\n" // not an opcode; see INT docs in pickletools.py
  val FALSE = "I00\n" // not an opcode; see INT docs in pickletools.py
  // Protocol 2
  val PROTO = 0x80 // identify pickle protocol
  val NEWOBJ = 0x81 // build object by applying cls.__new__ to argtuple
  val EXT1 = 0x82 // push object from extension registry; 1-byte index
  val EXT2 = 0x83 // ditto, but 2-byte index
  val EXT4 = 0x84 // ditto, but 4-byte index
  val TUPLE1 = 0x85 // build 1-tuple from stack top
  val TUPLE2 = 0x86 // build 2-tuple from two topmost stack items
  val TUPLE3 = 0x87 // build 3-tuple from three topmost stack items
  val NEWTRUE = 0x88 // push True
  val NEWFALSE = 0x89 // push False
  val LONG1 = 0x8a // push long from < 256 bytes
  val LONG4 = 0x8b // push really big long
  // Protocol 3 (Python 3.x)
  val BINBYTES = 'B' // push bytes; counted binary string argument
  val SHORT_BINBYTES = 'C' // "     " ; "      " "      " < 256 bytes
  // Protocol 4 (Python 3.4+)
  val SHORT_BINUNICODE = 0x8c // push short string; UTF-8 length < 256 bytes
  val BINUNICODE8 = 0x8d // push very long string
  val BINBYTES8 = 0x8e // push very long bytes string
  val EMPTY_SET = 0x8f // push empty set on the stack
  val ADDITEMS = 0x90 // modify set by adding topmost stack items
  val FROZENSET = 0x91 // build frozenset from topmost stack items
  val MEMOIZE = 0x94 // store top of the stack in memo
  val FRAME = 0x95 // indicate the beginning of a new frame
  val NEWOBJ_EX = 0x92 // like NEWOBJ but work with keyword only arguments
  val STACK_GLOBAL = 0x93 // same as GLOBAL but using names on the stacks
  // Protocol 5 (Python 3.8+)
  val BYTEARRAY8 = 0x96 // push bytearray
  val NEXT_BUFFER = 0x97 // push next out-of-band buffer
  val READONLY_BUFFER = 0x98 //  make top of stack readonly
}