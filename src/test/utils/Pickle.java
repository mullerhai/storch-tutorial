package torch.utils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.io.StreamCorruptedException;

/**
 * Objects to byte arrays and vice versa
 * 
 * @author filip
 * 
 */
public class Pickle {
	public static Object load(byte[] pickle) throws StreamCorruptedException,
			IOException, ClassNotFoundException {
		ByteArrayInputStream instream = new ByteArrayInputStream(pickle);
		ObjectInput oin = new ObjectInputStream(instream);
		Object o = oin.readObject();
		oin.close();
		instream.close();
		return o;
	}

	public static byte[] dump(Serializable s) throws IOException {
		ByteArrayOutputStream outstream = new ByteArrayOutputStream();
		ObjectOutput oout = new ObjectOutputStream(outstream);
		oout.writeObject(s);
		byte[] pickle = outstream.toByteArray();
		oout.close();
		outstream.close();
		return pickle;
	}
}