//package torch.keystone
//
//
//import java.awt.image.{DataBufferByte, BufferedImage}
//
//import java.awt.image.{BufferedImage, DataBufferByte}
//import java.io.InputStream
//import javax.imageio.ImageIO
//import java.awt.image.BufferedImage
//import java.io.{File, InputStream}
//import javax.imageio.ImageIO
//
////import keystoneml.pipelines.Logging
//
//object ImageUtils extends Logging {
//
//  /**
//   * Load image from file.
//   *
//   * @param fileBytes Bytes of an input file.
//   * @return
//   */
//  def loadImage(fileBytes: InputStream): Option[Image] = {
//    classOf[ImageIO].synchronized {
//      try {
//        val img = ImageIO.read(fileBytes)
//        if (img != null) {
//          if (img.getHeight() < 36 || img.getWidth() < 36) {
//            println(s"Ignoring SMALL IMAGE ${img.getHeight}x${img.getWidth()}")
//            None
//          } else {
//            if (img.getType() == BufferedImage.TYPE_3BYTE_BGR) {
//              val imgW = ImageConversions.bufferedImageToWrapper(img)
//              Some(imgW)
//            } else if (img.getType() == BufferedImage.TYPE_BYTE_GRAY) {
//              val imgW = ImageConversions.grayScaleImageToWrapper(img)
//              Some(imgW)
//            } else {
//              println(s"Ignoring image, not RGB or Grayscale of type ${img.getType}")
//              None
//            }
//          }
//        } else {
//          println(s"Failed to parse image, (result was null)")
//          None
//        }
//      } catch {
//        case e: Exception =>
//          println(s"Failed to parse image: due to ${e.getMessage}")
//          None
//      }
//    }
//  }
//
//  /**
//   * Writes image to file `fname`
//   *
//   * If Image is non-standard (that is, values not in [0,255], the "scale"
//   * argument can be passed. Currently assumes a 3 or 1-dimensional image.
//   *
//   * @param fname Destination filename.
//   * @param in    Input image.
//   * @param scale Scale image to [0,255]
//   * @return
//   */
//  def writeImage(fname: String, in: Image, scale: Boolean = false) = {
//    val bi = ImageConversions.imageToBufferedImage(in, scale)
//    val outf = new File(fname)
//    ImageIO.write(bi, "png", outf)
//  }
//
//
//  /**
//   * Converts an input image to Grayscale according to the NTSC standard weights for RGB images and
//   * using sqrt sum of squares for images with other numbers of channels.
//   *
//   * @param in Input image.
//   * @return Grayscaled image.
//   */
//  def toGrayScale(in: Image): Image = {
//    // From the Matlab docs for rgb2gray:
//    // rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B
//    // components: 0.2989 * R + 0.5870 * G + 0.1140 * B
//
//    val numChannels = in.metadata.numChannels
//    val out = new ChannelMajorArrayVectorizedImage(new Array(in.metadata.xDim * in.metadata.yDim),
//      ImageMetadata(in.metadata.xDim, in.metadata.yDim, 1))
//    var i = 0
//    while (i < in.metadata.xDim) {
//      var j = 0
//      while (j < in.metadata.yDim) {
//        var sumSq = 0.0
//        var k = 0
//        if (numChannels == 3) {
//          // Assume data is in BGR order. Todo - we should check the metadata for this.
//          val px = 0.2989 * in.get(i, j, 2) + 0.5870 * in.get(i, j, 1) + 0.1140 * in.get(i, j, 0)
//          out.put(i, j, 0, px)
//        }
//        else {
//          while (k < numChannels) {
//            sumSq = sumSq + (in.get(i, j, k) * in.get(i, j, k))
//            k = k + 1
//          }
//          val px = math.sqrt(sumSq / numChannels)
//          out.put(i, j, 0, px)
//        }
//        j = j + 1
//      }
//      i = i + 1
//    }
//    out
//  }
//
//  /**
//   * Apply a function to every pixel in the image.
//   * NOTE: This function creates a copy of the input image and does not affect the input image.
//   *
//   * @param in  image to apply function to
//   * @param fun function that maps pixels from input to output
//   * @return new image that is the result of applying the function.
//   */
//  def mapPixels(in: Image, fun: Double => Double): Image = {
//    val out = new ChannelMajorArrayVectorizedImage(
//      new Array[Double](in.metadata.xDim * in.metadata.yDim * in.metadata.numChannels),
//      ImageMetadata(in.metadata.xDim, in.metadata.yDim, in.metadata.numChannels))
//
//    var x, y, c = 0
//    while (x < in.metadata.xDim) {
//      y = 0
//      while (y < in.metadata.yDim) {
//        c = 0
//        while (c < in.metadata.numChannels) {
//          out.put(x, y, c, fun(in.get(x, y, c)))
//          c += 1
//        }
//        y += 1
//      }
//      x += 1
//    }
//    out
//  }
//
//  /**
//   * Crop an input image to the given bounding box described by
//   * (startX, startY, endX, endY).
//   *
//   * @param in     image to crop
//   * @param startX x-position (inclusive) to describe upper left corner of BB
//   * @param startY y-position (inclusive) to describe upper left corner of BB
//   * @param endX   x-position (exclusive) to describe lower right corner of BB
//   * @param endY   y-position (exclusive) to describe lower right corner of BB
//   * @return new image of size (endX - startX, endY - startY)
//   */
//  def crop(im: Image, startX: Int, startY: Int, endX: Int, endY: Int): Image = {
//    val xDim = im.metadata.xDim
//    val yDim = im.metadata.yDim
//    val nChannels = im.metadata.numChannels
//
//    if (startX < 0 || startX > xDim || endX < 0 || endX > xDim)
//      throw new IllegalArgumentException("invalid x coordiate given")
//    if (startY < 0 || startY > yDim || endY < 0 || endY > yDim)
//      throw new IllegalArgumentException("invalid y coordinate given")
//    if (startX > endX || startY > endY)
//      throw new IllegalArgumentException("startX > endX or startY > endY encountered")
//
//    val newXDim = endX - startX
//    val newYDim = endY - startY
//
//    val out = new Array[Double](newXDim * newYDim * nChannels)
//
//    var c = 0
//    while (c < nChannels) {
//      var s = startX
//      while (s < endX) {
//        var b = startY
//        while (b < endY) {
//          out(c + (s - startX) * nChannels +
//            (b - startY) * (endX - startX) * nChannels) = im.get(s, b, c)
//          b = b + 1
//        }
//        s = s + 1
//      }
//      c = c + 1
//    }
//
//    new ChannelMajorArrayVectorizedImage(out, ImageMetadata(newXDim, newYDim, nChannels))
//  }
//
//  /**
//   * Combine two images applying a function on corresponding pixels.
//   * Requires both images to be of the same size
//   *
//   * @param in  First input image
//   * @param in2 Second input image
//   * @param fun Function that takes in a pair of pixels and returns the pixel in the combined image
//   * @return Combined image
//   */
//  def pixelCombine(in: Image, in2: Image, fun: (Double, Double) => Double = _ + _): Image = {
//    require(in.metadata.xDim == in2.metadata.xDim &&
//      in.metadata.yDim == in2.metadata.yDim &&
//      in.metadata.numChannels == in2.metadata.numChannels,
//      "Images must have the same dimension.")
//
//    val out = new ChannelMajorArrayVectorizedImage(
//      new Array[Double](in.metadata.xDim * in.metadata.yDim * in.metadata.numChannels),
//      ImageMetadata(in.metadata.xDim, in.metadata.yDim, in.metadata.numChannels))
//
//    var x, y, c = 0
//    while (x < in.metadata.xDim) {
//      y = 0
//      while (y < in.metadata.yDim) {
//        c = 0
//        while (c < in.metadata.numChannels) {
//          out.put(x, y, c, fun(in.get(x, y, c), in2.get(x, y, c)))
//          c += 1
//        }
//        y += 1
//      }
//      x += 1
//    }
//    out
//  }
//
//
//  /**
//   * Convolves images with two one-dimensional filters.
//   *
//   * @param img     Image to be convolved.
//   * @param xFilter Horizontal convolution filter.
//   * @param yFilter Vertical convolution filter.
//   * @return Convolved image
//   */
//  def conv2D(img: Image, xFilter: Array[Double], yFilter: Array[Double]): Image = {
//    val paddedXDim = img.metadata.xDim + xFilter.length - 1
//    val paddedYDim = img.metadata.yDim + yFilter.length - 1
//    val imgPadded = new RowMajorArrayVectorizedImage(new Array[Double](paddedXDim * paddedYDim *
//      img.metadata.numChannels), ImageMetadata(paddedXDim, paddedYDim, img.metadata.numChannels))
//
//    val xPadLow = math.floor((xFilter.length - 1).toFloat / 2).toInt
//    // Since we go from 0 to paddedXDim
//    val xPadHigh = (paddedXDim - 1) - math.ceil((xFilter.length - 1).toFloat / 2).toInt
//
//    val yPadLow = math.floor((yFilter.length - 1).toFloat / 2).toInt
//    // Since we go from 0 to paddedYDim
//    val yPadHigh = (paddedYDim - 1) - math.ceil((yFilter.length - 1).toFloat / 2).toInt
//
//    var c = 0
//    while (c < img.metadata.numChannels) {
//      var y = 0
//      while (y < paddedYDim) {
//        var yVal = -1
//        if (y < yPadLow || y > yPadHigh) {
//          yVal = 0
//        }
//        var x = 0
//        while (x < paddedXDim) {
//          var xVal = -1
//          if (x < xPadLow || x > xPadHigh) {
//            xVal = 0
//          }
//
//          var px = 0.0
//          if (!(xVal == 0 || yVal == 0)) {
//            px = img.get(x - xPadLow, y - yPadLow, c)
//          }
//          imgPadded.put(x, y, c, px)
//          x = x + 1
//        }
//        y = y + 1
//      }
//      c = c + 1
//    }
//
//    val xFilterToUse = xFilter.reverse
//    val yFilterToUse = yFilter.reverse
//    val imgChannels = imgPadded.metadata.numChannels
//    val imgWidth = imgPadded.metadata.yDim
//    val imgHeight = imgPadded.metadata.xDim
//
//    val resWidth = imgWidth - yFilterToUse.length + 1
//    val resHeight = imgHeight - xFilterToUse.length + 1
//
//    // Storage area for intermediate output.
//    val midres = new ColumnMajorArrayVectorizedImage(
//      new Array[Double](resHeight * imgWidth * imgChannels),
//      ImageMetadata(resHeight, imgWidth, imgChannels))
//
//    // Storage for final output.
//    val res = new ColumnMajorArrayVectorizedImage(
//      new Array[Double](resWidth * resHeight * imgChannels),
//      ImageMetadata(resHeight, resWidth, imgChannels))
//
//    // First we do the rows.
//    var x = 0
//    var y, chan, i = 0
//    var tmp = 0.0
//
//    while (chan < imgChannels) {
//      y = 0
//      while (y < imgWidth) {
//        x = 0
//        while (x < resHeight) {
//          i = 0
//          tmp = 0.0
//          var idxToGet = x + y * paddedXDim + chan * paddedXDim * paddedYDim
//          while (i < xFilterToUse.length) {
//            tmp += imgPadded.getInVector(idxToGet + i) * xFilterToUse(i)
//            i += 1
//          }
//          midres.put(x, y, chan, tmp)
//          x += 1
//        }
//        y += 1
//      }
//      chan += 1
//    }
//
//    // Then we do the columns.
//    x = 0
//    y = 0
//    chan = 0
//    i = 0
//
//    while (chan < imgChannels) {
//      x = 0
//      while (x < resHeight) {
//        y = 0
//        while (y < resWidth) {
//          val idxToPut = y + x * resWidth + chan * resWidth * resHeight
//          var idxToGet = y + x * imgWidth + chan * imgWidth * resHeight
//          i = 0
//          tmp = 0.0
//          while (i < yFilterToUse.length) {
//            tmp += midres.getInVector(idxToGet + i) * yFilterToUse(i)
//            i += 1
//          }
//          res.putInVector(idxToPut, tmp)
//          y += 1
//        }
//        x += 1
//      }
//      chan += 1
//    }
//    res
//  }
//
//  /**
//   * Split an image into a number of images, one per channel of input image.
//   *
//   * @param in Input image to be split
//   * @return Array of images, one per channel of input image
//   */
//  def splitChannels(in: Image): Array[Image] = {
//    val out = new Array[Image](in.metadata.numChannels)
//    var c = 0
//    while (c < in.metadata.numChannels) {
//      val a = ChannelMajorArrayVectorizedImage(
//        new Array[Double](in.metadata.xDim * in.metadata.yDim),
//        ImageMetadata(in.metadata.xDim, in.metadata.yDim, 1))
//      var x = 0
//      while (x < in.metadata.xDim) {
//        var y = 0
//        while (y < in.metadata.yDim) {
//          a.put(x, y, 0, in.get(x, y, c))
//          y = y + 1
//        }
//        x = x + 1
//      }
//      out(c) = a
//      c = c + 1
//    }
//    out
//  }
//
//  /**
//   * Flip the image such that 
//   * flipImage(im)(x,y,z) = im(im.metadata.xDim-x-1,im.metadata.yDim-y-1,im.metadata.numChannels-z-1)
//   * for all valid (x,y,z).
//   *
//   * @param im An input image.
//   * @return A flipped image.
//   */
//  def flipImage(im: Image): Image = {
//    val size = im.metadata.xDim * im.metadata.yDim * im.metadata.numChannels
//    val res = new ChannelMajorArrayVectorizedImage(Array.fill[Double](size)(0.0), im.metadata)
//
//    for (
//      x <- 0 until im.metadata.xDim;
//      y <- 0 until im.metadata.yDim;
//      c <- 0 until im.metadata.numChannels
//    ) {
//      res.put(im.metadata.xDim - x - 1, im.metadata.yDim - y - 1, im.metadata.numChannels - c - 1, im.get(x, y, c))
//    }
//
//    res
//  }
//
//  /**
//   * Flip the image horizontally
//   * flipImage(im)(x,y,z) = im(x, im.metadata.yDim-y-1, z)
//   * for all valid (x,y,z).
//   *
//   * @param im An input image.
//   * @return A flipped image.
//   */
//  def flipHorizontal(im: Image): Image = {
//    val size = im.metadata.xDim * im.metadata.yDim * im.metadata.numChannels
//    val res = new ChannelMajorArrayVectorizedImage(Array.fill[Double](size)(0.0), im.metadata)
//
//    var cIdx = 0
//    while (cIdx < im.metadata.numChannels) {
//      var xIdx = 0
//      while (xIdx < im.metadata.xDim) {
//        var yIdxDest = im.metadata.yDim - 1
//        var yIdxSource = 0
//        while (yIdxDest >= 0) {
//          res.put(xIdx, yIdxDest, cIdx, im.get(xIdx, yIdxSource, cIdx))
//          yIdxDest = yIdxDest - 1
//          yIdxSource = yIdxSource + 1
//        }
//
//        xIdx += 1
//      }
//      cIdx += 1
//    }
//    res
//  }
//}
////import keystoneml.pipelines._
//
///**
// * A wrapper trait for images that might be stored in various ways.  Be warned
// * that using this wrapper probably introduces some inefficiency.  Also, images
// * are currently treated as immutable, which may introduce a serious
// * performance problem; in the future we may need to add a set() method.
// *
// * If you have a choice and performance matters to you, use
// * ChannelMajorArrayVectorizedImage, as it is likely to be the most efficient
// * implementation.
// */
//trait Image {
//  val metadata: ImageMetadata
//
//  /**
//   * Get the pixel value at (x, y, channelIdx).  Channels are indexed as
//   * follows:
//   *   - If the image is RGB, 0 => blue, 1 => green, 2 => red.
//   *   - If the image is RGB+alpha, 0 => blue, 1=> green, 2 => red, and
//   *     3 => alpha.
//   *   - Other channel schemes are unsupported; the only reason this matters
//   *     is that input converters (e.g. from BufferedImage to Image) need to
//   *     handle channels consistently.
//   */
//  def get(x: Int, y: Int, channelIdx: Int): Double
//
//  /**
//   * Put a pixel value at (x, y, channelIdx).
//   */
//  def put(x: Int, y: Int, channelIdx: Int, newVal: Double)
//
//  /**
//   * Returns a flat version of the image, represented as a single array.
//   * It is indexed as follows: The pixel value for (x, y, channelIdx)
//   * is at channelIdx + x*numChannels + y*numChannels*xDim.
//   *
//   * This implementation works for arbitrary image formats but it is
//   * inefficient.
//   */
//  def toArray: Array[Double] = {
//    val flat = new Array[Double](this.flatSize)
//    var y = 0
//    while (y < this.metadata.yDim) {
//      val runningOffsetY = y*this.metadata.numChannels*this.metadata.xDim
//      var x = 0
//      while (x < this.metadata.xDim) {
//        val runningOffsetX = runningOffsetY + x*this.metadata.numChannels
//        var channelIdx = 0
//        while (channelIdx < this.metadata.numChannels) {
//          flat(channelIdx + runningOffsetX) = get(x, y, channelIdx)
//          channelIdx += 1
//        }
//        x += 1
//      }
//      y += 1
//    }
//    flat
//  }
//
//  def getSingleChannelAsIntArray(): Array[Int] = {
//    if (this.metadata.numChannels > 1) {
//      throw new RuntimeException(
//        "Cannot call getSingleChannelAsIntArray on an image with more than one channel.")
//    }
//    var index = 0;
//    var flat = new Array[Int](this.metadata.xDim*this.metadata.yDim)
//    (0 until metadata.xDim).map({ x =>
//      (0 until metadata.yDim).map({ y =>
//        val px = get(x, y, 0);
//        if(px < 1) {
//          flat(index) = (255*px).toInt
//        }
//        else {
//          flat(index) = math.round(px).toInt
//        }
//        index += 1
//      })
//    })
//    flat
//  }
//
//  def getSingleChannelAsFloatArray(): Array[Float] = {
//    if (this.metadata.numChannels > 1) {
//      throw new RuntimeException(
//        "Cannot call getSingleChannelAsFloatArray on an image with more than one channel.")
//    }
//    var index = 0;
//    var flat = new Array[Float](this.metadata.xDim*this.metadata.yDim)
//    (0 until metadata.yDim).map({ y =>
//      (0 until metadata.xDim).map({ x =>
//        flat(index) = get(x, y, 0).toFloat
//        index += 1
//      })
//    })
//    flat
//  }
//
//  def flatSize: Int = {
//    metadata.numChannels*metadata.xDim*metadata.yDim
//  }
//
//
//  /**
//   * An inefficient implementation of equals().  Subclasses should override
//   * this if they can implement it more cheaply and anyone cares about such
//   * things.
//   */
//  override def equals(o: Any): Boolean = {
//    if (o == null || !o.isInstanceOf[Image]) {
//      false
//    } else {
//      val other = o.asInstanceOf[Image]
//      if (!this.metadata.equals(other.metadata)) {
//        false
//      } else {
//        for (xIdx <- (0 until metadata.xDim);
//             yIdx <- (0 until metadata.yDim);
//             channelIdx <- (0 until metadata.numChannels)) {
//          if (this.get(xIdx, yIdx, channelIdx) != other.get(xIdx, yIdx, channelIdx)) {
//            return false
//          }
//        }
//        true
//      }
//    }
//  }
//}
//
///**
// * Contains metadata about the storage format of an image.
// *
// * @param xDim is the height of the image(!)
// * @param yDim is the width of the image
// * @param numChannels is the number of color channels in the image
// */
//case class ImageMetadata(xDim: Int, yDim: Int, numChannels: Int)
//
///**
// * Wraps a byte array, where a byte is a color channel value.  This is the
// * format generated by Java's JPEG parser.
// *
// * VectorizedImage is indexed as follows: The pixel value for (x, y, channelIdx)
// *   is at channelIdx + y*numChannels + x*numChannels*yDim.
// */
//case class ByteArrayVectorizedImage(
//                                     vectorizedImage: Array[Byte],
//                                     override val metadata: ImageMetadata) extends VectorizedImage {
//  def imageToVectorCoords(x: Int, y: Int, channelIdx: Int): Int = {
//    channelIdx + y*metadata.numChannels + x*metadata.yDim*metadata.numChannels
//  }
//
//  def vectorToImageCoords(v: Int): Coordinate = {
//    coord.x = v / (metadata.yDim * metadata.numChannels)
//    coord.y = (v - (coord.x * metadata.yDim * metadata.numChannels)) / metadata.numChannels
//    coord.channelIdx = v - coord.y * metadata.numChannels - coord.x * metadata.yDim * metadata.numChannels
//    coord
//  }
//
//  // FIXME: This is correct but inefficient - every time we access the image we
//  // use several method calls (which are hopefully inlined) and a conversion
//  // from byte to double (which hopefully at least does not involve any
//  // boxing).
//  override def getInVector(vectorIdx: Int) = {
//    val signedValue = vectorizedImage(vectorIdx)
//    if (signedValue < 0) {
//      signedValue + 256
//    } else {
//      signedValue
//    }
//  }
//
//  override def putInVector(vectorIdx: Int, newVal: Double) = ???
//}
//
///**
// * VectorizedImage that indexed as follows: The pixel value for
// * (x, y, channelIdx) is at channelIdx + x*numChannels + y*numChannels*xDim.
// */
//case class ChannelMajorArrayVectorizedImage(
//                                             vectorizedImage: Array[Double],
//                                             override val metadata: ImageMetadata) extends VectorizedImage {
//  override def imageToVectorCoords(x: Int, y: Int, channelIdx: Int): Int = {
//    channelIdx + x * metadata.numChannels + y * metadata.xDim * metadata.numChannels
//  }
//
//  override def vectorToImageCoords(v: Int): Coordinate = {
//    coord.y = v / (metadata.xDim * metadata.numChannels)
//    coord.x = (v - (coord.y * metadata.xDim * metadata.numChannels)) / metadata.numChannels
//    coord.channelIdx = v - coord.x * metadata.numChannels - coord.y * metadata.xDim * metadata.numChannels
//    coord
//  }
//
//  override def getInVector(vectorIdx: Int) = vectorizedImage(vectorIdx)
//
//
//  override def putInVector(vectorIdx: Int, newVal: Double) = {
//    vectorizedImage(vectorIdx) = newVal
//  }
//
//  override def toArray = vectorizedImage
//}
//
///**
// * VectorizedImage that is indexed as follows: The pixel value for (x, y, channelIdx)
// *   is at y + x*yDim + channelIdx*yDim*xDim
// */
//case class ColumnMajorArrayVectorizedImage(
//                                            vectorizedImage: Array[Double],
//                                            override val metadata: ImageMetadata) extends VectorizedImage {
//  override def imageToVectorCoords(x: Int, y: Int, channelIdx: Int): Int = {
//    val cidx = channelIdx
//    y + x * metadata.yDim + cidx * metadata.yDim * metadata.xDim
//  }
//
//  override def vectorToImageCoords(v: Int): Coordinate = {
//    coord.channelIdx = v / (metadata.xDim * metadata.yDim)
//    coord.x = (v - (coord.channelIdx * metadata.xDim * metadata.yDim)) / metadata.yDim
//    coord.y = v - coord.x * metadata.yDim - coord.channelIdx * metadata.yDim * metadata.xDim
//    coord
//  }
//
//  override def getInVector(vectorIdx: Int) = {
//    vectorizedImage(vectorIdx)
//  }
//
//  override def putInVector(vectorIdx: Int, newVal: Double) = {
//    vectorizedImage(vectorIdx) = newVal
//  }
//}
//
///**
// * VectorizedImage which is indexed as follows: The pixel value for
// * (x, y, channelIdx) is at x + y*xDim + channelIdx*xDim*yDim.
// */
//case class RowMajorArrayVectorizedImage(
//                                         vectorizedImage: Array[Double],
//                                         override val metadata: ImageMetadata) extends VectorizedImage {
//  override def imageToVectorCoords(x: Int, y: Int, channelIdx: Int): Int = {
//    x + y * metadata.xDim + channelIdx * metadata.xDim * metadata.yDim
//  }
//
//  override def vectorToImageCoords(v: Int): Coordinate = {
//    coord.channelIdx = v / (metadata.xDim * metadata.yDim)
//    coord.y = (v - coord.channelIdx * metadata.xDim * metadata.yDim) / metadata.xDim
//    coord.x = v - coord.y * metadata.xDim - coord.channelIdx * metadata.xDim * metadata.yDim
//    coord
//  }
//
//  override def getInVector(vectorIdx: Int) = vectorizedImage(vectorIdx)
//
//  override def putInVector(vectorIdx: Int, newVal: Double) = {
//    vectorizedImage(vectorIdx) = newVal
//  }
//}
//
//
///**
// * Helper trait for implementing Images that wrap vectorized representations
// * of images.
// */
//trait VectorizedImage extends Image {
//  def imageToVectorCoords(x: Int, y: Int, channelIdx: Int): Int
//
//  def getInVector(vectorIdx: Int): Double
//
//  def putInVector(vectorIdx: Int, newVal: Double): Unit
//
//  override def get(x: Int, y: Int, channelIdx: Int) = {
//    getInVector(imageToVectorCoords(x, y, channelIdx))
//  }
//
//  override def put(x: Int, y: Int, channelIdx: Int, newVal: Double) = {
//    putInVector(imageToVectorCoords(x, y, channelIdx), newVal)
//  }
//
//  def vectorToImageCoords(v: Int): Coordinate
//
//  @transient lazy protected val coord: Coordinate = new Coordinate(0,0,0)
//
//  /**
//   * Returns an iterator of coordinate values based on the "natural" order
//   * of a Vectorized image. That is, this returns a value of the form (x,y,channel,value)
//   * in order.
//   *
//   * This method is optimized to avoid unnecessary memory allocation and designed
//   * to approach the performance of an equivalent `while` loop over the image pixels for
//   * speeding up things like Aggregation over an image regardless of underlying image ordering.
//   *
//   * An important restriction is that the reference to the returned `CoordinateValue`
//   * should not be modified or saved by the caller.
//   *
//   * @return
//   */
//  def iter(): Iterator[CoordinateValue] = new Iterator[CoordinateValue] {
//    var i = 0
//    val totSize = metadata.xDim*metadata.yDim*metadata.numChannels
//    var tup: Coordinate = null
//    var v: Double = 0.0
//    var cv: CoordinateValue = new CoordinateValue(0,0,0,0.0)
//
//    def hasNext = i < totSize
//
//    def next() = {
//      tup = vectorToImageCoords(i)
//      v = getInVector(i)
//      i += 1
//      cv.x = tup.x
//      cv.y = tup.y
//      cv.channelIdx = tup.channelIdx
//      cv.v = v
//      cv
//    }
//  }
//}
//
//class Coordinate(var x: Int, var y: Int, var channelIdx: Int)
//class CoordinateValue(var x: Int, var y: Int, var channelIdx: Int, var v: Double)
//
///**
// * Wraps a double array.
// *
// * @param vectorizedImage is indexed as follows: The pixel value for (x, y, channelIdx)
// *   is at y + x.metadata.yDim + channelIdx*metadata.yDim*metadata.xDim
// * @param metadata Image metadata.
// */
//case class RowColumnMajorByteArrayVectorizedImage(
//                                                   vectorizedImage: Array[Byte],
//                                                   override val metadata: ImageMetadata) extends VectorizedImage {
//  def imageToVectorCoords(x: Int, y: Int, channelIdx: Int): Int = {
//    val cidx = channelIdx
//
//    y + x*metadata.yDim + cidx*metadata.yDim*metadata.xDim
//  }
//
//  override def vectorToImageCoords(v: Int): Coordinate = {
//    coord.channelIdx = v / (metadata.xDim * metadata.yDim)
//    coord.x = (v - (coord.channelIdx * metadata.xDim * metadata.yDim)) / metadata.yDim
//    coord.y = v - coord.x * metadata.yDim - coord.channelIdx * metadata.yDim * metadata.xDim
//    coord
//  }
//
//  // FIXME: This is correct but inefficient - every time we access the image we
//  // use several method calls (which are hopefully inlined) and a conversion
//  // from byte to double (which hopefully at least does not involve any
//  // boxing).
//  override def getInVector(vectorIdx: Int) = {
//    val signedValue = vectorizedImage(vectorIdx)
//    if (signedValue < 0) {
//      signedValue + 256
//    } else {
//      signedValue
//    }
//  }
//
//  override def putInVector(vectorIdx: Int, newVal: Double) = ???
//}
//
///**
// * Represents a labeled image.
// *
// * @tparam L Type of the label.
// */
//trait AbstractLabeledImage[L] {
//  def image: Image
//  def label: L
//  def filename: Option[String]
//}
//
///**
// * A labeled image. Commonly used in Image classification.
// *
// * @param image An Image.
// * @param label A label. Should be in [0 .. K] where K is some number of unique labels.
// */
//case class LabeledImage(image: Image, label: Int, filename: Option[String] = None)
//  extends AbstractLabeledImage[Int]
//
///**
// * A multilabeled image. Commonly used in Image classification.
// *
// * @param image An Image.
// * @param label A set of labels. Should be an array with all elements in [0 .. K]
// *              where K is some number of unique labels.
// * @param filename A filename where this image was found. Useful for debugging.
// */
//case class MultiLabeledImage(image: Image, label: Array[Int], filename: Option[String] = None)
//  extends AbstractLabeledImage[Array[Int]]
//
//
//
//
//object ImageConversions {
//  /**
//   * Copied in small part from Mota's code here:
//   *   http://stackoverflow.com/a/9470843
//   */
//  def bufferedImageToWrapper(image: BufferedImage): Image = {
//    val pixels = image.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData()
//    val xDim = image.getHeight()
//    val yDim = image.getWidth()
//    val hasAlphaChannel = image.getAlphaRaster() != null
//    val numChannels = image.getType() match {
//      case BufferedImage.TYPE_3BYTE_BGR => 3
//      case BufferedImage.TYPE_4BYTE_ABGR => 4
//      case BufferedImage.TYPE_4BYTE_ABGR_PRE => 4
//      case BufferedImage.TYPE_BYTE_GRAY => 1
//      case _ => throw new RuntimeException("Unexpected Image Type " + image.getType())
//    }
//    val metadata = ImageMetadata(xDim, yDim, numChannels)
//    ByteArrayVectorizedImage(pixels, metadata)
//  }
//
//  def grayScaleImageToWrapper(image: BufferedImage): Image = {
//    val pixels = image.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData()
//    val xDim = image.getHeight()
//    val yDim = image.getWidth()
//    val numChannels = 3
//    val metadata = ImageMetadata(xDim, yDim, numChannels)
//
//    // Concatenate the grayscale image thrice to get three channels.
//    // TODO(shivaram): Is this the right thing to do ?
//    val allPixels = pixels.flatMap(p => Seq(p, p, p))
//    ByteArrayVectorizedImage(allPixels, metadata)
//  }
//
//
//  /**
//   * Converts an image to a buffered image.
//   * If Image is non-standard (that is, values not in (0,255), the "scale"
//   * argument can be passed. Currently assumes a 3 or 1-dimensional image.
//   * @param im An Image.
//   * @param scale Boolean indicating whether to scale or not.
//   * @return
//   */
//  def imageToBufferedImage(im: Image, scale: Boolean=false): BufferedImage = {
//    val canvas = new BufferedImage(im.metadata.yDim, im.metadata.xDim, BufferedImage.TYPE_INT_RGB)
//
//    //Scaling
//    val scalef: Double => Int = if (scale) {
//      val immin = im.toArray.min
//      val immax = im.toArray.max
//      d: Double => (255*(d-immin)/immax).toInt
//    } else {
//      d: Double => d.toInt
//    }
//
//    var x = 0
//    while (x < im.metadata.xDim) {
//      var y = 0
//      while (y < im.metadata.yDim) {
//
//        //Scale and pack into an rgb pixel.
//        val chanArr = im.metadata.numChannels match {
//          case 1 => Array(0,0,0)
//          case 3 => Array(0,1,2)
//        }
//
//        val pixVals = chanArr.map(c => im.get(x, y, c)).map(scalef)
//        val pix = (pixVals(0) << 16) | (pixVals(1) << 8) | pixVals(2)
//
//        //Note, BufferedImage has opposite canvas coordinate system from us.
//        //E.g. their x,y is our y,x.
//        canvas.setRGB(y, x, pix)
//        y += 1
//      }
//      x += 1
//    }
//
//    canvas
//  }
//
//}