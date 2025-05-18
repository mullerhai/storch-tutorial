package torch
package basic

import org.bytedeco.pytorch.global.torch as torchNative
import torch.internal.NativeConverters.fromNative
import torch.{---, ::, Slice}

object TensorSliceOps01 {

  //2d  Tensor
  def indexSelectRows(): Unit = {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    println(s"First row: ${tensor(0)}")
    println(s"Second row: ${tensor(1)}")
    println(s"Third row: ${tensor(2)}")
    println(s"Forth row: ${tensor(3)}")
    println(s"Last row: ${tensor(-1)}")
    println("Read row finish \r\t")
  }

  //2d Tensor
  def indexSelectColumns(): Unit = {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    println(s"First column: ${tensor(Slice(), 0)}")
    println(s"Second column: ${tensor(Slice(), 1)}")
    println(s"Third column: ${tensor(Slice(), 2)}")
    println(s"Forth column: ${tensor(Slice(), 3)}")
    println(s"Last column: ${tensor(Slice(), -1)}")
    println("Read column finish \r\t")
  }

  //2d Tensor
  def indexSelectColumnsTwo(): Unit = {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    println(s"First column: ${tensor(---, 0)}")
    println(s"Second column: ${tensor(---, 1)}")
    println(s"Third column: ${tensor(---, 2)}")
    println(s"Forth column: ${tensor(---, 3)}")
    println(s"Last column: ${tensor(---, -1)}")
    println("Read column finish \r\t")
  }

  //2d Tensor
  def indexSelectColumnsThree(): Unit = {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    println(s"First column: ${tensor(::, 0)}")
    println(s"Second column: ${tensor(::, 1)}")
    println(s"Third column: ${tensor(::, 2)}")
    println(s"Forth column: ${tensor(::, 3)}")
    println(s"Last column: ${tensor(::, -1)}")

    println("Read column finish \r\t")
  }

  def indexSelectColumnsFour(): Unit = {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    println(s"column select two columns ,index[0,1,2,3] : ${tensor(::, 0.::(1))}")
    println(s"column select two columns ,index[0,2] : ${tensor(::, 0.::(2))}")
    println(s"column select two columns ,index[0,3] : ${tensor(::, 0.::(3))}")
    println(s"column select one columns ,index[0] : ${tensor(::, 0.::(4))}")
    println(s"column select one columns ,index[0] : ${tensor(::, 0.::(5))}")
    println(s"column select one columns ,index[1] : ${tensor(::, 1.::(3))}")
    println("Read column finish \r\t")
  }

  def indexSelectColumnsFour2(): Unit = {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    println(s"column select two columns ,index[0,1,2,3] : ${tensor(---, 0.::(1))}")
    println(s"column select two columns ,index[0,2] : ${tensor(---, 0.::(2))}")
    println(s"column select two columns ,index[0,3] : ${tensor(---, 0.::(3))}")
    println(s"column select one columns ,index[0] : ${tensor(---, 0.::(4))}")
    println(s"column select one columns ,index[0] : ${tensor(---, 0.::(5))}")
    println(s"column select one columns ,index[1] : ${tensor(---, 1.::(3))}")
    println("Read column finish \r\t")
  }

  def indexSelectRowsFour(): Unit = {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    println(s"column select two row tensor( 0.::(1)),index[0,1,2,3] : ${tensor(0.::(1))}")
    println(s"column select two row tensor( 0.::(2)),index[0,2] : ${tensor(0.::(2))}")
    println(s"column select two row tensor( 0.::(3)) ,index[0,3] : ${tensor(0.::(3))}")
    println(s"column select one row tensor( 0.::(4)),index[0] : ${tensor(0.::(4))}")
    println(s"column select one row tensor( 0.::(5)),index[0] : ${tensor(0.::(5))}")
    println(s"column select two row tensor( 1.::(3)) ,index[1] : ${tensor(1.::(3))}")

  }

  def indexSelectRowsFifth(): Unit = {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    println(s"column select two row tensor(Seq(0,1)),index[0,1] : ${tensor(Seq(0, 1))}")
    println(s"column select two row tensor(Seq(0,2)),index[0,2] : ${tensor(Seq(0, 2))}")
    println(s"column select two row tensor(Seq(0,3)),index[0,3] : ${tensor(Seq(0, 3))}")
    println(s"column select one row tensor(Seq(3,0)),index[3,0] : ${tensor(Seq(3, 0))}")
    println(s"column select one row tensor(Seq(0,1,3)) index[0,1,3] : ${tensor(Seq(0, 1, 3))}")
    println(s"column select two row tensor(Seq(3,1,0,2)) ,index[3,1,0,2] : ${tensor(Seq(3, 1, 0, 2))}")

  }

  def indexSelectColumnsFifth(): Unit = {
    val tensor = torch.arange(0, 16).reshape(4, 4)
    println(s"column select two Columns tensor(Seq(0,1)),index[0,1] : ${tensor(::, Seq(0, 1))}")
    println(s"column select two Columns tensor(Seq(0,2)),index[0,2] : ${tensor(::, Seq(0, 2))}")
    println(s"column select two Columns tensor(Seq(0,3)),index[0,3] : ${tensor(::, Seq(0, 3))}")
    println(s"column select one Columns tensor(Seq(3,0)),index[3,0] : ${tensor(::, Seq(3, 0))}")
    println(s"column select one Columns tensor(Seq(0,1,3)) index[0,1,3] : ${tensor(::, Seq(0, 1, 3))}")
    println(s"column select two Columns tensor(Seq(3,1,0,2)) ,index[3,1,0,2] : ${tensor(::, Seq(3, 1, 0, 2))}")

  }

  def indexUpdateColumnsSix(): Unit = {
    val tensor = torch.arange(1, 17).reshape(4, 4).to(dtype = torch.float32)
    val zero = torch.zeros(Seq(4, 2))
    tensor.update(indices = Seq(::, Seq(2, 1)), values = zero.to(dtype = torch.float32))
    println(s"Index ${tensor} ")
    //Index tensor dtype=float32, shape=[4, 4], device=CPU
    //[[1.0000, 0.0000, 0.0000, 4.0000],
    // [5.0000, 0.0000, 0.0000, 8.0000],
    // [9.0000, 0.0000, 0.0000, 12.0000],
    // [13.0000, 0.0000, 0.0000, 16.0000]]
  }

  def indexUpdateRowsSix(): Unit = {
    val tensor = torch.arange(1, 17).reshape(4, 4).to(dtype = torch.float32)
    val zero = torch.zeros(Seq(2, 4))
    tensor.update(indices = Seq(Seq(1, 2)), values = zero.to(dtype = torch.float32))
    println(s"Index ${tensor}")
    //Index tensor dtype=float32, shape=[4, 4], device=CPU
    //[[1.0000, 2.0000, 3.0000, 4.0000],
    // [0.0000, 0.0000, 0.0000, 0.0000],
    // [0.0000, 0.0000, 0.0000, 0.0000],
    // [13.0000, 14.0000, 15.0000, 16.0000]]
  }

  def indexBackward(): Unit = {
    val x = torch.ones(Seq(2, 2), requires_grad = true)
    val y = x * 3 + 2
    val gfn = y.grad_fn()
    println(s"y ${y.grad_fn()}")
    println(s"y ${gfn.getptr().name().getString}")
    println(s"y ${gfn.getptr.name().getString}")
  }

  //https://blog.csdn.net/weicao1990/article/details/93599947
  def randSelect01(): Unit = {
    val a = torch.rand(Seq(4, 3, 28, 28))
    println(s"a 0 shape ${a(0).shape}") //a 0 shape ArraySeq(3, 28, 28)
    println(s"a 00 shape ${a(Seq(0), Seq(0)).shape}") //a 00 shape ArraySeq(1, 28, 28)
    println(s"a 1234 shape ${a(1, 2, 3, 4)}") // shape tensor dtype=float32, shape=[], device=CPU 0.4458

  }

  //# 选择第一张和第三张图
  //print(a.index_select(0, torch.tensor([0, 2])).shape)
  //
  //# 选择R通道和B通道
  //print(a.index_select(1, torch.tensor([0, 2])).shape)
  //
  //# 选择图像的0~8行
  //print(a.index_select(2, torch.arange(8)).shape)
  //torch.Size([2, 3, 28, 28])
  //torch.Size([4, 2, 28, 28])
  //torch.Size([4, 3, 8, 28])
  def randSelect02(): Unit = {
    val a = torch.rand(Seq(4, 3, 28, 28))
    val aa = a.index_select(0, torch.tensor(Seq(0, 2))).shape
    println(s"aa shape ${aa} , torch.Size([2, 3, 28, 28])")

    val b = a.index_select(1, torch.tensor(Seq(0, 2), requires_grad = false))
    println(s" b shape ${b.shape}, torch.Size([4, 2, 28, 28]) ")
    val c = a.index_select(2, torch.arange(0, 8)).shape
    println(s" c shape ${c} , torch.Size([4, 3, 8, 28])")


  }

  //import torch
  //
  //# 譬如：4张图片，每张三个通道，每个通道28行28列的像素
  //a = torch.rand(4, 3, 28, 28)
  //
  //# 在第一个维度上取后0和1，等同于取第一、第二张图片
  //print(a[:2].shape)
  //
  //# 在第一个维度上取0和1,在第二个维度上取0，
  //# 等同于取第一、第二张图片中的第一个通道
  //print(a[:2, :1, :, :].shape)
  //
  //# 在第一个维度上取0和1,在第二个维度上取1,2，
  //# 等同于取第一、第二张图片中的第二个通道与第三个通道
  //print(a[:2, 1:, :, :].shape)
  //
  //# 在第一个维度上取0和1,在第二个维度上取1,2，
  //# 等同于取第一、第二张图片中的第二个通道与第三个通道
  //print(a[:2, -2:, :, :].shape)
  //
  //# 使用step隔行采样
  //# 在第一、第二维度取所有元素，在第三、第四维度隔行采样
  //# 等同于所有图片所有通道的行列每个一行或者一列采样
  //# 注意：下面的代码不包括28
  //print(a[:, :, 0:28:2, 0:28:2].shape)
  //print(a[:, :, ::2, ::2].shape)  # 等同于上面语句

  //torch.Size([2, 3, 28, 28])
  //torch.Size([2, 1, 28, 28])
  //torch.Size([2, 2, 28, 28])
  //torch.Size([2, 2, 28, 28])
  def randSelect031(): Unit = {
    val a = torch.rand(Seq(4, 3, 28, 28))
    // print(a[:2].shape) # torch.Size([2, 3, 28, 28])
    //print(a[:2, :1, :, :].shape) #torch.Size([2, 1, 28, 28])
    //print(a[:2, 1:, :, :].shape) # torch.Size([2, 2, 28, 28])
    //print(a[:2, -2:, :, :].shape) # torch.Size([2, 2, 28, 28])
    //print(a[:, :, 0:28:2, 0:28:2].shape) #torch.Size([4, 3, 14, 14])
    //print(a[:, :, ::2, ::2].shape) #torch.Size([4, 3, 14, 14])
    println(s"a[:2].shape ${a(Seq(0, 1)).shape}  torch.Size([2, 3, 28, 28])")
    println(s"a[:2, :1, :, :].shape ${a(Seq(0, 1), Slice(0), ::, ---).shape}  torch.Size([2,1, 28, 28])")
    println(s"a[:2,  1:, :, :].shape) ${a(::, 1, ---).shape}  torch.Size([2,2, 28, 28])")
    println(s"a[:2, -2:, :, :].shape) ${a(::, 1, ---).shape}  torch.Size([2,2, 28, 28])")
    println(s"a[:, :, 0:28:2, 0:28:2].shape  ${a(---, ---, 0.::(28), Slice(0, 28)).shape} torch.Size([4,3, 14, 14])")
    println(s"a[:, :, ::2, ::2].shape  step  ${a(---, ---, 0.::(2), 0.::(2)).shape} torch.Size([4, 3, 14, 14])")
  }

  //import torch
  //
  //a = torch.rand(4, 3, 28, 28)
  //
  //# 等与a
  //print(a[...].shape)
  //
  //# 第一张图片的所有维度
  //print(a[0, ...].shape)
  //
  //# 所有图片第二通道的所有维度
  //print(a[:, 1, ...].shape)
  //
  //# 所有图像所有通道所有行的第一、第二列
  //print(a[..., :2].shape)
  //torch.Size([4, 3, 28, 28])
  //torch.Size([3, 28, 28])
  //torch.Size([4, 28, 28])
  //torch.Size([4, 3, 28, 2])
  def randSelect04(): Unit = {
    val a = torch.rand(Seq(4, 3, 28, 28))
    println(s"a(...) shape ${a(---).shape}  torch.Size([4, 3, 28, 28])")
    println(s"a(0,...) shape ${a(0, ---).shape}  torch.Size([3, 28, 28])")
    println(s"a[:, 1, ...].shape) ${a(::, 1, ---).shape}  torch.Size([4, 28, 28])")
    println(s"a[..., :2].shape  ${a(---, Seq(0, 1)).shape} torch.Size([4, 3, 28, 2])")
    println(s"a[..., :2].shape  step  ${a(---, 0.::(2)).shape} torch.Size([4, 3, 28, 14])")
  }

  //import torch
  //
  //a = torch.randn(3, 4)
  //print(a)
  //
  //# 生成a这个Tensor中大于0.5的元素的掩码
  //mask = a.ge(0.5)
  //print(mask)
  //
  //# 取出a这个Tensor中大于0.5的元素
  //val = torch.masked_select(a, mask)
  //print(val)
  //print(val.shape)
  //tensor([[ 0.2055, -0.7070,  1.1201,  1.3325],
  //        [-1.6459,  0.9635, -0.2741,  0.0765],
  //        [ 0.2943,  0.1206,  1.6662,  1.5721]])
  //tensor([[0, 0, 1, 1],
  //        [0, 1, 0, 0],
  //        [0, 0, 1, 1]], dtype=torch.uint8)
  //tensor([1.1201, 1.3325, 0.9635, 1.6662, 1.5721])
  //torch.Size([5])
  def randSelect05(): Unit = {
    val a = torch.randn(Seq(3, 4))
    println(s"a ${a}")
    val mask = a.ge(0.5)
    println(s"mask ${mask.to(dtype = torch.uint8)}")
    val maskSelect = torch.masked_select(a, mask)
    println(s"mask select ${maskSelect}")
  }

  //import torch
  //
  //a = torch.tensor([[3, 7, 2], [2, 8, 3]])
  //print(a)
  //print(torch.take(a, torch.tensor([0, 1, 5])))
  //tensor([[3, 7, 2],
  //        [2, 8, 3]])
  //tensor([3, 7, 3])
  def randSelect06(): Unit = {
    val input = torch.tensor(Seq(Seq(3, 7, 2), Seq(2, 8, 13)))
    println(input)
    val index = torch.Tensor(Seq(0, 2, 4)).to(dtype = torch.int32)
    println(s"index dtype  ${index.dtype}")
    val res = index.dtype match
      case torch.int64 => fromNative(torchNative.take(input.native, index.native))
      case torch.int32 => fromNative(torchNative.take(input.native, index.to(dtype = torch.int64).native))
    val b = torch.take(input, index)
    println(res)

  }

  def randSelect03(): Unit = {
    val a = torch.rand(Seq(4, 3, 28, 28))
    // print(a[:2].shape) # torch.Size([2, 3, 28, 28])
    //print(a[:2, :1, :, :].shape) #torch.Size([2, 1, 28, 28])
    //print(a[:2, 1:, :, :].shape) # torch.Size([2, 2, 28, 28])
    //print(a[:2, -2:, :, :].shape) # torch.Size([2, 2, 28, 28])
    //print(a[:, :, 0:28:2, 0:28:2].shape) #torch.Size([4, 3, 14, 14])
    //print(a[:, :, ::2, ::2].shape) #torch.Size([4, 3, 14, 14])
    // Slice(extract(start), extract(end), extract(step))
    // ::(step: Int | Option[Int]): Slice(start, None, None) Slice(step, None, start)
    // &&(end: Int | Option[Int])  Slice(start, end, Some(1))
    println(s"a[:2].shape ${a(Seq(0, 1)).shape}  torch.Size([2, 3, 28, 28])")
    println(s"a[:, :, 0:28:2, 0:28:2].shape  ${a(::, ::, Slice(0, 28, 2), Slice(0, 28, 2)).shape} torch.Size([4,3, 14, 14])")
    println(s"a[:, :, ::2, ::2].shape  step  ${a(::, ::, 0.::(2), 0.::(2)).shape} torch.Size([4, 3, 14, 14])")
    println(s"a[:2, :1, :, :].shape ${a(0.&&(2), 0.&&(1), ::, ---).shape}  torch.Size([2,1, 28, 28])")

    println(s"a[:2,  1:, :, :].shape) ${a(0.&&(2), 1.&&(3), ::, ---).shape}  torch.Size([2,2, 28, 28])")
    println(s"a[:2, -2:, :, :].shape) ${a(0.&&(2), -2.&&(3), ::, ---).shape}  torch.Size([2,2, 28, 28])")
  }

  def randSelect032(): Unit = {
    val tensor = torch.arange(0, 12).reshape(4, 3)
    val t1 = tensor(0.&&(2), 1.&&(3))
    val t2 = tensor(0.&&(2), -2.&&(3))
    println(s"tensor ${tensor}")
    println(s"tensor[:2,  1:].shape) ${tensor(0.&&(2), 1.&&(3)).shape}  torch.Size([2,2])")
    println(s"a[:2, -2:].shape) ${tensor(0.&&(2), -2.&&(3)).shape}  torch.Size([2,2])")
    println(s"t1 ${t1}")
    println(s"t2 ${t2}")
  }

  //tensor = torch.arange(0,48).reshape(4,3,4)
  //k = tensor[:2,:, 1:]
  //kb = tensor[:2, :,-2:]
  //print(tensor[:2,:, 1:].shape) #torch.Size([2, 3,3])
  //print(tensor[:2,:, -2:].shape) #torch.Size([2,3, 2])
  /*
  *  python  scala
  *   :       ::  --- [all]
  *   k:      k.::    [only start]
  *   -k:      -k.::  [only start ]
  *   :k      0.&&(k)  [only end]
  *   ::k     0.::(k)  [only step]
  *   s:e:t    slice(s,e,t) [start, end, step]
  *   (k,r, c,..)  Seq(k,r,c..) [Select some index]
  *  s::k     s.::(k)  [only start and step]
  *  s:e       s.&&(e) [ only start end ]
  *  :e:t      slice(0,e,t) [only end step]
  * */
  def randSelect033(): Unit = {
    val tensor = torch.arange(0, 48).reshape(4, 3, 4)
    val t1 = tensor(0.&&(2), ::, 1.&&(4))
    val t2 = tensor(0.&&(2), ::, -2.&&(4))
    val t3 = tensor(0.&&(2), ::, 1.::)
    val t4 = tensor(0.&&(2), ::, -2.::)
    val t5 = tensor(0.&&(2), Slice(0, 1, 2), -2.::) //tensor[:2,:1:2, -2:]//torch.Size([2, 1, 2])
    println(s"tensor ${tensor}")
    println(s"t1: tensor[:2,::,  1:].shape) ${tensor(0.&&(2), ::, 1.&&(4)).shape}  torch.Size([2,3,3])")

    println(s"t3: tensor[:2,::,  1:].shape) ${tensor(0.&&(2), ::, 1.::).shape}  torch.Size([2,3,3])")

    println(s"t2: tensor[:2,::, -2:].shape) ${tensor(0.&&(2), ::, -2.&&(4)).shape}  torch.Size([2,3,2])")
    println(s"t4: tensor[:2,::, -2:].shape) ${tensor(0.&&(2), ::, -2.::).shape}  torch.Size([2,3,2])")
    println
    println(s"t1 ${t1}")
    println(s"t3 ${t3}")
    println(s"t2 ${t2}")
    println(s"t4 ${t4}")
  }

  @main
  def main(): Unit = {
    randSelect034()
    //    randSelect033()
    //    randSelect03()
    //    randSelect04()
    //    randSelect05()
    //    randSelect02()
    //    randSelect01()
    //    randSelect06()
    //    indexSelectRows()
    //    indexSelectColumns()
    //    indexSelectColumnsTwo()
    //    indexSelectColumnsThree()
    //    indexSelectColumnsFour()
    //    indexSelectRowsFour()
    //    indexSelectColumnsFour2()
    //    indexSelectRowsFifth()
    //    indexSelectColumnsFifth()
    //    indexUpdateColumnsSix()
    //    indexUpdateRowsSix()

    //    indexBackward()
  }

  def randSelect034(): Unit = {
    val tensor = torch.arange(0, 320).reshape(4, 10, 8)
    val t1 = tensor(3.&&(6), Slice(0, 8, 3), -4.::) //tensor( 3:6, :8:3,-4:)
    println(s"t1 shape ${t1.shape} t1 shape ArraySeq(1, 3, 4)")
    println(s"t1 ${t1}")
    val t2 = tensor(3.&&(6))
    println(s"t2 shape ${t2.shape} t2 shape ArraySeq(1, 10, 8)")
    println(s"t2 ${t2}")
    //[[[244, 245, 246, 247],
    //  [268, 269, 270, 271],
    //  [292, 293, 294, 295]]]

    val t3 = tensor(2.&&(6), Slice(0, 8, 3), -4.&&(2)) //t1 = tensor[2:6, :8:3, -4:2]
    println(s"t3 shape ${t3.shape} t3 shape (2, 3, 0)")
    println(s"t3 ${t3}")
    val t4 = tensor(2.&&(6), 0.&#(8, 2), -4.&&(-1)) //#:(8.::(2))
    println(s"t4 ${t4.shape}")
    val t5 = tensor(2.&&(6), 0.#&(8.::(2)), -4.&&(-1))
    println(s"t5 ${t5.shape}")
    val t6 = tensor(2.&&(6), 8.&^(0.::(2)), -4.&&(-1))
    println(s"t6 ${t6.shape}")
  }
}



//
//class PositionalEncoding[D <: BFloat16 | Float32 : Default](d_model: Long, max_len: Long = 28 * 28)
//  extends HasParams[D] {
//
//  import torch.{---, Slice}
//
//  var tensor = torch.ones(Seq(4, 4))
//  println(s"First row: ${tensor(0)}")
//  // First row: tensor dtype=float32, shape=[4], device=CPU
//  // [1.0000, 1.0000, 1.0000, 1.0000]
//  println(s"First column: ${tensor(Slice(), 0)}")
//  // First column: tensor dtype=float32, shape=[4], device=CPU
//  // [1.0000, 1.0000, 1.0000, 1.0000]
//  println(s"Last column: ${tensor(---, -1)}")
//
//  val arr = Seq(max_len, d_model)
//  var encoding = torch.zeros(size = arr.map(_.toInt), dtype = this.paramType)
//  val position = torch.arange(0, max_len, dtype = this.paramType).unsqueeze(1)
//  val div_term =
//    torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.Tensor(10000.0)) / d_model))
//  val sinPosition = torch.sin(position * div_term).to(dtype = this.paramType)
//  val cosPosition = torch.cos(position * div_term).to(dtype = this.paramType)
//  val indexSin = torch.Tensor(Seq(0L, 1L))
//  val indexCos = torch.Tensor(Seq(1L, 1L))
//  encoding.index(::, 1.::(13)).add(sinPosition)
//  encoding.index(::, Seq[Long](2, 1, 13)).add(sinPosition)
//  encoding.index(::, 13).equal(sinPosition)
//  encoding.update(indices = Seq(2.::(21), 1.::(13)), values = sinPosition)
//  encoding.update(indices = Seq(---, 2.::(21), 1.::(13)), values = sinPosition)
//  encoding.update(indices = Seq(---, ::(21), 1.::(13)), values = sinPosition)
//  encoding.update(indices = Seq(---, 1.::, 1.::(13)), values = sinPosition)
//  encoding.update(indices = Seq(---, ::, 1.::(13)), values = sinPosition)
//  encoding = encoding.to(dtype = this.paramType)
//  encoding = torch.indexCopy(encoding, 0, indexSin, sinPosition)
//  encoding = torch.indexCopy(encoding, 0, indexCos, cosPosition)
//  encoding = encoding.unsqueeze(0)
//
//  // return x + self.encoding[: ,: x.size(1)].to(x.device)
//  def apply(x: torch.Tensor[D]): torch.Tensor[D] =
//    x.add(encoding).to(x.device)
//}


