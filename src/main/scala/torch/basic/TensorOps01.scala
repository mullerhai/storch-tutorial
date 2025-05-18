package torch.basic

import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.global.torch as torchNative
import org.bytedeco.pytorch.{AbstractTensor, Node, OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
import torch.data.dataset.ChunkSharedBatchDataset
import torch.nn
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam

object TensorOps01 {

  //  @main
  def main(): Unit =
    //    randSelect01()
    seqDataTensor()
    singleDataTensor()
    randnDataTensor()
    rawTensorGrad()

  def rawTensorGrad(): Unit = {
    val dfx = AbstractTensor.create(1.0)
    val dfw = AbstractTensor.create(2.0)
    dfw.set_requires_grad(true)
    val dfb = AbstractTensor.create(3.0)
    val dfy = dfx.mul(dfw).add(dfb)
    dfy.grad()
    println(s"before backward dfy grad ${dfy.grad()} grad require ${dfy.requires_grad()} dfy grad_fn ${dfy.grad_fn()}")
    //    dfy.set_requires_grad(true)
    //    dfw.set_requires_grad(true)
    dfy.backward()
    println(s"after backward dfy grad: ${dfy.grad()} grad require ${dfy.requires_grad()} dfy.grad_fn： ${dfy.grad_fn()}")
    println(s"after backward dfw grad: ${dfw.grad()} dfw.grad_fn： ${dfw.grad_fn()}")

  }

  def singleDataTensor(): Unit = {
    val x = torch.Tensor(1.0, requires_grad = true)
    val w = torch.Tensor(2.0, requires_grad = true)
    //    w.requiresGrad = true //will warning
    println(s"before backward w requiresGrad ${w.requiresGrad} w grad ${w.grad}")
    w.set_requires_grad(true)
    println(s"after set grad w requiresGrad ${w.requiresGrad} w grad ${w.grad}")
    val b = torch.Tensor(3.0, requiresGrad = true)
    val y = w * x + b
    println(s"y ${y}")
    println(s"y grad: ${y.native.grad} ,,w grad ${w.grad}")
    println(s" y before backward : grad fn ${y.grad_fn()} native fn ${y.native.grad_fn()}")
    println(s"before backward  y.grad: ${y.grad}")
    println(s"before backward  x.grad: ${x.grad}")
    println(s"before backward  w.grad: ${w.grad}")
    println(s"before backward  b.grad: ${b.grad}")
    //    y.requiresGrad = true
    println(s" y grad fn ${y.grad_fn()}")
    y.grad_fn()
    println(s"before backward y requiresGrad ${y.requiresGrad}")
    y.backward() //java.lang.RuntimeException: element 0 of tensors does not require grad and does not have a grad_fn
    println(s"after backward y requiresGrad ${y.requiresGrad}")
    println(s"y grad: ${y.native.grad} ,,w grad ${w.grad}")
    println(s"y after backward : grad fn ${y.grad_fn()} native fn ${y.native.grad_fn()}")
    println(s" after backward : y.grad: ${y.grad}")
    println(s" after backward : x.grad: ${x.grad}")
    println(s" after backward : w.grad: ${w.grad}")
    println(s"b.grad: ${b.grad}")
    println(s"y after backward : grad: ${y.native.grad}")

  }

  def randnDataTensor(): Unit = {
    val x1 = torch.randn(Seq(10, 3))
    val y1 = torch.randn(Seq(10, 2))
    val linear = nn.Linear(3, 2)
    println(s"weight ${linear.weight}")
    println(s"bias ${linear.bias}")
    val criterion = nn.loss.MSELoss()
    val optimizer = torch.optim.SGD(linear.parameters(true), lr = 0.01)
    optimizer.zeroGrad()
    val pred = linear(x1)
    println(s"pred ${pred.shape}")
    val loss = criterion(pred.to(dtype = torch.float32), y1.to(dtype = torch.float32))
    println(s"loss  ${loss.item}")
    loss.requiresGrad = true
    loss.backward() //java.lang.RuntimeException: Found dtype Double but expected FloatException raised from compute_types
    println(s"dL/dw ${linear.weight.grad}")
    println(s"dL/db ${linear.bias.grad}")
    println(s"dL/dw ${loss.grad_fn()}")
    optimizer.step()
  }

  def seqDataTensor(): Unit = {

    val dd = torch.Tensor(Seq(24d, 36d), requiresGrad = true).reshape(1, 2)
    val ww = torch.tensor(Seq(12d, 18d), requires_grad = true).reshape(2, 1)
    val bb = torch.tensor(2.0, requires_grad = true)
    ww.set_requires_grad(true)
    bb.set_requires_grad(true)
    bb.requiresGrad = true

    val x_train = torch.Tensor(Seq(3.3, 4.4, 5.5, 6.71, 6.93, 4.169, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.312, 7.993, 3.1), requiresGrad = true).view(5, 3).to(dtype = torch.float32)
    val y_train = torch.Tensor(Seq(1.7, 2.76, 2.09, 3.19, 1.694), requiresGrad = true).view(5, 1).to(dtype = torch.float32)
    //, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.68, 2.904, 1.3
    val model = nn.Linear(3, 5)
    val outputs = model(x_train)
    val criterion = nn.loss.MSELoss()
    val loss = criterion(outputs, y_train)
    val kk = dd.mul(ww).add(bb)
    //    val kk = x_train.add(y_train)
    loss.backward()
    println(s"loss after backward : grad: ${loss.native.grad} ,${loss.grad}  grad fn ${loss.grad_fn()} ")
    kk.set_requires_grad(true)
    kk.requiresGrad = true
    val wwNode: Node = ww.native.grad_fn()
    if wwNode == null then println("null ... node")
    println(s"wwNode ${wwNode}")
    println(s"ww before backward requiresGrad ${ww.requiresGrad} grad ${ww.grad} grad fn ${ww.grad_fn()}")
    println(s"bb before backward requiresGrad ${bb.requiresGrad} grad ${bb.grad} grad fn ${bb.grad_fn()}")
    println(s"kk before backward requiresGrad ${kk.requiresGrad} grad ${kk.grad} grad fn ${kk.grad_fn()}")

    //    kk.backward()  //pytorch grad can be implicitly created only for scalar outputs //https://blog.csdn.net/qq_39208832/article/details/117415229
    println(s"kk after backward : grad: ${kk.native.grad} ,${kk.grad} bb grad ${bb.grad} ")
    //    y.requiresGrad = true
    println(s" kk after backward : grad fn ${kk.grad_fn()} native fn ${kk.native.grad_fn()}")
    println(s"after backward kk.grad ${kk.grad}")
    println(s"after backward bb.grad ${bb.grad}")
    println(s"after backward dd.grad ${bb.grad}")
    //    println(b.grad)
  }


}

