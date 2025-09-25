import torch.nn.{Module, Parameter, functional as F}
import torch.optim.Optimizer
import torch.{Default, FloatNN, Tensor}
import torch.nn as nn
import torch.nn.modules.{HasParams, TensorModule}
import scala.collection.mutable.ArrayBuffer

class DeepIRTLayer[ParamType <: FloatNN: Default](val memorySize: Int, 
                                                  val memoryStateDim: Int, 
                                                  val isWrite: Boolean) extends TensorModule[ParamType] with HasParams[ParamType] {
  
  // 权重参数
  val erase = if (isWrite) Some(nn.Linear(memoryStateDim, memoryStateDim, bias = true)) else None
  val add = if (isWrite) Some(nn.Linear(memoryStateDim, memoryStateDim, bias = true)) else None
  
  // 初始化权重
  if (isWrite) {
    erase.get.reset_parameters()
    add.get.reset_parameters()
  }
  
  def addressing(controlInput: Tensor[ParamType], memory: Tensor[ParamType]): Tensor[ParamType] = {
    val similarityScore = controlInput.matmul(memory.t())
    F.softmax(similarityScore, dim = 1)
  }
  
  def read(
    memory: Tensor[ParamType], 
    controlInput: Option[Tensor[ParamType]] = None, 
    readWeight: Option[Tensor[ParamType]] = None
  ): Tensor[ParamType] = {
    val rw = readWeight.getOrElse(addressing(controlInput.get, memory))
    val readWeightReshaped = rw.view(-1, 1)
    val memoryReshaped = memory.view(-1, memoryStateDim)
    val rc = readWeightReshaped * memoryReshaped
    val readContent = rc.view(-1, memorySize, memoryStateDim)
    readContent.sum(dim = 1)
  }
  
  def write(
    controlInput: Tensor[ParamType], 
    memory: Tensor[ParamType], 
    writeWeight: Tensor[ParamType]
  ): Tensor[ParamType] = {
    assert(isWrite, "Cannot write with isWrite = false")
    
    val eraseSignal = F.sigmoid(erase.get(controlInput))
    val addSignal = F.tanh(add.get(controlInput))
    
    val eraseReshape = eraseSignal.view(-1, 1, memoryStateDim)
    val addReshape = addSignal.view(-1, 1, memoryStateDim)
    val writeWeightReshape = writeWeight.view(-1, memorySize, 1)
    
    val eraseMult = eraseReshape * writeWeightReshape
    val addMult = addReshape * writeWeightReshape
    
    memory * (torch.tensor(1) - eraseMult) + addMult
  }
  
  override def parameters: List[Tensor[ParamType]] = {
    val params = ArrayBuffer[Tensor[ParamType]]()
    if (isWrite) {
      params ++= erase.get.parameters
      params ++= add.get.parameters
    }
    params.toList
  }
  
  def apply(input: (Tensor[ParamType], Tensor[ParamType], Option[Tensor[ParamType]])): Tensor[ParamType] = {
    val (controlInput, memory, weight) = input
    if (isWrite) write(controlInput, memory, weight.get) else read(memory, Some(controlInput), weight)
  }

  override def apply(v1: Tensor[ParamType]): Tensor[ParamType] = ???
}

class DeepIRT[ParamType <: FloatNN: Default](
    val memorySize: Int,
    val memoryKeyStateDim: Int,
    val memoryValueStateDim: Int,
    initMemoryKey: Tensor[ParamType]
) extends Module with HasParams[ParamType] {
  
  val keyHead = DeepIRTLayer[ParamType](memorySize, memoryKeyStateDim, isWrite = false)
  val valueHead = DeepIRTLayer[ParamType](memorySize, memoryValueStateDim, isWrite = true)
  
  var memoryKey: Tensor[ParamType] = initMemoryKey
  var memoryValue: Option[Tensor[ParamType]] = None
  
  def init_value_memory(memoryValue: Tensor[ParamType]): Unit = {
    this.memoryValue = Some(memoryValue)
  }
  
  def attention(controlInput: Tensor[ParamType]): Tensor[ParamType] = {
    keyHead.addressing(controlInput, memoryKey)
  }
  
  def read(readWeight: Tensor[ParamType]): Tensor[ParamType] = {
    require(memoryValue.isDefined, "memoryValue not initialized. Call initValueMemory first.")
    valueHead.read(memoryValue.get, readWeight = Some(readWeight))
  }
  
  def write(writeWeight: Tensor[ParamType], controlInput: Tensor[ParamType]): Tensor[ParamType] = {
    require(memoryValue.isDefined, "memoryValue not initialized. Call initValueMemory first.")
    val newMemoryValue = valueHead.write(controlInput, memoryValue.get, writeWeight)
    memoryValue = Some(newMemoryValue)
    newMemoryValue
  }
  
  override def parameters: List[Tensor[ParamType]] = {
    keyHead.parameters ++ valueHead.parameters
  }
  
  def apply(controlInput: Tensor[ParamType]): Tensor[ParamType] = {
    val attnWeight = attention(controlInput)
    read(attnWeight)
  }
}


