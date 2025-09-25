//import org.bytedeco.javacpp.PointerScope;
////import org.bytedeco.pytorch.AbstractTensor;
////import org.bytedeco.pytorch.global.torch;
//import torch.*;
//import torch.Slice;
//
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//
//import torch.nn.modules.TensorModule;
//import torch.nn.modules.container.Sequential;
//import torch.nn.functional.Activations ;
//import torch.nn.functional.Vision ;
//import torch.nn.functional.Loss;
//import torch.ops.BLASOps;
//import torch.ops.ComparisonOps;
//import torch.ops.CreationOps;
//import torch.ops.RandomSamplingOps;
//import torch.ops.IndexingSlicingJoiningOps;
//import torch.optim.Adagrad;
////import torch.nn.functional.Functional;
//import torch.nn.Functional;
//import torch.Torch;
//
//import scala.collection.JavaConverters;
//import torch.nn.modules.linear.Linear as Li;
//
//class TorchOps implements BLASOps, ComparisonOps, CreationOps, RandomSamplingOps,IndexingSlicingJoiningOps{
//    
//}
//
//class  ActFun implements Activations , Vision , Loss {
//    
//}
//public class TensorSliceOps01 {
//
//    // 2D Tensor - 选择行
//    public static void indexSelectRows() {
//        var funcitonal = new Functional();
//        var torch = new Torch();
//        
//        var torchkk = package$.MODULE$;
//        
//        Default<Float64> Float64;
//        var fc = new Linear(34,23, false,Float64);
//        
//        var fc2 =  new Linear<Float64>(34,23, false, Float64);
//        List<TensorModule<Float64>> models = new ArrayList<>();
//        models.add(fc);
//        models.add(fc2);
//        TensorModule<Float64>[] modelArray = models.toArray(new TensorModule[0]);
//        //Seq<TensorModule<Float64>>
//        var scalaSeq = JavaConverters.asScalaIteratorConverter(models.iterator()).asScala().toSeq();
////        var models = new TensorModule<Float64>[](fc,fc2);
//        var sequtail = new Sequential<Float64>(scalaSeq)  ;
//        Linear<Float> linear = Linear.apply(20L, 30L, true);
//        var grd = new Adagrad();
//        torch.package$.MODULE$.grid_sample()
//        var actFun = new ActFun();
//        var torchOps = new TorchOps();
//        var kk = torchOps.randn();
////        System.err.println(actFun);
//        var tensor = new Float16Tensor();
////        var tensor2 = torch.nn.functional.Vision.;
//        actFun.affine_grid() 
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 16).reshape(4, 4);
//            System.out.printf("First row: %s\n", tensor.get(0));
//            System.out.printf("Second row: %s\n", tensor.get(1));
//            System.out.printf("Third row: %s\n", tensor.get(2));
//            System.out.printf("Forth row: %s\n", tensor.get(3));
//            System.out.printf("Last row: %s\n", tensor.get(-1));
//            System.out.println("Read row finish \r\t");
//            var seq = new Sequential<TensorModule<>>();
//            seq.
//                    
//        }
//    }
//
//    // 2D Tensor - 选择列
//    public static void indexSelectColumns() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 16).reshape(4, 4);
//            System.out.printf("First column: %s\n", tensor.get(Slice.all(), 0));
//            System.out.printf("Second column: %s\n", tensor.get(Slice.all(), 1));
//            System.out.printf("Third column: %s\n", tensor.get(Slice.all(), 2));
//            System.out.printf("Forth column: %s\n", tensor.get(Slice.all(), 3));
//            System.out.printf("Last column: %s\n", tensor.get(Slice.all(), -1));
//            System.out.println("Read column finish \r\t");
//        }
//    }
//
//    // 2D Tensor - 选择列（使用 --- 符号）
//    public static void indexSelectColumnsTwo() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 16).reshape(4, 4);
//            System.out.printf("First column: %s\n", tensor.get(Slice.all(), 0));
//            System.out.printf("Second column: %s\n", tensor.get(Slice.all(), 1));
//            System.out.printf("Third column: %s\n", tensor.get(Slice.all(), 2));
//            System.out.printf("Forth column: %s\n", tensor.get(Slice.all(), 3));
//            System.out.printf("Last column: %s\n", tensor.get(Slice.all(), -1));
//            System.out.println("Read column finish \r\t");
//        }
//    }
//
//    // 2D Tensor - 选择列（使用 :: 符号）
//    public static void indexSelectColumnsThree() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 16).reshape(4, 4);
//            System.out.printf("First column: %s\n", tensor.get(Slice.all(), 0));
//            System.out.printf("Second column: %s\n", tensor.get(Slice.all(), 1));
//            System.out.printf("Third column: %s\n", tensor.get(Slice.all(), 2));
//            System.out.printf("Forth column: %s\n", tensor.get(Slice.all(), 3));
//            System.out.printf("Last column: %s\n", tensor.get(Slice.all(), -1));
//            System.out.println("Read column finish \r\t");
//        }
//    }
//
//    // 选择特定列
//    public static void indexSelectColumnsFour() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 16).reshape(4, 4);
//            System.out.printf("column select two columns ,index[0,1,2,3] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(1)));
//            System.out.printf("column select two columns ,index[0,2] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(2)));
//            System.out.printf("column select two columns ,index[0,3] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(3)));
//            System.out.printf("column select one columns ,index[0] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(4)));
//            System.out.printf("column select one columns ,index[0] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(5)));
//            System.out.printf("column select one columns ,index[1] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(1).to(3)));
//            System.out.println("Read column finish \r\t");
//        }
//    }
//
//    // 选择特定列（使用 --- 符号）
//    public static void indexSelectColumnsFour2() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 16).reshape(4, 4);
//            System.out.printf("column select two columns ,index[0,1,2,3] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(1)));
//            System.out.printf("column select two columns ,index[0,2] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(2)));
//            System.out.printf("column select two columns ,index[0,3] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(3)));
//            System.out.printf("column select one columns ,index[0] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(4)));
//            System.out.printf("column select one columns ,index[0] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(0).to(5)));
//            System.out.printf("column select one columns ,index[1] : %s\n", 
//                tensor.get(Slice.all(), Slice.from(1).to(3)));
//            System.out.println("Read column finish \r\t");
//        }
//    }
//
//    // 选择特定行
//    public static void indexSelectRowsFour() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 16).reshape(4, 4);
//            System.out.printf("column select two row tensor( 0.::(1)),index[0,1,2,3] : %s\n", 
//                tensor.get(Slice.from(0).to(1)));
//            System.out.printf("column select two row tensor( 0.::(2)),index[0,2] : %s\n", 
//                tensor.get(Slice.from(0).to(2)));
//            System.out.printf("column select two row tensor( 0.::(3)) ,index[0,3] : %s\n", 
//                tensor.get(Slice.from(0).to(3)));
//            System.out.printf("column select one row tensor( 0.::(4)),index[0] : %s\n", 
//                tensor.get(Slice.from(0).to(4)));
//            System.out.printf("column select one row tensor( 0.::(5)),index[0] : %s\n", 
//                tensor.get(Slice.from(0).to(5)));
//            System.out.printf("column select two row tensor( 1.::(3)) ,index[1] : %s\n", 
//                tensor.get(Slice.from(1).to(3)));
//        }
//    }
//
//    // 使用序列选择行
//    public static void indexSelectRowsFifth() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 16).reshape(4, 4);
//            System.out.printf("column select two row tensor(Seq(0,1)),index[0,1] : %s\n", 
//                tensor.get(Arrays.asList(0L, 1L)));
//            System.out.printf("column select two row tensor(Seq(0,2)),index[0,2] : %s\n", 
//                tensor.get(Arrays.asList(0L, 2L)));
//            System.out.printf("column select two row tensor(Seq(0,3)),index[0,3] : %s\n", 
//                tensor.get(Arrays.asList(0L, 3L)));
//            System.out.printf("column select one row tensor(Seq(3,0)),index[3,0] : %s\n", 
//                tensor.get(Arrays.asList(3L, 0L)));
//            System.out.printf("column select one row tensor(Seq(0,1,3)) index[0,1,3] : %s\n", 
//                tensor.get(Arrays.asList(0L, 1L, 3L)));
//            System.out.printf("column select two row tensor(Seq(3,1,0,2)) ,index[3,1,0,2] : %s\n", 
//                tensor.get(Arrays.asList(3L, 1L, 0L, 2L)));
//        }
//    }
//
//    // 使用序列选择列
//    public static void indexSelectColumnsFifth() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 16).reshape(4, 4);
//            System.out.printf("column select two Columns tensor(Seq(0,1)),index[0,1] : %s\n", 
//                tensor.get(Slice.all(), Arrays.asList(0L, 1L)));
//            System.out.printf("column select two Columns tensor(Seq(0,2)),index[0,2] : %s\n", 
//                tensor.get(Slice.all(), Arrays.asList(0L, 2L)));
//            System.out.printf("column select two Columns tensor(Seq(0,3)),index[0,3] : %s\n", 
//                tensor.get(Slice.all(), Arrays.asList(0L, 3L)));
//            System.out.printf("column select one Columns tensor(Seq(3,0)),index[3,0] : %s\n", 
//                tensor.get(Slice.all(), Arrays.asList(3L, 0L)));
//            System.out.printf("column select one Columns tensor(Seq(0,1,3)) index[0,1,3] : %s\n", 
//                tensor.get(Slice.all(), Arrays.asList(0L, 1L, 3L)));
//            System.out.printf("column select two Columns tensor(Seq(3,1,0,2)) ,index[3,1,0,2] : %s\n", 
//                tensor.get(Slice.all(), Arrays.asList(3L, 1L, 0L, 2L)));
//        }
//    }
//
//    // 更新特定列的值
//    public static void indexUpdateColumnsSix() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(1, 17).reshape(4, 4).to(torch.float32());
//            Tensor zero = torch.zeros(Arrays.asList(4L, 2L));
//            tensor.update(new Object[]{Slice.all(), Arrays.asList(2L, 1L)}, zero.to(torch.float32()));
//            System.out.printf("Index %s \n", tensor);
//        }
//    }
//
//    // 更新特定行的值
//    public static void indexUpdateRowsSix() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(1, 17).reshape(4, 4).to(torch.float32());
//            Tensor zero = torch.zeros(Arrays.asList(2L, 4L));
//            tensor.update(new Object[]{Arrays.asList(1L, 2L)}, zero.to(torch.float32()));
//            System.out.printf("Index %s\n", tensor);
//        }
//    }
//
//    // 索引反向传播
//    public static void indexBackward() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor x = torch.ones(Arrays.asList(2L, 2L), true);
//            Tensor y = x.mul(3).add(2);
//            AbstractTensor gfn = y.grad_fn();
//            System.out.printf("y %s\n", y.grad_fn());
//            System.out.printf("y %s\n", gfn.getptr().name().getString());
//            System.out.printf("y %s\n", gfn.getptr().name().getString());
//        }
//    }
//
//    // 随机选择操作1
//    public static void randSelect01() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor a = torch.rand(Arrays.asList(4L, 3L, 28L, 28L));
//            System.out.printf("a 0 shape %s\n", a.get(0).shape());
//            System.out.printf("a 00 shape %s\n", a.get(Arrays.asList(0L), Arrays.asList(0L)).shape());
//            System.out.printf("a 1234 shape %s\n", a.get(1, 2, 3, 4));
//        }
//    }
//
//    // 随机选择操作2
//    public static void randSelect02() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor a = torch.rand(Arrays.asList(4L, 3L, 28L, 28L));
//            Tensor aa = a.index_select(0, torch.tensor(Arrays.asList(0L, 2L)));
//            System.out.printf("aa shape %s , torch.Size([2, 3, 28, 28])\n", aa.shape());
//
//            a.numpy();
//            Tensor b = a.index_select(1, torch.tensor(Arrays.asList(0L, 2L), false));
//            System.out.printf(" b shape %s, torch.Size([4, 2, 28, 28]) \n", b.shape());
//            Tensor c = a.index_select(2, torch.arange(0, 8));
//            System.out.printf(" c shape %s , torch.Size([4, 3, 8, 28])\n", c.shape());
//        }
//    }
//
//    // 随机选择操作3-1
//    public static void randSelect031() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor a = torch.rand(Arrays.asList(4L, 3L, 28L, 28L));
//            System.out.printf("a[:2].shape %s  torch.Size([2, 3, 28, 28])\n", 
//                a.get(Arrays.asList(0L, 1L)).shape());
//            System.out.printf("a[:2, :1, :, :].shape %s  torch.Size([2,1, 28, 28])\n", 
//                a.get(Arrays.asList(0L, 1L), Slice.from(0), Slice.all(), Slice.all()).shape());
//            System.out.printf("a[:2,  1:, :, :].shape) %s  torch.Size([2,2, 28, 28])\n", 
//                a.get(Slice.all(), 1, Slice.all(), Slice.all()).shape());
//            System.out.printf("a[:2, -2:, :, :].shape) %s  torch.Size([2,2, 28, 28])\n", 
//                a.get(Slice.all(), 1, Slice.all(), Slice.all()).shape());
//            System.out.printf("a[:, :, 0:28:2, 0:28:2].shape  %s torch.Size([4,3, 14, 14])\n", 
//                a.get(Slice.all(), Slice.all(), Slice.from(0).to(28), Slice.from(0).to(28)).shape());
//            System.out.printf("a[:, :, ::2, ::2].shape  step  %s torch.Size([4, 3, 14, 14])\n", 
//                a.get(Slice.all(), Slice.all(), Slice.from(0).step(2), Slice.from(0).step(2)).shape());
//        }
//    }
//
//    // 随机选择操作4
//    public static void randSelect04() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor a = torch.rand(Arrays.asList(4L, 3L, 28L, 28L));
//            System.out.printf("a(...) shape %s  torch.Size([4, 3, 28, 28])\n", 
//                a.get(Slice.all(), Slice.all(), Slice.all(), Slice.all()).shape());
//            System.out.printf("a(0,...) shape %s  torch.Size([3, 28, 28])\n", 
//                a.get(0, Slice.all(), Slice.all(), Slice.all()).shape());
//            System.out.printf("a[:, 1, ...].shape) %s  torch.Size([4, 28, 28])\n", 
//                a.get(Slice.all(), 1, Slice.all(), Slice.all()).shape());
//            System.out.printf("a[..., :2].shape  %s torch.Size([4, 3, 28, 2])\n", 
//                a.get(Slice.all(), Slice.all(), Slice.all(), Arrays.asList(0L, 1L)).shape());
//            System.out.printf("a[..., :2].shape  step  %s torch.Size([4, 3, 28, 14])\n", 
//                a.get(Slice.all(), Slice.all(), Slice.all(), Slice.from(0).step(2)).shape());
//        }
//    }
//
//    // 掩码选择
//    public static void randSelect05() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor a = torch.randn(Arrays.asList(3L, 4L));
//            System.out.printf("a %s\n", a);
//            Tensor mask = a.ge(0.5);
//            System.out.printf("mask %s\n", mask.to(torch.uint8()));
//            Tensor maskSelect = torch.masked_select(a, mask);
//            System.out.printf("mask select %s\n", maskSelect);
//        }
//    }
//
//    // take 操作
//    public static void randSelect06() {
//        try (PointerScope scope = new PointerScope()) {
//            List<List<Long>> inputData = Arrays.asList(
//                Arrays.asList(3L, 7L, 2L),
//                Arrays.asList(2L, 8L, 13L)
//            );
//            Tensor input = torch.tensor(inputData);
//            System.out.println(input);
//            Tensor index = torch.tensor(Arrays.asList(0L, 2L, 4L)).to(torch.int32());
//            System.out.printf("index dtype  %s\n", index.dtype());
//            
//            Tensor res;
//            if (index.dtype() == torch.int64()) {
//                res = torch.take(input, index);
//            } else {
//                res = torch.take(input, index.to(torch.int64()));
//            }
//            
//            Tensor b = torch.take(input, index);
//            System.out.println(res);
//        }
//    }
//
//    // 随机选择操作3
//    public static void randSelect03() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor a = torch.rand(Arrays.asList(4L, 3L, 28L, 28L));
//            System.out.printf("a[:2].shape %s  torch.Size([2, 3, 28, 28])\n", 
//                a.get(Arrays.asList(0L, 1L)).shape());
//            System.out.printf("a[:, :, 0:28:2, 0:28:2].shape  %s torch.Size([4,3, 14, 14])\n", 
//                a.get(Slice.all(), Slice.all(), Slice.from(0).to(28).step(2), Slice.from(0).to(28).step(2)).shape());
//            System.out.printf("a[:, :, ::2, ::2].shape  step  %s torch.Size([4, 3, 14, 14])\n", 
//                a.get(Slice.all(), Slice.all(), Slice.from(0).step(2), Slice.from(0).step(2)).shape());
//            System.out.printf("a[:2, :1, :, :].shape %s  torch.Size([2,1, 28, 28])\n", 
//                a.get(Slice.from(0).to(2), Slice.from(0).to(1), Slice.all(), Slice.all()).shape());
//            System.out.printf("a[:2,  1:, :, :].shape) %s  torch.Size([2,2, 28, 28])\n", 
//                a.get(Slice.from(0).to(2), Slice.from(1).to(3), Slice.all(), Slice.all()).shape());
//            System.out.printf("a[:2, -2:, :, :].shape) %s  torch.Size([2,2, 28, 28])\n", 
//                a.get(Slice.from(0).to(2), Slice.from(-2).to(3), Slice.all(), Slice.all()).shape());
//        }
//    }
//
//    // 随机选择操作3-2
//    public static void randSelect032() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 12).reshape(4, 3);
//            Tensor t1 = tensor.get(Slice.from(0).to(2), Slice.from(1).to(3));
//            Tensor t2 = tensor.get(Slice.from(0).to(2), Slice.from(-2).to(3));
//            System.out.printf("tensor %s\n", tensor);
//            System.out.printf("tensor[:2,  1:].shape) %s  torch.Size([2,2])\n", 
//                tensor.get(Slice.from(0).to(2), Slice.from(1).to(3)).shape());
//            System.out.printf("a[:2, -2:].shape) %s  torch.Size([2,2])\n", 
//                tensor.get(Slice.from(0).to(2), Slice.from(-2).to(3)).shape());
//            System.out.printf("t1 %s\n", t1);
//            System.out.printf("t2 %s\n", t2);
//        }
//    }
//
//    // 随机选择操作3-3
//    public static void randSelect033() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 48).reshape(4, 3, 4);
//            Tensor t1 = tensor.get(Slice.from(0).to(2), Slice.all(), Slice.from(1).to(4));
//            Tensor t2 = tensor.get(Slice.from(0).to(2), Slice.all(), Slice.from(-2).to(4));
//            Tensor t3 = tensor.get(Slice.from(0).to(2), Slice.all(), Slice.from(1));
//            Tensor t4 = tensor.get(Slice.from(0).to(2), Slice.all(), Slice.from(-2));
//            Tensor t5 = tensor.get(Slice.from(0).to(2), Slice.from(0).to(1).step(2), Slice.from(-2));
//            
//            System.out.printf("tensor %s\n", tensor);
//            System.out.printf("t1: tensor[:2,::,  1:].shape) %s  torch.Size([2,3,3])\n", 
//                tensor.get(Slice.from(0).to(2), Slice.all(), Slice.from(1).to(4)).shape());
//            System.out.printf("t3: tensor[:2,::,  1:].shape) %s  torch.Size([2,3,3])\n", 
//                tensor.get(Slice.from(0).to(2), Slice.all(), Slice.from(1)).shape());
//            System.out.printf("t2: tensor[:2,::, -2:].shape) %s  torch.Size([2,3,2])\n", 
//                tensor.get(Slice.from(0).to(2), Slice.all(), Slice.from(-2).to(4)).shape());
//            System.out.printf("t4: tensor[:2,::, -2:].shape) %s  torch.Size([2,3,2])\n", 
//                tensor.get(Slice.from(0).to(2), Slice.all(), Slice.from(-2)).shape());
//            System.out.println();
//            System.out.printf("t1 %s\n", t1);
//            System.out.printf("t3 %s\n", t3);
//            System.out.printf("t2 %s\n", t2);
//            System.out.printf("t4 %s\n", t4);
//        }
//    }
//
//    // 随机选择操作3-4
//    public static void randSelect034() {
//        try (PointerScope scope = new PointerScope()) {
//            Tensor tensor = torch.arange(0, 320).reshape(4, 10, 8);
//            Tensor t1 = tensor.get(Slice.from(3).to(6), Slice.from(0).to(8).step(3), Slice.from(-4));
//            System.out.printf("t1 shape %s t1 shape ArraySeq(1, 3, 4)\n", t1.shape());
//            System.out.printf("t1 %s\n", t1);
//            
//            Tensor t2 = tensor.get(Slice.from(3).to(6));
//            System.out.printf("t2 shape %s t2 shape ArraySeq(1, 10, 8)\n", t2.shape());
//            System.out.printf("t2 %s\n", t2);
//            
//            Tensor t3 = tensor.get(Slice.from(2).to(6), Slice.from(0).to(8).step(3), Slice.from(-4).to(2));
//            System.out.printf("t3 shape %s t3 shape (2, 3, 0)\n", t3.shape());
//            System.out.printf("t3 %s\n", t3);
//            
//            Tensor t4 = tensor.get(Slice.from(2).to(6), Slice.from(0).to(8).step(2), Slice.from(-4).to(-1));
//            System.out.printf("t4 %s\n", t4.shape());
//        }
//    }
//
//    public static void main(String[] args) {
//        try (PointerScope scope = new PointerScope()) {
//            randSelect034();
//            randSelect033();
//            randSelect03();
//            randSelect04();
//            randSelect05();
//            randSelect02();
//            randSelect01();
//            randSelect06();
//            indexSelectRows();
//            indexSelectColumns();
//            indexSelectColumnsTwo();
//            indexSelectColumnsThree();
//            indexSelectColumnsFour();
//            indexSelectRowsFour();
//            indexSelectColumnsFour2();
//            indexSelectRowsFifth();
//            indexSelectColumnsFifth();
//            indexUpdateColumnsSix();
//            indexUpdateRowsSix();
//            // indexBackward();
//        }
//    }
//}
