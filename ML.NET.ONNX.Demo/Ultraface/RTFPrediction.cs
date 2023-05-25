using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Windows;
using System.Windows.Interop;
using SkiaSharp;

namespace ML.NET.ONNX.Demo.Ultraface
{
    public class RTFPrediction
    {
        private readonly string modelFile = "version-RFB-640.onnx";
        private PredictionEngine<RTFInput, RTFOutput> predictionEngine = null;
        /// <summary>
        /// 人脸检测期望阈值
        /// </summary>
        public double Score { get; set; } = 0.9;

        /// <summary>
        /// 人脸检测框重叠计算的交并比
        /// </summary>
        public double IouThreshold { get; set; } = 0.4;
        public RTFPrediction()
        {
            MLContext context = new MLContext();
            var emptyData = new List<RTFInput>();
            var data = context.Data.LoadFromEnumerable(emptyData);
            var pipeline =
                context.Transforms.ResizeImages(
                        resizing: Microsoft.ML.Transforms.Image.ImageResizingEstimator.ResizingKind.Fill, //填充Resize
                        outputColumnName: "resize", //Resize的结果放置到 data列
                        imageWidth: RTFInput.ImageWidth,
                        imageHeight: RTFInput.ImageHeight,
                        inputColumnName: nameof(RTFInput.Image) //从Image属性来源,
                    )
                    .Append(
                        context.Transforms.ExtractPixels(
                            offsetImage: 127f,
                            scaleImage: 1 / 128f,
                            inputColumnName: "resize",
                            outputColumnName: "input")
                    ).Append(
                        context.Transforms.ApplyOnnxModel(
                            modelFile: modelFile,
                            inputColumnNames: new string[] { "input" },
                            outputColumnNames: new string[] { "scores", "boxes" }));
            var model = pipeline.Fit(data);
            predictionEngine = context.Model.CreatePredictionEngine<RTFInput, RTFOutput>(model); //生成预测引擎
        }
        /// <summary>
        /// 预测
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public ImageSource Predict(string path)
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var bitmap = MLImage.CreateFromStream(stream);
            var prediction = predictionEngine.Predict(new RTFInput() { Image = bitmap });
            var boxes = ParseBox(prediction);
            boxes = boxes.Where(b => b.Score > Score).OrderByDescending(s => s.Score).ToList();
            boxes = HardNms(boxes, IouThreshold);
            //boxes = HardNMS(boxes, 0.4);
            var bitmapImage = new BitmapImage(new Uri(path));
            var rtb = new RenderTargetBitmap(bitmap.Width, bitmap.Height, 96, 96, PixelFormats.Pbgra32);
            var dv = new DrawingVisual();
            using (var dc = dv.RenderOpen())
            {
                dc.DrawImage(bitmapImage, new Rect(0, 0, bitmap.Width, bitmap.Height));
                foreach (var item in boxes)
                {
                    dc.DrawRectangle(null, new Pen(Brushes.Red, 2),
                        new Rect(item.Rect.X * bitmap.Width, item.Rect.Y * bitmap.Height,
                            item.Rect.Width * bitmap.Width, item.Rect.Height * bitmap.Height));
                }
            }

            rtb.Render(dv);

            return rtb;
        }
        /// <summary>
        /// 转换输出结果为矩形框
        /// </summary>
        /// <param name="prediction"></param>
        /// <returns></returns>
        private List<Box> ParseBox(RTFOutput prediction)
        {
            var boxes = new List<Box>();
            for (int i = 0; i < prediction.Boxes.Length; i += 4)
            {
                boxes.Add(new Box
                {
                    X1 = prediction.Boxes[i],
                    Y1 = prediction.Boxes[i + 1],
                    X2 = prediction.Boxes[i + 2],
                    Y2 = prediction.Boxes[i + 3],
                    Score = prediction.Scores[i / 2 + 1]
                });
            }
            boxes = boxes.OrderByDescending(b => b.Score).ToList();
            return boxes.ToList();
        }
        /// <summary>
        /// 非极大值抑制,在目标检测中，经常会出现多个检测框（bounding box）重叠覆盖同一目标的情况，而我们通常只需要保留一个最佳的检测结果。非极大值抑制（Non-Maximum Suppression，NMS）就是一种常见的目标检测算法，用于在冗余的检测框中筛选出最佳的一个。
        /// NMS 原理是在对检测结果进行处理前，按照检测得分进行排序（一般检测得分越高，表明检测框越可能包含目标），然后选择得分最高的检测框加入结果中。接下来，遍历排序后的其余检测框，如果检测框之间的IoU（Intersection over Union，交并比）大于一定阈值，那么就将该检测框删除，因为被保留的那一个框已经足够表明目标的存在。
        /// </summary>
        /// <param name="boxes">检测框列表</param>
        /// <param name="iouThreshold">交并比</param>
        /// <returns>去掉重复后的检测框列表</returns>
        private List<Box> HardNms(List<Box> boxes, double iouThreshold)
        {
            // 按照分数从高到低排序
            boxes = boxes.OrderByDescending(b => b.Score).ToList();
            var selectedBoxes = new List<Box>();
            while (boxes.Count > 0)
            {
                // 选择分数最高的框
                var bestBox = boxes[0];
                selectedBoxes.Add(bestBox);
                boxes.RemoveAt(0);
                // 计算当前框与其它框的交并比
                for (int i = boxes.Count - 1; i >= 0; i--)
                {
                    var iou = CalculateIOU(bestBox, boxes[i]);
                    if (iou >= iouThreshold)
                    {
                        boxes.RemoveAt(i);
                    }
                }
            }

            return selectedBoxes;
        }

        /// <summary>
        /// 计算两个框的交集面积
        /// </summary>
        /// <param name="box1">框1</param>
        /// <param name="box2">框2</param>
        /// <returns></returns>
        private static double CalculateIOU(Box box1, Box box2)
        {
            var x1 = Math.Max(box1.X1, box2.X1);
            var y1 = Math.Max(box1.Y1, box2.Y1);
            var x2 = Math.Min(box1.X2, box2.X2);
            var y2 = Math.Min(box1.Y2, box2.Y2);

            if (x2 <= x1 || y2 <= y1) { return 0.0; }

            var intersectionArea = (x2 - x1) * (y2 - y1);
            var box1Area = (box1.X2 - box1.X1) * (box1.Y2 - box1.Y1);
            var box2Area = (box2.X2 - box2.X1) * (box2.Y2 - box2.Y1);
            var unionArea = box1Area + box2Area - intersectionArea;

            return intersectionArea / unionArea;
        }
    }


}
