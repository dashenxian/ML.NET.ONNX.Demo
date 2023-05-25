using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace ML.NET.ONNX.Demo.Emotion
{
    public class EmotionInput
    {
        public const int ImageWidth = 64;
        public const int ImageHeight = 64;
        [ImageType(ImageHeight, ImageWidth)]
        public MLImage Image { get; set; }
    }
}
