using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML.NET.ONNX.Demo.Ultraface
{
    public class RTFInput
    {
        public const int ImageWidth = 640;
        public const int ImageHeight = 480;
        [ImageType(ImageHeight, ImageWidth)]
        public MLImage Image { get; set; }
    }
}
