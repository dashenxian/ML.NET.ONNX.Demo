using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace ML.NET.ONNX.Demo.Ultraface
{
    public class Box
    {
        public float X1 { get; set; }
        public float Y1 { get; set; }
        public float X2 { get; set; }
        public float Y2 { get; set; }
        public float Score { get; set; }
        // 计算面积
        public float Area => (X2 - X1) * (Y2 - Y1);

        private Rect GetRect()
        {
            var hei = Y2 - Y1;
            var wid = X2 - X1;
            if (wid < 0)
            {
                wid = 0;
            }
            if (hei < 0)
            {
                hei = 0;
            }
            return new Rect(X1, Y1, wid, hei);
        }

        private Rect rect = Rect.Empty;

        public Rect Rect
        {
            get
            {
                if (rect.IsEmpty)
                {
                    rect = GetRect();
                }
                return rect;
            }
        }

    }

}
