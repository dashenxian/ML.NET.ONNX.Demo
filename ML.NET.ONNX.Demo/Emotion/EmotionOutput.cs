using Microsoft.ML.Data;

namespace ML.NET.ONNX.Demo.Emotion
{
    public class EmotionOutput
    {
        [ColumnName("Plus692_Output_0")]
        public float[] Result { get; set; }
    }
}
