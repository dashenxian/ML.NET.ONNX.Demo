using Microsoft.Win32;
using ML.NET.ONNX.Demo.Ultraface;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using ML.NET.ONNX.Demo.Emotion;

namespace ML.NET.ONNX.Demo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private RTFPrediction rtfPrediction = new RTFPrediction();
        private EmotionPrediction emotionPrediction = new EmotionPrediction();
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            if (dialog.ShowDialog().Value)
            {
                //image.Source =bitma;
                var result = rtfPrediction.Predict(dialog.FileName);
                if (result != null)
                {
                    image.Source = result;
                }
            }
        }

        private void Button_Click1(object sender, RoutedEventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            if (dialog.ShowDialog().Value)
            {
                var result = emotionPrediction.Predict(dialog.FileName);
                MessageBox.Show(result);
            }
        }
    }
}
