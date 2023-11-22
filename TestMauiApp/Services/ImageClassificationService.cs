using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using TestMauiApp.Source;

namespace TestMauiApp.Services
{
    public class ImageClassificationService
    {
        private readonly ImageRecognitionInception _imageRecognition;

        public ImageClassificationService()
        {
            _imageRecognition = new ImageRecognitionInception();
            _imageRecognition.InitConfig();
            _imageRecognition.PrepareData();
        }

        public async Task<List<string>> ClassifyImage()
        {
            // Convert the Stream to an NDArray or a format that the TensorFlow model can process
            // You might need a utility method similar to ReadTensorFromImageFile but for a Stream
        /*    NDArray imageNDArray = ConvertStreamToNDArray(imageStream);*/

            // Run the model
            var result = _imageRecognition.Run(); // Modify Run method to accept NDArray and return classification result
            return result;
        }
    }
}
