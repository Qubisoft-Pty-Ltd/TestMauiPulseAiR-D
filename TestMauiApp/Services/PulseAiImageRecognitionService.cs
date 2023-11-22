using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;
using TestMauiApp.Source;

namespace TestMauiApp.Services
{
    public class PulseAiImageRecognitionService
    {
        private readonly PulseAiImageRecognition _pulseAiRecognition;

        public PulseAiImageRecognitionService()
        {
            _pulseAiRecognition = new PulseAiImageRecognition();
            _pulseAiRecognition.InitConfig();
        }

        public async Task<List<string>> ClassifyImage()
        {
            try {
                var result = _pulseAiRecognition.Run();
                return result;
            } catch (Exception ex)
            {
                Debug.WriteLine(ex);
            }
      /*      var result = _pulseAiRecognition.Run();*/
            return [];
        }
    }
}
