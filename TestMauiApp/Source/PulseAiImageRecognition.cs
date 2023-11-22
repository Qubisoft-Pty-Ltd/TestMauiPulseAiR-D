using System.Collections.Generic;
using System.Diagnostics;
using System.Xml.Linq;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace TestMauiApp.Source;

public class PulseAiImageRecognition : SciSharpExample, IExample
{
    /* string dir = Directory.GetCurrentDirectory();*/
    string dir = "C:\\Users\\marcu\\Work\\Source\\PulseAi\\test_maui_app\\TestMauiApp\\resources";
    string pbFile = "pulseAiModel.pb";
    string labelFile = "labels.txt";
    List<NDArray> file_ndarrays = new List<NDArray>();

    public ExampleConfig InitConfig()
        => Config = new ExampleConfig
        {
            Name = "Image Recognition Inception",
            Enabled = true,
            IsImportingGraph = false
        };

    public List<string> Run()
    {
        // load image file
        file_ndarrays.Clear();
        var files = Directory.GetFiles(Path.Join(dir, "house_photos"));
        for (int i = 0; i < files.Length; i++)
        {
            var nd = ReadTensorFromImageFile(files[i]);
            file_ndarrays.Add(nd);
        }

        tf.compat.v1.disable_eager_execution();

        var graph = tf.Graph().as_default();
        var pathModel = Path.Join(dir, pbFile);
        //import GraphDef from pb file
        graph.Import(Path.Join(dir, pbFile));

        var input_name = "input";
        var output_name = "output";

        var output_operation = graph.OperationByName(output_name);
        var input_operation = graph.OperationByName(input_name);

        var labels = File.ReadAllLines(Path.Join(dir, labelFile));
        var result_labels = new List<string>();
        var sw = new Stopwatch();

        var sess = tf.Session(graph);
        foreach (var nd in file_ndarrays)
        {
            sw.Restart();
/*            var outputs = sess.run(new[] {graph.OperationByName("detected_boxes").outputs[0],
                                      graph.OperationByName("detected_scores").outputs[0],
                                      graph.OperationByName("detected_classes").outputs[0]},*/

            var outputs = sess.run(new[] {graph.OperationByName("output").outputs[0],
                                                  graph.OperationByName("output1").outputs[0],
                                                  graph.OperationByName("output2").outputs[0]},
                       (input_operation.outputs[0], nd));

            ProcessModelOutputs(outputs);

            sw.Stop();
        }

        return result_labels;
    }

    private NDArray ReadTensorFromImageFile(string file_name,
                            int input_height = 320,
                            int input_width = 320,
                            int input_mean = 117,
                            int input_std = 1)
    {
        var graph = tf.Graph().as_default();

        var file_reader = tf.io.read_file(file_name, "file_reader");
        var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
        var cast = tf.cast(decodeJpeg, tf.float32);
        var dims_expander = tf.expand_dims(cast, 0);
        var resize = tf.constant(new int[] { input_height, input_width });
        var bilinear = tf.image.resize_bilinear(dims_expander, resize);
        var sub = tf.subtract(bilinear, new float[] { input_mean });
        var normalized = tf.divide(sub, new float[] { input_std });

        var sess = tf.Session(graph);
        return sess.run(normalized);
    }

    private List<string> ProcessModelOutputs(NDArray[] Outputs) {


        var labels = File.ReadAllLines(Path.Join(dir, labelFile));
        List<string> result_labels = new List<string>();
        float detectionThreshold = 0.5f;


        // Extract detected boxes, scores, and classes
        var detectedBoxesNdArray = Outputs[0];

        float[] detectedBoxesNdArrayFloat = Outputs[0].ToArray<float>();
        var shape = detectedBoxesNdArray.shape;
        float[,] detectedBoxes = new float[shape[0], shape[1]];
        float[] detectedScores = Outputs[1].ToArray<float>();
        float[] detectedClasses = Outputs[2].ToArray<float>();
        var flatArray = detectedBoxesNdArray.numpy();

        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                detectedBoxes[i, j] = (float)flatArray[i, j];
            }
        }

        for (int i = 0; i < detectedScores.Length; i++)
        {
            if (detectedScores[i] < detectionThreshold)
                continue;

            float x1 = detectedBoxes[i, 0], y1 = detectedBoxes[i, 1], x2 = detectedBoxes[i, 2], y2 = detectedBoxes[i, 3];
            int classIndex = (int)detectedClasses[i];
            string label = labels[classIndex];

            Debug.WriteLine($"Detected {label} with score {detectedScores[i]} at [{x1}, {y1}, {x2}, {y2}]");

            // Add to result_labels or any other list if necessary
            result_labels.Add($"{label} ({detectedScores[i]}): [{x1}, {y1}, {x2}, {y2}]");
        }

        return result_labels;



    }
}