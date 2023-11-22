using System.Diagnostics;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

namespace TestMauiApp.Source;

public class ImageRecognitionInception : SciSharpExample, IExample
{
    /* string dir = Directory.GetCurrentDirectory();*/
    string dir = "C:\\Users\\marcu\\Work\\Source\\PulseAi\\test_maui_app\\TestMauiApp\\resources";
    string pbFile = "tensorflow_inception_graph.pb";
    string labelFile = "imagenet_comp_graph_label_strings.txt";
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
        tf.compat.v1.disable_eager_execution();

        PrepareData();

        var graph = tf.Graph().as_default();
        //import GraphDef from pb file
        graph.Import(Path.Join(dir, pbFile));

        var input_name = "input";
        var output_name = "output";

        var input_operation = graph.OperationByName(input_name);
        var output_operation = graph.OperationByName(output_name);

        var labels = File.ReadAllLines(Path.Join(dir, labelFile));
        var result_labels = new List<string>();
        var sw = new Stopwatch();

        var sess = tf.Session(graph);
        foreach (var nd in file_ndarrays)
        {
            sw.Restart();

            var results = sess.run(output_operation.outputs[0], (input_operation.outputs[0], nd));
            results = np.squeeze(results);
            int idx = np.argmax(results);

            Debug.WriteLine($"{labels[idx]} {results[idx]} in {sw.ElapsedMilliseconds}ms");
            result_labels.Add(labels[idx]);
        }

        return result_labels;
    }

    private NDArray ReadTensorFromImageFile(string file_name,
                            int input_height = 224,
                            int input_width = 224,
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

    public override void PrepareData()
    {
        // have to renable If I want to change resources and models to use
        /*        Directory.CreateDirectory(dir);

                // get model file
                string url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip";
                DownloadFile(url, dir, "inception5h.zip");

                ZipFile.ExtractToDirectory(Path.Join(dir, "inception5h.zip"), dir);

                // download sample picture
                Directory.CreateDirectory(Path.Join(dir, "img"));
                url = $"https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/label_image/data/grace_hopper.jpg";
                DownloadFile(url, Path.Join(dir, "img"), "grace_hopper.jpg");

                url = $"https://raw.githubusercontent.com/SciSharp/TensorFlow.NET/master/data/shasta-daisy.jpg";
                DownloadFile(url, Path.Join(dir, "img"), "shasta-daisy.jpg");*/

        // load image file
        file_ndarrays.Clear();
        var files = Directory.GetFiles(Path.Join(dir, "img"));
        for (int i = 0; i < files.Length; i++)
        {
            var nd = ReadTensorFromImageFile(files[i]);
            file_ndarrays.Add(nd);
        }
    }

    public static void DownloadFile(string url, string directory, string fileName)
    {
        using (var httpClient = new HttpClient())
        {
            var response = httpClient.GetAsync(url).Result;
            response.EnsureSuccessStatusCode();

            var filePath = Path.Combine(directory, fileName);
            using (var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                response.Content.CopyToAsync(fileStream).Wait();
            }
        }
    }
}