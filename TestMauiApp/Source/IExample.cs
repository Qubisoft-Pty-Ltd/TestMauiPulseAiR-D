using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;

namespace TestMauiApp.Source
{
    public interface IExample
    {
        ExampleConfig Config { get; set; }
        ExampleConfig InitConfig();
        List<string> Run();

        void BuildModel();

        /// <summary>
        /// Build dataflow graph, train and predict
        /// </summary>
        /// <returns></returns>
        void Train();
        string FreezeModel();
        void Test();

        void Predict();

        Graph ImportGraph();

        Graph BuildGraph();

        /// <summary>
        /// Prepare dataset
        /// </summary>
        void PrepareData();
    }
}
